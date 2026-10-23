/**
 * Summarize strategy — replaces oversized content with LLM-generated summaries.
 *
 * @internal
 */

import { logger } from '../../../logging/logger.js'
import { normalizeError } from '../../../errors.js'
import { Message, TextBlock, ToolResultBlock } from '../../../types/messages.js'
import { DocumentBlock, ImageBlock } from '../../../types/media.js'
import type { ContentBlock } from '../../../types/messages.js'
import type { Model } from '../../../models/model.js'
import type { InvocationContext } from '../../../agent/invocation-context.js'
import type { LocalAgent } from '../../../types/agent.js'
import type { ContextStrategy, ContextState } from '../../types.js'
import {
  flattenMessagesToContent,
  formatSummarized,
  summarizeContent,
  toolResultToContentBlocks,
  type SummarizeConfig,
} from '../../methods/summarize.js'
import { formatStashRefs } from '../../stash.js'
import {
  BaseOffloadStrategy,
  collectRemovableWithPair,
  insertMarker,
  spliceWithPairs,
  repairAlternation,
  type OffloadConditions,
  type OffloadTarget,
} from './base.js'

/**
 * Replaces `eligible` messages in `messages` with one `[Summarized: ...]` marker, in place. Each eligible
 * message's tool-pair partner is summarized with it; message 0 and the last message never are.
 *
 * @param messages - The history to edit
 * @param eligible - Messages in `messages` to summarize
 * @param model - Model that writes the summary
 * @param config - Summarization settings; only `systemPrompt` is read
 * @param invocationContext - The request the summary call's usage is added to
 * @returns Whether any messages were summarized
 * @internal
 */
export async function summarizeMessages(
  messages: Message[],
  eligible: Message[],
  model: Model,
  config?: SummarizeConfig,
  invocationContext?: InvocationContext
): Promise<boolean> {
  const safeSet = new Set<Message>()
  for (const message of eligible) {
    const index = messages.indexOf(message)
    if (index === -1) continue
    for (const removable of collectRemovableWithPair(messages, index)) {
      safeSet.add(removable)
    }
  }
  const safe = messages.filter((message) => safeSet.has(message))
  if (safe.length === 0) return false

  const contentBlocks = flattenMessagesToContent(safe)
  const summary = await summarizeContent(contentBlocks, model, config, invocationContext)
  if (!summary) return false

  const totalTokens = await model.countTokens(safe)
  const { lowestIndex } = spliceWithPairs(messages, safe)
  insertMarker(messages, lowestIndex, formatSummarized(`${safe.length} messages`, totalTokens, summary))

  repairAlternation(messages)
  logger.debug(`summarized=<${safe.length}>, tokens=<${totalTokens}> | batched summarization complete`)
  return true
}

export class SummarizeStrategy extends BaseOffloadStrategy {
  readonly name = 'offload:summarize'

  private readonly _config: SummarizeConfig

  constructor(target?: OffloadTarget, config?: SummarizeConfig, conditions?: OffloadConditions) {
    super(target, conditions)
    this._config = config ?? {}
  }

  when(conditions: OffloadConditions): ContextStrategy {
    return new SummarizeStrategy(this._target, this._config, conditions)
  }

  override async apply(context: ContextState): Promise<boolean> {
    const model = this._resolveModel(context.agent)
    if (!model) {
      logger.warn('no model available for summarization')
      return false
    }
    if (this._isMessageLevel && this._selectsByRecencyOnly) {
      if (!context.overflow && context.utilization < this._utilizationThreshold!) return false
      return this._compact(context, model)
    }
    if (!this._isMessageLevel && this._isRetroactive && model.serverCompaction === true) {
      return this._compact(context, model)
    }
    return super.apply(context)
  }

  /**
   * Whether the messages this strategy summarizes are fully determined by `preserveRecent`, which is
   * the only selection `Model.compact()` receives.
   */
  private get _selectsByRecencyOnly(): boolean {
    return (this._target === undefined || this._target === '*') && this._threshold === undefined
  }

  /** Replaces the history with `model.compact()`'s result. */
  private async _compact(context: ContextState, model: Model): Promise<boolean> {
    const { messages, agent } = context
    const compacted = await model.compact(
      messages,
      {
        preservedMessages: this._preserveRecent,
        // A server-side compaction must carry the live system prompt and tools, or the provider
        // treats it as a different conversation.
        ...(agent.systemPrompt !== undefined && { systemPrompt: agent.systemPrompt }),
        toolSpecs: agent.toolRegistry.list().map((tool) => tool.toolSpec),
        ...(this._config.systemPrompt !== undefined && { instructions: this._config.systemPrompt }),
      },
      context.invocationContext
    )
    if (!compacted) return false

    messages.splice(0, messages.length, ...compacted)
    return true
  }

  protected override async _applyPerMessage(context: ContextState): Promise<boolean> {
    const model = this._resolveModel(context.agent)
    if (!model) return false

    const { messages } = context
    if (messages.length <= 1) return false

    const eligible = await this._getEligibleMessages(context)
    if (eligible.length === 0) return false

    return summarizeMessages(messages, eligible, model, this._config, context.invocationContext)
  }

  protected async _replaceBlock(
    block: ContentBlock,
    tokens: number,
    message: Message,
    agent: LocalAgent,
    stashRefs: string[],
    invocationContext?: InvocationContext
  ): Promise<ContentBlock | null> {
    const model = this._resolveModel(agent)
    if (!model) return null

    if (block instanceof ToolResultBlock) {
      const summary = await this._trySummarize(
        toolResultToContentBlocks(block.content),
        model,
        message,
        invocationContext
      )
      if (!summary) return null

      logger.debug(`toolUseId=<${block.toolUseId}>, tokens=<${tokens}> | summarized tool result`)
      return new ToolResultBlock({
        toolUseId: block.toolUseId,
        status: block.status,
        content: [new TextBlock(`${formatSummarized('tool result', tokens, summary)}${formatStashRefs(stashRefs)}`)],
      })
    }

    if (block instanceof TextBlock) {
      const summary = await this._trySummarize([new TextBlock(block.text)], model, message, invocationContext)
      if (!summary) return null

      logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | summarized text block`)
      return new TextBlock(`${formatSummarized('text block', tokens, summary)}${formatStashRefs(stashRefs)}`)
    }

    if (block instanceof ImageBlock || block instanceof DocumentBlock) {
      const summary = await this._trySummarize([block], model, message, invocationContext)
      if (summary) {
        logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | summarized media block`)
        return new TextBlock(`${formatSummarized('media block', tokens, summary)}${formatStashRefs(stashRefs)}`)
      }
    }

    logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | offloaded media block`)
    return new TextBlock(`[Offloaded: ~${tokens} tokens]${formatStashRefs(stashRefs)}`)
  }

  /** Summarizes one block; a failure is logged and leaves the block unsummarized so the pass continues. */
  private async _trySummarize(
    content: ContentBlock[],
    model: Model,
    message: Message,
    invocationContext?: InvocationContext
  ): Promise<string | null> {
    try {
      return await summarizeContent(content, model, this._config, invocationContext)
    } catch (error) {
      logger.warn(
        `trackingId=<${message.trackingId}>, error=<${normalizeError(error).message}> | block summarization failed`
      )
      return null
    }
  }

  private _resolveModel(agent: LocalAgent): Model | undefined {
    return this._config.model ?? agent.auxModel
  }
}
