/**
 * Summarize strategy — replaces oversized content with LLM-generated summaries.
 *
 * @internal
 */

import { logger } from '../../../logging/logger.js'
import { normalizeError } from '../../../errors.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock, type Role } from '../../../types/messages.js'
import { DocumentBlock, ImageBlock } from '../../../types/media.js'
import type { ContentBlock } from '../../../types/messages.js'
import { Model, type SummarizeOptions, type SummarizeResult } from '../../../models/model.js'
import { unwrapInvocationContext, type InvocationContext } from '../../../agent/invocation-context.js'
import type { LocalAgent } from '../../../types/agent.js'
import type { ContextStrategy, ContextState } from '../../types.js'
import {
  blockSummaryRequest,
  DEFAULT_BLOCK_INSTRUCTIONS,
  formatSummarized,
  toolResultToContentBlocks,
  type SummarizeConfig,
} from '../../methods/summarize.js'
import { formatStashRefs } from '../../stash.js'
import { isPinned } from '../../../conversation-manager/compression/pin-message.js'
import {
  BaseOffloadStrategy,
  collectRemovableWithPair,
  repairAlternation,
  resolvePreserveCount,
  type OffloadConditions,
  type OffloadTarget,
} from './base.js'

/**
 * Returns the index of the first message a `Model.summarize()` call keeps verbatim.
 *
 * The cut never separates a `toolUse` from its `toolResult`: it moves forward to summarize the pair together,
 * or back when the `toolResult` is the last message, which the next model call responds to and which is never
 * summarized. A one-message prefix is not cut: after a pass it is that pass's summary, and re-summarizing it
 * alone frees nothing.
 *
 * @param messages - Full conversation history
 * @param preservedMessages - Count or ratio (between 0 and 1 exclusive) of trailing messages to keep; the last
 *   message is always kept, so 0 and 1 cut at the same place
 * @returns The split index; messages before it may be summarized. 0 when nothing should be
 * @internal
 */
export function compactionSplitIndex(messages: Message[], preservedMessages: number): number {
  const kept = resolvePreserveCount(messages.length, preservedMessages)
  let split = Math.min(messages.length - kept, messages.length - 1)
  if (split > 0 && splitsToolPair(messages, split)) {
    split = split < messages.length - 1 ? split + 1 : split - 1
  }
  return split > 1 ? split : 0
}

/** Returns the cut of a recency-only pass: the compaction cut, moved off a pinned last user message. */
function recencySplit(messages: Message[], preservedMessages: number): number {
  const split = compactionSplitIndex(messages, preservedMessages)
  const last = messages.length - 1
  const pinnedLast = split > 0 && split === last && messages[split]!.role === 'user' && isPinned(messages, split)
  if (!pinnedLast) return split
  // Placed behind a pinned last message, the summary would end the history, so one message stays.
  return split - (splitsToolPair(messages, split - 1) ? 2 : 1)
}

/** Whether a cut before `messages[index]` separates a tool result from the tool use before it. */
function splitsToolPair(messages: Message[], index: number): boolean {
  const hasToolResult = messages[index]!.content.some((block) => block instanceof ToolResultBlock)
  return hasToolResult && messages[index - 1]!.content.some((block) => block instanceof ToolUseBlock)
}

/**
 * Opens a compaction request whose selection starts with an assistant turn, since providers require a
 * conversation to start with a user message.
 */
const EXCERPT_OPENER = '[Earlier conversation omitted]'

/** Returns the indices of the messages in `selected` and their tool-pair partners, in history order. */
function withToolPairs(messages: Message[], selected: Message[]): number[] {
  const indexOf = new Map(messages.map((message, index) => [message, index]))
  const pairs = selected.map((message) => collectRemovableWithPair(messages, indexOf.get(message)!))
  const indices = new Set(pairs.flat().map((removable) => indexOf.get(removable)!))
  return [...indices].sort((left, right) => left - right)
}

/** Returns `selected` as `Model.summarize()` input: alternating roles, opening with a user turn. */
function compactionRequest(selected: Message[]): Message[] {
  const request = [...selected]
  repairAlternation(request)
  if (request[0]!.role === 'assistant') {
    request.unshift(new Message({ role: 'user', content: [new TextBlock(EXCERPT_OPENER)] }))
  }
  return request
}

/** Where a summary goes in the history, and its role. */
interface SummarySlot {
  start: number
  role: Role
}

const hasPinnedFlag = (message: Message): boolean => message.metadata?.custom?.pinned === true

/**
 * Returns the indices to summarize and where their summary goes, so no pinned message gains content, or undefined.
 *
 * A summary inside a pinned message would never be summarized again, so each pass would grow that message. The
 * summary goes at the first run of selected messages whose kept neighbors leave it a role: joining an unpinned
 * neighbor of that role, or standing alone between neighbors of the other role. Every other run is removed only
 * when that joins no pinned message and not message 0; the rest stay in the history.
 */
function summaryPlan(messages: Message[], indices: number[]): { indices: number[]; slot: SummarySlot } | undefined {
  const all = runs(indices)
  for (const slotRun of all) {
    const role = slotRole(messages, slotRun)
    if (!role) continue
    const kept = all.filter((run) => run === slotRun || removable(messages, run)).flat()
    return { indices: kept, slot: { start: slotRun[0]!, role } }
  }
  return undefined
}

/** Splits sorted `indices` into runs of consecutive indices. */
function runs(indices: number[]): number[][] {
  const result: number[][] = []
  for (const index of indices) {
    const last = result.at(-1)
    if (last && last.at(-1) === index - 1) last.push(index)
    else result.push([index])
  }
  return result
}

/** Whether removing `run` leaves alternating neighbors, or joins two unpinned ones other than message 0. */
function removable(messages: Message[], run: number[]): boolean {
  const start = run[0]!
  const after = messages[run.at(-1)! + 1]
  if (!after) return false
  if (start === 0) return after.role === 'user'
  const before = messages[start - 1]!
  if (before.role !== after.role) return true
  return start > 1 && !hasPinnedFlag(before) && !hasPinnedFlag(after)
}

/**
 * Returns the role of a summary in place of `run`, preferring to follow the message before it, if any fits.
 *
 * A role fits when no locked neighbor has it: a pinned neighbor, or message 0 when the run starts at 1.
 */
function slotRole(messages: Message[], run: number[]): Role | undefined {
  const start = run[0]!
  const before = messages[start - 1]
  const locked = [before, messages[run.at(-1)! + 1]].filter(
    (message): message is Message => !!message && hasPinnedFlag(message)
  )
  if (start === 1) locked.push(messages[0]!)
  if (!before) return locked.some((message) => message.role === 'user') ? undefined : 'user'
  const other: Role = before.role === 'assistant' ? 'user' : 'assistant'
  const roles: Role[] = locked.includes(before) ? [other, before.role] : [before.role, other]
  return roles.find((role) => !locked.some((message) => message.role === role))
}

/**
 * Replaces the messages at `indices` with `summary` at `slot`.
 *
 * Joining a kept message appends to it, which keeps its tracking id and block indices for stash refs. A user
 * summary leads the user message after it instead, so the conversation reads in order.
 */
function placeSummary(messages: Message[], indices: number[], slot: SummarySlot, summary: Message): void {
  const position = slot.start - indices.filter((index) => index < slot.start).length
  for (const index of [...indices].reverse()) messages.splice(index, 1)
  const carrier = carrierIndex(messages, position, slot.role)
  if (carrier === undefined) {
    const placed = summary.role === slot.role ? summary : new Message({ role: slot.role, content: summary.content })
    messages.splice(position, 0, placed)
  } else {
    const joined = messages[carrier]!
    messages[carrier] = new Message({
      role: joined.role,
      content: [...joined.content, ...summary.content],
      trackingId: joined.trackingId,
      ...(joined.metadata ? { metadata: joined.metadata } : {}),
    })
  }
  repairAlternation(messages)
}

/** Returns the index of the kept message the summary appends to, or undefined when it stands alone or leads. */
function carrierIndex(messages: Message[], position: number, role: Role): number | undefined {
  if (position > 1 && messages[position - 1]!.role === role) return position - 1
  if (position > 0 && role === 'assistant' && messages[position]?.role === role) return position
  return undefined
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
    if (!this._resolveModel(context.agent)) {
      logger.warn('no model available for summarization')
      return false
    }
    return super.apply(context)
  }

  /**
   * Returns the indices of the messages this pass summarizes, in history order, leaving out pinned ones.
   *
   * A recency-only pass takes the messages before the compaction cut; a target or threshold pass takes the
   * eligible messages and their tool pairs.
   */
  private async _select(context: ContextState): Promise<number[]> {
    const { messages } = context
    const recencyOnly = (this._target === undefined || this._target === '*') && this._threshold === undefined
    if (!recencyOnly) return withToolPairs(messages, await this._getEligibleMessages(context))
    const split = recencySplit(messages, this._preserveRecent)
    const selected = Array.from({ length: split }, (_, index) => index).filter((index) => !isPinned(messages, index))
    // After a pass behind a pinned message, the only unpinned message before the cut is that pass's summary.
    return selected.length > 1 ? selected : []
  }

  /** Replaces the selected messages with `model.summarize()`'s summary of them. */
  protected override async _applyPerMessage(context: ContextState): Promise<boolean> {
    const model = this._resolveModel(context.agent)
    if (!model) return false
    const selected = await this._select(context)
    if (selected.length === 0) return false
    const { messages, agent } = context
    const plan = summaryPlan(messages, selected)
    if (!plan) {
      logger.debug(`summarized=<${selected.length}> | no place outside pinned messages for the summary`)
      return false
    }
    const { indices, slot } = plan

    const request = compactionRequest(indices.map((index) => messages[index]!))
    const options = this._summarizeOptions(agent)
    // Server-side compaction replaces the start of the conversation, so a provider override only gets the
    // conversation's first messages, opening with its own user turn; any other selection gets the base summary.
    const prefix = indices.every((index, position) => index === position) && messages[0]!.role === 'user'
    const result = prefix
      ? await model.summarize(request, options)
      : await Model.prototype.summarize.call(model, request, options)
    unwrapInvocationContext(context.invocationContext)?.recordUsage(result.usage)
    if (result.summary.content.length === 0) {
      logger.warn(`summarized=<${indices.length}> | model summarize returned no summary | leaving history unchanged`)
      return false
    }

    placeSummary(messages, indices, slot, result.summary)
    logger.debug(`summarized=<${indices.length}> | compacted messages`)
    return true
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
    let result: SummarizeResult
    try {
      // A lone block is not the conversation's start, so it gets the base summary.
      result = await Model.prototype.summarize.call(model, blockSummaryRequest(content), {
        instructions: this._config.systemPrompt || DEFAULT_BLOCK_INSTRUCTIONS,
      })
    } catch (error) {
      logger.warn(
        `trackingId=<${message.trackingId}>, error=<${normalizeError(error).message}> | block summarization failed`
      )
      return null
    }
    unwrapInvocationContext(invocationContext)?.recordUsage(result.usage)
    return (result.summary.content[0] as TextBlock | undefined)?.text ?? null
  }

  /** Returns the `Model.summarize()` options of a message-level summary. */
  private _summarizeOptions(agent: LocalAgent): SummarizeOptions {
    return {
      // A server-side compaction must carry the live system prompt and tools, or the provider
      // treats it as a different conversation.
      ...(agent.systemPrompt !== undefined && { systemPrompt: agent.systemPrompt }),
      toolSpecs: agent.toolRegistry.list().map((tool) => tool.toolSpec),
      ...(this._config.systemPrompt !== undefined && { instructions: this._config.systemPrompt }),
    }
  }

  private _resolveModel(agent: LocalAgent): Model | undefined {
    return this._config.model ?? agent.auxModel
  }
}
