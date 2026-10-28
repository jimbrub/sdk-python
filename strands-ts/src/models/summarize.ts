import { type ContentBlock, Message, type StopReason, type SystemPrompt, TextBlock } from '../types/messages.js'
import { CitationsBlock } from '../types/citations.js'
import { MaxTokensError, ModelError } from '../errors.js'
import type { ToolSpec } from '../tools/types.js'
import type { ModelStreamEvent, Usage } from './streaming.js'
import type { Model, StreamAggregatedResult } from './model.js'

/**
 * Options for a {@link Model.summarize} call.
 */
export interface SummarizeOptions {
  /** The live conversation's system prompt, passed through unchanged. */
  systemPrompt?: SystemPrompt

  /** The live conversation's tool specifications, passed through unchanged. */
  toolSpecs?: ToolSpec[]

  /** Summarization instruction that replaces the default one. */
  instructions?: string
}

/**
 * Result of a {@link Model.summarize} call.
 */
export interface SummarizeResult {
  /** The message that replaces the summarized messages. Its content is empty when the model wrote no summary. */
  summary: Message

  /** Token usage of the summary call, absent when the provider reports none. */
  usage?: Usage
}

/**
 * Default {@link Model.summarize} instruction. Subject to change based on benchmarking.
 *
 * @internal
 */
export const DEFAULT_SUMMARIZE_INSTRUCTIONS =
  'Stop. Do not continue the task and do not call any tools. Summarize the conversation so far for a fresh ' +
  'context window that will replace it. Preserve key data, values, and identifiers; decisions and conclusions; ' +
  'errors (if present); and the context needed to continue the work. Output only the summary text.'

/**
 * The base {@link Model.summarize} implementation: a {@link Model.streamAggregated} call with `messages` plus one
 * trailing instruction.
 *
 * @internal
 * @param model - The model to stream from
 * @param messages - The messages to summarize, left unmodified
 * @param options - The live system prompt and tools, and a custom summarization instruction
 * @returns The summary message and the call's token usage
 */
export async function summarizeWithStream(
  model: Model,
  messages: Message[],
  options: SummarizeOptions = {}
): Promise<SummarizeResult> {
  const { instructions, ...streamOptions } = options
  const request = appendInstruction(messages, instructions || DEFAULT_SUMMARIZE_INSTRUCTIONS)
  const { reply, ...usage } = await readSummary(model.streamAggregated(request, streamOptions))
  const summary = (reply?.content ?? [])
    .flatMap(blockText)
    .map((text) => text.trim())
    .filter(Boolean)
    .join('\n')
  return { summary: new Message({ role: 'user', content: summary ? [new TextBlock(summary)] : [] }), ...usage }
}

/** Reads a {@link Model.streamAggregated} call; `reply` is set only when the reply is complete. */
async function readSummary(
  stream: AsyncGenerator<ModelStreamEvent | ContentBlock, StreamAggregatedResult, undefined>
): Promise<{ reply?: Message; usage?: Usage }> {
  const result: { reply?: Message; usage?: Usage } = {}
  try {
    const { stopReason, message } = await drain(stream, result)
    if (COMPLETE_STOP_REASONS.has(stopReason)) result.reply = message
  } catch (error) {
    if (!writesNoSummary(error)) throw error
  }
  return result
}

/** Drains `stream`, keeping its last reported usage in `state`, and returns its result. */
async function drain(
  stream: AsyncGenerator<ModelStreamEvent | ContentBlock, StreamAggregatedResult, undefined>,
  state: { usage?: Usage }
): Promise<StreamAggregatedResult> {
  let next = await stream.next()
  while (!next.done) {
    if (next.value.type === 'modelMetadataEvent' && next.value.usage) state.usage = next.value.usage
    next = await stream.next()
  }
  return next.value
}

/** Stop reasons that end a complete reply. */
const COMPLETE_STOP_REASONS: ReadonlySet<StopReason> = new Set(['endTurn', 'stopSequence'])

/** Whether `error` is a reply cut off at the output limit, or stopped on a tool call whose input does not parse. */
function writesNoSummary(error: unknown): boolean {
  return error instanceof MaxTokensError || (error instanceof ModelError && error.cause instanceof SyntaxError)
}

/** Returns the generated text of a text or citations block. */
function blockText(block: ContentBlock): string[] {
  if (block instanceof TextBlock) return [block.text]
  if (block instanceof CitationsBlock) return block.content.map((content) => content.text)
  return []
}

/** Returns `messages` with `instruction` as the final user turn, keeping only each message's role and content. */
function appendInstruction(messages: Message[], instruction: string): Message[] {
  const request = messages.map((message) => new Message({ role: message.role, content: message.content }))
  const block = new TextBlock(instruction)
  const last = request.at(-1)
  if (last?.role === 'user') {
    request[request.length - 1] = new Message({ role: 'user', content: [...last.content, block] })
  } else {
    request.push(new Message({ role: 'user', content: [block] }))
  }
  return request
}
