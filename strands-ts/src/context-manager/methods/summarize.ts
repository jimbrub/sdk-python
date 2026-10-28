/**
 * Summarize reduction method.
 *
 * Summaries come from `Model.summarize()`; these helpers configure it, format the summary markers, and convert tool
 * result content for it. Strategies handle selection and placement.
 */

import type { Model } from '../../models/model.js'
import { JsonBlock, Message, TextBlock } from '../../types/messages.js'
import { VideoBlock } from '../../types/media.js'
import type { ContentBlock, ToolResultContent } from '../../types/messages.js'

export const SUMMARIZED_PREFIX = '[Summarized:'

/**
 * Format a `[Summarized: ...]` marker with an optional summary body.
 *
 * @internal
 */
export function formatSummarized(description: string, tokens: number, summary?: string): string {
  const header = `${SUMMARIZED_PREFIX} ${description}, ~${tokens.toLocaleString('en-US')} tokens]`
  return summary ? `${header}\n\n${summary}` : header
}

/**
 * Configuration for the summarize method.
 *
 * @experimental
 */
export interface SummarizeConfig {
  /** Model used for summarization. Resolution order: `this model > agent.auxModel > agent.model`. */
  model?: Model

  /** Custom summarization instruction, replacing the default one sent to `Model.summarize()`. */
  systemPrompt?: string
}

/**
 * Returns the `Model.summarize()` input for one block: its content as a delimited user turn.
 *
 * @internal
 */
export function blockSummaryRequest(content: ContentBlock[]): Message[] {
  return [new Message({ role: 'user', content: [new TextBlock('<content>'), ...content, new TextBlock('</content>')] })]
}

/**
 * Converts ToolResultContent[] to ContentBlock[] `Model.summarize()` accepts, rendering JSON as text and video as a
 * placeholder.
 */
export function toolResultToContentBlocks(content: ToolResultContent[]): ContentBlock[] {
  return content.map((block) => {
    if (block instanceof JsonBlock) return new TextBlock(JSON.stringify(block.json, null, 2))
    if (block instanceof VideoBlock) return new TextBlock('[video omitted]')
    return block
  })
}
