import { describe, it, expect } from 'vitest'
import { toolResultToContentBlocks } from '../summarize.js'
import { ImageBlock, VideoBlock } from '../../../types/media.js'
import { JsonBlock, TextBlock } from '../../../types/messages.js'

describe('toolResultToContentBlocks', () => {
  it('renders JSON as text and video as a placeholder, and passes text and images through', () => {
    const bytes = new Uint8Array([1])
    const image = new ImageBlock({ format: 'png', source: { bytes } })
    const content = [
      new TextBlock('out'),
      new JsonBlock({ json: { ok: true } }),
      new VideoBlock({ format: 'mp4', source: { bytes } }),
      image,
    ]

    expect(toolResultToContentBlocks(content)).toEqual([
      new TextBlock('out'),
      new TextBlock(JSON.stringify({ ok: true }, null, 2)),
      new TextBlock('[video omitted]'),
      image,
    ])
  })
})
