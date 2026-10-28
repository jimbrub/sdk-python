"""Base :meth:`Model.summarize` implementation and its result type."""

from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..types.content import ContentBlock, Message, Messages, SystemContentBlock
from ..types.event_loop import Usage
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec

if TYPE_CHECKING:
    from .model import Model

# Subject to change based on benchmarking.
_DEFAULT_SUMMARIZE_INSTRUCTIONS = (
    "Stop. Do not continue the task and do not call any tools. Summarize the conversation so far for a fresh "
    "context window that will replace it. Preserve key data, values, and identifiers; decisions and conclusions; "
    "errors (if present); and the context needed to continue the work. Output only the summary text."
)


@dataclass
class SummarizeResult:
    """Result of a :meth:`Model.summarize` call.

    Attributes:
        summary: The message that replaces the summarized messages. Its content is empty when the model wrote
            no summary.
        usage: Token usage of the summary call, ``None`` when the provider reports none.
    """

    summary: Message
    usage: Usage | None = None


async def _summarize_with_stream(
    model: "Model",
    messages: Messages,
    *,
    tool_specs: list[ToolSpec] | None = None,
    system_prompt: str | None = None,
    system_prompt_content: list[SystemContentBlock] | None = None,
    instructions: str | None = None,
) -> SummarizeResult:
    """Summarize ``messages`` with a :meth:`Model.stream` call on ``messages`` plus one trailing instruction."""
    request = _append_instruction(messages, instructions or _DEFAULT_SUMMARIZE_INSTRUCTIONS)
    events = model.stream(request, tool_specs, system_prompt, system_prompt_content=system_prompt_content)
    summary, usage = await _read_summary(events)
    content = [ContentBlock(text=summary)] if summary else []
    return SummarizeResult(summary=Message(role="user", content=content), usage=usage)


async def _read_summary(events: AsyncIterable[StreamEvent]) -> tuple[str, Usage | None]:
    """Return the reply's text, one line per block, or "" unless the reply is complete, and the call's usage."""
    blocks: list[str] = [""]
    stop_reason = None
    usage: Usage | None = None
    async for event in events:
        if "contentBlockDelta" in event:
            blocks[-1] += event["contentBlockDelta"]["delta"].get("text", "")
        elif "contentBlockStop" in event:
            blocks.append("")
        elif "messageStop" in event:
            stop_reason = event["messageStop"]["stopReason"]
        elif "metadata" in event:
            usage = event["metadata"].get("usage", usage)

    if stop_reason not in ("end_turn", "stop_sequence"):
        return "", usage
    return "\n".join(block.strip() for block in blocks if block.strip()), usage


def _append_instruction(messages: Messages, instruction: str) -> Messages:
    """Return ``messages`` with ``instruction`` as the final user turn, keeping only each message's role and content."""
    request = [Message(role=message["role"], content=message["content"]) for message in messages]
    block = ContentBlock(text=instruction)
    if request and request[-1]["role"] == "user":
        request[-1] = Message(role="user", content=[*request[-1]["content"], block])
    else:
        request.append(Message(role="user", content=[block]))
    return request
