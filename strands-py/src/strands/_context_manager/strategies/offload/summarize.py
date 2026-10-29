"""Summarize strategy — replaces oversized content with LLM-generated summaries."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

from ....agent.conversation_manager.compression.pin_message import _has_pinned_flag, is_pinned
from ....event_loop.streaming import _record_usage
from ....models._summarize import _tool_result_to_content_blocks
from ....models.model import Model
from ....types.content import ContentBlock, Message, Messages, Role, _ensure_tracking_id
from ....types.tools import ToolResult, ToolResultContent
from ...methods.summarize import SummarizeConfig, _format_summarized
from ...stash import _format_stash_refs
from ...types import ContextState, is_tool_result_block, is_tool_use_block
from .base import (
    BaseOffloadStrategy,
    OffloadConditions,
    OffloadTarget,
    _build_conditions,
    _collect_removable_with_pair,
    _repair_alternation,
    _resolve_preserve_count,
)

if TYPE_CHECKING:
    from ....agent.agent import Agent
    from ....agent.invocation_context import InvocationContext

logger = logging.getLogger(__name__)

# Opens a compaction request whose selection starts with an assistant turn, since providers require a
# conversation to start with a user message.
_EXCERPT_OPENER = "[Earlier conversation omitted]"


def _compaction_split_index(messages: Messages, preserved_messages: int | float) -> int:
    """Return the index of the first message a ``Model.summarize()`` call keeps verbatim.

    The cut never separates a ``toolUse`` from its ``toolResult``: it moves forward to summarize the pair together,
    or back when the ``toolResult`` is the last message, which the next model call responds to and which is never
    summarized. A one-message prefix is not cut: after a pass it is that pass's summary, and re-summarizing it
    alone frees nothing.

    Args:
        messages: Full conversation history.
        preserved_messages: Count or ratio (between 0 and 1 exclusive) of trailing messages to keep; the last
            message is always kept, so 0 and 1 cut at the same place.

    Returns:
        The split index; messages before it may be summarized. 0 when nothing should be.
    """
    kept = _resolve_preserve_count(len(messages), preserved_messages)
    split = min(len(messages) - kept, len(messages) - 1)
    if split > 0 and _splits_tool_pair(messages, split):
        split = split + 1 if split < len(messages) - 1 else split - 1
    return split if split > 1 else 0


def _recency_split(messages: Messages, preserved_messages: int | float) -> int:
    """Return the cut of a recency-only pass: the compaction cut, moved off a pinned last user message."""
    split = _compaction_split_index(messages, preserved_messages)
    pinned_last = 0 < split == len(messages) - 1 and messages[split]["role"] == "user" and is_pinned(messages, split)
    if not pinned_last:
        return split
    # Placed behind a pinned last message, the summary would end the history, so one message stays.
    return split - (2 if _splits_tool_pair(messages, split - 1) else 1)


def _splits_tool_pair(messages: Messages, index: int) -> bool:
    """Whether a cut before ``messages[index]`` separates a tool result from the tool use before it."""
    has_tool_result = any(is_tool_result_block(block) for block in messages[index]["content"])
    return has_tool_result and any(is_tool_use_block(block) for block in messages[index - 1]["content"])


def _with_tool_pairs(messages: Messages, selected: Messages) -> list[int]:
    """Return the indices of the messages in ``selected`` and their tool-pair partners, in history order."""
    index_map = {id(message): index for index, message in enumerate(messages)}
    pairs = (_collect_removable_with_pair(messages, index_map[id(message)]) for message in selected)
    return sorted({index_map[id(removable)] for pair in pairs for removable in pair})


def _compaction_request(selected: Messages) -> Messages:
    """Return ``selected`` as ``Model.summarize()`` input: alternating roles, opening with a user turn."""
    request = list(selected)
    _repair_alternation(request)
    if request[0]["role"] == "assistant":
        request.insert(0, Message(role="user", content=[ContentBlock(text=_EXCERPT_OPENER)]))
    return request


def _summary_plan(messages: Messages, indices: list[int]) -> tuple[list[int], tuple[int, Role]] | None:
    """Return the indices to summarize and where their summary goes, so no pinned message gains content, or None.

    A summary inside a pinned message would never be summarized again, so each pass would grow that message. The
    summary goes at the first run of selected messages whose kept neighbors leave it a role: joining an unpinned
    neighbor of that role, or standing alone between neighbors of the other role. Every other run is removed only
    when that joins no pinned message and not message 0; the rest stay in the history.
    """
    runs = _runs(indices)
    for slot_run in runs:
        role = _slot_role(messages, slot_run)
        if role is not None:
            kept = [index for run in runs if run is slot_run or _removable(messages, run) for index in run]
            return kept, (slot_run[0], role)
    return None


def _runs(indices: list[int]) -> list[list[int]]:
    """Split sorted ``indices`` into runs of consecutive indices."""
    runs: list[list[int]] = []
    for index in indices:
        if runs and runs[-1][-1] == index - 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _removable(messages: Messages, run: list[int]) -> bool:
    """Whether removing ``run`` leaves alternating neighbors, or joins two unpinned ones other than message 0."""
    start, end = run[0], run[-1] + 1
    if end >= len(messages):
        return False
    after = messages[end]
    if start == 0:
        return after["role"] == "user"
    before = messages[start - 1]
    if before["role"] != after["role"]:
        return True
    return start > 1 and not _has_pinned_flag(before) and not _has_pinned_flag(after)


def _slot_role(messages: Messages, run: list[int]) -> Role | None:
    """Return the role of a summary in place of ``run``, preferring to follow the message before it, if any fits.

    A role fits when no locked neighbor has it: a pinned neighbor, or message 0 when the run starts at 1.
    """
    start, end = run[0], run[-1] + 1
    before = messages[start - 1] if start > 0 else None
    after = messages[end] if end < len(messages) else None
    locked = [message for message in (before, after) if message is not None and _has_pinned_flag(message)]
    if start == 1:
        locked.append(messages[0])
    if before is None:
        return None if any(message["role"] == "user" for message in locked) else "user"
    other: Role = "user" if before["role"] == "assistant" else "assistant"
    roles = (other, before["role"]) if any(message is before for message in locked) else (before["role"], other)
    return next((role for role in roles if not any(message["role"] == role for message in locked)), None)


def _place_summary(messages: Messages, indices: list[int], slot: tuple[int, Role], summary: Message) -> None:
    """Replace the messages at ``indices`` with ``summary`` at ``slot``.

    Joining a kept message appends to it, which keeps its tracking id and block indices for stash refs. A user
    summary leads the user message after it instead, so the conversation reads in order.
    """
    start, role = slot
    position = start - sum(1 for index in indices if index < start)
    for index in reversed(indices):
        del messages[index]
    carrier = _carrier(messages, position, role)
    if carrier is None:
        placed = summary.copy() if summary["role"] == role else Message(role=role, content=list(summary["content"]))
        _ensure_tracking_id(placed)
        messages.insert(position, placed)
    else:
        joined = messages[carrier].copy()
        joined["content"] = [*joined["content"], *summary["content"]]
        messages[carrier] = joined
    _repair_alternation(messages)


def _carrier(messages: Messages, position: int, role: Role) -> int | None:
    """Return the index of the kept message the summary appends to, or None when it stands alone or leads."""
    if position > 1 and messages[position - 1]["role"] == role:
        return position - 1
    if 0 < position < len(messages) and role == "assistant" and messages[position]["role"] == role:
        return position
    return None


class SummarizeStrategy(BaseOffloadStrategy):
    """Summarize strategy — replaces oversized content with LLM-generated summaries."""

    @property
    def name(self) -> str:
        """Strategy name."""
        return "offload:summarize"

    def __init__(
        self,
        target: OffloadTarget | None = None,
        config: SummarizeConfig | None = None,
        conditions: OffloadConditions | None = None,
    ) -> None:
        super().__init__(target, conditions)
        self._config: SummarizeConfig = config or {}

    def when(
        self,
        *,
        threshold: int | None = None,
        utilization: float | None = None,
        preserve_recent: int | float = 0,
    ) -> SummarizeStrategy:
        """Return a new instance with the given conditions applied."""
        return SummarizeStrategy(
            self._target,
            self._config,
            _build_conditions(threshold=threshold, utilization=utilization, preserve_recent=preserve_recent),
        )

    async def apply(self, context: ContextState) -> bool:
        """Apply summarization. Returns False if no model is available."""
        if not self._resolve_model(context.agent):
            logger.warning("strategy=<%s> | no model available for summarization", self.name)
            return False
        return await super().apply(context)

    async def _select(self, context: ContextState) -> list[int]:
        """Return the indices of the messages this pass summarizes, in history order, leaving out pinned ones.

        A recency-only pass takes the messages before the compaction cut; a target or threshold pass takes the
        eligible messages and their tool pairs.
        """
        messages = context.messages
        if self._target not in (None, "*") or self._threshold is not None:
            return _with_tool_pairs(messages, await self._get_eligible_messages(context))
        split = _recency_split(messages, self._preserve_recent)
        selected = [index for index in range(split) if not is_pinned(messages, index)]
        # After a pass behind a pinned message, the only unpinned message before the cut is that pass's summary.
        return selected if len(selected) > 1 else []

    async def _apply_per_message(self, context: ContextState) -> bool:
        """Replace the selected messages with ``model.summarize()``'s summary of them."""
        model = self._resolve_model(context.agent)
        if not model:
            return False
        indices = await self._select(context)
        if not indices:
            return False
        messages = context.messages
        plan = _summary_plan(messages, indices)
        if plan is None:
            logger.debug("summarized=<%s> | no place outside pinned messages for the summary", len(indices))
            return False
        indices, slot = plan
        agent = context.agent

        # Server-side compaction replaces the start of the conversation, so a provider override only gets the
        # conversation's first messages, opening with its own user turn; any other selection gets the base summary.
        prefix = indices == list(range(len(indices))) and messages[0]["role"] == "user"
        summarize = model.summarize if prefix else functools.partial(Model.summarize, model)
        result = await summarize(
            _compaction_request([messages[index] for index in indices]),
            # A server-side compaction must carry the live system prompt and tools, or the provider
            # treats it as a different conversation.
            tool_specs=agent.tool_registry.get_all_tool_specs(),
            system_prompt=agent.system_prompt,
            system_prompt_content=agent.system_prompt_content,
            instructions=self._config.get("system_prompt"),
        )
        if result.usage:
            _record_usage(context.invocation_context, result.usage)
        if not result.summary["content"]:
            logger.warning(
                "summarized=<%s> | model summarize returned no summary | leaving history unchanged", len(indices)
            )
            return False

        _place_summary(messages, indices, slot, result.summary)
        logger.debug("summarized=<%s> | compacted messages", len(indices))
        return True

    async def _replace_block(
        self,
        block: ContentBlock,
        tokens: int,
        message: Message,
        agent: Agent,
        stash_refs: list[str],
        invocation_context: InvocationContext | None = None,
    ) -> ContentBlock | None:
        model = self._resolve_model(agent)
        if not model:
            return None

        refs = _format_stash_refs(stash_refs)

        if "toolResult" in block:
            tool_result = block["toolResult"]
            content_blocks = _tool_result_to_content_blocks(tool_result["content"])
            summary = await self._try_summarize(content_blocks, model, message, invocation_context)
            if not summary:
                return None

            logger.debug("tool_use_id=<%s>, tokens=<%s> | summarized tool result", tool_result["toolUseId"], tokens)
            marker = _format_summarized("tool result", tokens, summary) + refs
            summarized_content: list[ToolResultContent] = [{"text": marker}]
            return ContentBlock(
                toolResult=ToolResult(
                    toolUseId=tool_result["toolUseId"],
                    status=tool_result["status"],
                    content=summarized_content,
                )
            )

        if "text" in block:
            summary = await self._try_summarize([ContentBlock(text=block["text"])], model, message, invocation_context)
            if not summary:
                return None

            logger.debug("tracking_id=<%s>, tokens=<%s> | summarized text block", message.get("tracking_id"), tokens)
            marker = _format_summarized("text block", tokens, summary) + refs
            return ContentBlock(text=marker)

        if "image" in block or "document" in block:
            summary = await self._try_summarize([block], model, message, invocation_context)
            if summary:
                logger.debug(
                    "tracking_id=<%s>, tokens=<%s> | summarized media block", message.get("tracking_id"), tokens
                )
                marker = _format_summarized("media block", tokens, summary) + refs
                return ContentBlock(text=marker)

        logger.debug("tracking_id=<%s>, tokens=<%s> | offloaded media block", message.get("tracking_id"), tokens)
        return ContentBlock(text=f"[Offloaded: ~{tokens} tokens]{refs}")

    async def _try_summarize(
        self,
        content: list[ContentBlock],
        model: Model,
        message: Message,
        invocation_context: InvocationContext | None = None,
    ) -> str | None:
        """Summarize one block; a failure is logged and leaves the block unsummarized so the pass continues."""
        try:
            # A lone block is not the conversation's start, so it gets the base summary.
            result = await Model.summarize(
                model,
                [Message(role="user", content=content)],
                instructions=self._config.get("system_prompt"),
            )
        except Exception as error:
            logger.warning(
                "tracking_id=<%s>, error=<%s> | block summarization failed", message.get("tracking_id"), error
            )
            return None
        if result.usage:
            _record_usage(invocation_context, result.usage)
        content = result.summary["content"]
        return content[0]["text"] if content else None

    def _resolve_model(self, agent: Agent) -> Model | None:
        return self._config.get("model") or agent.aux_model
