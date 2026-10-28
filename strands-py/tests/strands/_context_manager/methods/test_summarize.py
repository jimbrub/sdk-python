"""Tests for the summarize method module."""

from strands._context_manager.methods.summarize import _tool_result_to_content_blocks
from strands.types.content import ContentBlock


class TestToolResultToContentBlocks:
    """Tests for converting tool result content to content blocks."""

    def test_converts_text_content(self):
        content = [{"text": "hello"}]
        tru_result = _tool_result_to_content_blocks(content)
        assert tru_result == [ContentBlock(text="hello")]

    def test_passes_through_non_text_content(self):
        content = [{"image": {"format": "png", "source": {"bytes": b"data"}}}]
        tru_result = _tool_result_to_content_blocks(content)
        assert len(tru_result) == 1

    def test_handles_mixed_content(self):
        content = [
            {"text": "first"},
            {"image": {"format": "png", "source": {"bytes": b"data"}}},
            {"text": "second"},
        ]
        tru_result = _tool_result_to_content_blocks(content)
        assert len(tru_result) == 3
        assert tru_result[0] == ContentBlock(text="first")
        assert tru_result[2] == ContentBlock(text="second")


class TestFormatSummarized:
    """Tests for the summary marker."""

    def test_format_summarized_uses_expected_prefix(self):
        from strands._context_manager.methods.summarize import _format_summarized

        assert _format_summarized("test", 100).startswith("[Summarized:")


class TestToolResultToContentBlocksVideo:
    """Tests for video content in tool_result_to_content_blocks."""

    def test_renders_video_as_a_placeholder(self):
        content = [{"text": "clip"}, {"video": {"format": "mp4", "source": {"bytes": b"data"}}}]
        tru_result = _tool_result_to_content_blocks(content)
        assert tru_result == [ContentBlock(text="clip"), ContentBlock(text="[video omitted]")]


class TestToolResultToContentBlocksJson:
    """Tests for JSON content in tool_result_to_content_blocks."""

    def test_converts_json_content_to_text(self):
        content = [{"json": {"key": "value", "count": 42}}]
        tru_result = _tool_result_to_content_blocks(content)
        assert len(tru_result) == 1
        assert tru_result[0]["text"] == '{\n  "key": "value",\n  "count": 42\n}'

    def test_handles_mixed_json_and_text(self):
        content = [
            {"text": "plain text"},
            {"json": {"nested": True}},
        ]
        tru_result = _tool_result_to_content_blocks(content)
        assert len(tru_result) == 2
        assert tru_result[0] == ContentBlock(text="plain text")
        assert tru_result[1]["text"] == '{\n  "nested": true\n}'
