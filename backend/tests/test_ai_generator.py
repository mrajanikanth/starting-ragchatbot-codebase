"""
Tests for AIGenerator tool-calling behaviour in ai_generator.py.

Covers: direct text responses, tool-use dispatch, second-call message
structure, no-tools-in-second-call invariant, and exception propagation.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_text_response(text="Hello world"):
    """Simulate a Claude response that ends with plain text."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def make_tool_use_response(tool_name="search_course_content",
                           tool_input=None,
                           tool_id="toolu_abc123"):
    """Simulate a Claude response that requests a tool call."""
    tool_input = tool_input or {"query": "Python basics"}

    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = tool_input

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


@pytest.fixture
def generator():
    """AIGenerator with a mocked Anthropic client."""
    with patch("ai_generator.anthropic.Anthropic"):
        gen = AIGenerator("fake-key", "claude-sonnet-4-6")
    gen.client = MagicMock()
    return gen


# ---------------------------------------------------------------------------
# Direct (non-tool) responses
# ---------------------------------------------------------------------------

class TestDirectResponse:
    def test_returns_text_when_stop_reason_is_end_turn(self, generator):
        """generate_response() returns the text block when no tool is needed."""
        generator.client.messages.create.return_value = make_text_response("Direct answer.")

        result = generator.generate_response(query="What is Python?")

        assert result == "Direct answer."

    def test_api_called_with_user_query(self, generator):
        """The user's query appears in the messages list sent to Claude."""
        generator.client.messages.create.return_value = make_text_response()

        generator.generate_response(query="What is Python?")

        kwargs = generator.client.messages.create.call_args[1]
        messages = kwargs["messages"]
        assert any(
            m["role"] == "user" and "What is Python?" in m["content"]
            for m in messages
        )

    def test_api_called_with_system_prompt(self, generator):
        """System prompt is always passed to Claude."""
        generator.client.messages.create.return_value = make_text_response()

        generator.generate_response(query="Q")

        kwargs = generator.client.messages.create.call_args[1]
        assert "system" in kwargs
        assert kwargs["system"]  # non-empty

    def test_api_called_with_tools_when_provided(self, generator):
        """When tools are supplied, they appear in the first API call."""
        generator.client.messages.create.return_value = make_text_response()
        tools = [{"name": "search_course_content", "description": "...", "input_schema": {}}]

        generator.generate_response(query="Q", tools=tools)

        kwargs = generator.client.messages.create.call_args[1]
        assert "tools" in kwargs
        assert kwargs["tools"] == tools

    def test_api_not_called_with_tools_when_none_provided(self, generator):
        """When no tools are passed, the tools key is absent from the API call."""
        generator.client.messages.create.return_value = make_text_response()

        generator.generate_response(query="Q")

        kwargs = generator.client.messages.create.call_args[1]
        assert "tools" not in kwargs

    def test_conversation_history_included_in_system_prompt(self, generator):
        """Conversation history is appended to the system prompt."""
        generator.client.messages.create.return_value = make_text_response()

        generator.generate_response(
            query="Q",
            conversation_history="User: Hi\nAssistant: Hello"
        )

        kwargs = generator.client.messages.create.call_args[1]
        assert "User: Hi" in kwargs["system"]

    def test_exception_from_first_api_call_propagates(self, generator):
        """An API exception on the first call propagates (causes 'query failed')."""
        generator.client.messages.create.side_effect = Exception("Connection refused")

        with pytest.raises(Exception, match="Connection refused"):
            generator.generate_response(query="Q")

    def test_empty_content_raises_value_error(self, generator):
        """ValueError is raised when the API returns no text content blocks."""
        bad_response = MagicMock()
        bad_response.stop_reason = "end_turn"
        bad_response.content = []
        generator.client.messages.create.return_value = bad_response

        with pytest.raises(ValueError, match="No text block found"):
            generator.generate_response(query="Q")


# ---------------------------------------------------------------------------
# Tool-use dispatch and second API call
# ---------------------------------------------------------------------------

class TestToolExecution:
    def test_tool_manager_execute_tool_is_called_on_tool_use(self, generator):
        """When Claude requests a tool, execute_tool is called on the tool_manager."""
        tool_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "Python basics"},
            tool_id="tid_1",
        )
        final_response = make_text_response("Here are the results.")

        generator.client.messages.create.side_effect = [tool_response, final_response]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search results text"

        generator.generate_response(
            query="Tell me about Python",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics",
        )

    def test_tool_execution_result_included_in_second_call(self, generator):
        """The tool result appears as a tool_result block in the second API call."""
        tool_response = make_tool_use_response(tool_id="tid_1")
        final_response = make_text_response("Final answer")

        generator.client.messages.create.side_effect = [tool_response, final_response]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Found: lesson 1 content"

        generator.generate_response(
            query="Q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        messages = second_kwargs["messages"]
        tool_result_msg = next(
            (m for m in messages if m.get("role") == "user" and isinstance(m.get("content"), list)),
            None,
        )
        assert tool_result_msg is not None, "No user message with tool_result found in second call"
        result_block = tool_result_msg["content"][0]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == "tid_1"
        assert result_block["content"] == "Found: lesson 1 content"

    def test_second_api_call_has_no_tools_key(self, generator):
        """Second API call deliberately omits 'tools' to force a text answer."""
        tool_response = make_tool_use_response()
        final_response = make_text_response("Done")

        generator.client.messages.create.side_effect = [tool_response, final_response]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        generator.generate_response(
            query="Q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        assert "tools" not in second_kwargs, (
            "'tools' key must be absent from the second API call so Claude cannot loop"
        )

    def test_generate_response_returns_text_from_second_call(self, generator):
        """generate_response() returns the text from the second (final) Claude call."""
        tool_response = make_tool_use_response()
        final_response = make_text_response("Final synthesised answer")

        generator.client.messages.create.side_effect = [tool_response, final_response]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        result = generator.generate_response(
            query="Q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert result == "Final synthesised answer"

    def test_second_call_messages_include_assistant_tool_use_turn(self, generator):
        """The assistant's tool-use turn is re-sent as context in the second call."""
        tool_response = make_tool_use_response(tool_id="tid_2")
        final_response = make_text_response("OK")

        generator.client.messages.create.side_effect = [tool_response, final_response]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "R"

        generator.generate_response(
            query="Q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        messages = second_kwargs["messages"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1, "Expected exactly one assistant message in second call"

    def test_exception_from_second_api_call_propagates(self, generator):
        """An exception in the second API call propagates (causes 'query failed')."""
        tool_response = make_tool_use_response()

        generator.client.messages.create.side_effect = [
            tool_response,
            Exception("Rate limit exceeded"),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        with pytest.raises(Exception, match="Rate limit exceeded"):
            generator.generate_response(
                query="Q",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

    def test_exception_from_tool_manager_propagates(self, generator):
        """An exception raised by execute_tool propagates (causes 'query failed')."""
        tool_response = make_tool_use_response()
        generator.client.messages.create.return_value = tool_response

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = RuntimeError("DB is down")

        with pytest.raises(RuntimeError, match="DB is down"):
            generator.generate_response(
                query="Q",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

    def test_empty_content_in_second_response_raises_value_error(self, generator):
        """ValueError when the second (final) response has no text content blocks."""
        tool_response = make_tool_use_response()
        bad_final = MagicMock()
        bad_final.stop_reason = "end_turn"
        bad_final.content = []

        generator.client.messages.create.side_effect = [tool_response, bad_final]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        with pytest.raises(ValueError, match="No text block found"):
            generator.generate_response(
                query="Q",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

    def test_no_tool_call_made_when_stop_reason_is_end_turn(self, generator):
        """execute_tool is never called when Claude's stop_reason is end_turn."""
        generator.client.messages.create.return_value = make_text_response("Direct")
        mock_tool_manager = MagicMock()

        generator.generate_response(
            query="Q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_not_called()

    def test_two_api_calls_total_during_tool_use_flow(self, generator):
        """Exactly two API calls are made: one with tools, one without."""
        tool_response = make_tool_use_response()
        final_response = make_text_response("Done")

        generator.client.messages.create.side_effect = [tool_response, final_response]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "R"

        generator.generate_response(
            query="Q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert generator.client.messages.create.call_count == 2

    def test_skips_non_text_blocks_to_find_first_text_block(self, generator):
        """
        _extract_text skips non-TextBlock content blocks (e.g. ToolUseBlocks) and
        returns the first actual text.  Previously content[0].text raised AttributeError
        when the first block wasn't a TextBlock.
        """
        from anthropic.types import ToolUseBlock
        non_text_block = MagicMock()
        non_text_block.type = "tool_use"

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The answer is here."

        mixed_response = MagicMock()
        mixed_response.stop_reason = "end_turn"
        mixed_response.content = [non_text_block, text_block]
        generator.client.messages.create.return_value = mixed_response

        result = generator.generate_response(query="Q")
        assert result == "The answer is here."

    def test_raises_value_error_when_only_non_text_blocks_present(self, generator):
        """ValueError when the response has content but no TextBlock at all."""
        non_text_block = MagicMock()
        non_text_block.type = "tool_use"

        bad_response = MagicMock()
        bad_response.stop_reason = "end_turn"
        bad_response.content = [non_text_block]
        generator.client.messages.create.return_value = bad_response

        with pytest.raises(ValueError, match="No text block found"):
            generator.generate_response(query="Q")


# ---------------------------------------------------------------------------
# Regression: model ID must be a known-valid Anthropic model
# ---------------------------------------------------------------------------

class TestModelConfiguration:
    def test_model_is_not_deprecated_sonnet_4_date_format(self):
        """
        Regression: 'claude-sonnet-4-20250514' is not a valid Anthropic model ID.
        The Anthropic API returns a 404 NotFoundError for this model, which
        propagates as HTTP 500 → 'query failed' in the frontend for every query.
        The correct model is 'claude-sonnet-4-6' (or 'claude-sonnet-4-5').
        """
        from config import config

        deprecated_model = "claude-sonnet-4-20250514"
        assert config.ANTHROPIC_MODEL != deprecated_model, (
            f"Model is still set to the deprecated ID '{deprecated_model}'. "
            "Change config.ANTHROPIC_MODEL to a valid model such as 'claude-sonnet-4-6'."
        )

    def test_model_follows_valid_naming_convention(self):
        """Model ID must match a known Claude 4 naming pattern."""
        from config import config

        valid_prefixes = ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4")
        assert any(config.ANTHROPIC_MODEL.startswith(p) for p in valid_prefixes), (
            f"ANTHROPIC_MODEL '{config.ANTHROPIC_MODEL}' does not match a known Claude 4 pattern. "
            f"Expected one of: {valid_prefixes}"
        )
