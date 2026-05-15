"""
Tests for RAGSystem.query() in rag_system.py.

Covers: tool/tool_manager wiring, return-value shape, session history
integration, source harvesting/reset, and exception propagation paths
that manifest as 'query failed' in the frontend.
"""
import pytest
from unittest.mock import MagicMock, patch

from rag_system import RAGSystem


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def rag():
    """RAGSystem with all heavy dependencies replaced by MagicMocks."""
    cfg = MagicMock()
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.CHROMA_PATH = "./test_chroma_db"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "claude-sonnet-4-6"

    with (
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.VectorStore"),
        patch("rag_system.AIGenerator"),
        patch("rag_system.SessionManager"),
    ):
        system = RAGSystem(cfg)

    # Replace with fresh mocks so test assertions are clean
    system.ai_generator = MagicMock()
    system.tool_manager = MagicMock()
    system.session_manager = MagicMock()
    return system


# ---------------------------------------------------------------------------
# Return-value contract
# ---------------------------------------------------------------------------

class TestQueryReturnValue:
    def test_query_returns_tuple_of_answer_and_sources(self, rag):
        """query() returns a 2-tuple: (answer_str, sources_list)."""
        rag.ai_generator.generate_response.return_value = "The answer"
        rag.tool_manager.get_last_sources.return_value = []

        result = rag.query("What is Python?")

        assert isinstance(result, tuple) and len(result) == 2

    def test_query_returns_ai_generator_answer(self, rag):
        """The first element of the tuple is the text from AIGenerator."""
        rag.ai_generator.generate_response.return_value = "My answer"
        rag.tool_manager.get_last_sources.return_value = []

        answer, _ = rag.query("Q")

        assert answer == "My answer"

    def test_query_returns_sources_from_tool_manager(self, rag):
        """The second element is the source list from ToolManager."""
        rag.ai_generator.generate_response.return_value = "A"
        rag.tool_manager.get_last_sources.return_value = ["src1", "src2"]

        _, sources = rag.query("Q")

        assert sources == ["src1", "src2"]


# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------

class TestToolWiring:
    def test_query_passes_tools_to_ai_generator(self, rag):
        """ToolManager.get_tool_definitions() output is passed to generate_response."""
        rag.ai_generator.generate_response.return_value = "A"
        rag.tool_manager.get_last_sources.return_value = []
        rag.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content"}
        ]

        rag.query("Q")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "tools" in kwargs
        assert kwargs["tools"] == [{"name": "search_course_content"}]

    def test_query_passes_tool_manager_to_ai_generator(self, rag):
        """The same ToolManager instance is forwarded so tools can be executed."""
        rag.ai_generator.generate_response.return_value = "A"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Q")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs.get("tool_manager") is rag.tool_manager

    def test_query_resets_sources_after_retrieval(self, rag):
        """reset_sources() is called on every query so stale sources don't leak."""
        rag.ai_generator.generate_response.return_value = "A"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Q")

        rag.tool_manager.reset_sources.assert_called_once()

    def test_sources_are_retrieved_before_reset(self, rag):
        """get_last_sources() is called before reset_sources()."""
        call_order = []
        rag.ai_generator.generate_response.return_value = "A"
        rag.tool_manager.get_last_sources.side_effect = lambda: call_order.append("get") or []
        rag.tool_manager.reset_sources.side_effect = lambda: call_order.append("reset")

        rag.query("Q")

        assert call_order == ["get", "reset"], (
            "Sources must be retrieved before they are reset"
        )


# ---------------------------------------------------------------------------
# Session / history
# ---------------------------------------------------------------------------

class TestSessionHandling:
    def test_no_history_when_no_session_id(self, rag):
        """Without a session_id, conversation_history is None/absent."""
        rag.ai_generator.generate_response.return_value = "A"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Q")  # no session_id

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs.get("conversation_history") is None

    def test_history_fetched_when_session_id_provided(self, rag):
        """Session history is retrieved and passed to generate_response."""
        rag.session_manager.get_conversation_history.return_value = (
            "User: Hi\nAssistant: Hello"
        )
        rag.ai_generator.generate_response.return_value = "A"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Q", session_id="session_1")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs.get("conversation_history") == "User: Hi\nAssistant: Hello"

    def test_session_history_updated_after_query(self, rag):
        """add_exchange is called with the original query and AI response."""
        rag.ai_generator.generate_response.return_value = "The answer"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("What is Python?", session_id="session_1")

        rag.session_manager.add_exchange.assert_called_once_with(
            "session_1", "What is Python?", "The answer"
        )

    def test_session_not_updated_without_session_id(self, rag):
        """add_exchange is NOT called when no session_id is provided."""
        rag.ai_generator.generate_response.return_value = "A"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Q")

        rag.session_manager.add_exchange.assert_not_called()


# ---------------------------------------------------------------------------
# Exception propagation  ← these map directly to 'query failed' in the UI
# ---------------------------------------------------------------------------

class TestExceptionPropagation:
    def test_exception_from_ai_generator_propagates(self, rag):
        """Any exception from generate_response bubbles up to FastAPI → HTTP 500."""
        rag.ai_generator.generate_response.side_effect = Exception(
            "Anthropic API error"
        )

        with pytest.raises(Exception, match="Anthropic API error"):
            rag.query("What is Python?", session_id="session_1")

    def test_exception_from_tool_manager_propagates(self, rag):
        """Exception from get_tool_definitions propagates."""
        rag.tool_manager.get_tool_definitions.side_effect = RuntimeError("Tool crash")

        with pytest.raises(RuntimeError, match="Tool crash"):
            rag.query("Q")

    def test_exception_from_get_last_sources_propagates(self, rag):
        """Exception from get_last_sources propagates."""
        rag.ai_generator.generate_response.return_value = "A"
        rag.tool_manager.get_last_sources.side_effect = Exception("Source error")

        with pytest.raises(Exception, match="Source error"):
            rag.query("Q")

    def test_content_query_does_not_silently_return_none(self, rag):
        """query() always returns a tuple, never None (None would break the FastAPI model)."""
        rag.ai_generator.generate_response.return_value = "Some answer"
        rag.tool_manager.get_last_sources.return_value = []

        result = rag.query("What does lesson 1 cover?")

        assert result is not None
        answer, sources = result
        assert answer is not None

    def test_query_prompt_wraps_user_question(self, rag):
        """The user question is embedded in the prompt sent to the AI generator."""
        rag.ai_generator.generate_response.return_value = "A"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Explain neural networks", session_id="s1")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        query_sent = kwargs.get("query", "")
        assert "Explain neural networks" in query_sent
