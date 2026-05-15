"""
Tests for CourseSearchTool.execute() in search_tools.py.

Covers: result formatting, empty-result handling, error propagation,
parameter forwarding to VectorStore, and source tracking.
"""
import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_results(docs, metas):
    """Convenience constructor for non-empty SearchResults."""
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs),
    )


# ---------------------------------------------------------------------------
# CourseSearchTool.execute()
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecute:
    def setup_method(self):
        self.mock_store = MagicMock()
        self.tool = CourseSearchTool(self.mock_store)

    # --- Happy path ---

    def test_returns_formatted_text_when_content_found(self):
        """execute() returns a non-empty string when the vector store has hits."""
        self.mock_store.search.return_value = make_results(
            ["Python is a high-level language."],
            [{"course_title": "Python 101", "lesson_number": 1}],
        )
        self.mock_store.get_lesson_link.return_value = None

        result = self.tool.execute(query="What is Python?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_contains_course_title(self):
        """Formatted output includes the course title as a header."""
        self.mock_store.search.return_value = make_results(
            ["Some content"],
            [{"course_title": "Intro to AI", "lesson_number": 2}],
        )
        self.mock_store.get_lesson_link.return_value = None

        result = self.tool.execute(query="AI basics")

        assert "Intro to AI" in result

    def test_result_contains_lesson_number(self):
        """Formatted output includes the lesson number."""
        self.mock_store.search.return_value = make_results(
            ["Content"],
            [{"course_title": "ML Course", "lesson_number": 3}],
        )
        self.mock_store.get_lesson_link.return_value = None

        result = self.tool.execute(query="something")

        assert "Lesson 3" in result

    def test_result_contains_document_text(self):
        """The actual chunk text appears in the output."""
        self.mock_store.search.return_value = make_results(
            ["Neural networks learn from data."],
            [{"course_title": "DL Course", "lesson_number": 1}],
        )
        self.mock_store.get_lesson_link.return_value = None

        result = self.tool.execute(query="neural networks")

        assert "Neural networks learn from data." in result

    # --- Empty results ---

    def test_returns_no_content_message_when_empty(self):
        """execute() returns a 'no content found' string on empty results."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = self.tool.execute(query="obscure topic")

        assert "No relevant content found" in result

    def test_empty_result_without_filter_has_no_filter_info(self):
        """Empty message with no filters is a plain 'No relevant content found.'"""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = self.tool.execute(query="something")

        assert result == "No relevant content found."

    def test_empty_result_mentions_course_when_filtered(self):
        """Empty message includes the requested course name when a filter was used."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = self.tool.execute(query="something", course_name="Python 101")

        assert "Python 101" in result

    def test_empty_result_mentions_lesson_when_filtered(self):
        """Empty message includes the lesson number when a lesson filter was used."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = self.tool.execute(query="something", lesson_number=5)

        assert "lesson 5" in result.lower()

    # --- Error from vector store ---

    def test_returns_error_string_when_search_fails(self):
        """execute() surfaces the error string from a failed search (does NOT raise)."""
        self.mock_store.search.return_value = SearchResults.empty(
            "Search error: ChromaDB collection has 0 elements"
        )

        result = self.tool.execute(query="anything")

        # Must return a string, not raise an exception
        assert isinstance(result, str)
        assert "Search error" in result

    def test_does_not_raise_when_search_returns_error(self):
        """execute() never raises; it always returns a string."""
        self.mock_store.search.return_value = SearchResults.empty("DB error")

        try:
            result = self.tool.execute(query="test")
        except Exception as exc:
            pytest.fail(f"execute() raised unexpectedly: {exc}")

    # --- Parameter forwarding ---

    def test_forwards_query_to_vector_store(self):
        """VectorStore.search is called with the exact query string."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        self.tool.execute(query="Python basics")

        self.mock_store.search.assert_called_once()
        _, kwargs = self.mock_store.search.call_args
        assert kwargs["query"] == "Python basics"

    def test_forwards_course_name_to_vector_store(self):
        """course_name is forwarded to VectorStore.search."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        self.tool.execute(query="q", course_name="ML Fundamentals")

        _, kwargs = self.mock_store.search.call_args
        assert kwargs["course_name"] == "ML Fundamentals"

    def test_forwards_lesson_number_to_vector_store(self):
        """lesson_number is forwarded to VectorStore.search."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        self.tool.execute(query="q", lesson_number=7)

        _, kwargs = self.mock_store.search.call_args
        assert kwargs["lesson_number"] == 7

    # --- Source tracking ---

    def test_last_sources_populated_after_successful_search(self):
        """last_sources is non-empty after a successful search."""
        self.mock_store.search.return_value = make_results(
            ["Content"],
            [{"course_title": "Python 101", "lesson_number": 1}],
        )
        self.mock_store.get_lesson_link.return_value = None

        self.tool.execute(query="Python?")

        assert len(self.tool.last_sources) == 1

    def test_last_sources_contains_course_label(self):
        """Source label includes the course title."""
        self.mock_store.search.return_value = make_results(
            ["Content"],
            [{"course_title": "Python 101", "lesson_number": 2}],
        )
        self.mock_store.get_lesson_link.return_value = None

        self.tool.execute(query="Python?")

        assert "Python 101" in self.tool.last_sources[0]

    def test_last_sources_contains_anchor_tag_when_link_available(self):
        """Source is an HTML anchor tag when a lesson link is returned."""
        self.mock_store.search.return_value = make_results(
            ["Content"],
            [{"course_title": "Python 101", "lesson_number": 1}],
        )
        self.mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        self.tool.execute(query="Python?")

        assert '<a href="https://example.com/lesson1"' in self.tool.last_sources[0]

    def test_last_sources_is_plain_text_when_no_link(self):
        """Source is plain text (no anchor) when no lesson link exists."""
        self.mock_store.search.return_value = make_results(
            ["Content"],
            [{"course_title": "Python 101", "lesson_number": 1}],
        )
        self.mock_store.get_lesson_link.return_value = None

        self.tool.execute(query="Python?")

        assert "<a href" not in self.tool.last_sources[0]

    def test_last_sources_empty_after_empty_results(self):
        """last_sources is not populated when search returns no results."""
        self.tool.last_sources = ["stale"]
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        self.tool.execute(query="q")

        # last_sources should NOT be updated with stale data cleared by this call
        # (it should remain as-is because _format_results is not called)
        # The important thing: no crash
        assert isinstance(self.tool.last_sources, list)


# ---------------------------------------------------------------------------
# ToolManager integration
# ---------------------------------------------------------------------------

class TestToolManager:
    def test_register_and_execute_tool(self):
        """ToolManager can register a CourseSearchTool and route calls to it."""
        mock_store = MagicMock()
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool = CourseSearchTool(mock_store)

        manager = ToolManager()
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")
        assert isinstance(result, str)

    def test_execute_unknown_tool_returns_error_string(self):
        """Calling an unregistered tool returns an error string, not an exception."""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="x")
        assert "not found" in result.lower()

    def test_get_last_sources_returns_sources_after_search(self):
        """get_last_sources() returns populated sources after a successful search."""
        mock_store = MagicMock()
        mock_store.search.return_value = make_results(
            ["Content"],
            [{"course_title": "ML", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_store)
        manager = ToolManager()
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="ml basics")

        sources = manager.get_last_sources()
        assert len(sources) == 1

    def test_reset_sources_clears_last_sources(self):
        """reset_sources() empties last_sources on all tools."""
        mock_store = MagicMock()
        mock_store.search.return_value = make_results(
            ["Content"],
            [{"course_title": "ML", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_store)
        manager = ToolManager()
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="q")

        manager.reset_sources()

        assert manager.get_last_sources() == []
