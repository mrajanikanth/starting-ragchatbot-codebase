"""
Integration tests for the FastAPI endpoints defined in app.py.

All tests share a session-scoped TestClient (conftest.test_client) so the
app is only initialised once.  A function-scoped rag_mock injects a fresh
MagicMock into app_module.rag_system before each test and restores the
original afterward, giving full per-test isolation without recreating the
client.
"""
import pytest


class TestQueryEndpoint:
    """POST /api/query"""

    def test_success_returns_200_with_answer_and_sources(self, test_client, rag_mock):
        response = test_client.post("/api/query", json={"query": "What is Python?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Python is a high-level programming language."
        assert body["sources"] == ["Python 101 - Lesson 1", "Python 101 - Lesson 2"]

    def test_response_body_includes_session_id(self, test_client, rag_mock):
        response = test_client.post("/api/query", json={"query": "Q"})
        assert response.status_code == 200
        assert "session_id" in response.json()

    def test_creates_session_when_none_provided(self, test_client, rag_mock):
        response = test_client.post("/api/query", json={"query": "Q"})
        assert response.status_code == 200
        rag_mock.session_manager.create_session.assert_called_once()
        assert response.json()["session_id"] == "test-session-id"

    def test_uses_provided_session_id_without_creating_new(self, test_client, rag_mock):
        response = test_client.post(
            "/api/query",
            json={"query": "Q", "session_id": "existing-sid"},
        )
        assert response.status_code == 200
        assert response.json()["session_id"] == "existing-sid"
        rag_mock.session_manager.create_session.assert_not_called()

    def test_forwards_query_to_rag_system(self, test_client, rag_mock):
        test_client.post("/api/query", json={"query": "What is machine learning?"})
        rag_mock.query.assert_called_once()
        assert rag_mock.query.call_args[0][0] == "What is machine learning?"

    def test_sources_field_is_a_list(self, test_client, rag_mock):
        response = test_client.post("/api/query", json={"query": "Q"})
        assert isinstance(response.json()["sources"], list)

    def test_missing_query_field_returns_422(self, test_client, rag_mock):
        response = test_client.post("/api/query", json={})
        assert response.status_code == 422

    def test_rag_system_exception_returns_500(self, test_client, rag_mock):
        rag_mock.query.side_effect = Exception("Database error")
        response = test_client.post("/api/query", json={"query": "Q"})
        assert response.status_code == 500

    def test_500_detail_contains_the_error_message(self, test_client, rag_mock):
        rag_mock.query.side_effect = Exception("Something went wrong")
        response = test_client.post("/api/query", json={"query": "Q"})
        assert "Something went wrong" in response.json()["detail"]


class TestCoursesEndpoint:
    """GET /api/courses"""

    def test_success_returns_200_with_course_stats(self, test_client, rag_mock):
        response = test_client.get("/api/courses")
        assert response.status_code == 200
        body = response.json()
        assert body["total_courses"] == 2
        assert "Python 101" in body["course_titles"]
        assert "Machine Learning Fundamentals" in body["course_titles"]

    def test_response_has_required_fields(self, test_client, rag_mock):
        body = test_client.get("/api/courses").json()
        assert "total_courses" in body
        assert "course_titles" in body

    def test_course_titles_is_a_list(self, test_client, rag_mock):
        assert isinstance(test_client.get("/api/courses").json()["course_titles"], list)

    def test_analytics_exception_returns_500(self, test_client, rag_mock):
        rag_mock.get_course_analytics.side_effect = RuntimeError("Chroma error")
        assert test_client.get("/api/courses").status_code == 500


class TestNewSessionEndpoint:
    """POST /api/session/new"""

    def test_success_returns_200_with_session_id(self, test_client, rag_mock):
        response = test_client.post("/api/session/new", json={})
        assert response.status_code == 200
        assert "session_id" in response.json()

    def test_returns_session_id_from_session_manager(self, test_client, rag_mock):
        assert test_client.post("/api/session/new", json={}).json()["session_id"] == "test-session-id"

    def test_deletes_old_session_when_id_provided(self, test_client, rag_mock):
        test_client.post("/api/session/new", json={"old_session_id": "old-sid-123"})
        rag_mock.session_manager.delete_session.assert_called_once_with("old-sid-123")

    def test_does_not_delete_session_when_old_id_is_null(self, test_client, rag_mock):
        test_client.post("/api/session/new", json={})
        rag_mock.session_manager.delete_session.assert_not_called()

    def test_session_creation_exception_returns_500(self, test_client, rag_mock):
        rag_mock.session_manager.create_session.side_effect = Exception("Session error")
        assert test_client.post("/api/session/new", json={}).status_code == 500


class TestRootEndpoint:
    """GET / — served by the mocked static files mount."""

    def test_root_returns_200(self, test_client, rag_mock):
        assert test_client.get("/").status_code == 200

    def test_root_returns_html_content_type(self, test_client, rag_mock):
        content_type = test_client.get("/").headers.get("content-type", "")
        assert "text/html" in content_type