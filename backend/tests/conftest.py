import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Manual insertion keeps individual-file invocation working (e.g. running
# `uv run pytest backend/tests/test_foo.py` directly).  The pytest.ini_options
# pythonpath setting handles the standard `uv run pytest` from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Shared data fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_query_result():
    """Typical (answer, sources) return value for RAGSystem.query()."""
    return (
        "Python is a high-level programming language.",
        ["Python 101 - Lesson 1", "Python 101 - Lesson 2"],
    )


@pytest.fixture
def sample_course_analytics():
    """Typical return value for RAGSystem.get_course_analytics()."""
    return {
        "total_courses": 2,
        "course_titles": ["Python 101", "Machine Learning Fundamentals"],
    }


# ── App / API test fixtures ───────────────────────────────────────────────────

async def _mock_static_asgi(scope, receive, send):
    """
    Minimal async ASGI callable that replaces StaticFiles in the test env.

    app.py mounts StaticFiles("../frontend") at module level; the frontend
    directory does not exist during tests.  This stub responds to any HTTP
    request with 200 so that the "/" route smoke-test passes and the app
    initialises cleanly.
    """
    if scope["type"] == "http":
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/html; charset=utf-8")],
        })
        await send({
            "type": "http.response.body",
            "body": b"<html><body>Mock frontend</body></html>",
            "more_body": False,
        })


@pytest.fixture(scope="session")
def app_module():
    """
    Import app.py once per session with module-level side effects neutralised.

    app.py has two module-level calls that break in a test environment:
      1. rag_system = RAGSystem(config)  — hits ChromaDB + the embedding model
      2. app.mount("/", StaticFiles("../frontend"))  — directory absent in tests

    Both are patched before the import so the module loads without errors.
    The mock StaticFiles instance (_mock_static_asgi) is a real async ASGI
    function, so requests to "/" still receive a well-formed HTTP response.
    """
    mock_static_cls = MagicMock(return_value=_mock_static_asgi)

    sys.modules.pop("app", None)  # ensure a clean import inside the patches

    with (
        patch("rag_system.RAGSystem"),
        patch("fastapi.staticfiles.StaticFiles", mock_static_cls),
    ):
        import app as _app  # noqa: PLC0415

    return _app


@pytest.fixture(scope="session")
def test_client(app_module):
    """Session-scoped TestClient; app lifespan events run once per session."""
    from fastapi.testclient import TestClient

    with TestClient(app_module.app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def rag_mock(app_module, sample_query_result, sample_course_analytics):
    """
    Per-test mock RAGSystem injected into the live app module.

    Replaces app_module.rag_system for the duration of one test, then
    restores the original so tests are fully isolated from each other.
    """
    mock = MagicMock()
    mock.query.return_value = sample_query_result
    mock.get_course_analytics.return_value = sample_course_analytics
    mock.session_manager.create_session.return_value = "test-session-id"
    mock.session_manager.delete_session.return_value = None

    original = app_module.rag_system
    app_module.rag_system = mock
    yield mock
    app_module.rag_system = original