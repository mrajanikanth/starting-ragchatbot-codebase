# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project is managed with `uv` and requires Python ≥ 3.13. **Use `uv` for ALL dependency management and for anything that runs Python in the project env — never invoke `pip` directly, and never hand-edit `pyproject.toml` deps without re-running `uv lock`/`uv sync`.** `pip` (or manual edits) outside `uv` will desync `uv.lock` from the installed environment.

```bash
uv sync                                          # install / sync deps from uv.lock
uv add <pkg>                                     # add a dependency (not `pip install <pkg>`)
uv add --dev <pkg>                               # add a dev-only dependency
uv remove <pkg>                                  # remove a dependency
uv lock --upgrade-package <pkg> && uv sync       # upgrade one package
./run.sh                                         # start the app (cd backend && uvicorn --reload --port 8000)
cd backend && uv run uvicorn app:app --reload --port 8000   # equivalent manual start
uv run python path/to/file.py                    # run a Python file in the project env (NEVER `python file.py`)
uv run python -c "from backend.config import config; print(config)"   # run arbitrary Python in the env
```

There is no test suite, linter, or formatter configured — `pyproject.toml` only declares runtime deps. Don't invent commands for tooling that isn't wired up.

Required env var (in `.env` at the repo root, read by `backend/config.py` via `python-dotenv`):

```
ANTHROPIC_API_KEY=sk-ant-...
```

Web UI at http://localhost:8000, Swagger at http://localhost:8000/docs.

## Architecture

This is a tool-calling RAG chatbot: a FastAPI backend serves a vanilla-JS frontend and orchestrates a two-call dance with Claude where retrieval happens via an Anthropic **tool**, not by stuffing context into the prompt.

### Request flow (the "big picture" that spans files)

1. **`frontend/script.js`** POSTs `{query, session_id}` to `/api/query`. `session_id` is `null` on first turn; the server mints one and the browser caches it for the rest of the page lifetime.
2. **`backend/app.py`** validates the request, creates a session if needed, and calls `RAGSystem.query()`.
3. **`backend/rag_system.py`** is the orchestrator. It pulls conversation history (as a formatted string, not as real `messages`), grabs tool definitions from `ToolManager`, and hands everything to `AIGenerator`.
4. **`backend/ai_generator.py`** makes **two** Anthropic API calls when a tool is used:
   - Call #1: with `tools=[...]` and `tool_choice="auto"`. If `stop_reason == "tool_use"`, it executes every tool block.
   - Call #2: same messages + `tool_result` blocks appended, but **`tools` is deliberately omitted** to force a text answer. This is what enforces the system prompt's "one search per query maximum" rule — there's no architectural support for multi-hop tool use.
5. **`backend/search_tools.py`** — the only registered tool is `search_course_content` on `CourseSearchTool`. It calls `VectorStore.search()` and stashes the formatted source list on `self.last_sources` as a side effect.
6. **`backend/vector_store.py`** runs two-stage retrieval against ChromaDB: if `course_name` was supplied, it fuzzy-resolves the canonical title against the `course_catalog` collection first, then queries `course_content` with a `where` filter for `(course_title, lesson_number)`.
7. Back in `RAGSystem.query()`, sources are harvested from `ToolManager.get_last_sources()` (out-of-band — they never travel through Claude's response payload), the exchange is appended to session history, and `(answer, sources)` is returned.
8. The frontend renders the answer through `marked.parse()` and shows sources in a `<details>` disclosure.

### Document ingestion

On FastAPI startup (`app.py:88-98`), `RAGSystem.add_course_folder("../docs")` scans for `.pdf/.docx/.txt` files. `DocumentProcessor.process_course_document()` parses a fixed header (`Course Title:`, `Course Link:`, `Course Instructor:`) then walks `Lesson N: <title>` markers, optionally consuming a following `Lesson Link:` line. Each lesson body is sentence-chunked (regex-based, abbreviation-aware) into 800-char chunks with 100-char sentence-level overlap. Chunks are inserted into two Chroma collections: `course_catalog` (one row per course, lesson list serialised as a JSON string in metadata because Chroma metadata is scalar-only) and `course_content` (per-chunk). Ingestion is idempotent — courses whose titles are already in the catalog are skipped.

### Conventions worth knowing

- **Course title is the primary key.** It's used as the Chroma `id` in `course_catalog` and as the metadata foreign key in `course_content`. Two courses with the same title will collide.
- **Sessions are in-memory only** (`SessionManager.sessions` dict). Restarting the server wipes them; the browser's cached `session_id` will then point at nothing and the server will silently treat it as a new session because `get_conversation_history` returns `None` for unknown ids.
- **History is sent as a string in the system prompt**, not as prior `messages`. Claude cannot see tool-use turns from earlier exchanges.
- **`MAX_HISTORY = 2`** in `config.py` means 2 *exchanges* (4 messages) are kept — the trim happens in `SessionManager.add_message` as `max_history * 2`.
- **Assistant output is rendered as HTML via `marked.parse` with no sanitiser** (`frontend/script.js:120`). User messages are escaped; assistant messages are not.
- **Chunk context prefixes are inconsistent**: in `document_processor.py`, non-final lessons prefix only the *first* chunk with `"Lesson N content: ..."`, while the final lesson prefixes *every* chunk with `"Course <title> Lesson N content: ..."`. The two branches (lines ~186 and ~234) have drifted — likely a bug, treat carefully if changing chunking.
- **`backend/main.py` is unused scaffold** — the real entrypoint is `backend/app.py` via `run.sh`.
- **ChromaDB lives at `backend/chroma_db/`** (gitignored). Delete the directory to force full re-ingest on next startup.
- **Embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers`; downloaded on first use, cached under `~/.cache/huggingface`.
- **Model**: `claude-sonnet-4-20250514` hardcoded in `config.py:13`. Change it there, not in `ai_generator.py`.