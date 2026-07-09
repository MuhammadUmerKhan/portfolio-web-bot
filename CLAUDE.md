# CLAUDE.md — Project Context for AI Agents

> This file provides comprehensive project context for any AI coding agent.
> Read this file in full before making any changes to the codebase.

---

## Project Overview

**Name**: Portfolio Web Bot — Production-Grade Hybrid RAG Personal Assistant
**Owner**: Muhammad Umer Khan
**Purpose**: A personal AI assistant that answers questions about Umer's skills, projects, work experience, and education using a hybrid Retrieval-Augmented Generation (RAG) pipeline. It is deployed as a FastAPI backend consumed by Umer's Vercel-hosted portfolio frontends.
**Stack**: Python 3.12, FastAPI, LangChain, LangGraph, Groq (LLM), BAAI/bge-base-en-v1.5 (embeddings), Qdrant Cloud (vector DB), FlashRank (reranker), Logfire (observability)
**Package Manager**: `uv` (with `pyproject.toml` and `uv.lock`)

---

## Architecture

```
Sources (PDF resume + GitHub READMEs from 70+ repos)
        │
        ▼
Ingestion & Chunking  (app/ingestion/)
        │
        ├──────────────┐
        ▼              ▼
  Vector index     Knowledge graph
  (Qdrant Cloud)   (in-memory adjacency list)
        │              │
        └──────┬───────┘
               ▼
     LangGraph Agentic Planner  (tool-calling routing)
               │
               ▼
     Hybrid retrieval (dense Qdrant + BM25 + RRF + FlashRank rerank)
               │
               ▼
     Responder node (persona prompt + context fusion)
               │
               ▼
     Portkey API Gateway (Routing/Fallback strategy: llama-3.3-70b -> llama-3.1-8b)
               │
               ▼
         Final answer
```

Qdrant Cloud is the **single source of truth**. BM25 index and knowledge graph are derived, in-memory structures rebuilt from Qdrant on every boot. No local data files are required at runtime.

---

## Directory Structure

```
portfolio-web-bot/
├── main.py                          # FastAPI app entry point
├── PLAN.md                          # Master roadmap (13 phases, checklist-style)
├── pyproject.toml                   # uv package manifest
├── .env / .env.example              # Environment variables (never commit .env)
│
├── app/
│   ├── core/
│   │   ├── config.py                # Pydantic Settings (validates all env vars)
│   │   ├── logging.py               # Logfire structured logging setup
│   │   ├── circuit_breaker.py       # AsyncCircuitBreaker for Qdrant failure resilience
│   │   ├── embeddings.py            # Local BAAI/bge-base-en-v1.5 embeddings factory
│   │   └── __init__.py              # Exports: get_settings, setup_logging, get_logger, get_embeddings_model
│   │
│   ├── agents/
│   │   ├── graph.py                 # LangGraph StateGraph compiler (guard -> planner → retriever → responder)
│   │   ├── state.py                 # AgentState TypedDict (messages, retrieved_docs, graph_context)
│   │   └── nodes/
│   │       ├── guard.py             # Input Guardrails Node (rejects off-topic/injection)
│   │       ├── planner.py           # Tool-calling classifier (binds vector_search + graph_search)
│   │       ├── retriever.py         # Executes tool calls, serializes Documents to dicts for checkpointer safety
│   │       └── responder.py         # Builds final prompt with persona rules, generates answer
│   │
│   ├── gateway/
│   │   └── client.py                # Portkey ChatOpenAI client wrapper with fallback configs
│   │
│   ├── guardrails/
│   │   ├── rails.py                 # NeMo Guardrails configuration and gating function
│   │   └── config/                  # Colang scripts (.co) and config.yml
│   │
│   ├── ingestion/
│   │   ├── loader/
│   │   │   ├── pdf_loader.py        # PyPDFLoader wrapper
│   │   │   └── markdown_loader.py   # GitHub raw markdown loader with _meta.json resolution
│   │   ├── chunking/
│   │   │   ├── markdown_chunker.py  # MarkdownHeaderTextSplitter + recursive fallback
│   │   │   └── recursive_chunker.py # Generic RecursiveCharacterTextSplitter
│   │   ├── processor.py             # Regex entity extractor (tech stack, years, employers vs platforms)
│   │   └── graph_builder.py         # Compiles adjacency-list knowledge graph from chunk metadata
│   │
│   └── services/
│       ├── chatbot.py               # CustomDocChatbot (orchestrator) + CustomHybridRetriever
│       ├── graph_service.py         # In-memory graph query service (entity word-boundary matching)
│       └── retrieval/
│           ├── qdrant_service.py     # Direct Qdrant SDK client (query_points + scroll pagination)
│           └── ranking_service.py    # FlashRank local cross-encoder reranker
│
├── scripts/
│   ├── fetch_github_sources.py      # GitHub API fetcher (pulls READMEs from all repos)
│   └── ingest.py                    # CLI: load → chunk → extract → embed → upsert to Qdrant
│
├── scratch/
│   ├── interactive_test.py          # VS Code cell-based interactive tester (exportable to .ipynb)
│   └── interactive_test.ipynb       # Jupyter notebook for component-level testing
│
├── data/                            # Git-ignored. Raw + processed data (local dev only)
│   ├── raw/github/                  # Fetched GitHub READMEs and _meta.json files
│   └── processed/
│       ├── chunks.json              # Serialized chunks (local cache, not needed at runtime)
│       └── knowledge_graph.json     # Compiled graph (local cache, not needed at runtime)
│
├── assets/
│   └── Muhammad_Umer_Khan_AI_Resume.pdf
│
└── docs/                            # Documentation artifacts
```

---

## Environment Variables

All variables are validated at startup by Pydantic Settings in `app/core/config.py`. See `.env.example` for the full list. Critical ones:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Primary LLM (Groq) |
| `QDRANT_API_KEY` | Qdrant Cloud authentication |
| `QDRANT_END_POINT` | Qdrant Cloud cluster URL |
| `QDRANT_COLLECTION_NAME` | Vector collection name (default: `personal_kb`) |
| `GITHUB_TOKEN` | Fine-grained PAT for source fetching |
| `LOGFIRE_TOKEN` | Pydantic Logfire observability |
| `LANGSMITH_API_KEY` | LangSmith tracing (optional) |

---

## Commands

```bash
# Install dependencies
uv sync

# Activate virtual environment
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Fetch GitHub sources (run once or when repos change)
python scripts/fetch_github_sources.py

# Run ingestion pipeline (parse → chunk → embed → upsert to Qdrant)
python scripts/ingest.py

# Start the FastAPI development server
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Key Design Decisions & Gotchas

### LangGraph Serialization
The checkpointer (`MemorySaver`) uses msgpack under the hood. Storing LangChain `Document` objects in `AgentState` causes `TypeError: Type is not msgpack serializable`. **Always serialize Documents to plain dicts** (`{"content": ..., "metadata": ...}`) before saving to state. The `retrieved_docs` field in `AgentState` is `List[dict]`, not `List[Document]`.

### Embedding Model
We use local `BAAI/bge-base-en-v1.5` (768 dimensions) via HuggingFace, NOT a remote API. This avoids rate limits entirely and keeps the vector space consistent. Never mix embedding models — all vectors in Qdrant were embedded with this model.

### Qdrant as Single Source of Truth
The `data/` directory is git-ignored and not needed at runtime. On boot, `CustomDocChatbot.__init__` scrolls the entire Qdrant collection via `fetch_all_chunks()` (with 5-step retry backoff for idle cluster wake-up) and rebuilds BM25 + knowledge graph in memory. This enables stateless deployment.

### Working Directory Sensitivity
`PROJECT_ROOT` in `config.py` is resolved relative to `__file__`, not `os.getcwd()`. This is critical because Jupyter notebooks shift the working directory to `scratch/`, breaking relative path resolution. Always use `PROJECT_ROOT` for file paths.

### Entity Extraction: Employers vs Platforms
The processor distinguishes between `EMPLOYER_KEYWORDS` (SMIT, Saylani, Revera Innovations, etc. — real work/education relationships) and `PLATFORM_KEYWORDS` (GitHub, OpenAI, Google, etc. — tools/vendors). They produce different edge types in the knowledge graph. Never conflate them.

### Graph Topology (LangGraph Agent)
```
START → planner → [tool_calls?] → retriever → responder → END
                → [no tools]   ─────────────────────────→ END
```
The planner uses ChatGroq with bound tools (`vector_search`, `graph_search`) to decide routing dynamically. Greetings skip retrieval entirely. Knowledge-graph queries call `graph_search`. Factual queries call `vector_search`. Complex queries call both.

---

## Coding Conventions

- Use `get_logger(__name__)` for logging (Logfire-backed structured logging).
- Use `get_settings()` (cached via `@lru_cache`) for configuration — never read `.env` directly.
- Use `get_embeddings_model()` for embeddings — never instantiate `HuggingFaceEmbeddings` directly.
- All retrieval services live under `app/services/retrieval/`.
- All LangGraph agent nodes live under `app/agents/nodes/`.
- Ingestion code (loaders, chunkers, processor, graph builder) lives under `app/ingestion/`.
- Scripts that are run manually or in CI live under `scripts/`.
- Temporary test/debug files go in `scratch/`.
- The `src/` directory has been retired. All code now lives under `app/`.

---

## Progress (Phases Completed)

| Phase | Status | Summary |
|---|---|---|
| **0** — Accounts & secrets | ✅ Done | API keys secured, `.env` configured |
| **1** — Multi-source ingestion | ✅ Done | PDF + 70 GitHub repos → 898 chunks with metadata |
| **2** — Embeddings & Qdrant | ✅ Done | Local BGE embeddings, Qdrant Cloud collection created & synced |
| **3** — Hybrid retrieval + reranking | ✅ Done | Dense + BM25 + RRF fusion + FlashRank reranking (top 4) |
| **4a** — Knowledge graph | ✅ Done | Adjacency-list graph (Project, Skill, Company, Year, Platform) |
| **4b** — Qdrant as single source | ✅ Done | BM25 + graph rebuilt from Qdrant at boot, stateless deploy |
| **5** — Agentic RAG router | ✅ Done | LangGraph workflow with dynamic tool-calling planner |
| **6** — Guardrails | ✅ Done | Input/output guardrails (prompt injection, PII, off-topic) |
| **7** — LLM gateway | 🔲 Next | Portkey OSS gateway with fallback chain |
| **8** — Observability | 🟡 Partial | Logfire wired; LangSmith/Langfuse + metrics endpoint pending |
| **9** — Evaluation | 🔲 Planned | RAGAS + DeepEval eval suite |
| **10** — API hardening | 🔲 Planned | Rate limiting, auth, Qdrant keep-alive |
| **11** — Voice interface | 🔲 Planned | Vapi/Retell integration (optional) |
| **12** — Deployment & CI/CD | 🔲 Planned | Docker, Render/HF Spaces, GitHub Actions |
| **13** — Documentation | 🔲 Planned | README rewrite, demo recording |

---

## Phase 7 — LLM Gateway (Current Phase)

The next phase to implement. Key requirements from `docs/PLAN.md`:
- **Portkey Gateway**: Self-host the Portkey OSS gateway or use the managed API.
- **Fallback Chain**: Configure resilient LLM routing from Groq `llama-3.3-70b-versatile` → Groq `gpt-oss-120b` → Google `gemini-2.5-flash`.
- **Circuit Breaking**: Implement fail-safes for upstream API outages.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Welcome message |
| `POST` | `/chat` | Main chat endpoint. Body: `{"query": "..."}`. Returns: `{"reply": "..."}` |
| `GET` | `/health` | Health check (verifies chatbot initialization) |

---

## Important References

- **Master Roadmap**: `docs/PLAN.md` — the authoritative source for all phase requirements, tech choices, and design reasoning.
- **Configuration Schema**: `app/core/config.py` — all environment variables and their types/defaults.
- **Agent Workflow**: `app/agents/graph.py` — the LangGraph state machine definition.
- **Chatbot Orchestrator**: `app/services/chatbot.py` — `CustomDocChatbot` class that wires everything together.
- **Guardrails Context**: `docs/08_GUARDRAILS.md` — architecture and integration patterns for the NeMo security layer.
