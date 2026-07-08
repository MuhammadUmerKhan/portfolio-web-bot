# PLAN.md — Production-Grade Hybrid RAG Personal Assistant

> Tracking document for turning the current v1 baseline (in-memory FAISS + BM25, resume-only) into a
> reliable, scalable, fast, optimized, and secured hybrid RAG system — built entirely on free-tier
> infrastructure, and portfolio-ready.
>
> Check items off as you go (`- [ ]` → `- [x]`). Each phase has: **what to build**, **exact tech
> choices with reasoning**, **free-tier facts (verified, with dates)**, and **reliability / scalability
> / security notes**. Where a number could drift (rate limits, pricing), it's flagged with
> "⚠️ verify live" — these change often and you should re-check the provider dashboard before relying
> on them for anything time-sensitive.

---

## 0. Current state audit (what already exists in this repo)

Read directly from the repo before writing this plan, so nothing here contradicts what's already built:

- `main.py` — FastAPI app, CORS configured for your Vercel frontends, `/`, `/chat`, `/health` endpoints. Working.
- `src/rag_pipeline.py` — `CustomDocChatbot`: loads one PDF (`PyPDFLoader`), splits with
  `RecursiveCharacterTextSplitter` (800/300), embeds with a **custom OpenRouter embeddings wrapper**
  (`nvidia/llama-nemotron-embed-vl-1b-v2:free`), stores in **in-memory FAISS**, retrieves with an
  **EnsembleRetriever (FAISS + BM25, 50/50 weights)**, answers via **Groq (`openai/gpt-oss-120b`)**,
  memory via `ConversationBufferMemory`, TTL query cache (`cachetools`, 10 min).
- `src/config.py` — env-driven config, resume path hardcoded to one PDF.
- `requirements.txt` — **already lists** the full target stack: `qdrant-client`, `flashrank`,
  `langgraph`, `nemoguardrails`, `portkey-ai`, `logfire`, `langsmith`, `langfuse`, `ragas`, `deepeval`,
  `google-generativeai`, `langchain-google-genai`, `unstructured`, `python-pptx`, `python-docx`,
  `pdfplumber`. **None of these are wired into code yet** — they're aspirational dependencies.
- `app/ingestion/{loader,chunking}` and `app/services/` — empty scaffolds, ready to be filled.

**Read: this project is not starting from zero.** It's starting from a working, honest "traditional
RAG" demo (README.md already calls it that) with the production dependencies pre-installed but not
yet implemented. The phases below are about *implementing* what's already declared, in the right
order, on top of the existing `app/` scaffold rather than a rewrite.

### Known issues to fix, not carry forward
- [ ] `EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"` via OpenRouter's free embedding
  endpoint is not a dependable production choice — OpenRouter's free-tier models are frequently
  rotated, rate-limited unpredictably, and not guaranteed to stay available. Replace with Gemini's
  `gemini-embedding-001` (Phase 2).
- [ ] In-memory FAISS is rebuilt from scratch **on every process start** (`load_pdf` → `split_documents`
  → `FAISS.from_documents` all run inside `setup_and_query`, called per query if `vector_db` isn't
  cached correctly) — this is slow and not scalable past one document. Replace with persistent Qdrant
  (Phase 2).
- [ ] Single hardcoded PDF (`RESUME_PATH`) — needs to become a multi-source ingestion pipeline
  (Phase 1).
- [ ] No guardrails, no reranking, no evals, no auth, no rate limiting despite being listed as
  dependencies — this is most of the work in Phases 4–9.

---

## 1. Target architecture (reference)

```
Sources (MD, PDF, DOCX, PPTX)
        │
        ▼
Ingestion & chunking  (app/ingestion)
        │
        ├──────────────┐
        ▼              ▼
  Vector index     Knowledge graph
  (Qdrant)         (entities/relations, lightweight)
        │              │
        └──────┬───────┘
               ▼
     Guardrails + LangGraph planner   (input safety, routing)
               │
               ▼
     Hybrid retrieval (dense + BM25 + graph + rerank)
               │
               ▼
     Guarded response (output check, citations, reply)
```

Cross-cutting, wrapping every node above: **observability** (Logfire/LangSmith), **LLM gateway with
fallback** (Portkey), **caching** (in-memory/Redis), **rate limiting & auth** (FastAPI middleware).

---

## 2. Tech stack decisions (with reasoning)

| Layer | Choice | Why | Free-tier reality (verify live) |
|---|---|---|---|
| Embeddings | **Gemini `gemini-embedding-001`** via `google-generativeai` (already in requirements.txt) | Already free on Gemini API, 768/1536/3072-dim via Matryoshka truncation, 2048-token input, no card required. More stable than free OpenRouter embedding models. | Free on Gemini API free tier, subject to rate limits; official pricing page lists $0.15/1M tokens on paid tier if you ever exceed free quota. ⚠️ verify current TPM/RPD in Google AI Studio → Quotas, these were cut 50-80% in Dec 2025 and have moved since. |
| Primary LLM | **Groq — `llama-3.3-70b-versatile`** (fallback: `openai/gpt-oss-120b`, already your current model) | Fastest inference (280-394 TPS on 70B), OpenAI-compatible endpoint, no card required. | ~30 RPM / ~12K TPM / ~100K TPD per model on free tier as of June 2026, applied at the org level (not per key). ⚠️ verify at console.groq.com/settings/limits — Groq has changed these more than once in 2025-2026. |
| Secondary/fallback LLM | **Gemini 2.5 Flash-Lite or Flash** via Portkey fallback config | Different provider = real fallback, not just a different model on the same rail. | ~10-15 RPM, 250K TPM, 1,000-1,500 RPD as of mid-2026. Pro models are paid-only since April 2026 — don't design around free Gemini Pro. |
| Vector DB | **Qdrant Cloud free cluster** (already `qdrant-client` in requirements.txt) | Persistent, filterable payloads, purpose-built client-server vector DB — the "production" signal a portfolio should show, vs FAISS-in-memory. | Free tier: 1 node, 0.5 vCPU, 1 GB RAM, 4 GB disk, permanent, no card. Supports ~1M vectors at 768 dims. **Auto-suspends after 1 week idle, deleted after 4 weeks idle** — you need a keep-alive ping (Phase 10) or accept manual reactivation. |
| Reranker | **FlashRank** (already in requirements.txt) | Local cross-encoder, zero network round-trip, zero rate limits, zero cost — genuinely free, not just free-tier. | No limits — it runs on your CPU. |
| Orchestration | **LangGraph** (already in requirements.txt) | You already know this from AuraClaw — reuse the same mental model for a much smaller graph (planner → retrieve → guardrail → respond). | N/A, local library. |
| Guardrails | **NeMo Guardrails** (already in requirements.txt) + a small custom input/output filter | Open source, self-hosted, reuse your DineMate red-teaming knowledge directly. | Free, self-hosted, no external calls required for the rails engine itself (though the rails can call an LLM for checks — route those through Groq free tier too). |
| LLM Gateway | **Portkey** — self-host the **open-source gateway** (github.com/Portkey-AI/gateway), not just the hosted product | The OSS gateway gives you fallback/routing/caching with no request limit at all. The *hosted* Portkey Dev tier (10,000 logs/month, 3-day retention) is fine as an optional add-on for a dashboard, but don't depend on it for the gateway itself. | OSS gateway: no cap, self-hosted. Hosted dashboard free tier: 10K logs/mo, 3-day retention — logs past that just stop recording, requests still succeed. |
| Observability | **Logfire** (Personal plan — already in requirements.txt) | Purpose-built for Pydantic/FastAPI/LLM traces, SQL-queryable, generous free tier. | 10M spans/month, 1 seat, 3 projects, 30-day retention, permanently free as of 2026 pricing. |
| Evals | **RAGAS + DeepEval** (already in requirements.txt) | RAGAS gives faithfulness/relevancy/precision metrics; DeepEval wraps them in pytest so evals run in CI. | Both are open-source libraries — free. The only cost is the LLM calls they make as a judge, which you route through your Groq free tier. |
| Hosting (API) | **Render free tier** to start, **Hugging Face Spaces (Docker)** as an always-warm alternative | Render: real permanent free tier, but web services sleep after 15 min idle (30-60s cold start on next request) — acceptable for a portfolio demo, not for a paid product. HF Spaces Docker: free, and community CPU Spaces don't sleep the same way, good for a public-facing demo bot. | Render free: 750 hrs/mo web service, cold starts. HF Spaces: free CPU tier, persistent for public Spaces. ⚠️ verify current Render bandwidth caps — they were cut from 100GB to 5GB (Hobby) in April 2026. |
| Frontend | Existing Vercel deployments (already in your CORS config) | Already built — no change needed here. | Vercel free tier, already in use. |

---

## 3. Phases

### Phase 0 — Accounts, secrets, repo hygiene
- [x] Create/confirm free accounts: Google AI Studio (Gemini), Groq Console, Qdrant Cloud, Logfire,
      LangSmith or Langfuse (pick one primary — see Phase 8 note), Render or HF Spaces.
- [x] Rotate/scope API keys — never reuse a key across dev and the deployed portfolio instance; a
      leaked key on a public-facing bot is a real risk, not a hypothetical one.
- [x] Confirm `.env` is in `.gitignore` (already present) and `.env.example` lists every var with a
      placeholder, no real values.
- [x] Decide: does the knowledge base include anything sensitive (phone number, address, private
      client details from past work)? Explicitly exclude those from ingestion — a public assistant
      should only ever surface what you'd put on a public resume.

### Phase 1 — Multi-source ingestion pipeline
Builds on the existing empty `app/ingestion/loader` and `app/ingestion/chunking` folders.

- [x] **Source acquisition — GitHub fetcher** (`app/ingestion/fetchers/github_fetcher.py`,
      run via `scripts/fetch_github_sources.py`): pulls `README.md` + `docs/*.md` from every repo
      (public + private, forks excluded) owned by your GitHub account, using an authenticated
      `/user/repos` call (5,000 req/hr limit — confirmed unauthenticated calls hit the 60/hr ceiling
      within two test requests). Idempotent via a git-blob-SHA manifest
      (`data/raw/github/_manifest.json`) — safe to re-run or schedule. Saves per-repo metadata
      (`_meta.json`: description, language, topics, `pushed_at`) alongside the markdown for Phase 1's
      metadata-extraction step and Phase 4's knowledge graph. `data/` is git-ignored — private repo
      content must never enter this repo's git history.
      - [x] Create a fine-grained GitHub PAT (read-only Contents + Metadata) and add it as
            `GITHUB_TOKEN` in `.env`.
      - [x] Run `python scripts/fetch_github_sources.py` and confirm `data/raw/github/` populates.
- [x] **Loaders** (`app/ingestion/loader/`): one loader per source type —
      `pypdf`/`pdfplumber` for PDFs (implemented using LangChain's standard `PyPDFLoader`), `python-docx` for Word, `unstructured` for Markdown/HTML,
      `python-pptx` if you ever add slide content. Each loader returns a common `RawDocument` shape.
- [x] **Header-aware chunking for Markdown** (`app/ingestion/chunking/`): your repo READMEs are
      already structured with headers and bullets — use a markdown-header splitter (LangChain's
      `MarkdownHeaderTextSplitter` + recursive character splitter fallback) preserving logical context boundaries and `start_index`.
- [x] **Generic recursive splitter as fallback** for unstructured text (resume PDF, plain notes) —
      keep something close to the existing 800/300 config, but make chunk size configurable per
      source type.
- [x] **Entity/metadata extraction at ingestion time**: for each chunk, extract lightweight metadata —
      project name, company, tech stack mentioned, date range if present. This is what Phase 4's graph
      layer will consume; doing it once at ingestion is cheaper than doing it per-query.
- [x] **Idempotent ingestion**: hash each source file's content; use that hash as part of the Qdrant
      point ID (implemented using deterministic `uuid.uuid5` UUIDs to ensure full compatibility with Qdrant points) so re-running ingestion updates instead of duplicating vectors.
- [x] **A single `ingest.py` CLI script** that walks a `data/` folder, runs load → chunk → extract, and prints a summary (files processed, chunks created) — this becomes
      your reproducible pipeline, not a notebook you run once and forget.

### Phase 2 — Embedding model & persistent vector store
- [x] **Local BGE Embeddings Factory** (`app/core/embeddings.py`): Swap `OpenRouterEmbeddings` in `rag_pipeline.py` for a local open-source `BAAI/bge-base-en-v1.5` HuggingFace embedding model (768 dimensions). (Switched from Gemini due to 100 requests/min free tier rate limits which caused 429 exceptions on batch runs, and to keep the vector space 100% consistent since mixing embedding models degrades retrieval quality).
- [x] **Decide embedding dimensionality**: 768 is the recommended floor for quality vs. storage — with Qdrant's free 4GB disk limit, 768-dim keeps you comfortably under the ~1M-vector free-tier ceiling.
- [x] **Qdrant Collection Creation** (`scripts/ingest.py`): Create the remote collection dynamically with cosine distance and 768 dimensions on Qdrant Cloud.
- [x] **Separate Ingestion from Retrieval** (`scripts/ingest.py`): Replace lazy FAISS indexing with a one-time Qdrant upsert during ingestion and a fast query-time client search.
- [x] **BM25 Cache Optimization** (`src/rag_pipeline.py`): Precompute and load the BM25 index once from the local processed chunks cache on startup instead of parsing per query.
- [x] **Reliability**: Configure Qdrant client connection timeouts and wrap remote calls safely.

### Phase 3 — Hybrid retrieval + reranking
- [x] **Direct SDK Dense Search** (`app/services/retrieval/qdrant_service.py`): Combine direct Qdrant SDK `query_points` client queries with local BM25 index search.
- [x] **FlashRank Reranking Service** (`app/services/retrieval/ranking_service.py`): Add a local CPU-bound `ms-marco-MiniLM-L-12-v2` reranking step after candidate retrieval to narrow the top 10 fused results down to the top 4.
- [x] **Query Cache Normalization** (`src/rag_pipeline.py`): Add string normalization (lowercase, clean spaces, strip punctuation) to cache keys for the TTLCache query layer.

### Phase 4 — Lightweight knowledge graph (the hybrid differentiator)
- [x] Do **not** reach for full Microsoft GraphRAG — it's built for corpora orders of magnitude larger than a personal knowledge base and requires community-summarization passes that cost real LLM budget you don't need to spend.
- [x] **Adjacency graph compilation** (`app/ingestion/graph_builder.py`): compile a structured adjacency-list graph of `Project`, `Company` (real employers/institutes only), `Platform` (vendors/tools), `Skill`, and `Year` based on chunk metadata co-occurrences, serializing it to [knowledge_graph.json](file:///c:/portfolio-web-bot/data/processed/knowledge_graph.json).
- [x] **Graph Query Service** (`app/services/graph_service.py`): load the graph and implement entity word-boundary matching. For matching queries (e.g. "what projects did you build with FastAPI?"), retrieve neighbors and format them into human-readable context statements.
- [x] **Retriever Integration** (`src/rag_pipeline.py`): integrate `GraphService` inside `CustomHybridRetriever`. Prepend any extracted graph context as a standard `Document` at index 0 of final documents list for LLM prioritization.

### Phase 5 — Orchestration: LangGraph planner
- [ ] Replace `ConversationalRetrievalChain` (deprecated pattern, and it can't make a routing decision)
      with a small LangGraph graph: `classify_query → {vector_search | graph_search | both} →
      rerank → generate → guardrail_check`.
- [ ] The planner node's only job: decide whether the query is a lookup question (→ vector), a
      relationship question (→ graph), or ambiguous (→ both, merge results). Keep this classification
      cheap — a small Groq call or even a rule-based keyword check before reaching for an LLM call.
- [ ] Preserve conversation memory — LangGraph has its own state/checkpoint pattern; migrate
      `ConversationBufferMemory`'s role (last-20-messages) into the graph state.

### Phase 6 — Guardrails (input + output)
Directly reuses your DineMate red-teaming work — same threat model, smaller blast radius.
- [ ] **Input rails**: block prompt injection attempts, off-topic requests, and attempts to extract
      system prompt/config. NeMo Guardrails' Colang rails handle the pattern-based cases; back them
      with the same style of classifier you used in DineMate for the harder cases.
- [ ] **Output rails**: block fabricated claims not present in retrieved context (a lightweight
      faithfulness check — does the answer's claims trace back to retrieved chunks), and block PII
      leakage (phone/email should only appear when the contact-info intent is detected, not leak into
      unrelated answers).
- [ ] **Explicit refusal path**: if the guardrail fires, return a clear, honest "I can't help with
      that" rather than a silently degraded answer — this is both a safety and a UX decision.
- [ ] Write down your threat model in `docs/threat-model.md` (injection via chat input, injection via a
      malicious "uploaded" document if you ever add user uploads, jailbreak attempts, PII/resume-data
      exfiltration) — this document itself is a portfolio artifact, not just internal notes.

### Phase 7 — LLM gateway & reliability
- [ ] Self-host the open-source Portkey gateway (Docker, per their quickstart) in front of Groq +
      Gemini, configured with a fallback chain: Groq `llama-3.3-70b-versatile` → Groq
      `openai/gpt-oss-120b` → Gemini `2.5-flash`. Free, no log-volume cap on the OSS gateway itself.
- [ ] Circuit breaker around every external call (Groq, Gemini, Qdrant) — reuse the pattern from
      AuraClaw's Pillar work rather than reinventing it: track consecutive failures, open the circuit
      after N failures, half-open after a cooldown.
- [ ] Idempotent retries with exponential backoff on 429s specifically — Groq and Gemini both return
      `retry-after` headers; respect them instead of guessing.

### Phase 8 — Observability
- [x] Wire `logfire.configure()` + `logfire.instrument_fastapi()` into `main.py` (package already
      installed) — this alone gets you request/response traces, latency, and LLM call spans with
      almost no code.
- [ ] Pick **one** of LangSmith/Langfuse as the LangGraph-specific tracer (both are in
      requirements.txt — running both is redundant for a solo project). LangSmith is already
      referenced via `@traceable` in the current code, so keep LangSmith unless you have a specific
      reason to prefer Langfuse's self-hosted option.
- [ ] Track per-query token count, cost estimate (even if $0 on free tier, log what it *would* cost —
      useful for the portfolio writeup), and latency broken down by retrieval vs. generation.
- [ ] Add a `/metrics` or admin-only endpoint that surfaces these aggregates — a real dashboard is a
      stronger portfolio signal than raw logs.

### Phase 9 — Evaluation
- [ ] Write 20-30 labeled question/answer pairs a recruiter would plausibly ask ("what's your
      experience with LangGraph", "what companies have you worked at", "tell me about DineMate").
- [ ] Run RAGAS metrics (faithfulness, answer relevancy, context precision/recall) against this set —
      this becomes both a regression test and a portfolio-README number.
- [ ] Wrap in DeepEval so `pytest` runs the eval suite in CI on every ingestion or prompt change —
      catching regressions before they reach the deployed bot, not after.

### Phase 10 — API hardening & security
- [ ] Rate limiting middleware on `/chat` (e.g. `slowapi`) — this is public-facing, unrate-limited
      today.
- [ ] Basic API key or origin-check auth if you want to restrict `/chat` to your portfolio frontend
      specifically, separate from the open CORS list.
- [ ] Qdrant free-tier keep-alive: a lightweight scheduled ping (GitHub Actions cron, free) so the
      cluster doesn't auto-suspend after a week of low traffic.
- [ ] Confirm no secrets ever appear in Logfire/LangSmith traces (both support redaction — configure
      it explicitly rather than assuming defaults are safe).

### Phase 11 — Voice interface (optional, high differentiation)
- [ ] You already have Vapi/Retell experience — wire either as a thin voice layer calling the same
      `/chat` endpoint. This is the single biggest "this isn't a course clone" signal for recruiters,
      and it's additive, not a rebuild.

### Phase 12 — Deployment & CI/CD
- [ ] Dockerize the FastAPI app (single `Dockerfile`, already has `pyproject.toml`/`uv.lock` for
      reproducible installs).
- [ ] Deploy to Render free tier first (simplest path, already CORS-configured for your Vercel
      frontends); evaluate HF Spaces if cold starts on Render prove annoying for demo purposes.
- [ ] GitHub Actions: run the Phase 9 eval suite + lint on every push; deploy on merge to `main`.

### Phase 13 — Documentation & portfolio polish
- [ ] `README.md` rewrite: architecture diagram (can reuse the one already generated in this
      conversation), the threat model doc, eval numbers, and an honest "what's free vs. what would
      cost money at scale" section — that honesty is itself a signal of production maturity.
- [ ] Record a short demo (text + voice) for the portfolio site rather than relying on recruiters to
      type queries themselves.

---

## 4. Open decisions / things to verify before you build (don't assume)

- [ ] Confirm current Gemini embedding rate limits in Google AI Studio → Quotas the week you start
      Phase 2 — free-tier numbers have moved twice in the last year.
- [ ] Confirm current Groq per-model RPM/TPM at console.groq.com/settings/limits before finalizing the
      Portkey fallback order.
- [ ] Confirm Qdrant free-cluster inactivity window (currently: suspend after 1 week, delete after 4)
      hasn't changed before relying on the keep-alive cadence in Phase 10.
- [ ] Decide LangSmith vs. Langfuse as primary tracer (Phase 8) — don't run both long-term, it's
      redundant instrumentation for a solo project.
- [ ] Decide whether Neo4j is worth it for Phase 4's graph layer, or whether a JSON/SQLite adjacency
      structure is enough — default recommendation above is the lightweight option; revisit only if
      the entity graph genuinely outgrows it.

---

## 5. Progress log
*(add a dated line here each time you complete a phase, so this file doubles as a build diary)*

- `2026-07-07` — Plan created. Repo audited: v1 traditional RAG baseline confirmed working
  (FAISS + BM25 + Groq), production dependencies present but unwired.
- `2026-07-07` — Phase 0 confirmed complete (secrets secured). Phase 1 source-acquisition step built:
  `app/ingestion/fetchers/github_fetcher.py` + `scripts/fetch_github_sources.py`, pulling
  README + docs/*.md from public and private repos. Logic validated against GitHub's live API
  (Trees + Contents endpoints) before delivery; confirmed unauthenticated rate limit (60/hr) to
  justify the token requirement.
- `2026-07-07` — Fixed a real gap: the fetcher was only catching root-level README.md and a
  top-level docs/ folder, missing nested READMEs in monorepo-style layouts
  (e.g. `backend/README.md`, `packages/ui/README.md`). Filter now matches README.md and docs/*.md
  at any depth, while excluding vendored/noise directories (node_modules, vendor, .venv, etc.).
  Verified the new filter logic against edge cases before editing the shipped file.
- `2026-07-07` — Refactored legacy ad-hoc configuration and custom JSON logging to a production-grade
  system using Pydantic Settings (`app/core/config.py`) and unified Logfire structured logging
  (`app/core/logging.py`). Updated all code references, set up unbuffered console streams with UTF-8
  enforcement on Windows, and verified correct FastAPI startup. Deleted deprecated files `src/config.py`
  and `src/logger.py`.
- `2026-07-07` — Completed Phase 1 Ingestion Loaders & Chunking Pipeline. Created modular PDF and Markdown
  loaders (`app/ingestion/loader/`), header-aware and recursive splitters (`app/ingestion/chunking/`),
  and an entities extractor (`app/ingestion/processor.py`). Developed the CLI `scripts/ingest.py` which
  successfully parsed, chunked (898 chunks), and serialized Umer's resume PDF and 70 GitHub repository
  markdowns with idempotent content-hash IDs and tech stack tagging.
- `2026-07-08` — Completed Phase 2 Embeddings Factory & Qdrant Cloud Integration. Created `app/core/embeddings.py`
  providing local open-source `BAAI/bge-base-en-v1.5` embeddings (768 dimensions) via HuggingFace for zero-cost,
  unlimited local embedding generation. Switched from Gemini embeddings to prevent vector space mixing (which
  degrades search accuracy) and to bypass the 100 requests/minute free-tier rate limits. Integrated Qdrant batch
  upserts into `scripts/ingest.py` (using cosine distance) and eagerly connected the chatbot retriever in
  `src/rag_pipeline.py`. Optimized the RAG path by pre-building the `BM25Retriever` once at server startup from the
  local serialized chunks cache.
- `2026-07-08` — Completed Phase 4 Lightweight Knowledge Graph. Created `app/ingestion/graph_builder.py` which compiles
  an adjacency list knowledge graph (Project, Skill, Company, Year) based on chunk metadata, saving it as `knowledge_graph.json`.
  Developed `app/services/graph_service.py` to match entity word boundaries and return structured relationship contexts at
  query time. Integrated this graph context lookup directly inside `CustomHybridRetriever` in `src/rag_pipeline.py` as a
  top-priority Document injected into the RAG context.

