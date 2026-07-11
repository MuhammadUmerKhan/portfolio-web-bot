# 🏗️ Portfolio Bot — System Architecture

> A production-grade, **Agentic Hybrid RAG** personal assistant engineered for speed, safety, and deep observability. Every design decision was made with production constraints in mind — free-tier limits, zero-disk deployments, and recruiter-grade transparency.

---

## 📐 System Overview

The full system is composed of 8 distinct layers. The diagram below shows how they connect:

```mermaid
graph LR

    subgraph INTERFACE ["Interface Layer"]
        direction TB
        FRONTEND["Vercel Frontend\nPortfolio Site"]
        API["FastAPI\n/query  /health"]
    end

    subgraph SAFETY ["Safety Gate"]
        direction TB
        RATE["slowapi\n5 req/min"]
        GUARD{"NeMo Guardrails\nColang Input Rules"}
    end

    subgraph AGENT ["LangGraph ReAct Agent"]
        direction TB
        GNODE["Guard Node\napp/agents/nodes/guard.py"]
        ANODE["Agent Node\napp/agents/nodes/agent.py"]
        TNODE["Tools Node\napp/agents/nodes/tools.py"]
        MEM[("MemorySaver\nThread-Scoped History")]
    end

    subgraph RETRIEVAL ["Retrieval Engine"]
        direction TB
        QD[("Qdrant Cloud\nVector DB - 768-dim")]
        BM25["BM25 Index\nIn-Memory Sparse"]
        RRF["RRF Fusion\nDense + Sparse"]
        FR["FlashRank\nCross-Encoder - Local CPU"]
        KG["Knowledge Graph\nIn-Memory JSON Adjacency"]
    end

    subgraph GATEWAY ["LLM Gateway - Portkey"]
        direction TB
        PK["Portkey Cloud\nFallback - Retry - Cache"]
        G1["gpt-oss-120b\nPrimary - Groq LPU"]
        G2["gpt-oss-20b\nFallback - Groq LPU"]
    end

    subgraph INGEST ["Ingestion Pipeline (Offline)"]
        direction TB
        GH["GitHub API\nfetch_github_sources.py"]
        PDF["Resume PDF\nPyPDFLoader"]
        SPLIT["Header-Aware Splitter\n800 chars - 300 overlap"]
        EMB["BAAI/bge-base-en-v1.5\nLocal - 768-dim - HuggingFace"]
    end

    subgraph OBS ["Observability (Three-Pillar)"]
        direction LR
        LF["Logfire\nInfra Spans"]
        LS["LangSmith\nAgent Traces"]
        PKO["Portkey\nLLM Cost + Latency"]
    end

    FRONTEND -->|"POST /query + thread_id"| API
    API --> RATE --> GNODE
    GNODE --> GUARD
    GUARD -->|"blocked"| FRONTEND
    GUARD -->|"safe"| ANODE
    ANODE -->|"Tool Call"| TNODE
    TNODE --> QD
    TNODE --> KG
    QD --> BM25 --> RRF --> FR --> ANODE
    ANODE -->|"Final Answer"| FRONTEND
    MEM <-.->|"Checkpoint"| ANODE
    ANODE --> PK --> G1
    PK -.->|"429/503 fallback"| G2
    GH --> SPLIT
    PDF --> SPLIT
    SPLIT --> EMB --> QD
    API -..->|"spans"| LF
    ANODE -..->|"traces"| LS
    PK -..->|"metrics"| PKO

    classDef ui        fill:#2563EB,stroke:#1E40AF,color:#fff
    classDef safety    fill:#DC2626,stroke:#991B1B,color:#fff
    classDef agent     fill:#7C3AED,stroke:#5B21B6,color:#fff
    classDef retrieval fill:#059669,stroke:#065F46,color:#fff
    classDef gateway   fill:#D97706,stroke:#92400E,color:#fff
    classDef ingest    fill:#4F46E5,stroke:#3730A3,color:#fff
    classDef obs       fill:#0D9488,stroke:#0F766E,color:#fff
    classDef memory    fill:#6D28D9,stroke:#4C1D95,color:#fff

    class FRONTEND,API ui
    class RATE,GUARD,GNODE safety
    class ANODE,TNODE agent
    class QD,BM25,RRF,FR,KG retrieval
    class PK,G1,G2 gateway
    class GH,PDF,SPLIT,EMB ingest
    class LF,LS,PKO obs
    class MEM memory
```

---

## 1. Ingestion Pipeline

The ingestion pipeline is a **one-time, offline process** that converts raw documents into a queryable knowledge base. It is idempotent — safe to re-run with content-hash deduplication.

### 1.1 Source Acquisition

```mermaid
flowchart LR
    A["GitHub Fine-grained PAT\nRead-only Contents + Metadata"] --> B["GitHub Trees API\nFlat path manifest per repo"]
    B --> C{Filter}
    C -->|"README.md at any depth"| D["Raw Markdown"]
    C -->|"docs at any depth"| D
    C -->|"node_modules / vendor / .venv"| E["Skipped"]
    F["Local Resume PDF"] --> G["PyPDFLoader\npage-by-page text extraction"]
    D --> H["Raw Content Buffer"]
    G --> H
```

- Pulls from **public and private repositories** using a fine-grained PAT.
- **Git blob SHA manifest** prevents re-fetching unchanged files on incremental runs.
- Filter matches `README.md` and `docs/*.md` at **any nesting depth**.

### 1.2 Chunking

```mermaid
flowchart TD
    Raw["Raw Document"] --> DT{Document Type}
    DT -->|"Markdown"| MH["MarkdownHeaderTextSplitter\nPreserves H1 / H2 / H3 boundaries"]
    DT -->|"PDF"| RC["RecursiveCharacterTextSplitter\n800 chars - 300 overlap"]
    MH --> RC
    RC --> Chunks["Chunks + Metadata\nchunk_id - source_url - doc_type"]
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 800 chars | Fits inside the 2048-token embedding input limit |
| Overlap | 300 chars | Preserves context across chunk boundaries |
| Markdown mode | Header-aware | Keeps section titles with their content |
| PDF mode | Recursive | Handles free-form resume prose |

### 1.3 Embedding and Storage

```mermaid
flowchart LR
    Chunks["Text Chunks"] --> EMB["BAAI/bge-base-en-v1.5\nHuggingFace - Local CPU\n768 dimensions"]
    EMB --> QD[("Qdrant Cloud\nCosine Distance\nFree Tier")]
    Chunks --> KG["knowledge_graph.json\nProject - Skill - Company - Year\nAdjacency List"]
```

**Why local BGE instead of a cloud embedding API?**

| | Cloud Embeddings (Gemini) | Local BGE |
|--|--|--|
| Free rate limit | 100 RPM — 429s on batch | Unlimited |
| Network latency | Round-trip per batch | None — in-process |
| Consistency | Risk if model version changes | Pinned, stable |
| Batch ingest risk | High — 898 chunks | Zero |

---

## 2. Retrieval Engine

Every tool call triggers a **3-stage hybrid retrieval pipeline**.

```mermaid
flowchart TD
    Q["User Query"] --> S1A["Stage 1A — Dense Vector Search\nBAAI/bge-base-en-v1.5 768-dim\nQdrant ANN - top 30"]
    Q --> S1B["Stage 1B — Sparse Keyword Search\nBM25 Index - In-Memory\nTF-IDF keyword matching - top 30"]

    S1A --> RRF["Stage 2 — Reciprocal Rank Fusion\n1 / (k + rank) per document\n60 fused candidates"]
    S1B --> RRF

    RRF --> STRIP["Emoji Stripping\nPrevents cross-encoder score collapse"]
    STRIP --> FR["Stage 3 — FlashRank Reranker\nms-marco-MiniLM-L-12-v2\nONNX Quantized Cross-Encoder\nLocal CPU - under 100ms\ntop 5 docs"]

    FR --> LLM["Groq LLM\nopenai/gpt-oss-120b\nFinal answer synthesis"]
```

| Stage | Method | Strength | Weakness |
|-------|--------|----------|---------|
| Dense (Qdrant) | ANN on 768-dim vectors | Finds semantically similar content | Misses exact keyword matches |
| Sparse (BM25) | TF-IDF keyword frequency | Exact match for names, titles | No semantic understanding |
| RRF | Rank fusion | Best of both worlds | Returns 60 candidates — still too many |
| FlashRank | Cross-encoder reranking | Deeply accurate relevance scoring | Only feasible on small candidate set |

---

## 3. Knowledge Graph

The Knowledge Graph answers **relational queries** that semantic similarity alone cannot satisfy — e.g., *"What tech stacks did you use in 2024?"*

```mermaid
graph TD
    BOOT["Server Startup\nfetch_all_chunks from Qdrant"] --> GB["graph_builder.py\nScans chunk metadata payloads\nBuilds adjacency list"]
    GB --> KGJ[("knowledge_graph.json\nIn-Memory\nProject - Skill - Company - Year")]
    QUERY["User Query"] --> GS["graph_service.py\nWord-boundary entity matcher"]
    KGJ --> GS
    GS --> CTX["Structured Context\nInjected as priority Document"]
    CTX --> ANODE["Agent Node"]
```

**Key design:** The graph is built entirely from **Qdrant metadata payloads** at startup. No Neo4j, no extra credentials. The deployment stays stateless and single-service.

---

## 4. LangGraph ReAct Agent

### 4.1 Why ReAct Instead of a Pipeline?

The original pipeline (`Planner -> Retriever -> Responder`) caused **Context Amnesia**: on follow-up questions, the Planner generates a new search query and overwrites previous context.

```mermaid
flowchart LR
    subgraph OLD ["Old Pipeline - Context Amnesia"]
        P["Planner"] --> R["Retriever\nFetches NEW docs"] --> RS["Responder"]
        P2["Follow-up"] --> R2["Retriever\nOverwrites context!"] --> RS2["Responder\nForgets previous turn"]
    end

    subgraph NEW ["ReAct Agent - Persistent Memory"]
        A["Agent Node"] -->|"Needs context"| T["Tools Node"]
        T -->|"Appends ToolMessage"| A
        A -->|"Knows from history"| FINAL["Final Answer\n0 extra DB calls"]
    end
```

### 4.2 The Execution Loop

```mermaid
flowchart TD
    START(["User Input"]) --> GUARD["Guard Node\nNeMo Guardrails intent classification"]
    GUARD -->|"jailbreak / off-topic"| END1(["END - Hardcoded refusal"])
    GUARD -->|"safe"| AGENT["Agent Node\nGroq gpt-oss-120b via Portkey\nPersona + bound tools"]

    AGENT -->|"Has answer from history"| ANSWER["Final AIMessage"]
    AGENT -->|"Needs fresh context"| TOOLS["Tools Node"]

    TOOLS -->|"search_vector_db"| VEC["Hybrid Retrieval\nRRF + FlashRank"]
    TOOLS -->|"search_graph_db"| GDB["Knowledge Graph\nEntity Traversal"]

    VEC -->|"ToolMessage appended to state"| AGENT
    GDB -->|"ToolMessage appended to state"| AGENT

    ANSWER --> END2(["END"])
    MEM[("MemorySaver\nper thread_id")] <-.->|"Checkpoint every step"| AGENT
```

### 4.3 State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rail_fired: bool
```

The `add_messages` reducer **appends** — never replaces. Every `ToolMessage` result is inserted directly into the shared state, giving the LLM full recall of every search it has performed.

---

## 5. Safety Layer (NeMo Guardrails)

```mermaid
flowchart TD
    USER["User Message"] --> NR["NeMo Guardrails\ngpt-oss-20b - Fast Classification"]
    NR --> INTENT{Colang Intent Match}

    INTENT -->|"user attempt jailbreak"| JB["Hardcoded refusal\nrail_fired = True\nGraph exits immediately"]
    INTENT -->|"user ask off topic"| OT["Branded refusal\nrail_fired = True\nGraph exits immediately"]
    INTENT -->|"user says greeting"| GR["Return greeting\nNo RAG, no LLM call"]
    INTENT -->|"user ask capabilities"| CAP["Return capabilities\nNo RAG, no LLM call"]
    INTENT -->|"No match"| PASS["Pass to Agent Node\nFull RAG pipeline"]
```

**Why gpt-oss-20b for the guard?** The guard runs on every request. Its task — intent classification — is semantically simple. The 20B model is fast enough and accurate enough, without the latency overhead of the 120B model.

| Threat | Layer | Mitigation |
|--------|-------|-----------|
| Prompt injection | Guard Node | Colang jailbreak flow with semantic matching |
| Off-topic abuse | Guard Node | Branded refusal — no tokens wasted |
| Resource exhaustion | API middleware | slowapi — 5 requests per minute per IP |
| Cross-origin attacks | API middleware | CORS allow-list: 3 known frontend origins |
| PII in traces | Observability | LangSmith — fully auditable |
| Output hallucination | RAG grounding | Strict context-only prompting |

---

## 6. LLM Gateway (Portkey)

```mermaid
sequenceDiagram
    participant App as Python App
    participant PK as Portkey Cloud Gateway
    participant G1 as Groq gpt-oss-120b
    participant G2 as Groq gpt-oss-20b

    App->>PK: ChatOpenAI call + Config-ID header
    PK->>G1: Route to primary target
    G1-->>PK: 429 Too Many Requests
    PK->>G1: Retry 1
    G1-->>PK: 429 Too Many Requests
    PK->>G1: Retry 2
    G1-->>PK: 429 Too Many Requests
    Note over PK: 3 retries exhausted — fallback
    PK->>G2: Route to fallback target
    G2-->>PK: 200 OK
    PK-->>App: Response (app is unaware of the fallback)
```

**Why ChatOpenAI not ChatGroq?** `ChatGroq` is hardwired to `api.groq.com`. `ChatOpenAI` accepts a custom `base_url`, making it the standard way to inject any proxy layer without writing a custom LangChain client.

---

## 7. Observability — Three-Pillar Stack

```mermaid
graph TD
    REQ["Incoming Request"] --> LF["Logfire\nInfrastructure Layer"]
    REQ --> LS["LangSmith\nAgent Graph Layer"]
    REQ --> PK["Portkey\nLLM Economics Layer"]

    LF --> LFD["API response time end-to-end\nQdrant query timing\nGuardrail evaluation duration\nIngestion pipeline performance"]
    LS --> LSD["Full LangGraph state transitions\nExact system prompts sent to Groq\nPlannerOutput Pydantic schema\nTool call inputs and raw outputs"]
    PK --> PKD["Tokens per query prompt + completion\nPure LLM latency\nFallback trigger frequency\nCache hit rate"]
```

All three share the same `thread_id`. A single ID lets you trace: Logfire API span → LangSmith graph trace → exact Portkey LLM call.

---

## 8. Evaluation Suite

```mermaid
flowchart TD
    GD[("Golden Dataset\n1 automated sample\n15 samples in full dataset")] --> P1["Phase 1 - Response Generation\nPOST /query for each question\nCaptures actual_response - actual_contexts"]

    P1 --> P2["Phase 2 - DeepEval Scoring\nDedicated JUDGE_GROQ key via Portkey\n40s cooldown between samples\nThreshold = 0.7"]

    P2 --> M1["FaithfulnessMetric\nIs the answer grounded in the retrieved docs?"]
    P2 --> M2["AnswerRelevancyMetric\nDoes the answer address the actual question?"]

    M1 & M2 --> CLI["CLI Rich Report\nscripts/run_evals.py\nExit 0 = pass - Exit 1 = fail"]
```

> **Dataset note:** Limited to 1 sample for CI runs to avoid Groq 429 on the Judge key (6,000 TPM free tier). Full 15-question dataset preserved in `golden_dataset_full.json`.

---

## 9. Design Decisions Table

| Decision | Chosen Approach | Why | Trade-off |
|----------|----------------|-----|-----------|
| Embeddings | Local BAAI/bge-base-en-v1.5 | Zero API calls, no 429 risk, pinned version | 768-dim vs 3072-dim cloud — sufficient for this domain |
| Agent pattern | ReAct loop | Natively solves Context Amnesia on follow-ups | More complex state management |
| LLM routing | Portkey cloud gateway | Hot-swap providers without redeploy | Adds Portkey as external dependency |
| Retrieval | RRF fusion Qdrant + BM25 | Consistently beats either retriever alone | BM25 rebuilt in-memory at every startup |
| Reranking | FlashRank local CPU ONNX | Zero latency, zero cost, no external API | Quantized model adds 30-50 MB to bundle |
| Guardrails | Input-only NeMo | Protects at gate, avoids double LLM call | No output-side PII redaction |
| Auth | CORS allow-list only | API key in browser JS is always visible | Acceptable for read-only public portfolio bot |
| Graph storage | In-memory JSON adjacency | Zero extra DB, stateless deploy | Not suitable for millions of nodes |
| Memory | MemorySaver per thread_id | Multi-turn conversations, zero extra infrastructure | In-memory only — resets on server restart |
| Free vs paid | 100% free-tier stack | Production thinking within real constraints | Rate limits require pacing in eval suite |

---

## 📚 Deep Dive References

| Layer | Document |
|-------|---------|
| Ingestion | [02 — Ingestion Engine](docs/02_INGESTION_ENGINE.md) |
| Node Architecture | [03 — Node Intelligence](docs/03_NODE_INTELLIGENCE.md) |
| Observability | [04 — Tracing and Observability](docs/04_TRACING_AND_OBSERVABILITY.md) |
| Environment Setup | [05 — Environment Variables](docs/05_ENVIRONMENT_VARIABLES.md) |
| Reranking | [07 — FlashRank Reranking](docs/07_FLASHRANK_RERANKING.md) |
| Guardrails | [08 — NeMo Guardrails](docs/08_GUARDRAILS.md) |
| LLM Gateway | [09 — LLM Gateway](docs/09_LLM_GATEWAY.md) |
| ReAct Agent | [10 — Agent Architecture](docs/10_AGENT.md) |
| Threat Model | [Threat Model](docs/threat-model.md) |
| Eval Theory | [11 — Evals](docs/11_EVALS.md) |
| Eval Pipeline | [12 — Evals Pipeline](docs/12_EVALS_PIPELINE.md) |
| Build Log | [PLAN.md](docs/PLAN.md) |