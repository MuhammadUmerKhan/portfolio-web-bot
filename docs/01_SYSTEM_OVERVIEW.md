# 🤖 Portfolio Web Bot: System Architecture

A production-grade, state-of-the-art hybrid RAG personal assistant built for speed, scalability, and deep observability. This platform leverages **LangGraph** to handle complex reasoning and a highly optimized free-tier stack for document and relational intelligence.

---

## 🌟 Vision
Most RAG systems fail because they treat every query the same, leading to slow responses and context bloat. Our **Agentic RAG** distinguishes between:
1.  **Conversational Queries**: "Hi", "Who are you?", "How are you?"
2.  **Semantic Queries (Vector)**: "What is your experience with React?", "Tell me about your portfolio project."
3.  **Relational Queries (Graph)**: "What projects did you build in 2023?", "What tech stack connects your web projects?"

By using a **Planner-Retriever-Responder** architecture driven by Pydantic structured outputs, we ensure that deep technical answers are grounded in the precise retrieval method needed, while conversational interactions remain fluid, fast, and entirely bypass the database.

---

## 🏗️ High-Level Flow
```mermaid
sequenceDiagram
    participant User
    participant UI as Vercel Frontend
    participant Agent as Agent Brain (FastAPI / LangGraph)
    participant DB as Knowledge Base (Qdrant Cloud)

    User->>UI: Asks Question
    UI->>Agent: Request with thread_id
    Agent->>Agent: Planner classifies intent & assigns `search_type`
    
    alt Semantic or Relational Context Needed
        Agent->>DB: Execute Vector/BM25 and/or Graph Lookup
        DB-->>Agent: Raw Chunks & Extracted Relations
        Agent->>Agent: Reciprocal Rank Fusion (RRF) & FlashRank Reranking
        Agent->>Agent: Responder synthesizes context
    else Conversational
        Agent->>Agent: Planner outputs Direct Response instantly
    end
    
    Agent->>User: Synthesized Answer
```

---

## 📂 Project Organization
*   **`app/`**: The core Python package containing the LangGraph Agent, Retrieval Pipelines, and Services.
*   **`data/`**: The local ground-truth documentation used for ingestion (git-ignored).
*   **`docs/`**: This documentation suite, capturing the architecture and tracking progress.
*   **`scripts/`**: Developer CLI scripts for ingesting sources and processing the graph.

---

## 🚀 Quick Navigation
1.  **Ingestion Engine**: [02_INGESTION_ENGINE.md](02_INGESTION_ENGINE.md)
2.  **Node Intelligence**: [03_NODE_INTELLIGENCE.md](03_NODE_INTELLIGENCE.md)
3.  **FlashRank Reranking**: [07_FLASHRANK_RERANKING.md](07_FLASHRANK_RERANKING.md)
4.  **Tracing & Observability**: [04_TRACING_AND_OBSERVABILITY.md](04_TRACING_AND_OBSERVABILITY.md)
5.  **Environment Variables**: [05_ENVIRONMENT_VARIABLES.md](05_ENVIRONMENT_VARIABLES.md)
6.  **Known Gotchas**: [06_KNOWN_GOTCHAS.md](06_KNOWN_GOTCHAS.md)
7.  **Implementation Plan & Log**: [PLAN.md](PLAN.md)
8.  **Agent Context**: [../CLAUDE.md](../CLAUDE.md)
