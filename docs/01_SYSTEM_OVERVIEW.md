# 🤖 Portfolio Web Bot: System Architecture

A production-grade, state-of-the-art hybrid RAG personal assistant built for speed, scalability, and deep observability. This platform leverages **LangGraph** to handle complex reasoning and a highly optimized free-tier stack for document and relational intelligence.

---

## 🌟 Vision
Most RAG systems fail because they treat every query the same, leading to slow responses and context bloat. Our **Agentic RAG** distinguishes between:
1.  **Conversational Queries**: "Hi", "Who are you?", "How are you?"
2.  **Semantic Queries (Vector)**: "What is your experience with React?", "Tell me about your portfolio project."
3.  **Relational Queries (Graph)**: "What projects did you build in 2023?", "What tech stack connects your web projects?"

By using a **Tool-Calling ReAct Agent** architecture, we ensure that deep technical answers are grounded in the precise retrieval method needed. The Agent intrinsically understands short-term memory, perfectly handles follow-up questions without re-searching the database, and iteratively calls semantic or relational tools only when fresh context is required.

---

## 🏗️ High-Level Flow (ReAct Agent)
```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "primaryColor": "#7C3AED",
    "primaryTextColor": "#ffffff",
    "primaryBorderColor": "#5B21B6",
    "secondaryColor": "#1E1B4B",
    "secondaryTextColor": "#ffffff",
    "tertiaryColor": "#0F172A",
    "tertiaryTextColor": "#94A3B8",
    "noteBkgColor": "#1E293B",
    "noteTextColor": "#E2E8F0",
    "activationBkgColor": "#4C1D95",
    "activationBorderColor": "#7C3AED",
    "signalColor": "#A78BFA",
    "signalTextColor": "#ffffff",
    "labelBoxBkgColor": "#1E1B4B",
    "labelBoxBorderColor": "#7C3AED",
    "labelTextColor": "#C4B5FD",
    "loopTextColor": "#E2E8F0",
    "actorBkg": "#1E1B4B",
    "actorBorder": "#7C3AED",
    "actorTextColor": "#ffffff",
    "actorLineColor": "#6D28D9",
    "sequenceNumberColor": "#ffffff",
    "background": "#0F172A"
  }
}}%%
sequenceDiagram
    participant User
    participant UI as 🖥️ Vercel Frontend
    participant Agent as 🧠 ReAct Agent (LangGraph)
    participant DB as 🗄️ Knowledge Base (Qdrant Cloud)

    User->>UI: Asks Question
    UI->>Agent: POST /query + thread_id
    Agent->>Agent: 🛡️ NeMo Guardrails — check for injection/off-topic

    rect rgb(76, 29, 149)
        Note over Agent, DB: 🔄 ReAct Loop
        Agent->>Agent: Reads message history. Is answer already known?
        alt Needs Semantic Context
            Agent->>DB: 🔍 search_vector_db (RRF + FlashRank → top 5)
            DB-->>Agent: Top 5 Semantic Chunks
        else Needs Relational Context
            Agent->>DB: 🕸️ search_graph_db (entity traversal)
            DB-->>Agent: Structured Relations
        else Answer in History
            Agent->>Agent: ✅ No DB call needed
        end
        Agent->>Agent: Reason on tool outputs
    end

    Agent->>User: 💬 Synthesized Answer
```

---

## 📂 Project Organization
*   **`app/`**: The core Python package containing the LangGraph Agent, Retrieval Pipelines, Services, Guardrails, and Portkey Gateway.
*   **`data/`**: The local ground-truth documentation used for ingestion (git-ignored).
*   **`docs/`**: This documentation suite, capturing the architecture and tracking progress.
*   **`scripts/`**: Developer CLI scripts for ingesting sources, testing guardrails, and testing gateways.

---

## 🚀 Quick Navigation
1.  **Ingestion Engine**: [02_INGESTION_ENGINE.md](02_INGESTION_ENGINE.md)
2.  **Legacy Node Architecture (Deprecated)**: [03_NODE_INTELLIGENCE.md](03_NODE_INTELLIGENCE.md)
3.  **FlashRank Reranking**: [07_FLASHRANK_RERANKING.md](07_FLASHRANK_RERANKING.md)
4.  **Guardrails**: [08_GUARDRAILS.md](08_GUARDRAILS.md)
5.  **LLM Gateway**: [09_LLM_GATEWAY.md](09_LLM_GATEWAY.md)
6.  **Agent Architecture**: [10_AGENT.md](10_AGENT.md)
7.  **Threat Model**: [threat-model.md](threat-model.md)
8.  **Evals Strategy**: [11_EVALS.md](11_EVALS.md)
9.  **Evals Pipeline**: [12_EVALS_PIPELINE.md](12_EVALS_PIPELINE.md)
10. **CI/CD & Dockerization**: [13_CI_CD.md](13_CI_CD.md)
11. **Implementation Plan & Log**: [PLAN.md](PLAN.md)
12. **Agent Context**: [../CLAUDE.md](../CLAUDE.md)

---

> **Next →** [02 — Ingestion Engine](02_INGESTION_ENGINE.md)
