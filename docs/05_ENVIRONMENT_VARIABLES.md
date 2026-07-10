# 🔑 Environment Variables & Configuration

The project uses a `.env` file for local development. All configuration is managed via **Pydantic Settings** in `app/core/config.py` for strict type safety. Copy `.env.example` to `.env` and fill in your values before running anything.

---

## 🧠 LLMs

| Variable | Description | Example |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Primary key for Groq LLM calls (Planner + Responder nodes) | `gsk_...` |
| `FALL_GROQ_API_KEY` | Fallback Groq key | `gsk_...` |

---

## 🌐 GitHub Integration

| Variable | Description | Example |
| :--- | :--- | :--- |
| `GITHUB_TOKEN` | Fine-grained PAT to fetch repositories | `github_pat_...` |
| `GITHUB_USERNAME` | Umer's GitHub username | `MuhammadUmerKhan` |

---

## 🗄️ Vector Database (Qdrant)

| Variable | Description | Example |
| :--- | :--- | :--- |
| `QDRANT_API_KEY` | Qdrant Cloud access token | `xyz...` |
| `QDRANT_END_POINT` | Full URL of your Qdrant Cloud cluster | `https://your-cluster.cloud.qdrant.io:6333` |
| `QDRANT_CLUSTER_ID` | The ID string for the cluster | `...` |
| `QDRANT_COLLECTION_NAME` | The collection namespace to use | `personal_kb` |

---

## 🕵️ Observability

| Variable | Description | Example |
| :--- | :--- | :--- |
| `LOGFIRE_TOKEN` | Pydantic Logfire token — traces every API call, parsing step, and retrieval span | `logfire_...` |
| `LANGSMITH_API_KEY` | LangSmith token — records LangGraph node transitions, prompts, and token usage | `lsv2_...` |
| `LANGCHAIN_TRACING_V2` | Enable/disable LangSmith tracing | `true` |
| `LANGCHAIN_ENDPOINT` | LangSmith API endpoint | `https://api.smith.langchain.com` |

---

## 🔧 Core App Settings

| Variable | Description | Example |
| :--- | :--- | :--- |
| `HF_HOME` | HuggingFace cache directory for downloading the BGE embedding model | `/tmp/huggingface` |
| `MODEL_NAME` | The primary Groq model to use | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | The HuggingFace sentence transformer to use | `BAAI/bge-base-en-v1.5` |
| `RESUME_PATH` | Path to the PDF resume | `assets/Muhammad_Umer_Khan_AI_Resume.pdf` |

---

## 🔒 Security Best Practices
1.  **Never** commit your `.env` file to Git — it is in `.gitignore`.
2.  Use `.env.example` as the template when onboarding new developers.
