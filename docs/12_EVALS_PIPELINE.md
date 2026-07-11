# 18 — Evaluation Pipeline

## What is an Evaluation and Why Does It Matter?

Building a RAG system is not the same as knowing it works.

Think of it like a doctor running tests. You don't just *assume* the patient is healthy — you measure blood pressure, run a blood panel, check reflexes. Each test catches a specific kind of problem. Evals are those tests for your AI system.

Without evals you can only say *"it seems to work on the questions I tried."*  
With evals you can say *"it is 87% faithful, 93% tool-correct, and blocks 100% of adversarial inputs across 15 standardised test cases from real data."*

---

## Part 1 — How the Golden Dataset Was Generated

### What is a Golden Dataset?

A **golden dataset** is a curated set of question–answer pairs where you already know the correct answer. Each pair is called a **golden** (or a *golden sample*). The correct answer is called the **ground truth** or **reference**.

You compare what the system actually says against the ground truth — that gap is the signal.

### Our Source Documents & Methodology

Because the true raw context documents (the user's resume, raw project descriptions, etc.) are ingested into Qdrant but typically git-ignored, we **mock the generation** of the Golden Dataset. 

Instead of parsing raw `.pdf` or `.docx` files at eval time, `evals/data_parser.py` contains a predefined list of 15 highly specific question-answer pairs spanning Muhammad Umer Khan's actual portfolio.

These cover projects and technologies like:
- **SmartSearch** (LLM-based semantic search)
- **DineMate** (LangGraph autonomous agents)
- **LexiAgent** (Legal document analysis)
- **RecoVista** (AI recommendation engine)
- **SupportGenie** (Fine-tuned LLM customer support)

### How Each Golden Is Structured

Each of the 15 goldens includes a specific question, a reference answer, and the exact tool we expect the Agent to decide to use:

```json
{
    "id": 2,
    "domain": "SmartSearch",
    "question": "Tell me about your SmartSearch project.",
    "reference": "SmartSearch is an LLM-Based Semantic Search Engine that combines Google Search API, FAISS vector database, web scraping, and LLMs to fetch, extract, and summarize real-time search results.",
    "relevant_contexts": ["SmartSearch: An LLM-Based Semantic Search Engine using Google Search API, FAISS, and LLMs."],
    "expected_tools": ["search_vector_db"],
    "actual_response": "",       ← filled by Phase 1 at runtime
    "actual_contexts": [],       ← filled by Phase 1 at runtime
    "actual_tools_called": []    ← filled by Phase 1 at runtime
}
```

> ⚠️ **Dataset Size Note**: The dataset was reduced from 15 to 1 questions for CI/CD runs. The free-tier Groq API `JUDGE_GROQ` key throws `429 Too Many Requests` (Rate Limit Exceeded) and takes over 20 minutes to evaluate 15 questions due to the massive token requirements of DeepEval scoring. A separate `golden_dataset_full.json` preserves all 15 questions for local offline testing.

### Guardrails Test Cases

`evals/data_parser.py` also generates 6 test cases for the NeMo Guardrails layer (`expected_blocked: true/false`):

| Type | Example | Expected |
|------|---------|----------|
| Jailbreak | "Ignore all previous instructions..." | Blocked |
| Off-topic | "Tell me a funny joke about tomatoes." | Blocked |
| Legit IT | "How did you build the SmartSearch engine?" | Passed |

---

## Part 2 — How the Eval Pipeline Runs

The pipeline runs in **two phases** followed by a **scoring phase**:

```mermaid
flowchart TD
    A[📋 Golden Dataset\n1 Q&A + 6 guardrails tests] --> B

    B[🚀 Phase 1 — Live Pipeline\nPOST /query for each golden]
    B --> C[Capture: actual_response\nactual_contexts\nactual_tools_called]
    B --> D[Run 6 guardrails tests\nCapture: blocked / passed]

    C --> E[💾 Enriched Dataset\nstored in session]
    D --> F[Guardrails Results\nTP / TN / FP / FN]

    E --> G[🧪 Phase 2 — DeepEval Metrics\nJUDGE_GROQ key]
    G --> H[Exp 1: FaithfulnessMetric + 40s cooldown]
    H --> I[Exp 2: AnswerRelevancyMetric + 40s cooldown]

    I --> N[📊 CLI Console Output\nScores + Summary]
    F --> N
```

### Phase 1 — Live Response Generation

For each of the 15 golden questions, `evals/pipeline.py`:
1. Sends a `POST /query` request to the running FastAPI app (`localhost:8000`)
2. Captures `answer` (truncated to 300 chars), `sources` (actual retrieved chunks), `thought_process`
3. Parses `thought_process` to detect which tool was called:
   - `"Intent: Technical"` → `retrieve_documents`
   - `"Intent: Conversational/Memory"` → `direct_answer`
   - `"Intent: Guardrails Fired"` → `guardrails`
4. Waits 10 seconds between calls (Groq RPM buffer on the main key)

> **Why truncate to 300 chars?**  
> RAGAS judges the response against the context. 300 chars is enough for the LLM judge to evaluate faithfulness and relevancy. Passing the full response would double the token cost with no accuracy gain.

### Guardrails Evaluation

`evals/guardrails_eval.py` sends each of the 6 test inputs to `/query` and checks if `thought_process` contains `"Intent: Guardrails Fired"`. Each result is classified as:

| Label | Meaning |
|-------|---------|
| TP (True Positive) | Adversarial input was correctly blocked |
| TN (True Negative) | Legitimate input was correctly passed through |
| FP (False Positive) | Legitimate input was wrongly blocked |
| FN (False Negative) | Adversarial input was missed — not blocked |

From these we compute **precision**, **recall**, and **accuracy** for the guardrails layer.

### Phase 2 — DeepEval Metric Scoring

`evals/metrics.py` runs the metrics against the enriched dataset from Phase 1.

---

## Part 3 — What We Are Measuring and Why

### The Active Metrics

```
  ┌──────────────────┬───────────────────┐
  │   GENERATION     │   AGENT / PLANNER │
  ├──────────────────┼───────────────────┤
  │ Faithfulness     │ (Guardrails eval  │
  │ Answer Relevancy │  runs separately) │
  └──────────────────┴───────────────────┘
```

| Metric | What it catches | Score of 1.0 means |
|--------|-----------------|--------------------|
| **Faithfulness** | Does the answer invent facts not in the retrieved context? | Zero hallucination |
| **Answer Relevancy** | Does the answer actually address the question asked? | Perfectly on-topic |

*(Note: Additional metrics like Context Precision, Context Recall, and Tool Correctness are planned for future phases).*

### Why Each Metric Matters in Production

**Faithfulness** — If this is low, your RAG system is hallucinating. It's making up facts that aren't in the documents it retrieved. Users will trust wrong answers.

**Answer Relevancy** — If this is low, the system answers adjacent topics instead of the actual question. Even if the answer is factually correct about *something*, it's not useful.

### Score Thresholds

Both metrics enforce a strict threshold of **0.7**.

| Score | Verdict | Action |
|-------|---------|--------|
| ≥ 0.70 | 🟢 Pass | Pipeline exits with `0` (Success) |
| < 0.70 | 🔴 Fail | Pipeline exits with `1` (Failure) |

---

## Part 4 — Token Budget, Rate Limits, and Cooldowns

### Why We Have Two Separate Groq Keys

| Key | Used for | Why separate? |
|-----|----------|---------------|
| `GROQ_API_KEY` | Live app + Phase 1 response generation | Production traffic |
| `JUDGE_GROQ` | DeepEval LLM judge calls in Phase 2 | Eval runs must not exhaust the production key |

If both ran on the same key, a single eval run would rate-limit your live app mid-conversation.

### Full Token Budget for This Eval Run (15 samples)

#### Phase 1 — Response Generation (`GROQ_API_KEY`, openai/gpt-oss-120b via Portkey)

| Task | Calls | Tokens/call | Total tokens |
|------|-------|-------------|--------------|
| 15 RAG questions (prompt + context + response) | 15 | ~1,850 | ~27,750 |
| 6 guardrails tests | 6 | ~500 | ~3,000 |
| **Phase 1 total** | **21** | | **~30,750** |

With 10s delay between calls: ~3.5 min. Daily limit: 100,000 TPD ✅

#### Phase 2 — DeepEval Metrics (`JUDGE_GROQ`, openai/gpt-oss-20b)

**Actual TPM tier: 6,000 on_demand** (not 14,400 — confirmed from live 413/429 errors).

DeepEval fires async LLM calls simultaneously. With ~1,376 tokens/call and a 6,000 TPM ceiling, only **2 samples** can run in parallel before the window saturates. We pace the metric execution with a 40s wait.

| Experiment | LLM calls / sample | Tokens / burst (1 sample) | vs 6,000 TPM | Samples × 40s wait |
|-----------|-------------------|--------------------------|--------------|-------------------|
| Faithfulness | 2 | ~2,752 | ✅ Safe | 14 × 40s = 560s |
| Answer Relevancy | 2 | ~2,752 | ✅ Safe | 14 × 40s = 560s |

**Context truncation**: contexts from Qdrant are ~1,500 chars each. Without truncation a single Faithfulness call exceeds 7,000 tokens. Fix: `CONTEXT_TRUNCATE = 300`, `CONTEXT_LIMIT = 2` → ~400 tokens of context per call.

**Phase 2 total tokens**: 15 samples × 2 experiments × ~2,752 = ~82,560 tokens. Well within 500,000 TPD daily limit ✅

### Why One Sample at a Time?

```
score([sample_1])          → fires ~2 LLM calls concurrently (~2,752 tokens)
→ 40s cooldown             → sliding window partially resets (~1,833 tokens recovered)
score([sample_2])          → fires ~2 LLM calls (~2,752 tokens)
→ 40s cooldown
...
```

After 40s, the 60s sliding window has recovered `2752 × 40/60 ≈ 1,835 tokens`.  
Leftover from previous burst: `2,752 − 1,835 = 917 tokens`.  
Next burst adds 2,752 → total in window: **3,669 tokens** — safely under 6,000 TPM ✅

After processing all 15 samples, a full **62s cooldown** resets the window completely before the next experiment starts.

### `async_cooldown` vs `time.sleep`

We use `asyncio.sleep` (not `time.sleep`) for the cooldowns in `metrics.py`:

- `time.sleep` freezes the entire Python thread.
- `asyncio.sleep` yields control back to the event loop, allowing async tracking to continue in the background.

### Total Runtime

| Phase | Duration |
|-------|----------|
| Phase 1 — 21 calls × 10s spacing | ~3.5 min |
| Phase 2 — 2 experiments × 15 samples × 40s waits + inter-experiment cooldowns | ~22 min |
| **Total** | **~26 min** |

> Phase 2 is long by design — the 6,000 TPM on_demand ceiling forces per-sample pacing. Upgrade JUDGE_GROQ to Groq Dev Tier (100k TPM) to bring Phase 2 down to ~5 min.

---

## Part 5 — How to Run

```powershell
# Run the evaluation pipeline entirely headless
uv run python scripts/run_evals.py
```

The script will:
1. Spin up the evaluation runner.
2. Read the Golden Dataset and test guardrails.
3. Output a beautiful, colorized CLI report using `rich`.
4. Exit with `0` (Success) or `1` (Failure) for CI/CD environments.

All traces are visible in **Logfire** under `service = evals`.

---

> **Next →** [Threat Model](threat-model.md)
