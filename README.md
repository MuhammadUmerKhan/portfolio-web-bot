# Traditional RAG Bot 🤖  
**`README.md` – Simple, Reliable, Context-Aware**

---

## What is This?  
A **classic Retrieval-Augmented Generation (RAG)** chatbot that reads **your resume PDF** and answers questions **only from the document**. No hallucinations — just facts, paraphrased clearly.  

Perfect for:  
- Personal AI portfolios  
- Resume Q&A bots  
- Hackathon demos  

---

## How It Works (Step-by-Step)  

<img src="../assets/traditional-rag-mermaid-diagram.svg" width="700" height="400" />

| Step | What Happens | Tool |
|------|--------------|------|
| 1️⃣ **Load PDF** | Reads `Muhammad_Umer_Khan_AI_Resume.pdf` | `PyPDFLoader` |
| 2️⃣ **Chunk Text** | Splits into 800-char overlapping pieces | `RecursiveCharacterTextSplitter` |
| 3️⃣ **Embed** | Converts text to vectors | `BAAI/bge-small-en-v1.5` (CPU) |
| 4️⃣ **Store** | Saves vectors in memory | `FAISS` (fast similarity search) |
| 5️⃣ **Query** | User asks → search → retrieve top 3 chunks | `MMR Search` |
| 6️⃣ **Answer** | LLM uses context + smart prompt → clean reply | `Groq + openai/gpt-oss-120b` |

> **Memory**: Remembers last 20 messages for natural chat flow  

---

## Features That Shine  

| Feature | Why It Matters |
|--------|----------------|
| **No Hallucinations** | Answers **only** from your resume |
| **Smart Prompt** | Professional tone, bullets, emojis, contact info |
| **Caching** | Repeated questions? Instant reply! |
| **Fast & Light** | Runs on CPU, no GPU needed |
| **API Ready** | FastAPI + CORS for React frontend |

---

## Setup in 3 Steps  

```bash
# 1. Clone & enter
git clone <your-repo>
cd rag-bot-traditional

# 2. Install
pip install -r requirements.txt

# 3. Add your key
echo "GROQ_API_KEY=your_groq_key_here" > .env
```

Place your resume:  
`../assets/Muhammad_Umer_Khan_AI_Resume.pdf`

---

## Run It!  

```bash
uvicorn app:app --reload
```

Server starts at: `http://localhost:8000`

### Test Endpoints  
| Method | Endpoint | Example |
|--------|----------|--------|
| `GET` | `/` | `{"message": "Hello, I am Muhammad Umer Khan's AI Bot!"}` |
| `POST` | `/chat` | `{"query": "What are your AI skills?"}` |
| `GET` | `/health` | `{"status": "healthy"}` |

---

## Example Q&A  

**You:** _"What projects have you built?"_  
**Bot:**  
> I’ve developed:  
> - **Diagnosify** – LLM-powered medical report analyzer with OCR & RAG  
> - **Banking FAQ Bot** – Fine-tuned Mistral-7B with QLoRA, Streamlit + MongoDB  

---

## Deploy Anywhere  

- Hugging Face Spaces  
- Vercel (via serverless)  
- Docker container  

---

> **Traditional RAG = Simplicity + Accuracy**  
> Great for **production-ready**, **reliable** resume bots.  