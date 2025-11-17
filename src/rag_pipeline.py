import httpx, os, cachetools, warnings
from langsmith import traceable
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from src.config import GROQ_API_KEY, RESUME_PATH, MODEL_NAME, EMBEDDING_MODEL
from src.logger import logging

# Ignore warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class CustomDocChatbot:
    """A RAG-based chatbot for answering questions using a resume PDF."""
    
    query_cache = cachetools.TTLCache(maxsize=500, ttl=600)

    def __init__(self):
        """Initialize the chatbot with LLM, embeddings, and RAG chain."""
        self.llm = self.configure_llm()
        self.embeddings = None
        self.vector_db = None
        self.memory = None
        self.qa_chain = None
        self.http_client = httpx.AsyncClient(timeout=15.0)
        logger.info({"message": "🤖 CustomDocChatbot initialized"})

    def configure_llm(self):
        """Configure the Groq LLM with specified model and API key."""
        try:
            llm = ChatGroq(
                model_name=MODEL_NAME,
                temperature=0.5,
                groq_api_key=GROQ_API_KEY
            )
            logger.info({"message": "✅ Groq LLM configured successfully"})
            return llm
        except Exception as e:
            logger.error({"message": f"❌ Failed to configure LLM: {str(e)}"})
            raise 
    
    @traceable(run_type="tool", name="Embeddings_Initializer")
    def configure_embedding_model(self):
        """Configure OpenAI embeddings."""
        try:
            embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                dimensions=None,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info({"message": "✅ OpenAI embeddings configured"})
            return embeddings
        except Exception as e:
            logger.error({"message": f"❌ Failed to configure OpenAI embeddings: {str(e)}"})
            raise

    @lru_cache(maxsize=1)
    @traceable(run_type="tool", name="PDF_Loader")
    def load_pdf(self):
        """Load the resume PDF from the specified path."""
        try:
            if not os.path.exists(RESUME_PATH):
                raise FileNotFoundError(f"PDF not found at {RESUME_PATH}")
            loader = PyPDFLoader(RESUME_PATH)
            docs = loader.load()
            logger.info({"message": f"📄 Loaded {len(docs)} pages from {RESUME_PATH}"})
            return docs
        except Exception as e:
            logger.error({"message": f"❌ Error loading PDF: {str(e)}"})
            raise

    @traceable(run_type="tool", name="Text_Splitter")
    def split_documents(self, docs):
        """Split documents into chunks."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=300, add_start_index=True
            )
            splits = text_splitter.split_documents(docs)
            logger.info({"message": f"📑 Created {len(splits)} chunks"})
            return splits
        except Exception as e:
            logger.error({"message": f"❌ Error splitting documents: {str(e)}"})
            raise

    @traceable(run_type="chain", name="RAG_Pipeline")
    def setup_and_query(self, question: str):
        """Set up the RAG pipeline and process a query in a single traceable run."""
        try:
            # Load and split documents
            docs = self.load_pdf()
            splits = self.split_documents(docs)

            # Initialize in-memory FAISS vector store
            if self.embeddings is None:
                self.embeddings = self.configure_embedding_model()

            self.vector_db = FAISS.from_documents(splits, self.embeddings)
            logger.info({"message": "🔍 FAISS vector store initialized in-memory"})

            # Initialize retriever
            retriever = self.vector_db.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4}
            )
            logger.info({"message": "🔍 FAISS retriever initialized"})

            # Set up conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True,
                memory_limit=20
            )

            # Define prompt template
            prompt_template = """
                You are Muhammad Umer Khan — a professional, polite, and passionate AI Engineer 🤖 dedicated to clear, accurate, and helpful communication.
                ✅ Only answer what is **explicitly asked** in the question — avoid extra or unrelated details.
                📝 Paraphrase from the context in your own words, keeping answers short and easy to read (1–3 sentences).
                🔹 Use bullet points only when listing multiple items, skills, experiences, or contact details, make use of emoji in response.
                🚫 If the answer is **not found** in the provided context, reply politely with:
                "I'm sorry, that information isn't available in my current context. 😊 Feel free to ask about my skills, projects, or how to contact me."
                💬 Always keep the tone clear, friendly, and professional, with light use of emojis for a human touch.
                            
                📬 If the question is about contacting you, respond with:
                    "You can reach me at:
                    - Phone: +923432187868 📞
                    - Email: muhammadumerk546@gmail.com 📧
                    - LinkedIn: https://www.linkedin.com/in/muhammad-umer-khan-61729b260/ 🔗"
                            
                Context: {context}
                Question: {question}
                Answer:
            """
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

            # Initialize RAG chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=False,
                verbose=False
            )
            logger.info({"message": "🚀 QA chain initialized"})

            # Process query
            result = self.qa_chain.invoke({"question": question})
            response = result["answer"].strip()
            logger.info({"message": f"💬 Query: {question} | Answer: {response}"})
            return response
        except Exception as e:
            logger.error({"message": f"❌ Error in RAG pipeline: {str(e)}"})
            raise

    async def query(self, question: str) -> str:
        """Process a user query through the RAG chain with caching."""
        try:
            if question in self.query_cache:
                response = self.query_cache[question]
                logger.info({"message": f"💾 Cache hit for query: {question}"})
                return response

            response = self.setup_and_query(question)
            self.query_cache[question] = response
            return response
        except Exception as e:
            logger.error({"message": f"❌ Query error: {str(e)}"})
            raise

    async def shutdown(self):
        """Clean up resources on shutdown."""
        await self.http_client.aclose()
        logger.info({"message": "🛑 HTTP client closed"})