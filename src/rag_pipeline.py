import httpx
import os
import json
import cachetools
import warnings
from typing import List
from pathlib import Path
from langsmith import traceable
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from app.core import get_settings, get_logger, get_embeddings_model

# Ignore warnings
warnings.filterwarnings("ignore")

settings = get_settings()
logger = get_logger(__name__)

class CustomDocChatbot:
    """A RAG-based chatbot for answering questions using a remote Qdrant store and local BM25."""
    
    query_cache = cachetools.TTLCache(maxsize=500, ttl=600)

    def __init__(self):
        """Initialize the chatbot eagerly, setting up connection to LLM, embeddings, Qdrant, and BM25."""
        self.llm = self.configure_llm()
        self.embeddings = self.configure_embedding_model()
        self.http_client = httpx.AsyncClient(timeout=15.0)
        
        # 1. Eagerly connect to remote Qdrant store
        from qdrant_client import QdrantClient
        from langchain_community.vectorstores import Qdrant
        
        logger.info({"message": f"🔍 Connecting to Qdrant Cloud: {settings.qdrant.endpoint}"})
        client = QdrantClient(
            url=settings.qdrant.endpoint,
            api_key=settings.qdrant.api_key.get_secret_value(),
            timeout=30.0
        )
        self.vector_db = Qdrant(
            client=client,
            collection_name="personal_kb",
            embeddings=self.embeddings
        )
        logger.info({"message": "✅ Eagerly connected to Qdrant Cloud vector store"})

        # 2. Precompute/initialize the local BM25 retriever
        self.bm25_retriever = self._init_bm25_retriever()
        
        # 3. Configure retrieval strategies
        qdrant_retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, qdrant_retriever], 
            weights=[0.5, 0.5]
        )
        logger.info({"message": "✅ Ensemble (Hybrid) retriever initialized"})

        # 4. Set up conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            memory_limit=20
        )

        # 5. Define prompt template
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

        # 6. Initialize RAG chain eagerly
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False,
            verbose=False
        )
        logger.info({"message": "🚀 Eager QA chain setup complete. Chatbot is fully operational."})

    def configure_llm(self):
        """Configure the Groq LLM with specified model and API key."""
        try:
            llm = ChatGroq(
                model_name=settings.app.model_name,
                temperature=0.5,
                groq_api_key=settings.groq.api_key.get_secret_value()
            )
            logger.info({"message": "✅ Groq LLM configured successfully"})
            return llm
        except Exception as e:
            logger.error({"message": f"❌ Failed to configure LLM: {str(e)}"})
            raise 
    
    @traceable(run_type="tool", name="Embeddings_Initializer")
    def configure_embedding_model(self):
        """Configure embedding model (with local fallback)."""
        try:
            embeddings = get_embeddings_model()
            return embeddings
        except Exception as e:
            logger.error({"message": f"❌ Failed to configure embeddings: {str(e)}"})
            raise

    def _init_bm25_retriever(self) -> BM25Retriever:
        """Loads serialized chunks from local cache and initializes BM25Retriever."""
        from langchain_core.documents import Document
        
        chunks_path = Path("data/processed/chunks.json")
        documents = []
        
        if chunks_path.exists():
            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                
                for item in chunks_data:
                    doc = Document(
                        page_content=item["content"],
                        metadata={
                            "source_type": item["source_type"],
                            "source_path": item["source_path"],
                            "project_name": item.get("project_name"),
                            "project_language": item.get("project_language"),
                            "project_topics": item.get("project_topics", []),
                            "relative_path": item.get("relative_path"),
                            "page_number": item.get("page_number"),
                            "headers": item.get("headers", {}),
                            "metadata": item.get("metadata", {})
                        }
                    )
                    documents.append(doc)
                logger.info({"message": f"📚 Loaded {len(documents)} cache chunks for BM25 retriever"})
            except Exception as e:
                logger.error({"message": f"❌ Error loading local chunk cache: {str(e)}"})
                
        if not documents:
            logger.warning({"message": "⚠️ Chunks cache file empty or missing. BM25 initializing with fallback."})
            documents = [Document(page_content="Muhammad Umer Khan is an AI Engineer and developer.")]
            
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = 3
        return retriever

    @lru_cache(maxsize=1)
    @traceable(run_type="tool", name="PDF_Loader")
    def load_pdf(self):
        """Deprecated legacy loader helper, kept for compatibility."""
        try:
            if not settings.app.resume_path.exists():
                raise FileNotFoundError(f"PDF not found at {settings.app.resume_path}")
            loader = PyPDFLoader(str(settings.app.resume_path))
            docs = loader.load()
            return docs
        except Exception as e:
            logger.error({"message": f"❌ Error loading PDF: {str(e)}"})
            raise

    @traceable(run_type="tool", name="Text_Splitter")
    def split_documents(self, docs):
        """Deprecated legacy splitter helper, kept for compatibility."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=300, add_start_index=True
            )
            splits = text_splitter.split_documents(docs)
            return splits
        except Exception as e:
            logger.error({"message": f"❌ Error splitting documents: {str(e)}"})
            raise

    @traceable(run_type="chain", name="RAG_Pipeline")
    def setup_and_query(self, question: str):
        """Processes a query using the pre-initialized eager hybrid QA chain."""
        try:
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