import httpx
import os
import re
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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import ConfigDict
from app.core import get_settings, get_logger, get_embeddings_model
from app.services.retrieval import QdrantRetrievalService, RankingService

# Ignore warnings
warnings.filterwarnings("ignore")

settings = get_settings()
logger = get_logger(__name__)

class CustomHybridRetriever(BaseRetriever):
    """
    Custom LangChain retriever that combines dense Qdrant search
    and local BM25 keyword search using Reciprocal Rank Fusion (RRF)
    and applies local FlashRank reranking.
    """
    dense_service: QdrantRetrievalService
    sparse_retriever: BM25Retriever
    reranker: RankingService
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1. Fetch dense candidates from Qdrant Cloud (top 10)
        dense_docs = self.dense_service.retrieve(query, limit=10)
        
        # 2. Fetch sparse candidates from BM25 local index (top 10)
        self.sparse_retriever.k = 10
        sparse_docs = self.sparse_retriever.invoke(query)
        
        # 3. Fuse dense and sparse candidates using RRF
        fused_docs = self._rrf(dense_docs, sparse_docs, k=60)
        
        # 4. Rerank candidate list using local FlashRank model (top 4 final candidates)
        final_docs = self.reranker.rerank(query, fused_docs, top_n=4)
        return final_docs

    def _rrf(self, dense_docs: List[Document], sparse_docs: List[Document], k: int = 60) -> List[Document]:
        """Performs Reciprocal Rank Fusion (RRF) on dense and sparse retrievals."""
        rrf_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(dense_docs):
            content = doc.page_content
            doc_map[content] = doc
            rrf_scores[content] = rrf_scores.get(content, 0.0) + (1.0 / (rank + k))
            
        for rank, doc in enumerate(sparse_docs):
            content = doc.page_content
            doc_map[content] = doc
            rrf_scores[content] = rrf_scores.get(content, 0.0) + (1.0 / (rank + k))
            
        sorted_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused_documents = []
        for content in sorted_contents:
            doc = doc_map[content]
            # Copy to prevent original cache reference mutation
            new_doc = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
            new_doc.metadata["rrf_score"] = rrf_scores[content]
            fused_documents.append(new_doc)
            
        return fused_documents

class CustomDocChatbot:
    """A RAG-based chatbot for answering questions using a remote Qdrant store and local BM25."""
    
    query_cache = cachetools.TTLCache(maxsize=500, ttl=600)

    def __init__(self):
        """Initialize the chatbot eagerly, setting up LLM, Custom RRF Retriever, and memory."""
        self.llm = self.configure_llm()
        self.embeddings = self.configure_embedding_model()
        self.http_client = httpx.AsyncClient(timeout=15.0)
        
        # 1. Initialize direct Qdrant retrieval service
        self.dense_service = QdrantRetrievalService()
        self.vector_db = self.dense_service.client  # Checked in /health endpoint
        
        # 2. Precompute/initialize the local BM25 retriever
        self.bm25_retriever = self._init_bm25_retriever()
        
        # 3. Initialize FlashRank reranker service
        self.ranking_service = RankingService()
        
        # 4. Initialize our custom hybrid retriever with FlashRank reranking
        retriever = CustomHybridRetriever(
            dense_service=self.dense_service,
            sparse_retriever=self.bm25_retriever,
            reranker=self.ranking_service
        )
        logger.info({"message": "✅ Custom hybrid retriever (RRF + FlashRank) eagerly initialized"})

        # 5. Set up conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            memory_limit=20
        )

        # 6. Define prompt template
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

        # 7. Initialize RAG chain eagerly
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

    def _normalize_query(self, query: str) -> str:
        """Normalize query string for query cache matching."""
        # Lowercase and strip whitespace
        q = query.lower().strip()
        # Clean extra internal whitespace
        q = re.sub(r'\s+', ' ', q)
        # Strip trailing punctuation (like ?, ., !)
        q = re.sub(r'[?.!]+$', '', q)
        return q

    async def query(self, question: str) -> str:
        """Process a user query through the RAG chain with caching."""
        try:
            normalized_q = self._normalize_query(question)
            if normalized_q in self.query_cache:
                response = self.query_cache[normalized_q]
                logger.info({"message": f"💾 Cache hit for query: {question} (normalized: {normalized_q})"})
                return response

            response = self.setup_and_query(question)
            self.query_cache[normalized_q] = response
            return response
        except Exception as e:
            logger.error({"message": f"❌ Query error: {str(e)}"})
            raise

    async def shutdown(self):
        """Clean up resources on shutdown."""
        await self.http_client.aclose()
        logger.info({"message": "🛑 HTTP client closed"})