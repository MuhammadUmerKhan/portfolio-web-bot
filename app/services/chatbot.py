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
from app.core import get_settings, get_embeddings_model
from app.services.retrieval import QdrantRetrievalService, RankingService
from app.services.graph_service import GraphService
from app.ingestion.graph_builder import build_graph
from app.agents.graph import create_agent_graph
import logfire

# Ignore warnings
warnings.filterwarnings("ignore")

settings = get_settings()

class CustomHybridRetriever(BaseRetriever):
    """
    Custom LangChain retriever that combines dense Qdrant search,
    local BM25 keyword search using Reciprocal Rank Fusion (RRF),
    and local FlashRank reranking.
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
        
        # 4. Rerank candidate list using local FlashRank model (top 5 final candidates)
        final_docs = self.reranker.rerank(query, fused_docs, top_n=5)
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
        
        # Fetch all chunks once from Qdrant Cloud to rebuild all services dynamically
        chunks_data = self.dense_service.fetch_all_chunks()
        
        # 2. Precompute/initialize the BM25 retriever using fetched chunks
        self.bm25_retriever = self._init_bm25_retriever(chunks_data)
        
        # 3. Initialize FlashRank reranker service
        self.ranking_service = RankingService()
        
        # 4. Compile the Knowledge Graph adjacency dictionary and load GraphService
        graph_dict = build_graph(chunks_data)
        self.graph_service = GraphService(graph_dict=graph_dict)
        
        # 5. Initialize our custom hybrid retriever with FlashRank reranking (no forced graph injection)
        self.retriever = CustomHybridRetriever(
            dense_service=self.dense_service,
            sparse_retriever=self.bm25_retriever,
            reranker=self.ranking_service
        )
        logfire.info("✅ Custom hybrid retriever (RRF + FlashRank) eagerly initialized")

        # 6. Initialize stateful LangGraph agent workflow
        self.agent = create_agent_graph(self)
        self.qa_chain = self.agent  # Alias for health check endpoint compatibility
        logfire.info("🚀 LangGraph agent workflow compiled. Chatbot is fully operational.")

    def configure_llm(self):
        """Configure the Groq LLM with specified model and API key."""
        try:
            llm = ChatGroq(
                model_name=settings.app.model_name,
                temperature=0.5,
                groq_api_key=settings.groq.api_key.get_secret_value()
            )
            logfire.info("✅ Groq LLM configured successfully")
            return llm
        except Exception as e:
            logfire.error("❌ Failed to configure LLM: {error}", error=str(e))
            raise 
    
    @traceable(run_type="tool", name="Embeddings_Initializer")
    def configure_embedding_model(self):
        """Configure embedding model (with local fallback)."""
        try:
            embeddings = get_embeddings_model()
            return embeddings
        except Exception as e:
            logfire.error("❌ Failed to configure embeddings: {error}", error=str(e))
            raise

    def _init_bm25_retriever(self, chunks_data: list[dict]) -> BM25Retriever:
        """Initializes BM25Retriever from dynamically retrieved chunk data."""
        from langchain_core.documents import Document
        
        documents = []
        try:
            for item in chunks_data:
                meta = item.get("metadata", {})
                # Standardize fields inside metadata
                meta.setdefault("source_type", item.get("source_type"))
                meta.setdefault("source_path", item.get("source_path"))
                meta.setdefault("project_name", item.get("project_name"))
                
                doc = Document(
                    page_content=item.get("content", ""),
                    metadata=meta
                )
                documents.append(doc)
            logfire.info("📚 Initialized BM25 retriever with {count} document chunks", count=len(documents))
        except Exception as e:
            logfire.error("❌ Error initializing BM25 retriever: {error}", error=str(e))
            
        if not documents:
            logfire.warning("⚠️ No chunks available. BM25 initializing with fallback.")
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
            logfire.error("❌ Error loading PDF: {error}", error=str(e))
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
            logfire.error("❌ Error splitting documents: {error}", error=str(e))
            raise

    @traceable(run_type="chain", name="RAG_Pipeline")
    def setup_and_query(self, question: str, thread_id: str = "default") -> str:
        """Processes a query using the pre-initialized stateful LangGraph agent workflow."""
        try:
            from langchain_core.messages import HumanMessage
            
            config = {"configurable": {"thread_id": thread_id}}
            state_input = {
                "messages": [HumanMessage(content=question)],
                "retrieved_docs": [],
                "graph_context": ""
            }
            
            result_state = self.agent.invoke(state_input, config=config)
            
            # The final AI message response is the last message in the sequence
            response = result_state["messages"][-1].content.strip()
            logfire.info("💬 Query: {question} | Answer: {response}", question=question, response=response)
            return response
        except Exception as e:
            logfire.error("❌ Error in RAG pipeline: {error}", error=str(e))
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
                logfire.info("💾 Cache hit for query: {question} (normalized: {normalized_q})", question=question, normalized_q=normalized_q)
                return response

            response = self.setup_and_query(question)
            self.query_cache[normalized_q] = response
            return response
        except Exception as e:
            logfire.error("❌ Query error: {error}", error=str(e))
            raise

    async def shutdown(self):
        """Clean up resources on shutdown."""
        await self.http_client.aclose()
        logfire.info("🛑 HTTP client closed")
