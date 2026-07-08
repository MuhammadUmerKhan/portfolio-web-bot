# %% [markdown]
# # RAG Chatbot & GraphRAG Pipeline Interactive Tester 🚀
# Use VS Code's interactive window (click "Run Cell" above each block) to test and debug 
# each component of Umer's portfolio chatbot dynamically.

# %% [markdown]
# ### Cell 1: Environment & Settings Initialization
# Loads environment variables, configures settings, and initializes logging.

# %%
import os
import sys
from pathlib import Path

# Resolve project root dynamically (searches upward for 'app' directory)
current_dir = Path(os.getcwd()).resolve()
project_root = current_dir
for parent in [current_dir] + list(current_dir.parents):
    if (parent / "app").exists():
        project_root = parent
        break

sys.path.append(str(project_root))

from app.core import get_settings, setup_logging, get_logger, get_embeddings_model
setup_logging()
logger = get_logger("interactive_tester")
settings = get_settings()

print("✅ Settings & Env loaded.")
print(f"Project root added to path: {project_root}")
print(f"Active collection: {settings.qdrant.collection_name}")
print(f"Active LLM: {settings.app.model_name}")

# %% [markdown]
# ### Cell 2: Test Qdrant Scroll Loader (Phase 4b)
# Fetches all points from Qdrant Cloud. This simulates what happens at server startup.

# %%
from app.services.retrieval.qdrant_service import QdrantRetrievalService

print("Connecting to Qdrant Cloud and scrolling collection...")
qdrant_service = QdrantRetrievalService()
chunks_data = qdrant_service.fetch_all_chunks()

print(f"\n✅ Scroll complete. Fetched {len(chunks_data)} total chunks from Qdrant Cloud.")
if chunks_data:
    print("\nSample Chunk Payload:")
    import pprint
    pprint.pprint(chunks_data[0])

# %% [markdown]
# ### Cell 3: Test Dynamic BM25 In-Memory Indexing
# Tests how the BM25 index is compiled from the fetched Qdrant payloads in RAM.

# %%
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

print("Compiling BM25 index in RAM...")
documents = []
for item in chunks_data:
    meta = item.get("metadata", {})
    meta.setdefault("source_type", item.get("source_type"))
    meta.setdefault("source_path", item.get("source_path"))
    meta.setdefault("project_name", item.get("project_name"))
    
    doc = Document(
        page_content=item.get("content", ""),
        metadata=meta
    )
    documents.append(doc)

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# Test query
query = "DineMate"
results = bm25_retriever.invoke(query)

print(f"✅ BM25 indexed {len(documents)} documents successfully.")
print(f"\nBM25 Search Results for '{query}':")
for idx, r in enumerate(results):
    print(f"\nMatch {idx+1} [Source: {r.metadata.get('source_path')}]:")
    print(r.page_content[:200] + "...")

# %% [markdown]
# ### Cell 4: Test In-Memory Knowledge Graph Compilation & Search (Phase 4a)
# Tests graph builder compiling the adjacency list in RAM and GraphService resolving queries.

# %%
from app.ingestion.graph_builder import build_graph
from app.services.graph_service import GraphService

print("Compiling lightweight Knowledge Graph adjacency list...")
graph_dict = build_graph(chunks_data)
graph_service = GraphService(graph_dict=graph_dict)

# Test query
graph_query = "FastAPI"
graph_context = graph_service.query_graph(graph_query)

print(f"\n✅ Knowledge graph built with {len(graph_dict)} unique nodes.")
print(f"\nGraph Search Context lookup for '{graph_query}':")
print(graph_context if graph_context else "No entity matched in graph.")

# %% [markdown]
# ### Cell 5: Test FlashRank Reranker (Phase 3)
# Tests combining Dense + Sparse outputs and reranking them on CPU.

# %%
from app.services.retrieval.ranking_service import RankingService

print("Initializing local FlashRank CPU cross-encoder...")
ranking_service = RankingService()

# 1. Fetch dense matches from Qdrant
dense_docs = qdrant_service.retrieve("FastAPI projects", limit=5)
# 2. Fetch sparse matches from BM25
sparse_docs = bm25_retriever.invoke("FastAPI projects")

# 3. Combine
combined = dense_docs + sparse_docs

# 4. Rerank
reranked = ranking_service.rerank("FastAPI projects", combined, top_n=2)

print("\n✅ FlashRank Rerank complete.")
print("\nTop 2 Reranked Matches:")
for idx, r in enumerate(reranked):
    score = r.metadata.get('rerank_score') or 0.0
    print(f"\nRank {idx+1} (Score: {score:.4f}):")
    print(r.page_content[:200] + "...")

# %% [markdown]
# ### Cell 6: Test Stateful LangGraph Agent E2E (Phase 5)
# Starts the compiled chatbot service and executes a conversation thread to verify state.

# %%
from app.services.chatbot import CustomDocChatbot

print("Eagerly booting CustomDocChatbot (loads Qdrant, builds BM25 & Graph)...")
chatbot = CustomDocChatbot()

# %%
# Ask greeting (routes to direct chat)
q1 = "Hello, what is your name?"
print(f"Query: {q1}")
ans1 = chatbot.setup_and_query(q1, thread_id="test_session")
print(f"Answer: {ans1}\n")

# Ask semantic project query (routes to tools retrieval)
q2 = "What did Umer build for Society Management?"
print(f"Query: {q2}")
ans2 = chatbot.setup_and_query(q2, thread_id="test_session")
print(f"Answer: {ans2}\n")

# Follow up to test conversation memory
q3 = "What technologies were used in that project?"
print(f"Query: {q3}")
ans3 = chatbot.setup_and_query(q3, thread_id="test_session")
print(f"Answer: {ans3}\n")
