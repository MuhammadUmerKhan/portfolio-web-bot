import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core import setup_logging, get_settings, get_embeddings_model
import logfire
from app.ingestion.loader import PDFLoader, MarkdownLoader
from app.ingestion.processor import IngestionProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Set up observability logs (configures Logfire)
setup_logging()

def main():
    settings = get_settings()
    
    # Paths to parse
    resume_path = settings.app.resume_path
    github_raw_dir = Path("data/raw/github")
    
    logfire.info("Starting document ingestion pipeline...")
    
    raw_docs = []
    
    # 1. Parse PDF Resume
    if resume_path.exists():
        with logfire.span("Parse PDF Resume"):
            logfire.info("Loading resume PDF from: {path}", path=resume_path)
            pdf_loader = PDFLoader(resume_path)
            pdf_docs = pdf_loader.load()
            raw_docs.extend(pdf_docs)
            logfire.info("Successfully parsed resume page count: {count}", count=len(pdf_docs))
    else:
        logfire.warning("Resume PDF not found at: {path}. Skipping.", path=resume_path)
        
    # 2. Parse GitHub repositories markdown corpus
    if github_raw_dir.exists():
        with logfire.span("Parse GitHub Repositories"):
            logfire.info("Loading GitHub repositories raw markdown from: {path}", path=github_raw_dir)
            md_loader = MarkdownLoader(github_raw_dir)
            md_docs = md_loader.load()
            raw_docs.extend(md_docs)
            logfire.info("Successfully parsed GitHub markdown files count: {count}", count=len(md_docs))
    else:
        logfire.warning("GitHub raw directory not found: {path}. Skipping.", path=github_raw_dir)
        
    if not raw_docs:
        logfire.error("No documents loaded. Ingestion pipeline aborted.")
        sys.exit(1)
        
    # 3. Coordinate chunking, ID generation, and metadata extraction
    processor = IngestionProcessor(chunk_size=800, chunk_overlap=300)
    all_chunks = []
    
    with logfire.span("Chunking and Metadata Extraction"):
        for doc in raw_docs:
            chunks = processor.process_document(doc)
            all_chunks.extend(chunks)
            
        logfire.info("Chunking and metadata extraction completed.")
        logfire.info("Total input documents: {raw} | Total chunks created: {chunks}", raw=len(raw_docs), chunks=len(all_chunks))
        
    if not all_chunks:
        logfire.warning("No chunks generated. Pipeline aborted.")
        return
        
    # 4. Initialize Embeddings Model (with local SentenceTransformer fallback)
    embeddings_model = get_embeddings_model()
    
    # 5. Embed chunk contents in batches
    with logfire.span("Generate Vector Embeddings"):
        logfire.info("Generating vector embeddings for {count} chunks...", count=len(all_chunks))
        batch_size = 100
        embeddings_list = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            
            logfire.info(
                "Embedding batch {batch}/{total} ({size} texts)...", 
                batch=(i // batch_size) + 1, 
                total=(len(all_chunks) - 1) // batch_size + 1, 
                size=len(batch_texts)
            )
            
            batch_embeddings = embeddings_model.embed_documents(batch_texts)
            embeddings_list.extend(batch_embeddings)
            
            # Add a tiny sleep to respect API limits if batch count is large
            if i + batch_size < len(all_chunks):
                time.sleep(0.5)
                
        logfire.info("Successfully generated {count} embeddings.", count=len(embeddings_list))
        
    # 6. Initialize Qdrant Client
    logfire.info("Connecting to Qdrant Cloud: {endpoint}", endpoint=settings.qdrant.endpoint)
    qdrant_client = QdrantClient(
        url=settings.qdrant.endpoint,
        api_key=settings.qdrant.api_key.get_secret_value(),
        timeout=30.0
    )
    
    collection_name = settings.qdrant.collection_name
    
    # Create Qdrant collection if it doesn't exist
    if not qdrant_client.collection_exists(collection_name):
        logfire.info("Collection '{collection}' does not exist. Creating collection...", collection=collection_name)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,  # text-embedding-004 defaults to 768 dimensions
                distance=Distance.COSINE
            )
        )
        logfire.info("Collection '{collection}' created successfully.", collection=collection_name)
    else:
        logfire.info("Collection '{collection}' already exists.", collection=collection_name)
        
    # Prepare points structure for upsert
    points = []
    for chunk, vector in zip(all_chunks, embeddings_list):
        payload = chunk.model_dump()
        point_id = payload.pop("id")
        
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )
        
    # 7. Batch upsert points to Qdrant
    with logfire.span("Upsert to Qdrant Cloud"):
        logfire.info("Upserting {count} points to Qdrant Cloud...", count=len(points))
        qdrant_batch_size = 200
        for i in range(0, len(points), qdrant_batch_size):
            batch_points = points[i:i + qdrant_batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch_points
            )
            logfire.info(
                "Upserted points {start} to {end}...", 
                start=i + 1, 
                end=min(i + qdrant_batch_size, len(points))
            )
            
        logfire.info("Qdrant Cloud ingestion sync complete.")
    
    # 8. Serialize local cache (data/processed/chunks.json) for BM25 retrieval
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.json"
    
    serialized_chunks = [chunk.model_dump() for chunk in all_chunks]
    output_file.write_text(json.dumps(serialized_chunks, indent=2), encoding="utf-8")
    
    logfire.info("Saved all {count} chunks to processed storage: {file}", count=len(all_chunks), file=output_file)

if __name__ == "__main__":
    main()
