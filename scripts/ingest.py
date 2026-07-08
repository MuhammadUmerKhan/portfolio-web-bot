import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core import setup_logging, get_logger, get_settings, get_embeddings_model
from app.ingestion.loader import PDFLoader, MarkdownLoader
from app.ingestion.processor import IngestionProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Set up observability logs
setup_logging()
logger = get_logger("ingestion_pipeline")

def main():
    settings = get_settings()
    
    # Paths to parse
    resume_path = settings.app.resume_path
    github_raw_dir = Path("data/raw/github")
    
    logger.info("Starting document ingestion pipeline...")
    
    raw_docs = []
    
    # 1. Parse PDF Resume
    if resume_path.exists():
        logger.info("Loading resume PDF from: %s", resume_path)
        pdf_loader = PDFLoader(resume_path)
        pdf_docs = pdf_loader.load()
        raw_docs.extend(pdf_docs)
        logger.info("Successfully parsed resume page count: %d", len(pdf_docs))
    else:
        logger.warning("Resume PDF not found at: %s. Skipping.", resume_path)
        
    # 2. Parse GitHub repositories markdown corpus
    if github_raw_dir.exists():
        logger.info("Loading GitHub repositories raw markdown from: %s", github_raw_dir)
        md_loader = MarkdownLoader(github_raw_dir)
        md_docs = md_loader.load()
        raw_docs.extend(md_docs)
        logger.info("Successfully parsed GitHub markdown files count: %d", len(md_docs))
    else:
        logger.warning("GitHub raw directory not found: %s. Skipping.", github_raw_dir)
        
    if not raw_docs:
        logger.error("No documents loaded. Ingestion pipeline aborted.")
        sys.exit(1)
        
    # 3. Coordinate chunking, ID generation, and metadata extraction
    processor = IngestionProcessor(chunk_size=800, chunk_overlap=300)
    all_chunks = []
    
    for doc in raw_docs:
        chunks = processor.process_document(doc)
        all_chunks.extend(chunks)
        
    logger.info("Chunking and metadata extraction completed.")
    logger.info("Total input documents: %d | Total chunks created: %d", len(raw_docs), len(all_chunks))
    
    if not all_chunks:
        logger.warning("No chunks generated. Pipeline aborted.")
        return
        
    # 4. Initialize Embeddings Model (with local SentenceTransformer fallback)
    embeddings_model = get_embeddings_model()
    
    # 5. Embed chunk contents in batches
    logger.info("Generating vector embeddings for %d chunks...", len(all_chunks))
    batch_size = 100
    embeddings_list = []
    
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_texts = [chunk.content for chunk in batch_chunks]
        
        logger.info(
            "Embedding batch %d/%d (%d texts)...", 
            (i // batch_size) + 1, 
            (len(all_chunks) - 1) // batch_size + 1, 
            len(batch_texts)
        )
        
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        embeddings_list.extend(batch_embeddings)
        
        # Add a tiny sleep to respect API limits if batch count is large
        if i + batch_size < len(all_chunks):
            time.sleep(0.5)
            
    logger.info("Successfully generated %d embeddings.", len(embeddings_list))
    
    # 6. Initialize Qdrant Client
    logger.info("Connecting to Qdrant Cloud: %s", settings.qdrant.endpoint)
    qdrant_client = QdrantClient(
        url=settings.qdrant.endpoint,
        api_key=settings.qdrant.api_key.get_secret_value(),
        timeout=30.0
    )
    
    collection_name = settings.qdrant.collection_name
    
    # Create Qdrant collection if it doesn't exist
    if not qdrant_client.collection_exists(collection_name):
        logger.info("Collection '%s' does not exist. Creating collection...", collection_name)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,  # text-embedding-004 defaults to 768 dimensions
                distance=Distance.COSINE
            )
        )
        logger.info("Collection '%s' created successfully.", collection_name)
    else:
        logger.info("Collection '%s' already exists.", collection_name)
        
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
    logger.info("Upserting %d points to Qdrant Cloud...", len(points))
    qdrant_batch_size = 200
    for i in range(0, len(points), qdrant_batch_size):
        batch_points = points[i:i + qdrant_batch_size]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
        logger.info(
            "Upserted points %d to %d...", 
            i + 1, 
            min(i + qdrant_batch_size, len(points))
        )
        
    logger.info("Qdrant Cloud ingestion sync complete.")
    
    # 8. Serialize local cache (data/processed/chunks.json) for BM25 retrieval
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.json"
    
    serialized_chunks = [chunk.model_dump() for chunk in all_chunks]
    output_file.write_text(json.dumps(serialized_chunks, indent=2), encoding="utf-8")
    
    logger.info("Saved all %d chunks to processed storage: %s", len(all_chunks), output_file)

if __name__ == "__main__":
    main()
