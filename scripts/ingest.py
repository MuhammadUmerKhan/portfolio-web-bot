import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core import setup_logging, get_logger, get_settings
from app.ingestion.loader import PDFLoader, MarkdownLoader
from app.ingestion.processor import IngestionProcessor

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
        return
        
    # 3. Coordinate chunking, ID generation, and metadata extraction
    processor = IngestionProcessor(chunk_size=800, chunk_overlap=300)
    all_chunks = []
    
    for doc in raw_docs:
        chunks = processor.process_document(doc)
        all_chunks.extend(chunks)
        
    logger.info("Chunking and metadata extraction completed.")
    logger.info("Total input documents: %d | Total chunks created: %d", len(raw_docs), len(all_chunks))
    
    if not all_chunks:
        logger.warning("No chunks generated. Serialization aborted.")
        return
        
    # Log sample chunk detail for debugging / trace validation
    sample = all_chunks[0]
    logger.info(
        "Sample Chunk details:\n"
        "- Chunk ID: %s\n"
        "- Source File: %s\n"
        "- Source Type: %s\n"
        "- Headers: %s\n"
        "- Extracted Metadata: %s\n"
        "- Content Snippet: %s...",
        sample.id, sample.source_path, sample.source_type, sample.headers, sample.metadata, sample.content[:150]
    )
    
    # 4. Serialize to disk (data/processed/chunks.json)
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.json"
    
    serialized_chunks = [chunk.model_dump() for chunk in all_chunks]
    output_file.write_text(json.dumps(serialized_chunks, indent=2), encoding="utf-8")
    
    logger.info("Saved all %d chunks to processed storage: %s", len(all_chunks), output_file)

if __name__ == "__main__":
    main()
