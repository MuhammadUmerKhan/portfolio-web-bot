import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.loader.base import RawDocument
from app.ingestion.chunking.base import Chunk

class RecursiveChunker:
    """
    Standard recursive character-based chunker for unstructured text like resume PDFs.
    Generates deterministic SHA-256 chunk IDs.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 300):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def split(self, doc: RawDocument) -> list[Chunk]:
        # create_documents (not split_text) is required for add_start_index to actually
        # populate start_index metadata — split_text silently ignores that setting.
        sub_docs = self.text_splitter.create_documents([doc.content])
        chunks = []
        
        for sub_doc in sub_docs:
            sub_split = sub_doc.page_content
            start_index = sub_doc.metadata.get("start_index")

            # Deterministic UUID (not a raw hash) — Qdrant point IDs must be an
            # unsigned integer or a UUID, so a 64-char sha256 hexdigest would be
            # rejected. uuid5 keeps this fully idempotent: same input -> same ID.
            chunk_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.source_path}:{sub_split}")
            )
            
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=sub_split,
                    source_type=doc.source_type,
                    source_path=doc.source_path,
                    project_name=doc.project_name,
                    project_description=doc.project_description,
                    project_url=doc.project_url,
                    project_language=doc.project_language,
                    project_topics=doc.project_topics,
                    project_pushed_at=doc.project_pushed_at,
                    is_private=doc.is_private,
                    page_number=doc.page_number,
                    start_index=start_index,
                )
            )
            
        return chunks
