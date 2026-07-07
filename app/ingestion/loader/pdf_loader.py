from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from app.ingestion.loader.base import RawDocument, SourceType

class PDFLoader:
    """
    Parses PDF documents using LangChain's PyPDFLoader and returns a list of page-by-page RawDocuments.
    """
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load(self) -> list[RawDocument]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF document not found at: {self.file_path}")
            
        loader = PyPDFLoader(str(self.file_path))
        pages = loader.load()
        documents = []
        
        for page in pages:
            # LangChain's page number metadata is 0-indexed
            page_num = page.metadata.get("page", 0) + 1
            documents.append(
                RawDocument(
                    content=page.page_content.strip(),
                    source_type=SourceType.RESUME_PDF,
                    source_path=str(self.file_path.resolve()),
                    page_number=page_num,
                )
            )
            
        return documents
