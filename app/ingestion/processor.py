import re
from typing import Dict, Any
from app.ingestion.loader import RawDocument, SourceType
from app.ingestion.chunking import Chunk, MarkdownChunker, RecursiveChunker

# Comprehensive technology keywords for rule-based metadata extraction
TECH_KEYWORDS = [
    # Languages
    "Python", "JavaScript", "TypeScript", "HTML", "CSS", "SQL", "Bash", "C++", "Java", "Go", "Rust",
    # Frameworks & Libraries
    "FastAPI", "React", "Node.js", "Next.js", "Streamlit", "LangChain", "LangGraph", 
    "PyTorch", "TensorFlow", "scikit-learn", "Hugging Face", "Uvicorn", "Flask", "Django",
    # Databases & Indexing
    "Qdrant", "FAISS", "PostgreSQL", "SQLite", "MySQL", "Neo4j", "Redis", "MongoDB",
    # Tools, Platforms & Methods
    "RAG", "Agentic AI", "NLP", "Computer Vision", "Docker", "Git", "CI/CD", "GitHub Actions",
    "Render", "Vercel", "AWS", "GCP", "LLM", "LLMs", "Llama", "Gemini", "Claude", 
    "OpenRouter", "Portkey", "Logfire", "LangSmith", "Langfuse", "RAGAS", "DeepEval", 
    "NeMo Guardrails", "Guardrails", "Transformers", "Vapi", "Retell"
]

# Company or educational organizations mentioned in resume/github
COMPANY_KEYWORDS = [
    "SMIT", "Saylani", "Vercel", "Render", "GitHub", "Google", "OpenAI", "Anthropic", "NVIDIA"
]

def extract_lightweight_metadata(content: str) -> Dict[str, Any]:
    """
    Scans document content using case-insensitive keyword searches
    to extract technical stacks, dates/years, and mentioned organizations.
    """
    metadata = {}
    content_lower = content.lower()
    
    # 1. Tech stack extraction
    matched_tech = []
    for tech in TECH_KEYWORDS:
        # (?!\w) instead of a trailing \b: \b never matches right after a
        # non-word character (e.g. the second '+' in "C++"), so keywords ending
        # in symbols would otherwise silently never be detected.
        pattern = r'\b' + re.escape(tech.lower()) + r'(?!\w)'
        if re.search(pattern, content_lower):
            matched_tech.append(tech)
    if matched_tech:
        metadata["tech_stack"] = matched_tech

    # 2. Company/organization extraction
    matched_companies = []
    for company in COMPANY_KEYWORDS:
        pattern = r'\b' + re.escape(company.lower()) + r'(?!\w)'
        if re.search(pattern, content_lower):
            matched_companies.append(company)
    if matched_companies:
        metadata["companies"] = matched_companies

    # 3. Year extraction (20xx matches)
    years = re.findall(r'\b(20\d{2})\b', content)
    if years:
        metadata["years"] = sorted(list(set(years)))

    return metadata

class IngestionProcessor:
    """
    Coordinates loaders and chunkers to process raw document inputs
    and enriches split chunks with parsed metadata (entities).
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 300):
        self.md_chunker = MarkdownChunker(chunk_size, chunk_overlap)
        self.pdf_chunker = RecursiveChunker(chunk_size, chunk_overlap)

    def process_document(self, doc: RawDocument) -> list[Chunk]:
        # Choose the right chunker structural parsing format
        if doc.source_type == SourceType.GITHUB_README:
            chunks = self.md_chunker.split(doc)
        else:
            chunks = self.pdf_chunker.split(doc)
            
        # Enrich each chunk with parsed lightweight entity metadata. update()
        # rather than assignment, so anything the chunker already put in
        # chunk.metadata (e.g. future fields) isn't silently wiped out.
        for chunk in chunks:
            chunk.metadata.update(extract_lightweight_metadata(chunk.content))
            
        return chunks
