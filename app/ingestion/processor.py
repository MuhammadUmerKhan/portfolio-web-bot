import re
from datetime import datetime
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

# Real employer / educational-institute relationships only — i.e. places
# Muhammad has actually worked or studied, not tools or APIs his projects use.
# Keeping this separate from PLATFORM_KEYWORDS is what stops the graph from
# implying "worked at OpenAI" just because a README calls OpenAI's API.
EMPLOYER_KEYWORDS = [
    "SMIT", "Saylani", "SaylaniTech", "Revera Innovations", "Bright Solutions",
]

# Third-party platforms, model providers, and hosting/tooling vendors
# mentioned in project docs — these describe what a project *uses*, not who
# Muhammad works for.
PLATFORM_KEYWORDS = [
    "GitHub", "Vercel", "Render", "Google", "OpenAI", "Anthropic", "NVIDIA",
]

# Plausible bounds for a project/career year mention — filters out obvious
# false positives from the bare `20\d{2}` regex (copyright years, license
# years, version strings) that don't reflect an actual project timeline.
MIN_PLAUSIBLE_YEAR = 2018

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

    # 2. Employer/institute extraction — only genuine work/education relationships,
    # not tools or APIs a project happens to use.
    matched_employers = []
    for employer in EMPLOYER_KEYWORDS:
        pattern = r'\b' + re.escape(employer.lower()) + r'(?!\w)'
        if re.search(pattern, content_lower):
            matched_employers.append(employer)
    if matched_employers:
        metadata["companies"] = matched_employers

    # 3. Platform/vendor extraction — tracked separately from employers so the
    # graph never conflates "mentions OpenAI's API" with "worked at OpenAI".
    matched_platforms = []
    for platform in PLATFORM_KEYWORDS:
        pattern = r'\b' + re.escape(platform.lower()) + r'(?!\w)'
        if re.search(pattern, content_lower):
            matched_platforms.append(platform)
    if matched_platforms:
        metadata["platforms"] = matched_platforms

    # 4. Year extraction, bounded to a plausible career window (2018 → current
    # year) — an unbounded \b(20\d{2})\b match picks up copyright years,
    # license years, and version strings that aren't real project dates.
    current_year = datetime.now().year
    years_found = re.findall(r'\b(20\d{2})\b', content)
    plausible_years = sorted(
        {y for y in years_found if MIN_PLAUSIBLE_YEAR <= int(y) <= current_year}
    )
    if plausible_years:
        metadata["years"] = plausible_years

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
