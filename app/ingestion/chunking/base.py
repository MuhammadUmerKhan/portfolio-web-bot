from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from app.ingestion.loader.base import SourceType

class Chunk(BaseModel):
    id: str  # Deterministic UUID (uuid5), required for use as a Qdrant point ID
    content: str
    source_type: SourceType
    source_path: str
    project_name: Optional[str] = None
    project_description: Optional[str] = None
    project_url: Optional[str] = None
    project_language: Optional[str] = None
    project_topics: list[str] = Field(default_factory=list)
    project_pushed_at: Optional[str] = None
    is_private: bool = False
    relative_path: Optional[str] = None
    page_number: Optional[int] = None
    start_index: Optional[int] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
