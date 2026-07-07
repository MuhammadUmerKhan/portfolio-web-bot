"""
Shared document schema for the ingestion pipeline.

Every loader (markdown, PDF, and any future source type) returns a list
of RawDocument — a normalized shape that app/ingestion/chunking consumes
regardless of where the content originally came from.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    GITHUB_README = "github_readme"
    RESUME_PDF = "resume_pdf"


class RawDocument(BaseModel):
    content: str

    source_type: SourceType
    source_path: str  # absolute/relative filesystem path this came from

    # Populated for GITHUB_README documents (from the repo's _meta.json,
    # written by app/ingestion/fetchers/github_fetcher.py). Left as None
    # for other source types rather than guessing values that don't apply.
    project_name: Optional[str] = None
    project_description: Optional[str] = None
    project_url: Optional[str] = None
    project_language: Optional[str] = None
    project_topics: list[str] = Field(default_factory=list)
    project_pushed_at: Optional[str] = None
    is_private: bool = False
    relative_path: Optional[str] = None  # path within the repo, e.g. "app/README.md"

    # Populated for RESUME_PDF documents
    page_number: Optional[int] = None
