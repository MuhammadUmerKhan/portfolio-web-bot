import json
from pathlib import Path
from app.ingestion.loader.base import RawDocument, SourceType

class MarkdownLoader:
    """
    Scans a directory for Markdown files (*.md) and extracts their content,
    recursively mapping repository configuration from parent _meta.json files.
    """
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def load(self) -> list[RawDocument]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
            
        documents = []
        
        # Search recursively for markdown files
        for md_path in self.root_dir.rglob("*.md"):
            if md_path.is_file() and not md_path.name.startswith("_"):
                content = md_path.read_text(encoding="utf-8", errors="replace").strip()
                if not content:
                    continue  # skip empty README/docs stubs — nothing to embed
                
                # Check for repository configuration (_meta.json) in parent folders
                meta_data = {}
                parent = md_path.parent
                while parent and parent != self.root_dir.parent:
                    meta_path = parent / "_meta.json"
                    if meta_path.exists():
                        try:
                            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                        break
                    parent = parent.parent
                
                # Relative path within the repo, computed structurally from the known
                # root_dir rather than by matching the repo name as a string — the
                # string-match approach breaks if _meta.json is missing/malformed, and
                # is fragile if a repo name ever collides with another path segment.
                relative_path = None
                try:
                    parts = md_path.relative_to(self.root_dir).parts
                    if len(parts) > 1:
                        relative_path = "/".join(parts[1:])
                    else:
                        relative_path = parts[0]
                except ValueError:
                    pass
                
                documents.append(
                    RawDocument(
                        content=content,
                        source_type=SourceType.GITHUB_README,
                        source_path=str(md_path.resolve()),
                        project_name=meta_data.get("name"),
                        project_description=meta_data.get("description"),
                        project_url=meta_data.get("url"),
                        project_language=meta_data.get("language"),
                        project_topics=meta_data.get("topics", []),
                        project_pushed_at=meta_data.get("pushed_at"),
                        is_private=meta_data.get("private", False),
                        relative_path=relative_path,
                    )
                )
                
        return documents
