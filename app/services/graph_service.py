import json
import re
from pathlib import Path
from app.core import get_logger

logger = get_logger(__name__)

class GraphService:
    """Service to load the lightweight knowledge graph and inject relationship context at query time."""
    
    def __init__(self):
        self.graph_path = Path("data/processed/knowledge_graph.json")
        self.graph = {}
        self.load_graph()

    def load_graph(self):
        """Loads compiled knowledge graph JSON from processed storage."""
        if not self.graph_path.exists():
            logger.warning("⚠️ Knowledge graph JSON not found at %s. Relational queries will bypass graph context.", self.graph_path)
            return
            
        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                self.graph = json.load(f)
            logger.info("✅ Knowledge graph loaded successfully with %d nodes.", len(self.graph))
        except Exception as e:
            logger.error("❌ Failed to load knowledge graph: %s", str(e))

    def query_graph(self, query: str) -> str:
        """
        Scans query for entity keywords, fetches matching neighbors,
        and constructs a structured context string.
        
        Args:
            query (str): The raw search query.
            
        Returns:
            str: Formatted context block of relationships, or empty string.
        """
        if not self.graph:
            return ""
            
        query_lower = query.lower()
        matched_contexts = []
        
        # Scan nodes for word-boundary matches in query
        for node_key, node_data in self.graph.items():
            node_name = node_data["name"]
            node_type = node_data["type"]
            
            # Match using word boundaries to avoid false substring matches
            pattern = r'\b' + re.escape(node_name.lower()) + r'\b'
            if re.search(pattern, query_lower):
                neighbors = node_data.get("neighbors", [])
                if not neighbors:
                    continue
                    
                # Group neighbors by relationship type for clean output
                grouped = {}
                for n in neighbors:
                    target_key = n["target"]
                    relation = n["relation"]
                    
                    target_name = target_key.split(":", 1)[1] if ":" in target_key else target_key
                    grouped.setdefault(relation, []).append(target_name)
                    
                # Build human readable relationship descriptions
                relations_desc = []
                for rel, targets in grouped.items():
                    targets_str = ", ".join(f"'{t}'" for t in sorted(list(set(targets))))
                    rel_readable = rel.replace("_", " ")
                    relations_desc.append(f"{rel_readable} {targets_str}")
                    
                if relations_desc:
                    context_line = f"'{node_name}' ({node_type}) is associated with: " + "; ".join(relations_desc)
                    matched_contexts.append(context_line)
                    
        if matched_contexts:
            context_block = (
                "\n\n[Lightweight Knowledge Graph Context]\n" + 
                "\n".join(f"- {ctx}" for ctx in matched_contexts)
            )
            logger.info("🎯 Knowledge Graph hit: matched %d entities.", len(matched_contexts))
            return context_block
            
        return ""
