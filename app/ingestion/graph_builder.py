import json
from pathlib import Path

def build_graph(chunks: list[dict]) -> dict:
    """
    Parses a list of chunks (matching chunks.json format) to construct a lightweight 
    adjacency list knowledge graph of Projects, Skills, Companies, Platforms, and Years.
    """
    nodes = {}  # { "Type:Name": {"type": type, "name": name} }
    edges = []  # List of tuples: (source_key, target_key, relationship_type)
    
    def add_node(node_type, name):
        if not name:
            return None
        # Use clean casing
        clean_name = name.strip()
        key = f"{node_type}:{clean_name}"
        if key not in nodes:
            nodes[key] = {"type": node_type, "name": clean_name}
        return key
        
    def add_edge(source, target, rel_type):
        if not source or not target or source == target:
            return
        edge = (source, target, rel_type)
        if edge not in edges:
            edges.append(edge)

    for chunk in chunks:
        # Extract metadata fields
        meta = chunk.get("metadata", {})
        project = chunk.get("project_name")
        tech_stack = meta.get("tech_stack", [])
        companies = meta.get("companies", [])   # real employer/institute relationships only
        platforms = meta.get("platforms", [])   # vendors/tools mentioned — not employers
        years = meta.get("years", [])
        
        # 1. Project mapping
        project_key = None
        if project:
            project_key = add_node("Project", project)
            
        # 2. Skill mapping
        for tech in tech_stack:
            tech_key = add_node("Skill", tech)
            if project_key:
                # Add bidirectional relationships
                add_edge(project_key, tech_key, "built_with")
                add_edge(tech_key, project_key, "used_in")
                
        # 3. Employer/institute mapping — genuine work/education relationships only
        for company in companies:
            company_key = add_node("Company", company)
            if project_key:
                add_edge(project_key, company_key, "associated_with")

        # 4. Platform/vendor mapping — kept as a distinct node type so the graph
        # never implies employment just because a README mentions calling an
        # API or deploying to a hosting provider.
        for platform in platforms:
            platform_key = add_node("Platform", platform)
            if project_key:
                add_edge(project_key, platform_key, "uses_platform")
                
        # 5. Temporal mapping — only for the project and its real employer/institute.
        # Platforms are deliberately excluded: a vendor mentioned alongside a
        # year in the same chunk isn't "active in" that year.
        for year in years:
            year_key = add_node("Year", year)
            if project_key:
                add_edge(project_key, year_key, "active_in")
            for company in companies:
                company_key = add_node("Company", company)
                add_edge(company_key, year_key, "active_in")

    # Serialize nodes to adjacency format
    adjacency = {}
    for key in nodes:
        adjacency[key] = {
            "type": nodes[key]["type"],
            "name": nodes[key]["name"],
            "neighbors": []
        }
        
    for source, target, rel in edges:
        adjacency[source]["neighbors"].append({
            "target": target,
            "relation": rel
        })
        
    return adjacency

def main():
    """
    CLI entrypoint: Parses data/processed/chunks.json and serializes the 
    resulting graph to data/processed/knowledge_graph.json.
    """
    chunks_path = Path("data/processed/chunks.json")
    graph_path = Path("data/processed/knowledge_graph.json")
    
    if not chunks_path.exists():
        print(f"❌ Chunks cache not found at {chunks_path}. Please run ingestion first.")
        return
        
    print("Loading processed chunks for graph extraction...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    adjacency = build_graph(chunks)
    
    # Ensure parent output dir exists
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(adjacency, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Knowledge graph compiled successfully.")
    print(f"Output saved to: {graph_path}")

if __name__ == "__main__":
    main()
