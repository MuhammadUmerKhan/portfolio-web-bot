from app.agents.state import AgentState

def get_retriever_node(chatbot_instance):
    """
    Returns the retrieval execution node.
    It reads the 'search_query' from the state and executes both dense/sparse 
    retrieval and knowledge graph lookups.
    """
    def execute_retriever(state: AgentState) -> dict:
        """
        Executes retrieval using the refined search query from the planner.
        """
        search_query = state.get("search_query", "")
        search_type = state.get("search_type", "both")
        
        if not search_query or search_type == "none":
            return {"retrieved_docs": [], "graph_context": ""}

        retrieved_docs = []
        graph_context = ""

        # 1. Fetch dense/sparse docs through the hybrid retriever
        if search_type in ["vector", "both"]:
            docs = chatbot_instance.retriever.invoke(search_query)
            for doc in docs:
                retrieved_docs.append({
                    "page_content": f"--- Document Source: {doc.metadata.get('source_path', 'Unknown')} ---\n{doc.page_content}",
                    "metadata": doc.metadata
                })
            
        # 2. Fetch context from the knowledge graph
        if search_type in ["graph", "both"]:
            graph_context = chatbot_instance.graph_service.query_graph(search_query)

        return {
            "retrieved_docs": retrieved_docs,
            "graph_context": graph_context
        }
        
    return execute_retriever, []
