from app.agents.state import AgentState

def get_trad_rag_node(chatbot_instance):
    """
    Returns the Traditional RAG node.
    Executes dense/sparse retrieval using the vector database and BM25.
    """
    from langsmith import traceable

    @traceable(name="Traditional RAG Node")
    def execute_trad_rag(state: AgentState) -> dict:
        search_query = state.get("search_query", "")
        retrieved_docs = []
        
        if search_query:
            docs = chatbot_instance.retriever.invoke(search_query)
            for doc in docs:
                retrieved_docs.append({
                    "page_content": f"--- Document Source: {doc.metadata.get('source_path', 'Unknown')} ---\n{doc.page_content}",
                    "metadata": doc.metadata
                })
        return {"retrieved_docs": retrieved_docs}
        
    return execute_trad_rag

def get_graph_rag_node(chatbot_instance):
    """
    Returns the Graph RAG node.
    Executes relational lookups against the Knowledge Graph.
    """
    from langsmith import traceable

    @traceable(name="Graph RAG Node")
    def execute_graph_rag(state: AgentState) -> dict:
        search_query = state.get("search_query", "")
        graph_context = ""
        
        if search_query:
            graph_context = chatbot_instance.graph_service.query_graph(search_query)
        return {"graph_context": graph_context}
        
    return execute_graph_rag
