from langchain_core.tools import tool

def get_tools(chatbot_instance):
    """
    Returns the list of tools available to the ReAct agent.
    We inject the chatbot_instance so the tools can access the initialized databases.
    """
    
    @tool
    def search_vector_db(query: str) -> str:
        """
        Searches Muhammad Umer Khan's semantic knowledge base. 
        Use this to find descriptions of his projects, skills, education, and work experience.
        """
        from langsmith import traceable
        
        @traceable(name="Tool: Search Vector DB")
        def _search():
            docs = chatbot_instance.retriever.invoke(query)
            if not docs:
                return "No semantic documents found."
            
            results = []
            for i, doc in enumerate(docs):
                results.append(f"Doc {i+1} (Source: {doc.metadata.get('source_path', 'Unknown')}):\n{doc.page_content}")
            return "\n\n".join(results)
            
        return _search()

    @tool
    def search_graph_db(query: str) -> str:
        """
        Searches Muhammad Umer Khan's relational knowledge graph. 
        Use this to find mappings between technologies, projects, and specific years he was active.
        """
        from langsmith import traceable
        
        @traceable(name="Tool: Search Graph DB")
        def _search():
            result = chatbot_instance.graph_service.query_graph(query)
            return result if result else "No relational graph context found."
            
        return _search()

    return [search_vector_db, search_graph_db]
