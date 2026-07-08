from langchain_core.messages import SystemMessage
from app.agents.state import AgentState
from app.agents.nodes.retriever import get_retriever_node

def get_planner_node(chatbot_instance):
    """
    Returns the planner node function. Binds vector_search and graph_search
    tools to the chatbot's LLM for routing.
    """
    # Retrieve the tools list bound to the chatbot instance
    _, tools = get_retriever_node(chatbot_instance)
    
    # Bind tools to the configured Groq LLM
    llm_with_tools = chatbot_instance.llm.bind_tools(tools)

    def call_planner(state: AgentState) -> dict:
        """
        Routes the user query by either triggering tools or responding directly.
        """
        system_prompt = SystemMessage(content=(
            "You are Umer's AI chatbot routing assistant.\n"
            "Your job is to decide whether the user's question requires searching Umer's "
            "experience, skills, projects, and codebase using your tools ('vector_search' and 'graph_search').\n\n"
            "Guidelines:\n"
            "1. If the user's message is a greeting, chitchat, self-intro, or general conversational message "
            "not asking about Umer's credentials/skills (e.g. 'hi', 'how are you', 'tell me a joke'), do NOT call any tools. Just reply directly.\n"
            "2. If it is a semantic or general question about Umer's projects/skills/experience, call the 'vector_search' tool.\n"
            "3. If it is a relational query asking about what tech Umer built what projects with, what years "
            "he was active, or connections, call the 'graph_search' tool.\n"
            "4. If it requires both, call both tools."
        ))
        
        # Build prompt: system instructions + existing conversation history
        messages = [system_prompt] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return call_planner
