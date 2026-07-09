from langchain_core.messages import SystemMessage, AIMessage
from pydantic import BaseModel, Field
from app.agents.state import AgentState
from typing import Literal

class PlannerOutput(BaseModel):
    is_conversational: bool = Field(description="True if the message is a greeting (hi, hello) or can be answered using ONLY the conversation history.")
    search_query: str = Field(description="The highly optimized search query if is_conversational is False. Empty string if True.")
    search_type: Literal["vector", "graph", "both", "none"] = Field(description="If is_conversational is False, specify the search type needed: 'vector' for semantic questions (skills/experience), 'graph' for relational questions (tech stack connections/years), or 'both'. Use 'none' if conversational.")
    direct_response: str = Field(description="If is_conversational is True, provide the friendly direct response here. Otherwise, empty string.")

def get_planner_node(chatbot_instance):
    """
    Returns the planner node function. Uses structured output to return either
    a conversational response or a refined search query.
    """
    structured_llm = chatbot_instance.llm.with_structured_output(PlannerOutput)

    def call_planner(state: AgentState) -> dict:
        """
        Routes the user query by outputting a search query or answering directly.
        """
        history_msgs = [m for m in state["messages"][:-1] if getattr(m, 'type', '') != "system"]
        history = "\n".join([f"{m.type}: {m.content}" for m in history_msgs[-4:]])
        user_message = state["messages"][-1].content
        
        prompt = f"""
        You are an intelligent Assistant Planner for Muhammad Umer Khan's portfolio AI.

        CONVERSATION HISTORY: {history}

        LATEST MESSAGE: "{user_message}"

        Task:
        1. If the latest message is a greeting, chitchat, or a question answered entirely by the history above, set `is_conversational`=true, set `search_type`="none", output a friendly `direct_response`, and leave `search_query` empty.
        2. If it's a question requiring fresh context, set `is_conversational`=false, leave `direct_response` empty, output a highly optimized, dense `search_query`, and set `search_type` using these rules:
            - "vector": For semantic questions about Umer's skills, project descriptions, work experience, or education (Traditional RAG).
            - "graph": For relational questions about technology stack mappings, specific years he was active, or connections between projects and skills (Graph Knowledge Base).
            - "both": Only if the question heavily requires both semantic descriptions and relational mapping.
        """

        response = structured_llm.invoke([SystemMessage(content=prompt)])
        
        if response.is_conversational:
            return {
                "messages": [AIMessage(content=response.direct_response)],
                "search_query": "",
                "search_type": "none"
            }
        else:
            return {
                "search_query": response.search_query,
                "search_type": response.search_type
            }

    return call_planner
