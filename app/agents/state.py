from typing import Annotated, Sequence, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state of our LangGraph agent.
    
    Attributes:
        messages: The sequence of messages representing the chat history.
        retrieved_docs: List of documents fetched from vector search.
        graph_context: Extracted relations context from the knowledge graph.
        search_query: The refined search query from the planner node.
        search_type: The type of retrieval to perform ('vector', 'graph', or 'both').
        rail_fired: Whether a guardrail triggered, indicating the graph should early-exit.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_docs: List[dict]
    graph_context: str
    search_query: str
    search_type: str
    rail_fired: bool