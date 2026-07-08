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
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_docs: List[dict]
    graph_context: str