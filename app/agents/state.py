from typing import Annotated, Sequence, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state of our LangGraph agent.
    
    Attributes:
        messages: The sequence of messages representing the chat history. Tool calls and responses are tracked natively here.
        rail_fired: Whether a guardrail triggered, indicating the graph should early-exit.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rail_fired: bool