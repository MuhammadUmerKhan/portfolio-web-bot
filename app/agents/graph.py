from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import get_planner_node
from app.agents.nodes.retriever import get_retriever_node
from app.agents.nodes.responder import get_responder_node

def create_agent_graph(chatbot_instance):
    """
    Constructs and compiles the stateful LangGraph agent workflow.
    
    Workflow structure:
    START -> planner -> [execute tools if called] -> responder -> END
                     -> [reply directly if no tools] -----------> END
    """
    
    # 1. Instantiate the state graph
    workflow = StateGraph(AgentState)
    
    # 2. Retrieve node functions
    planner_node = get_planner_node(chatbot_instance)
    retriever_node, _ = get_retriever_node(chatbot_instance)
    responder_node = get_responder_node(chatbot_instance)
    
    # 3. Add nodes to the graph
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("responder", responder_node)
    
    # 4. Set up conditional routing from planner
    def route_after_planner(state: AgentState):
        """
        Determines if the graph should execute retrieval tools or end.
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "retriever"
        return END

    # 5. Define edges
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            END: END
        }
    )
    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", END)
    
    # 6. Compile graph with memory checkpointer
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    return compiled_graph
