from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.guard import get_guard_node
from app.agents.nodes.planner import get_planner_node
from app.agents.nodes.retriever import get_retriever_node
from app.agents.nodes.responder import get_responder_node

def create_agent_graph(chatbot_instance):
    """
    Constructs and compiles the stateful LangGraph agent workflow.
    
    Workflow structure:
    START -> guard -> [if rail fired] ---------------------------> END
                   -> [if clean] -> planner -> [if retrieval] -> retriever -> responder -> END
                                            -> [if no retrieval] -> responder -> END
    """
    from langsmith import traceable
    
    # 1. Instantiate the state graph
    workflow = StateGraph(AgentState)
    
    # 2. Retrieve node functions
    guard_node = get_guard_node(chatbot_instance)
    planner_node = get_planner_node(chatbot_instance)
    retriever_node, _ = get_retriever_node(chatbot_instance)
    responder_node = get_responder_node(chatbot_instance)
    
    # 3. Add nodes to the graph
    workflow.add_node("guard", guard_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("responder", responder_node)
    
    # 4. Set up conditional routing
    def route_after_guard(state: AgentState):
        """
        Determines if a guardrail fired and we should short-circuit to END.
        """
        if state.get("rail_fired", False):
            return END
        return "planner"
        
    def route_after_planner(state: AgentState):
        """
        Determines if the graph should execute retrieval tools or end.
        """
        search_query = state.get("search_query", "")
        if search_query:
            return "retriever"
        return "responder"

    # 5. Define edges
    workflow.add_edge(START, "guard")
    workflow.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            END: END,
            "planner": "planner"
        }
    )
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            "responder": "responder"
        }
    )
    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", END)
    
    # 6. Compile graph with memory checkpointer
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    return compiled_graph
