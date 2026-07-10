from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from app.agents.state import AgentState
from app.agents.nodes.guard import get_guard_node
from app.agents.nodes.agent import get_agent_node
from app.agents.nodes.tools import get_tools

def create_agent_graph(chatbot_instance):
    """
    Constructs and compiles the stateful LangGraph agent workflow.
    
    Workflow structure (ReAct Loop):
    START -> guard -> [if rail fired] -----------------------> END
                   -> [if clean] -> agent <---> tools
                                          -> [no tools needed] -> END
    """
    
    # 1. Instantiate the state graph
    workflow = StateGraph(AgentState)
    
    # 2. Retrieve node functions
    guard_node = get_guard_node(chatbot_instance)
    agent_node = get_agent_node(chatbot_instance)
    tools = get_tools(chatbot_instance)
    tool_node = ToolNode(tools)
    
    # 3. Add nodes to the graph
    workflow.add_node("guard", guard_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # 4. Set up conditional routing
    def route_after_guard(state: AgentState):
        """Determines if a guardrail fired and we should short-circuit to END."""
        if state.get("rail_fired", False):
            return END
        return "agent"
        
    def route_after_agent(state: AgentState):
        """
        Determines if the agent decided to call a tool or if it gave a final answer.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM returned tool calls, route to the tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
            
        # Otherwise, the LLM has generated its final response
        return END

    # 5. Define edges
    workflow.add_edge(START, "guard")
    
    workflow.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            END: END,
            "agent": "agent"
        }
    )
    
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # The tools node always routes back to the agent to process the tool output
    workflow.add_edge("tools", "agent")
    
    # 6. Compile the graph with memory check-pointing
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app
