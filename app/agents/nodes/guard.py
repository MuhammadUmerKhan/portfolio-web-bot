from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.guardrails.rails import guard
import logfire

def get_guard_node(chatbot_instance):
    """
    Returns the guard node function bound to the chatbot instance.
    This node intercepts the user message and passes it through NeMo Guardrails.
    """
    
    async def guard_node(state: AgentState):
        """
        Runs the user query through the input guardrails.
        If a rail fires, we return the canned response and set rail_fired=True.
        """
        # Get the latest user message
        messages = state.get("messages", [])
        if not messages:
            return {"rail_fired": False}
            
        latest_message = messages[-1].content
        
        # Run through NeMo rails
        fired, response = await guard(latest_message)
        
        if fired and response:
            logfire.info("🛡️ Guard node blocked query and returning predefined response.")
            return {
                "messages": [AIMessage(content=response)],
                "rail_fired": True
            }
            
        return {"rail_fired": False}
        
    return guard_node
