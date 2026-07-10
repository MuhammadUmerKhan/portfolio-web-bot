import asyncio
import logfire
from app.services.chatbot import CustomDocChatbot
from langchain_core.messages import HumanMessage

async def main():
    bot = CustomDocChatbot()
    
    config = {"configurable": {"thread_id": "react-debug-1"}}
    
    print("\n=== USER: Tell me about your SmartSearch project in detail ===")
    state_input = {"messages": [HumanMessage(content="Tell me about your SmartSearch project in detail")]}
    
    async for event in bot.agent.astream(state_input, config=config, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            last_msg.pretty_print()

    print("\n=== USER: What technologies did you use for it? (Testing Context/Memory) ===")
    state_input = {"messages": [HumanMessage(content="What technologies did you use for it?")]}
    
    async for event in bot.agent.astream(state_input, config=config, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            last_msg.pretty_print()

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(main())
