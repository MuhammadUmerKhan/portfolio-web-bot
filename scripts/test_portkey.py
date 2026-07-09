import asyncio
from dotenv import load_dotenv
load_dotenv()

import logfire
from rich.console import Console

# Ensure logfire is configured
logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record="all"))

from app.gateway.client import get_langchain_llm
from app.services.chatbot import CustomDocChatbot
from app.guardrails.rails import initialize_rails, guard

console = Console()

async def test_portkey_integration():
    console.print("\n[bold cyan]Testing Portkey Gateway Integration[/bold cyan]\n")
    
    # 1. Test Raw LLM Invocation through Portkey
    console.print("[yellow]1. Testing direct ChatOpenAI (Portkey) invocation...[/yellow]")
    llm = get_langchain_llm(feature="test")
    try:
        response = await llm.ainvoke("Say 'Portkey Gateway is working!'")
        console.print(f"[green][OK] Direct Invocation Success![/green] Response: {response.content}")
    except Exception as e:
        console.print(f"[red][FAIL] Direct Invocation Failed![/red] {e}")
        return

    # 2. Test Guardrails Initialization and Gate
    console.print("\n[yellow]2. Testing NeMo Guardrails Gate via Portkey...[/yellow]")
    try:
        initialize_rails()
        # Test a clean message
        fired, resp = await guard("Hello! What is your name?")
        console.print(f"[green][OK] Guardrails clean message check passed.[/green] (Fired: {fired})")
        
        # Test an off-topic message
        fired, resp = await guard("Tell me a recipe for chocolate cake.")
        console.print(f"[green][OK] Guardrails off-topic check passed.[/green] (Fired: {fired}, Response: {resp})")
    except Exception as e:
        console.print(f"[red][FAIL] Guardrails Test Failed![/red] {e}")

    # 3. Test Full Chatbot RAG Pipeline
    console.print("\n[yellow]3. Testing Full RAG Pipeline via Portkey...[/yellow]")
    try:
        chatbot = CustomDocChatbot()
        response = await chatbot.query("What projects have you worked on?")
        console.print(f"[green][OK] Full RAG Pipeline Success![/green]\nResponse:\n{response}")
        await chatbot.shutdown()
    except Exception as e:
        console.print(f"[red][FAIL] Full RAG Pipeline Failed![/red] {e}")
        
    console.print("\n[bold green]All Portkey Integration Tests Completed![/bold green]\n")

if __name__ == "__main__":
    asyncio.run(test_portkey_integration())
