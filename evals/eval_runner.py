import json
import time
import asyncio
import sys
from rich.console import Console
from rich.table import Table

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from app.services.chatbot import CustomDocChatbot
from langchain_core.messages import HumanMessage
from evals.custom_eval_model import PortkeyEvalLLM

console = Console()

class EvalRunner:
    def __init__(self):
        self.bot = CustomDocChatbot()
        self.eval_model = PortkeyEvalLLM()
        self.dataset_path = "data/golden_dataset.json"
        self.enriched_dataset_path = "data/enriched_golden_dataset.json"

    async def phase_1_generate_responses(self):
        console.print("[bold cyan]🚀 Phase 1: Generating Live Responses[/bold cyan]")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            goldens = json.load(f)

        enriched_goldens = []
        for i, golden in enumerate(goldens):
            console.print(f"[{i+1}/{len(goldens)}] Asking: [yellow]{golden['question']}[/yellow]")
            
            # Invoke Agent
            user_input = golden["question"]
            state_input = {"messages": [HumanMessage(content=user_input)]}
            config = {"configurable": {"thread_id": f"eval_{hash(user_input)}"}}
            
            # Agent has Guardrails so it must be run async
            final_state = await self.bot.agent.ainvoke(state_input, config=config)
            messages = final_state.get("messages", [])
            
            actual_response = ""
            actual_contexts = []
            
            for msg in messages:
                if msg.type == "ai" and msg.content:
                    actual_response = msg.content
                elif msg.type == "tool":
                    actual_contexts.append(msg.content)
            
            # If no context was retrieved, add a dummy to avoid DeepEval crash
            if not actual_contexts:
                actual_contexts = ["No external context retrieved."]

            golden["actual_response"] = actual_response
            golden["actual_contexts"] = actual_contexts
            enriched_goldens.append(golden)
            
            # Rate limit pacing
            if i < len(goldens) - 1:
                console.print("   [dim]Waiting 10s to respect Groq RPM limits...[/dim]")
                await asyncio.sleep(10)

        with open(self.enriched_dataset_path, "w", encoding="utf-8") as f:
            json.dump(enriched_goldens, f, indent=4)
        console.print("[bold green]✅ Phase 1 Complete![/bold green]\n")
        return enriched_goldens

    async def phase_2_evaluate_metrics(self, enriched_goldens):
        console.print("[bold cyan]🧪 Phase 2: RAGAS Metric Scoring[/bold cyan]")
        console.print("[dim]Using strict 40s pacing to avoid Groq 6,000 TPM limit on the Judge key.[/dim]\n")
        
        # Initialize Metrics
        faithfulness = FaithfulnessMetric(threshold=0.7, model=self.eval_model)
        relevancy = AnswerRelevancyMetric(threshold=0.7, model=self.eval_model)

        results = []
        passed_count = 0

        for i, golden in enumerate(enriched_goldens):
            console.print(f"[{i+1}/{len(enriched_goldens)}] Evaluating: [yellow]{golden['question']}[/yellow]")
            
            test_case = LLMTestCase(
                input=golden["question"],
                actual_output=golden["actual_response"],
                expected_output=golden["reference"],
                retrieval_context=golden["actual_contexts"]
            )

            # Evaluate Faithfulness
            await faithfulness.a_measure(test_case)
            f_score = faithfulness.score
            f_reason = faithfulness.reason

            # Evaluate Relevancy
            await relevancy.a_measure(test_case)
            r_score = relevancy.score
            r_reason = relevancy.reason

            results.append({
                "id": golden["id"],
                "question": golden["question"],
                "faithfulness": f_score,
                "relevancy": r_score,
                "pass": f_score >= 0.7 and r_score >= 0.7
            })

            if f_score >= 0.7 and r_score >= 0.7:
                passed_count += 1
                console.print(f"   ✅ [green]PASS[/green] (F: {f_score:.2f}, R: {r_score:.2f})")
            else:
                console.print(f"   ❌ [red]FAIL[/red] (F: {f_score:.2f}, R: {r_score:.2f})")

            if i < len(enriched_goldens) - 1:
                console.print("   [dim]Waiting 40s to recover token window...[/dim]")
                await asyncio.sleep(40)

        self.print_final_report(results, passed_count, len(enriched_goldens))
        
        # Return True if all passed, False otherwise
        return passed_count == len(enriched_goldens)

    def print_final_report(self, results, passed, total):
        console.print("\n[bold magenta]📊 Final Evaluation Report[/bold magenta]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("ID", style="dim", width=4)
        table.add_column("Question")
        table.add_column("Faithfulness")
        table.add_column("Relevancy")
        table.add_column("Status")

        for res in results:
            status = "[green]PASS[/green]" if res["pass"] else "[red]FAIL[/red]"
            table.add_row(
                str(res["id"]),
                res["question"],
                f"{res['faithfulness']:.2f}",
                f"{res['relevancy']:.2f}",
                status
            )
        console.print(table)
        pass_rate = (passed / total) * 100
        console.print(f"\n[bold]Overall Pass Rate: {pass_rate:.1f}% ({passed}/{total})[/bold]")

async def main():
    runner = EvalRunner()
    enriched = await runner.phase_1_generate_responses()
    success = await runner.phase_2_evaluate_metrics(enriched)
    if not success:
        console.print("[bold red]❌ Pipeline Failed: Not all evaluations met the 0.7 threshold.[/bold red]")
        sys.exit(1)
    else:
        console.print("[bold green]✅ Pipeline Passed Successfully![/bold green]")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
