"""
run_interactive.py  --  Interactive Demo
=========================================
Type any question. Both systems answer it. You see the difference yourself.

  Semantic RAG  ->  one vector search -> chunks stuffed into a prompt
                    -> ONE GPT-4o call -> synthesized answer (no loop, no tools)

  Agentic       ->  GPT-4o in a reasoning loop -> decides what to search
                    -> calls tools (search, calculate, date) -> synthesized answer

Both return a natural-language answer. The quality gap is visible in the output.

Run:  python run_interactive.py
"""

import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")   # suppress httpx SSL warnings

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, ".")
from shared.kb import build_knowledge_base, semantic_search
from shared.display import console

from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box

import importlib
_ag = importlib.import_module("02_agentic_retrieval")
run_agentic_query = _ag.run_agentic_query
make_openai_client = _ag.make_openai_client

# Also import the single-shot RAG runner from 03_comparison_demo
_comp = importlib.import_module("03_comparison_demo")
run_semantic_rag = _comp.run_semantic_rag


# ---------------------------------------------------------------------------
# Example queries shown to the user on startup
# ---------------------------------------------------------------------------
EXAMPLES = [
    ("1", "SIMPLE",
     "What is the database migration policy?"),
    ("2", "MULTI-HOP",
     "Who is responsible for the service that caused the November 2024 latency incident "
     "and what was the root cause?"),
    ("3", "AGGREGATION",
     "What is the combined recovery time if both the Auth service "
     "and the Payment Gateway fail simultaneously?"),
    ("4", "CONDITIONAL",
     "We want to deploy a new feature to the Checkout service this Friday afternoon. "
     "What approvals do we need and who should we contact?"),
]

COMPLEXITY_COLORS = {
    "SIMPLE":      "blue",
    "MULTI-HOP":   "yellow",
    "AGGREGATION": "cyan",
    "CONDITIONAL": "magenta",
}

# Plain-English explanation of each query type shown after the comparison
COMPLEXITY_WHAT_IT_MEANS = {
    "SIMPLE":
        "Answer lives in a single document. One search is enough.\n"
        "Both approaches work equally well here.",

    "MULTI-HOP":
        "Answer requires chaining facts across 2+ documents.\n"
        "e.g. Incident -> service name -> architecture doc -> owner -> team directory -> email.\n"
        "Semantic RAG does ONE search and stops. If it missed a doc, that info is gone.\n"
        "Agentic keeps searching until the full chain is complete.",

    "AGGREGATION":
        "Answer requires collecting numbers from multiple docs and doing arithmetic.\n"
        "e.g. Auth RTO (10 min) + Payment RTO (15 min) = 25 min combined.\n"
        "Semantic RAG may or may not add correctly -- it has no calculator tool.\n"
        "Agentic explicitly calls calculate(10 + 15) = 25 and cites both source docs.",

    "CONDITIONAL":
        "Answer depends on IF/THEN logic plus real-world state (today's date/time).\n"
        "e.g. 'Can we deploy Friday afternoon?' = policy lookup + is-it-friday? + who to call.\n"
        "Semantic RAG quotes the policy but cannot tell you if THIS Friday is blocked.\n"
        "Agentic calls get_today_info(), checks the day and time, then applies the rule.",
}


def show_welcome():
    console.print()
    console.print(Panel(
        "[bold cyan]SEMANTIC RAG vs AGENTIC RETRIEVAL[/bold cyan]  --  Interactive Demo\n\n"
        "Same knowledge base. Same GPT-4o model. Different retrieval strategy.\n\n"
        "[blue]Semantic RAG[/blue]  does ONE vector search, stuffs the chunks into a prompt,\n"
        "              and calls GPT-4o ONCE. Fast. Simple. Limited by that one search.\n\n"
        "[green]Agentic[/green]      puts GPT-4o in a loop. It decides what to search, calls tools\n"
        "              (search, calculator, date lookup), reflects, and synthesizes.\n\n"
        "Type your own question, or pick one of the examples below.",
        border_style="cyan",
        padding=(1, 2),
    ))


def show_examples():
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    table.add_column("Pick", style="bold", width=5)
    table.add_column("Type", width=14)
    table.add_column("Question", width=80)
    for num, ctype, q in EXAMPLES:
        color = COMPLEXITY_COLORS[ctype]
        table.add_row(num, f"[{color}]{ctype}[/{color}]", q)
    console.print(table)
    console.print("[dim]Enter a number (1-4) to use an example, or type your own question.[/dim]")
    console.print("[dim]Type 'quit' to exit.\n[/dim]")


def get_query() -> str | None:
    try:
        raw = input("Your question: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw or raw.lower() in ("quit", "q", "exit"):
        return None
    # let user pick an example by number
    for num, _, q in EXAMPLES:
        if raw == num:
            console.print(f"[dim]Using: {q}[/dim]\n")
            return q
    return raw


def detect_complexity(query: str) -> str:
    """Rough heuristic to label the query type for display purposes."""
    q = query.lower()
    if any(w in q for w in ["combined", "total", "sum", "both", "simultaneously"]):
        return "AGGREGATION"
    if any(w in q for w in ["friday", "monday", "deploy", "weekend", "today", "this week"]):
        return "CONDITIONAL"
    if any(w in q for w in ["who", "responsible", "caused", "led to", "owner", "team lead"]):
        return "MULTI-HOP"
    return "SIMPLE"


def run_semantic_section(query: str, collection, client) -> tuple[str, list[dict]]:
    """
    Run single-shot semantic RAG and print the result.
    Returns (answer, results_list).
    """
    console.print()
    console.rule("[bold blue]SEMANTIC RAG  (single-shot)[/bold blue]")
    console.print(
        "[dim]ONE vector search -> chunks into prompt -> ONE GPT-4o call\n"
        "No loop. No tools. Model can only use what the first search returned.[/dim]\n"
    )

    answer, results = run_semantic_rag(query, collection, client)

    console.print(
        Panel(
            answer,
            title="[bold blue]Semantic RAG Answer[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    return answer, results


def run_agentic_section(query: str, collection, client) -> str:
    """Run agentic retrieval with live chain-of-thought output. Returns final answer."""
    console.print()
    console.rule("[bold green]AGENTIC RETRIEVAL  (multi-step loop)[/bold green]")
    console.print(
        "[dim]GPT-4o reasons, searches multiple times, uses tools, reflects, synthesizes.\n"
        "Watch every step below:[/dim]\n"
    )
    answer = run_agentic_query(query, collection, client)
    return answer


def show_comparison(
    query: str,
    complexity: str,
    sem_answer: str,
    sem_results: list[dict],
    ag_answer: str,
):
    """Print both synthesized answers side by side with a plain-English verdict."""
    console.print()
    console.rule("[bold white]SIDE-BY-SIDE COMPARISON[/bold white]")

    color = COMPLEXITY_COLORS.get(complexity, "white")

    # Explain the query type and why it matters
    console.print(
        Panel(
            COMPLEXITY_WHAT_IT_MEANS[complexity],
            title=f"[bold {color}]Why this is a '{complexity}' query[/bold {color}]",
            border_style=color,
            padding=(0, 1),
        )
    )
    console.print()

    # Side-by-side: both are synthesized answers (apples to apples)
    retrieved_ids = ", ".join(r["id"] for r in sem_results)
    table = Table(
        box=box.SIMPLE_HEAD,
        show_lines=True,
        padding=(0, 1),
        width=140,
    )
    table.add_column(
        f"[bold blue]SEMANTIC RAG[/bold blue]\n"
        f"1 search -> 1 LLM call\n"
        f"Retrieved: {retrieved_ids}",
        ratio=1,
    )
    table.add_column(
        "[bold green]AGENTIC RETRIEVAL[/bold green]\n"
        "Multi-step loop + tools\n"
        "Searched until complete",
        ratio=1,
    )
    table.add_row(sem_answer, ag_answer)
    console.print(table)

    # Verdict
    verdicts = {
        "SIMPLE": (
            "[blue]Both approaches give similar answers here.[/blue]\n"
            "One search is enough. Agentic adds latency without much benefit."
        ),
        "MULTI-HOP": (
            "[green]Agentic wins.[/green]\n\n"
            "Look for the email address in both answers.\n"
            "Semantic RAG likely never fetched TEAM-001 so it's missing contact details.\n"
            "Agentic followed the chain: incident -> owner -> directory -> email."
        ),
        "AGGREGATION": (
            "[green]Agentic wins on precision.[/green]\n\n"
            "Semantic RAG might state the numbers but cannot guarantee it added them.\n"
            "Agentic called calculate() explicitly and cited both source documents."
        ),
        "CONDITIONAL": (
            "[green]Agentic wins.[/green]\n\n"
            "Semantic RAG quoted the policy but cannot resolve 'this Friday afternoon'\n"
            "to a concrete yes/no.\n"
            "Agentic called get_today_info(), checked the actual UTC time, and told\n"
            "you exactly what to do right now."
        ),
    }
    verdict = verdicts.get(complexity, "")
    console.print(
        Panel(verdict, title="[bold]Verdict[/bold]", border_style="white", padding=(0, 1))
    )


def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print(Panel(
            "[red]OPENAI_API_KEY not set.[/red]\n"
            "Add to .env file:  OPENAI_API_KEY=sk-proj-...",
            border_style="red",
        ))
        sys.exit(1)

    show_welcome()

    console.print("[dim]Building knowledge base (first run downloads ~80 MB model)...[/dim]")
    collection = build_knowledge_base()
    client = make_openai_client()

    while True:
        console.print()
        console.rule()
        show_examples()

        query = get_query()
        if query is None:
            console.print("\n[dim]Goodbye.[/dim]\n")
            break

        complexity = detect_complexity(query)
        color = COMPLEXITY_COLORS[complexity]
        console.print(
            f"\n[dim]Query type detected:[/dim] [{color}]{complexity}[/{color}]  "
            f"[dim](this determines how much agentic will outperform semantic)[/dim]\n"
        )

        # Run both approaches
        sem_answer, sem_results = run_semantic_section(query, collection, client)
        ag_answer = run_agentic_section(query, collection, client)

        # Show both answers side by side
        show_comparison(query, complexity, sem_answer, sem_results, ag_answer)

        console.print("\n[dim]Press Enter to ask another question, or type 'quit' to exit.[/dim]")


if __name__ == "__main__":
    main()
