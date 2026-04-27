"""
run_interactive.py  --  Interactive Demo
=========================================
Type any question. Both systems answer it. You see the difference yourself.

FAIR COMPARISON:
  Both approaches use GPT-4o, the same ChromaDB index, and the same
  utility tools (calculate, get_today_info).

  Semantic RAG  ->  ONE fixed vector search -> chunks into prompt
                    -> GPT-4o with calculate + get_today_info (no search loop)
                    -> synthesized answer limited to what that one search found

  Agentic       ->  GPT-4o in a loop -> decides what to search and when
                    -> search_knowledge_base + calculate + get_today_info
                    -> synthesized answer after all needed docs are found

The ONLY difference: search_knowledge_base in a loop.
Giving that tool to semantic RAG would make it agentic by definition.

Run:  python run_interactive.py
"""

import sys
import os
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

# Import agentic runner and tool executor from 02_agentic_retrieval
_ag = importlib.import_module("02_agentic_retrieval")
run_agentic_query = _ag.run_agentic_query
make_openai_client = _ag.make_openai_client

# Import semantic RAG runner (with fair tool access) from 03_comparison_demo
_comp = importlib.import_module("03_comparison_demo")
run_semantic_rag = _comp.run_semantic_rag


# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
EXAMPLES = [
    ("1", "SIMPLE",
     "What is the database migration policy?"),
    ("2", "MULTI-HOP",
     "Who is responsible for the service that caused the November 2024 latency incident, "
     "what was the root cause, and what is their contact email?"),
    ("3", "AGGREGATION",
     "What is the combined recovery time if the Auth service, Payment Gateway, "
     "and Checkout service all fail simultaneously?"),
    ("4", "CONDITIONAL",
     "A developer wants to deploy the Checkout service and run a database migration "
     "this Friday at 2pm UTC. What approvals are needed for both, and who should they contact?"),
]

COMPLEXITY_COLORS = {
    "SIMPLE":      "blue",
    "MULTI-HOP":   "yellow",
    "AGGREGATION": "cyan",
    "CONDITIONAL": "magenta",
}

# What each query type means and where the gap will show up
COMPLEXITY_NOTES = {
    "SIMPLE": (
        "Answer is in one document. One search is enough.\n"
        "Both approaches will give similar answers here."
    ),
    "MULTI-HOP": (
        "Needs a chain of documents: A -> B -> C.\n"
        "Semantic RAG did one search and stopped. If a doc in the chain\n"
        "was not in the top-4, its information is simply missing.\n"
        "Agentic followed the trail document by document.\n"
        "Look for the email address -- it is the clearest indicator of the gap."
    ),
    "AGGREGATION": (
        "Needs numbers from 2 docs + arithmetic.\n"
        "Both had the calculate tool -- gap narrows here.\n"
        "Agentic is more reliable because it explicitly searches for each doc.\n"
        "Semantic RAG depends on whether both docs landed in the one search."
    ),
    "CONDITIONAL": (
        "Needs policy + today's date + contact lookup.\n"
        "Both had get_today_info -- date gap closes.\n"
        "Remaining gap: if the team directory (TEAM-001) was not retrieved,\n"
        "semantic RAG cannot name the approvers."
    ),
}


def show_welcome():
    console.print()
    console.print(Panel(
        "[bold cyan]SEMANTIC RAG vs AGENTIC RETRIEVAL[/bold cyan]  --  Interactive Demo\n\n"
        "[bold]Fair comparison:[/bold] same model, same index, same utility tools.\n"
        "The ONLY difference is whether the LLM controls the retrieval loop.\n\n"
        "[blue]Semantic RAG[/blue]   ONE vector search -> chunks in prompt\n"
        "               -> GPT-4o with calculate + get_today_info\n"
        "               Cannot search again. Limited to what that search returned.\n\n"
        "[green]Agentic[/green]       GPT-4o decides what to search and when\n"
        "               -> search + calculate + get_today_info in a loop\n"
        "               Keeps going until the answer is complete.\n\n"
        "Type your own question or pick an example below.",
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
    for num, _, q in EXAMPLES:
        if raw == num:
            console.print(f"[dim]Using: {q}[/dim]\n")
            return q
    return raw


def detect_complexity(query: str) -> str:
    """Rough heuristic to label the query type for display."""
    q = query.lower()
    if any(w in q for w in ["combined", "total", "sum", "both", "simultaneously"]):
        return "AGGREGATION"
    if any(w in q for w in ["friday", "monday", "deploy", "weekend", "today", "this week"]):
        return "CONDITIONAL"
    if any(w in q for w in ["who", "responsible", "caused", "led to", "owner", "team lead"]):
        return "MULTI-HOP"
    return "SIMPLE"


def run_semantic_section(query: str, collection, client) -> tuple[str, list[dict]]:
    """Run single-shot semantic RAG with fair tool access and print result."""
    console.print()
    console.rule("[bold blue]SEMANTIC RAG  (single-shot)[/bold blue]")
    console.print(
        "[dim]ONE vector search -> chunks into prompt\n"
        "-> GPT-4o with calculate + get_today_info  (no search loop)[/dim]\n"
    )

    answer, results = run_semantic_rag(query, collection, client)

    console.print(Panel(
        answer,
        title="[bold blue]Semantic RAG Answer[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))
    return answer, results


def run_agentic_section(query: str, collection, client) -> str:
    """Run agentic retrieval with live chain-of-thought output."""
    console.print()
    console.rule("[bold green]AGENTIC RETRIEVAL  (multi-step loop)[/bold green]")
    console.print(
        "[dim]GPT-4o reasons, searches multiple times, uses all tools, synthesizes.\n"
        "Every step below is live:[/dim]\n"
    )
    return run_agentic_query(query, collection, client)


def show_comparison(
    query: str,
    complexity: str,
    sem_answer: str,
    sem_results: list[dict],
    ag_answer: str,
):
    """Print both synthesized answers side by side with a verdict."""
    console.print()
    console.rule("[bold white]SIDE-BY-SIDE COMPARISON[/bold white]")

    color = COMPLEXITY_COLORS.get(complexity, "white")
    note = COMPLEXITY_NOTES.get(complexity, "")

    console.print(Panel(
        note,
        title=f"[bold {color}]'{complexity}' query -- what to look for[/bold {color}]",
        border_style=color,
        padding=(0, 1),
    ))
    console.print()

    retrieved_ids = ", ".join(r["id"] for r in sem_results)
    table = Table(
        box=box.SIMPLE_HEAD,
        show_lines=True,
        padding=(0, 1),
        width=140,
    )
    table.add_column(
        f"[bold blue]SEMANTIC RAG[/bold blue]\n"
        f"1 search -> GPT-4o + calculate + get_today_info\n"
        f"Retrieved: {retrieved_ids}",
        ratio=1,
    )
    table.add_column(
        "[bold green]AGENTIC RETRIEVAL[/bold green]\n"
        "Loop: search + calculate + get_today_info\n"
        "Searched until complete",
        ratio=1,
    )
    table.add_row(sem_answer, ag_answer)
    console.print(table)


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
            f"\n[dim]Query type detected:[/dim] [{color}]{complexity}[/{color}]\n"
        )

        # Run both approaches
        sem_answer, sem_results = run_semantic_section(query, collection, client)
        ag_answer = run_agentic_section(query, collection, client)

        # Side-by-side
        show_comparison(query, complexity, sem_answer, sem_results, ag_answer)

        console.print("\n[dim]Press Enter to ask another question, or type 'quit' to exit.[/dim]")


if __name__ == "__main__":
    main()
