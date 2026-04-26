"""
run_interactive.py  --  THE MAIN DEMO
======================================
You type any question. Both systems answer it. You see the difference yourself.

  Semantic  -->  finds the most similar chunks in the KB, returns raw text
  Agentic   -->  reasons step by step, calls tools, synthesizes a full answer

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
from rich.columns import Columns
from rich.text import Text
from rich import box

import importlib
_ag = importlib.import_module("02_agentic_retrieval")
run_agentic_query = _ag.run_agentic_query
make_openai_client = _ag.make_openai_client

# ---------------------------------------------------------------------------
# Example queries shown to the user on startup
# ---------------------------------------------------------------------------
EXAMPLES = [
    ("1", "SIMPLE",      "What is the database migration policy?"),
    ("2", "MULTI-HOP",   "Who is responsible for the service that caused the November 2024 latency incident and what was the root cause?"),
    ("3", "AGGREGATION", "What is the combined recovery time if both the Auth service and the Payment Gateway fail simultaneously?"),
    ("4", "CONDITIONAL", "We want to deploy a new feature to the Checkout service this Friday afternoon. What approvals do we need and who should we contact?"),
]

COMPLEXITY_COLORS = {
    "SIMPLE":      "blue",
    "MULTI-HOP":   "yellow",
    "AGGREGATION": "cyan",
    "CONDITIONAL": "magenta",
}

COMPLEXITY_WHAT_IT_MEANS = {
    "SIMPLE":
        "Answer lives in a single document. One search is enough.\n"
        "Both systems handle this equally well.",
    "MULTI-HOP":
        "Answer requires chaining facts across 2+ documents.\n"
        "e.g. Incident doc -> service name -> architecture doc -> owner -> team directory -> email.\n"
        "Semantic search cannot use result #1 to shape search #2.\n"
        "Agentic search does -- it keeps searching until the chain is complete.",
    "AGGREGATION":
        "Answer requires collecting numbers from multiple documents and doing math.\n"
        "e.g. Auth RTO (10 min) + Payment RTO (15 min) = 25 min combined.\n"
        "Semantic returns both chunks but cannot add. Agentic uses a calculator tool.",
    "CONDITIONAL":
        "Answer depends on IF/THEN logic combined with real-world state (e.g. today's date).\n"
        "e.g. 'Can we deploy Friday afternoon?' requires: policy lookup + date check + contact list.\n"
        "Semantic finds the policy but has no concept of 'today' or conditional branching.",
}


def show_welcome():
    console.print()
    console.print(Panel(
        "[bold cyan]AGENTIC vs SEMANTIC RETRIEVAL[/bold cyan]  --  Interactive Demo\n\n"
        "Same knowledge base. Same question. Two completely different approaches.\n\n"
        "[blue]Semantic[/blue]  finds the most similar document chunks using vector similarity.\n"
        "           Fast. No reasoning. Returns raw text. Fails on complex questions.\n\n"
        "[green]Agentic[/green]   uses GPT-4o to reason about what to search, searches multiple\n"
        "           times, uses tools (calculator, date lookup), and writes a full answer.\n\n"
        "Type your own question, or pick one of the examples below.",
        border_style="cyan",
        padding=(1, 2),
    ))


def show_examples():
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    table.add_column("Pick", style="bold", width=5)
    table.add_column("Type", width=14)
    table.add_column("Question", width=75)
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
    """Rough heuristic to label the query type for display."""
    q = query.lower()
    if any(w in q for w in ["combined", "total", "sum", "both", "simultaneously"]):
        return "AGGREGATION"
    if any(w in q for w in ["friday", "monday", "deploy", "weekend", "today", "this week"]):
        return "CONDITIONAL"
    if any(w in q for w in ["who", "responsible", "caused", "led to", "owner", "team lead"]):
        return "MULTI-HOP"
    return "SIMPLE"


def run_semantic_section(query: str, collection) -> tuple[str, list[dict]]:
    """Run semantic search and print results. Returns (combined_text, results)."""
    console.print()
    console.rule("[bold blue]SEMANTIC RETRIEVAL[/bold blue]")
    console.print("[dim]One-shot vector similarity search -- no reasoning, no follow-up[/dim]\n")

    t0 = time.perf_counter()
    results = semantic_search(collection, query, n_results=4)
    elapsed = (time.perf_counter() - t0) * 1000

    if not results:
        console.print("[red]No results found.[/red]")
        return "", []

    # Show each retrieved doc
    for i, r in enumerate(results, 1):
        sim = r["similarity"]
        sim_color = "green" if sim > 0.65 else ("yellow" if sim > 0.45 else "red")
        snippet = r["document"][:300].replace("\n", " ")
        console.print(Panel(
            f"[dim]Doc ID:[/dim]     [cyan]{r['id']}[/cyan]\n"
            f"[dim]Title:[/dim]      {r['metadata'].get('title', '?')}\n"
            f"[dim]Similarity:[/dim] [{sim_color}]{sim:.4f}[/{sim_color}]  "
            f"({'high match' if sim > 0.65 else 'medium match' if sim > 0.45 else 'low match'})\n\n"
            f"{snippet}...",
            title=f"[bold blue]Result #{i}[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        ))

    # What semantic gives you as an "answer"
    top = results[0]
    console.print(Panel(
        f"[bold]Semantic cannot synthesize.[/bold] The best it can do is return the top chunk:\n\n"
        f"[italic]{top['document'][:400]}...[/italic]\n\n"
        f"[dim]Retrieved {len(results)} chunks in {elapsed:.0f} ms.\n"
        f"If the answer spans multiple docs, the user has to connect the dots manually.[/dim]",
        title="[bold blue]What Semantic Returns as 'Answer'[/bold blue]",
        border_style="blue",
        padding=(0, 1),
    ))

    combined = " ".join(r["document"] for r in results)
    return combined, results


def run_agentic_section(query: str, collection, client) -> str:
    """Run agentic retrieval and print live. Returns final answer."""
    console.print()
    console.rule("[bold green]AGENTIC RETRIEVAL[/bold green]")
    console.print("[dim]GPT-4o reasons, decomposes, searches multiple times, uses tools, synthesizes[/dim]\n")

    answer = run_agentic_query(query, collection, client)
    return answer


def show_comparison(query: str, complexity: str, sem_results: list[dict], ag_answer: str):
    """Print a clear side-by-side comparison at the end."""
    console.print()
    console.rule("[bold white]SIDE-BY-SIDE COMPARISON[/bold white]")

    color = COMPLEXITY_COLORS.get(complexity, "white")

    # Explain what this complexity type means
    console.print(Panel(
        COMPLEXITY_WHAT_IT_MEANS[complexity],
        title=f"[bold {color}]Why this query is '{complexity}'[/bold {color}]",
        border_style=color,
        padding=(0, 1),
    ))

    console.print()

    # Left: what semantic gave
    sem_doc_ids = [r["id"] for r in sem_results]
    sem_top_text = sem_results[0]["document"][:350] if sem_results else "No results."

    # Right: what agentic gave
    ag_text = ag_answer[:500] if ag_answer else "No answer."

    table = Table(box=box.SIMPLE_HEAD, show_lines=True, padding=(0, 1), width=120)
    table.add_column("[bold blue]SEMANTIC[/bold blue]\n(returns chunks, no synthesis)", ratio=1)
    table.add_column("[bold green]AGENTIC[/bold green]\n(synthesized answer)", ratio=1)

    table.add_row(
        f"Docs retrieved: {', '.join(sem_doc_ids)}\n\n{sem_top_text}...",
        ag_text + ("..." if len(ag_answer) > 500 else ""),
    )
    console.print(table)

    # Verdict
    if complexity == "SIMPLE":
        verdict = (
            "[blue]For this query type, both work.[/blue]\n"
            "The question has one answer in one document.\n"
            "Agentic adds reasoning overhead without much benefit here."
        )
    else:
        verdict = (
            f"[green]Agentic wins on this '{complexity}' query.[/green]\n"
            "Semantic returned raw chunks. A human would need to read, connect,\n"
            "and reason across them to get the actual answer.\n"
            "Agentic did all of that automatically and gave you a complete answer."
        )
    console.print(Panel(verdict, title="[bold]Verdict[/bold]", border_style="white", padding=(0, 1)))


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

    console.print("[dim]Building knowledge base (first time downloads ~80MB model)...[/dim]")
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
            f"[dim](affects how much agentic will outperform semantic)[/dim]\n"
        )

        # Run both
        sem_text, sem_results = run_semantic_section(query, collection)
        ag_answer = run_agentic_section(query, collection, client)

        # Compare
        show_comparison(query, complexity, sem_results, ag_answer)

        console.print("\n[dim]Press Enter to ask another question, or type 'quit' to exit.[/dim]")


if __name__ == "__main__":
    main()
