"""
03_comparison_demo.py
---------------------
Runs BOTH semantic and agentic retrieval on the same 4 queries.

What you see for each query
----------------------------
  [1] SEMANTIC  -- the actual text of every retrieved document chunk,
                   so you can read what the system found and judge
                   whether it could ever answer the question on its own.

  [2] AGENTIC   -- the LIVE chain of thought:
                     * every reasoning step the agent prints
                     * every tool call with its arguments
                     * every tool result that comes back
                   Exactly what happens inside the agent loop, step by step.

  [3] FINAL COMPARISON -- both outputs side by side so you can compare
                          the synthesized agentic answer against the raw
                          semantic chunks at a glance.

Run:  python 03_comparison_demo.py
"""

import sys
import time
import os

from dotenv import load_dotenv
load_dotenv()

import openai
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.columns import Columns
from rich.text import Text
from rich import box

sys.path.insert(0, ".")
from shared.kb import build_knowledge_base, semantic_search
from shared.display import console, print_header
import importlib
_ag = importlib.import_module("02_agentic_retrieval")
run_agentic_query = _ag.run_agentic_query
make_openai_client = _ag.make_openai_client


# ---------------------------------------------------------------------------
# Query definitions
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "label": "Q1 -- Simple lookup",
        "complexity": "SIMPLE",
        "complexity_color": "blue",
        "complexity_explanation": (
            "The answer lives entirely in ONE document (POL-002).\n"
            "One vector search is enough. Both systems handle this well."
        ),
        "query": "What is the database migration policy?",
        "required_docs": ["POL-002"],
    },
    {
        "label": "Q2 -- Multi-hop",
        "complexity": "MULTI-HOP",
        "complexity_color": "yellow",
        "complexity_explanation": (
            "Needs 3 documents chained together:\n"
            "  INC-001  ->  'Payment Gateway caused the incident'\n"
            "  ARCH-001 ->  'Payment Gateway is owned by Sarah Chen'\n"
            "  TEAM-001 ->  'sarah.chen@technova.io'\n\n"
            "Semantic search treats each search independently.\n"
            "It cannot use result #1 to decide what to search for next.\n"
            "The agentic system does -- it keeps going until the chain is complete."
        ),
        "query": (
            "Who is responsible for the service that caused the "
            "November 2024 latency incident and what was the root cause?"
        ),
        "required_docs": ["INC-001", "ARCH-001", "TEAM-001"],
    },
    {
        "label": "Q3 -- Aggregation",
        "complexity": "AGGREGATION",
        "complexity_color": "cyan",
        "complexity_explanation": (
            "Needs numbers from 2 separate documents AND arithmetic:\n"
            "  ARCH-002 (Auth Service)    ->  RTO = 10 minutes\n"
            "  ARCH-001 (Payment Gateway) ->  RTO = 15 minutes\n"
            "  10 + 15 = 25 minutes combined\n\n"
            "Semantic returns both docs but CANNOT add the numbers.\n"
            "The user would have to do the math manually.\n"
            "The agentic system calls the 'calculate' tool to do it."
        ),
        "query": (
            "What is the combined recovery time if both the Auth service "
            "and the Payment Gateway fail simultaneously?"
        ),
        "required_docs": ["ARCH-001", "ARCH-002"],
    },
    {
        "label": "Q4 -- Conditional + date",
        "complexity": "CONDITIONAL",
        "complexity_color": "magenta",
        "complexity_explanation": (
            "Needs policy + real-world date awareness + contact lookup:\n"
            "  POL-001  ->  Friday after 15:00 UTC is a deployment freeze\n"
            "  TODAY    ->  Is it Friday? What time is it?\n"
            "  TEAM-001 ->  Who are the approvers and what are their emails?\n\n"
            "Semantic search finds the policy text but has NO concept of 'today'.\n"
            "It cannot resolve a conditional rule into a concrete yes/no answer.\n"
            "The agentic system calls 'get_today_info' then applies the policy logic."
        ),
        "query": (
            "We want to deploy a new feature to the Checkout service "
            "this Friday afternoon. What approvals do we need and who "
            "should we contact?"
        ),
        "required_docs": ["POL-001", "TEAM-001"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def show_semantic_section(q: dict, collection) -> tuple[str, list[dict]]:
    """
    Run semantic search and print every retrieved chunk in a readable panel.
    Returns (combined_text, results_list).
    """
    console.print()
    console.rule("[bold blue]SEMANTIC RETRIEVAL[/bold blue]")
    console.print(
        "[dim]One-shot vector similarity -- no reasoning, no follow-up searches[/dim]\n"
    )

    t0 = time.perf_counter()
    results = semantic_search(collection, q["query"], n_results=4)
    elapsed = (time.perf_counter() - t0) * 1000

    if not results:
        console.print("[red]No results found.[/red]")
        return "", []

    console.print(
        f"[blue]Retrieved {len(results)} document chunks in {elapsed:.0f} ms.[/blue]\n"
        f"[dim]Docs needed to answer correctly: "
        f"[yellow]{', '.join(q['required_docs'])}[/yellow][/dim]\n"
    )

    for i, r in enumerate(results, 1):
        sim = r["similarity"]
        sim_color = "green" if sim > 0.65 else ("yellow" if sim > 0.45 else "red")
        # Show full document text, not just a snippet
        doc_text = r["document"]
        console.print(
            Panel(
                f"[dim]Doc ID:[/dim]     [cyan]{r['id']}[/cyan]\n"
                f"[dim]Title:[/dim]      {r['metadata'].get('title', '?')}\n"
                f"[dim]Category:[/dim]   {r['metadata'].get('category', '?')}\n"
                f"[dim]Similarity:[/dim] [{sim_color}]{sim:.4f}[/{sim_color}]"
                f"  ({'high match' if sim > 0.65 else 'medium match' if sim > 0.45 else 'low match'})\n\n"
                f"{doc_text}",
                title=f"[bold blue]Chunk #{i}[/bold blue]",
                border_style="blue",
                padding=(0, 1),
            )
        )

    console.print(
        Panel(
            "[bold]Semantic cannot synthesize or reason.[/bold]\n\n"
            "It returned the chunks above. If the answer requires:\n"
            "  - connecting information across multiple docs   -> user has to do it manually\n"
            "  - adding numbers                               -> user has to do it manually\n"
            "  - knowing today's date to apply a policy rule  -> not possible at all",
            title="[bold blue]What Semantic CAN and CANNOT do[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )
    )

    combined = " ".join(r["document"] for r in results)
    return combined, results


def show_agentic_section(q: dict, collection, client) -> str:
    """
    Run agentic retrieval with live chain-of-thought output.
    Returns final answer string.
    """
    console.print()
    console.rule("[bold green]AGENTIC RETRIEVAL -- LIVE CHAIN OF THOUGHT[/bold green]")
    console.print(
        "[dim]GPT-4o reasons step by step, decides what to search, "
        "calls tools, reflects, synthesizes[/dim]\n"
    )
    console.print(
        "[dim]Watch each step below -- this is the reasoning that "
        "semantic search does NOT have:[/dim]\n"
    )

    answer = run_agentic_query(q["query"], collection, client)
    return answer


def show_final_comparison(q: dict, sem_results: list[dict], ag_answer: str):
    """Print both outputs side by side for easy comparison."""
    console.print()
    console.rule("[bold white]=== FINAL COMPARISON ===[/bold white]")

    color = q["complexity_color"]

    # Explain what this query type means
    console.print(
        Panel(
            q["complexity_explanation"],
            title=f"[bold {color}]Why this is a '{q['complexity']}' query[/bold {color}]",
            border_style=color,
            padding=(0, 1),
        )
    )
    console.print()

    # Semantic side -- show top 2 chunks
    sem_ids = [r["id"] for r in sem_results]
    sem_chunks_text = ""
    for i, r in enumerate(sem_results[:2], 1):
        sem_chunks_text += f"[cyan]{r['id']}[/cyan] ({r['metadata'].get('title', '?')}):\n"
        sem_chunks_text += r["document"][:300]
        if len(r["document"]) > 300:
            sem_chunks_text += "..."
        sem_chunks_text += "\n\n"
    if not sem_chunks_text:
        sem_chunks_text = "No results."

    sem_panel_content = (
        f"[dim]Docs found: {', '.join(sem_ids)}[/dim]\n"
        f"[dim]Needed:     {', '.join(q['required_docs'])}[/dim]\n\n"
        f"{sem_chunks_text}"
        f"[dim](Showing top 2 of {len(sem_results)} chunks.)\n"
        f"The user would need to read all chunks and reason across them manually.[/dim]"
    )

    # Agentic side -- the synthesized answer
    ag_panel_content = ag_answer

    # Print as a side-by-side table
    table = Table(
        box=box.SIMPLE_HEAD,
        show_lines=True,
        padding=(0, 1),
        width=130,
    )
    table.add_column(
        "[bold blue]SEMANTIC RETRIEVAL[/bold blue]\n"
        "(raw chunks, no synthesis)",
        ratio=1,
    )
    table.add_column(
        "[bold green]AGENTIC RETRIEVAL[/bold green]\n"
        "(synthesized, cited answer)",
        ratio=1,
    )
    table.add_row(sem_panel_content, ag_panel_content)
    console.print(table)

    # Verdict
    if q["complexity"] == "SIMPLE":
        verdict = (
            "[blue]Both systems work for this query.[/blue]\n\n"
            "The answer is in a single document, so one vector search finds it.\n"
            "Agentic adds reasoning overhead here without much extra benefit.\n"
            "This is the ONE case where semantic is competitive."
        )
    elif q["complexity"] == "MULTI-HOP":
        verdict = (
            "[green]Agentic wins.[/green]\n\n"
            "Semantic returned chunks from separate docs. A human would need to:\n"
            "  1. Read INC-001 to find 'Payment Gateway'\n"
            "  2. Know to then search ARCH-001 for the owner\n"
            "  3. Know to then search TEAM-001 for the email\n\n"
            "The agent did all three searches automatically by following the trail."
        )
    elif q["complexity"] == "AGGREGATION":
        verdict = (
            "[green]Agentic wins.[/green]\n\n"
            "Semantic retrieved both RTO values but cannot add them.\n"
            "The agent found '10 minutes' and '15 minutes', called calculate(10+15),\n"
            "and gave you '25 minutes combined' directly in the answer."
        )
    else:
        verdict = (
            "[green]Agentic wins.[/green]\n\n"
            "Semantic found the policy text but cannot tell you whether\n"
            "this specific Friday afternoon is inside the freeze window.\n"
            "The agent called get_today_info(), applied the policy rule,\n"
            "looked up the approvers, and gave you a concrete action plan."
        )

    console.print(
        Panel(
            verdict,
            title="[bold]Verdict[/bold]",
            border_style="white",
            padding=(0, 1),
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print(
            Panel(
                "[red]OPENAI_API_KEY not set.[/red]\n"
                "Add it to a [bold].env[/bold] file:  OPENAI_API_KEY=sk-proj-...\n\n"
                "You can still run [bold]python 01_semantic_retrieval.py[/bold] without a key.",
                title="Missing API Key",
                border_style="red",
            )
        )
        sys.exit(1)

    print_header(
        "AGENTIC vs SEMANTIC  --  Side-by-Side Demo",
        "Watch the agentic chain of thought, then compare both outputs",
    )

    console.print(
        Panel(
            "[bold]For each query you will see:[/bold]\n\n"
            "  [blue]1. SEMANTIC[/blue]  -- Every document chunk the vector search returned,\n"
            "               printed in full so you can read what it found.\n\n"
            "  [green]2. AGENTIC[/green]  -- The LIVE chain of thought: every reasoning step,\n"
            "               every tool call (with arguments), every tool result.\n"
            "               This is what semantic search does not have.\n\n"
            "  [white]3. COMPARISON[/white] -- Both final outputs side by side, with a verdict.",
            title="[bold]What you are about to see[/bold]",
            border_style="white",
            padding=(0, 1),
        )
    )

    console.print("\n[dim]Building knowledge base...[/dim]")
    collection = build_knowledge_base()
    client = make_openai_client()

    for q in QUERIES:
        console.print()
        console.print()
        console.rule(f"[bold white]{q['label']}[/bold white]")
        console.print(f"\n[yellow]Query:[/yellow] {q['query']}\n")

        # 1. Semantic
        sem_text, sem_results = show_semantic_section(q, collection)

        # 2. Agentic (live chain of thought prints inside run_agentic_query)
        ag_answer = show_agentic_section(q, collection, client)

        # 3. Side-by-side comparison + verdict
        show_final_comparison(q, sem_results, ag_answer)

        console.print()
        console.print("[dim]" + "-" * 100 + "[/dim]")

    console.print()
    console.print(
        Panel(
            "[bold]The pattern you just saw:[/bold]\n\n"
            "  SIMPLE queries      -> both systems competitive\n"
            "  MULTI-HOP queries   -> semantic misses the chain; agentic follows it\n"
            "  AGGREGATION queries -> semantic returns numbers; agentic adds them\n"
            "  CONDITIONAL queries -> semantic finds the rule; agentic applies it\n\n"
            "[dim]The agentic system is slower (multiple LLM calls + tool round-trips)\n"
            "but it actually answers the question rather than returning raw chunks.[/dim]",
            title="[bold]Summary[/bold]",
            border_style="white",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    run_comparison()
