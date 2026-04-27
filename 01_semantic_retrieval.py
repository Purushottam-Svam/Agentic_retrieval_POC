"""
01_semantic_retrieval.py
------------------------
Baseline: pure vector-similarity (semantic) retrieval.

Pipeline:
    Query  ->  embed(query)  ->  cosine-similarity(KB vectors)  ->  top-K chunks  ->  show results

No reasoning. No query decomposition. No follow-up. Just nearest neighbors.

Run:  python 01_semantic_retrieval.py
"""

import sys
import time
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from rich.table import Table
from rich import box

sys.path.insert(0, ".")
from shared.kb import build_knowledge_base, semantic_search
from shared.display import (
    console,
    print_header,
    print_query,
    print_semantic_results,
    print_final_answer,
)

# -- Test queries that expose semantic retrieval's limits ----------------------
QUERIES = [
    {
        "label": "Q1 - Simple (both approaches work)",
        "query": "What is the database migration policy?",
        "expected_docs": ["POL-002"],
    },
    {
        "label": "Q2 - Multi-hop (who owns the service that caused the Nov 2024 incident?)",
        "query": "Who is responsible for the service that caused the November 2024 latency incident, what was the root cause, and what is their contact email?",
        "expected_docs": ["INC-001", "ARCH-001", "TEAM-001"],
    },
    {
        "label": "Q3 - Aggregation (combined RTO if Auth + Payment both fail)",
        "query": "What is the combined recovery time if the Auth service, Payment Gateway, and Checkout service all fail simultaneously?",
        "expected_docs": ["ARCH-001", "ARCH-002", "ARCH-003"],
    },
    {
        "label": "Q4 - Conditional (deploy new feature this Friday, what approvals?)",
        "query": "A developer wants to deploy the Checkout service and run a database migration this Friday at 2pm UTC. What approvals are needed for both, and who should they contact?",
        "expected_docs": ["POL-001", "POL-002", "TEAM-001"],
    },
]


def run_semantic_demo():
    print_header(
        "SEMANTIC RETRIEVAL  (Baseline)",
        "Pure cosine-similarity vector search -- no reasoning, no follow-up",
    )

    collection = build_knowledge_base()

    for q in QUERIES:
        console.print()
        console.rule(f"[bold blue]{q['label']}[/bold blue]")
        print_query(q["query"], mode="SEMANTIC")

        t0 = time.perf_counter()
        results = semantic_search(collection, q["query"], n_results=4)
        elapsed = time.perf_counter() - t0

        print_semantic_results(results, q["query"])

        # Show which expected docs were found vs missed
        found_ids = {r["id"] for r in results}
        expected = set(q["expected_docs"])
        hit = expected & found_ids
        miss = expected - found_ids

        status_lines = []
        if hit:
            status_lines.append(f"[green][+] Found relevant docs:[/green] {', '.join(sorted(hit))}")
        if miss:
            status_lines.append(f"[red][-] Missed relevant docs:[/red] {', '.join(sorted(miss))}")
        status_lines.append(f"[dim]Retrieval time: {elapsed*1000:.1f} ms[/dim]")

        # "Answer" = just first result's snippet, since semantic retrieval doesn't synthesize
        top_snippet = results[0]["document"][:300] if results else "No results."
        console.print(
            Panel(
                f"[dim italic]Semantic retrieval returns chunks, not answers.\n"
                f"Best-match chunk (Doc {results[0]['id'] if results else '?'}):[/dim italic]\n\n"
                f"{top_snippet}...\n\n"
                + "\n".join(status_lines),
                title="[bold blue][doc] Semantic Result[/bold blue]",
                border_style="blue",
            )
        )

    console.print()
    console.rule("[bold]What Semantic Retrieval CANNOT do[/bold]")
    limits = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    limits.add_column(style="red bold", width=4)
    limits.add_column(width=90)
    limits.add_row("[X]", "Connect facts across multiple documents (multi-hop reasoning)")
    limits.add_row("[X]", "Aggregate numbers from separate documents (e.g., sum RTOs)")
    limits.add_row("[X]", "Apply conditional logic (e.g., Friday deployment rules)")
    limits.add_row("[X]", "Ask follow-up questions when initial results are insufficient")
    limits.add_row("[X]", "Synthesize a natural-language answer -- only returns chunks")
    console.print(limits)
    console.print(
        "\n[dim]-> Run [bold]python 02_agentic_retrieval.py[/bold] to see how agentic retrieval handles the same queries.[/dim]\n"
    )


if __name__ == "__main__":
    run_semantic_demo()
