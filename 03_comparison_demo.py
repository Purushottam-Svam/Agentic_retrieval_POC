"""
03_comparison_demo.py
---------------------
Side-by-side comparison of two RAG approaches on the same 4 queries.

APPROACH 1 -- Semantic RAG (single-shot)
  Step 1: One vector search  -> top-4 chunks from ChromaDB
  Step 2: Stuff those chunks into a prompt
  Step 3: ONE GPT-4o call    -> synthesized answer based only on those chunks
  No loop. No tool calls. No follow-up. The model can only work with
  whatever the single search happened to return.

APPROACH 2 -- Agentic Retrieval (multi-step loop)
  Step 1: GPT-4o reads the query and DECIDES what to search for
  Step 2: Calls search_knowledge_base() -- may repeat with different queries
  Step 3: Calls calculate() if arithmetic is needed
  Step 4: Calls get_today_info() if the question involves dates
  Step 5: Reflects -- "do I have everything I need?"
  Step 6: Synthesizes a complete answer with document citations

Both approaches use the SAME model (GPT-4o) and the SAME ChromaDB index.
The ONLY difference is the retrieval strategy.

What the user sees:
  [1] Semantic RAG  -- which docs were fetched + the one-shot answer
  [2] Agentic       -- the live chain of thought (every search, every tool call,
                       every result) + the final synthesized answer
  [3] Comparison    -- both final answers side by side so you can judge quality

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
# Each query is designed to expose a specific weakness of single-shot RAG.
# The "limitation" field describes exactly what semantic RAG will get wrong.

QUERIES = [
    {
        "label": "Q1 -- Simple lookup",
        "complexity": "SIMPLE",
        "complexity_color": "blue",
        "complexity_explanation": (
            "The answer lives entirely in ONE document (POL-002).\n"
            "A single vector search finds it. Both approaches handle this equally.\n"
            "This is the ONE case where single-shot RAG is competitive."
        ),
        "query": "What is the database migration policy?",
        "required_docs": ["POL-002"],
        "limitation": (
            "No real weakness here. The answer is in one doc, one search finds it.\n"
            "Both approaches will give a similar quality answer."
        ),
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
            "Single-shot RAG does ONE search and gets whatever comes back.\n"
            "It cannot use result #1 to decide what to search for next.\n"
            "So it will likely answer 'Sarah Chen' but MISS the email address\n"
            "because TEAM-001 was never retrieved."
        ),
        "query": (
            "Who is responsible for the service that caused the "
            "November 2024 latency incident and what was the root cause?"
        ),
        "required_docs": ["INC-001", "ARCH-001", "TEAM-001"],
        "limitation": (
            "Single-shot RAG will find 'Payment Gateway' and 'Sarah Chen' but\n"
            "will likely MISS the email (sarah.chen@technova.io) because it never\n"
            "searched TEAM-001 -- it had no way to know it needed to."
        ),
    },
    {
        "label": "Q3 -- Aggregation",
        "complexity": "AGGREGATION",
        "complexity_color": "cyan",
        "complexity_explanation": (
            "Needs numbers from 2 documents AND arithmetic:\n"
            "  ARCH-002 (Auth Service)    ->  RTO = 10 minutes\n"
            "  ARCH-001 (Payment Gateway) ->  RTO = 15 minutes\n"
            "  10 + 15 = 25 minutes combined\n\n"
            "Single-shot RAG may retrieve both docs. But when asked to add,\n"
            "the LLM sometimes gets it right by guessing, sometimes just lists\n"
            "the numbers separately without calculating.\n\n"
            "Agentic retrieval explicitly calls calculate(10 + 15) and cites\n"
            "both documents in the answer."
        ),
        "query": (
            "What is the combined recovery time if both the Auth service "
            "and the Payment Gateway fail simultaneously?"
        ),
        "required_docs": ["ARCH-001", "ARCH-002"],
        "limitation": (
            "Single-shot RAG may or may not add the numbers -- it depends on\n"
            "whether both docs were retrieved and whether the LLM does the math.\n"
            "Agentic explicitly calls calculate(10 + 15) = 25, so the answer\n"
            "is always precise and traceable."
        ),
    },
    {
        "label": "Q4 -- Conditional + date",
        "complexity": "CONDITIONAL",
        "complexity_color": "magenta",
        "complexity_explanation": (
            "Needs policy + real-world date awareness + contact lookup:\n"
            "  POL-001  ->  Friday after 15:00 UTC is a deployment freeze\n"
            "  TODAY    ->  Is it Friday? What time is it?\n"
            "  TEAM-001 ->  Who are the approvers and their emails?\n\n"
            "Single-shot RAG finds the policy text but has NO concept of today.\n"
            "It cannot tell you 'yes you can deploy' or 'no, it is after 15:00'.\n"
            "It will quote the rule without applying it to your actual situation.\n\n"
            "Agentic calls get_today_info(), checks the day and time, applies\n"
            "the rule, and gives you a concrete action plan."
        ),
        "query": (
            "We want to deploy a new feature to the Checkout service "
            "this Friday afternoon. What approvals do we need and who "
            "should we contact?"
        ),
        "required_docs": ["POL-001", "TEAM-001"],
        "limitation": (
            "Single-shot RAG will quote the Friday freeze policy but CANNOT\n"
            "tell you whether today/this-Friday is inside the freeze window.\n"
            "Agentic calls get_today_info() and resolves the condition for you."
        ),
    },
]

# System prompt for the single-shot semantic RAG call.
# Deliberately restrictive: the LLM can ONLY use what was retrieved.
# This surfaces the gap caused by incomplete retrieval.
SEMANTIC_RAG_SYSTEM_PROMPT = """You are an enterprise knowledge base assistant for TechNova Inc.

Rules you MUST follow:
1. Answer ONLY using the document chunks provided below the question.
2. Do NOT use any knowledge outside those chunks.
3. If a piece of information is not present in the chunks, explicitly say:
   "This information was not in the retrieved documents."
4. Do NOT search for more information. Do NOT ask follow-up questions.
5. Give one complete, direct answer. Cite the document ID in square brackets, e.g. [INC-001].

This is a SINGLE-SHOT response. You get exactly the documents listed — nothing more."""


# ---------------------------------------------------------------------------
# Semantic RAG: single-shot retrieval + one LLM call
# ---------------------------------------------------------------------------

def run_semantic_rag(
    query: str,
    collection,
    client: openai.OpenAI,
) -> tuple[str, list[dict]]:
    """
    Single-shot RAG pipeline:
      1. One vector search  -> top-4 chunks
      2. Stuff chunks into a prompt as context
      3. One GPT-4o call    -> synthesized answer
      4. Return (answer, results)

    The LLM is explicitly told it cannot search further.
    This surfaces the retrieval gap for complex queries.
    """
    # Step 1: one vector search, take top 4 results
    t0 = time.perf_counter()
    results = semantic_search(collection, query, n_results=4)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    console.print(
        f"  [dim]Retrieved {len(results)} chunks in {retrieval_ms:.0f} ms: "
        f"{', '.join(r['id'] for r in results)}[/dim]"
    )

    # Step 2: build context block from the retrieved chunks
    # Each chunk is labelled with its doc ID and title so the LLM can cite them.
    context_parts = []
    for r in results:
        context_parts.append(
            f"[{r['id']} -- {r['metadata'].get('title', '?')}]\n{r['document']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Step 3: single GPT-4o call -- no tools, no loop
    user_prompt = (
        f"Retrieved document chunks:\n\n{context}\n\n"
        f"Question: {query}"
    )

    t1 = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        # No 'tools' parameter -- this call cannot make tool calls
        messages=[
            {"role": "system", "content": SEMANTIC_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    llm_ms = (time.perf_counter() - t1) * 1000

    answer = response.choices[0].message.content or "(no response)"
    console.print(
        f"  [dim]One GPT-4o call completed in {llm_ms:.0f} ms.[/dim]"
    )

    return answer, results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_semantic_rag_section(q: dict, collection, client) -> tuple[str, list[dict]]:
    """Print the semantic RAG section: retrieved docs + one-shot answer."""
    console.print()
    console.rule("[bold blue]APPROACH 1 -- Semantic RAG (single-shot)[/bold blue]")
    console.print(
        "[dim]One vector search -> chunks stuffed into prompt -> one GPT-4o call -> answer\n"
        "No loop. No tool calls. The model sees only whatever the first search returned.[/dim]\n"
    )

    answer, results = run_semantic_rag(q["query"], collection, client)

    # Show which docs were retrieved vs which were actually needed
    retrieved_ids = [r["id"] for r in results]
    needed_ids = q["required_docs"]
    found = [d for d in needed_ids if d in retrieved_ids]
    missed = [d for d in needed_ids if d not in retrieved_ids]

    retrieval_summary = (
        f"[dim]Docs retrieved:[/dim] [cyan]{', '.join(retrieved_ids)}[/cyan]\n"
        f"[dim]Docs needed:   [/dim] [yellow]{', '.join(needed_ids)}[/yellow]\n"
    )
    if found:
        retrieval_summary += f"[green][+] Found needed docs:[/green] {', '.join(found)}\n"
    if missed:
        retrieval_summary += (
            f"[red][-] Never retrieved:[/red] {', '.join(missed)} "
            f"-- the model had no chance to use these\n"
        )

    console.print(
        Panel(
            retrieval_summary,
            title="[bold blue]What the single search fetched[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )
    )

    console.print(
        Panel(
            answer,
            title="[bold blue]Semantic RAG Answer (one-shot)[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    return answer, results


def show_agentic_section(q: dict, collection, client) -> str:
    """Print the agentic section: live chain of thought + final answer."""
    console.print()
    console.rule("[bold green]APPROACH 2 -- Agentic Retrieval (multi-step loop)[/bold green]")
    console.print(
        "[dim]GPT-4o decides what to search, searches multiple times, uses tools,\n"
        "reflects on what it found, then synthesizes a complete answer.\n"
        "Watch every step below -- this is the reasoning that single-shot RAG does not have.[/dim]\n"
    )

    # run_agentic_query prints each reasoning step, tool call, and tool result live
    answer = run_agentic_query(q["query"], collection, client)
    return answer


def show_final_comparison(q: dict, sem_answer: str, sem_results: list[dict], ag_answer: str):
    """Print both synthesized answers side by side with a verdict."""
    console.print()
    console.rule("[bold white]=== FINAL COMPARISON ===[/bold white]")

    color = q["complexity_color"]

    # Explain what makes this query type hard for single-shot RAG
    console.print(
        Panel(
            f"[bold]Query type:[/bold] {q['complexity']}\n\n"
            f"{q['complexity_explanation']}\n\n"
            f"[bold red]Expected gap:[/bold red]\n{q['limitation']}",
            title=f"[bold {color}]Why does query type matter?[/bold {color}]",
            border_style=color,
            padding=(0, 1),
        )
    )

    console.print()

    # Side-by-side table -- both are synthesized answers now
    table = Table(
        box=box.SIMPLE_HEAD,
        show_lines=True,
        padding=(0, 1),
        width=140,
    )
    table.add_column(
        "[bold blue]SEMANTIC RAG[/bold blue]\n"
        "Single search -> one GPT-4o call\n"
        f"(retrieved: {', '.join(r['id'] for r in sem_results)})",
        ratio=1,
    )
    table.add_column(
        "[bold green]AGENTIC RETRIEVAL[/bold green]\n"
        "Multi-step loop -> tools -> synthesized\n"
        f"(needed: {', '.join(q['required_docs'])})",
        ratio=1,
    )
    table.add_row(sem_answer, ag_answer)
    console.print(table)

    # Verdict based on complexity type
    verdicts = {
        "SIMPLE": (
            "white",
            "[blue]Both approaches competitive here.[/blue]\n\n"
            "The answer is in one document. One search finds it.\n"
            "Both produce similar quality answers.\n"
            "Agentic adds latency without much benefit for simple queries."
        ),
        "MULTI-HOP": (
            "green",
            "[green]Agentic wins.[/green]\n\n"
            "Look for the email address (sarah.chen@technova.io) in both answers.\n"
            "Semantic RAG's single search likely never reached TEAM-001,\n"
            "so it could not include the contact detail.\n"
            "Agentic followed the trail: incident -> service owner -> team directory."
        ),
        "AGGREGATION": (
            "green",
            "[green]Agentic wins on traceability.[/green]\n\n"
            "Semantic RAG may guess '25 minutes' but cannot prove how it got there.\n"
            "Agentic explicitly called calculate(10 + 15) = 25 and cited both\n"
            "ARCH-001 and ARCH-002, so the answer is verifiable step by step."
        ),
        "CONDITIONAL": (
            "green",
            "[green]Agentic wins.[/green]\n\n"
            "Semantic RAG quoted the Friday freeze policy but could not tell you\n"
            "whether THIS Friday afternoon is inside the freeze window.\n"
            "Agentic called get_today_info(), checked the actual day and UTC time,\n"
            "then applied the policy rule to give you a concrete yes/no answer."
        ),
    }
    border_color, verdict_text = verdicts.get(q["complexity"], ("white", ""))

    console.print(
        Panel(
            verdict_text,
            title="[bold]Verdict[/bold]",
            border_style=border_color,
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
                "Add it to a [bold].env[/bold] file:  OPENAI_API_KEY=sk-proj-...",
                title="Missing API Key",
                border_style="red",
            )
        )
        sys.exit(1)

    print_header(
        "Semantic RAG  vs  Agentic Retrieval  --  Side-by-Side",
        "Same model (GPT-4o). Same index (ChromaDB). Different retrieval strategy.",
    )

    console.print(
        Panel(
            "[bold]Both approaches use GPT-4o and the same ChromaDB knowledge base.\n"
            "The ONLY difference is HOW they retrieve:[/bold]\n\n"
            "  [blue]Semantic RAG[/blue]   --  ONE vector search -> chunks stuffed into prompt -> ONE LLM call\n"
            "  [green]Agentic[/green]        --  LLM decides what to search -> loop -> tools -> synthesize\n\n"
            "For simple queries: both give similar answers.\n"
            "For complex queries (multi-hop, aggregation, conditional):\n"
            "  Semantic RAG is limited by what its ONE search happened to return.\n"
            "  Agentic keeps going until it has everything it needs.",
            title="[bold]What you are comparing[/bold]",
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

        # Approach 1: single-shot RAG
        sem_answer, sem_results = show_semantic_rag_section(q, collection, client)

        # Approach 2: agentic loop (prints live chain of thought)
        ag_answer = show_agentic_section(q, collection, client)

        # Side-by-side comparison + verdict
        show_final_comparison(q, sem_answer, sem_results, ag_answer)

        console.print()
        console.print("[dim]" + "-" * 120 + "[/dim]")

    # Final summary
    console.print()
    console.print(
        Panel(
            "[bold]The pattern across all 4 queries:[/bold]\n\n"
            "  SIMPLE      -> both work. Single-shot RAG is faster and cheaper here.\n"
            "  MULTI-HOP   -> semantic RAG misses docs it never knew to search for.\n"
            "                 Agentic follows the chain until the trail is complete.\n"
            "  AGGREGATION -> semantic RAG may guess the math. Agentic proves it\n"
            "                 with an explicit calculate() tool call and citations.\n"
            "  CONDITIONAL -> semantic RAG quotes the rule but cannot apply it.\n"
            "                 Agentic checks today's date and resolves the condition.\n\n"
            "[dim]Use single-shot RAG when queries are simple and speed matters.\n"
            "Use agentic retrieval when answers require reasoning across multiple\n"
            "documents, arithmetic, or real-world context like dates.[/dim]",
            title="[bold]When to use which approach[/bold]",
            border_style="white",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    run_comparison()
