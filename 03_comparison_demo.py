"""
03_comparison_demo.py
---------------------
Side-by-side comparison of two RAG approaches on the same 4 queries.

APPROACH 1 -- Semantic RAG (single-shot retrieval, fair tool access)
  Step 1: ONE fixed vector search  -> top-4 chunks from ChromaDB
  Step 2: Chunks stuffed into a prompt as context
  Step 3: GPT-4o call WITH calculate + get_today_info tools
          -> can do math, can check today's date
          -> CANNOT search again (search_knowledge_base not in its tools)
  The model is limited by whatever that one search returned.

APPROACH 2 -- Agentic Retrieval (multi-step loop, all tools)
  Step 1: GPT-4o reads the query and DECIDES what to search for
  Step 2: Calls search_knowledge_base() -- repeats with different queries
  Step 3: Calls calculate() if arithmetic is needed
  Step 4: Calls get_today_info() if the question involves dates
  Step 5: Reflects -- "do I have everything I need?" -- searches more if not
  Step 6: Synthesizes a complete answer with document citations

WHY THIS IS THE FAIR COMPARISON:
  Both approaches use the SAME model (GPT-4o), SAME ChromaDB index,
  and the SAME utility tools (calculate, get_today_info).

  The ONLY difference is search_knowledge_base:
    Semantic RAG  -> ONE fixed search, decided before the LLM runs
    Agentic       -> LLM decides what to search, when, and how many times

  Giving search_knowledge_base to semantic RAG in a loop would make it
  agentic retrieval -- that tool IS what defines the agentic approach.

What the user sees:
  [1] Semantic RAG  -- docs retrieved + any tool calls it made + its answer
  [2] Agentic       -- live chain of thought (every search, tool call, result)
                       + final synthesized answer
  [3] Comparison    -- both answers side by side + verdict on what differs

Run:  python 03_comparison_demo.py
"""

import sys
import json
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

# Import from 02_agentic_retrieval (numeric filename -- needs importlib)
_ag = importlib.import_module("02_agentic_retrieval")
run_agentic_query = _ag.run_agentic_query
make_openai_client = _ag.make_openai_client
execute_tool = _ag.execute_tool   # reuse the same tool executor (calculate, get_today_info)


# ---------------------------------------------------------------------------
# Query definitions
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "label": "Q1 -- Simple lookup",
        "complexity": "SIMPLE",
        "complexity_color": "blue",
        "complexity_explanation": (
            "Answer lives in ONE document (POL-002).\n"
            "One search finds it. No tools needed. Both approaches are equivalent."
        ),
        "query": "What is the database migration policy?",
        "required_docs": ["POL-002"],
        "expected_gap": (
            "No meaningful gap. Both approaches give a similar answer.\n"
            "Agentic adds latency without benefit for simple lookups."
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
            "Semantic RAG does ONE search and stops -- no tool can fix a missing\n"
            "document that was never retrieved. If TEAM-001 was not in the top-4\n"
            "results, the email is simply not available to the model.\n\n"
            "Agentic follows the chain: reads INC-001, searches for the service\n"
            "owner, reads ARCH-001, searches for the contact, reads TEAM-001."
        ),
        "query": (
            "Who is responsible for the service that caused the "
            "November 2024 latency incident, what was the root cause, "
            "and what is their contact email?"
        ),
        "required_docs": ["INC-001", "ARCH-001", "TEAM-001"],
        "expected_gap": (
            "CLEAREST GAP -- both have the same tools, the only difference is retrieval.\n"
            "Semantic RAG answer will likely MISS sarah.chen@technova.io because\n"
            "TEAM-001 was probably not in the top-4 results for this query.\n"
            "Agentic will have the full chain including the email."
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
            "Both approaches now have the calculate tool.\n"
            "If semantic RAG's one search returns both ARCH-001 and ARCH-002\n"
            "(likely, since both match the query), it can get the right answer.\n"
            "The gap here is smaller -- it's mainly about citation traceability."
        ),
        "query": (
            "What is the combined recovery time if both the Auth service "
            "and the Payment Gateway fail simultaneously?"
        ),
        "required_docs": ["ARCH-001", "ARCH-002"],
        "expected_gap": (
            "GAP NARROWS -- both have calculate tool.\n"
            "If both docs were retrieved, semantic RAG may give the correct answer.\n"
            "Agentic is more reliable because it explicitly searches for each service\n"
            "separately and cites both source documents."
        ),
    },
    {
        "label": "Q4 -- Conditional + date",
        "complexity": "CONDITIONAL",
        "complexity_color": "magenta",
        "complexity_explanation": (
            "Needs policy + date check + contact lookup:\n"
            "  POL-001  ->  Friday after 15:00 UTC is a deployment freeze\n"
            "  TODAY    ->  Is it Friday? What time is it?\n"
            "  TEAM-001 ->  Who are the approvers and their emails?\n\n"
            "Both approaches now have get_today_info.\n"
            "Semantic RAG can check the date -- but only if it retrieved POL-001\n"
            "AND TEAM-001 in its one search. TEAM-001 might have been missed.\n"
            "Agentic will search for the policy, then separately search for the\n"
            "approvers, ensuring nothing is missed."
        ),
        "query": (
            "We want to deploy a new feature to the Checkout service "
            "this Friday afternoon. What approvals do we need and who "
            "should we contact?"
        ),
        "required_docs": ["POL-001", "TEAM-001"],
        "expected_gap": (
            "GAP PARTIALLY NARROWS -- both have get_today_info.\n"
            "Semantic RAG can now resolve the date condition.\n"
            "But if TEAM-001 was not retrieved, it still cannot name the approvers.\n"
            "Agentic searches for both the policy and the team separately."
        ),
    },
]

# ---------------------------------------------------------------------------
# Tool definitions for semantic RAG
# ---------------------------------------------------------------------------
# Semantic RAG gets the SAME utility tools as agentic (calculate, get_today_info)
# but NOT search_knowledge_base -- giving it that tool in a loop would make
# it agentic retrieval by definition.
SEMANTIC_RAG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a simple arithmetic expression and return the result. "
                "Use this when you need to add, subtract, multiply, or divide numbers "
                "found in the document chunks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A safe arithmetic expression, e.g. '15 + 10' or '(22 + 10) / 60'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_today_info",
            "description": "Returns today's date, day of week, and time in UTC. Use this when the question involves dates, days, or time-sensitive policies.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # NOTE: search_knowledge_base is intentionally NOT here.
    # Semantic RAG gets ONE fixed retrieval pass before the LLM runs.
    # Adding search_knowledge_base here would turn it into agentic retrieval.
]

# System prompt for semantic RAG.
# The model knows it has calculate and get_today_info but NOT search.
SEMANTIC_RAG_SYSTEM_PROMPT = """You are an enterprise knowledge base assistant for TechNova Inc.

You have been given a set of retrieved document chunks to answer a question.

Rules:
1. Answer ONLY using the document chunks provided. Do not use outside knowledge.
2. If information is missing from the chunks, explicitly say so -- do not guess.
3. You may call calculate() if arithmetic is needed on values found in the chunks.
4. You may call get_today_info() if the question involves dates or time-sensitive policies.
5. You do NOT have a search tool. You cannot retrieve more documents.
   Work only with what was provided.
6. Cite the document ID in square brackets, e.g. [INC-001], for every fact you state.
7. Give one complete, direct answer."""


# ---------------------------------------------------------------------------
# Semantic RAG runner
# ---------------------------------------------------------------------------

def run_semantic_rag(
    query: str,
    collection,
    client: openai.OpenAI,
) -> tuple[str, list[dict]]:
    """
    Single-shot RAG with fair tool access:
      1. ONE vector search -> top-4 chunks (retrieval is fixed, not LLM-driven)
      2. Chunks stuffed into a prompt as context
      3. GPT-4o call WITH calculate + get_today_info (same as agentic, fair)
         but WITHOUT search_knowledge_base (that would make it agentic)
      4. Small loop only to handle utility tool call responses
      5. Return synthesized answer

    The model can do math and check dates but cannot search for more documents.
    This isolates the retrieval strategy as the ONLY variable between approaches.
    """
    # ── Step 1: ONE fixed vector search ──────────────────────────────────────
    t0 = time.perf_counter()
    results = semantic_search(collection, query, n_results=4)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    retrieved_ids = [r["id"] for r in results]
    console.print(
        f"  [dim]Vector search: {len(results)} chunks in {retrieval_ms:.0f} ms "
        f"-> {', '.join(retrieved_ids)}[/dim]"
    )

    # ── Step 2: build context from retrieved chunks ───────────────────────────
    context_parts = []
    for r in results:
        context_parts.append(
            f"[{r['id']} -- {r['metadata'].get('title', '?')}]\n{r['document']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ── Step 3: call GPT-4o with utility tools (no search) ───────────────────
    messages = [
        {"role": "system", "content": SEMANTIC_RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Retrieved document chunks:\n\n{context}\n\n"
                f"Question: {query}"
            ),
        },
    ]

    tool_calls_log = []   # track what tools semantic RAG called (for display)

    # Small loop -- ONLY to handle calculate / get_today_info responses.
    # This is NOT a retrieval loop. The model cannot call search_knowledge_base.
    while True:
        t1 = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            tools=SEMANTIC_RAG_TOOLS,
            messages=messages,
        )
        llm_ms = (time.perf_counter() - t1) * 1000

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        tool_calls = msg.tool_calls or []

        # No more tool calls -- we have the final answer
        if finish_reason == "stop" or not tool_calls:
            answer = msg.content or "(no response)"
            console.print(f"  [dim]GPT-4o call: {llm_ms:.0f} ms | "
                          f"utility tool calls made: {len(tool_calls_log)}[/dim]")
            return answer, results

        # Execute each utility tool call and feed result back
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            tool_input = json.loads(tc.function.arguments)
            result_str = execute_tool(tc.function.name, tool_input, collection)
            tool_calls_log.append((tc.function.name, tool_input, result_str))

            # Show each utility tool call inline so the user can see them
            args_display = ", ".join(f"{k}={v}" for k, v in tool_input.items())
            console.print(
                f"  [bold blue]>> TOOL[/bold blue] [cyan]{tc.function.name}[/cyan]"
                f"({args_display})"
                f"  [dim]-> {result_str[:80]}[/dim]"
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_semantic_rag_section(q: dict, collection, client) -> tuple[str, list[dict]]:
    """Run single-shot RAG and print the retrieved docs + answer."""
    console.print()
    console.rule("[bold blue]APPROACH 1 -- Semantic RAG[/bold blue]")
    console.print(
        "[dim]ONE vector search -> chunks into prompt -> GPT-4o with calculate + get_today_info\n"
        "No search loop. Cannot retrieve more documents. Limited to what the first search found.[/dim]\n"
    )

    answer, results = run_semantic_rag(q["query"], collection, client)

    # Show which needed docs were retrieved vs missed
    retrieved_ids = [r["id"] for r in results]
    needed = q["required_docs"]
    found = [d for d in needed if d in retrieved_ids]
    missed = [d for d in needed if d not in retrieved_ids]

    doc_summary = (
        f"[dim]Docs retrieved:[/dim]  [cyan]{', '.join(retrieved_ids)}[/cyan]\n"
        f"[dim]Docs needed:   [/dim]  [yellow]{', '.join(needed)}[/yellow]\n"
    )
    if found:
        doc_summary += f"[green][+] Needed docs found:[/green] {', '.join(found)}\n"
    if missed:
        doc_summary += (
            f"[red][-] Never retrieved: {', '.join(missed)}[/red]  "
            f"-- model had no chance to use these\n"
        )

    console.print(Panel(
        doc_summary,
        title="[bold blue]What the single search fetched[/bold blue]",
        border_style="blue",
        padding=(0, 1),
    ))

    console.print(Panel(
        answer,
        title="[bold blue]Semantic RAG Answer[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))

    return answer, results


def show_agentic_section(q: dict, collection, client) -> str:
    """Run agentic retrieval -- live chain of thought prints inside run_agentic_query."""
    console.print()
    console.rule("[bold green]APPROACH 2 -- Agentic Retrieval[/bold green]")
    console.print(
        "[dim]GPT-4o in a loop: decides what to search, searches multiple times,\n"
        "uses calculate + get_today_info + search_knowledge_base, reflects, synthesizes.\n"
        "Watch every step below:[/dim]\n"
    )
    return run_agentic_query(q["query"], collection, client)


def show_final_comparison(
    q: dict,
    sem_answer: str,
    sem_results: list[dict],
    ag_answer: str,
):
    """Print both synthesized answers side by side with an honest verdict."""
    console.print()
    console.rule("[bold white]=== FINAL COMPARISON ===[/bold white]")

    color = q["complexity_color"]

    # Context panel: what this query type means and what gap to expect
    console.print(Panel(
        f"[bold]Query type:[/bold]  {q['complexity']}\n\n"
        f"{q['complexity_explanation']}\n\n"
        f"[bold]Expected gap:[/bold]\n{q['expected_gap']}",
        title=f"[bold {color}]What to look for in the answers[/bold {color}]",
        border_style=color,
        padding=(0, 1),
    ))
    console.print()

    # Side-by-side: both are synthesized GPT-4o answers
    retrieved_ids = ", ".join(r["id"] for r in sem_results)
    table = Table(
        box=box.SIMPLE_HEAD,
        show_lines=True,
        padding=(0, 1),
        width=140,
    )
    table.add_column(
        "[bold blue]SEMANTIC RAG[/bold blue]\n"
        "1 search -> GPT-4o + calculate + get_today_info\n"
        f"Retrieved: {retrieved_ids}",
        ratio=1,
    )
    table.add_column(
        "[bold green]AGENTIC RETRIEVAL[/bold green]\n"
        "Loop -> search + calculate + get_today_info\n"
        f"Needed: {', '.join(q['required_docs'])}",
        ratio=1,
    )
    table.add_row(sem_answer, ag_answer)
    console.print(table)

    # Verdict -- honest about where gap remains and where it closes
    verdicts = {
        "SIMPLE": (
            "white",
            "[blue]No meaningful gap.[/blue]\n\n"
            "Both approaches give equivalent answers for simple lookups.\n"
            "Semantic RAG is faster and cheaper here. Use it."
        ),
        "MULTI-HOP": (
            "green",
            "[green]Clear gap -- this is the core agentic advantage.[/green]\n\n"
            "Both had the same utility tools. The ONLY difference was retrieval.\n"
            "Look for sarah.chen@technova.io in both answers.\n"
            "Semantic RAG did one search -- if TEAM-001 wasn't in the top-4, the email\n"
            "was simply unavailable, no matter how smart the model is.\n"
            "Agentic followed the chain: incident -> service -> owner -> directory -> email."
        ),
        "AGGREGATION": (
            "cyan",
            "[cyan]Gap narrows -- both had the calculate tool.[/cyan]\n\n"
            "If semantic RAG retrieved both ARCH-001 and ARCH-002, it likely got the\n"
            "right answer (25 minutes). Agentic is more reliable because it searched\n"
            "for each service explicitly and always cites both source documents.\n"
            "For aggregation, agentic wins on reliability and traceability, not always answer."
        ),
        "CONDITIONAL": (
            "magenta",
            "[magenta]Gap partially narrows -- both had get_today_info.[/magenta]\n\n"
            "Semantic RAG can now check today's date and apply the policy rule.\n"
            "The remaining gap: if TEAM-001 was not in the top-4 results, semantic RAG\n"
            "cannot name the approvers or give their contact details.\n"
            "Agentic searched for the policy, then searched for the team separately."
        ),
    }
    border_color, verdict_text = verdicts.get(q["complexity"], ("white", ""))
    console.print(Panel(
        verdict_text,
        title="[bold]Verdict[/bold]",
        border_style=border_color,
        padding=(0, 1),
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print(Panel(
            "[red]OPENAI_API_KEY not set.[/red]\n"
            "Add it to a [bold].env[/bold] file:  OPENAI_API_KEY=sk-proj-...",
            title="Missing API Key",
            border_style="red",
        ))
        sys.exit(1)

    print_header(
        "Semantic RAG  vs  Agentic Retrieval  --  Fair Comparison",
        "Same model. Same index. Same utility tools. Only difference: retrieval strategy.",
    )

    console.print(Panel(
        "[bold]What makes this comparison fair:[/bold]\n\n"
        "  Both use GPT-4o\n"
        "  Both use the same ChromaDB knowledge base\n"
        "  Both have: [cyan]calculate[/cyan] + [cyan]get_today_info[/cyan] tools\n\n"
        "  The ONLY difference:\n"
        "    [blue]Semantic RAG[/blue]  ->  ONE fixed vector search before the LLM runs.\n"
        "                     Cannot search again. Limited by what that search returned.\n"
        "    [green]Agentic[/green]       ->  LLM decides what to search, when, and how many times.\n"
        "                     Keeps searching until it has everything it needs.\n\n"
        "  [dim]Giving Semantic RAG the search tool in a loop would make it agentic.\n"
        "  That is the exact line between the two approaches.[/dim]",
        title="[bold]What you are comparing[/bold]",
        border_style="white",
        padding=(0, 1),
    ))

    console.print("\n[dim]Building knowledge base...[/dim]")
    collection = build_knowledge_base()
    client = make_openai_client()

    for q in QUERIES:
        console.print()
        console.print()
        console.rule(f"[bold white]{q['label']}[/bold white]")
        console.print(f"\n[yellow]Query:[/yellow] {q['query']}\n")

        # Approach 1: single-shot RAG with fair tool access
        sem_answer, sem_results = show_semantic_rag_section(q, collection, client)

        # Approach 2: agentic loop (prints live chain of thought)
        ag_answer = show_agentic_section(q, collection, client)

        # Side-by-side comparison + verdict
        show_final_comparison(q, sem_answer, sem_results, ag_answer)

        console.print()
        console.print("[dim]" + "-" * 120 + "[/dim]")

    console.print()
    console.print(Panel(
        "[bold]What the 4 queries collectively prove:[/bold]\n\n"
        "  SIMPLE      -> both approaches equivalent. Use semantic RAG (faster, cheaper).\n\n"
        "  MULTI-HOP   -> clearest gap. Same tools, same model. Semantic RAG got one\n"
        "                 search, missed the chain. Agentic followed it end to end.\n\n"
        "  AGGREGATION -> gap narrows with fair tool access. Both can calculate now.\n"
        "                 Agentic still more reliable -- searches each doc explicitly.\n\n"
        "  CONDITIONAL -> gap narrows. Both can check today's date. Remaining gap:\n"
        "                 whether the right docs landed in that one search or not.\n\n"
        "[bold]Core insight:[/bold]\n"
        "The agentic advantage is not about having more tools.\n"
        "It is about the LLM controlling the retrieval -- deciding what to look for,\n"
        "following chains across documents, and searching until the answer is complete.",
        title="[bold]Summary[/bold]",
        border_style="white",
        padding=(1, 2),
    ))


if __name__ == "__main__":
    run_comparison()
