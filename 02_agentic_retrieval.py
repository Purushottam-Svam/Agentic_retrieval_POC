"""
02_agentic_retrieval.py
-----------------------
Agentic Retrieval using Claude as the reasoning engine.

Pipeline (visible at every step):
    Query
      +-> [Step 1] Agent analyzes query complexity
      +-> [Step 2] Agent decomposes into sub-queries
      +-> [Step 3] Tool calls: search_knowledge_base(sub_query_1), ...
      +-> [Step 4] Agent reflects: "Do I have enough info?"
      +-> [Step 5] Optional follow-up tool calls
      +-> [Step 6] Agent synthesizes final answer with citations

Tools available to the agent:
  - search_knowledge_base(query, n_results) -> vector search
  - calculate(expression)                   -> arithmetic
  - get_today_info()                        -> date/day-of-week

Run:  python 02_agentic_retrieval.py
"""

import sys
import json
import time
import math
import datetime
import os
from typing import Any

from dotenv import load_dotenv
load_dotenv()  # picks up .env if present

import openai
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from rich.text import Text
from rich import box

sys.path.insert(0, ".")
from shared.kb import build_knowledge_base, semantic_search
from shared.display import (
    console,
    print_header,
    print_query,
    print_tool_call,
    print_tool_result,
    print_agent_thought,
    print_final_answer,
    print_step,
)

MODEL = "gpt-4o"

# -- Tool definitions (passed to OpenAI) --------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the TechNova enterprise knowledge base using semantic similarity. "
                "Returns the most relevant document chunks for the given query. "
                "You can call this multiple times with different queries to gather information "
                "from different angles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant knowledge base documents.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-6). Default 3.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression and return the result.",
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
            "description": "Returns today's date, day of week, and time in UTC.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SYSTEM_PROMPT = """You are an expert enterprise knowledge retrieval agent for TechNova Inc.
Your job is to answer questions accurately by searching the company knowledge base.

Guidelines:
1. Always start by searching the knowledge base -- never guess facts.
2. For complex questions, decompose into sub-questions and search for each.
3. If a search result lacks detail, run another search with a more specific query.
4. When numbers must be combined (e.g., summing RTOs), use the calculate tool.
5. When the question involves dates or days of week, use get_today_info.
6. After gathering all facts, synthesize a clear, complete answer with document references (e.g., [ARCH-001]).
7. Be explicit about what you found and from which documents.

Think step by step. Show your reasoning."""


def execute_tool(tool_name: str, tool_input: dict, collection) -> str:
    """Execute a tool call and return the string result."""
    if tool_name == "search_knowledge_base":
        query = tool_input["query"]
        n = tool_input.get("n_results", 3)
        results = semantic_search(collection, query, n_results=n)
        parts = []
        for r in results:
            parts.append(
                f"[Doc {r['id']} | {r['metadata'].get('title', '?')} | sim={r['similarity']:.3f}]\n"
                f"{r['document']}"
            )
        return "\n\n---\n\n".join(parts) if parts else "No results found."

    elif tool_name == "calculate":
        expr = tool_input["expression"]
        # Safe eval: only allow digits, operators, parens, spaces, dots
        safe_chars = set("0123456789+-*/(). ")
        if not all(c in safe_chars for c in expr):
            return f"Error: unsafe expression '{expr}'"
        try:
            result = eval(expr, {"__builtins__": {}})  # noqa: S307
            return f"{expr} = {result}"
        except Exception as e:
            return f"Error: {e}"

    elif tool_name == "get_today_info":
        now = datetime.datetime.utcnow()
        return (
            f"Today: {now.strftime('%A, %B %d, %Y')} | "
            f"UTC time: {now.strftime('%H:%M')} | "
            f"Day of week: {now.strftime('%A')}"
        )
    else:
        return f"Unknown tool: {tool_name}"


def make_openai_client() -> openai.OpenAI:
    """Create OpenAI client with SSL verification disabled for corporate proxies."""
    import httpx
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return openai.OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))


def run_agentic_query(
    query: str,
    collection,
    client: openai.OpenAI,
    label: str = "",
) -> str:
    """Run one query through the agentic loop. Returns final answer string."""
    if label:
        console.rule(f"[bold green]{label}[/bold green]")

    print_query(query, mode="AGENTIC")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    step = 0
    total_tool_calls = 0
    t0 = time.perf_counter()

    while True:
        step += 1
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            tools=TOOLS,
            messages=messages,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        text_content = msg.content or ""
        tool_calls = msg.tool_calls or []

        # Only show intermediate reasoning when the agent is about to call tools.
        # On the final turn (no tool calls / finish_reason==stop) we skip this
        # so the same text doesn't appear twice (once as "thought", once as "answer").
        if text_content.strip() and tool_calls:
            print_agent_thought(text_content)

        if finish_reason == "stop" or not tool_calls:
            final_text = text_content or "(no text response)"
            elapsed = time.perf_counter() - t0
            console.print(
                f"\n  [dim]Agentic loop finished: {step} iteration(s), "
                f"{total_tool_calls} tool call(s), {elapsed*1000:.0f} ms total[/dim]"
            )
            print_final_answer(final_text, label="Agentic Answer")
            return final_text

        # Append assistant message (with tool_calls) to history
        messages.append({
            "role": "assistant",
            "content": text_content,
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

        # Execute each tool call and append result
        for tc in tool_calls:
            total_tool_calls += 1
            tool_input = json.loads(tc.function.arguments)
            print_tool_call(tc.function.name, tool_input)
            result_str = execute_tool(tc.function.name, tool_input, collection)
            display_result = result_str[:200] + "..." if len(result_str) > 200 else result_str
            print_tool_result(display_result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })


# -- Test queries (same as 01_semantic_retrieval.py) --------------------------
QUERIES = [
    {
        "label": "Q1 - Simple (database migration policy)",
        "query": "What is the database migration policy?",
    },
    {
        "label": "Q2 - Multi-hop (who owns the service that caused the Nov 2024 incident?)",
        "query": "Who is responsible for the service that caused the November 2024 latency incident, what was the root cause, and what is their contact email?",
    },
    {
        "label": "Q3 - Aggregation (combined RTO if Auth + Payment both fail)",
        "query": "What is the combined recovery time if the Auth service, Payment Gateway, and Checkout service all fail simultaneously?",
    },
    {
        "label": "Q4 - Conditional (deploy new feature this Friday, what approvals?)",
        "query": "A developer wants to deploy the Checkout service and run a database migration this Friday at 2pm UTC. What approvals are needed for both, and who should they contact?",
    },
]


def run_agentic_demo():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print(
            Panel(
                "[red]OPENAI_API_KEY not set.[/red]\n"
                "Set it with:\n"
                "  [bold]export OPENAI_API_KEY=sk-proj-...[/bold]\n"
                "or create a [bold].env[/bold] file in this directory.",
                title="Missing API Key",
                border_style="red",
            )
        )
        sys.exit(1)

    print_header(
        "AGENTIC RETRIEVAL  (GPT-4o as reasoning engine)",
        "Query decomposition -> multi-step tool calls -> reflection -> synthesized answer",
    )

    collection = build_knowledge_base()
    import httpx
    client = openai.OpenAI(
        api_key=api_key,
        http_client=httpx.Client(verify=False),  # bypass corporate SSL proxy
    )

    answers = {}
    for q in QUERIES:
        console.print()
        answer = run_agentic_query(q["query"], collection, client, label=q["label"])
        answers[q["label"]] = answer
        console.print()

    console.print()
    console.rule("[bold]What Agentic Retrieval CAN do (that semantic cannot)[/bold]")
    from rich.table import Table
    wins = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    wins.add_column(style="green bold", width=4)
    wins.add_column(width=90)
    wins.add_row("[+]", "Decompose complex questions into sub-queries automatically")
    wins.add_row("[+]", "Retrieve from multiple documents and reason across them (multi-hop)")
    wins.add_row("[+]", "Use arithmetic tools to aggregate numbers (e.g., sum RTOs)")
    wins.add_row("[+]", "Apply conditional logic with date/policy awareness")
    wins.add_row("[+]", "Reflect on partial results and issue follow-up searches")
    wins.add_row("[+]", "Synthesize a complete natural-language answer with citations")
    console.print(wins)
    console.print(
        "\n[dim]-> Run [bold]python 03_comparison_demo.py[/bold] for a direct side-by-side comparison.[/dim]\n"
    )


if __name__ == "__main__":
    run_agentic_demo()
