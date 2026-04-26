"""
Rich terminal display utilities used by all demo scripts.
Windows-safe: forces UTF-8 output and uses ASCII-friendly symbols.
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console(highlight=True)


def print_header(title: str, subtitle: str = ""):
    console.print()
    console.rule(f"[bold magenta]{title}[/bold magenta]")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]", justify="center")
    console.print()


def print_query(query: str, mode: str = ""):
    tag = f"[bold cyan]{mode}[/bold cyan] " if mode else ""
    console.print(
        Panel(
            f"{tag}[yellow]{query}[/yellow]",
            title="[bold]Query[/bold]",
            border_style="yellow",
        )
    )


def print_semantic_results(results: list[dict], query: str):
    table = Table(
        title="Semantic Retrieval -- Top Results",
        box=box.SIMPLE_HEAD,
        border_style="blue",
        show_lines=True,
    )
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Doc ID", style="cyan", width=10)
    table.add_column("Title", style="bold", width=38)
    table.add_column("Similarity", justify="right", width=10)
    table.add_column("Snippet (first 120 chars)", width=45)

    for i, r in enumerate(results, 1):
        sim = r["similarity"]
        sim_color = "green" if sim > 0.6 else ("yellow" if sim > 0.4 else "red")
        snippet = r["document"][:120].replace("\n", " ") + "..."
        table.add_row(
            str(i),
            r["id"],
            r["metadata"].get("title", "-"),
            f"[{sim_color}]{sim:.4f}[/{sim_color}]",
            snippet,
        )
    console.print(table)


def print_step(step_num: int, title: str, content: str, color: str = "cyan"):
    console.print(
        Panel(
            content,
            title=f"[bold {color}]Step {step_num}: {title}[/bold {color}]",
            border_style=color,
            padding=(0, 1),
        )
    )


def print_tool_call(tool_name: str, args: dict):
    args_str = ", ".join(f"{k}=[italic]{v}[/italic]" for k, v in args.items())
    console.print(f"  [bold green]>> TOOL CALL[/bold green] [cyan]{tool_name}[/cyan]({args_str})")


def print_tool_result(result_summary: str):
    console.print(f"  [bold yellow]<< TOOL RESULT[/bold yellow] {result_summary}")


def print_agent_thought(thought: str):
    console.print(
        Panel(
            f"[italic dim]{thought}[/italic dim]",
            title="[bold magenta][*] Agent Reasoning[/bold magenta]",
            border_style="magenta",
            padding=(0, 1),
        )
    )


def print_final_answer(answer: str, label: str = "Final Answer", color: str = "green"):
    console.print(
        Panel(
            answer,
            title=f"[bold {color}][OK] {label}[/bold {color}]",
            border_style=color,
            padding=(1, 2),
        )
    )


def print_comparison_table(query: str, semantic_answer: str, agentic_answer: str):
    console.print()
    console.rule("[bold white]SIDE-BY-SIDE COMPARISON[/bold white]")
    console.print(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    table = Table(box=box.SIMPLE_HEAD, show_lines=True, width=120)
    table.add_column("Semantic Retrieval", style="blue", ratio=1)
    table.add_column("Agentic Retrieval", style="green", ratio=1)
    table.add_row(semantic_answer, agentic_answer)
    console.print(table)


def print_score_comparison(
    query_label: str,
    semantic_score: int,
    agentic_score: int,
    criteria: str,
):
    bar = lambda score: "#" * score + "." * (10 - score)
    console.print(
        f"  [dim]{query_label}[/dim]\n"
        f"  Semantic: [blue]{bar(semantic_score)}[/blue] {semantic_score}/10\n"
        f"  Agentic:  [green]{bar(agentic_score)}[/green] {agentic_score}/10\n"
        f"  [dim]Criteria: {criteria}[/dim]"
    )
