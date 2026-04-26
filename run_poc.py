"""
run_poc.py  --  Master entry point
----------------------------------
Runs all three demos in sequence with clear section breaks.

Usage:
    python run_poc.py              # runs all three demos
    python run_poc.py --semantic   # semantic only (no API key needed)
    python run_poc.py --agentic    # agentic only
    python run_poc.py --compare    # side-by-side comparison
"""

import sys
import os
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()  # picks up .env if present

console = Console()

BANNER = """
+==============================================================================+
|          AGENTIC RETRIEVAL  POC  --  TechNova Enterprise Knowledge Base       |
|                                                                              |
|  Semantic Retrieval:   cosine-similarity only, returns chunks                |
|  Agentic Retrieval:    Claude reasons, decomposes, searches, reflects        |
+==============================================================================+
"""


def main():
    parser = argparse.ArgumentParser(description="Agentic Retrieval POC")
    parser.add_argument("--semantic", action="store_true", help="Run semantic demo only")
    parser.add_argument("--agentic", action="store_true", help="Run agentic demo only")
    parser.add_argument("--compare", action="store_true", help="Run comparison demo only")
    parser.add_argument("--rebuild-kb", action="store_true", help="Force-rebuild the knowledge base")
    args = parser.parse_args()

    console.print(BANNER, style="bold cyan")

    if args.rebuild_kb:
        console.print("[yellow]Rebuilding knowledge base from scratch...[/yellow]")
        sys.path.insert(0, ".")
        from shared.kb import build_knowledge_base
        build_knowledge_base(force_rebuild=True)
        console.print("[green]Done.[/green]")
        return

    run_all = not (args.semantic or args.agentic or args.compare)

    import importlib

    if args.semantic or run_all:
        console.rule("[bold blue]DEMO 1 of 3 -- Semantic Retrieval[/bold blue]")
        mod = importlib.import_module("01_semantic_retrieval")
        mod.run_semantic_demo()

    if args.agentic or run_all:
        console.rule("[bold green]DEMO 2 of 3 -- Agentic Retrieval[/bold green]")
        mod = importlib.import_module("02_agentic_retrieval")
        mod.run_agentic_demo()

    if args.compare or run_all:
        console.rule("[bold magenta]DEMO 3 of 3 -- Side-by-Side Comparison[/bold magenta]")
        mod = importlib.import_module("03_comparison_demo")
        mod.run_comparison()


if __name__ == "__main__":
    main()
