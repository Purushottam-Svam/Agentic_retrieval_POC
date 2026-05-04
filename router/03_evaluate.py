"""
03_evaluate.py
--------------
Tests the trained router against our 4 standard queries.

For each query this script:
  1. Runs the router -> SIMPLE / COMPLEX / UNCERTAIN + confidence score
  2. Runs Semantic RAG  (always, regardless of router decision)
  3. Runs Agentic       (always, regardless of router decision)
  4. Scores information richness of both answers automatically
  5. Shows a VERDICT: was the router right? what would have been lost?

Information richness scoring:
  Counts signals that indicate completeness of an answer:
    - Email addresses found        (+3 each)  <- hardest to get, needs multi-hop
    - Named people found           (+2 each)
    - Specific numbers/RTOs found  (+1 each)
  Agentic score >> Semantic score  -> COMPLEX was the right route
  Agentic score ~= Semantic score  -> SIMPLE would have been fine (saved tokens)

Run:
    python router/03_evaluate.py

Requires:
    router/model/  (built by 02_train.py)
    OPENAI_API_KEY in .env
"""

import os
import sys
import re
import importlib
import warnings
warnings.filterwarnings("ignore")

# Allow imports from project root (shared/, data/, 02_agentic_retrieval, etc.)
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.text import Text
from rich import box

# Import router (our new package)
from router.router_model import QueryRouter

# Import existing systems WITHOUT modifying them
_ag   = importlib.import_module("02_agentic_retrieval")
_comp = importlib.import_module("03_comparison_demo")

run_agentic_query = _ag.run_agentic_query
make_openai_client = _ag.make_openai_client
run_semantic_rag   = _comp.run_semantic_rag

from shared.kb import build_knowledge_base

console = Console()

# ---------------------------------------------------------------------------
# The 4 standard queries with ground truth labels
# (what a human expert would say the correct route is, and why)
# ---------------------------------------------------------------------------
EVAL_QUERIES = [
    {
        "label":       "Q1 - Simple",
        "query":       "What is the database migration policy?",
        "ground_truth": "SIMPLE",
        "reason":      "Answer is entirely in POL-002. One search finds it. No chaining.",
    },
    {
        "label":       "Q2 - Multi-hop",
        "query":       "Who is responsible for the service that caused the November 2024 latency incident, what was the root cause, and what is their contact email?",
        "ground_truth": "COMPLEX",
        "reason":      "Needs INC-001 -> ARCH-001 -> TEAM-001. Email only in TEAM-001. Three hops.",
    },
    {
        "label":       "Q3 - Aggregation",
        "query":       "What is the total recovery time we should plan for if all customer-facing services go down at once?",
        "ground_truth": "COMPLEX",
        "reason":      "Needs RTO from ARCH-001 + ARCH-002 + ARCH-003, then arithmetic. Multi-doc + math.",
    },
    {
        "label":       "Q4 - Conditional",
        "query":       "A developer wants to deploy the Checkout service and run a database migration this Friday at 2pm UTC. What approvals are needed for both, and who should they contact?",
        "ground_truth": "COMPLEX",
        "reason":      "Needs POL-001 (deploy) + POL-002 (migration) + TEAM-001 (contacts) + date check.",
    },
]


# ---------------------------------------------------------------------------
# Information richness scorer
# ---------------------------------------------------------------------------

EMAIL_RE   = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
NUMBER_RE  = re.compile(r"\b\d+\s*(?:minutes?|hours?|min|hrs?|%)\b", re.IGNORECASE)

# Names we know should appear in complete answers
KNOWN_NAMES = [
    "sarah chen", "david park", "alex rivera",
    "priya mehta", "marcus webb", "omar hassan", "james liu",
]

def richness_score(text: str) -> tuple[int, dict]:
    """
    Score how information-rich an answer is.

    Returns:
        (score, breakdown)
        score     : integer — higher = more complete
        breakdown : dict with counts of each signal type
    """
    t = text.lower()

    emails  = EMAIL_RE.findall(text)
    numbers = NUMBER_RE.findall(text)
    names   = [n for n in KNOWN_NAMES if n in t]

    score = len(emails) * 3 + len(names) * 2 + len(numbers) * 1

    breakdown = {
        "emails":  emails,
        "names":   names,
        "numbers": numbers,
        "score":   score,
    }
    return score, breakdown


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def compute_verdict(
    router_decision: str,
    ground_truth: str,
    sem_score: int,
    ag_score: int,
) -> tuple[str, str]:
    """
    Returns (verdict_label, explanation).

    Router correct  = decision matches ground truth
    Router wrong    = decision doesn't match ground truth

    But we also check the actual answer quality gap:
      Large gap (ag >> sem) + router said COMPLEX  -> CORRECT + MEANINGFUL SAVE
      Large gap (ag >> sem) + router said SIMPLE   -> WRONG   + DANGEROUS (user gets incomplete answer)
      Small gap             + router said SIMPLE   -> CORRECT + TOKEN SAVE (both answers good)
      Small gap             + router said COMPLEX  -> SAFE OVERROUTE (spent extra tokens, no harm)
    """
    gap = ag_score - sem_score
    router_correct = (router_decision in ("COMPLEX", "UNCERTAIN") and ground_truth == "COMPLEX") or \
                     (router_decision == "SIMPLE" and ground_truth == "SIMPLE")

    if router_correct and ground_truth == "COMPLEX" and gap >= 3:
        return "[bold green]CORRECT[/bold green]", \
               f"Router sent to Agentic. Agentic was richer by {gap} pts. Right call."
    elif router_correct and ground_truth == "SIMPLE":
        return "[bold green]CORRECT[/bold green]", \
               f"Router sent to Semantic RAG. Both answers similar (gap={gap}). Saved tokens."
    elif not router_correct and ground_truth == "COMPLEX" and gap >= 3:
        return "[bold red]WRONG -- DANGEROUS[/bold red]", \
               f"Router said SIMPLE, but Agentic was richer by {gap} pts. " \
               f"User would have gotten incomplete answer."
    elif not router_correct and ground_truth == "SIMPLE":
        return "[bold yellow]SAFE OVERROUTE[/bold yellow]", \
               f"Router said COMPLEX, but SIMPLE was sufficient (gap={gap}). " \
               f"Wasted tokens, no harm to answer quality."
    elif gap < 3 and ground_truth == "COMPLEX":
        return "[bold yellow]ACCEPTABLE[/bold yellow]", \
               f"Ground truth is COMPLEX but answers similar (gap={gap}). " \
               f"Router decision had minimal real impact."
    else:
        return "[yellow]UNCLEAR[/yellow]", f"Scores close (gap={gap}). Hard to say definitively."


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print(Panel("[red]OPENAI_API_KEY not set.[/red]", border_style="red"))
        sys.exit(1)

    console.print()
    console.print(Panel(
        "[bold cyan]QUERY ROUTER EVALUATION[/bold cyan]\n\n"
        "For each query:\n"
        "  1. Router classifies -> SIMPLE / COMPLEX\n"
        "  2. Both Semantic RAG and Agentic run (to compare what each produces)\n"
        "  3. Answer richness scored automatically (emails, names, numbers)\n"
        "  4. Verdict: was the router right? what would have been lost?\n\n"
        "[dim]Existing code is untouched. Router is a separate package.[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # Load router (once)
    console.print("\n[dim]Loading router model...[/dim]")
    try:
        router = QueryRouter()
        console.print("[green]Router loaded.[/green]")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    # Build KB + client (once)
    console.print("[dim]Building knowledge base...[/dim]")
    collection = build_knowledge_base()
    client     = make_openai_client()

    results = []

    for q in EVAL_QUERIES:
        console.print()
        console.rule(f"[bold white]{q['label']}[/bold white]")
        console.print(f"\n[bold]Query:[/bold] {q['query']}")
        console.print(f"[dim]Ground truth: {q['ground_truth']} | Reason: {q['reason']}[/dim]\n")

        # -- Step 1: Router classification --
        decision, confidence = router.route(q["query"])
        color = "green" if decision == "SIMPLE" else ("red" if decision == "COMPLEX" else "yellow")
        console.print(Panel(
            f"Decision   : [{color}]{decision}[/{color}]\n"
            f"Confidence : {confidence:.1%}\n"
            f"Ground truth: {q['ground_truth']}\n"
            f"Match      : {'[green]YES[/green]' if decision == q['ground_truth'] else '[red]NO[/red]'}",
            title="[bold]Router Output[/bold]",
            border_style=color,
            padding=(0, 1),
        ))

        # -- Step 2: Run Semantic RAG --
        console.rule("[blue]Semantic RAG[/blue]")
        sem_answer, sem_docs = run_semantic_rag(q["query"], collection, client)
        sem_score, sem_breakdown = richness_score(sem_answer)

        # -- Step 3: Run Agentic --
        console.rule("[green]Agentic Retrieval[/green]")
        ag_answer = run_agentic_query(q["query"], collection, client)
        ag_score, ag_breakdown = richness_score(ag_answer)

        # -- Step 4: Score and verdict --
        verdict_label, verdict_explanation = compute_verdict(
            decision, q["ground_truth"], sem_score, ag_score
        )

        # Richness breakdown table
        rich_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
        rich_table.add_column("Signal",       width=20)
        rich_table.add_column("Semantic RAG", width=40)
        rich_table.add_column("Agentic",      width=40)

        rich_table.add_row(
            "Emails found",
            str(sem_breakdown["emails"])  or "none",
            str(ag_breakdown["emails"])   or "none",
        )
        rich_table.add_row(
            "Names found",
            ", ".join(sem_breakdown["names"])  or "none",
            ", ".join(ag_breakdown["names"])   or "none",
        )
        rich_table.add_row(
            "Numbers found",
            ", ".join(sem_breakdown["numbers"])  or "none",
            ", ".join(ag_breakdown["numbers"])   or "none",
        )
        rich_table.add_row(
            "[bold]Richness score[/bold]",
            f"[bold]{sem_score}[/bold]",
            f"[bold]{ag_score}[/bold]",
        )

        console.print("\n[bold]Information Richness Comparison:[/bold]")
        console.print(rich_table)

        console.print(Panel(
            f"Router said    : [{color}]{decision}[/{color}] ({confidence:.1%} confidence)\n"
            f"Ground truth   : {q['ground_truth']}\n"
            f"Richness gap   : Semantic={sem_score}  Agentic={ag_score}  Gap={ag_score - sem_score}\n\n"
            f"Verdict        : {verdict_label}\n"
            f"Explanation    : {verdict_explanation}",
            title="[bold white]VERDICT[/bold white]",
            border_style="white",
            padding=(0, 1),
        ))

        results.append({
            "label":       q["label"],
            "decision":    decision,
            "confidence":  confidence,
            "ground_truth": q["ground_truth"],
            "correct":     decision == q["ground_truth"],
            "sem_score":   sem_score,
            "ag_score":    ag_score,
            "gap":         ag_score - sem_score,
            "verdict":     verdict_label,
        })

    # ---------------------------------------------------------------------------
    # Summary table across all 4 queries
    # ---------------------------------------------------------------------------
    console.print()
    console.rule("[bold white]ROUTER EVALUATION SUMMARY[/bold white]")

    summary = Table(box=box.SIMPLE_HEAD, show_lines=True, padding=(0, 1), width=130)
    summary.add_column("Query",        width=20)
    summary.add_column("Router",       width=12)
    summary.add_column("Confidence",   width=12)
    summary.add_column("Ground Truth", width=13)
    summary.add_column("Match",        width=8)
    summary.add_column("Sem Score",    width=10)
    summary.add_column("Ag Score",     width=9)
    summary.add_column("Gap",          width=5)
    summary.add_column("Verdict",      width=30)

    n_correct = 0
    for r in results:
        match_str = "[green]YES[/green]" if r["correct"] else "[red]NO[/red]"
        if r["correct"]:
            n_correct += 1
        summary.add_row(
            r["label"],
            r["decision"],
            f"{r['confidence']:.1%}",
            r["ground_truth"],
            match_str,
            str(r["sem_score"]),
            str(r["ag_score"]),
            str(r["gap"]),
            r["verdict"],
        )

    console.print(summary)
    accuracy = n_correct / len(results)
    console.print(
        f"\nRouter accuracy on standard queries: "
        f"[bold]{'[green]' if accuracy >= 0.75 else '[red]'}{n_correct}/{len(results)} ({accuracy:.0%})[/bold][/]"
    )
    console.print(
        "\n[dim]Note: Router was trained on synthetic data. Accuracy improves with "
        "real query logs and periodic retraining.[/dim]\n"
    )


if __name__ == "__main__":
    run_evaluation()
