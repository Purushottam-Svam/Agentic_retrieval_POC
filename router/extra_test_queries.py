"""
extra_test_queries.py
---------------------
50 additional queries to test the trained router beyond the 4 standard ones.

Coverage:
  - 25 SIMPLE  (Q01-Q25): single-doc lookups, factual, no chaining needed
  - 25 COMPLEX (Q26-Q50): multi-hop, aggregation, date-conditional, compound

Each query has:
  - id            : Q01..Q50
  - query         : the question text
  - expected      : SIMPLE or COMPLEX
  - type          : sub-type for understanding what makes it that class
  - why           : plain-English reason for the label

Run this file to classify all 50 queries through the trained router:
    python router/extra_test_queries.py

Output is printed to console and also saved to router/extra_test_outputs.json
"""

import os
import sys
import json
import time

# Run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# -----------------------------------------------------------------------
# The 50 queries
# -----------------------------------------------------------------------

QUERIES = [

    # ===================================================================
    # SIMPLE queries (Q01 - Q25)
    # One document contains the full answer. One search is enough.
    # ===================================================================

    {
        "id": "Q01",
        "query": "What is the RTO for the Auth service?",
        "expected": "SIMPLE",
        "type": "single-fact lookup",
        "why": "RTO is a single field in ARCH-002. One search finds it."
    },
    {
        "id": "Q02",
        "query": "Who leads the Payments team?",
        "expected": "SIMPLE",
        "type": "single-fact lookup",
        "why": "Team lead for Payments is in ARCH-001 or TEAM-001. One search."
    },
    {
        "id": "Q03",
        "query": "What is the deployment freeze window on Fridays?",
        "expected": "SIMPLE",
        "type": "policy lookup",
        "why": "Friday deployment cutoff is a single fact in POL-001."
    },
    {
        "id": "Q04",
        "query": "What does RTO stand for?",
        "expected": "SIMPLE",
        "type": "definition",
        "why": "Terminology definition in a single doc. No chaining."
    },
    {
        "id": "Q05",
        "query": "Who is Omar Hassan?",
        "expected": "SIMPLE",
        "type": "person lookup",
        "why": "Omar Hassan's role is in TEAM-001. Single doc."
    },
    {
        "id": "Q06",
        "query": "What is the on-call escalation policy?",
        "expected": "SIMPLE",
        "type": "policy lookup",
        "why": "Escalation policy lives in one policy document."
    },
    {
        "id": "Q07",
        "query": "What services does the Auth service depend on?",
        "expected": "SIMPLE",
        "type": "architecture lookup",
        "why": "Dependencies listed in ARCH-002. Single doc."
    },
    {
        "id": "Q08",
        "query": "What is the checkout service RTO?",
        "expected": "SIMPLE",
        "type": "single-fact lookup",
        "why": "RTO is a single field in ARCH-003."
    },
    {
        "id": "Q09",
        "query": "How many rows trigger the pt-online-schema-change requirement?",
        "expected": "SIMPLE",
        "type": "policy lookup",
        "why": "The 10M row threshold is one fact in POL-002."
    },
    {
        "id": "Q10",
        "query": "What is a P1 incident?",
        "expected": "SIMPLE",
        "type": "definition",
        "why": "P1 definition is a single entry in FAQ or policy doc."
    },
    {
        "id": "Q11",
        "query": "Who is the VP of Engineering at TechNova?",
        "expected": "SIMPLE",
        "type": "person lookup",
        "why": "James Liu is listed in TEAM-001. One search."
    },
    {
        "id": "Q12",
        "query": "What is the payment gateway's primary payment processor?",
        "expected": "SIMPLE",
        "type": "architecture lookup",
        "why": "Integration details (Stripe primary) in ARCH-001."
    },
    {
        "id": "Q13",
        "query": "What monitoring tool does TechNova use for PagerDuty alerts?",
        "expected": "SIMPLE",
        "type": "tool lookup",
        "why": "Monitoring tooling mentioned in a single FAQ or runbook doc."
    },
    {
        "id": "Q14",
        "query": "What is the RPO for the Analytics pipeline?",
        "expected": "SIMPLE",
        "type": "single-fact lookup",
        "why": "RPO is a single field in ARCH-004."
    },
    {
        "id": "Q15",
        "query": "What is the purpose of the staging dry-run in database migrations?",
        "expected": "SIMPLE",
        "type": "policy explanation",
        "why": "Dry-run rationale is explained in POL-002. One doc."
    },
    {
        "id": "Q16",
        "query": "Who leads the Data Engineering team?",
        "expected": "SIMPLE",
        "type": "person lookup",
        "why": "Priya Mehta's role in TEAM-001 or ARCH-004. One search."
    },
    {
        "id": "Q17",
        "query": "What Jira tag must be used for production deployment tickets?",
        "expected": "SIMPLE",
        "type": "policy lookup",
        "why": "The 'prod-deploy' tag requirement is one fact in POL-001."
    },
    {
        "id": "Q18",
        "query": "What was the root cause of the incident in Q1 2025?",
        "expected": "SIMPLE",
        "type": "incident lookup",
        "why": "Root cause of a specific incident is in one INC-* doc."
    },
    {
        "id": "Q19",
        "query": "What is the Analytics pipeline's data ingestion volume per day?",
        "expected": "SIMPLE",
        "type": "architecture lookup",
        "why": "2TB/day figure is one fact in ARCH-004."
    },
    {
        "id": "Q20",
        "query": "What does the Checkout service do?",
        "expected": "SIMPLE",
        "type": "service description",
        "why": "Service description is the opening paragraph of ARCH-003."
    },
    {
        "id": "Q21",
        "query": "What version of the Payment Gateway caused the November 2024 incident?",
        "expected": "SIMPLE",
        "type": "incident lookup",
        "why": "Version 3.7.2 is a single fact in INC-001."
    },
    {
        "id": "Q22",
        "query": "What is the rollback version the team used to resolve the November 2024 incident?",
        "expected": "SIMPLE",
        "type": "incident lookup",
        "why": "Rollback to 3.7.1 is a single fact in INC-001."
    },
    {
        "id": "Q23",
        "query": "How long did the November 2024 latency spike last?",
        "expected": "SIMPLE",
        "type": "incident lookup",
        "why": "Duration (14:00-16:30 UTC) is in INC-001. One search."
    },
    {
        "id": "Q24",
        "query": "What Kafka pipeline does the Analytics service use?",
        "expected": "SIMPLE",
        "type": "architecture lookup",
        "why": "Kafka -> Spark -> Snowflake is described in ARCH-004."
    },
    {
        "id": "Q25",
        "query": "What is the TechNova policy for hotfix deployments?",
        "expected": "SIMPLE",
        "type": "policy lookup",
        "why": "Hotfix policy is one section in POL-001."
    },

    # ===================================================================
    # COMPLEX queries (Q26 - Q50)
    # Require chaining, aggregation, date logic, or compound lookups.
    # ===================================================================

    {
        "id": "Q26",
        "query": "What is the combined RTO for Auth and Checkout if both go down simultaneously?",
        "expected": "COMPLEX",
        "type": "aggregation",
        "why": "Needs ARCH-002 (Auth RTO=10) + ARCH-003 (Checkout RTO=20) + arithmetic."
    },
    {
        "id": "Q27",
        "query": "Which team lead should I contact if the service that caused the November 2024 incident fails again?",
        "expected": "COMPLEX",
        "type": "multi-hop",
        "why": "INC-001 → service name → ARCH-001 → team lead name. Two hops."
    },
    {
        "id": "Q28",
        "query": "Can I deploy to production this Saturday, and if so who approves it?",
        "expected": "COMPLEX",
        "type": "date-conditional",
        "why": "Needs today's date, then POL-001 weekend rules, then TEAM-001 contacts."
    },
    {
        "id": "Q29",
        "query": "What is the email of the lead of the service with the highest RTO?",
        "expected": "COMPLEX",
        "type": "multi-hop + aggregation",
        "why": "Needs all ARCH-* docs to compare RTOs, then TEAM-001 for the email of the winner."
    },
    {
        "id": "Q30",
        "query": "If I need approval from both the DBA lead and the VP of Engineering, who do I email?",
        "expected": "COMPLEX",
        "type": "multi-person lookup",
        "why": "Needs TEAM-001 twice (two different roles). Involves combining two lookups."
    },
    {
        "id": "Q31",
        "query": "Which services were involved in incidents in 2024 and who owns each one?",
        "expected": "COMPLEX",
        "type": "multi-hop + multi-doc",
        "why": "Needs INC-001, INC-002, INC-003 each → then ARCH-* for each owner."
    },
    {
        "id": "Q32",
        "query": "What is the total downtime exposure (sum of all RTOs) across every service in the knowledge base?",
        "expected": "COMPLEX",
        "type": "aggregation",
        "why": "Needs ARCH-001 + ARCH-002 + ARCH-003 + ARCH-004 RTOs summed. Full scan + math."
    },
    {
        "id": "Q33",
        "query": "I want to deploy the Auth service on Monday morning at 9am UTC. What approvals do I need and who do I contact?",
        "expected": "COMPLEX",
        "type": "date-conditional + contact lookup",
        "why": "Needs date check (POL-001), then team lead name (ARCH-002), then email (TEAM-001)."
    },
    {
        "id": "Q34",
        "query": "What is the contact email of the person responsible for the service that had the connection pool bug?",
        "expected": "COMPLEX",
        "type": "multi-hop",
        "why": "INC-001 (connection pool = Payment Gateway) → ARCH-001 (Sarah Chen) → TEAM-001 (email)."
    },
    {
        "id": "Q35",
        "query": "If all four TechNova services fail, what is the maximum time before full recovery?",
        "expected": "COMPLEX",
        "type": "aggregation",
        "why": "Needs all four ARCH-* RTOs, then find the max (not sum). Multi-doc comparison."
    },
    {
        "id": "Q36",
        "query": "Which service has the lowest RTO and what team is responsible for it?",
        "expected": "COMPLEX",
        "type": "aggregation + lookup",
        "why": "Must retrieve all RTOs, find minimum, then identify the team. Multi-doc."
    },
    {
        "id": "Q37",
        "query": "What approvals are needed to run a DB migration and deploy a new microservice on the same Friday?",
        "expected": "COMPLEX",
        "type": "compound policy",
        "why": "POL-001 (deploy) + POL-002 (migration) + date check (Friday window) all needed."
    },
    {
        "id": "Q38",
        "query": "What were the root causes of all three 2024 incidents and do any share a common pattern?",
        "expected": "COMPLEX",
        "type": "multi-doc synthesis",
        "why": "Needs INC-001 + INC-002 + INC-003 and reasoning across them."
    },
    {
        "id": "Q39",
        "query": "Who do I page if both the Auth service and the Payment Gateway are down at 3am?",
        "expected": "COMPLEX",
        "type": "multi-person + on-call",
        "why": "Needs ARCH-001 + ARCH-002 team leads, then TEAM-001 for both contacts."
    },
    {
        "id": "Q40",
        "query": "What is the difference in RTO between the fastest and slowest customer-facing service?",
        "expected": "COMPLEX",
        "type": "aggregation + arithmetic",
        "why": "Needs all ARCH-* RTOs, finds min and max, subtracts. Multi-doc + math."
    },
    {
        "id": "Q41",
        "query": "Can a developer with no direct team lead deploy a critical service, and what is the escalation path?",
        "expected": "COMPLEX",
        "type": "policy + contact chain",
        "why": "POL-001 for rules, then TEAM-001 for escalation contacts. Two docs."
    },
    {
        "id": "Q42",
        "query": "What is the email address of the person who owns the service with the highest number of historical incidents?",
        "expected": "COMPLEX",
        "type": "multi-hop + aggregation",
        "why": "Count incidents per service across INC-*, find the most, then chain to ARCH-* then TEAM-001."
    },
    {
        "id": "Q43",
        "query": "Deploy Checkout and Auth simultaneously this Thursday — how many approvals in total and from whom?",
        "expected": "COMPLEX",
        "type": "compound + date-conditional",
        "why": "Two deployments × POL-001 rules + date check + TEAM-001 contacts. High complexity."
    },
    {
        "id": "Q44",
        "query": "What is the combined RTO of all services owned by the team leads mentioned in the November 2024 incident report?",
        "expected": "COMPLEX",
        "type": "multi-hop + aggregation",
        "why": "INC-001 → service → ARCH-001 (RTO). May expand to multiple services. Chain + math."
    },
    {
        "id": "Q45",
        "query": "If I run a migration tonight and it fails, who do I call first and what is the rollback procedure?",
        "expected": "COMPLEX",
        "type": "multi-doc procedural",
        "why": "POL-002 (rollback requirement) + RUN-* (runbook steps) + TEAM-001 (DBA contact)."
    },
    {
        "id": "Q46",
        "query": "Which team lead has the most services under their ownership?",
        "expected": "COMPLEX",
        "type": "aggregation across docs",
        "why": "Must scan all ARCH-* docs, count services per lead, find max. Full corpus scan."
    },
    {
        "id": "Q47",
        "query": "What would the total planned downtime window be if we schedule maintenance for every service back-to-back?",
        "expected": "COMPLEX",
        "type": "aggregation + arithmetic",
        "why": "Sum of all RTOs across ARCH-001 through ARCH-004. Multi-doc + math."
    },
    {
        "id": "Q48",
        "query": "Who are the two people I need to get approval from to deploy a critical service on a Friday before the cutoff?",
        "expected": "COMPLEX",
        "type": "policy + contact lookup",
        "why": "POL-001 (critical service needs team lead + VP) → TEAM-001 for both contacts."
    },
    {
        "id": "Q49",
        "query": "What is the email of the DBA lead and the Commerce team lead, and are they the same person?",
        "expected": "COMPLEX",
        "type": "multi-person lookup",
        "why": "Two separate TEAM-001 lookups for different roles, then comparison."
    },
    {
        "id": "Q50",
        "query": "Which service had a latency incident in November 2024, what is its RTO, and is that RTO above or below average?",
        "expected": "COMPLEX",
        "type": "multi-hop + aggregation + comparison",
        "why": "INC-001 → ARCH-001 (RTO=15) → all ARCH-* for average → comparison. Full chain."
    },
]


# -----------------------------------------------------------------------
# Runner: classify all 50 queries and print + save results
# -----------------------------------------------------------------------

def run():
    print("\n=== Router: 50 Extra Test Queries ===\n")
    print(f"Total queries: {len(QUERIES)}  |  "
          f"Expected SIMPLE: {sum(1 for q in QUERIES if q['expected']=='SIMPLE')}  |  "
          f"Expected COMPLEX: {sum(1 for q in QUERIES if q['expected']=='COMPLEX')}\n")

    from router.router_model import QueryRouter

    try:
        router = QueryRouter()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Warm-up call so model load time is excluded from per-query timing
    router.route("warm-up query")

    results = []
    correct = 0

    print(f"{'ID':<5} {'Expected':<10} {'Got':<12} {'Conf':>6}  {'Match':<6}  Query")
    print("-" * 110)

    t_start = time.perf_counter()
    for q in QUERIES:
        decision, confidence = router.route(q["query"])
        match = decision == q["expected"]
        if match:
            correct += 1

        match_str = "YES" if match else "NO "
        flag      = "" if match else "  <-- MISMATCH"
        query_short = q["query"][:70] + ("..." if len(q["query"]) > 70 else "")
        print(f"{q['id']:<5} {q['expected']:<10} {decision:<12} {confidence:>5.1%}  {match_str:<6}  {query_short}{flag}")

        results.append({
            "id":          q["id"],
            "query":       q["query"],
            "expected":    q["expected"],
            "type":        q["type"],
            "why":         q["why"],
            "decision":    decision,
            "confidence":  confidence,
            "correct":     match,
        })

    elapsed = (time.perf_counter() - t_start) * 1000

    accuracy = correct / len(QUERIES)
    n_simple  = sum(1 for r in results if r["expected"] == "SIMPLE")
    n_complex = sum(1 for r in results if r["expected"] == "COMPLEX")
    correct_simple  = sum(1 for r in results if r["expected"] == "SIMPLE"  and r["correct"])
    correct_complex = sum(1 for r in results if r["expected"] == "COMPLEX" and r["correct"])

    print("\n" + "=" * 110)
    print(f"Overall accuracy : {correct}/{len(QUERIES)} ({accuracy:.0%})")
    print(f"SIMPLE accuracy  : {correct_simple}/{n_simple} ({correct_simple/n_simple:.0%})")
    print(f"COMPLEX accuracy : {correct_complex}/{n_complex} ({correct_complex/n_complex:.0%})")
    print(f"Total inference  : {elapsed:.0f} ms  ({elapsed/len(QUERIES):.1f} ms per query)")

    # False negatives (most dangerous)
    fn = [r for r in results if r["expected"] == "COMPLEX" and r["decision"] == "SIMPLE"]
    if fn:
        print(f"\nDANGEROUS MISROUTES (COMPLEX classified as SIMPLE): {len(fn)}")
        for r in fn:
            print(f"  [{r['id']}] {r['query'][:80]}")
    else:
        print("\nNo dangerous misroutes (COMPLEX classified as SIMPLE). Good.")

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), "extra_test_outputs.json")
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": len(results),
                "correct": correct,
                "accuracy": round(accuracy, 4),
                "simple_accuracy":  round(correct_simple / n_simple, 4),
                "complex_accuracy": round(correct_complex / n_complex, 4),
                "total_inference_ms": round(elapsed, 1),
                "avg_inference_ms":   round(elapsed / len(QUERIES), 2),
                "false_negatives": len(fn),
            },
            "results": results
        }, f, indent=2)

    # Analysis of mismatches
    mismatches = [r for r in results if not r["correct"]]
    if mismatches:
        print(f"\n--- Mismatch Analysis ---")
        for r in mismatches:
            q_data = next(q for q in QUERIES if q["id"] == r["id"])
            print(f"\n  [{r['id']}] Expected {r['expected']}, got {r['decision']} ({r['confidence']:.1%})")
            print(f"  Query : {r['query']}")
            print(f"  Type  : {q_data['type']}")
            print(f"  Why expected {r['expected']}: {q_data['why']}")
            print(f"  Why router said {r['decision']}: Query phrasing resembles {r['decision'].lower()} patterns in training data.")

    print(f"\nFull results saved to: router/extra_test_outputs.json\n")


if __name__ == "__main__":
    run()
