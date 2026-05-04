"""
01_generate_data.py
-------------------
Generates synthetic training data for the query router classifier.

Strategy:
  Step 1 - 80 hardcoded seed examples (40 SIMPLE + 40 COMPLEX)
            These are guaranteed quality ground truth.
  Step 2 - GPT-4o generates 220 more examples (110 per class)
            to reach ~300 total for solid classifier training.
  Step 3 - Saves everything to router/training_data.json

Run:
    python router/01_generate_data.py

Output:
    router/training_data.json   -- (query, label) pairs ready for training
"""

import os
import sys
import json
import random

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import httpx
import openai

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "training_data.json")

# ---------------------------------------------------------------------------
# STEP 1 -- Hardcoded seed examples (guaranteed quality)
# ---------------------------------------------------------------------------

SIMPLE_SEEDS = [
    "What is the database migration policy?",
    "What are the on-call hours for engineers?",
    "What is the RTO for the Auth service?",
    "Who leads the Auth service team?",
    "What is the deployment freeze window?",
    "Where is the incident runbook documented?",
    "What is the RTO for the Payment Gateway?",
    "What is the TechNova on-call escalation policy?",
    "Who is the team lead for the Checkout service?",
    "What does RTO stand for?",
    "What is the deployment approval process?",
    "How many documents are in the knowledge base?",
    "What services does TechNova operate?",
    "What is the database migration approval requirement?",
    "What is the maximum allowed downtime for critical services?",
    "Who is responsible for infrastructure at TechNova?",
    "What is the difference between RTO and RPO?",
    "How do I escalate a P1 incident?",
    "What is the on-call rotation schedule?",
    "What version of the deployment policy is current?",
    "Who approves database migrations at TechNova?",
    "What is the Auth service architecture?",
    "What services depend on the Payment Gateway?",
    "What is the standard SLA for critical services?",
    "Where do I find the incident post-mortem template?",
    "What is the team lead email format at TechNova?",
    "What is the checkout service RTO?",
    "What are the deployment cutoff hours on Friday?",
    "Who is Marcus Webb?",
    "What is the Analytics Pipeline RTO?",
    "What monitoring tools does TechNova use?",
    "How do I request DBA review for a migration?",
    "What is the staging environment dry-run requirement?",
    "Who is the VP of Engineering at TechNova?",
    "What is the definition of a P1 incident?",
    "Where are team contact emails documented?",
    "What is the rollback procedure for failed deployments?",
    "Who is Priya Mehta?",
    "What is the policy for hotfix deployments?",
    "What are the steps in the incident response runbook?",
]

COMPLEX_SEEDS = [
    "Who is responsible for the service that caused the November 2024 latency incident, and what is their contact email?",
    "What is the total recovery time if all customer-facing services go down at once?",
    "A developer wants to deploy the Checkout service and run a database migration this Friday at 2pm UTC. What approvals are needed?",
    "Which team lead owns the service with the highest RTO, and how do I reach them?",
    "If Auth service and Payment Gateway both go down, what is the combined recovery time?",
    "Who caused the most recent incident and what is their manager's contact?",
    "What approvals does a developer need to deploy a critical service on a Friday afternoon?",
    "Which services have RTOs above 15 minutes, and who leads those teams?",
    "What was the root cause of the November 2024 incident and which team was responsible?",
    "If I need to run a database migration and deploy to production on the same day, who do I need approval from?",
    "What is the total downtime exposure across all customer-facing services combined?",
    "Who should I contact if both the Auth service and the Checkout service are down?",
    "What were all the incidents in 2024 and who owned the affected services?",
    "Which service has the longest recovery time and what is its team lead's email?",
    "Can I deploy the Checkout service on a Friday at 3pm UTC, and if so who approves it?",
    "What is the email of the person who owns the service that had the database connection pool issue?",
    "If a developer wants to push a critical hotfix during the deployment freeze, who must they contact?",
    "What are the combined approval requirements for deploying a new service and running a migration simultaneously?",
    "Who owns the service that failed in November 2024 and how many incidents has that service had?",
    "What is the sum of RTOs for Auth, Payment Gateway, and Checkout services?",
    "Which team lead should be contacted first if both database and deployment issues occur simultaneously?",
    "What would be the total downtime if the three highest-RTO services all failed at once?",
    "Who approved the last deployment to the Payment Gateway service?",
    "What approvals are needed to deploy on a Monday versus a Friday for a critical service?",
    "If I want to do a migration this Thursday evening at 6pm UTC, what is the approval chain?",
    "Which incidents in the knowledge base involved database issues and who were the responsible owners?",
    "What is the contact information for the team lead of the service with the shortest RTO?",
    "Compare the RTOs of all customer-facing services and identify the highest risk one.",
    "What is the email of the DBA lead and when should I contact them during a migration?",
    "Who do I contact for both deployment approval and DBA sign-off, and are those the same person?",
    "What were the root causes across all three TechNova incidents and which teams were involved?",
    "If the Analytics Pipeline fails, which team lead handles it and what is their escalation path?",
    "What is the earliest I can deploy to production on a Friday and who approves it for a critical service?",
    "Which service caused the most recent outage and what was its impact on other services?",
    "What are all the required approvals if I am a developer with no team lead and need to hotfix production?",
    "Calculate the total RTO exposure across all services documented in the architecture docs.",
    "Who should I page first if both Auth service and database systems are failing simultaneously?",
    "What were the contributing factors to all 2024 incidents, and do any share a root cause?",
    "If I need to deploy three services and run two migrations in one day, walk me through the complete approval chain.",
    "What is the combined on-call contact chain for a simultaneous failure of Payment Gateway and Checkout?",
]


# ---------------------------------------------------------------------------
# STEP 2 -- GPT-4o augmentation to reach ~300 total
# ---------------------------------------------------------------------------

SIMPLE_AUGMENTATION_PROMPT = """You are generating training data for a query classifier.

Generate exactly 110 SIMPLE enterprise knowledge base queries.

SIMPLE means: the answer is in ONE document. One vector search is sufficient.
No chaining across documents. No arithmetic. No date logic needed.
Pattern: "What is X?", "Who is Y?", "Where is Z documented?", "What does A mean?"

Domain: TechNova Inc. internal engineering KB.
Topics: deployment policies, incident runbooks, service architecture, team directory, on-call procedures, SLAs.

Rules:
- Each query must be unique and realistic (something a real engineer would ask)
- No compound questions ("X and Y?")
- No aggregation ("total", "combined", "sum", "all services")
- No multi-hop ("who owns the service that caused...")
- No date-conditional logic ("this Friday", "before the cutoff")
- Vary length: some short ("What is the RTO for Auth?"), some longer

Return a JSON array of 110 strings. Nothing else."""

COMPLEX_AUGMENTATION_PROMPT = """You are generating training data for a query classifier.

Generate exactly 110 COMPLEX enterprise knowledge base queries.

COMPLEX means: answering requires ONE OR MORE of:
  - Chaining across multiple documents (A -> B -> C)
  - Aggregating numbers from multiple docs ("total RTO across all services")
  - Date/time conditional logic ("deploy this Friday at 2pm, what approvals?")
  - Cross-referencing two unrelated policies simultaneously

Domain: TechNova Inc. internal engineering KB.
Topics: deployment policies, incident runbooks, service architecture, team directory, on-call procedures, SLAs.

Rules:
- Each query must be unique and realistic
- Vary the complexity type: some multi-hop, some aggregation, some date-conditional, some compound
- Make them feel natural — what a stressed engineer would actually ask at 2am
- Some should require contacting multiple people (approval chains)
- Include "who AND email" chains, "total downtime" sums, "Friday deploy + migration" conditionals

Return a JSON array of 110 strings. Nothing else."""


def _extract_json_array(text: str) -> list:
    """Extract the first complete JSON array from text, ignoring anything after it."""
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in response")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON array, got {type(obj)}")
    return obj


def generate_augmentation(client: openai.OpenAI, prompt: str, label: str) -> list[dict]:
    """Call GPT-4o to generate augmentation examples."""
    print(f"  Generating {label} augmentation examples via GPT-4o...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        queries = _extract_json_array(raw)
        return [{"query": q.strip(), "label": label} for q in queries if isinstance(q, str) and q.strip()]
    except Exception as e:
        print(f"  [WARNING] GPT-4o augmentation failed for {label}: {e}")
        print("  Continuing with seed data only.")
        return []


def main():
    print("\n=== Query Router: Training Data Generation ===\n")

    # Build seed examples
    seeds = (
        [{"query": q, "label": "SIMPLE"}  for q in SIMPLE_SEEDS] +
        [{"query": q, "label": "COMPLEX"} for q in COMPLEX_SEEDS]
    )
    print(f"Step 1: Loaded {len(SIMPLE_SEEDS)} SIMPLE seeds + {len(COMPLEX_SEEDS)} COMPLEX seeds")
    print(f"        Total seeds: {len(seeds)}")

    # GPT-4o augmentation
    api_key = os.environ.get("OPENAI_API_KEY", "")
    all_data = seeds[:]

    if api_key:
        print("\nStep 2: GPT-4o augmentation...")
        client = openai.OpenAI(
            api_key=api_key,
            http_client=httpx.Client(verify=False),
        )
        simple_aug  = generate_augmentation(client, SIMPLE_AUGMENTATION_PROMPT,  "SIMPLE")
        complex_aug = generate_augmentation(client, COMPLEX_AUGMENTATION_PROMPT, "COMPLEX")
        all_data.extend(simple_aug)
        all_data.extend(complex_aug)
        print(f"  Added {len(simple_aug)} SIMPLE + {len(complex_aug)} COMPLEX augmented examples")
    else:
        print("\nStep 2: OPENAI_API_KEY not set -- skipping GPT-4o augmentation.")
        print("        Training on seed data only (80 examples). Accuracy will be lower.")

    # Shuffle so SIMPLE/COMPLEX are interleaved (important for training stability)
    random.seed(42)
    random.shuffle(all_data)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_data, f, indent=2)

    n_simple  = sum(1 for d in all_data if d["label"] == "SIMPLE")
    n_complex = sum(1 for d in all_data if d["label"] == "COMPLEX")
    print(f"\nStep 3: Saved {len(all_data)} examples to router/training_data.json")
    print(f"        SIMPLE: {n_simple} | COMPLEX: {n_complex}")
    print("\nNext step: python router/02_train.py\n")


if __name__ == "__main__":
    main()
