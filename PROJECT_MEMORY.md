# Agentic Retrieval POC — Project Memory

> Read this file to fully understand the project without touching any other file.

---

## What This Project Is (Plain English)

This project answers one question:

**"Is normal AI search good enough for complex questions — or do we need something smarter?"**

We built two systems, ran the same questions through both, and compared the results side by side.

- **System 1 — Semantic Search (the normal way):** Ask a question, find the most similar text in the database, return it. Done. No thinking involved.
- **System 2 — Agentic Retrieval (the smarter way):** Ask a question, let an AI (Claude) *reason* about what it needs to find, search multiple times, use tools, connect the dots, then write a proper answer.

---

## The Fake Company We Built It On

To make it realistic, we created a fake company called **TechNova Inc.** and gave it 16 internal documents:

| Document Type | Examples |
|---|---|
| Incident Reports | "Payment Gateway caused a latency spike in Nov 2024" |
| Architecture Docs | "Auth service owned by David Park, RTO = 10 minutes" |
| Policies | "Deployments on Fridays after 3pm UTC are blocked" |
| Team Directory | "Sarah Chen — Payments Lead — sarah.chen@technova.io" |
| Runbooks | "If Redis fails, enable AUTH_FALLBACK_MODE=true" |
| FAQs | "Deployment freeze: Nov 24 – Dec 2 for Black Friday" |

These 16 documents are the "knowledge base" — like a company's internal wiki.

---

## The 4 Questions We Asked Both Systems

### Q1 — Simple
> "What is the database migration policy?"

The answer is in ONE document. Both systems find it easily. No difference.

---

### Q2 — Multi-hop (requires connecting 2+ documents)
> "Who is responsible for the service that caused the November 2024 latency incident?"

To answer this correctly you need to:
1. Read the incident report → "Payment Gateway caused it"
2. Read the Payment Gateway architecture doc → "owned by Sarah Chen"
3. Read the team directory → "sarah.chen@technova.io"

**Semantic search:** Returns the incident report chunk. Stops. Doesn't connect to Sarah Chen.

**Agentic retrieval:** Searches for the incident → finds Payment Gateway → searches for its owner → finds Sarah Chen → returns her name + email + root cause. Complete answer.

---

### Q3 — Aggregation (requires math across documents)
> "What is the combined recovery time if both Auth AND Payment Gateway fail simultaneously?"

Auth RTO = 10 minutes (in one doc). Payment RTO = 15 minutes (in another doc).
Answer = 10 + 15 = **25 minutes**.

**Semantic search:** Returns both documents. The user has to manually read, find the numbers, and add them.

**Agentic retrieval:** Retrieves both docs, extracts the numbers, calls a calculator tool, returns "25 minutes."

---

### Q4 — Conditional + Date-aware
> "We want to deploy to Checkout this Friday afternoon. What approvals do we need?"

To answer: read deployment policy + check what day of week Friday is + check the Friday cutoff (3pm UTC) + look up who the approvers are.

**Semantic search:** Returns the deployment policy doc. Doesn't know it's Friday. Doesn't flag the time cutoff. Doesn't pull the contact names.

**Agentic retrieval:** Checks today's date, reads the policy, identifies the Friday 15:00 UTC blocker, looks up Alex Rivera + Marcus Webb + James Liu, tells you the deployment is likely blocked and who to call.

---

## The Score

| Query | Semantic | Agentic |
|---|---|---|
| Q1 — Simple | 7/10 | 8/10 |
| Q2 — Multi-hop | 3/10 | 9/10 |
| Q3 — Aggregation | 2/10 | 9/10 |
| Q4 — Conditional | 3/10 | 9/10 |
| **Total** | **15/40** | **35/40** |

---

## What Each File Does

```
run_poc.py                   ← Start here. Runs all 3 demos.
01_semantic_retrieval.py     ← Demo 1: shows normal vector search results
02_agentic_retrieval.py      ← Demo 2: shows Claude reasoning + tool calls live
03_comparison_demo.py        ← Demo 3: runs both on same queries, shows scorecard

data/corpus.py               ← The 16 TechNova documents (the "knowledge base")
shared/kb.py                 ← Sets up ChromaDB + local AI embeddings
shared/display.py            ← Terminal display helpers (colors, tables, panels)

chroma_db/                   ← Auto-created folder where vectors are stored
.env.example                 ← Copy this to .env and add your API key
```

---

## How to Run It

### Step 1 — Set your API key
```
Copy .env.example → rename to .env
Open .env → add your Anthropic API key:  ANTHROPIC_API_KEY=sk-ant-...
```

### Step 2 — Run semantic search (no API key needed)
```bash
python 01_semantic_retrieval.py
```

### Step 3 — Run agentic retrieval (needs API key)
```bash
python 02_agentic_retrieval.py
```

### Step 4 — See the side-by-side comparison
```bash
python 03_comparison_demo.py
```

---

## Technical Stack (What Powers It)

| Component | Technology | Why |
|---|---|---|
| Vector store | ChromaDB | Stores and searches document embeddings |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local model, no API key needed |
| Agent brain | Claude claude-sonnet-4-6 (Anthropic) | Reasons, decomposes queries, calls tools |
| Terminal UI | Rich | Colored panels, tables, live output |
| Language | Python 3.11 | — |

---

## Key People in the Fake Knowledge Base

| Person | Role | Email | Owns |
|---|---|---|---|
| Sarah Chen | Payments Lead | sarah.chen@technova.io | Payment Gateway (RTO 15 min) |
| David Park | Auth Lead | david.park@technova.io | Auth Service (RTO 10 min) |
| Alex Rivera | Commerce Lead | alex.rivera@technova.io | Checkout Service (RTO 20 min) |
| Priya Mehta | Data Eng Lead | priya.mehta@technova.io | Analytics Pipeline |
| Marcus Webb | Platform Reliability Lead | marcus.webb@technova.io | Infrastructure |
| Omar Hassan | DBA Lead | omar.hassan@technova.io | Databases |
| James Liu | VP Engineering | james.liu@technova.io | Must approve critical deploys |

