# Agentic Retrieval POC — Project Memory

> Read this file to fully understand the project without reading any other file.
> Updated whenever the architecture or design decisions change.

---

## What This Project Is (Plain English)

This project answers one question:

**"Is a single-shot RAG system good enough for complex enterprise questions — or do we need agentic retrieval?"**

We built two systems on the same knowledge base, ran the same questions through both, and compared the synthesized answers side by side.

- **System 1 — Semantic RAG (single-shot):** One fixed vector search -> chunks stuffed into a prompt -> one GPT-4o call -> answer. Fast. Limited to whatever that one search returned.
- **System 2 — Agentic Retrieval (multi-step loop):** GPT-4o in a loop. Decides what to search, searches multiple times, uses tools, reflects on what it found, synthesizes a complete answer.

---

## Why the Comparison Is Fair

Both systems use:
- The **same model**: GPT-4o
- The **same ChromaDB index**: all 16 TechNova documents
- The **same utility tools**: `calculate` (math) and `get_today_info` (date lookup)

The **ONLY difference**:

| | Semantic RAG | Agentic |
|---|---|---|
| `search_knowledge_base` | Called ONCE before LLM runs (fixed) | LLM calls it as many times as needed |
| `calculate` | YES — in the same GPT-4o call | YES |
| `get_today_info` | YES — in the same GPT-4o call | YES |

Giving `search_knowledge_base` to Semantic RAG in a loop would make it Agentic by definition. That tool IS what defines the agentic approach.

---

## The Fake Company We Built It On

Fictional company: **TechNova Inc.** with 16 internal documents across 6 categories:

| Category | Doc IDs | Contents |
|---|---|---|
| incident | INC-001, INC-002, INC-003 | Post-mortems with root causes and owners |
| architecture | ARCH-001, ARCH-002, ARCH-003, ARCH-004 | Service specs: owner, RTO, dependencies |
| policy | POL-001, POL-002, POL-003 | Deployment, DB migration, on-call policies |
| team | TEAM-001 | Full engineering directory with names and emails |
| runbook | RUN-001, RUN-002 | Step-by-step operational runbooks |
| faq | FAQ-001, FAQ-002, FAQ-003 | Common engineering FAQs |

Documents are plain Python strings in `data/corpus.py`. ChromaDB stores their vector embeddings in `chroma_db/` (auto-created, gitignored).

---

## The 4 Test Queries

### Q1 — Simple
> "What is the database migration policy?"

Answer is in ONE document (POL-002). One search finds it. Both systems equivalent.
**Expected gap:** none.

---

### Q2 — Multi-hop (chain across 3 documents)
> "Who is responsible for the service that caused the November 2024 latency incident, what was the root cause, and what is their contact email?"

Requires chaining across exactly 3 documents:
1. INC-001  → "Payment Gateway service caused it" (service name only, no owner name)
2. ARCH-001 → "Payment Gateway is owned by Sarah Chen" (owner name, no email)
3. TEAM-001 → "sarah.chen@technova.io" (email only here)

**Corpus design rule (critical):**
- Incident docs (INC-*) name the service but NOT the team lead
- Architecture docs (ARCH-*) name the team lead but NOT their email
- TEAM-001 is the ONLY document with contact emails for team leads

This forces genuine multi-hop. Without this rule, INC-001 naming Sarah Chen directly
would let both systems answer from a single doc -- no chain needed, no gap visible.

**Semantic RAG:** One search. Gets INC-001 (root cause, service name). Probably gets
ARCH-001 (owner name). Almost certainly misses TEAM-001 (email not returned by a query
about "latency incident root cause"). Answer will have name but NO email.
**Agentic:** INC-001 → "Payment Gateway" → searches for owner → ARCH-001 → "Sarah Chen"
→ searches for contact → TEAM-001 → "sarah.chen@technova.io". Complete answer.
**Expected gap:** CLEAREST gap. Same tools. Pure retrieval difference.

---

### Q3 — Aggregation (math across 2 documents)
> "Combined recovery time if both Auth AND Payment Gateway fail simultaneously?"

- ARCH-002 → Auth RTO = 10 minutes
- ARCH-001 → Payment Gateway RTO = 15 minutes
- 10 + 15 = **25 minutes**

**Semantic RAG:** Has `calculate` tool now. If both docs were retrieved (likely — both match the query), it can give the correct answer.
**Agentic:** Explicitly searches for each service, always cites both source documents.
**Expected gap:** NARROWS with fair tool access. Gap is now about reliability and traceability, not the math itself.

---

### Q4 — Conditional + date-aware
> "Deploy to Checkout this Friday afternoon. What approvals do we need?"

Requires: POL-001 (Friday freeze rule) + `get_today_info()` (is it actually Friday?) + TEAM-001 (approver contacts).

**Semantic RAG:** Has `get_today_info` now — can resolve the date condition. Gap remains if TEAM-001 was not retrieved (cannot name approvers).
**Agentic:** Searches policy, checks date, then separately searches for the team.
**Expected gap:** PARTIALLY NARROWS. Date condition closes. Approver contact may still be missing.

---

## Architecture

```
User Query
    |
    +── SEMANTIC RAG PATH ──────────────────────────────────────────────────────
    |     1. ONE vector search -> top-4 chunks from ChromaDB
    |     2. Chunks stuffed into prompt as context
    |     3. GPT-4o call with tools: [calculate, get_today_info]
    |        -> may call calculate or get_today_info if needed
    |        -> CANNOT call search_knowledge_base (not in its tools)
    |     4. Returns synthesized answer
    |
    +── AGENTIC RETRIEVAL PATH ─────────────────────────────────────────────────
          GPT-4o reasoning loop
            |
            +-- search_knowledge_base(query)    <- hits the same ChromaDB index
            +-- calculate(expression)           <- Python eval (safe char whitelist)
            +-- get_today_info()                <- UTC date + day of week
            |
            reflect: "do I have enough? what do I still need?"
            |
            synthesize: complete answer with [DOC-ID] citations
```

---

## File Structure

```
Agentic_retrieval_Poc_1/
│
├── .env                       # GITIGNORED -- your real OpenAI API key lives here
├── .env.example               # Placeholder template -- safe to commit
├── .gitignore                 # Excludes: .env, .venv, chroma_db, __pycache__, .claude
├── requirements.txt           # Pinned dependencies
│
├── run_poc.py                 # Master entry point (argparse: --semantic/--agentic/--compare)
├── run_interactive.py         # Interactive: type your question, both systems answer
│
├── 01_semantic_retrieval.py   # Shows raw vector search results (no synthesis)
├── 02_agentic_retrieval.py    # Core agentic loop: TOOLS, SYSTEM_PROMPT, run_agentic_query()
├── 03_comparison_demo.py      # Side-by-side: Semantic RAG vs Agentic on all 4 queries
│
├── data/
│   ├── __init__.py
│   └── corpus.py              # 16 TechNova documents as plain Python strings
│
└── shared/
    ├── __init__.py
    ├── kb.py                  # ChromaDB setup + semantic_search() used by all scripts
    └── display.py             # Rich terminal helpers: print_tool_call, print_final_answer, etc.
```

---

## Key Design Decisions

### Tool access (current, fair design)
- Semantic RAG tools: `calculate`, `get_today_info` (NO `search_knowledge_base`)
- Agentic tools: all three — `search_knowledge_base`, `calculate`, `get_today_info`
- Rationale: utility tools (math, date) are not retrieval. Giving `search_knowledge_base` in a loop to Semantic RAG = making it Agentic.

### Module importing (numeric filenames)
Python cannot `import 02_agentic_retrieval` directly. Use:
```python
import importlib
_ag = importlib.import_module("02_agentic_retrieval")
run_agentic_query = _ag.run_agentic_query
execute_tool = _ag.execute_tool
```
`03_comparison_demo.py` imports `execute_tool` from `02_agentic_retrieval` to reuse the same tool executor for Semantic RAG's utility tool calls.

### Corporate SSL proxy fix
```python
def make_openai_client():
    import httpx
    return openai.OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))
```
Corporate networks re-sign HTTPS traffic. Python rejects the cert. `verify=False` bypasses this. Remove if not behind a corporate proxy.

### Local embeddings (no API key for search)
Model: `all-MiniLM-L6-v2` via `chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction`.
First run downloads ~80MB model from HuggingFace and caches it. Subsequent runs load instantly.
ChromaDB persists to `chroma_db/` (gitignored, auto-created).

### Windows UTF-8
All string literals use ASCII-safe chars: `[+]`/`[-]` not ✓/✗, `->` not →.
Run commands prefix: `PYTHONIOENCODING=utf-8 python ...`

### Duplicate reasoning box fix (02_agentic_retrieval.py)
`print_agent_thought()` only fires when `tool_calls` is non-empty (intermediate steps).
On the final turn (no tool calls), only `print_final_answer()` fires — same text never appears twice.

---

## Run Commands

```bash
# Side-by-side comparison (recommended starting point)
PYTHONIOENCODING=utf-8 python 03_comparison_demo.py

# Interactive -- ask your own question
PYTHONIOENCODING=utf-8 python run_interactive.py

# Individual demos
PYTHONIOENCODING=utf-8 python 01_semantic_retrieval.py   # no API key needed
PYTHONIOENCODING=utf-8 python 02_agentic_retrieval.py

# Master entry point
PYTHONIOENCODING=utf-8 python run_poc.py              # all three
PYTHONIOENCODING=utf-8 python run_poc.py --semantic
PYTHONIOENCODING=utf-8 python run_poc.py --agentic
PYTHONIOENCODING=utf-8 python run_poc.py --compare
PYTHONIOENCODING=utf-8 python run_poc.py --rebuild-kb
```

---

## Git Workflow

Branch: `main` is always stable. All changes go on `feature/` branches.

```bash
git checkout -b feature/<description>
# make changes
git add <files>
git commit -m "type: short description\n\nLonger explanation of why."
git push -u origin feature/<description>
git checkout main
git merge --no-ff feature/<description> -m "Merge ..."
git push origin main
```

Commit type prefixes: `feat:` new feature, `fix:` bug fix, `refactor:` cleanup, `docs:` docs/comments, `chore:` config/gitignore.

---

## Key People in the Knowledge Base

| Person | Role | Email | Owns |
|---|---|---|---|
| Sarah Chen | Payments Lead | sarah.chen@technova.io | Payment Gateway (RTO 15 min) |
| David Park | Auth Lead | david.park@technova.io | Auth Service (RTO 10 min) |
| Alex Rivera | Commerce Lead | alex.rivera@technova.io | Checkout Service (RTO 20 min) |
| Priya Mehta | Data Eng Lead | priya.mehta@technova.io | Analytics Pipeline |
| Marcus Webb | Platform Reliability Lead | marcus.webb@technova.io | Infrastructure |
| Omar Hassan | DBA Lead | omar.hassan@technova.io | Databases |
| James Liu | VP Engineering | james.liu@technova.io | Must approve critical deploys |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| openai | 2.26.0 | GPT-4o via chat completions + tool calling |
| chromadb | 1.5.5 | Vector store (local, persisted) |
| sentence-transformers | 5.2.3 | Local embeddings — all-MiniLM-L6-v2 |
| rich | 14.3.3 | Terminal UI |
| python-dotenv | 1.2.0 | Load .env API key |
