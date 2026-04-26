# Agentic Retrieval vs Semantic Retrieval — POC

A working, runnable proof-of-concept that demonstrates **why agentic retrieval beats plain
semantic (vector) search** on complex enterprise queries — with every step of the reasoning
visible in the terminal.

---

## What this project shows

| Capability | Semantic Search | Agentic Retrieval |
|---|---|---|
| Find a fact in one document | YES | YES |
| Chain facts across multiple documents (multi-hop) | NO | YES |
| Add numbers from separate documents (aggregation) | NO | YES |
| Apply a policy rule based on today's date (conditional) | NO | YES |
| Explain *why* it gave the answer | NO | YES |

Semantic search returns the closest-matching text chunks — fast and simple. It cannot reason,
chain, or calculate. Agentic retrieval puts a large language model (GPT-4o) in a loop where
it decides what to search, calls tools, reflects on what it found, and synthesises a complete
answer with citations.

---

## Architecture

```
User Query
    |
    +----> SEMANTIC PATH
    |        embed(query) -> cosine-similarity(ChromaDB) -> top-K chunks -> return raw text
    |
    +----> AGENTIC PATH
             GPT-4o reasoning loop
               |
               +-- search_knowledge_base(sub_query)   <- hits ChromaDB (same index)
               +-- calculate(expression)              <- Python eval (safe whitelist)
               +-- get_today_info()                   <- UTC date + day of week
               |
               reflect: "do I have enough? what do I still need?"
               |
               synthesize: full natural-language answer with [DOC-ID] citations
```

The **same ChromaDB index** is used by both paths. The difference is entirely in whether an
LLM is reasoning about what to search for and what to do with the results.

---

## Knowledge Base (TechNova Inc.)

16 plain-text documents about a fictional company, split across 6 categories:

| Category | Doc IDs | Contents |
|---|---|---|
| incident | INC-001, INC-002, INC-003 | Incident post-mortems with root causes and owners |
| architecture | ARCH-001, ARCH-002, ARCH-003, ARCH-004 | Service specs: owner, RTO, dependencies |
| policy | POL-001, POL-002, POL-003 | Deployment, DB migration, on-call policies |
| team | TEAM-001 | Full engineering directory with names and emails |
| runbook | RUN-001, RUN-002 | Step-by-step ops runbooks |
| faq | FAQ-001, FAQ-002, FAQ-003 | Common engineering FAQs |

The corpus is deliberately designed so that:
- **Q1** (DB migration policy) is answerable from one document — semantic works fine
- **Q2** (who caused the Nov 2024 incident?) requires INC-001 -> ARCH-001 -> TEAM-001 — semantic misses the chain
- **Q3** (combined RTO if Auth + Payment fail) requires two docs + arithmetic — semantic returns numbers but cannot add them
- **Q4** (deploy this Friday afternoon?) requires a policy + today's date + approver lookup — semantic finds the policy but cannot apply conditional logic

---

## Query Types Explained

**SIMPLE** — Answer lives in one document. One search is enough. Both systems handle this equally.

**MULTI-HOP** — Answer requires following a chain across 2+ documents.
Example: _Incident doc → "Payment Gateway caused it" → Architecture doc → "owned by Sarah Chen" → Team directory → "sarah.chen@technova.io"_
Semantic search cannot use result #1 to decide what to search for next. Agentic search can.

**AGGREGATION** — Answer requires collecting numbers from multiple documents and doing arithmetic.
Example: _Auth RTO (10 min) + Payment RTO (15 min) = 25 min combined._
Semantic returns both chunks but cannot add them. The agentic system calls a `calculate` tool.

**CONDITIONAL** — Answer depends on IF/THEN logic combined with real-world state (today's date).
Example: _"Can we deploy Friday afternoon?" = policy says no after 15:00 UTC + check today's day/time._
Semantic finds the policy text. It has no concept of "today" and cannot resolve the condition.

---

## File Structure

```
Agentic_retrieval_Poc_1/
│
├── .env.example               # Template — copy to .env and add your OpenAI key
├── .gitignore                 # Excludes .env, .venv, chroma_db, __pycache__
├── requirements.txt           # All Python dependencies
│
├── run_poc.py                 # Master entry point (run all demos, or individual ones)
├── run_interactive.py         # Type your own question, see both systems answer live
│
├── 01_semantic_retrieval.py   # Demo 1: pure vector similarity baseline (no API key needed)
├── 02_agentic_retrieval.py    # Demo 2: GPT-4o agentic loop with tools
├── 03_comparison_demo.py      # Demo 3: both systems on the same 4 queries, side by side
│
├── data/
│   ├── __init__.py
│   └── corpus.py              # 16-document TechNova knowledge corpus (plain Python strings)
│
└── shared/
    ├── __init__.py
    ├── kb.py                  # ChromaDB setup + semantic_search() used by all demos
    └── display.py             # Rich terminal display helpers shared by all scripts
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Purushottam-Svam/Agentic_retrieval_POC.git
cd Agentic_retrieval_POC

python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env and replace sk-proj-YOUR_OPENAI_KEY_HERE with your real key
```

Get a key at https://platform.openai.com/api-keys

### 3. First run downloads the embedding model

The local embedding model (`all-MiniLM-L6-v2`, ~80 MB) is downloaded from HuggingFace on the
first run and cached by `sentence-transformers`. No API key needed for embeddings.

---

## Running the demos

> **Windows users:** prefix every command with `PYTHONIOENCODING=utf-8` to avoid encoding errors.
> Example: `PYTHONIOENCODING=utf-8 python 03_comparison_demo.py`

### Option A — Best demo: side-by-side comparison (recommended starting point)

```bash
PYTHONIOENCODING=utf-8 python 03_comparison_demo.py
```

For each of the 4 queries you will see:
1. **Semantic section** — every retrieved document chunk printed in full
2. **Agentic section** — the live chain of thought: every reasoning step, every tool call with arguments, every tool result
3. **Comparison** — both final outputs side by side with a plain-English verdict

### Option B — Interactive: type your own question

```bash
PYTHONIOENCODING=utf-8 python run_interactive.py
```

### Option C — Individual demos

```bash
# Semantic baseline only (no OpenAI key needed)
PYTHONIOENCODING=utf-8 python 01_semantic_retrieval.py

# Agentic only
PYTHONIOENCODING=utf-8 python 02_agentic_retrieval.py
```

### Option D — Master entry point

```bash
PYTHONIOENCODING=utf-8 python run_poc.py              # all three in sequence
PYTHONIOENCODING=utf-8 python run_poc.py --semantic   # semantic only
PYTHONIOENCODING=utf-8 python run_poc.py --agentic    # agentic only
PYTHONIOENCODING=utf-8 python run_poc.py --compare    # comparison only
PYTHONIOENCODING=utf-8 python run_poc.py --rebuild-kb # force-rebuild ChromaDB index
```

---

## How agentic retrieval works (step by step)

```
1. User sends query
2. GPT-4o reads the query and the system prompt
3. GPT-4o decides: "this is complex, I need to search for X first"
4. GPT-4o emits a tool_call: search_knowledge_base(query="X")
5. Our code runs the ChromaDB search and returns the chunks
6. GPT-4o reads the results and reflects: "I found Y, but I still need Z"
7. GPT-4o emits another tool_call: search_knowledge_base(query="Z")
8. ... continues until it has everything
9. If arithmetic needed: tool_call: calculate("10 + 15")
10. If date needed:      tool_call: get_today_info()
11. GPT-4o synthesizes a complete answer with [DOC-ID] citations
12. Loop exits when GPT-4o returns finish_reason="stop"
```

Every step 3-11 is printed to the terminal in the demo so you can watch the reasoning happen.

---

## Key people in the corpus (for multi-hop queries)

| Name | Role | Email | Owns |
|---|---|---|---|
| Sarah Chen | Payments Team Lead | sarah.chen@technova.io | Payment Gateway (RTO 15 min) |
| David Park | Auth Team Lead | david.park@technova.io | Auth Service (RTO 10 min) |
| Alex Rivera | Commerce Team Lead | alex.rivera@technova.io | Checkout Service (RTO 20 min) |
| Marcus Webb | Platform Reliability Lead | marcus.webb@technova.io | Infrastructure |
| Omar Hassan | DBA Lead | omar.hassan@technova.io | Database policies |
| James Liu | VP Engineering | james.liu@technova.io | Must approve critical deploys |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| openai | 2.26.0 | GPT-4o chat completions + tool calling |
| chromadb | 1.5.5 | Vector database (local, persisted to disk) |
| sentence-transformers | 5.2.3 | Local embeddings — `all-MiniLM-L6-v2` model |
| rich | 14.3.3 | Terminal UI (panels, tables, colors) |
| python-dotenv | 1.2.0 | Load API key from `.env` file |

---

## Notes

- ChromaDB is stored locally in `chroma_db/` (gitignored). On first run it is created
  automatically and reused on subsequent runs.
- The `verify=False` flag in `make_openai_client()` is for corporate networks where an SSL
  proxy re-signs HTTPS traffic. Remove it if you are not behind such a proxy.
- Python cannot `import` modules with numeric-leading filenames directly. Scripts that need
  `02_agentic_retrieval` use `importlib.import_module("02_agentic_retrieval")`.
