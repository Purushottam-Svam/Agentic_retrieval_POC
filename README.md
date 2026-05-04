# Agentic Retrieval vs Semantic Retrieval — POC

A working, runnable proof-of-concept that demonstrates **why agentic retrieval beats plain
semantic (vector) search** on complex enterprise queries — with every step of the reasoning
visible in the terminal.

---

## Project Progress

This project was built in three phases, each adding a layer on top of the previous one.

### Phase 1 — Semantic RAG Baseline ✅
Built a pure vector-search pipeline: embed query → cosine similarity in ChromaDB → return top-K chunks → pass to GPT-4o for synthesis. This is the standard RAG approach. Fast (2–3s) and cheap (1 LLM call), but cannot reason, chain, or calculate. Serves as the baseline to beat.

### Phase 2 — Agentic Retrieval Loop ✅
Replaced the single-shot search with a GPT-4o reasoning loop that decides what to search, calls tools iteratively, reflects on what it found, and synthesises a complete answer with citations. Three tools available: `search_knowledge_base`, `calculate`, `get_today_info`. This handles multi-hop, aggregation, and date-conditional queries that semantic RAG cannot. Costs 3–6x more tokens on simple questions, but gives the right answer on hard ones.

**Side-by-side evaluation across 4 designed queries:**

| Query Type | Semantic RAG | Agentic |
|---|---|---|
| Q1 — Simple policy lookup | ✅ Correct | ✅ Correct |
| Q2 — Multi-hop (incident → service → email) | ❌ Missed email | ✅ Found full chain |
| Q3 — Aggregation (combined RTO across services) | ❌ Wrong total | ✅ Correct math |
| Q4 — Conditional (Friday deploy + migration approvals) | ❌ Incomplete | ✅ Full approval chain |

### Phase 3 — Query Router (ML Classifier) ✅
The problem with Phase 2: sending every query to the Agentic loop wastes 5–7x tokens on simple questions with zero accuracy gain. Built a lightweight **DistilBERT binary classifier** that reads the query and decides in ~8ms whether to route it to Semantic RAG (SIMPLE) or the Agentic loop (COMPLEX). The model is fine-tuned on 281 labeled examples generated with GPT-4o augmentation.

**Router performance:**

| Metric | Result |
|---|---|
| Validation accuracy | **100%** (57/57) |
| False negatives (COMPLEX routed to SIMPLE — dangerous) | **0** |
| 50-query test suite accuracy | **92%** (46/50) |
| Inference latency | **~8ms per query** |
| Router accuracy on 4 standard queries | **4/4 (100%)** |

The 4 mismatches in the test suite are all SIMPLE queries over-routed to COMPLEX (safe overroute — wastes tokens but never gives an incomplete answer).

---

## What this project shows

| Capability | Semantic Search | Agentic Retrieval |
|---|---|---|
| Find a fact in one document | YES | YES |
| Chain facts across multiple documents (multi-hop) | NO | YES |
| Add numbers from separate documents (aggregation) | NO | YES |
| Apply a policy rule based on today's date (conditional) | NO | YES |
| Explain *why* it gave the answer | NO | YES |
| Route query intelligently to save tokens | NO | YES (via router) |

---

## Architecture

### Without Router (Phase 1 & 2)
```
User Query
    |
    +----> SEMANTIC PATH
    |        embed(query) -> cosine-similarity(ChromaDB) -> top-K chunks -> GPT-4o synthesis
    |
    +----> AGENTIC PATH
             GPT-4o reasoning loop
               |
               +-- search_knowledge_base(sub_query)
               +-- calculate(expression)
               +-- get_today_info()
               |
               reflect -> synthesize -> answer with [DOC-ID] citations
```

### With Router (Phase 3)
```
User Query
    |
    v
+---------------------------+
|  QueryRouter (~8ms)       |   DistilBERT fine-tuned classifier
|  router/router_model.py   |   threshold = 0.65
+----------+----------------+
           |
    +------+-------+
 SIMPLE           COMPLEX / UNCERTAIN
    |                   |
    v                   v
Semantic RAG       Agentic Loop
(1 search,         (N searches,
 1 LLM call,        reflection,
 ~2s, cheap)        synthesis,
                    ~5-20s, accurate)
    |                   |
    +--------+----------+
             v
           Answer
```

The router eliminates wasteful agentic calls on simple lookups while ensuring
complex queries always get the full reasoning treatment.

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
- **Q2** (who caused the Nov 2024 incident?) requires INC-001 → ARCH-001 → TEAM-001 — semantic misses the chain
- **Q3** (combined RTO if all services fail) requires three docs + arithmetic — semantic returns numbers but cannot add them
- **Q4** (deploy this Friday afternoon?) requires policy + today's date + approver lookup — semantic finds the policy but cannot apply conditional logic

---

## Query Types Explained

**SIMPLE** — Answer lives in one document. One search is enough. Both systems handle this equally. Router sends to Semantic RAG.

**MULTI-HOP** — Answer requires following a chain across 2+ documents.
Example: _Incident doc → "Payment Gateway caused it" → Architecture doc → "owned by Sarah Chen" → Team directory → "sarah.chen@technova.io"_

**AGGREGATION** — Answer requires collecting numbers from multiple documents and doing arithmetic.
Example: _Auth RTO (10 min) + Payment RTO (15 min) + Checkout RTO (20 min) = 45 min total._

**CONDITIONAL** — Answer depends on IF/THEN logic combined with real-world state (today's date).
Example: _"Can we deploy Friday afternoon?" = policy says no after 15:00 UTC + check today's day/time._

---

## File Structure

```
Agentic_retrieval_Poc_1/
│
├── .env.example               # Template — copy to .env and add your OpenAI key
├── .gitignore                 # Excludes .env, .venv, chroma_db, router/model/
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
├── shared/
│   ├── __init__.py
│   ├── kb.py                  # ChromaDB setup + semantic_search() used by all demos
│   └── display.py             # Rich terminal display helpers shared by all scripts
│
└── router/                    # Phase 3: Query Router (standalone ML package)
    ├── __init__.py
    ├── router_model.py        # QueryRouter class — loads model, classifies in ~8ms
    ├── 01_generate_data.py    # Step 1: generate training data via GPT-4o augmentation
    ├── 02_train.py            # Step 2: fine-tune DistilBERT, saves to router/model/
    ├── 03_evaluate.py         # Step 3: evaluate router + both retrieval systems end-to-end
    ├── extra_test_queries.py  # 50-query standalone test suite (no API key needed)
    ├── training_data.json     # 281 labeled queries (144 SIMPLE / 137 COMPLEX)
    ├── extra_test_outputs.json# Latest test suite results
    ├── ROUTER_GUIDE.md        # Deep-dive: architecture, training, fine-tuning explained
    └── model/                 # [gitignored] trained weights — run 02_train.py to generate
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

## Running the Query Router (Phase 3)

The router is a standalone package — it does not change any existing file.

> **Run all commands from the project root**, not from inside `router/`.

### Step 1 — Generate training data (~30s, costs ~$0.05)

```bash
python router/01_generate_data.py
```

Calls GPT-4o to generate ~220 augmented examples on top of 80 hardcoded seeds.
Saves 281 labeled queries to `router/training_data.json` (144 SIMPLE / 137 COMPLEX).

### Step 2 — Train the model (~15 min on CPU, free)

```bash
python router/02_train.py
```

Fine-tunes DistilBERT for up to 5 epochs with early stopping.
Saves the best checkpoint to `router/model/` (gitignored — train once locally).

Expected result:
```
              precision    recall  f1-score   support
      SIMPLE       1.00      1.00      1.00        35
     COMPLEX       1.00      1.00      1.00        22
    accuracy                           1.00        57
False negatives (COMPLEX routed to SIMPLE): 0/22
[OK] Training complete. Model saved to router/model/
```

### Step 3 — Full end-to-end evaluation (~5 min, costs ~$0.20)

```bash
python router/03_evaluate.py
```

Runs both retrieval systems on all 4 standard queries, scores answer richness,
and shows whether the router made the right call for each.

### Quick standalone test (no API key, no KB, ~instant)

```bash
python router/extra_test_queries.py
```

Runs 50 queries through the trained router only. Shows classification and
confidence for each. Useful to verify the model is working after training.

---

## How agentic retrieval works (step by step)

```
1.  User sends query
2.  GPT-4o reads the query and the system prompt
3.  GPT-4o decides: "this is complex, I need to search for X first"
4.  GPT-4o emits a tool_call: search_knowledge_base(query="X")
5.  Our code runs the ChromaDB search and returns the chunks
6.  GPT-4o reads the results and reflects: "I found Y, but I still need Z"
7.  GPT-4o emits another tool_call: search_knowledge_base(query="Z")
8.  ... continues until it has everything
9.  If arithmetic needed: tool_call: calculate("10 + 15 + 20")
10. If date needed:       tool_call: get_today_info()
11. GPT-4o synthesizes a complete answer with [DOC-ID] citations
12. Loop exits when GPT-4o returns finish_reason="stop"
```

Every step 3–11 is printed to the terminal in the demo so you can watch the reasoning happen.

---

## How the Query Router works

The router is a fine-tuned **DistilBERT** model (66 MB) — a smaller, faster version of BERT
that runs in ~8ms on CPU.

```
Query text
    |
    v
Tokenizer (DistilBERT subword tokens)
    |
    v
6 Transformer layers (self-attention)
    |
    v
[CLS] token vector (768 numbers — entire sentence meaning)
    |
    v
Linear(768 → 2)    ← the only new weights trained
    |
    v
Softmax → [p_simple, p_complex]
    |
    v
Threshold (0.65):
  p_complex >= 0.65  → COMPLEX
  p_simple  >= 0.65  → SIMPLE
  neither            → UNCERTAIN → defaults to COMPLEX (safer)
```

Training used only the final classification head. DistilBERT's 66M pre-trained
weights were kept frozen — we just taught it a new two-way decision on top of
language understanding it already had.

---

## Key people in the corpus (for multi-hop queries)

| Name | Role | Email | Owns |
|---|---|---|---|
| Sarah Chen | Payments Team Lead | sarah.chen@technova.io | Payment Gateway (RTO 15 min) |
| David Park | Auth Team Lead | david.park@technova.io | Auth Service (RTO 10 min) |
| Alex Rivera | Commerce Team Lead | alex.rivera@technova.io | Checkout Service (RTO 20 min) |
| Priya Mehta | Data Engineering Lead | priya.mehta@technova.io | Analytics Pipeline (RTO 30 min) |
| Marcus Webb | Platform Reliability Lead | marcus.webb@technova.io | Infrastructure |
| Omar Hassan | DBA Lead | omar.hassan@technova.io | Database policies |
| James Liu | VP Engineering | james.liu@technova.io | Must approve critical deploys |

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | GPT-4o chat completions + tool calling |
| `chromadb` | Vector database (local, persisted to disk) |
| `sentence-transformers` | Local embeddings — `all-MiniLM-L6-v2` |
| `rich` | Terminal UI (panels, tables, colors) |
| `python-dotenv` | Load API key from `.env` file |
| `torch` | PyTorch — required for DistilBERT inference and training |
| `transformers` | HuggingFace Transformers — DistilBERT model + tokenizer |
| `datasets` | HuggingFace Datasets — required by Trainer in `02_train.py` |
| `accelerate` | HuggingFace Accelerate — required by HuggingFace Trainer |
| `scikit-learn` | Classification report + confusion matrix in `02_train.py` |

---

## Notes

- ChromaDB is stored locally in `chroma_db/` (gitignored). On first run it is created
  automatically and reused on subsequent runs.
- The router model weights (`router/model/`) are gitignored — they are ~256 MB each.
  Clone the repo and run `python router/02_train.py` once to generate them locally (~15 min).
- The `verify=False` flag in `make_openai_client()` is for corporate networks where an SSL
  proxy re-signs HTTPS traffic. Remove it if you are not behind such a proxy.
- Python cannot `import` modules with numeric-leading filenames directly. Scripts that need
  `02_agentic_retrieval` use `importlib.import_module("02_agentic_retrieval")`.
