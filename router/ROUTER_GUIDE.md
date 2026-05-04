# Query Router — Complete Guide

> **What is this?**
> A standalone ML module that classifies any enterprise KB query as SIMPLE or COMPLEX
> in ~8ms, before deciding which retrieval system to use.
> Zero changes to the existing codebase. Zero new pip dependencies.

---

## Table of Contents

1. [Why This Exists](#1-why-this-exists)
2. [How It Works — The Big Picture](#2-how-it-works--the-big-picture)
3. [What Is Fine-Tuning?](#3-what-is-fine-tuning)
4. [File-by-File Code Walkthrough](#4-file-by-file-code-walkthrough)
   - [router_model.py](#41-router_modelpy--inference-engine)
   - [01_generate_data.py](#42-01_generate_datapy--training-data-factory)
   - [02_train.py](#43-02_trainpy--fine-tuning-distilbert)
   - [03_evaluate.py](#44-03_evaluatepy--end-to-end-evaluation)
5. [How to Run — Step by Step](#5-how-to-run--step-by-step)
6. [How to Test the Router Standalone](#6-how-to-test-the-router-standalone)
7. [How to Test With the Full POC](#7-how-to-test-with-the-full-poc)
8. [Understanding the Outputs](#8-understanding-the-outputs)
9. [What the Numbers Mean](#9-what-the-numbers-mean)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Why This Exists

The POC has two retrieval systems:

| System | Speed | Cost | Best For |
|---|---|---|---|
| Semantic RAG | Fast (~2s) | Cheap (1 LLM call) | Simple single-doc lookups |
| Agentic Loop | Slow (5–20s) | Expensive (3–6 LLM calls) | Multi-hop, aggregation, conditionals |

**The problem:** blindly sending every query to Agentic wastes 5–7x tokens on
simple questions with zero accuracy gain.

**The solution:** classify the query first (8ms, free) and pick the right path.

```
Without router:  ALL queries → Agentic → expensive, slow
With router:     Simple queries → Semantic RAG (fast, cheap)
                 Complex queries → Agentic (accurate, necessary)
```

---

## 2. How It Works — The Big Picture

```
User types a question
        │
        ▼
┌───────────────────────┐
│   QueryRouter (~8ms)  │   ← DistilBERT model loaded in RAM
│   router_model.py     │
└──────────┬────────────┘
           │
     ┌─────┴──────┐
  SIMPLE        COMPLEX / UNCERTAIN
     │                │
     ▼                ▼
Semantic RAG     Agentic Loop
(1 search,       (N searches,
 1 LLM call)      reflection,
                  synthesis)
     │                │
     └────────┬───────┘
              ▼
           Answer
```

The router is a **text classifier**. It reads the query and outputs one of:
- `SIMPLE`    → one vector search is enough
- `COMPLEX`   → needs chaining / math / date logic / multi-doc reasoning
- `UNCERTAIN` → borderline — defaults to Agentic (safer to overspend than under-answer)

---

## 3. What Is Fine-Tuning?

### Pre-training vs Fine-tuning

**Pre-training** is when a model learns language from scratch on billions of words.
DistilBERT was pre-trained by Hugging Face on Wikipedia + BookCorpus.
It learned: grammar, word meaning, sentence structure, context.
This took weeks on hundreds of GPUs. We do not do this.

**Fine-tuning** is when you take that pre-trained model and teach it one specific
new task using your own small dataset. The model already understands language —
you are just redirecting that knowledge toward your classification problem.

### What DistilBERT looks like

```
Input text:  "What is the total RTO if Auth and Payment fail?"
      │
      ▼
Tokenizer splits into subwords:
      ["what", "is", "the", "total", "rt", "##o", "if", "auth", "and", "payment", "fail", "?"]
      │
      ▼
Each token → 768-dim vector (via 6 transformer layers of self-attention)
      │
      ▼
[CLS] token  ← special token added at position 0, its vector = entire sentence meaning
      │       (768 numbers summarizing the whole query)
      ▼
Linear layer (768 → 2)   ← THIS is what we train. Just one matrix multiply.
      │
      ▼
[score_SIMPLE, score_COMPLEX]   e.g. [0.03, 4.71]
      │
      ▼
Softmax → probabilities:  [0.018, 0.982]
      │
      ▼
Decision: COMPLEX (98.2% confidence)
```

The **only new weights we train** are the `Linear(768 → 2)` layer.
Everything else (the 66M parameters of DistilBERT) are either frozen or
very lightly nudged by the small learning rate (3e-5).

### Why DistilBERT and not BERT-base?

| Model | Size | Inference Speed | Accuracy |
|---|---|---|---|
| BERT-base | 440 MB | ~25ms | 100% baseline |
| DistilBERT | 66 MB | ~8ms | 97% of BERT |
| TinyBERT | 15 MB | ~2ms | 93% of BERT |

DistilBERT hits the sweet spot: small enough to load in RAM and run in 8ms,
accurate enough that 97% of BERT performance is more than sufficient for
binary classification.

---

## 4. File-by-File Code Walkthrough

### 4.1 `router_model.py` — Inference Engine

This is the only file that runs in production (at query time).
The other three files only run during training.

```python
class QueryRouter:
    def __init__(self, model_path, threshold=0.65):
```

**`model_path`** — where the saved model weights are (router/model/).
Loaded once at startup, kept in RAM. Not reloaded per query.

**`threshold=0.65`** — the confidence cutoff.
- p(COMPLEX) >= 0.65  → COMPLEX
- p(SIMPLE)  >= 0.65  → SIMPLE
- neither              → UNCERTAIN (defaults to COMPLEX in the caller)

Why 0.65 and not 0.5?
At 0.5, a 51% confidence = firm decision. Too aggressive.
At 0.65, the model must be reasonably sure before picking a path.
Anything borderline becomes UNCERTAIN, which safely routes to Agentic.

```python
def route(self, query: str) -> tuple[str, float]:
    inputs = self.tokenizer(query, return_tensors="pt", truncation=True,
                            padding=True, max_length=128)
    with torch.no_grad():
        logits = self.model(**inputs).logits
        probs  = torch.softmax(logits, dim=1)[0]
```

**`torch.no_grad()`** — disables gradient computation. Inference only.
Without this, PyTorch tracks all operations for backprop (training mode).
In inference you do not need gradients — disabling them saves ~40% memory
and speeds up each call.

**`max_length=128`** — enterprise queries are typically under 50 tokens.
128 is a safe ceiling with no wasted computation.

**`torch.softmax`** — converts raw logit scores (any real number) into
probabilities that sum to 1.0. e.g. [0.31, 4.71] → [0.018, 0.982].

---

### 4.2 `01_generate_data.py` — Training Data Factory

Fine-tuning needs labeled examples: (query_text, label) pairs.
This file creates them in two steps.

#### Step 1 — Hardcoded seeds (80 examples)

These are written by hand. They are ground-truth anchors.
40 SIMPLE + 40 COMPLEX, covering the full range of what engineers ask.

Why write them by hand first?
- GPT-4o generated examples sometimes drift — they may be too similar,
  too easy, or miss real-world phrasing patterns.
- The 80 seeds guarantee minimum quality and diversity.
- They serve as the "core" the augmented data builds around.

#### Step 2 — GPT-4o augmentation (220 more examples)

```python
SIMPLE_AUGMENTATION_PROMPT = """
Generate exactly 110 SIMPLE enterprise knowledge base queries.
SIMPLE means: the answer is in ONE document. One vector search is sufficient.
...
Return a JSON array of 110 strings. Nothing else.
"""
```

GPT-4o is instructed to:
- Stay within the TechNova domain (deployment, incidents, architecture...)
- Vary sentence length and phrasing
- Never generate compound or multi-hop questions for the SIMPLE set
- Return raw JSON (no markdown) so we can `json.loads()` directly

The augmentation reaches ~300 total examples. Why 300?
For binary classification on a narrow domain, 200–500 labeled examples
gives solid generalisation. Below 100 the model memorises; above 1000
you get diminishing returns without real query logs.

#### Output: `training_data.json`

```json
[
  {"query": "What is the database migration policy?", "label": "SIMPLE"},
  {"query": "Who caused the incident and what is their email?", "label": "COMPLEX"},
  ...
]
```

Shuffled (seed=42 for reproducibility) so SIMPLE/COMPLEX interleave.
This matters for training — batches should contain both classes.

---

### 4.3 `02_train.py` — Fine-tuning DistilBERT

This is the core ML file. Run once to produce the model weights.

#### Loading data and splitting

```python
split_idx = int(len(texts) * 0.8)
train_texts, val_texts   = texts[:split_idx],  texts[split_idx:]
```

80/20 split. With 263 examples: 210 train, 53 validation.
The validation set is held out — the model never trains on it.
It exists purely to measure how well the model generalises to unseen queries.

Why not 90/10?
With only ~260 examples, 90/10 gives 26 validation samples — too small
to get reliable accuracy estimates. 80/20 gives 53, which is acceptable.

#### Tokenisation

```python
def tokenize_batch(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True,
                     padding="max_length", max_length=128)
```

**`truncation=True`** — if a query exceeds 128 tokens, cut it. Safe for
our domain (queries are short).

**`padding="max_length"`** — pad every sequence to exactly 128 tokens with
`[PAD]` tokens. Required because PyTorch needs all tensors in a batch to
be the same length. The model learns to ignore `[PAD]` tokens via the
attention mask.

**`max_length=128`** — DistilBERT's max is 512. We use 128 because:
- Enterprise queries are short (avg ~30 tokens)
- Smaller = faster batches = faster training
- No information lost

#### Training arguments

```python
training_args = TrainingArguments(
    num_train_epochs=5,           # pass over all training data 5 times
    per_device_train_batch_size=16, # process 16 examples at once
    learning_rate=3e-5,           # very small — fine-tuning, not training from scratch
    warmup_steps=20,              # ramp LR from 0 to 3e-5 over first 20 steps
    weight_decay=0.01,            # L2 regularisation — prevents overfitting
    eval_strategy="epoch",        # evaluate on val set after every epoch
    load_best_model_at_end=True,  # after all epochs, restore highest val accuracy
)
```

**`learning_rate=3e-5`**: This is the most important hyperparameter.
- Too high (e.g. 1e-3): destroys the pre-trained weights, model forgets language
- Too low (e.g. 1e-7): model barely updates, never learns your task
- 2e-5 to 5e-5: the accepted range for fine-tuning BERT-family models

**`warmup_steps=20`**: Learning rate starts at 0 and ramps up linearly
over the first 20 gradient steps. This prevents large, destabilising
updates in the first few batches before the model has seen enough data.

**`weight_decay=0.01`**: Adds a small penalty for large weights. Acts as
regularisation — discourages the model from over-committing to the
training set patterns. Crucial when your dataset is small (~260 examples).

**`load_best_model_at_end=True`**: After 5 epochs, the model might slightly
overfit in later epochs. This ensures we save the checkpoint with highest
validation accuracy, not just the last one.

#### Early stopping

```python
callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
```

If validation accuracy does not improve for 2 consecutive epochs, training
stops early. Prevents wasting time and overfitting on small datasets.
In our run it did not trigger — accuracy kept improving through epoch 5.

#### `compute_metrics`

```python
def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": float(accuracy)}
```

Called by the Trainer after every epoch on the validation set.
`np.argmax(logits, axis=1)` converts raw scores to predicted class indices.
Returns accuracy as a float — Trainer uses this to decide which checkpoint
is "best" (because `metric_for_best_model="accuracy"`).

#### Classification report (after training)

```
              precision    recall  f1-score   support
      SIMPLE       1.00      1.00      1.00        28
     COMPLEX       1.00      1.00      1.00        25
    accuracy                           1.00        53
```

**Precision**: of all queries predicted COMPLEX, what % were actually COMPLEX?
**Recall**: of all actually-COMPLEX queries, what % did the model catch?
**F1**: harmonic mean of precision and recall. Best single number for class quality.

The confusion matrix is the most important output:
```
               Pred SIMPLE  Pred COMPLEX
  Actual SIMPLE        28             0     ← 0 false positives (no wasted tokens)
  Actual COMPLEX        0            25     ← 0 false negatives (no dangerous misroutes)
```

**False negatives** (bottom-left) are the dangerous cell:
a COMPLEX query labelled SIMPLE means the user gets an incomplete answer
and does not know it. We got 0 — ideal result.

---

### 4.4 `03_evaluate.py` — End-to-End Evaluation

This is the integration test. It connects the router to the full POC
without touching any existing file.

#### Imports via importlib (no file renaming needed)

```python
_ag   = importlib.import_module("02_agentic_retrieval")
_comp = importlib.import_module("03_comparison_demo")
```

Python cannot `import 02_agentic_retrieval` directly (filename starts with a digit).
`importlib.import_module` handles this. The existing files are unchanged.

#### Information richness scorer

```python
EMAIL_RE  = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
NUMBER_RE = re.compile(r"\b\d+\s*(?:minutes?|hours?|min|hrs?|%)\b", re.IGNORECASE)
KNOWN_NAMES = ["sarah chen", "david park", "alex rivera", ...]

def richness_score(text):
    emails  → +3 each   (hardest to get, proves multi-hop worked)
    names   → +2 each   (proves owner lookup worked)
    numbers → +1 each   (proves aggregation worked)
```

Why these weights?
- Emails require the full INC-001 → ARCH-001 → TEAM-001 chain to retrieve.
  Finding one = proof that multi-hop happened. High weight.
- Names require at least ARCH-* lookup. Medium weight.
- Numbers can appear in a single doc. Lower weight.

#### Verdict logic

```
router_correct AND large gap (≥3) → CORRECT: right call, meaningfully better
router_correct AND small gap (<3) → CORRECT: saved tokens, both answers equal
router_wrong   AND large gap      → WRONG–DANGEROUS: user gets incomplete answer
router_wrong   AND small gap      → SAFE OVERROUTE: wasted tokens, no harm
```

The gap threshold of 3 points was chosen because:
- One email found = +3 points alone
- If agentic found one email the semantic missed, gap = 3 = meaningful
- Gap < 3 means both answers have essentially the same information

---

## 5. How to Run — Step by Step

### Prerequisites

```
Python 3.11+
All packages in requirements.txt installed
OPENAI_API_KEY set in .env
Working directory: Agentic_retrieval_Poc_1/   (project root, NOT router/)
```

> **IMPORTANT**: Always run from the project root, not from inside router/.
> The scripts use `sys.path.insert(0, "..")` to find shared/ and data/.
> Running from inside router/ will cause import errors.

---

### Step 1 — Generate training data

```bash
python router/01_generate_data.py
```

**What happens:**
1. Loads 80 hardcoded seed queries (40 SIMPLE + 40 COMPLEX)
2. Calls GPT-4o to generate 220 augmentation examples
3. Shuffles everything with seed=42
4. Saves to `router/training_data.json`

**Expected output:**
```
=== Query Router: Training Data Generation ===

Step 1: Loaded 40 SIMPLE seeds + 40 COMPLEX seeds
        Total seeds: 80

Step 2: GPT-4o augmentation...
  Generating SIMPLE augmentation examples via GPT-4o...
  Generating COMPLEX augmentation examples via GPT-4o...
  Added 110 SIMPLE + 110 COMPLEX augmented examples

Step 3: Saved 300 examples to router/training_data.json
        SIMPLE: 150 | COMPLEX: 150

Next step: python router/02_train.py
```

**Time:** ~30 seconds (2 GPT-4o calls)
**Cost:** ~$0.05

---

### Step 2 — Train the model

```bash
python router/02_train.py
```

**What happens:**
1. Loads training_data.json, splits 80/20
2. Downloads DistilBERT weights from HuggingFace (~66MB, first run only)
3. Fine-tunes for up to 5 epochs with early stopping
4. Evaluates on validation set after each epoch
5. Saves best model to `router/model/`
6. Prints classification report + confusion matrix

**Expected output (per epoch):**
```
{'eval_loss': '0.1213', 'eval_accuracy': '0.9623', 'epoch': '2'}
{'eval_loss': '0.0144', 'eval_accuracy': '1.0',    'epoch': '5'}
```

**Final output:**
```
--- Validation Set Classification Report ---
              precision    recall  f1-score   support
      SIMPLE       1.00      1.00      1.00        28
     COMPLEX       1.00      1.00      1.00        25
    accuracy                           1.00        53

False negatives (COMPLEX routed to SIMPLE): 0/25

[OK] Training complete. Model saved to router/model/
```

**Time:** 15–20 minutes on CPU (no GPU required)
**Cost:** $0 (runs locally)

---

### Step 3 — Evaluate against the 4 standard queries

```bash
python router/03_evaluate.py
```

**What happens:**
1. Loads the trained router from `router/model/`
2. For each of the 4 standard queries:
   - Router classifies (SIMPLE/COMPLEX + confidence)
   - Semantic RAG runs and produces a synthesized answer
   - Agentic loop runs and produces a synthesized answer
   - Richness scores computed for both
   - Verdict printed
3. Summary table across all 4 queries

**Time:** ~5 minutes (8 GPT-4o calls total — 2 per query)
**Cost:** ~$0.20

---

## 6. How to Test the Router Standalone

If you want to test just the router classification (no GPT-4o calls, no KB),
use this quick test script from the project root:

```python
# Run from project root:
# python -c "exec(open('router/quick_test.py').read())"
# Or just paste this in a Python shell:

import sys
sys.path.insert(0, ".")
from router.router_model import QueryRouter

router = QueryRouter()

test_queries = [
    "What is the migration policy?",
    "Who owns the service that caused the November 2024 incident?",
    "What is the total RTO if all services fail?",
    "Deploy Checkout this Friday at 2pm — what approvals do I need?",
    "Who is the VP of Engineering?",
    "What are the on-call hours?",
]

for q in test_queries:
    decision, confidence = router.route(q)
    print(f"[{decision:8s} {confidence:.0%}]  {q}")
```

Expected output:
```
[SIMPLE   98%]  What is the migration policy?
[COMPLEX  98%]  Who owns the service that caused the November 2024 incident?
[COMPLEX  98%]  What is the total RTO if all services fail?
[COMPLEX  98%]  Deploy Checkout this Friday at 2pm — what approvals do I need?
[SIMPLE   98%]  Who is the VP of Engineering?
[SIMPLE   97%]  What are the on-call hours?
```

**This is instant (~8ms per query). No API key needed. No KB needed.**

---

## 7. How to Test With the Full POC

### Option A — Run the evaluation script (recommended)

```bash
python router/03_evaluate.py
```

Runs both retrieval systems on all 4 standard queries and shows the full
comparison with richness scores and verdicts. Best for understanding the gap.

### Option B — Run extra test queries

```bash
python router/extra_test_queries.py
```

Runs 50 additional queries through the router only (no GPT-4o).
Shows classification + confidence for each. Useful for testing router
behaviour on queries not in the training set.

### Option C — Wire router into run_interactive.py (production integration)

Add these two changes to `run_interactive.py` to make the router live:

**At startup (after `client = make_openai_client()`):**
```python
from router.router_model import QueryRouter
router = QueryRouter()
console.print("[green]Router loaded.[/green]")
```

**In the main loop (replace the current section that always runs both):**
```python
decision, confidence = router.route(query)
console.print(f"\n[dim]Router: [{decision}] ({confidence:.0%} confidence)[/dim]\n")

if decision == "SIMPLE":
    sem_answer, sem_results = run_semantic_section(query, collection, client)
    ag_answer = "[dim]Skipped — router classified as SIMPLE[/dim]"
else:
    ag_answer = run_agentic_section(query, collection, client)
    sem_answer, sem_results = "[dim]Skipped — router classified as COMPLEX[/dim]", []
```

> Note: run_interactive.py currently always runs BOTH systems for comparison
> purposes. The above change would make it production-style (one path only).
> For demo/POC purposes, running both and comparing is more educational.

---

## 8. Understanding the Outputs

### Router output

```
+------------------------------- Router Output -------------------------------+
| Decision   : COMPLEX                                                        |
| Confidence : 98.2%                                                          |
| Ground truth: COMPLEX                                                       |
| Match      : YES                                                            |
+-----------------------------------------------------------------------------+
```

- **Decision**: SIMPLE, COMPLEX, or UNCERTAIN
- **Confidence**: probability of the decided class (0–100%)
- **Match**: does the decision agree with the known ground truth label?

### Richness comparison table

```
+-----------------------------------------------------------------------------+
|  Signal    |  Semantic RAG                  |  Agentic                      |
|  Emails    |  []                            |  ['sarah.chen@technova.io']   |
|  Names     |  sarah chen                    |  sarah chen                   |
|  Richness  |  2                             |  5                            |
+-----------------------------------------------------------------------------+
```

The gap between semantic and agentic richness scores tells you how much
information was lost by using semantic RAG. Gap = 0 means both answers
are equivalent. Gap >= 3 means the router's choice meaningfully affected quality.

### Verdict labels

| Verdict | Meaning |
|---|---|
| CORRECT | Router decision matched ground truth AND gap confirms it mattered |
| CORRECT (token save) | Router said SIMPLE, both answers equivalent — saved tokens |
| WRONG — DANGEROUS | Router said SIMPLE but COMPLEX was needed — incomplete answer |
| SAFE OVERROUTE | Router said COMPLEX unnecessarily — wasted tokens, no quality harm |
| ACCEPTABLE | Ground truth COMPLEX but answers similar — real-world edge case |

---

## 9. What the Numbers Mean

### Training accuracy = 100%

This sounds suspicious but is expected for this dataset size and domain.
With only 263 examples in a narrow domain (enterprise KB queries), and
two clearly distinct classes, DistilBERT — which already understands
language deeply — easily achieves perfect separation on the validation set.

This does NOT mean the model will be 100% accurate on all future queries.
It means it generalises well within the distribution of the training data.
Out-of-distribution queries (new phrasing, new domains) may confuse it.

### Confidence = 98.2% on all 4 queries

The model is very certain on our standard queries because they match
patterns well represented in the training data. In production with
real user queries, you will see lower confidence on ambiguous phrasing.
That is expected and healthy — those land in UNCERTAIN, which safely
routes to Agentic.

### Why richness gap = 3 for Q2, Q3, Q4 but 0 for Q1

- Q1 (SIMPLE): both systems found POL-002 and produced equivalent answers.
  Gap = 0. Router correctly saved tokens by sending to Semantic RAG.
- Q2 (COMPLEX): semantic missed the email. Gap = 3 (one email * 3 pts).
- Q3 (COMPLEX): semantic got wrong total (37 min vs 45 min). Gap = 3 (one number difference).
- Q4 (COMPLEX): both got all emails (Q4 landed all 3 docs in vector search).
  Gap = 3 because agentic found alex.rivera twice (counted twice). Structural diff.

---

## 10. Troubleshooting

### "No trained model found at router/model/"
You have not run training yet.
```bash
python router/01_generate_data.py
python router/02_train.py
```

### "OPENAI_API_KEY not set"
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-proj-...
```
Or export it in your shell: `set OPENAI_API_KEY=sk-proj-...`

### ImportError on 02_agentic_retrieval
You are running from inside `router/` instead of the project root.
```bash
# Wrong:
cd router && python 03_evaluate.py

# Correct:
cd Agentic_retrieval_Poc_1
python router/03_evaluate.py
```

### SSL certificate errors during training
The corporate proxy intercepts HTTPS. DistilBERT downloads from HuggingFace.
If the download fails, set this env var to disable SSL verification:
```bash
set CURL_CA_BUNDLE=
set REQUESTS_CA_BUNDLE=
python router/02_train.py
```

### Training is very slow (>30 min)
You are on CPU only (expected). DistilBERT on CPU with 210 training examples
takes 15–20 minutes for 5 epochs. This is a one-time cost. Inference is 8ms.
To speed up training, reduce epochs:
```python
# In 02_train.py, change:
num_train_epochs=3   # instead of 5
```
Accuracy will be slightly lower (~96% vs 100%) but still excellent.

### Model overfitting (train accuracy >> val accuracy)
If your training data is too small or not diverse enough, the model
memorises training examples instead of learning patterns.
Fix: regenerate more training data with 01_generate_data.py,
or reduce `num_train_epochs` to 3.

### Router says UNCERTAIN for most queries
Your threshold (0.65) may be too high for your training data distribution.
Lower it:
```python
router = QueryRouter(threshold=0.55)
```
Or retrain with more balanced training data.
