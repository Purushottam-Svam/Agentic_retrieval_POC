"""
02_train.py
-----------
Fine-tunes DistilBERT on the generated training data.

Why DistilBERT over BERT-base?
  - 66MB vs 440MB model size
  - 2x faster inference on CPU
  - 97% of BERT accuracy for classification tasks
  - Perfect for binary classification on short queries

What this script does:
  1. Loads router/training_data.json
  2. Splits 80/20 into train/validation
  3. Fine-tunes DistilBERT (adds a 2-class head on top)
  4. Evaluates accuracy + per-class metrics on validation set
  5. Saves best model to router/model/

Run:
    python router/02_train.py

Output:
    router/model/    -- saved model + tokenizer (load with QueryRouter)
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH  = os.path.join(os.path.dirname(__file__), "training_data.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")
BASE_MODEL = "distilbert-base-uncased"

LABEL2ID = {"SIMPLE": 0, "COMPLEX": 1}
ID2LABEL = {0: "SIMPLE", 1: "COMPLEX"}


def load_data() -> tuple[list[str], list[int]]:
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        print("Run  python router/01_generate_data.py  first.")
        sys.exit(1)

    with open(DATA_PATH) as f:
        raw = json.load(f)

    texts  = [d["query"] for d in raw]
    labels = [LABEL2ID[d["label"]] for d in raw]

    n_simple  = labels.count(0)
    n_complex = labels.count(1)
    print(f"Loaded {len(texts)} examples  |  SIMPLE: {n_simple}  |  COMPLEX: {n_complex}")
    return texts, labels


def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,      # more than enough for enterprise queries (~30 tokens avg)
    )


def compute_metrics(eval_pred) -> dict:
    """Called by Trainer after each eval epoch. Returns accuracy."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": float(accuracy)}


def print_classification_report(model, tokenizer, val_dataset):
    """Detailed per-class metrics after training completes."""
    print("\n--- Validation Set Classification Report ---")
    texts  = list(val_dataset["text"])
    labels = list(val_dataset["label"])

    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = logits.argmax(dim=1).numpy()

    print(classification_report(labels, preds, target_names=["SIMPLE", "COMPLEX"]))

    cm = confusion_matrix(labels, preds)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"               Pred SIMPLE  Pred COMPLEX")
    print(f"  Actual SIMPLE     {cm[0][0]:>5}         {cm[0][1]:>5}")
    print(f"  Actual COMPLEX    {cm[1][0]:>5}         {cm[1][1]:>5}")

    # False negatives (COMPLEX classified as SIMPLE) are the dangerous ones
    false_negatives = cm[1][0]
    total_complex   = cm[1][0] + cm[1][1]
    print(f"\nFalse negatives (COMPLEX routed to SIMPLE): {false_negatives}/{total_complex}")
    print("These are the dangerous misroutes -- user would get incomplete answer.")


def main():
    print("\n=== Query Router: DistilBERT Fine-tuning ===\n")

    texts, labels = load_data()

    # 80/20 split
    split_idx = int(len(texts) * 0.8)
    train_texts, val_texts   = texts[:split_idx],  texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    print(f"Train: {len(train_texts)} | Validation: {len(val_texts)}")

    # Tokenizer
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL)

    # Build HuggingFace datasets
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds   = Dataset.from_dict({"text": val_texts,   "label": val_labels})

    train_ds = train_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    val_ds   = val_ds.map(lambda b: tokenize_batch(b, tokenizer),   batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(  type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Model
    print(f"Loading model: {BASE_MODEL}")
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Training config
    # Small dataset -> few epochs, small LR to avoid overfitting
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,            # fine-tuning LR (not training from scratch)
        warmup_steps=20,               # gradual LR warmup prevents early instability
        weight_decay=0.01,             # mild regularization
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,   # restore best checkpoint after all epochs
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",              # no wandb/tensorboard
        fp16=False,                    # CPU safe
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        # stops if val accuracy doesn't improve for 2 consecutive epochs
    )

    print("\nStarting training...\n")
    trainer.train()

    # Save best model + tokenizer
    print(f"\nSaving best model to: {MODEL_PATH}")
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    # Detailed evaluation
    # Re-create val dataset without tensor format for classification_report
    val_ds_plain = Dataset.from_dict({"text": val_texts, "label": val_labels})
    print_classification_report(model, tokenizer, val_ds_plain)

    print(f"\n[OK] Training complete. Model saved to router/model/")
    print("Next step: python router/03_evaluate.py\n")


if __name__ == "__main__":
    main()
