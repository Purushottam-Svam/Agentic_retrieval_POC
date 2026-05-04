"""
router_model.py
---------------
QueryRouter: loads the saved DistilBERT model and classifies a query as
SIMPLE or COMPLEX in ~8ms on CPU.

Usage:
    from router.router_model import QueryRouter
    router = QueryRouter()           # loads model once
    decision, confidence = router.route("What is the migration policy?")
    # -> ("SIMPLE", 0.94)
"""

import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Default path: router/model/ relative to this file
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

LABEL_MAP = {0: "SIMPLE", 1: "COMPLEX"}


class QueryRouter:
    """
    Binary text classifier: SIMPLE vs COMPLEX query.

    SIMPLE  -> one vector search is enough, use Semantic RAG
    COMPLEX -> needs multi-hop / aggregation / date logic, use Agentic loop

    Threshold logic:
      p_complex >= threshold          -> COMPLEX
      p_complex <= (1 - threshold)    -> SIMPLE
      in between                      -> UNCERTAIN (routes to COMPLEX, safer)
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, threshold: float = 0.65):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"No trained model found at '{model_path}'.\n"
                f"Run  python router/02_train.py  first."
            )
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.threshold = threshold

    def route(self, query: str) -> tuple[str, float]:
        """
        Classify a query.

        Returns:
            (decision, confidence)
            decision   : "SIMPLE" | "COMPLEX" | "UNCERTAIN"
            confidence : probability of the returned class (0.0 - 1.0)
        """
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]

        p_simple  = probs[0].item()
        p_complex = probs[1].item()

        if p_complex >= self.threshold:
            return "COMPLEX", round(p_complex, 4)
        elif p_simple >= self.threshold:
            return "SIMPLE", round(p_simple, 4)
        else:
            # uncertain — bias toward COMPLEX (safer: overspend tokens vs give wrong answer)
            return "UNCERTAIN", round(max(p_simple, p_complex), 4)
