"""
router/
-------
Standalone query routing package. Zero changes to existing codebase.

Files:
  router_model.py      - QueryRouter class (inference only, loads saved model)
  01_generate_data.py  - generates synthetic training data using GPT-4o
  02_train.py          - fine-tunes DistilBERT, saves model to router/model/
  03_evaluate.py       - classifies our 4 standard queries, runs both retrieval
                         systems, compares answers, shows verdict

Run order:
  python router/01_generate_data.py
  python router/02_train.py
  python router/03_evaluate.py
"""
