# E3 — DistilBERT fine-tuning

Compares learning rates 2e-5 and 5e-5 on D1 Validation. The selected checkpoint is the primary
seed model at `models/e3/lr2e-5/`; the unselected 5e-5 checkpoint is omitted.

```bash
python experiments/e3/run_e3.py
python experiments/e3/analyse_e3.py
```

Final tables, figures and `handoff/distilbert_scores.parquet` are under `results/e3/`.
