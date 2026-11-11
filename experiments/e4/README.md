# E4 — RoBERTa fine-tuning and cost analysis

Compares RoBERTa learning rates and freezes the D1 Validation winner. The selected checkpoint is
`models/e4/lr2e-5/` in standard safetensors format. The exploratory follow-up model is omitted.

```bash
python -m experiments.e4.train train
python -m experiments.e4.evaluate score
python -m experiments.e4.figures
```

Use `requirements/e4.txt` in a separate environment. Final tables, figures and the E6 raw-score
input are under `results/e4/`.
