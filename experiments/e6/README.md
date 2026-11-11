# E6 — Unified threshold and false-alarm analysis

Reads the retained `results/e*/handoff/*_scores.parquet` files, selects `tF1` and `t5` using only
D1 Validation, then evaluates the frozen policies on D1–D4.

```bash
python experiments/e6/run_e6.py
```

Authoritative outputs are `thresholds.csv`, `metrics.csv`, `notinject_false_alarms.csv`,
`predictions.parquet`, `summary.md` and the three final figures under `results/e6/`.
