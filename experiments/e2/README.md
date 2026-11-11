# E2 — BiLSTM initialization experiment

Compares random and GloVe initialization using frozen D1 training/validation data. The selected
submission checkpoint is `models/e2/bilstm_random_best.pt`; the unselected GloVe checkpoint is
omitted from this cleaned package.

```bash
python experiments/e2/run_e2.py
python experiments/e2/score_d4.py
```

Final tables, figures and the E6 raw-score input are under `results/e2/`.
