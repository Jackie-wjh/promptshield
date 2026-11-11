# E5 — Typo-augmentation robustness

Keeps the E1 Combined feature space fixed and compares controlled typo-augmentation policies.
The cleaned package keeps the clean reference and the selected `aug_random_p30_x1` checkpoint.

```bash
python experiments/e5/run_e5.py
python experiments/e5/build_main_table.py
```

Final paired metrics, robustness figures and the two E6 raw-score files are under `results/e5/`.
