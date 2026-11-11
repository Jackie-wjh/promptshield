# Model checkpoints

This directory contains the selected lightweight checkpoints needed for the Streamlit demo and direct reproduction:

- `e1/word.joblib`, `char.joblib`, `combined.joblib`
- `e2/bilstm_random_best.pt`
- `e5/clean.joblib`, `aug_random_p30_x1.joblib`

The selected E3 DistilBERT and E4 RoBERTa fine-tuned weight files are not committed because they exceed GitHub's regular per-file limit. Their experiment code, frozen configurations, raw-score handoffs, metrics and figures are included in the repository.

Recreate the omitted checkpoints with:

```bash
python -m pip install -r requirements/e3.txt
python experiments/e3/run_e3.py

python -m pip install -r requirements/e4.txt
python -m experiments.e4.train
```

See [`../MODEL_MANIFEST.json`](../MODEL_MANIFEST.json) for hashes and expected sizes.
