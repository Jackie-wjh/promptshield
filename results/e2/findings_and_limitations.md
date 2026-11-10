# E2 BiLSTM: Findings and Limitations

## 1. Experimental Findings
In this experiment, we evaluated bidirectional LSTM (BiLSTM) text classifiers with Masked Mean Pooling and Dropout (0.3) on the primary training set (D1: S-Labs, N=11,046). We strictly compared two controlled embedding initialization strategies: **Random normal initialization** versus **Pretrained GloVe 6B (100d)** vectors (vocabulary size = 9,386, GloVe coverage = 92.94%, OOV rate = 7.06%). Both conditions were trained using Adam (lr=1e-3, batch=64, max_epochs=8, patience=2) with checkpoint selection driven solely by D1 Validation Macro-F1 at threshold 0.5.

Counter-intuitively, the **Random initialization achieved slightly superior validation performance** (Macro-F1: 0.9869, Best Epoch 1) compared to the GloVe initialization (Macro-F1: 0.9848, Best Epoch 3). This occurs because GloVe representations originate from general-domain corpora (Wikipedia/Gigaword), where directive words like "ignore", "override", or "instruction" carry neutral connotations. In contrast, prompt injection attacks rely on domain-specific adversarial patterns. The randomly initialized embedding was capable of directly fitting these distinct adversarial keyword features without overcoming general-domain priors. Furthermore, with over 11,000 training examples, D1 provided sufficient domain supervision, diminishing the cold-start advantages of static pretraining.

## 2. Cross-Source Transfer & Generalization
Evaluating the selected model (`bilstm_random`, SHA-256: `6ec73987d1c64bedbd75cc3dfb5a734751f11900eb5818b6851cdb5e943f9cdd`) on external held-out test splits revealed severe distribution shifts:
- **D1 In-Domain Test (N=2,101)**: Macro-F1 = 0.9619.
- **D2 Out-of-Domain Test (deepset, N=116)**: Macro-F1 drops to 0.8448 (-11.7 percentage points).
- **D3 Agentic Test (N=545)**: Macro-F1 drops drastically to 0.7151 (-24.7 percentage points).

Crucially, on D3 the severe degradation is **not caused by a deficiency in attack recall**; the BiLSTM successfully identifies 96.43% of actual attacks (270/280, Attack Recall = 96.43%). Instead, the bottleneck stems from an **acute false-alarm rate on benign inputs, where more than half of legitimate queries are falsely flagged as attacks (135/265, Benign FPR = 50.94%)**. Because Agentic prompts frequently incorporate formal administrative, authorization, and system-interaction vocabulary, the pooled recurrent representations over-index on these technical keywords, inducing massive false-positive distribution shift.

## 3. Limitations
1. **Context Truncation and Fixed Capacity**: Sequences were capped at 256 tokens. BiLSTM compresses sequence representations into a fixed 256-dimensional vector via masked mean pooling, diluting critical semantic nuance in long-tail complex inputs.
2. **Static Vocabulary Vulnerability**: While inference latency is exceptionally fast (median latency ~0.54 ms on CPU), the fixed vocabulary cannot adapt to obfuscated keywords (e.g., character insertions or cross-lingual terms), causing out-of-vocabulary blind spots under distribution transfer.
