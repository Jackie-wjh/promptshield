# E2 BiLSTM: Findings and Limitations

## 1. Experimental Findings
In this experiment, we evaluated bidirectional LSTM (BiLSTM) text classifiers with Masked Mean Pooling and Dropout (0.3) on the primary training set (D1: S-Labs, N=11,046). We strictly compared two controlled embedding initialization strategies: **Random normal initialization** versus **Pretrained GloVe 6B (100d)** vectors (vocabulary size = 9,386, GloVe coverage = 92.94%, OOV rate = 7.06%). Both conditions were trained using Adam (lr=1e-3, batch=64, max_epochs=8, patience=2) with checkpoint selection driven solely by D1 Validation Macro-F1 at threshold 0.5.

Counter-intuitively, the **Random initialization achieved slightly superior validation performance** (Macro-F1: 0.9869, Best Epoch 1) compared to the GloVe initialization (Macro-F1: 0.9848, Best Epoch 3). This occurs because GloVe representations originate from general-domain corpora (Wikipedia/Gigaword), where directive words like "ignore", "override", or "instruction" carry neutral connotations. In contrast, prompt injection attacks rely on domain-specific adversarial patterns. The randomly initialized embedding was capable of directly fitting these distinct adversarial keyword features without overcoming general-domain priors. Furthermore, with over 11,000 training examples, D1 provided sufficient domain supervision, diminishing the cold-start advantages of static pretraining.

## 2. Cross-Source Transfer & Generalization
Evaluating the selected model (`bilstm_random`) on external held-out test splits revealed severe distribution shifts:
- **D1 In-Domain Test**: Macro-F1 = 0.9619 (high precision and recall).
- **D2 Out-of-Domain Test (deepset)**: Macro-F1 drops to 0.8448 (-11.7 percentage points).
- **D3 Agentic Test (5K complex prompts)**: Macro-F1 drops drastically to 0.7151 (-24.7 percentage points).

The frozen checkpoint was subsequently scored on all 339 benign D4 NotInject prompts using the checkpoint's saved training vocabulary. E6 reports 156 false alarms at threshold 0.5 (FPR 46.0%, 95% CI 40.8%–51.3%), 189 at tF1 (55.8%, 50.4%–60.9%), and 252 at t5 (74.3%, 69.4%–78.7%). The D1-selected recall-oriented threshold therefore transfers poorly to this benign stress set.

This severe degradation validates our hypothesis that recurrent sequential models overfit to surface-level lexicons in the training domain. When confronted with complex, multi-sentence agentic prompts (D3) where attack intents are masked inside legitimate tool descriptions, BiLSTM fails to maintain high recall.

## 3. Limitations
1. **Context Truncation and Fixed Capacity**: Sequences were capped at 256 tokens. BiLSTM compresses sequence representations into a fixed 256-dimensional vector via masked mean pooling, diluting critical attack triggers in long-tail inputs.
2. **Static Vocabulary Vulnerability**: While inference latency is exceptionally fast (median latency ~0.54 ms on CPU), the fixed vocabulary cannot adapt to obfuscated keywords (e.g., character insertions or leetspeak), making it susceptible to evasion without subword tokenization.
