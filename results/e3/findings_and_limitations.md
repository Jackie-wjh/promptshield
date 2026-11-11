# E3 Findings and Limitations

E3 held the tokenizer, training data, batch size, schedule, epoch budget, seed and evaluation code
fixed and changed only the DistilBERT-base-uncased fine-tuning learning rate (2e-5 versus 5e-5).
Both rates were trained for three epochs and every candidate was compared on D1 validation only.
Validation Macro F1 at threshold 0.5 peaked at the first epoch for both rates — 0.9904 at 2e-5 and
0.9854 at 5e-5 — and then declined monotonically (2e-5: 0.9828, 0.9772; 5e-5: 0.9838, 0.9813). The
frozen checkpoint is 2e-5 at epoch 1. That 0.50-point margin is smaller than the spread produced by
two extra seeds of the frozen configuration (0.76 points on validation, 1.43 points on D1 test), so
these data do not establish that 2e-5 is the better rate. What they do show is that three epochs
overfit 11,046 training rows, and that 5e-5 degrades more slowly than 2e-5 as training continues.

In domain, DistilBERT is clearly stronger than the E1 Combined TF-IDF baseline at the shared 0.5
threshold: D1 test Macro F1 0.9729 against 0.9451, a paired class-stratified bootstrap difference of
+2.77 points (95% CI 1.86 to 3.68, 1,000 resamples). The lead widens on D2, where Macro F1 reaches
0.8441 against 0.6782 (+16.59 points, CI 9.65 to 25.93) and attack recall rises from 43.3% to 75.0%.

That advantage does not transfer. On the template-generated D3 agentic source the model flags 256 of
265 benign inputs (FPR 96.6%) while recalling 100% of attacks, so Macro F1 falls to 0.3760 against
0.6320 for E1; the paired FPR difference is +52.83 points (CI 46.79 to 58.49). The same over-defence
appears on benign-only NotInject: 227 of 339 false alarms (FPR 0.670, CI 0.618 to 0.718) against 78
of 339 for E1. A model that has learned imperative-instruction surface cues from one annotation
policy appears to treat instruction-like benign text as hostile.

Limitations: one training source and one annotation policy; scores are uncalibrated softmax outputs,
not probabilities of maliciousness; only the selected configuration was re-run with extra seeds;
D3 is template-generated, so it stresses template behaviour rather than production traffic; latency
comes from a single laptop GPU and is not comparable to server hardware. Truncation was measured at
0 of 11,046 training and 0 of 5,086 evaluation rows (longest input 211 tokens), so the 256-token
budget is not a confound here.
