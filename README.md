# When the Mathematics Is Right: Floating-Point Softmax, and What a Language Model Computes

CS771 (Autumn 2026), Mini-Project 2. Both problems are about the gap between what an ML model is supposed to compute and what actually happens in practice.

## Contents

| File | What it is |
|---|---|
| `problem1.ipynb` | Floating-point failures in log-sum-exp and the softmax cross-entropy gradient, and a numerically stable gradient. |
| `problem2.ipynb` | A tiny world of Markov-chain mixtures: latent-variable inference, EM, a small transformer, and in-context learning as Bayesian inference. |

Both notebooks are executed top to bottom with fixed seeds, and contain the predictions, derivations, code, plots and written answers.

## Problem 1: Correct mathematics meets floating-point arithmetic

- **Part I.** The naive $\log(e^a+e^b)$ overflows to `inf` in both float32 and float64 even though $F(1000,999)\approx1000.31$. Subtracting the maximum fixes it, with the derivation extended to $n$ terms.
- **Part II.** Compare against `torch.logaddexp` / `torch.logsumexp`, including very negative inputs (underflow).
- **Part III.** The library's cross-entropy gradient $g_y=p_y-1$ collapses to exactly 0 once $p_y$ rounds to 1:
  - float32: $m\approx16.75$
  - float64: $m\approx36.75$

  We explain the exact threshold: the denominator $1+4e^{-m}$ is accumulated term by term, so each term is rounded away once $e^{-m}<\varepsilon/2$, giving $m>\ln(2/\varepsilon)$ (16.64 / 36.74).
- **Part IV.** `stable_grad` avoids the subtraction by writing $g_y=-\sum_{j\neq y}p_j$ and $p_j=\exp(z_j-\mathrm{lse}(\mathbf z))$.
- **Part V.** Three plots (correct-class gradient, gradient-sum residual, relative error), with exact zeros shown on a labelled bottom line, plus tests with a non-zero target index, different $K$ and extreme logit gaps.

| | float32 | float64 |
|---|---|---|
| library $g_y$ first exactly 0 at $m$ | 16.75 | 36.75 |
| all library gradients 0 at $m$ | 104 | 745.25 |
| `stable_grad` max relative error | 7.0e-7 | 3.8e-15 |

`stable_grad` removes the cancellation but not the limits of the number format: beyond the underflow point every method returns 0.

## Problem 2: Is a language model secretly doing latent-variable inference?

Text over 8 tokens is generated from a mixture of 4 Markov chains. We compare the exact Bayes-optimal predictor, EM, and a 2-layer transformer (~105k parameters) trained only on next-token prediction.

- **Part 1.** The posterior over the concept reaches more than 95% on the true concept after about 5 tokens (median).
- **Part 2.** EM from 5 random starts: 4 of 5 reach the same optimum (−86.3088 per sequence), one gets stuck at a local optimum (−94.20). EM's component numbering differs from the true one (label switching).
- **Part 3.** Test cross-entropy (nats) and perplexity:

| Predictor | CE | Perplexity |
|---|---|---|
| Oracle (knows the concept) | 1.347 | 3.85 |
| Bayes (true tables) | 1.369 | 3.93 |
| EM mixture (227 params) | 1.369 | 3.93 |
| Transformer (105k params) | 1.385 | 3.99 |
| Bigram (no latent variable) | 1.818 | 6.16 |

- **Part 4.** The transformer's KL to Bayes is at most 0.044 nats at every position (peak near position 6, where the posterior is still resolving), against 0.33–0.47 for the bigram.
- **Part 5.** A linear probe recovers the posterior from hidden states with held-out $R^2$ = 0.81 (trained), 0.25 (random weights), 0.07 (input embeddings only).
- **Part 6.** On an unseen chain, the transformer trained on 4 concepts behaves like the 4-concept Bayes predictor and does not adapt. A transformer trained on fresh chains behaves like the in-context counting predictor (CE 1.550 vs 1.543), at a cost of about 0.39 nats on the original concepts. Each is approximately Bayes-optimal under its own training prior.

## Running the notebooks

Everything runs on a laptop or Google Colab with no GPU needed (the two transformer training runs take about 2–4 minutes each on CPU).

```bash
pip install torch numpy scipy pandas matplotlib scikit-learn jupyter
jupyter notebook problem1.ipynb   # runs in about a minute
jupyter notebook problem2.ipynb   # runs in about 6-10 minutes on CPU
```

Both notebooks print their environment in the first cells. Our runs used PyTorch on CPU with fixed seeds.

## Group members

- Allala Srisharan Reddy
- Ponish Bhatia
- Rohan Rajeev Bharadwaj
- Saanvi Singh
- Sanath Koundinya

## References

- S. M. Xie, A. Raghunathan, P. Liang, T. Ma. *An Explanation of In-context Learning as Implicit Bayesian Inference.* ICLR 2022.
- A. Raventós, M. Paul, F. Chen, S. Ganguli. *Pretraining Task Diversity and the Emergence of Non-Bayesian In-context Learning for Regression.* NeurIPS 2023.
