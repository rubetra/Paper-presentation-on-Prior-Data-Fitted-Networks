
# Transformers Can Do Bayesian Inference

> Paper presentation · Seminar in Advanced Topics in Machine Learning · ETH Zürich (Fall 2025) · Sophie Haldemann

Presentation of the paper Müller, S., Hollmann, N., Pineda, S., Grabocka, J., & Hutter, F. (2022). *Transformers Can Do Bayesian Inference*. Published at ICLR 2022. University of Freiburg, Charité Berlin, Bosch Center for AI.

<br>

<img width="1413" height="750" alt="image" src="https://github.com/user-attachments/assets/fe18995f-d13e-443b-a08d-186980648042" />

---

## Overview

This presentation covers **Prior-Data Fitted Networks (PFNs)** — a meta-learning approach that trains a transformer on synthetic datasets sampled from a prior, enabling it to perform approximate Bayesian inference in a single forward pass at test time.

<img width="1600" height="896" alt="Screen Recording 2026-04-25 at 11 41 49" src="https://github.com/user-attachments/assets/05d4254e-9a25-451a-b11f-affc12059fc7" />


---

## Motivation

Classical Bayesian inference requires computing the **Posterior Predictive Distribution (PPD)**:

$$p(y^* \mid x^*, D) = \int p(y^* \mid x^*, \theta)\, p(\theta \mid D)\, d\theta$$

This integral is intractable in most real-world settings. Approximation methods such as MCMC and Variational Inference (VI) are typically slow and expensive, which limits their use in low-data regimes where uncertainty estimation matters most.

$$\rightarrow$$ Can a deep learning model be *pretrained* to learn the PPD mapping directly?

---

## Method

### Prior-Data Fitted Networks (PFNs)

A PFN is a transformer $T_\theta$ trained to approximate Bayesian inference via three stages:

**1. Prior sampling**
Generate $K$ synthetic datasets $D^{(1)}, \dots, D^{(K)}$ by sampling from a prior $p(t)$ over supervised learning tasks.

**2. Pretraining**
Optimize $T_\theta$ to minimize the negative log-likelihood of held-out test points across all $K$ datasets:

$$\ell_\theta = -\sum_{i=1}^{K} \log q_\theta\!\left(y^{(i)} \mid x^{(i)}, D^{(i)}\right)$$

This objective is equivalent to minimizing the expected KL-divergence between the true PPD and the model's approximation.

**3. Inference**
The trained $T_{\hat{\theta}}$ performs instant Bayesian inference for any query point $x^*$ given a new dataset in just **1 forward pass**.

### Transformer Adaptations

- **Variable train/test splits** — no fixed context size
- **No positional encodings** — makes the model invariant to input permutations
- **Novel regression head (Riemann Distribution)** — treats regression as classification over a discretized output grid, enabling the transformer to output a full predictive distribution rather than a point estimate

---

## Results

### Toy Example: Gaussian Process Regression

The prior samples $(x, y)$-pairs from a Gaussian Process:
- Sample $x$ from the unit cube
- Compute kernel matrix $K_{ij} = k(x_i, x_j)$
- Sample $y \sim \mathcal{N}(0, K)$

PFNs successfully learn to approximate the GP posterior predictive distribution, with performance improving with more training.

### Binary Classification (BNN Prior)

Evaluated on 21 real-world datasets (20 subsets each), simplified to balanced binary classification with max. 60 features, 30/70 split.

| Metric | PFN-BNN | Log. Reg. | GP | KNN | BNN | Catboost | XGB |
|---|---|---|---|---|---|---|---|
| Mean rank ROC AUC | **2.786 ★★★** | 4.690 | 5.286 | 6.214 | 5.000 | 4.833 | 3.357 |
| Loss/Tie/Win vs PFN-BNN | — | 16/1/4 | 17/0/4 | 17/1/3 | 17/1/3 | 13/1/7 | 12/2/7 |
| Expected Calibration Error | **0.025** | 0.157 | 0.095 | 0.093 | 0.089 | 0.157 | 0.066 |
| Benchmark Time | GPU: **0:00:13** / CPU: 0:04:23 | 0:09:56 | 0:24:30 | 0:00:34 | 12:04:41 | 2:05:20 | 20:59:46 |

PFNs achieve the best mean rank AUROC and the fastest GPU inference time by a large margin.

### Few-Shot Image Classification (Omniglot)

With fine-tuning on 30 alphabets (500,000 synthetic pretraining episodes), PFNs reach performance **on par with state-of-the-art** models on 20 held-out alphabets, demonstrating that the PFN framework extends beyond tabular data.

---

## Discussion

**Strengths**
- High speedup during inference compared to all baselines (1 forward pass)
- Synthetic training data from the prior is cheap and unlimited
- Good calibration (lowest ECE among most baselines)
- Close approximation of the true PPD
- Minimal constraints: any prior one can sample from is sufficient

**Limitations**
- High one-time pretraining cost
- Evaluated on small, simplified datasets — limited real-world scale
- Designing a meaningful prior is non-trivial
- Not interpretable: no access to the latent posterior

---

## Take-Home Messages

- PFNs are transformers that **meta-learn Bayesian inference** by training on synthetic datasets sampled from a prior
- **Fast at inference** (1 forward pass) but require upfront pretraining
- Use a **novel Riemann regression head** that reframes regression as a classification problem
- Excel on **small tabular data** and enable **few-shot learning** with fine-tuning

---

## Further Developments

| Work | Description |
|---|---|
| TabPFNv2.5 (Grinsztajn et al., 2025) | Improved tabular classification using a prior over Structural Causal Models |
| Do-PFN (Robertson et al., 2025) | Causal inference via simulated interventions and conditional interventional distributions |
| Mamba4Cast (Bhethanabhotla et al., 2024) | Zero-shot time series forecasting using Mamba instead of Transformer for faster inference |

---

## Tools

R, Python, PowerPoint. Pretraining animation created with [Claude](https://claude.ai) (Anthropic).

---

## Reference

Müller, S., Hollmann, N., Pineda, S., Grabocka, J., & Hutter, F. (2022). Transformers Can Do Bayesian Inference. *International Conference on Learning Representations (ICLR 2022)*. https://arxiv.org/abs/2112.10510
