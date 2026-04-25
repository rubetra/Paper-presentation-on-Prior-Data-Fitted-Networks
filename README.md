# Transformers Can Do Bayesian Inference

> Paper presentation · Seminar in Advanced Topics in Machine Learning · ETH Zürich (Fall 2025) · Sophie Haldemann

<br>

This repository contains my seminar presentation of:
Müller, S., Hollmann, N., Pineda, S., Grabocka, J., & Hutter, F. (2022). *Transformers Can Do Bayesian Inference*. ICLR 2022.


## Overview

This presentation covers **Prior-Data Fitted Networks (PFNs)**, a meta-learning approach that trains a transformer on synthetic datasets sampled from a prior, enabling it to perform approximate Bayesian inference in a single forward pass at test time.


<img width="1600" height="896" alt="Untitled (1)" src="https://github.com/user-attachments/assets/30397db1-6635-4e6d-8d51-6e5f13b7df59" />

<br>



## Motivation

- Classical Bayesian inference requires computing the **Posterior Predictive Distribution (PPD)**:

  $$p(y^{\ast} \mid x^{\ast}, D) = \int p(y^{\ast} \mid x^{\ast}, \theta) \, p(\theta \mid D) \, d\theta$$

- The PPD is intractable in most real-world settings
- Approximation methods such as MCMC and Variational Inference are typically slow and expensive

→ Can a deep learning model be *pretrained* to learn the PPD mapping directly?

<br>



## Method

### Prior-Data Fitted Networks (PFNs)

A PFN is a transformer $T_\theta$ trained to approximate Bayesian inference:

**1. Prior sampling:** Generate $K$ synthetic datasets $D^{(1)}, \dots, D^{(K)}$ by sampling from a prior $p(t)$ over supervised learning tasks.

**2. Pretraining:** Optimize $T_\theta$ to minimize the negative log-likelihood of held-out test points across all $K$ datasets:

$$\ell_\theta = -\sum_{i=1}^{K} \log\ q_\theta \left(y^{(i)} \mid x^{(i)}, D^{(i)}\right)$$

 This objective is equivalent to minimizing the expected KL-divergence between the true PPD and the model's approximation.

**3. Inference:** The trained $T_{\hat{\theta}}$ performs instant Bayesian inference for a query point $x^*$ given a new dataset in just **1 forward pass**.

### Transformer Adaptations

- **Variable train/test splits**: makes the model flexible in terms of context size
- **No positional encodings**: makes the model invariant to input permutations
- **Novel regression head (Riemann Distribution)**: treats regression as classification over a discretized output grid

<br>


## Results

### Toy Example: Gaussian Process Regression

The prior samples $(x, y)$-pairs from a Gaussian Process:
- Sample $x$ from the unit cube
- Compute kernel matrix $K_{ij} = k(x_i, x_j)$
- Sample $y \sim \mathcal{N}(0, K)$


<img width="899" height="370" alt="image" src="https://github.com/user-attachments/assets/29d3fea5-19f5-4ffb-8f5c-b5148bb22d47" />

<img width="877" height="362" alt="image" src="https://github.com/user-attachments/assets/11af4a68-b02c-49a3-b998-38a41a9fcab6" />

<br>

### Binary Classification (Bayesian Neural Network Prior)

<img width="923" height="247" alt="image" src="https://github.com/user-attachments/assets/4542c0b3-fd18-4ff9-8101-59069a4a7cb0" />

Evaluated on 21 real-world datasets (20 subsets each), simplified to balanced binary classification with max. 60 features, 30/70 split.

| Metric | PFN-BNN | Log. Reg. | GP | KNN | BNN | Catboost | XGB |
|---|---|---|---|---|---|---|---|
| Mean rank ROC AUC | **2.786 ★★★** | 4.690 | 5.286 | 6.214 | 5.000 | 4.833 | 3.357 |
| Loss/Tie/Win vs PFN-BNN | — | 16/1/4 | 17/0/4 | 17/1/3 | 17/1/3 | 13/1/7 | 12/2/7 |
| Expected Calibration Error | **0.025** | 0.157 | 0.095 | 0.093 | 0.089 | 0.157 | 0.066 |
| Benchmark Time | GPU: **0:00:13** / CPU: 0:04:23 | 0:09:56 | 0:24:30 | 0:00:34 | 12:04:41 | 2:05:20 | 20:59:46 |

PFNs achieve the best mean rank AUROC and the fastest GPU inference time.

<br>

### Few-Shot Image Classification (Omniglot)

PFNs are pretrained on 500'000 synthetic datasets of "made-up" characters sampled from a prior
(jittery stroke combinations), then fine-tuned on 30 real alphabets, and evaluated on 20 held-out
alphabets. Performance is **on par with state-of-the-art** models, demonstrating that PFN generalises beyond tabular data.

<img width="1429" height="502" alt="image" src="https://github.com/user-attachments/assets/bbf58c0e-c41c-481a-b359-1cfd7a8e5bea" />

| Method | Vanilla BNN<br>(Liu & Wang, 2016) | MAML<br>(Finn et al., 2017) | MLAP<br>(Amit & Meir, 2018) | BMAML<br>(Yoon et al., 2018) | PACOH-NN<br>(Rothfuss et al., 2021) | **PFN**<br>(this work) |
|---|---|---|---|---|---|---|
| Accuracy | 0.795 ± 0.006 | 0.693 ± 0.013 | 0.700 ± 0.014 | 0.764 ± 0.025 | **0.885 ± 0.090** | **0.865 ± 0.019** |

<br>

## Discussion

**Strengths**
- High speedup during inference compared to all baselines (1 forward pass)
- Synthetic training data from the prior is cheap and unlimited
- Good calibration (lowest ECE among most baselines)
- Close approximation of the true PPD
- Minimal constraints (any prior one can sample from is sufficient)

**Limitations**
- High one-time pretraining cost
- Evaluated on small, simplified datasets (limited real-world scale)
- Designing a meaningful prior is non-trivial
- Not very interpretable (no access to the latent posterior)

<br>

## Take-Home Messages

- PFNs are transformers that **meta-learn Bayesian inference** by training on synthetic datasets sampled from a prior
- **Fast at inference** (1 forward pass) but require upfront pretraining
- Use a **novel Riemann regression head** that reframes regression as a classification problem
- Excel on **small tabular data** and enable **few-shot learning** with fine-tuning

<br>

## Further Developments

- **TabPFNv2.5 (Grinsztajn et al., 2025)**: Improved tabular classification using a prior over Structural Causal Models
- **Do-PFN (Robertson et al., 2025)**: Causal inference via simulated interventions and conditional interventional distributions
- **Mamba4Cast (Bhethanabhotla et al., 2024)**: Zero-shot time series forecasting using Mamba instead of Transformer for faster inference


<br>

## Reference

Müller, S., Hollmann, N., Pineda, S., Grabocka, J., & Hutter, F. (2022). Transformers Can Do Bayesian Inference. *International Conference on Learning Representations (ICLR 2022)*. https://arxiv.org/abs/2112.10510
