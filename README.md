# Scaling Laws for Neural Language Models
### Paper Presentation for DS 5690: Generative AI Models in Theory & Practice
**Presenter**: Kanu Shetkar ([@Kshetkar1](https://github.com/Kshetkar1))
**Vanderbilt University | Fall 2025**

---

## 📄 Paper Information

**Authors**: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, et al.
**Organization**: OpenAI & Johns Hopkins University
**Published**: January 2020
**Link**: [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

---

## 🎯 The Problem

Before 2020, AI companies had **no systematic way** to decide:
- Should we build bigger models?
- Or train smaller models on more data?
- How much compute do we need?

**Stakes**: GPT-3 cost **$4-12 million** to train. Get it wrong = millions wasted.

---

## 🔬 What This Paper Did

OpenAI trained **400+ models** across **7 orders of magnitude** of scale.

**Discovery**: Language model performance follows **precise mathematical power laws**.

---

## 📊 The Three Scaling Laws
<img width="416" height="212" alt="image" src="https://github.com/user-attachments/assets/c14ef1fe-6ad0-44c5-8ffb-08be400db7b1" />


### Law 1: Performance vs Model Size
```
L(N) = (Nc / N)^0.076
```
**10x parameters** → **5% loss reduction**

---

### Law 2: Performance vs Data Size
```
L(D) = (Dc / D)^0.095
```
**10x data** → **6% loss reduction**

---

### Law 3: Performance vs Compute Budget
```
L(C) = (Cc / C)^0.050
```

**Optimal allocation**:
- Parameters scale as **C^0.73**
- Data scales as **C^0.27**

**Translation**: For every 10x compute, use **5.4x more parameters** but only **2x more data**.

---

## 💡 Key Insight

**Old way**: Train small models to convergence
**New way**: Train huge models, stop early

**Example**:
- **Before**: 340M params, 3.3B tokens → Loss 3.2 nats ($7K)
- **After**: 1B params, 1B tokens → Loss 2.7 nats ($7K) = **16% better!**

---

## 🤖 Formal Algorithms

### Algorithm 1: Compute-Optimal Resource Allocation

**Input:** $C \in \mathbb{R}^+$ (compute budget in PetaFLOP-days)

**Parameters:**
- $\alpha_N = 0.73$, $\alpha_D = 0.27$ (scaling exponents)
- $k_N = 0.3$, $k_D = 3.2$ (hardware coefficients)

**Algorithm:**

1. $N_{\text{optimal}} \leftarrow k_N \times C^{\alpha_N}$ $\triangleright$ Optimal model size
2. $D_{\text{optimal}} \leftarrow k_D \times C^{\alpha_D}$ $\triangleright$ Optimal dataset size
3. $C_{\text{actual}} \leftarrow \frac{6 \times N_{\text{optimal}} \times D_{\text{optimal}}}{10^{15}}$ $\triangleright$ Verify constraint
4. **if** $\frac{|C_{\text{actual}} - C|}{C} > 0.1$ **then**
5. &nbsp;&nbsp;&nbsp;&nbsp;Print "Warning"
6. **return** $(N_{\text{optimal}}, D_{\text{optimal}})$

**Example**: C = 1.0 PetaFLOP-day → N = 1.3B params, D = 3.2B tokens

**Key Property**: Predicted GPT-3 performance to within 0.05 nats before spending $4M!

---

### Algorithm 2: Predict Loss from Resources

**Input:** $N$ (parameters), $D$ (tokens)

**Parameters:** $N_c = 8.8 \times 10^{13}$, $D_c = 5.4 \times 10^{13}$, $\alpha_N = 0.076$, $\alpha_D = 0.095$

**Algorithm:**

1. $\text{term}_N \leftarrow \left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D}$ $\triangleright$ Parameter constraint
2. $\text{term}_D \leftarrow \frac{D_c}{D}$ $\triangleright$ Data constraint
3. $L \leftarrow (\text{term}_N + \text{term}_D)^{\alpha_D}$ $\triangleright$ Bottleneck formula
4. **return** $L$

**Complexity:** $O(1)$

---

### Algorithm 3: Early Stopping Criterion

**Input:** $N$, $S$ (training step), $B$ (batch size), $C_{\text{budget}}$

**Algorithm:**

1. $C_{\text{used}} \leftarrow \frac{6 \times N \times B \times S}{10^{15}}$ $\triangleright$ Compute used
2. $\text{tokens}_{\text{seen}} \leftarrow B \times S$ $\triangleright$ Tokens processed
3. $(N_{\text{opt}}, D_{\text{opt}}) \leftarrow$ ComputeOptimalAllocation$(C_{\text{budget}})$
4. **if** $C_{\text{used}} \geq C_{\text{budget}}$ **then return** True
5. **if** $N \approx N_{\text{opt}}$ **and** $\text{tokens}_{\text{seen}} \geq D_{\text{opt}}$ **then return** True
6. **if** $N > 2 \times N_{\text{opt}}$ **then return** True
7. **return** False

**Key Property**: Training to convergence wastes 100x compute!

**Complexity:** $O(1)$

---

## 🤔 Question 1: Wide vs Narrow Architecture

**You have a budget of 100 million parameters. Which architecture performs better?**

**A) Wide & Shallow**: 6 layers, 2048 dimensions (~100M params)
**B) Narrow & Deep**: 48 layers, 512 dimensions (~100M params)

<details>
<summary><b>Click for answer</b></summary>

**Answer: They perform almost identically!**

The paper found < 0.1 nats difference in loss at the same parameter count $N$.

**Key Finding**: Architecture shape (wide vs narrow, shallow vs deep) matters **far less** than total parameter count.

This held from 768K to 1.5B parameters. At 100B+ scale, architecture innovations (sparse attention, MoE) become important for efficiency/context.

</details>

---

## 🏗️ The Architecture Finding

**Test**: Two models with same $N$:
- **Wide & shallow**: 6 layers, 2048 dimensions
- **Narrow & deep**: 48 layers, 512 dimensions

**Result**: Nearly identical performance (< 0.1 nats difference)

**Conclusion**: At fixed $N$, architecture size (depth vs width) barely matters. **Scale dominates.**

---

## 🤔 Question 2: Why Architecture Still Matters?

If architecture size barely matters, why do labs spend time on architecture search?

<details>
<summary><b>Click to discuss</b></summary>

**1. Inference Efficiency**
Training depends on $N$, but **inference cost** depends on architecture. Sparse models are cheaper to serve.

**2. Specialized Tasks**
Vision, long-context, code models need different architectures.

**3. Hardware Constraints**
At trillion-parameter scale, need model parallelism, memory-efficient designs.

**4. Post-Training**
RLHF and fine-tuning have different dynamics than pretraining.

**Bottom line**: Architecture **size** matters less. Architecture **design** matters a lot!

</details>

---

## 🌟 Impact

### Before vs After

| Before (2018-2019) | After (2020+) |
|-------------------|---------------|
| Small models to convergence | Large models, stop early |
| BERT: 340M, 3.3B tokens | GPT-3: 175B, 300B tokens |
| Trial & error | Predictive formulas |

---

### The Chinchilla Correction (2022)

DeepMind tested at 10-100x more compute than original paper.

**Finding**: GPT-3 and Gopher were **undertrained on data**.

**Chinchilla** (70B params, 1.4T tokens) beat **Gopher** (280B params, 300B tokens) with same compute.

**Revised laws**: N and D should scale more equally (N ~ C^0.50, D ~ C^0.50) not the extreme ratio from original paper.

---

### The LLaMA Revolution (2023)

Meta trained smaller models on way more data:
- LLaMA-13B (1T tokens) matches GPT-3-175B (300B tokens)
- **10x cheaper inference**
- Enabled **open-source LLM ecosystem**

---

## 🔬 Critical Analysis

### Key Limitations

1. **Dataset Homogeneity**: Only English web text (WebText2)
2. **Architecture**: Only decoder-only Transformers (GPT-2 style)
3. **Post-Training Ignored**: No RLHF or instruction tuning
4. **Emergent Abilities**: Power laws predict smooth improvements, not sudden jumps
5. **Inference Costs**: Optimizes training, ignores serving costs

---

### What the Authors Overlooked

1. **Data Quality**: Treating all tokens equally is wrong. High-quality data gives 5-10x better efficiency.
2. **Data Wall**: Internet has ~5T tokens. GPT-4 scale needs 10T+. We're running out.
3. **Fine-Tuning**: Smaller models with RLHF can match larger base models.
4. **Architecture at Scale**: "Architecture doesn't matter" breaks down beyond 1.5B params.

---

### Have Others Disputed This?

**Yes!** The Chinchilla revision (2022) directly challenged the original allocation:
- **Original**: N ~ C^0.73, D ~ C^0.27
- **Chinchilla**: N ~ C^0.50, D ~ C^0.50

**Why?** Scaling exponents themselves change with scale!

**Other disputes**:
- Emergent abilities debate (Schaeffer et al. 2023)
- LLaMA "overtraining" (1T tokens for 7B params violates laws but works better)
- Compute estimation (6ND may be off by 2x)

---

## 📊 Quick Reference

| Law | Formula | 10x Resource → |
|-----|---------|----------------|
| **L(N)** | $(N_c/N)^{0.076}$ | 5% ↓ loss |
| **L(D)** | $(D_c/D)^{0.095}$ | 6% ↓ loss |
| **L(C)** | $(C_c/C)^{0.050}$ | 3% ↓ loss |

---

| Model | Year | Params | Data | Strategy |
|-------|------|--------|------|----------|
| BERT | 2018 | 340M | 3.3B | Convergence |
| GPT-2 | 2019 | 1.5B | 40B | Convergence |
| GPT-3 | 2020 | 175B | 300B | Early stop ✓ |
| Chinchilla | 2022 | 70B | 1.4T | Refined laws ✓ |

---

## 🎯 Key Takeaways

**1. Scale is Predictable**
Performance follows power laws. Forecast before spending millions.

**2. Bigger Models Are Sample-Efficient**
Train large models briefly > train small models to convergence.

**3. Architecture Size Matters Less Than You Think**
At fixed params, depth vs width has minimal impact. Focus on scale.

---

## 🔗 Resources

1. [Original Paper](https://arxiv.org/abs/2001.08361) - Kaplan et al., 2020
2. [Chinchilla Paper](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022
3. [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)
4. [Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701)
5. [EleutherAI Blog](https://blog.eleuther.ai/scaling-laws/)

---

## 📖 Citation

```bibtex
@article{kaplan2020scaling,
  title={Scaling Laws for Neural Language Models},
  author={Kaplan, Jared and McCandlish, Sam and Henighan, Tom and Brown, Tom B and others},
  journal={arXiv preprint arXiv:2001.08361},
  year={2020}
}
```

---

**Questions?** [@Kshetkar1](https://github.com/Kshetkar1)
**Repository**: [Scaling-Laws-Paper-Presentation](https://github.com/Kshetkar1/Scaling-Laws-For-Neural-Language-Models-Paper-Presentation)

*Presented October 2025*
