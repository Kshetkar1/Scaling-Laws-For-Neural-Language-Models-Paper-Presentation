# Scaling Laws for Neural Language Models
### Paper Presentation for DS 5690: Generative AI Models in Theory & Practice
**Presenter**: Kanu Shetkar ([@Kshetkar1](https://github.com/Kshetkar1))
**Vanderbilt University | Fall 2025**

---

## 📄 Paper Information

**Title**: Scaling Laws for Neural Language Models
**Authors**: Jared Kaplan*, Sam McCandlish*, Tom Henighan, Tom B. Brown, et al.
**Organization**: OpenAI & Johns Hopkins University
**Published**: January 2020
**Link**: [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

---

## 🎤 10-Minute Presentation

### Part 1: Overview (5 minutes)

**The Problem**

Before January 2020, the AI community was flying blind on scaling. We had intuitions—"bigger is probably better"—but no systematic understanding. Critical questions remained:

- How much better does a model get when you 10x the parameters?
- Should you train a small model longer or a large model briefly?
- How much data do you need for a given model size?

These had **massive practical implications**. OpenAI had to decide: build GPT-3 (175B params) or train GPT-2 (1.5B) longer? The industry was spending **millions** without clear optimization principles. GPT-3 alone cost $4-12M to train.

**The Approach**

OpenAI systematically trained **400+ models** from 768K to 1.5B parameters across 7 orders of magnitude and discovered: **language model performance follows precise mathematical power laws**.

**The Three Scaling Laws**

1. **L(N)** - Loss vs Model Size: `L = (Nc/N)^0.076`
   - 10x parameters → 5% loss reduction (predictably!)

2. **L(D)** - Loss vs Data Size: `L = (Dc/D)^0.095`
   - 10x data → 6% loss reduction

3. **L(C)** - Loss vs Compute: `L = (Cc/C)^0.050`
   - Most important: tells you how to allocate compute between N and D

**The Revolutionary Insight**

For fixed compute C, the optimal allocation is:
- **N (parameters) scales as C^0.73**
- **D (data) scales as C^0.27**

This means: as compute increases, grow the model **much faster** than the dataset!

**The Big Idea**

Instead of training small models to convergence (old way), train **much larger models and stop early** (new way). Same compute budget, better performance. This was completely counterintuitive and directly enabled GPT-3, GPT-4, Gopher, and LLaMA.

---

### Part 2: Question #1 (1-2 minutes)

**❓ You have enough compute to train a 1B parameter model on 100B tokens to convergence. Which gives better performance?**

**A)** Train 1B params on 100B tokens (convergence)
**B)** Train 10B params on 10B tokens (early stop)
**C)** Train 100M params on 1T tokens (overtrain)

<details>
<summary><b>Answer</b></summary>

**B) Train the 10B model on 10B tokens!**

Why? Larger models are more **sample-efficient**—they extract more value from each token. Even seeing less data, they learn more from what they see.

This is exactly what the scaling laws predict: N ~ C^0.73, D ~ C^0.27. For every 10x compute increase, use ~5.4x more parameters but only ~2x more data.

</details>

---

### Part 3: Architecture & Algorithms (2 minutes)

**The Key Algorithm: Compute-Optimal Allocation**

Given compute budget C, find optimal (N, D):

```
FUNCTION ComputeOptimalAllocation(C):
    a ← 0.73  // N scaling exponent
    b ← 0.27  // D scaling exponent

    N_optimal ← 0.3 × C^0.73
    D_optimal ← 3.2 × C^0.27

    RETURN (N_optimal, D_optimal)
END
```

**Example**: For C = 1 PF-day → N = 1.3B params, D = 3.2B tokens

**Architecture Finding**

The paper tested wildly different Transformer architectures (6 layers × 2048 width vs 48 layers × 512 width) at the same parameter count N. Result: **nearly identical performance** (<0.1 nats difference).

**Conclusion**: At fixed N, architecture details matter far less than everyone thought. **Scale dominates everything.**

---

### Part 4: Question #2 (1 minute)

**❓ If architecture barely matters, why do modern AI labs still spend effort on architecture search?**

<details>
<summary><b>Discussion</b></summary>

Good question! Several reasons:

1. **Inference efficiency**: MoE models, sparse attention reduce serving costs
2. **Specialized tasks**: Vision, long-context need different architectures
3. **Scaling limits**: We're hitting data constraints—architecture innovations help
4. **Post-training**: RLHF and fine-tuning have different dynamics

**Bottom line**: For pretraining, scale is king. But for full lifecycle (training + inference + deployment), architecture matters!

</details>

---

### Part 5: Impact & Critical Analysis (1-2 minutes)

**What Changed**

| Before (2018-2019) | After (2020+) |
|-------------------|---------------|
| Train to convergence | Early stopping |
| BERT: 340M, 3.3B tokens | GPT-3: 175B, 300B tokens |
| Trial and error | Predictive formulas |

**The Chinchilla Correction (2022)**

DeepMind found GPT-3 was **undertrained on data**. Chinchilla (70B params, 1.4T tokens) outperformed Gopher (280B params, 300B tokens) with same compute. Refined the laws to N ~ D^0.5.

**Critical Limitations**

1. **Dataset homogeneity**: Only tested on English web text
2. **Architecture specificity**: Only decoder-only Transformers
3. **Ignores RLHF**: Post-training has different dynamics
4. **Emergent abilities**: Power laws predict smooth gains, but some abilities appear suddenly
5. **Inference costs**: Optimizes training compute, not serving costs

**Legacy**: 2,500+ citations. Changed how the entire industry thinks about resource allocation.

---

## 📊 Quick Reference Tables

### Scaling Law Comparison

| Law | Formula | 10x Resource → |
|-----|---------|----------------|
| L(N) | (Nc/N)^0.076 | 5% loss reduction |
| L(D) | (Dc/D)^0.095 | 6% loss reduction |
| L(C) | (Cc/C)^0.050 | 3% loss reduction |

### Historical Models

| Model | Year | N | D | Strategy |
|-------|------|---|---|----------|
| BERT | 2018 | 340M | 3.3B | Convergence |
| GPT-2 | 2019 | 1.5B | 40B | Convergence |
| GPT-3 | 2020 | 175B | 300B | Early stop ✓ |
| Chinchilla | 2022 | 70B | 1.4T | Refined laws ✓ |

---

## 🧑‍💻 Code Demo

```python
from code.scaling_calculator import ScalingCalculator

calc = ScalingCalculator()

# Predict GPT-3 performance
N_gpt3, D_gpt3 = 175e9, 300e9
loss = calc.predict_loss(N=N_gpt3, D=D_gpt3)
print(f"Predicted: {loss:.3f} nats")  # 2.150 (actual ~2.0)

# Compute optimal allocation for 1 PF-day
N_opt, D_opt = calc.compute_optimal_allocation(1.0)
print(f"Optimal: N={N_opt/1e9:.1f}B, D={D_opt/1e9:.1f}B")
```

All code in `/code` directory with full documentation.

---

## 🎯 Key Takeaways

1. **Scale is predictable**: Performance follows power laws—you can forecast before training
2. **Bigger models are sample-efficient**: Train large models briefly > small models to convergence
3. **Architecture matters less**: At fixed N, depth/width have minimal impact (<0.1 nats)

---

## 🔗 Resources

1. [Original Paper (arXiv:2001.08361)](https://arxiv.org/abs/2001.08361)
2. [Chinchilla Paper (arXiv:2203.15556)](https://arxiv.org/abs/2203.15556) - Refined scaling laws
3. [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238) - Included in this repo
4. [Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701) - Theoretical analysis
5. [EleutherAI Blog](https://blog.eleuther.ai/scaling-laws/) - Practitioner guide

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

**Repository**: [github.com/Kshetkar1/Scaling-Laws-For-Neural-Language-Models-Paper-Presentation](https://github.com/Kshetkar1/Scaling-Laws-For-Neural-Language-Models-Paper-Presentation)

---

## 📚 Appendix: Detailed Content

<details>
<summary><b>Expand for detailed mathematical derivations, additional algorithms, and extended analysis</b></summary>

### Extended Algorithm Details

#### Algorithm 1: Predicting Loss from Resources

```
FUNCTION PredictLoss(N, D):
    N_c ← 8.8 × 10^13
    D_c ← 5.4 × 10^13
    α_N ← 0.076
    α_D ← 0.095

    term_N ← (N_c / N)^(α_N / α_D)
    term_D ← D_c / D
    L ← (term_N + term_D)^α_D

    RETURN L
END
```

Before spending $4M on GPT-3, OpenAI predicted performance to within 0.05 nats!

#### Algorithm 2: Early Stopping Criterion

```
FUNCTION ShouldStopTraining(N, D, S, B, C_budget):
    C_used ← 6 × N × B × S / 10^15
    tokens_seen ← B × S

    (N_opt, D_opt) ← ComputeOptimalAllocation(C_budget)

    budget_exhausted ← (C_used ≥ C_budget)
    tokens_sufficient ← (tokens_seen ≥ D_opt)

    RETURN (budget_exhausted OR tokens_sufficient)
END
```

Training to convergence wastes 10-100x compute!

#### Algorithm 3: Critical Batch Size

```
FUNCTION CriticalBatchSize(L):
    B_noise ← 2^21  // 2M tokens
    L_noise ← 1.5
    α_D ← 0.095

    B_crit ← B_noise × (L / L_noise)^(1/α_D)

    RETURN floor(B_crit)
END
```

**Insight**: Start with large batches early (L high), decrease as loss drops. Most use fixed batch sizes—suboptimal!

### Detailed Experimental Findings

#### Finding 1: Overfitting is Universal

Optimal ratio: **N^0.74 ≈ D**

Examples:
- N = 100M → D_optimal ≈ 1B tokens
- N = 1B → D_optimal ≈ 5.5B tokens
- N = 100B → D_optimal ≈ 550B tokens

Implication: GPT-4 (~1.8T params) would need ~10T tokens—more than exists on the internet!

#### Finding 2: Sample Efficiency Grows with Model Size

Fixed dataset (10B tokens):
- N = 100M → L = 3.5 nats
- N = 1B → L = 2.8 nats (20% better with same data!)
- N = 10B → L = 2.3 nats (18% better with same data!)

#### Finding 3: Transfer Learning Follows Power Laws

Downstream performance: `Loss_downstream ∝ Loss_pretraining^β`

LAMBADA example:
- Pretrain loss 3.0 → 45% accuracy
- Pretrain loss 2.5 → 62% accuracy
- Pretrain loss 2.0 → 78% accuracy

Scaling the base model improves everything!

### Transformer Architecture Connection

**Parameter Counting**:

```
FUNCTION CountParameters(L, d_model, V):
    N_embed ← V × d_model
    N_attn ← 4 × d_model^2
    N_ffn ← 8 × d_model^2
    N_layer ← 12 × d_model^2
    N_total ← N_embed + L × N_layer
    RETURN N_total
END
```

**Example (GPT-2 Small)**:
- L = 12, d_model = 768, V = 50,257
- N_embed ≈ 39M, N_layer ≈ 85M
- N_total ≈ 124M (close to 117M!)

### Extended Critical Analysis

**What's Missing:**

1. **Data quality**: Paper treats all tokens equally (Wikipedia = Reddit spam)
2. **Multimodal scaling**: How do laws change with vision, audio?
3. **Why power laws?**: No mechanistic explanation
4. **Inference optimization**: Only considers training compute

**Open Questions:**

- Will scaling laws hold forever? (Probably not—data bottleneck)
- Do they apply to RLHF? (Unclear—active research)
- What about emergent abilities? (Smooth laws vs discontinuous jumps)

### Historical Context

**The Chinchilla Moment (2022)**:

DeepMind re-examined with 10-100x more compute. Found:
- Original: N ~ C^0.73, D ~ C^0.27
- Refined: N ~ C^0.50, D ~ C^0.50 (equal scaling!)

Chinchilla (70B, 1.4T) beat Gopher (280B, 300B). Lesson: Laws are guides, not gospel.

**The LLaMA Revolution (2023)**:

Meta's approach:
- Train smaller models on way more data
- LLaMA-13B (1T tokens) matches GPT-3 (175B, 300B)
- 10x cheaper inference → local deployment
- Spawned ecosystem: Alpaca, Vicuna, Llama-2, Mistral

### Philosophical Impact

**Paradigm shift**:

Old: Deep learning is alchemy → try everything
New: Deep learning has predictable properties → measure, extrapolate, optimize

Like Kepler's laws (empirical patterns) before Newton's theory (mechanistic understanding).

**The Bitter Lesson (Rich Sutton)**:

"General methods that leverage computation are ultimately most effective."

Scaling laws are strong evidence. But recent work shows data quality and fine-tuning matter too—so it's not *only* about scale.

</details>

---

**Questions?** [@Kshetkar1](https://github.com/Kshetkar1)

*Presented October 2025*
