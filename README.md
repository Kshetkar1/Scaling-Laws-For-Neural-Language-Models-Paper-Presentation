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

## 🎤 The Presentation

Have you ever wondered how companies like OpenAI decide whether to build something like GPT-3 or just train GPT-2 for longer? I mean, we're talking about decisions that cost **millions of dollars**. Before this paper came out in January 2020, the AI community was basically flying blind when it came to these choices.

Let me paint you a picture of what it was like back then.

### The Problem

So imagine you're at OpenAI in late 2019. You've got a fixed compute budget—let's say enough to run powerful GPUs for a few months. And you're sitting in a meeting room trying to decide: should we build a model with 175 billion parameters, or should we take our existing 1.5 billion parameter model and just train it on way more data?

Everyone has opinions, right? Some people are saying "bigger is better!" Others are saying "no, we need more data!" And honestly, **nobody really knew**. We had intuitions—"bigger is probably better"—but that was it. Just intuitions.

These weren't just academic debates. These decisions had **massive practical implications**:
- Training GPT-3 ended up costing somewhere between $4 and $12 million
- Google DeepMind was making similar decisions with their compute budgets
- The entire industry was spending millions of dollars on training runs without any clear optimization principles

So the stakes were incredibly high. Get it wrong, and you waste millions of dollars or miss out on critical performance gains.

### What This Paper Did

This is where this paper comes in, and I have to say, what the authors at OpenAI did was pretty revolutionary.

They systematically trained **over 400 models**—ranging from tiny 768 thousand parameter models all the way up to 1.5 billion parameters. They varied everything you can think of: depth, width, attention heads, you name it. And they did this across **seven orders of magnitude** of scale.

And then they discovered something remarkable: **language model performance follows precise mathematical power laws**.

Now, I know "power laws" might sound abstract, so let me break this down.

### The Three Scaling Laws

Think of power laws like compound interest. You know how when you put money in a savings account, each additional dollar doesn't give you the same absolute return, but the pattern is predictable? Power laws work the same way.

The paper found three fundamental laws:

**Law 1: Performance vs Model Size**
```
L(N) = (Nc / N)^0.076
```

What does this mean in plain English? If you **10x the number of parameters**, you get about a **5% reduction in loss**. And this holds consistently from 1 million parameters all the way to 1 billion parameters!

**Law 2: Performance vs Data Size**
```
L(D) = (Dc / D)^0.095
```

If you **10x your training data**, you get about a **6% reduction in loss**. Notice that data is slightly more efficient than parameters.

**Law 3: Performance vs Compute Budget**
```
L(C) = (Cc / C)^0.050
```

And this is the most important one. This law tells you: given a fixed compute budget C, how should you allocate it between model size and data size?

### The Game-Changing Insight

Here's where it gets really interesting. The optimal allocation, according to these laws, is:
- **Parameters should scale as C^0.73**
- **Data should scale as C^0.27**

Let me repeat that because it's counterintuitive: as you increase compute, you should grow your model **much faster** than your dataset!

For every 10x increase in compute, you should use about **5.4x more parameters** but only **2x more data**.

This was completely backwards from what everyone thought. The conventional wisdom was: train small models on tons of data until they converge. But the scaling laws said: no, train **huge models** on relatively modest data and **stop early**.

### Why This Changed Everything

Let me show you what this meant in practice. Instead of training small models to convergence—the old way—you should train much larger models and stop well before they've "seen enough" data.

Here's a concrete example. Before scaling laws:
- BERT-Large: 340M parameters, 3.3B tokens, trained to convergence
- Cost: about $7,000 in GPU time
- Result: Loss around 3.2 nats

After scaling laws, with the **same budget**:
- Optimal: 1B parameters, 1B tokens, stop at ~10% convergence
- Cost: Same $7,000
- Result: Loss around 2.7 nats—that's **16% better**!

This directly enabled GPT-3, GPT-4, Gopher, and all the modern LLMs we use today.

---

## 🤔 Question for the Class

Alright, before we go deeper, let me ask you all something. Think about this for a second.

**You have enough compute to train a 1 billion parameter model on 100 billion tokens until it fully converges. Which of these strategies do you think would give you the best final performance?**

**A)** Train that 1B parameter model on all 100B tokens until loss stops improving
**B)** Train a 10B parameter model (10x bigger!) on just 10B tokens and stop early
**C)** Train a tiny 100M parameter model on 1 trillion tokens (10x more data!)

Take a moment. What's your gut instinct?

<details>
<summary><b>Click here for the answer</b></summary>

The answer is **B**—train the 10 billion parameter model on 10 billion tokens!

I know, I know. It feels wrong, right? You're thinking: "Wait, that model is seeing way less data. How can it be better?"

Here's why it works. Larger models are what we call **sample-efficient**. They extract more value from each token they see. So even though the 10B model sees less data overall, it learns more from what it does see.

This is exactly what the scaling laws predict. Remember those exponents? N ~ C^0.73, D ~ C^0.27. The model size grows much faster than the data size.

And this is why GPT-3—with 175 billion parameters trained on 300 billion tokens—outperformed models that were trained to convergence on way more data. It's all about that sample efficiency.

</details>

---

## 🤖 The Formal Algorithm

Now let's get a bit more technical. How do you actually use these scaling laws in practice?

Here's the key algorithm. If you're a practitioner and you have a compute budget C, this is how you find the optimal model size and dataset size:

```
FUNCTION ComputeOptimalAllocation(C):
    // These are the scaling exponents from the paper
    a ← 0.73  // How N scales with C
    b ← 0.27  // How D scales with C

    // These coefficients depend on your hardware
    N_coeff ← 0.3
    D_coeff ← 3.2

    // Calculate optimal allocations
    N_optimal ← N_coeff × C^0.73
    D_optimal ← D_coeff × C^0.27

    RETURN (N_optimal, D_optimal)
END
```

Let's walk through an example. Say you have a budget of 1 PetaFLOP-day. Plug that in:
- N_optimal = 0.3 × 1^0.73 ≈ **1.3 billion parameters**
- D_optimal = 3.2 × 1^0.27 ≈ **3.2 billion tokens**

Before spending $4 million on training GPT-3, OpenAI could use this formula to **predict its performance** to within 0.05 nats. That's incredibly powerful!

### The Architecture Finding

Here's something else that shocked everyone. The paper tested wildly different Transformer architectures:
- A wide, shallow model: 6 layers with 2048 dimensions
- A narrow, deep model: 48 layers with 512 dimensions

Both had the same total parameter count N. And you know what? They performed **nearly identically**—less than 0.1 nats difference!

The conclusion? At fixed N, **architecture details matter far less than everyone thought**. Scale dominates everything.

---

## 🤔 Second Question

This brings me to another question. If architecture barely matters, why do modern AI labs like Google, Meta, and OpenAI still spend so much time and effort on architecture search and optimization?

Think about it. We just said scale is what matters. So why all the fuss about architecture?

<details>
<summary><b>Click to discuss</b></summary>

Great question! And there are actually several really good reasons:

**1. Inference Efficiency**

While training performance only depends on N, **inference cost**—actually serving the model to users—depends heavily on architecture. Mixture-of-Experts models, sparse attention patterns... these can make inference way cheaper.

**2. Specialized Tasks**

The scaling laws are for general language modeling. But for specific tasks:
- Vision models need different architectures (like Vision Transformers or ConvNets)
- Long-context models need modified attention mechanisms
- Code models benefit from specialized tokenization

**3. Scaling Limits**

As models approach a trillion parameters, we're hitting constraints:
- Running out of high-quality training text
- Data bottlenecks where architecture innovations help
- Hardware limitations that different architectures handle differently

**4. Post-Training Dynamics**

The scaling laws only apply to pretraining. But RLHF and instruction-tuning have different dynamics. A smaller, well-tuned model can outperform a larger base model.

**The bottom line**: For pretraining, scale is king. But for the **full lifecycle**—training, inference, deployment, fine-tuning—architecture absolutely matters!

</details>

---

## 🌟 Impact and What Changed

Let me show you what actually changed in the industry after this paper came out.

### Before vs After

| Before (2018-2019) | After (2020+) |
|-------------------|---------------|
| Train small models to convergence | Train large models, stop early |
| BERT: 340M params, 3.3B tokens | GPT-3: 175B params, 300B tokens |
| Trial and error | Predictive formulas |
| "More data is always better" | "Bigger models are sample-efficient" |

### The Chinchilla Correction

Now here's where it gets interesting. In 2022, DeepMind revisited these scaling laws with way more compute—like 10 to 100 times more than this original paper.

And they found something surprising: **GPT-3 and Gopher were actually undertrained on data**!

They trained a model called Chinchilla—70 billion parameters on 1.4 trillion tokens. And it **outperformed Gopher**, which had 280 billion parameters but only 300 billion tokens. Same training compute, but Chinchilla was better because it had better allocation.

They refined the scaling laws to show that N and D should actually scale more equally with C—not the extreme N >> D that the original paper suggested.

The lesson? The original paper was **directionally correct**—scale matters!—but the quantitative optimum keeps shifting as we test at larger scales. This is healthy scientific iteration. The laws are guides, not gospel.

### The LLaMA Revolution

And then Meta took this even further with LLaMA in 2023. They trained smaller models on way more data:
- LLaMA-13B trained on 1 trillion tokens matches GPT-3's 175B trained on 300B
- Inference is **10x cheaper** because the model is smaller
- This enabled local deployment and sparked the entire open-source LLM ecosystem

---

## 🔬 Critical Analysis

Now let me be critical for a moment. What are the limitations of this work?

**1. Dataset Homogeneity**
All experiments used English web text (WebText2). Do these laws hold for:
- Non-English languages, especially low-resource ones?
- Specialized domains like code, math, or scientific text?
- Multimodal data like images, video, audio?

**2. Architecture Specificity**
Only tested decoder-only Transformers like GPT-2. What about:
- Encoder-decoder models like T5?
- Mixture-of-experts models?
- Non-Transformer architectures like Mamba or state-space models?

**3. Post-Training is Ignored**
The paper only looks at pretraining—predicting the next token. But modern LLMs have:
- Instruction tuning
- RLHF (reinforcement learning from human feedback)

Do scaling laws apply there? We're still figuring that out.

**4. Emergent Abilities**
Some capabilities—like chain-of-thought reasoning—seem to **suddenly appear** at certain scales. Power laws predict smooth improvements, not these discontinuous jumps. This is still hotly debated.

**5. Inference Costs**
The paper optimizes training compute but completely ignores inference compute. GPT-3 with 175B parameters is expensive to serve. Chinchilla with 70B is way cheaper, even if training cost was similar.

### The Bigger Picture

This paper is a **milestone in empirical science**. It doesn't solve the **theory** of deep learning—we still don't know *why* power laws emerge. But it gives us:
- Predictive tools that work in practice
- A framework for resource allocation
- Inspiration for all the follow-up work

It's like Kepler's laws of planetary motion—empirical patterns that came before Newton's theory gave us the mechanistic understanding.

And the **legacy**? Over 2,500 citations in just a few years. Changed how the entire industry thinks about resource allocation.

---

## 📊 Quick Reference

### The Three Laws

| Law | Formula | 10x Resource → |
|-----|---------|----------------|
| **L(N)** | (Nc/N)^0.076 | 5% loss reduction |
| **L(D)** | (Dc/D)^0.095 | 6% loss reduction |
| **L(C)** | (Cc/C)^0.050 | 3% loss reduction |

### Historical Evolution

| Model | Year | Parameters | Data | Strategy |
|-------|------|-----------|------|----------|
| BERT | 2018 | 340M | 3.3B | Convergence |
| GPT-2 | 2019 | 1.5B | 40B | Convergence |
| GPT-3 | 2020 | 175B | 300B | **Early stop** ✓ |
| Chinchilla | 2022 | 70B | 1.4T | **Refined laws** ✓ |

---

## 🧑‍💻 Code Demo

Here's the code I wrote that implements these scaling laws:

```python
from code.scaling_calculator import ScalingCalculator

calc = ScalingCalculator()

# Predict GPT-3's performance before training
N_gpt3 = 175e9  # 175 billion parameters
D_gpt3 = 300e9  # 300 billion tokens

predicted_loss = calc.predict_loss(N=N_gpt3, D=D_gpt3)
print(f"Predicted: {predicted_loss:.3f} nats")
# Output: 2.150 nats (actual was ~2.0 - very close!)

# Find optimal allocation for your budget
C = 1.0  # 1 PetaFLOP-day
N_opt, D_opt = calc.compute_optimal_allocation(C)
print(f"Optimal: {N_opt/1e9:.1f}B params, {D_opt/1e9:.1f}B tokens")
```

All the code is in the `/code` directory. You can run it yourself!

---

## 🎯 Key Takeaways

If you remember nothing else from this presentation, remember these three things:

**1. Scale is Predictable**
Performance follows power laws. You can forecast performance before spending millions on training. This was revolutionary.

**2. Bigger Models Are Sample-Efficient**
Train large models briefly rather than small models to convergence. The optimal allocation: N grows much faster than D as compute increases.

**3. Architecture Matters Less Than You Think**
At fixed parameter count, depth and width have minimal impact. For pretraining, focus on scale, not architecture search.

---

## 🔗 Resources

Want to dive deeper? Here are the key papers:

1. **[Original Scaling Laws Paper](https://arxiv.org/abs/2001.08361)** - Kaplan et al., 2020
2. **[Chinchilla Paper](https://arxiv.org/abs/2203.15556)** - Refined scaling laws, 2022
3. **[Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)** - Mathematical foundations
4. **[Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701)** - Theoretical analysis
5. **[EleutherAI Blog on Scaling Laws](https://blog.eleuther.ai/scaling-laws/)** - Practitioner guide

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

## 📚 Appendix: Extended Details

<details>
<summary><b>Click to expand: Additional algorithms, detailed findings, and extended analysis</b></summary>

### Additional Algorithms

#### Predicting Loss from Resources

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

#### Early Stopping Criterion

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

#### Critical Batch Size

```
FUNCTION CriticalBatchSize(L):
    B_noise ← 2^21  // 2M tokens
    L_noise ← 1.5
    α_D ← 0.095

    B_crit ← B_noise × (L / L_noise)^(1/α_D)

    RETURN floor(B_crit)
END
```

Start with large batches early, decrease as loss drops. Counter-intuitive but optimal!

### Detailed Experimental Findings

#### Overfitting is Universal

Optimal ratio: **N^0.74 ≈ D**

- 100M params → need 1B tokens
- 1B params → need 5.5B tokens
- 100B params → need 550B tokens

GPT-4 (~1.8T params) would need ~10 trillion tokens—more text than exists on the internet!

#### Sample Efficiency Grows with Model Size

Fixed dataset of 10B tokens:
- 100M params → Loss 3.5 nats
- 1B params → Loss 2.8 nats (20% better on same data!)
- 10B params → Loss 2.3 nats (18% better on same data!)

#### Transfer Learning Follows Power Laws

Downstream performance: `Loss_downstream ∝ Loss_pretraining^β`

LAMBADA example:
- Pretrain loss 3.0 → 45% accuracy
- Pretrain loss 2.5 → 62% accuracy
- Pretrain loss 2.0 → 78% accuracy

### Parameter Counting in Transformers

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

GPT-2 Small: L=12, d_model=768, V=50,257
- N_embed ≈ 39M
- N_layer ≈ 85M
- N_total ≈ 124M

### What's Still Missing

1. **Data quality**: All tokens treated equally (Wikipedia = Reddit spam)
2. **Multimodal scaling**: How do laws change with vision, audio?
3. **Why power laws?**: No mechanistic explanation
4. **Inference costs**: Only optimizes training

### Open Questions

- Will scaling laws hold forever? (Probably not—data bottleneck coming)
- Do they apply to RLHF? (Active research area)
- What about emergent abilities? (Smooth laws vs sudden jumps)

### The Bitter Lesson Connection

Rich Sutton's "Bitter Lesson" (2019):
> "General methods that leverage computation are ultimately most effective."

Scaling laws are strong evidence for this. But recent work shows data quality and fine-tuning matter too—so it's not *only* about scale.

</details>

---

**Questions? Want to discuss?** Find me at [@Kshetkar1](https://github.com/Kshetkar1)

**Repository**: [github.com/Kshetkar1/Scaling-Laws-For-Neural-Language-Models-Paper-Presentation](https://github.com/Kshetkar1/Scaling-Laws-For-Neural-Language-Models-Paper-Presentation)

*Presented October 2025*
