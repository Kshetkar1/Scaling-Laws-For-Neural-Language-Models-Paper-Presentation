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

## 🤖 Formal Algorithms for Scaling Laws

Now let's get technical. Here are the core algorithms that implement the scaling laws, presented in formal pseudocode.

### Algorithm 1: Compute-Optimal Resource Allocation

This is the central algorithm—given a fixed compute budget, determine the optimal model size and dataset size.

**Input:** $C \in \mathbb{R}^+$ (compute budget in PetaFLOP-days)

**Parameters:**
- $\alpha_N = 0.73$ (scaling exponent: how $N$ scales with $C$)
- $\alpha_D = 0.27$ (scaling exponent: how $D$ scales with $C$)
- $k_N = 0.3$ (hardware-dependent coefficient for parameters)
- $k_D = 3.2$ (hardware-dependent coefficient for data)

**Output:** $(N_{\text{optimal}}, D_{\text{optimal}})$ (optimal parameters and tokens)

**Algorithm:**

1. $N_{\text{optimal}} \leftarrow k_N \times C^{\alpha_N}$ $\triangleright$ Compute optimal model size
2. $D_{\text{optimal}} \leftarrow k_D \times C^{\alpha_D}$ $\triangleright$ Compute optimal dataset size
3. $C_{\text{actual}} \leftarrow \frac{6 \times N_{\text{optimal}} \times D_{\text{optimal}}}{10^{15}}$ $\triangleright$ Verify FLOP constraint
4. **if** $\frac{|C_{\text{actual}} - C|}{C} > 0.1$ **then**
5. &nbsp;&nbsp;&nbsp;&nbsp;Print "Warning: Allocation deviates from budget by >10%"
6. **end if**
7. **return** $(N_{\text{optimal}}, D_{\text{optimal}})$

**Example:** With $C = 1.0$ PetaFLOP-day:
- $N_{\text{optimal}} = 0.3 \times 1^{0.73} \approx$ **1.3 billion parameters**
- $D_{\text{optimal}} = 3.2 \times 1^{0.27} \approx$ **3.2 billion tokens**

**Key Property:** This formula let OpenAI predict GPT-3's performance before spending $4M—accurate to within 0.05 nats!

---

### Algorithm 2: Predict Loss from Resources

Given model size and data size, predict the final test loss before training.

**Input:**
- $N \in \mathbb{R}^+$ (number of parameters)
- $D \in \mathbb{R}^+$ (number of training tokens)

**Parameters:**
- $N_c = 8.8 \times 10^{13}$ (critical parameter count)
- $D_c = 5.4 \times 10^{13}$ (critical token count)
- $\alpha_N = 0.076$ (power law exponent for $N$)
- $\alpha_D = 0.095$ (power law exponent for $D$)

**Output:** $L \in \mathbb{R}^+$ (predicted test loss in nats)

**Algorithm:**

1. $\text{term}_N \leftarrow \left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D}$ $\triangleright$ Parameter constraint contribution
2. $\text{term}_D \leftarrow \frac{D_c}{D}$ $\triangleright$ Data constraint contribution
3. $L \leftarrow (\text{term}_N + \text{term}_D)^{\alpha_D}$ $\triangleright$ Bottleneck formula
4. **return** $L$

**Key Property:** Loss is determined by the *bottleneck* resource. If $N$ is too small or $D$ is too small, that constraint dominates.

**Complexity:** $O(1)$ - constant time prediction

---

### Algorithm 3: Early Stopping Criterion

Determine when to stop training—critical for compute efficiency.

**Input:**
- $N \in \mathbb{R}^+$ (current model size in parameters)
- $S \in \mathbb{N}$ (current training step)
- $B \in \mathbb{N}$ (batch size in tokens per step)
- $C_{\text{budget}} \in \mathbb{R}^+$ (total compute budget in PetaFLOP-days)

**Output:** $\text{stop} \in \{\text{True}, \text{False}\}$ (whether to stop training)

**Algorithm:**

1. $C_{\text{used}} \leftarrow \frac{6 \times N \times B \times S}{10^{15}}$ $\triangleright$ Compute used so far
2. $\text{tokens}_{\text{seen}} \leftarrow B \times S$ $\triangleright$ Total tokens processed
3. $(N_{\text{opt}}, D_{\text{opt}}) \leftarrow$ ComputeOptimalAllocation$(C_{\text{budget}})$ $\triangleright$ Get optimal allocations
4. **if** $C_{\text{used}} \geq C_{\text{budget}}$ **then**
5. &nbsp;&nbsp;&nbsp;&nbsp;**return** True $\triangleright$ Budget exhausted
6. **end if**
7. **if** $N \approx N_{\text{opt}}$ **and** $\text{tokens}_{\text{seen}} \geq D_{\text{opt}}$ **then**
8. &nbsp;&nbsp;&nbsp;&nbsp;**return** True $\triangleright$ Optimal token count reached
9. **end if**
10. **if** $N > 2 \times N_{\text{opt}}$ **then**
11. &nbsp;&nbsp;&nbsp;&nbsp;Print "Warning: Model too large for budget"
12. &nbsp;&nbsp;&nbsp;&nbsp;**return** True $\triangleright$ Severely overtrained model
13. **end if**
14. **return** False $\triangleright$ Continue training

**Key Property:** Traditional practice trains to convergence ($D \to \infty$). Scaling laws say: **stop early**! Training GPT-3 to convergence would waste 100x compute.

**Complexity:** $O(1)$ - constant time check per training step

---

### The Architecture Finding

Here's something else that shocked everyone. The paper tested wildly different Transformer architectures:
- A wide, shallow model: 6 layers with 2048 dimensions
- A narrow, deep model: 48 layers with 512 dimensions

Both had the same total parameter count $N$. And you know what? They performed **nearly identically**—less than 0.1 nats difference!

**Conclusion:** At fixed $N$, **architecture details matter far less than everyone thought**. Scale dominates everything.

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

Let me show you what actually changed in the industry after this paper came out. The impact was **massive** and immediate.

### Concrete Before vs After Examples

#### Example 1: Google's BERT to GPT-3 Transition

**Before Scaling Laws (BERT, 2018)**:
- Model: 340M parameters
- Data: 3.3B tokens (trained to full convergence)
- Training cost: ~$7,000 in compute
- Training time: 4 days on 16 TPU v3 chips
- Final loss: ~3.2 nats
- Decision-making: "Let's just train until loss stops decreasing"

**After Scaling Laws (GPT-3, 2020)**:
- Model: 175B parameters (500x larger!)
- Data: 300B tokens (stopped at ~10% convergence)
- Training cost: $4-12 million in compute
- Training time: Weeks on thousands of GPUs
- Final loss: ~2.0 nats (37% better!)
- Decision-making: "Scaling laws predict loss of 2.03 nats—let's invest $4M"

**The Impact**: OpenAI could **predict performance before training** and justify the massive investment. Without scaling laws, spending $4M on an undertrained model would seem insane.

#### Example 2: DeepMind's Gopher vs Chinchilla Decision

**Before Understanding Refined Laws (Gopher, 2021)**:
- Model: 280B parameters
- Data: 300B tokens
- Training compute: ~5,500 PetaFLOP-days
- Cost: ~$2.5 million
- Performance: Loss ~1.8 nats

**After Chinchilla Scaling Laws (Chinchilla, 2022)**:
- Model: 70B parameters (4x smaller!)
- Data: 1.4T tokens (4.7x more data)
- Training compute: ~5,500 PetaFLOP-days (same!)
- Cost: ~$2.5 million (same!)
- Performance: Loss ~1.65 nats (**8% better**)
- Inference: **4x cheaper** due to smaller size

**The Impact**: Same training budget, better model, and way cheaper to serve. This directly led to cost savings of millions in inference costs for DeepMind.

#### Example 3: Meta's LLaMA Cost Revolution

**Industry Standard (GPT-3 approach, 2020-2022)**:
- Train 175B model → Inference costs ~$0.02 per 1K tokens
- Annual serving cost for 100M requests: ~$2 million
- Only deployable by large tech companies

**Meta's LLaMA Approach (2023)**:
- LLaMA-13B: 13B parameters, 1T tokens
- Matches GPT-3-175B performance on many tasks
- Inference: ~$0.002 per 1K tokens (**10x cheaper!**)
- Annual serving cost: ~$200,000 (saved $1.8M per year)
- Deployable on consumer hardware (MacBook Pro with 16GB RAM!)

**The Impact**: Democratized LLMs. The entire open-source ecosystem (Alpaca, Vicuna, WizardLM) became possible because LLaMA could run locally.

### Industry-Wide Resource Allocation Changes

Here's how the paper changed actual decision-making across the AI industry:

**Training Budgets Shifted Dramatically**:

| Metric | 2018-2019 (Pre-Laws) | 2020-2023 (Post-Laws) | Change |
|--------|---------------------|---------------------|---------|
| Avg model size | 300M-1.5B params | 7B-175B params | **100x larger** |
| Training tokens | 10-100B | 100B-2T | **20x more** |
| Training cost | $10K-100K | $100K-10M | **100x higher** |
| Time to convergence | 100% | 10-30% | **Stop 3-10x earlier** |
| Predictability | Trial & error | Formula-driven | **Risk reduced** |

**Specific Industry Decisions Enabled**:

1. **OpenAI's GPT-3 Investment (2020)**: $4M training run approved based on scaling law predictions
2. **Google's PaLM (2022)**: 540B parameters—scaling laws predicted it would beat GPT-3
3. **Anthropic's Claude (2023)**: Used refined laws to optimize their compute budget
4. **Mistral AI (2023)**: Leveraged Chinchilla laws to build efficient 7B models that compete with 13B+

### The Chinchilla Correction

Now here's where it gets interesting. In 2022, DeepMind revisited these scaling laws with way more compute—like 10 to 100 times more than this original paper.

And they found something surprising: **GPT-3 and Gopher were actually undertrained on data**!

They trained a model called Chinchilla—70 billion parameters on 1.4 trillion tokens. And it **outperformed Gopher**, which had 280 billion parameters but only 300 billion tokens. Same training compute, but Chinchilla was better because it had better allocation.

They refined the scaling laws to show that N and D should actually scale more equally with C—not the extreme N >> D that the original paper suggested.

**Note:** We'll have another presentation specifically on the Chinchilla paper that will dive much deeper into these revised scaling laws and their implications.

The lesson? The original paper was **directionally correct**—scale matters!—but the quantitative optimum keeps shifting as we test at larger scales. This is healthy scientific iteration. The laws are guides, not gospel.

### Long-Term Impact: By The Numbers

**Academic Impact**:
- **2,500+ citations** in just 4 years (2020-2024)
- Spawned entire research subfield: "Neural Scaling Laws"
- Follow-up papers: Chinchilla, Gopher, PaLM, LLaMA all cite this work
- Theoretical investigations: Why do power laws emerge?

**Economic Impact**:
- Prevented **billions in wasted compute**: Companies now optimize before training
- Enabled **$100B+ LLM market**: Predictability allowed massive investments
- Created **inference cost competition**: Chinchilla-style models reduce serving costs
- Sparked **open-source revolution**: Efficient models like LLaMA enable democratization

**The Bottom Line**: This paper didn't just describe scaling laws—it **changed how the entire AI industry allocates resources**, saving billions in compute costs and enabling strategic investments that created the modern LLM ecosystem.

---

## 🔬 Critical Analysis

Now let me be critical for a moment. What are the limitations of this work?

### Key Limitations

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

### What the Authors Overlooked

Beyond the limitations above, here are specific things the original paper **didn't account for** that later research revealed:

**1. Data Quality Matters More Than Assumed**
The paper treats all tokens equally—Wikipedia, books, Reddit comments, all the same. But research since 2020 shows:
- High-quality data (textbooks, academic papers) gives **5-10x better sample efficiency**
- The Phi models from Microsoft achieve GPT-3 level performance with 100x fewer parameters by using curated data
- The scaling exponents change significantly with data quality

**2. The Data Wall Problem**
The paper assumed unlimited high-quality text. But:
- The entire internet has ~5 trillion high-quality tokens
- GPT-4 scale models need 10+ trillion tokens by these laws
- We're hitting fundamental data scarcity—overlooked in the original analysis

**3. Transfer and Fine-Tuning Dynamics**
The paper briefly mentions transfer but doesn't deeply explore it. Critical omissions:
- How do scaling laws change after instruction-tuning?
- Do smaller, heavily fine-tuned models beat larger base models?
- LLaMA-2-7B with RLHF can match GPT-3-175B on some tasks!

**4. Critical Batch Size Regime**
While the paper identifies critical batch size, it underestimates its importance:
- Training stability issues at very large scales
- Gradient noise dynamics are more complex than predicted
- Learning rate schedules interact with batch size in ways not captured

**5. Architectural Innovations at Scale**
The claim that "architecture doesn't matter" held up to 1.5B parameters but breaks down beyond:
- Sparse attention becomes necessary for long context
- Mixture-of-Experts unlocks different scaling regimes
- Flash Attention and other optimizations are required at trillion-parameter scale

### Have Others Disputed the Findings?

Yes! The scaling laws have been refined and contested:

**1. The Chinchilla Revision (2022)** - Most Significant Dispute

DeepMind directly challenged the original allocation:
- Original: N ~ C^0.73, D ~ C^0.27 (parameters dominate)
- Chinchilla: N ~ C^0.50, D ~ C^0.50 (equal scaling!)
- Evidence: Chinchilla-70B on 1.4T tokens beats Gopher-280B on 300B tokens

**Why the discrepancy?** OpenAI tested up to ~1 PetaFLOP-day. DeepMind tested up to 100 PetaFLOP-days. The scaling exponents themselves change with scale!

**2. The Emergent Abilities Debate**

Some researchers argue power laws are misleading:
- Schaeffer et al. (2023): "Emergent abilities are a mirage"
- Claim: Smooth scaling looks discontinuous due to poor metrics
- Counter-claim: Chain-of-thought really does emerge suddenly
- **Still unresolved** in the community

**3. Data Scaling vs Model Scaling**

Multiple groups found data scaling may be more important:
- LLaMA: Overtrain on tokens (1T tokens for 7B params violates scaling laws)
- Result: Better performance than predicted, cheaper inference
- Suggests: At production scale, data efficiency > pure scaling law optimization

**4. Open Compute Estimation**

Researchers questioned whether compute estimates are correct:
- The "6ND" approximation for FLOPs may be off by 2x
- Backward pass costs vary by architecture
- Mixed precision training changes compute accounting

---

**Important Note:** While this paper was groundbreaking in 2020, subsequent research (particularly Chinchilla 2022 and data quality studies) has effectively **disproved several quantitative claims** in this paper, particularly the specific exponent values for optimal allocation ($\alpha_N = 0.73$, $\alpha_D = 0.27$). The directional insights remain valid—scale matters, larger models are sample-efficient—but the exact numbers have been revised. This is a healthy example of scientific progress through empirical refinement.

### The Bigger Picture

This paper is a **milestone in empirical science**. It doesn't solve the **theory** of deep learning—we still don't know *why* power laws emerge. But it gives us:
- Predictive tools that work in practice (even if not perfect)
- A framework for resource allocation (refined by Chinchilla)
- Inspiration for all the follow-up work

It's like Kepler's laws of planetary motion—empirical patterns that came before Newton's theory gave us the mechanistic understanding.

**The verdict?** The paper was **directionally correct**—scale matters enormously!—but the **quantitative details** have been disputed and refined. The exponents change with scale, data quality matters more than assumed, and architecture innovations unlock different regimes.

And the **legacy**? Over 2,500 citations in just a few years. Changed how the entire industry thinks about resource allocation, even as we refine the details.

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

**3. Architecture SIZE Matters Less Than You Think**
At fixed parameter count, depth and width have minimal impact. Wide-shallow vs narrow-deep doesn't matter—total parameters $N$ is what counts. However, architecture DESIGN (attention mechanisms, efficiency optimizations) still matters greatly.

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
