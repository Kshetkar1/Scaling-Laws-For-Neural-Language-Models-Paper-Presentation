# Scaling Laws for Neural Language Models
## 📝 PRESENTATION SCRIPT (Word-for-Word Speaking Notes)

**Presenter**: Kanu Shetkar
**Course**: DS 5690 - Generative AI Models in Theory & Practice
**Vanderbilt University | Fall 2025**

---

## Opening and Paper Information

Good morning everyone. Today I'm excited to talk about a paper that fundamentally changed how we train large language models. This is "Scaling Laws for Neural Language Models" by Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, and their colleagues at OpenAI and Johns Hopkins University. It was published in January 2020, and you can find it on arXiv with the identifier 2001.08361. In just under five years, this paper has accumulated over 2,500 citations, which tells you just how influential it's been in shaping modern AI development.

---

## The Problem

So let me set the stage for you. Picture this: it's late 2019, and you're a researcher at OpenAI or Google DeepMind. You have a massive compute budget—we're talking millions of dollars worth of GPU time. And you're sitting in a meeting room with your team, trying to answer some really fundamental questions. Should we build bigger models with more parameters? Or should we train smaller models on way more data? How much compute do we actually need to hit our performance targets?

Before this paper came out in 2020, AI companies had absolutely no systematic way to answer these questions. It was all intuition, gut feeling, and trial and error. Some people would say "bigger is always better!" Others would argue "no, we just need more training data!" But nobody really knew for sure. And here's the kicker: the stakes were incredibly high. Training GPT-3 ended up costing somewhere between four and twelve million dollars. If you get the resource allocation wrong, you could waste millions of dollars. Or worse, you could miss out on critical performance improvements that your competitors achieve. This wasn't just an academic question—it was a multi-million dollar business decision that companies were making blind.

---

## What This Paper Did

This is where this paper comes in, and what the authors did was truly remarkable. They took a completely empirical, scientific approach to answering these questions. OpenAI trained over 400 different language models—that's not a typo, over 400 models!—spanning seven orders of magnitude in scale. Think about that range for a second. They went from tiny models with 768,000 parameters all the way up to 1.5 billion parameters. They systematically varied everything: the number of layers, the hidden dimension size, the number of attention heads, the dataset size, the training duration. They left no stone unturned.

And then they made their groundbreaking discovery: language model performance follows precise mathematical power laws. Now, I know when you hear "power laws" it might sound abstract or theoretical, but stay with me because this is actually incredibly practical and predictable.

---

## The Three Scaling Laws - Introduction

Think of power laws like compound interest at a bank. You know how when you deposit money in a savings account, each additional dollar doesn't give you the same absolute return, but the growth pattern is predictable? You can calculate exactly how much you'll have in ten years using a formula. Power laws work the same way. Each additional parameter or data token gives you diminishing returns, but those returns follow a precise mathematical curve that lets you forecast future performance.

The paper discovered three fundamental scaling laws that govern language model performance. Let me walk you through each one.

---

## Law 1: Performance vs Model Size

The first law relates performance to model size, measured in parameters. The formula is L of N equals N-c divided by N, all raised to the power of 0.076. Now, let me translate that into plain English for you. If you multiply your model size by ten—so if you go from one billion parameters to ten billion parameters—you get approximately a five percent reduction in loss. And here's what's remarkable: this relationship holds consistently across a huge range. The paper showed it works from one million parameters all the way up to one billion parameters. It's not just true at one scale—it's a universal pattern.

Think of N as your model's capacity—how many parameters it has to work with. The bigger N gets, the better your loss L becomes, but with diminishing returns. Doubling your model size doesn't double your performance; it follows this precise power law with an exponent of 0.076.

---

## Law 2: Performance vs Data Size

The second law relates performance to data size, measured in tokens. The formula is L of D equals D-c divided by D, raised to the power of 0.095. What does this mean? If you multiply your training data by ten—say you go from 10 billion tokens to 100 billion tokens—you get approximately a six percent reduction in loss.

Notice something interesting here: the exponent for data, 0.095, is slightly larger than the exponent for parameters, which was 0.076. This means data is slightly more efficient than parameters on a per-unit basis. Each additional token of data gives you a bit more bang for your buck than each additional parameter. But—and this is crucial—data and parameters are not interchangeable in terms of compute cost, which brings us to the third law.

---

## Law 3: Performance vs Compute Budget - The Most Important One

The third law is the most important one for practical applications. It relates performance to your total compute budget. The formula is L of C equals C-c divided by C, raised to the power of 0.050. This law answers the critical question: given a fixed compute budget C—measured in PetaFLOP-days, which is a unit of computational work—how should you allocate that budget between model size and data size?

Here's where it gets really interesting and counterintuitive. The optimal allocation that the paper derived is this: parameters should scale as C to the power of 0.73, and data should scale as C to the power of 0.27. Let me emphasize what this means. As you increase your compute budget, you should grow your model size much, much faster than your dataset size. For every ten-x increase in compute, you should use about 5.4 times more parameters but only about two times more data.

This was completely backwards from conventional wisdom at the time. Before this paper, the standard practice was to pick a model architecture, pick a dataset, and then train until the model converges—until the loss stops improving. The bigger your compute budget, the longer you'd train, but you'd keep the model size relatively fixed. The scaling laws said: no, flip that around! Spend your compute on bigger models trained for less time.

---

## The Key Insight: Train Large Models, Stop Early

Let me make this concrete with a comparison that really drives home how revolutionary this insight was. The old way of thinking was: train small models to convergence. Let them see the data over and over until they've squeezed out every drop of performance. The new way that emerged from these scaling laws is: train huge models, stop early, well before convergence. Don't let them see all the data multiple times. Just make them big and give them a single pass or a modest number of passes.

Let me give you a real example from the paper. Using the old approach, you'd train a 340 million parameter model on 3.3 billion tokens until convergence. This costs about seven thousand dollars in GPU time, and you end up with a loss around 3.2 nats. Using the new scaling law approach with the exact same seven thousand dollar budget, you'd train a one billion parameter model on just one billion tokens and stop early—way before convergence. Same cost, but you get a loss around 2.7 nats. That's sixteen percent better performance for the same money!

This insight directly enabled the creation of GPT-3, which has 175 billion parameters but was only trained on 300 billion tokens—just about two epochs over the data. It enabled GPT-4, Gopher, and basically all the modern large language models we use today. Without this paper, we'd still be training small models to convergence and wondering why they weren't performing as well as we hoped.

---

## Formal Algorithms - Introduction

Now let's get into the mathematical machinery that implements these scaling laws. I want to walk you through three core algorithms that the paper presents. These are written in formal pseudocode notation, and I'm going to explain each one step by step so you understand exactly how they work. If you're the kind of person who wants to actually implement these scaling laws in practice, this is the section you need to pay attention to.

---

## Algorithm 1: Compute-Optimal Resource Allocation

The first algorithm is called Compute-Optimal Resource Allocation, and this is really the heart of the paper. Given a fixed compute budget, this algorithm tells you exactly how big your model should be and how much data you should use. Let me walk through it line by line.

The input to this algorithm is C, which is your compute budget measured in PetaFLOP-days. A PetaFLOP-day is one quadrillion floating-point operations per second sustained for an entire day. To give you a sense of scale, training GPT-3 took about 3,000 to 4,000 PetaFLOP-days.

The algorithm has four key parameters that were empirically determined from the 400-plus models they trained. First, we have alpha-N equals 0.73. This is the exponent that determines how model size scales with compute. Second, we have alpha-D equals 0.27. This is the exponent for data scaling. Notice these two add up to 1.0, which makes sense because we're allocating a fixed resource. Third, we have k-N equals 0.3, which is a hardware-dependent coefficient for model size. And fourth, we have k-D equals 3.2, which is the hardware coefficient for data size.

Now let's walk through the algorithm step by step. Step one: Calculate N-optimal as k-N times C raised to the alpha-N power. In code terms, that's N-optimal equals 0.3 times C to the 0.73. This gives us our optimal model size in billions of parameters. The intuition here is that as we increase our compute budget C, the model size grows at this fast rate determined by the 0.73 exponent.

Step two: Calculate D-optimal as k-D times C raised to the alpha-D power. In code, that's D-optimal equals 3.2 times C to the 0.27. This gives us our optimal dataset size in billions of tokens. Notice that data grows much slower than model size because that exponent is only 0.27.

Step three: Verify the constraint. We calculate C-actual as 6 times N-optimal times D-optimal, divided by ten to the fifteenth. This is checking that our allocation actually uses up the compute budget we said we had. The factor of 6 comes from the fact that in a forward-backward pass through a Transformer, you do roughly six operations per parameter per token—one forward, two backward, plus some overhead.

Steps four and five: Check if the absolute value of C-actual minus C, divided by C, is greater than 0.1—that's a ten percent error tolerance. If so, print a warning. This is basically a sanity check to make sure our allocation is feasible.

Step six: Return the pair N-optimal and D-optimal. That's your answer—how big your model should be and how much data you should use.

Let me give you a concrete example. Suppose C equals 1.0 PetaFLOP-day. Then N-optimal equals 0.3 times 1.0 to the 0.73, which is approximately 0.3 times 1.0, so about 1.3 billion parameters. And D-optimal equals 3.2 times 1.0 to the 0.27, which is approximately 3.2 billion tokens. So for one PetaFLOP-day of compute, you should train a 1.3 billion parameter model on 3.2 billion tokens.

And here's the key property that makes this so powerful: this formula let OpenAI predict GPT-3's performance to within 0.05 nats before spending four million dollars on training! Think about that. They could plug in their compute budget, get N-optimal and D-optimal, plug those into the loss prediction formula, and know within a very tight margin what their final performance would be. No more guesswork, no more expensive trial and error. That's incredibly powerful for planning and resource allocation.

---

## Algorithm 2: Predict Loss from Resources

The second algorithm is Predict Loss from Resources. This is the flip side of Algorithm 1. Instead of asking "given compute, what's the optimal allocation?", this asks "given a specific model size and data size, what loss will I achieve?" This is useful for forecasting performance before you've committed to training.

The inputs are simple: N, your model size in parameters, and D, your dataset size in tokens. You've already decided these, or you want to evaluate a hypothetical scenario.

The parameters are four constants that were fitted from the empirical data. First, N-c equals 8.8 times ten to the thirteenth—that's 88 trillion. This is a characteristic scale for parameters. Second, D-c equals 5.4 times ten to the thirteenth—that's 54 trillion. This is a characteristic scale for data. Third, alpha-N equals 0.076—that's the parameter exponent we saw in Law 1. Fourth, alpha-D equals 0.095—that's the data exponent from Law 2.

Now the algorithm itself is beautifully simple. It's only four steps, and the complexity is order one—constant time. You don't need to loop over anything or do expensive computations.

Step one: Calculate term-N as N-c divided by N, raised to the power of alpha-N divided by alpha-D. So that's 88 trillion divided by N, raised to the 0.076 divided by 0.095, which is 0.8. This term represents the constraint from having finite parameters. If your model is small compared to N-c, this term is large and dominates your loss.

Step two: Calculate term-D as D-c divided by D. That's 54 trillion divided by D. This term represents the constraint from having finite data. If you have little data compared to D-c, this term is large.

Step three: Calculate L as term-N plus term-D, all raised to the alpha-D power. So L equals the sum of those two terms, raised to the 0.095. This is called the "bottleneck formula." The idea is that your final loss is determined by whichever resource—parameters or data—is your bottleneck. The sum captures both constraints.

Step four: Return L. That's your predicted test loss in nats.

Let me give you an example to make this concrete. Suppose you're evaluating GPT-3, which has N equals 175 billion parameters and D equals 300 billion tokens. Term-N equals 88 trillion divided by 175 billion, which is about 503, raised to the 0.8 power, which is about 106. Term-D equals 54 trillion divided by 300 billion, which is 180. Then L equals 106 plus 180, which is 286, raised to the 0.095 power. That gives approximately 1.73 nats. The actual GPT-3 loss was about 2.0 nats, so the prediction is in the right ballpark. The discrepancy comes from the fact that GPT-3 wasn't perfectly optimally trained according to these laws, plus there's measurement noise and other factors.

The key insight from this algorithm is that you can predict performance before training. No more flying blind. You can run this calculation in milliseconds and decide whether a proposed model configuration is worth the cost.

---

## Algorithm 3: Early Stopping Criterion

The third algorithm is the Early Stopping Criterion. This one is super practical because it tells you when to stop training. Remember, the whole insight from the scaling laws is to train large models and stop early. But how early is early? This algorithm answers that question.

The inputs are: N, your model size in parameters; S, the current training step number; B, your batch size in tokens; and C-budget, your total compute budget in PetaFLOP-days.

The algorithm has seven steps, and like Algorithm 2, the complexity is constant time—order one.

Step one: Calculate C-used as 6 times N times B times S, divided by ten to the fifteenth. This computes how much compute you've used so far. Again, that factor of 6 accounts for forward pass, backward pass, and overhead. B times S gives you total tokens processed, and 6 times N times that gives you total FLOPs, which we convert to PetaFLOPs by dividing by ten to the fifteen.

Step two: Calculate tokens-seen as B times S. This is straightforward—how many tokens has your model seen so far in training?

Step three: Call ComputeOptimalAllocation with C-budget to get N-opt and D-opt. This is calling Algorithm 1 to figure out what the optimal model size and data size would be for your budget.

Now we have three stopping conditions. Step four: If C-used is greater than or equal to C-budget, return True. This is the obvious one—you've run out of compute, so stop.

Step five: If N is approximately equal to N-opt and tokens-seen is greater than or equal to D-opt, return True. This condition says: if you built the right-sized model according to the scaling laws, and you've seen the right amount of data, then stop. You're done. Don't keep training past this point because you're wasting compute.

Step six: If N is greater than two times N-opt, return True. This is a safety condition. If your model is way bigger than optimal for your budget—more than double—then stop early. You won't have enough compute to properly train it, so cut your losses.

Step seven: If none of the above conditions are met, return False. Keep training.

Here's the key property of this algorithm: it saves you massive amounts of compute. The paper showed that training to full convergence—the traditional approach—can waste one hundred times more compute than stopping at the optimal point! Think about that. If you're spending a million dollars on a training run, you could achieve the same performance for ten thousand dollars by stopping at the right time. Or you could use that million dollars much more efficiently to train a bigger model for less time and get way better performance. This algorithm is the practical tool that makes that possible.

---

## Question 1: Architecture Intuition

Now, before I talk about the architecture findings, let me test your intuition with a question. Imagine you have a budget of exactly 100 million parameters—that's your constraint, you can't go over it. You need to choose an architecture for your Transformer language model. Which of these two options do you think would give better performance?

Option A: Wide and Shallow. Six layers, each with 2048 dimensions. If you do the math on a Transformer with these dimensions, that comes out to roughly 100 million parameters total.

Option B: Narrow and Deep. 48 layers—so eight times as many layers—but each layer only has 512 dimensions. This also comes out to approximately 100 million parameters.

Think about it for a moment. We've all heard conventional wisdom about deep learning, right? "Deeper is better." "Width gives you capacity but depth gives you abstraction." Which architecture would you bet on?

[PAUSE FOR 30 SECONDS]

Okay, here's the surprising answer that shocked everyone when this paper came out: they perform almost identically! The paper ran this exact experiment, and they found that at the same parameter count N, these two wildly different architectures had less than 0.1 nats difference in loss. That's within noise levels. For all practical purposes, they're the same.

The key finding here is that architecture shape—wide versus narrow, shallow versus deep—matters far less than the total parameter count. This was really shocking because all the conventional wisdom in deep learning said things like "deeper networks learn hierarchical features better" or "wider networks generalize better because they're less overparameterized." But the scaling laws showed: at fixed N, just count the parameters. The way you arrange them matters much less than you'd think.

Now, I need to give you an important caveat. This finding held true in the range of models they tested, which went from 768,000 parameters up to 1.5 billion parameters. At much larger scales—like 100 billion parameters and beyond—architecture innovations like sparse attention, Mixture-of-Experts, different positional encodings, these do start to matter. But they matter for different reasons: they help with efficiency, they enable longer context windows, they make training more stable. The basic principle that "parameter count dominates performance" still holds, but the architectural details become important for practical engineering reasons.

---

## The Architecture Finding - Details

Let me show you exactly what they tested. They took two models with the exact same total parameter count N. One model was wide and shallow: six layers with 2048 dimensions per layer. The other model was narrow and deep: 48 layers with 512 dimensions per layer. These are radically different architectures, right? One model does six passes through the data with lots of computation per pass. The other does 48 passes with less computation per pass. Conventional wisdom would say they'd behave very differently.

But the result was clear: nearly identical performance. Less than 0.1 nats difference in loss. The conclusion? At fixed N, architectural details like depth versus width barely matter. Scale dominates everything. If you want better performance, get more parameters. Don't agonize over whether to make your model wider or deeper—just make it bigger.

---

## Question 2: Why Architecture Still Matters

This naturally leads to a question that I know some of you are thinking. If architecture size barely matters, and it's all about parameter count, then why do modern AI labs like OpenAI, Google, Meta, Anthropic—why do they spend so much time and effort on architecture search and innovation? Think about it. We just said scale is king. So why all the research papers about new attention mechanisms, new layer types, new architectural designs?

Well, there are several really important reasons why architecture still matters a lot in practice. Let me walk through them.

First, inference efficiency. While training performance depends primarily on N, inference cost—actually serving the model to users in production—depends heavily on architecture. A dense 175 billion parameter model like GPT-3 costs a fortune to run. You need multiple high-end GPUs just to load it into memory. But a Mixture-of-Experts model with the same total parameter count can be much cheaper to run because only a subset of parameters are active for each token. Sparse attention patterns can reduce the quadratic cost of attention down to linear. These architectural innovations don't necessarily improve pretraining loss at fixed N, but they make deployment economically feasible.

Second, specialized tasks. The scaling laws in this paper are specifically for general language modeling—predicting the next token in a sequence of text. But in practice, we care about lots of different tasks. Vision models need to process images, which have very different structure than text. Vision Transformers, ConvNets, these architectures matter a lot for vision. Long-context models that need to handle, say, an entire book in context—they need modified attention mechanisms like sparse attention or state-space models. Code models benefit from specialized tokenization and architectural tweaks that capture code structure better. So while scale matters most for general language modeling, architecture matters a lot when you move to specialized domains.

Third, hardware constraints and scaling limits. As models approach a trillion parameters and beyond, we're hitting real practical constraints. We're running out of high-quality training text on the internet. Data bottlenecks mean we need architectural innovations that learn more efficiently from data—that's part of why data quality has become such a focus. Memory limitations on GPUs mean we need model parallelism, pipeline parallelism, and architectures designed to be split across machines efficiently. Communication bandwidth between GPUs becomes a bottleneck, so architecture design that minimizes communication is critical.

And fourth, post-training dynamics. The scaling laws in this paper only apply to pretraining—the phase where you're just predicting the next token. But modern language models go through additional phases: instruction tuning, where you teach them to follow instructions; and RLHF, reinforcement learning from human feedback, where you align them with human preferences. These post-training phases have different dynamics. A smaller model that's been heavily tuned with RLHF can outperform a larger base model. Architecture might matter more in these regimes.

So the bottom line is this: for pretraining from scratch, scale is king. If you want the best pretraining loss, focus on getting more parameters and more data according to the scaling laws. But for the full lifecycle of an AI system—training, fine-tuning, deployment, inference at scale—architecture absolutely matters. It's not that architecture doesn't matter; it's that for the specific question of pretraining loss at fixed compute, parameter count dominates over architectural details.

---

## Impact and What Changed After This Paper

Let me show you concretely what changed in the AI industry after this paper came out. It's actually pretty dramatic if you look at the before and after.

Before this paper—so we're talking 2018, 2019—the dominant paradigm was to train small models to convergence. BERT, which came out in 2018, had 340 million parameters and was trained on 3.3 billion tokens until the loss stopped improving. Everything was trial and error. You'd pick a model size, pick a dataset, train it, see how it did, maybe try again with different hyperparameters. The guiding motto was basically "more data is always better" and "train until convergence."

After 2020, when this paper came out, everything flipped. The new paradigm became: train large models and stop early. GPT-3 in 2020 had 175 billion parameters—that's over 500 times bigger than BERT—trained on 300 billion tokens. Crucially, it was not trained to convergence. It saw each token about twice on average, then they stopped. We started using predictive formulas. Before spending millions on a training run, labs would use the scaling laws to forecast performance. And the guiding principle became "bigger models are sample-efficient"—they learn more from each token.

You can see this shift in the models that were released. BERT and GPT-2 in the before era: relatively small, trained to convergence. GPT-3, Gopher, Chinchilla, LLaMA, all the modern models in the after era: much larger, stopped early according to scaling principles.

---

## The Chinchilla Correction

Now here's where the story gets interesting and shows you how science is supposed to work. In 2022, DeepMind published a paper on Chinchilla that revisited these scaling laws at much larger scales. They tested with ten to a hundred times more compute than the original paper. And they found something important: GPT-3 and Gopher were actually undertrained on data relative to what's truly optimal!

Let me explain what they did. They trained a model called Chinchilla with 70 billion parameters on 1.4 trillion tokens. And they compared it to Gopher, which had 280 billion parameters but was only trained on 300 billion tokens—following the original scaling laws from this paper. Both models used roughly the same amount of training compute. And guess what? Chinchilla outperformed Gopher despite being four times smaller!

DeepMind's revised scaling laws said that N and D should actually scale more equally with compute. Instead of N scaling as C to the 0.73 and D as C to the 0.27—that extreme ratio where you grow parameters much faster than data—Chinchilla found they should both scale closer to C to the 0.50. Much more balanced.

So what's going on here? Why did the scaling exponents change? The answer is that scaling laws themselves can evolve as you test at different scales. The original paper tested up to about one PetaFLOP-day of compute. DeepMind tested up to 100 PetaFLOP-days—two orders of magnitude larger. At these larger scales, different constraints start to bind. The sample efficiency advantage of larger models might not be as extreme as it appears at smaller scales.

This is actually healthy scientific iteration. The original paper was directionally correct—scale matters enormously, and there are predictable power laws! But the exact quantitative optimum keeps getting refined as we test at larger and larger scales. The scaling laws are guides, not gospel. They give you a framework for thinking about resource allocation, but you need to keep calibrating them with empirical data at the scale you're actually working at.

The lesson for practitioners? Use scaling laws as a starting point for planning, but if you have the resources, do your own calibration experiments at your target scale. Don't treat the exponents as immutable constants—they're empirical fits that can shift.

---

## The LLaMA Revolution

And then Meta took things even further with LLaMA in 2023. They trained a series of smaller models on way more data than the original scaling laws would suggest. LLaMA-13B, with 13 billion parameters, was trained on one trillion tokens. According to the original scaling laws, that's "overtraining"—you're using way more data than optimal for that model size.

But guess what? It worked incredibly well. LLaMA-13B matches GPT-3's 175 billion parameters on many benchmarks. It's thirteen times smaller, which means inference is about ten times cheaper. You can run LLaMA-13B on a single high-end GPU. GPT-3 requires a cluster. This made local deployment feasible for the first time and directly sparked the entire open-source LLM ecosystem we have today.

So what does this tell us? The scaling laws from the original paper gave us the insight that scale is predictable and that bigger models are sample-efficient. But the exact allocation between model size and data size depends on your objective function. If you're optimizing purely for pretraining loss at a fixed training compute budget, use the scaling laws as written. But if you're optimizing for deployment cost, inference efficiency, or making models accessible to people without huge compute clusters, then training smaller models on more data—even past the "optimal" point—can make a lot of sense.

This is why research is iterative. Each paper builds on the last, refines the understanding, and reveals new tradeoffs.

---

## Critical Analysis - Key Limitations

Now let me put on my critical lens for a moment. As important as this paper is, it has limitations. Any good scientist should be able to critique even foundational work. So let's talk about what this paper doesn't cover and where we should be cautious about generalizing.

First, dataset homogeneity. All the experiments in this paper used English web text, specifically a dataset called WebText2. It's basically web pages crawled from the internet, filtered for quality. But that raises questions: do these scaling laws hold for non-English languages, especially low-resource languages that have way less training data available? What about specialized domains like code, mathematics, or scientific papers? What about multimodal data—images, video, audio? The paper doesn't test any of that. So when you're working with a non-English or specialized domain, you should take these scaling laws as a starting point, not gospel.

Second, architecture specificity. They only tested one type of architecture: decoder-only Transformers, specifically the GPT-2 style. What about encoder-decoder models like T5? What about Mixture-of-Experts models that activate different subsets of parameters? What about completely non-Transformer architectures like state-space models or Mamba? We don't know if the same scaling laws apply. Recent work suggests they might be reasonably general, but it's an open question.

Third, post-training is ignored. The paper is entirely about pretraining—predicting the next token. But modern LLMs have a whole second phase: instruction tuning to make them follow instructions, and RLHF to align them with human preferences. Do the same scaling laws apply to RLHF? That's still an active research area. There are hints that smaller, well-aligned models can punch above their weight compared to their pretraining loss would suggest.

Fourth, emergent abilities. Some capabilities, like chain-of-thought reasoning or few-shot learning on complex tasks, seem to suddenly appear at certain scales. Power laws predict smooth, continuous improvements, not these discontinuous jumps. How do we reconcile that? There's an active debate in the community about whether emergent abilities are real or just artifacts of how we measure things. But either way, the smooth power laws don't capture this phenomenon.

And fifth, inference costs. The paper optimizes training compute but completely ignores inference compute. But in practice, most of the cost of a deployed LLM isn't training—it's inference. Serving billions of requests to users. GPT-3 with 175 billion parameters is incredibly expensive to serve. Chinchilla with 70 billion parameters is way cheaper. If inference cost matters for your application, you need to factor that into your allocation decisions, and this paper doesn't help with that.

---

## What the Authors Overlooked

Beyond the limitations I just mentioned, let me highlight specific things that later research revealed were oversimplified or overlooked in the original paper.

First, data quality matters way more than the paper assumed. In this paper, all tokens are treated equally. A Wikipedia article counts the same as a Reddit comment, which counts the same as a random blog post. But research since 2020 has shown that high-quality data—like textbooks, academic papers, high-quality books—gives you five to ten times better sample efficiency. Microsoft's Phi models achieve GPT-3 level performance with a hundred times fewer parameters by using very carefully curated, high-quality data. The scaling exponents themselves change significantly depending on data quality. So the paper's assumption that "a token is a token" is too simplistic.

Second, the data wall problem. The paper implicitly assumed unlimited high-quality training data. Just scale up D according to the formula, right? But we're hitting real limits. The entire publicly accessible internet has maybe five trillion high-quality tokens, depending on how you count. GPT-4 scale models, if you follow the original scaling laws, would need ten trillion plus tokens. We're running out of text! This has led to a huge focus on synthetic data, data augmentation, and curriculum learning—ways to extract more value from limited data. The original paper didn't foresee this bottleneck.

Third, transfer and fine-tuning dynamics are more complex than the paper suggested. The paper briefly mentions transfer learning but doesn't deeply explore it. How do scaling laws change after instruction tuning? Do smaller, heavily fine-tuned models beat larger base models? The answer is: yes, often! LLaMA-2-7B with RLHF can match or beat GPT-3-175B on instruction-following tasks. That's a huge practical consideration that the scaling laws for pretraining don't capture.

Fourth, critical batch size. The paper identifies something called the "critical batch size"—the point beyond which making your batches bigger doesn't speed up training. But it underestimates how important this is. At very large scales, training stability issues arise. Gradient noise dynamics are more complex than the simple power laws predict. Learning rate schedules interact with batch size in subtle ways that the paper doesn't fully capture.

And fifth, architectural innovations at scale. The claim that "architecture doesn't matter" held up to 1.5 billion parameters in their experiments. But it breaks down beyond that scale. At tens or hundreds of billions of parameters, sparse attention becomes necessary for long context. Mixture-of-Experts unlocks different scaling regimes—you can have a trillion total parameters but only activate 10 billion per token. Flash Attention and other memory optimizations are required to even fit these models on GPUs. Architecture matters a lot at the scales we're actually deploying today.

---

## Have Others Disputed the Findings?

Yes, absolutely. And this is a good thing! Science advances through disagreement and refinement. Let me walk you through the major disputes and revisions.

The most significant one is the Chinchilla Revision I mentioned earlier. DeepMind directly challenged the original allocation in 2022. The original paper said N scales as C to the 0.73, D scales as C to the 0.27—parameters dominate, data is secondary. Chinchilla said: no, they should scale more equally, both as C to the 0.50. The evidence? Chinchilla with 70 billion parameters on 1.4 trillion tokens beats Gopher with 280 billion parameters on 300 billion tokens, same compute budget. That's a direct empirical refutation of the original exponents.

Why the discrepancy? Scale matters. OpenAI tested up to about one PetaFLOP-day of compute in this paper. DeepMind tested up to 100 PetaFLOP-days—two orders of magnitude larger. The scaling exponents themselves are not constant—they change with the regime you're in!

There's also the emergent abilities debate. Some researchers, notably Schaeffer and colleagues in a 2023 paper, argued that "emergent abilities are a mirage." They claim that what looks like sudden jumps in capability are actually smooth improvements that appear discontinuous because we're using the wrong metrics or evaluation methods. Others counter-claim that certain capabilities, like chain-of-thought reasoning, really do emerge suddenly at scale. This is still unresolved, but it challenges the smooth power law story.

Then there's the data scaling versus model scaling debate. Multiple research groups have found that data scaling may be more important than the original paper suggested. LLaMA deliberately "overtrains" relative to the original scaling laws—one trillion tokens for seven billion parameters. That should be suboptimal according to the original formulas. But the result is better performance than predicted, and crucially, much cheaper inference because the model is smaller. This suggests that when you're optimizing for deployment rather than just training loss, the optimal point shifts.

And finally, there's been pushback on compute estimation. Researchers have questioned whether the compute estimates in the paper are accurate. The "6ND" approximation for FLOPs—six operations per parameter per token—might be off by a factor of two depending on implementation details. Backward pass costs vary by architecture. Mixed precision training changes the compute accounting because not all operations are full-precision. These might seem like technical details, but if your compute estimates are wrong by 2x, your scaling law calibration is off.

The bigger picture? This paper was a massive step forward and directionally correct. Scale matters enormously, and it's predictable! But the quantitative details—the exact exponents, the exact optimal allocation—these have been disputed, refined, and updated as we've tested at larger scales and with different objectives. That's how science is supposed to work.

---

## The Bigger Picture and Legacy

Let me zoom out and put this paper in historical context. This paper is a milestone in what I'd call empirical science of deep learning. It doesn't solve the fundamental theory of why deep learning works or why power laws emerge. We still don't have a theoretical derivation from first principles that explains these exponents. But it gives us predictive tools that work incredibly well in practice, even if we don't fully understand why.

Think of it like Kepler's laws of planetary motion from the early 1600s. Kepler observed that planets move in ellipses, not circles, and he derived precise mathematical laws for their motion. But he didn't know why. He didn't have Newton's theory of gravity that came decades later. The laws were empirical patterns—they predicted planetary positions incredibly well, but the mechanistic understanding came later. This scaling laws paper is like that. We have the patterns, we can make predictions, but the deep theoretical understanding of why neural networks follow power laws is still an open problem.

So what's the verdict on this paper? It was directionally correct—scale matters enormously! You can predict performance with simple formulas. Bigger models are sample-efficient. These insights are absolutely correct and have stood the test of time. But the quantitative details have been disputed and refined. The exact exponents change with scale. Data quality matters more than assumed. Architecture innovations unlock different regimes at larger scales. And fine-tuning dynamics are more complex than pretraining alone suggests.

But the legacy of this paper is undeniable. It has over 2,500 citations in under five years. It fundamentally changed how the entire AI industry thinks about resource allocation. Before this paper, planning a multi-million dollar training run was mostly guesswork. After this paper, you could plug in your budget, get optimal N and D, and forecast your final performance. Even as we refine the exact numbers, the framework this paper established is still what everyone uses.

This paper enabled GPT-3, which in turn enabled ChatGPT, which brought LLMs to mainstream awareness. It enabled informed decisions about compute allocation that saved companies millions of dollars. And it sparked a whole subfield of research on scaling laws—not just for language models, but for vision, for multimodal models, for RL agents, for diffusion models. The idea that you can systematically study how AI systems improve with scale, and derive predictive laws, that's the lasting contribution.

---

## Key Takeaways - What You Should Remember

Alright, we're coming to the end here. If you remember nothing else from this presentation, remember these three core takeaways.

First: Scale is predictable. Performance follows power laws. You can forecast performance before spending millions of dollars on training. This was a revolutionary insight in 2020, and it remains true today even as we refine the exact numbers. You don't have to guess and pray anymore—you can calculate.

Second: Bigger models are sample-efficient. You should train large models briefly rather than small models to convergence. The optimal allocation has N growing much faster than D as you increase compute. This is counterintuitive, it goes against the old conventional wisdom, but it's been validated over and over. It's why GPT-3 worked so well, it's why modern LLMs are so large.

And third: Architecture size matters less than you think for pretraining. At a fixed parameter count, depth versus width has minimal impact on pretraining loss. For pretraining, focus on scale—getting more parameters—not on agonizing over architectural details. Now, remember the caveats: architecture matters a lot for inference efficiency, for specialized tasks, for post-training. But for pretraining performance at fixed N, scale dominates.

---

## Conclusion

So let me wrap this all up. This paper, "Scaling Laws for Neural Language Models" by Kaplan, McCandlish, and their colleagues at OpenAI, fundamentally changed how we think about training large language models. It gave us mathematical tools—simple power law formulas—to predict performance and optimize resource allocation. It enabled the creation of GPT-3 with 175 billion parameters in 2020, which then enabled GPT-4, and which ultimately led to ChatGPT bringing AI to the mainstream. It enabled the entire modern LLM ecosystem.

Now, as I've discussed, the exact quantitative predictions have been refined by later work, especially the Chinchilla paper from DeepMind in 2022. The exponents aren't set in stone—they change with scale and with your objectives. Data quality matters more than the original paper assumed. Post-training dynamics are more complex. But the core insight remains absolutely valid: scale matters, and it's predictable. That framework for thinking about AI development is the lasting legacy of this paper.

Thank you so much for your attention. I'm happy to take any questions you might have.

---

## Resources and References

For those of you who want to dive deeper into this topic, let me point you to some key resources.

First, obviously, the original paper by Kaplan et al., "Scaling Laws for Neural Language Models" from January 2020. That's arXiv 2001.08361. Read it, understand the methodology, look at the plots—they're very informative.

Second, the Chinchilla paper by Hoffmann et al. from DeepMind, "Training Compute-Optimal Large Language Models" from March 2022. That's arXiv 2203.15556. This is the key revision that updated the scaling exponents.

Third, for a more formal treatment of the algorithms and mathematics, check out "Formal Algorithms for Transformers" by Phuong and Hutter from 2022, arXiv 2207.09238. This has the pseudocode and complexity analysis spelled out in detail.

Fourth, if you want the theoretical perspective on why power laws might emerge, read "Explaining Neural Scaling Laws" by Bahri et al. from 2021, arXiv 2102.06701. This tries to derive scaling laws from first principles using statistical mechanics arguments.

And fifth, for a really accessible explanation and community perspective, check out the EleutherAI blog post on scaling laws at blog.eleuther.ai/scaling-laws/. EleutherAI is an open-source AI research collective that's done a lot of work replicating and extending these findings.

Those five resources will give you a comprehensive understanding of scaling laws, from the original empirical findings to the theoretical explanations to the practical updates we use today.

Thank you again, and I'll now take questions.
