Opening and Paper Information (1.5 minutes)

Hello, my name is Kanu and I'll be talking about Scaling Laws for Neural Language Models,  a paper that was published in January 2020, written by Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, and their colleagues at OpenAI and Johns Hopkins University.

The Problem

Picture this: it's late 2019, and you're a researcher at OpenAI with a massive compute budget—millions of dollars worth of GPU time. You're trying to answer fundamental questions: Should we build bigger models with more parameters? Or train smaller models on more data? How much compute do we need?

Before this paper, AI companies had no systematic way to answer these questions. It was all intuition and trial and error. The stakes were incredibly high—training GPT-3 cost four to twelve million dollars. Get the resource allocation wrong, and you waste millions.

What This Paper Did

OpenAI trained over 400 different language models spanning seven orders of magnitude in scale—from 768,000 parameters up to 1.5 billion parameters. They systematically varied everything: layers, dimensions, attention heads, dataset size, training duration.

Their groundbreaking discovery: language model performance follows precise mathematical power laws. Think of power laws like compound interest at a bank—each additional resource gives you diminishing returns, but those returns follow a precise mathematical curve that lets you forecast future performance.

The Three Scaling Laws (2 minutes)

The paper discovered three fundamental scaling laws.

Law 1: Performance vs Model Size

The formula is L(N) = (N_c / N)^0.076. In plain English: if you multiply your model size by ten, you get approximately a 5% reduction in loss. This relationship holds consistently from one million to one billion parameters—it's a universal pattern.

Law 2: Performance vs Data Size

The formula is L(D) = (D_c / D)^0.095. If you multiply your training data by ten, you get approximately a 6% reduction in loss.

Notice the exponent for data, 0.095, is slightly larger than for parameters, 0.076. This means data is slightly more efficient per unit. But they're not interchangeable in terms of compute cost, which brings us to the third law.

Law 3: Performance vs Compute Budget - The Most Important One

This law relates performance to your total compute budget: L(C) = (C_c / C)^0.050.

Here's the critical finding: parameters should scale as C^0.73, and data should scale as C^0.27. As you increase compute, you should grow your model size much faster than your dataset size. For every 10x increase in compute, use about 5.4 times more parameters but only 2 times more data.

This was completely backwards from conventional wisdom. Before this paper, you'd pick a model, pick a dataset, and train until convergence. The scaling laws said: flip that around! Spend your compute on bigger models trained for less time.

The Key Insight: Train Large Models, Stop Early

Let me give you a concrete example. Using the old approach: train a 340 million parameter model on 3.3 billion tokens until convergence. Cost: $7,000, loss: 3.2 nats. Using the new scaling law approach with the same budget: train a 1 billion parameter model on just 1 billion tokens and stop early. Same cost, but loss: 2.7 nats. That's 16% better performance for the same money!

This insight directly enabled GPT-3, GPT-4, Gopher, and all modern large language models.

Formal Algorithms and Pseudocode (2 minutes)

Now let me show you the practical algorithms that implement these scaling laws. I'll walk through the pseudocode very briefly.

[SHOW PSEUDOCODE SLIDE]

There are three key algorithms here:

Algorithm 1: Compute-Optimal Resource Allocation. Given a compute budget C, this tells you exactly how big your model should be and how much data to use. The key formulas are N_optimal = 0.3 × C^0.73 for model size, and D_optimal = 3.2 × C^0.27 for data size. For example, with 1 PetaFLOP-day of compute, you should train a 1.3 billion parameter model on 3.2 billion tokens.

Algorithm 2: Predict Loss from Resources. This is the flip side—given a model size N and data size D, it predicts what loss you'll achieve. It uses the bottleneck formula where loss depends on whichever resource—parameters or data—is your constraint. This is what let OpenAI predict GPT-3's performance to within 0.05 nats before spending four million dollars.

Algorithm 3: Early Stopping Criterion. This tells you when to stop training. It checks if you've used your compute budget, if you've seen enough data for your model size, or if your model is too big for your budget. The key property: training to full convergence wastes 100 times more compute than stopping at the optimal point!

These three algorithms are incredibly practical—they turned scaling from guesswork into predictable science. All three run in constant time, O(1), so you can make these calculations in milliseconds.

Question 1: Architecture Intuition (1 minute)

Before I talk about the architecture findings, let me test your intuition with a question. You have a budget of exactly 100 million parameters. Which architecture performs better?

Option A: Wide and Shallow - 6 layers, 2048 dimensions

Option B: Narrow and Deep - 48 layers, 512 dimensions

Think about it. We've all heard "deeper is better" in deep learning, right?

[PAUSE 15 SECONDS]

Here's the surprising answer: they perform almost identically! Less than 0.1 nats difference in loss—within noise levels.

The key finding: At fixed parameter count, architecture shape—wide versus narrow, shallow versus deep—matters far less than total parameter count. This shocked everyone because conventional wisdom said depth matters for learning hierarchical features. But the scaling laws showed: at fixed N, just count the parameters. Scale dominates everything.

Important caveat: This held true from 768,000 to 1.5 billion parameters. At much larger scales—100 billion and beyond—architecture innovations like sparse attention and Mixture-of-Experts do matter, but for different reasons: efficiency, longer context, training stability.

Question 2: Why Architecture Still Matters (1.5 minutes)

This leads to an obvious question: if architecture size barely matters and it's all about parameter count, why do modern AI labs spend so much time on architecture innovation?

There are several important reasons:

First, inference efficiency. While training performance depends on N, inference cost depends heavily on architecture. GPT-3 with 175 billion parameters costs a fortune to run in production. But a Mixture-of-Experts model with the same parameter count can be much cheaper because only a subset of parameters are active for each token.

Second, specialized tasks. These scaling laws are for general language modeling—next token prediction. But vision models need different architectures for images. Long-context models need sparse attention. Code models benefit from specialized tokenization. Architecture matters when you move to specialized domains.

Third, hardware constraints. At trillion-parameter scales, we're hitting real limits. We're running out of high-quality training text. Memory limitations need model parallelism. Communication bandwidth between GPUs becomes a bottleneck.

Fourth, post-training dynamics. These laws only apply to pretraining. Modern LLMs go through instruction tuning and RLHF. A smaller model with heavy RLHF can outperform a larger base model.

Bottom line: for pretraining from scratch, scale is king. But for the full lifecycle—training, fine-tuning, deployment, inference—architecture absolutely matters.

Impact and What Changed (2 minutes)

Let me show you what changed in the AI industry after this paper.

Before 2020: Train small models to convergence. BERT had 340 million parameters, trained on 3.3 billion tokens until loss stopped improving. Everything was trial and error. The motto was "train until convergence."

After 2020: Train large models and stop early. GPT-3 had 175 billion parameters—500 times bigger than BERT—trained on 300 billion tokens. Not trained to convergence—it saw each token about twice, then they stopped. We started using predictive formulas. The principle became "bigger models are sample-efficient."

The Chinchilla Correction (2022)

Here's where the story gets interesting. DeepMind revisited these scaling laws at much larger scales—10 to 100 times more compute. They found GPT-3 and Gopher were actually undertrained on data!

They trained Chinchilla with 70 billion parameters on 1.4 trillion tokens and compared it to Gopher with 280 billion parameters on 300 billion tokens—both using the same compute. Chinchilla outperformed Gopher despite being four times smaller!

DeepMind's revised laws said N and D should scale more equally—both as C^0.50, not the extreme 0.73/0.27 ratio. Why did the exponents change? Because scaling laws themselves evolve as you test at different scales. The original paper tested up to 1 PetaFLOP-day. DeepMind tested up to 100 PetaFLOP-days—two orders of magnitude larger.

This is healthy scientific iteration. The original paper was directionally correct—scale matters and it's predictable! —but the exact quantitative optimum keeps getting refined.

The LLaMA Revolution (2023)

Meta took things further with LLaMA. They trained LLaMA-13B with 13 billion parameters on one trillion tokens—way more data than the original scaling laws suggested. According to those laws, this is "overtraining."

But it worked incredibly well. LLaMA-13B matches GPT-3's 175 billion parameters on many benchmarks. It's thirteen times smaller, meaning inference is ten times cheaper. You can run it on a single GPU instead of a cluster. This made local deployment feasible and sparked the entire open-source LLM ecosystem.

The lesson: if you're optimizing purely for pretraining loss, use the scaling laws as written. But if you're optimizing for deployment cost and inference efficiency, training smaller models on more data can make a lot of sense.

Critical Analysis (1.5 minutes)

As important as this paper is, it has limitations.

First, dataset homogeneity. All experiments used English web text—WebText2. Do these laws hold for non-English languages? Low-resource languages? Specialized domains like code or mathematics? Multimodal data? We don't know.

Second, architecture specificity. They only tested decoder-only Transformers, GPT-2 style. What about encoder-decoder models like T5? Mixture-of-Experts? State-space models? The generality is unclear.

Third, post-training is ignored. The paper is entirely about pretraining. But modern LLMs have instruction tuning and RLHF. Smaller, well-aligned models can punch above their weight compared to what their pretraining loss would suggest.

Fourth, data quality matters more than assumed. The paper treats all tokens equally—a Wikipedia article counts the same as a Reddit comment. But research shows high-quality data gives 5 to 10 times better sample efficiency. Microsoft's Phi models achieve GPT-3 level performance with 100 times fewer parameters using carefully curated data.

Fifth, the data wall problem. The paper assumed unlimited training data. But the entire internet has maybe 5 trillion high-quality tokens. GPT-4 scale models need 10+ trillion tokens. We're running out of text!

Have others disputed this? Absolutely! The Chinchilla revision directly challenged the original allocation. There's debate about emergent abilities—some researchers claim they're mirages, others say they're real discontinuous jumps. LLaMA's "overtraining" worked better than predicted. And compute estimation—the "6ND" approximation—might be off by 2x depending on implementation.

The bigger picture: this paper was directionally correct and a massive step forward. But the exact exponents and optimal allocations have been refined as we've tested at larger scales. That's how science works.

Key Takeaways (1 minute)

If you remember nothing else, remember these three core takeaways:

First: Scale is predictable. Performance follows power laws. You can forecast performance before spending millions. This was revolutionary in 2020 and remains true today. You don't have to guess anymore—you can calculate.

Second: Bigger models are sample-efficient. Train large models briefly rather than small models to convergence. The optimal allocation has N growing much faster than D. This is counterintuitive but validated over and over. It's why GPT-3 worked, it's why modern LLMs are so large.

Third: Architecture size matters less than you think for pretraining. At fixed parameter count, depth versus width has minimal impact on pretraining loss. Focus on scale, not architectural details. But remember: architecture matters for inference efficiency, specialized tasks, and post-training.

Conclusion (30 seconds)

This paper fundamentally changed how we think about training large language models. It gave us mathematical tools to predict performance and optimize resource allocation. It enabled GPT-3, GPT-4, and ultimately ChatGPT bringing AI to the mainstream.

The exact predictions have been refined—the exponents aren't set in stone, data quality matters more, post-training is complex. But the core insight remains valid: scale matters, and it's predictable. That framework is the lasting legacy of this paper.

Thank you so much for your attention. I'm happy to take any questions.
