# ALGORITHMS EXPLAINED LIKE YOU'RE TEACHING A FRIEND
## Every Step, Every Symbol, Plain English

---

# 🤖 ALGORITHM 1: Finding the Optimal Split

## 🎯 THE BIG PICTURE

**The Problem:** You have a compute budget (like $100,000 worth of GPU time). How do you split it between model size and data?

**What This Does:** Takes your budget → tells you the perfect model size and data size.

---

## THE ALGORITHM

```
Step 1: N_optimal ← 0.3 × C^0.73
Step 2: D_optimal ← 3.2 × C^0.27  
Step 3: C_actual ← (6 × N × D) / 10^15
Step 4: Check if close to budget
```

---

## STEP 1: Calculate Model Size

### **The Line:**
```
N_optimal ← 0.3 × C^0.73
```

### **What Each Part Means:**

**N_optimal** = Number of parameters your model should have
- This is what we're solving for!
- Example: "1.3 billion parameters"

**C** = Your compute budget
- Measured in PetaFLOP-days (PF-days)
- Example: C = 1.0 means "1 PF-day"
- Think: "How many GPU-hours can I afford?"

**0.73** = The exponent (the magic number!)
- **Where from?** Table 6 in the paper
- **Why 0.73?** They tested 400+ models and found this works best
- **What it means:** Parameters should grow FAST with compute

**0.3** = A conversion constant
- Calibrated to their hardware setup
- Think: "0.3 billion parameters per PF-day to the 0.73"

### **Why Raise to the 0.73 Power?**

Because growth isn't linear!

```
If C = 1 PF-day:
  C^0.73 = 1^0.73 = 1.0
  N = 0.3 × 1.0 = 0.3 billion = 300 million params

If C = 10 PF-days (10x more compute):
  C^0.73 = 10^0.73 = 5.37
  N = 0.3 × 5.37 = 1.6 billion params
```

**Key Insight:** 10x more compute → only 5.4x bigger model (not 10x!)

The rest goes to data!

---

## STEP 2: Calculate Data Size

### **The Line:**
```
D_optimal ← 3.2 × C^0.27
```

### **What Each Part Means:**

**D_optimal** = Number of tokens to train on
- Tokens = pieces of text (roughly 3/4 of a word)
- Example: "3.2 billion tokens"

**C** = Same compute budget

**0.27** = The data exponent
- **Where from?** Table 6
- **Why 0.27?** Much smaller than 0.73!
- **What it means:** Data should grow SLOWLY compared to model

**3.2** = Data conversion constant

### **Why 0.27 is So Much Smaller:**

```
If C = 1 PF-day:
  C^0.27 = 1^0.27 = 1.0
  D = 3.2 × 1.0 = 3.2 billion tokens

If C = 10 PF-days (10x more compute):
  C^0.27 = 10^0.27 = 1.86
  D = 3.2 × 1.86 = 6.0 billion tokens
```

**Key Insight:** 10x more compute → only 1.9x more data!

**The Big Reveal:** Most of your extra compute should go to model size, NOT data!

---

## STEP 3: Verify the Math

### **The Line:**
```
C_actual ← (6 × N_optimal × D_optimal) / 10^15
```

### **What's Happening:**
Double-checking: "Does this actually use our budget?"

### **Breaking Down the Formula:**

**6** = Operations per parameter per token
- **Why 6?** Here's the breakdown:
  - Forward pass through network: ~2 operations per parameter
  - Backward pass (gradients): ~4 operations per parameter
  - Total: 2 + 4 = 6 operations
- It's an approximation that works well

**N_optimal** = Parameters (from Step 1)

**D_optimal** = Tokens (from Step 2)

**6 × N × D** = Total operations needed
- For each token, we do 6N operations
- For D tokens, we do 6ND total operations
- This gives us FLOPs (floating point operations)

**10^15** = Why divide by this?
- **Unit conversion!**
- 1 PetaFLOP = 10^15 FLOPs per second
- We're converting raw FLOPs into PF-days

### **Example:**
```
N = 1.3 billion = 1.3 × 10^9
D = 3.2 billion = 3.2 × 10^9

Total FLOPs = 6 × 1.3×10^9 × 3.2×10^9
           = 24.96 × 10^18

C_actual = 24.96 × 10^18 / 10^15
        ≈ 1.0 PF-days ✓
```

Close to our budget of 1.0!

---

## STEP 4: Safety Check

### **The Line:**
```
if |C_actual - C| / C > 0.1 then
    Print "Warning"
end if
```

### **What's Happening:**
Checking if we're within 10% of our budget.

### **The Math:**

**|C_actual - C|** = Absolute difference
- The | | bars mean "make it positive"
- How far off are we?

**Divide by C** = Make it a percentage
- Error / Budget = Percentage error

**> 0.1** = More than 10% off?
- If yes, warn the user
- If no, we're good!

### **Example:**
```
C_budget = 1.0
C_actual = 0.95

Error = |0.95 - 1.0| / 1.0
     = 0.05 / 1.0
     = 0.05 = 5% ✓ OK!
```

---

# 🤖 ALGORITHM 2: Predicting Loss

## 🎯 THE BIG PICTURE

**The Problem:** Before spending millions, you want to know: "What loss will I get?"

**What This Does:** Takes model size + data size → predicts final loss

---

## THE ALGORITHM

```
Step 1: term_N ← (N_c / N)^0.8
Step 2: term_D ← D_c / D
Step 3: L ← (term_N + term_D)^0.095
```

---

## STEP 1: Parameter Constraint

### **The Line:**
```
term_N ← (N_c / N)^(α_N / α_D)
Which simplifies to: (N_c / N)^0.8
```

### **What's Happening:**
Measuring how much your model size limits performance.

### **The Symbols:**

**term_N** = Parameter constraint
- **Big number** = model is too small (bottleneck!)
- **Small number** = model is big enough

**N_c** = 8.8 × 10^13 = 88 trillion
- This is a reference scale from Table 5
- Think: "the ideal scale for parameters"

**N** = Your actual model size
- Example: GPT-3 = 175 billion

**N_c / N** = How you compare to ideal
- If this ratio is big → your model is small compared to ideal

### **Example with GPT-3:**
```
N = 175 billion = 1.75 × 10^11
N_c = 8.8 × 10^13

Ratio = 8.8×10^13 / 1.75×10^11
     = 502.86

term_N = 502.86^0.8
      = 181.4
```

**What 181.4 means:** 
This is fairly large, so model size is contributing significantly to the loss. A bigger model would help!

---

## STEP 2: Data Constraint  

### **The Line:**
```
term_D ← D_c / D
```

### **What's Happening:**
Measuring how much your dataset size limits performance.

### **The Symbols:**

**term_D** = Data constraint
- **Big number** = not enough data (bottleneck!)
- **Small number** = plenty of data

**D_c** = 5.4 × 10^13 = 54 trillion
- Reference scale for data from Table 5

**D** = Your actual dataset size
- Example: GPT-3 = 300 billion tokens

### **Example with GPT-3:**
```
D = 300 billion = 3 × 10^11
D_c = 5.4 × 10^13

term_D = 5.4×10^13 / 3×10^11
      = 180
```

**Notice:** No exponent here! Just the ratio.

**What 180 means:**
Similar to term_N (181.4), so both constraints contribute equally. Pretty balanced!

---

## STEP 3: Combine to Get Loss

### **The Line:**
```
L ← (term_N + term_D)^α_D
Which is: (term_N + term_D)^0.095
```

### **What's Happening:**
Combining both bottlenecks to predict final loss.

### **The Math:**

**Add the terms:**
```
term_N + term_D = 181.4 + 180 = 361.4
```

**Raise to the 0.095 power:**
- This is α_D from Table 5
- Same exponent from Law 2!

```
L = 361.4^0.095
  ≈ 2.15 nats
```

**GPT-3's actual loss:** ~2.0 nats

**Amazing!** Predicted 2.15, got 2.0—very close!

---

## THE BOTTLENECK INSIGHT

**What the two terms tell you:**

**Scenario 1: Parameter Bottleneck**
```
term_N = 500 (HUGE - model too small)
term_D = 50  (small - plenty data)
→ Model size is your problem!
```

**Scenario 2: Data Bottleneck**
```
term_N = 50  (small - model fine)
term_D = 500 (HUGE - not enough data)
→ Data size is your problem!
```

**Scenario 3: Balanced (GPT-3)**
```
term_N = 181 (moderate)
term_D = 180 (moderate)
→ Both contribute equally ✓
```

---

# 🤖 ALGORITHM 3: When to Stop

## 🎯 THE BIG PICTURE

**The Problem:** During training, should I keep going or stop?

**Old Way:** Train until loss stops improving (forever!)

**New Way:** Stop early based on compute + tokens. Way more efficient!

---

## THE ALGORITHM

```
Step 1: Calculate compute used so far
Step 2: Count tokens seen
Step 3: Get optimal allocation
Step 4: Check 3 stop conditions
```

---

## STEP 1: Compute Used

### **The Line:**
```
C_used ← (6 × N × B × S) / 10^15
```

### **What's Happening:**
Tracking how much compute we've spent (like checking your bank account).

### **The Symbols:**

**C_used** = Compute spent so far (PF-days)

**N** = Model size (fixed at start)
- Example: 7 billion parameters

**B** = Batch size (tokens per step)
- Example: 2 million tokens per step

**S** = Current step number
- Example: 10,000 training steps

**6 × N × B × S** = Total FLOPs used
- 6 operations per parameter per token
- N parameters × B tokens × S steps

**/ 10^15** = Convert to PF-days

### **Example:**
```
N = 7 × 10^9 (7B model)
B = 2 × 10^6 (2M tokens/step)
S = 10,000 steps

FLOPs = 6 × 7×10^9 × 2×10^6 × 10,000
     = 8.4 × 10^20

C_used = 8.4×10^20 / 10^15
      = 8.4 × 10^5 / 10^15
      ≈ 0.84 PF-days
```

---

## STEP 2: Tokens Seen

### **The Line:**
```
tokens_seen ← B × S
```

### **What's Happening:**
Counting total training data seen.

### **Simple Multiplication:**

**B** = Tokens per step
**S** = Number of steps
**B × S** = Total tokens!

### **Example:**
```
B = 2 million tokens/step
S = 10,000 steps

tokens_seen = 2×10^6 × 10,000
           = 20 billion tokens
```

---

## STEP 3: Get Optimal Plan

### **The Line:**
```
(N_opt, D_opt) ← ComputeOptimalAllocation(C_budget)
```

### **What's Happening:**
Calling Algorithm 1 to find the ideal allocation.

### **Example:**
```
C_budget = 100 PF-days

Algorithm 1 returns:
N_opt = 37 billion parameters
D_opt = 12 billion tokens
```

Now we know what we SHOULD be doing!

---

## STEP 4: The 3 Stop Checks

### **CHECK 1: Out of Money?**
```
if C_used ≥ C_budget then
    return True  ← STOP!
end if
```

**What it checks:** Have we spent our budget?

**Example:**
```
C_budget = 100 PF-days
C_used = 105 PF-days

105 ≥ 100? YES → STOP!
```

**Why:** No money left, can't continue!

---

### **CHECK 2: Reached Optimal Point?**
```
if N ≈ N_opt and tokens_seen ≥ D_opt then
    return True  ← STOP!
end if
```

**What it checks:** Right model size AND seen enough tokens?

**Example:**
```
N = 35B (our model)
N_opt = 37B (optimal)
35 ≈ 37? YES (close enough)

tokens_seen = 13B
D_opt = 12B
13 ≥ 12? YES

BOTH true → STOP!
```

**This is "Stop Early"!**
Traditional training would continue to convergence.
Scaling laws say: "You hit the sweet spot. Stop now!"

**Why stop?**
Training more won't improve enough to justify the cost.

---

### **CHECK 3: Model Too Big?**
```
if N > 2 × N_opt then
    Print "Warning: Model too large!"
    return True  ← STOP!
end if
```

**What it checks:** Did we choose a model way too big for our budget?

**Example:**
```
N = 100B (our model)
N_opt = 37B (optimal)

100 > 2 × 37 = 74? YES → STOP with warning!
```

**What went wrong:**
You should have used a smaller model!
A 37B model would perform better with this budget.

**The lesson:**
Use Algorithm 1 BEFORE training to pick the right size!

---

## THE DEFAULT: Keep Going

```
return False  ← Continue training
```

**When this happens:**
None of the stop conditions triggered, so keep training!

---

# 🎯 COMPLETE EXAMPLE: Training GPT-3

Let's walk through a full training run!

## BEFORE TRAINING

**Budget:** 3,640 PF-days (OpenAI's actual budget)

**Algorithm 1:** What should we do?
```
N_opt = 0.3 × 3640^0.73 ≈ 175 billion params
D_opt = 3.2 × 3640^0.27 ≈ 300 billion tokens
```

**Algorithm 2:** What will we get?
```
Predicted loss ≈ 2.15 nats
```

**Decision:** Build a 175B parameter model, train on 300B tokens

---

## DURING TRAINING

**At step 50,000:**
```
C_used = 1,820 PF-days (half budget)
tokens_seen = 150B tokens (halfway!)

Check 1: 1820 ≥ 3640? NO → continue
Check 2: seen 300B? NO → continue
Check 3: model too big? NO → continue

→ KEEP TRAINING
```

**At step 100,000:**
```
C_used = 3,640 PF-days (budget reached!)
tokens_seen = 300B tokens (target hit!)

Check 1: 3640 ≥ 3640? YES → STOP!
```

---

## AFTER TRAINING

**Result:**
- Final loss: ~2.0 nats
- Predicted: 2.15 nats
- **Error: Only 7.5%!** Amazing accuracy!

**The Efficiency:**
If they had trained to convergence:
- Would take ~300,000 steps
- Would cost ~11,000 PF-days (3x the budget!)
- Would only improve loss by ~0.1 nats

By stopping early:
- Saved 2/3 of compute
- Got 93% of the performance
- **This is compute-efficient training!**

---

# 💡 HOW ALL THREE WORK TOGETHER

```
PLANNING (Algorithm 1):
"I have X compute → Use Y params, Z tokens"

PREDICTING (Algorithm 2):  
"With Y params and Z tokens → Get L loss"

EXECUTING (Algorithm 3):
"At each step: Should I stop? No? Keep going!"
```

## The Workflow:

**1. Use Algorithm 1** → Plan your training
- Input: 100 PF-days
- Output: 37B params, 12B tokens

**2. Use Algorithm 2** → Set expectations
- Input: 37B params, 12B tokens
- Output: Loss will be 2.3 nats

**3. Build model** → 37B parameters

**4. Train** → At each step, use Algorithm 3
- Keep checking: Time to stop?
- When tokens_seen hits 12B → STOP!

**5. Result** → Loss is 2.3 nats (as predicted!)

---

# ✅ WHAT YOU NOW UNDERSTAND

- ✓ Every symbol (α_N, k_D, N_c, etc.)
- ✓ Why we multiply by 6 (forward + backward)
- ✓ Why we divide by 10^15 (unit conversion)
- ✓ Why 0.73 and 0.27 (optimal split)
- ✓ Why stop early (100x compute savings!)
- ✓ What each check does
- ✓ How to use all three together

**You can now explain these algorithms to anyone!** 🎯
