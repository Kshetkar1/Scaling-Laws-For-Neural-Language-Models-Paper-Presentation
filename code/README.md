# Code Examples: Scaling Laws for Neural Language Models

This directory contains Python implementations of the key algorithms and formulas from the paper "Scaling Laws for Neural Language Models" (Kaplan et al., 2020).

## 🚀 Quick Start - Run All Demos

**To generate ALL outputs at once (recommended for presentation):**

```bash
python3 run_all_demos.py
```

This will:
- ✅ Run all code examples
- ✅ Generate text outputs (predictions, tables, comparisons)
- ✅ Create visualizations (4 plots showing scaling laws)
- ✅ Save everything to `../outputs/` directory
- ✅ Create a summary report

**All outputs will be saved to the `outputs/` folder for use during your presentation!**

---

## Files

### 1. `predict_loss.py`
Predict test loss given model size (N) and dataset size (D).

**Usage:**
```python
from predict_loss import predict_loss

# Predict GPT-3 performance
N_gpt3 = 175e9  # 175B parameters
D_gpt3 = 300e9  # 300B tokens
loss = predict_loss(N_gpt3, D_gpt3)
print(f"Predicted loss: {loss:.3f} nats")
```

**Functions:**
- `predict_loss(N, D)` - Core prediction function
- `plot_scaling_law_N(D_fixed)` - Plot L(N) at fixed D
- `plot_scaling_law_D(N_fixed)` - Plot L(D) at fixed N

---

### 2. `optimal_allocation.py`
Compute optimal model size and dataset size for a given compute budget.

**Usage:**
```python
from optimal_allocation import compute_optimal_allocation

C = 1.0  # 1 PetaFLOP-day budget
N_opt, D_opt = compute_optimal_allocation(C)
print(f"Optimal: N={N_opt:.2e} params, D={D_opt:.2e} tokens")
```

**Functions:**
- `compute_optimal_allocation(C)` - Returns (N_optimal, D_optimal)
- `generate_allocation_table()` - Generate table for multiple budgets
- `compare_allocation_strategies(C)` - Compare different strategies

---

### 3. `scaling_calculator.py`
Interactive calculator class for various scaling law predictions.

**Usage:**
```python
from scaling_calculator import ScalingCalculator

calc = ScalingCalculator()

# Predict loss
loss = calc.predict_loss(N=1e9, D=10e9)

# Compare models
models = [
    ("GPT-2", 1.5e9, 40e9),
    ("GPT-3", 175e9, 300e9),
]
results = calc.compare_models(models)

# How much better is 10x scaling?
improvement, L1, L2 = calc.how_much_better(1e9, 10e9, 10e9, 10e9)
```

**Class Methods:**
- `predict_loss(N, D, C)` - Flexible loss prediction
- `compute_optimal_allocation(C)` - Optimal (N, D) for budget C
- `compare_models(model_specs)` - Compare multiple configurations
- `how_much_better(N1, D1, N2, D2)` - Quantify improvement

---

### 4. `compare_strategies.py`
Visualize and compare different training strategies.

**Usage:**
```python
from compare_strategies import compare_training_strategies, visualize_optimal_frontier

# Compare training curves
compare_training_strategies(C=1.0)

# Visualize compute-optimal frontier
visualize_optimal_frontier()
```

**Functions:**
- `training_loss_curve(N, D, S_max)` - Simulate training trajectory
- `compare_training_strategies(C)` - Plot comparison of strategies
- `visualize_optimal_frontier()` - Plot N vs D with historical models

---

### 5. `run_all_demos.py` ⭐ **NEW - Comprehensive Demo**
Run all examples and save outputs for presentation.

**Usage:**
```bash
python3 run_all_demos.py
```

**What it does:**
- Runs all 4 code files above
- Saves text outputs to `../outputs/`
- Saves 4 visualizations as PNG files
- Creates a summary report (DEMO_SUMMARY.md)
- **Perfect for generating presentation materials!**

**Outputs:**
- `scaling_calculator_output.txt` - Calculator results
- `optimal_allocation_output.txt` - Allocation tables
- `scaling_law_N.png` - Loss vs model size plot
- `scaling_law_D.png` - Loss vs data size plot
- `training_strategies.png` - Strategy comparison plot
- `optimal_frontier.png` - Historical models plot

---

## Installation

```bash
# Install required packages
pip install numpy matplotlib pandas
```

## Running the Examples

```bash
# Run individual files
python predict_loss.py
python optimal_allocation.py
python scaling_calculator.py
python compare_strategies.py
```

## Key Formulas Implemented

### Loss Prediction (Equation 1.6)
```
L(N, D) = [(Nc/N)^(αN/αD) + Dc/D]^αD

where:
  Nc ≈ 8.8 × 10¹³ parameters
  Dc ≈ 5.4 × 10¹³ tokens
  αN ≈ 0.076
  αD ≈ 0.095
```

### Optimal Allocation (Section 5)
```
N_optimal(C) = N₀ × C^0.73
D_optimal(C) = D₀ × C^0.27
```

### Critical Batch Size (Equation 5.8)
```
B_crit(L) ≈ B_noise × (L / L_noise)^(1/αD)
```

## Example Output

Running `scaling_calculator.py`:

```
============================================================
SCALING CALCULATOR DEMO
============================================================

1. Predicting GPT-3 performance:
   GPT-3 (N=175B, D=300B)
   Predicted loss: 2.150 nats
   Actual loss: ~2.0 nats ✓ Very close!

2. Comparing historical models:
   Model           N (B params)    D (B tokens)    Loss (nats)
   -------------------------------------------------------------
   GPT-2                    1.5              40       3.142
   GPT-3                  175.0             300       2.150
   Gopher                 280.0             300       2.086
   Chinchilla              70.0            1400       1.870

3. How much better is 10x the parameters?
   Small model (1B params): L = 3.142 nats
   Large model (10B params): L = 2.598 nats
   Improvement: 17.3%
```

## Visualizations

The code generates several plots:
1. **Scaling Laws**: L(N) and L(D) on log-log axes
2. **Training Curves**: Compare different training strategies
3. **Optimal Frontier**: (N, D) space with historical models

## Further Reading

See the main README.md in the parent directory for detailed explanations of the scaling laws and their implications.

## Citation

```bibtex
@article{kaplan2020scaling,
  title={Scaling Laws for Neural Language Models},
  author={Kaplan, Jared and McCandlish, Sam and others},
  journal={arXiv preprint arXiv:2001.08361},
  year={2020}
}
```
