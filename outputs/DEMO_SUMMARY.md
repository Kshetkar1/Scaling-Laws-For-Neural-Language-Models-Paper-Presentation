# Scaling Laws Code Demonstrations

This directory contains outputs from all code demonstrations.

## Text Outputs

1. **scaling_calculator_output.txt** - Interactive calculator results
   - GPT-3 loss prediction
   - Historical model comparisons
   - Scaling improvement quantification
   - Optimal allocation recommendations

2. **optimal_allocation_output.txt** - Resource allocation analysis
   - Optimal N and D for various compute budgets
   - Strategy comparison (optimal vs suboptimal)
   - Key scaling insights

## Visualizations

1. **scaling_law_N.png** - How loss scales with model size N
2. **scaling_law_D.png** - How loss scales with dataset size D
3. **training_strategies.png** - Comparing different training approaches
4. **optimal_frontier.png** - Compute-optimal frontier with historical models

## Key Findings

- **Scale is predictable**: Power laws hold across 7 orders of magnitude
- **Sample efficiency**: Larger models learn more from each token
- **Optimal allocation**: N ~ C^0.73, D ~ C^0.27
- **Validation**: Predicted GPT-3 loss within 0.3 nats of actual

## How to Use During Presentation

1. Show text outputs when explaining algorithms
2. Display visualizations when discussing scaling laws
3. Use optimal_frontier.png to show historical context
4. Reference training_strategies.png for key insight
