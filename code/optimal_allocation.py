"""
Compute optimal model size and dataset size for a given compute budget.
Based on the scaling laws from Kaplan et al. (2020).
"""

import numpy as np
import pandas as pd
from predict_loss import predict_loss


def compute_optimal_allocation(C):
    """
    Given compute budget C (in PF-days), return optimal (N, D).

    Args:
        C: Compute budget in PetaFLOP-days

    Returns:
        (N_optimal, D_optimal): Optimal model size and dataset size
    """
    # Optimal allocation exponents (from paper's Appendix B)
    a = 0.73  # N scales as C^0.73
    b = 0.27  # D scales as C^0.27

    # Coefficients (hardware-dependent, approximate)
    N_coeff = 0.3
    D_coeff = 3.2

    N_optimal = N_coeff * (C ** a) * 1e9  # Convert to raw parameters
    D_optimal = D_coeff * (C ** b) * 1e9  # Convert to raw tokens

    return N_optimal, D_optimal


def generate_allocation_table(C_values=None):
    """Generate a table of optimal allocations for different compute budgets."""
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # PF-days

    results = []

    for C in C_values:
        N, D = compute_optimal_allocation(C)
        L = predict_loss(N, D)
        results.append({
            'Compute (PF-days)': C,
            'Optimal N (params)': f'{N:.2e}',
            'Optimal D (tokens)': f'{D:.2e}',
            'Predicted Loss (nats)': f'{L:.2f}'
        })

    df = pd.DataFrame(results)
    return df


def compare_allocation_strategies(C):
    """
    Compare different allocation strategies for a fixed compute budget.

    Args:
        C: Compute budget in PF-days
    """
    # Strategy 1: Optimal allocation
    N_opt, D_opt = compute_optimal_allocation(C)
    L_opt = predict_loss(N_opt, D_opt)

    # Strategy 2: Over-allocate to N (too large model, not enough data)
    N_large = N_opt * 3
    D_small = D_opt / 3
    L_large = predict_loss(N_large, D_small)

    # Strategy 3: Over-allocate to D (too small model, too much data)
    N_small = N_opt / 3
    D_large = D_opt * 3
    L_small = predict_loss(N_small, D_large)

    results = {
        'Strategy': ['Optimal', 'Over-allocate N', 'Over-allocate D'],
        'N (params)': [f'{N_opt:.2e}', f'{N_large:.2e}', f'{N_small:.2e}'],
        'D (tokens)': [f'{D_opt:.2e}', f'{D_small:.2e}', f'{D_large:.2e}'],
        'Loss (nats)': [f'{L_opt:.3f}', f'{L_large:.3f}', f'{L_small:.3f}']
    }

    return pd.DataFrame(results)


if __name__ == '__main__':
    print("=" * 60)
    print("OPTIMAL ALLOCATION TABLE")
    print("=" * 60)
    df = generate_allocation_table()
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("COMPARING STRATEGIES (C = 1.0 PF-days)")
    print("=" * 60)
    comparison = compare_allocation_strategies(1.0)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("1. As compute increases, N grows MUCH faster than D (N ~ C^0.73 vs D ~ C^0.27)")
    print("2. Over-allocating to N hurts less than over-allocating to D")
    print("3. The optimal frontier balances N and D to minimize loss")
