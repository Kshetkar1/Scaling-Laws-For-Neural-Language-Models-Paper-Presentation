"""
Predict test loss given model size (N) and dataset size (D).
Based on the scaling laws from Kaplan et al. (2020).
"""

import numpy as np
import matplotlib.pyplot as plt


def predict_loss(N, D):
    """
    Predict test loss given model size and data size.

    Args:
        N: Number of model parameters
        D: Dataset size in tokens

    Returns:
        Predicted cross-entropy loss in nats
    """
    N_c = 8.8e13  # Critical parameter count
    D_c = 5.4e13  # Critical dataset size
    alpha_N = 0.076
    alpha_D = 0.095

    term_N = (N_c / N) ** (alpha_N / alpha_D)
    term_D = D_c / D
    L = (term_N + term_D) ** alpha_D
    return L


def plot_scaling_law_N(D_fixed=10e9, save_path=None):
    """Plot how loss scales with model size N at fixed data size D."""
    N_values = np.logspace(6, 11, 50)  # 1M to 100B parameters
    losses = [predict_loss(N, D_fixed) for N in N_values]

    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, losses, linewidth=2, color='#2E86AB')
    plt.xlabel('Model Size N (parameters)', fontsize=14)
    plt.ylabel('Test Loss (nats)', fontsize=14)
    plt.title(f'Scaling Law: L(N) at D = {D_fixed/1e9:.0f}B tokens', fontsize=16)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_scaling_law_D(N_fixed=1e9, save_path=None):
    """Plot how loss scales with dataset size D at fixed model size N."""
    D_values = np.logspace(6, 12, 50)  # 1M to 1T tokens
    losses = [predict_loss(N_fixed, D) for D in D_values]

    plt.figure(figsize=(10, 6))
    plt.loglog(D_values, losses, linewidth=2, color='#A23B72')
    plt.xlabel('Dataset Size D (tokens)', fontsize=14)
    plt.ylabel('Test Loss (nats)', fontsize=14)
    plt.title(f'Scaling Law: L(D) at N = {N_fixed/1e9:.0f}B params', fontsize=16)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Example: Predict GPT-3 performance
    N_gpt3 = 175e9  # 175 billion parameters
    D_gpt3 = 300e9  # 300 billion tokens

    predicted_loss = predict_loss(N_gpt3, D_gpt3)
    print(f"Predicted GPT-3 loss: {predicted_loss:.3f} nats")
    print(f"Actual GPT-3 loss: ~2.0 nats (very close!)")

    # Plot scaling laws
    print("\nGenerating scaling law plots...")
    plot_scaling_law_N(D_fixed=10e9)
    plot_scaling_law_D(N_fixed=1e9)
