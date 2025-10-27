"""
Compare different training strategies with the same compute budget.
Visualizes training curves and final performance.
"""

import matplotlib.pyplot as plt
import numpy as np
from predict_loss import predict_loss


def training_loss_curve(N, D, S_max, B=2**19):
    """
    Simulate loss curve during training.

    Args:
        N: Model parameters
        D: Dataset size (tokens)
        S_max: Maximum training steps
        B: Batch size (tokens per step)

    Returns:
        (steps, losses): Arrays of training steps and corresponding losses
    """
    steps = np.linspace(1, S_max, 100)
    tokens_seen = steps * B

    # Effective data size grows with steps (up to total D)
    D_eff = np.minimum(tokens_seen, D)
    losses = [predict_loss(N, d) for d in D_eff]
    return steps, losses


def compare_training_strategies(C=1.0, save_path=None):
    """
    Compare three different training strategies with the same compute budget.

    Args:
        C: Compute budget in PF-days (for illustration purposes)
        save_path: Optional path to save the figure
    """
    # Three strategies with roughly the same compute
    strategies = [
        ('Small model, long training', 100e6, 10e9, 50000),
        ('Medium model, medium training', 500e6, 5e9, 20000),
        ('Large model, short training', 2e9, 2e9, 5000),
    ]

    plt.figure(figsize=(12, 7))
    colors = ['#E63946', '#F77F00', '#06A77D']

    for (name, N, D, S_max), color in zip(strategies, colors):
        steps, losses = training_loss_curve(N, D, S_max)
        final_loss = losses[-1]
        plt.plot(steps, losses,
                label=f'{name}\nN={N/1e9:.1f}B, D={D/1e9:.0f}B → L={final_loss:.2f}',
                linewidth=2.5, color=color)

    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Test Loss (nats)', fontsize=14)
    plt.title('Comparing Training Strategies (Fixed Compute Budget)', fontsize=16)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)

    # Add annotation
    plt.text(0.02, 0.98,
             'Key Insight: Larger models trained briefly\nbeat smaller models trained longer!',
             transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_optimal_frontier(save_path=None):
    """
    Visualize the compute-optimal frontier in (N, D) space.
    """
    # Generate points along different compute budgets
    C_values = np.logspace(-3, 2, 20)  # 0.001 to 100 PF-days
    N_optimal = []
    D_optimal = []

    for C in C_values:
        N = 0.3 * (C ** 0.73) * 1e9
        D = 3.2 * (C ** 0.27) * 1e9
        N_optimal.append(N)
        D_optimal.append(D)

    plt.figure(figsize=(10, 8))
    plt.loglog(D_optimal, N_optimal, linewidth=3, color='#2E86AB',
               label='Optimal frontier (N ~ D^0.74)')

    # Plot historical models
    models = {
        'BERT': (3.3e9, 340e6, 'undertrained'),
        'GPT-2': (40e9, 1.5e9, 'undertrained'),
        'GPT-3': (300e9, 175e9, 'near optimal'),
        'Gopher': (300e9, 280e9, 'undertrained'),
        'Chinchilla': (1.4e12, 70e9, 'near optimal'),
    }

    colors_map = {'undertrained': '#E63946', 'near optimal': '#06A77D'}

    for name, (D, N, status) in models.items():
        plt.scatter(D, N, s=200, color=colors_map[status],
                   edgecolors='black', linewidth=1.5, zorder=5)
        plt.text(D * 1.3, N, name, fontsize=11, verticalalignment='center')

    plt.xlabel('Dataset Size D (tokens)', fontsize=14)
    plt.ylabel('Model Size N (parameters)', fontsize=14)
    plt.title('Compute-Optimal Frontier: Historical Models', fontsize=16)
    plt.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='#2E86AB', lw=3, label='Optimal frontier'),
        Patch(facecolor='#06A77D', edgecolor='black', label='Near optimal'),
        Patch(facecolor='#E63946', edgecolor='black', label='Undertrained (need more data)')
    ]
    plt.legend(handles=legend_elements, fontsize=11, loc='upper left')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("Generating comparison visualizations...")
    print("\n1. Comparing training strategies...")
    compare_training_strategies(C=1.0)

    print("\n2. Visualizing the compute-optimal frontier...")
    visualize_optimal_frontier()

    print("\nDone! Close the plot windows to exit.")
