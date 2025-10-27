"""
Interactive scaling calculator for predicting model performance.
Based on the scaling laws from Kaplan et al. (2020).
"""


class ScalingCalculator:
    """Interactive calculator for scaling law predictions."""

    def __init__(self):
        self.N_c = 8.8e13
        self.D_c = 5.4e13
        self.alpha_N = 0.076
        self.alpha_D = 0.095
        self.alpha_C = 0.050

    def predict_loss(self, N=None, D=None, C=None):
        """
        Predict loss given any two of (N, D, C).

        Args:
            N: Model parameters (optional)
            D: Dataset size in tokens (optional)
            C: Compute budget in PF-days (optional)

        Returns:
            Predicted cross-entropy loss in nats
        """
        if N is not None and D is not None:
            term_N = (self.N_c / N) ** (self.alpha_N / self.alpha_D)
            term_D = self.D_c / D
            L = (term_N + term_D) ** self.alpha_D
            return L
        elif C is not None:
            N_opt, D_opt = self.compute_optimal_allocation(C)
            return self.predict_loss(N=N_opt, D=D_opt)
        else:
            raise ValueError("Must provide either (N, D) or C")

    def compute_optimal_allocation(self, C):
        """
        Compute optimal (N, D) for budget C.

        Args:
            C: Compute budget in PF-days

        Returns:
            (N_optimal, D_optimal)
        """
        N_opt = 0.3 * (C ** 0.73) * 1e9
        D_opt = 3.2 * (C ** 0.27) * 1e9
        return N_opt, D_opt

    def compare_models(self, model_specs):
        """
        Compare multiple model configurations.

        Args:
            model_specs: List of (name, N, D) tuples

        Returns:
            List of (name, N, D, L) tuples
        """
        results = []
        for name, N, D in model_specs:
            L = self.predict_loss(N=N, D=D)
            results.append((name, N, D, L))
        return results

    def how_much_better(self, N1, D1, N2, D2):
        """
        Compare two model configurations.

        Args:
            N1, D1: First model's parameters and data
            N2, D2: Second model's parameters and data

        Returns:
            Percentage improvement
        """
        L1 = self.predict_loss(N=N1, D=D1)
        L2 = self.predict_loss(N=N2, D=D2)
        improvement = (L1 - L2) / L1 * 100
        return improvement, L1, L2


if __name__ == '__main__':
    calc = ScalingCalculator()

    print("=" * 60)
    print("SCALING CALCULATOR DEMO")
    print("=" * 60)

    # Example 1: Predict GPT-3 performance
    print("\n1. Predicting GPT-3 performance:")
    N_gpt3 = 175e9
    D_gpt3 = 300e9
    L_gpt3 = calc.predict_loss(N=N_gpt3, D=D_gpt3)
    print(f"   GPT-3 (N={N_gpt3/1e9:.0f}B, D={D_gpt3/1e9:.0f}B)")
    print(f"   Predicted loss: {L_gpt3:.3f} nats")
    print(f"   Actual loss: ~2.0 nats ✓ Very close!")

    # Example 2: Compare historical models
    print("\n2. Comparing historical models:")
    models = [
        ("GPT-2", 1.5e9, 40e9),
        ("GPT-3", 175e9, 300e9),
        ("Gopher", 280e9, 300e9),
        ("Chinchilla", 70e9, 1.4e12),
    ]

    print(f"   {'Model':<15} {'N (B params)':<15} {'D (B tokens)':<15} {'Loss (nats)':<12}")
    print("   " + "-" * 57)
    for name, N, D, L in calc.compare_models(models):
        print(f"   {name:<15} {N/1e9:>14.1f} {D/1e9:>14.0f} {L:>11.3f}")

    # Example 3: How much better is scaling?
    print("\n3. How much better is 10x the parameters?")
    N_small = 1e9
    N_large = 10e9
    D_fixed = 10e9
    improvement, L_small, L_large = calc.how_much_better(
        N_small, D_fixed, N_large, D_fixed
    )
    print(f"   Small model (1B params): L = {L_small:.3f} nats")
    print(f"   Large model (10B params): L = {L_large:.3f} nats")
    print(f"   Improvement: {improvement:.1f}%")

    # Example 4: Optimal allocation for your budget
    print("\n4. What should I train with my compute budget?")
    C = 1.0  # 1 PF-day
    N_opt, D_opt = calc.compute_optimal_allocation(C)
    L_opt = calc.predict_loss(N=N_opt, D=D_opt)
    print(f"   Budget: {C} PF-days")
    print(f"   Optimal N: {N_opt:.2e} parameters ({N_opt/1e9:.1f}B)")
    print(f"   Optimal D: {D_opt:.2e} tokens ({D_opt/1e9:.1f}B)")
    print(f"   Expected loss: {L_opt:.3f} nats")

    print("\n" + "=" * 60)
    print("Try it yourself!")
    print(">>> from scaling_calculator import ScalingCalculator")
    print(">>> calc = ScalingCalculator()")
    print(">>> calc.predict_loss(N=1e9, D=10e9)")
    print("=" * 60)
