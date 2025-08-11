#!/usr/bin/env python3
"""
Delta-Hedging P&L Simulator CLI

Run discrete delta-hedging simulations and analyze P&L distributions.
"""

import argparse
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from src.quant.hedging import simulate_delta_hedge  # noqa: E402


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Delta-Hedging P&L Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Option parameters
    parser.add_argument("--S0", type=float, default=100.0, help="Initial stock price")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to expiry (years)")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument(
        "--option-type", choices=["call", "put"], default="call", help="Option type"
    )

    # Simulation parameters
    parser.add_argument(
        "--n-paths", type=int, default=10000, help="Number of simulation paths"
    )
    parser.add_argument(
        "--steps-per-year", type=int, default=252, help="Time steps per year"
    )
    parser.add_argument(
        "--rebalance-every", type=int, default=1, help="Rebalance frequency (in steps)"
    )

    # Fee parameters
    parser.add_argument(
        "--fee-bps", type=float, default=0.0, help="Proportional fee (basis points)"
    )
    parser.add_argument(
        "--fixed-fee", type=float, default=0.0, help="Fixed fee per trade"
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def plot_pnl_histogram(pnl_paths, summary, args, output_path):
    """Create and save P&L histogram."""
    plt.figure(figsize=(12, 8))

    # Main histogram
    plt.subplot(2, 2, 1)
    plt.hist(pnl_paths, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    plt.axvline(
        summary["mean"],
        color="red",
        linestyle="--",
        label=f"Mean: {summary['mean']:.4f}",
    )
    plt.axvline(
        summary["p50"],
        color="green",
        linestyle="--",
        label=f"Median: {summary['p50']:.4f}",
    )
    plt.xlabel("P&L")
    plt.ylabel("Frequency")
    plt.title("Delta-Hedge P&L Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Summary statistics
    plt.subplot(2, 2, 2)
    stats_text = f"""
    Summary Statistics:

    Mean P&L: {summary['mean']:.4f}
    Std Dev: {summary['std']:.4f}
    Sharpe (Ann): {summary['sharpe_annualized']:.4f}

    Percentiles:
    P5: {summary['p5']:.4f}
    P50: {summary['p50']:.4f}
    P95: {summary['p95']:.4f}
    """
    plt.text(
        0.1,
        0.5,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
    )
    plt.axis("off")

    # Parameters
    plt.subplot(2, 2, 3)
    param_text = f"""
    Parameters:

    S₀: {args.S0}
    K: {args.K}
    T: {args.T}
    r: {args.r}
    σ: {args.sigma}
    Type: {args.option_type}

    Paths: {args.n_paths:,}
    Steps/Year: {args.steps_per_year}
    Rebalance: Every {args.rebalance_every} steps

    Fees: {args.fee_bps} bps + ${args.fixed_fee}
    """
    plt.text(
        0.1,
        0.5,
        param_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )
    plt.axis("off")

    # P&L vs time (if we have enough data)
    plt.subplot(2, 2, 4)
    if len(pnl_paths) > 100:
        # Show sample of paths
        sample_size = min(100, len(pnl_paths))
        sample_indices = np.random.choice(len(pnl_paths), sample_size, replace=False)
        sample_pnl = pnl_paths[sample_indices]

        plt.hist(sample_pnl, bins=20, alpha=0.6, color="orange", edgecolor="black")
        plt.xlabel("P&L (Sample)")
        plt.ylabel("Frequency")
        plt.title(f"Sample of {sample_size} Paths")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(
            0.5,
            0.5,
            "Not enough data for sample plot",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved P&L histogram → {output_path}")


def main():
    """Main function."""
    args = parse_args()

    print("Delta-Hedging P&L Simulator")
    print("=" * 50)
    print(
        f"Parameters: S₀={args.S0}, K={args.K}, T={args.T}, r={args.r}, σ={args.sigma}"
    )
    print(f"Option: {args.option_type}")
    print(f"Simulation: {args.n_paths:,} paths, {args.steps_per_year} steps/year")
    print(f"Rebalancing: Every {args.rebalance_every} steps")
    print(f"Fees: {args.fee_bps} bps + ${args.fixed_fee}")
    print()

    # Run simulation
    print("Running delta-hedge simulation...")
    result = simulate_delta_hedge(
        S0=args.S0,
        K=args.K,
        T=args.T,
        r=args.r,
        sigma=args.sigma,
        option_type=args.option_type,
        n_paths=args.n_paths,
        steps_per_year=args.steps_per_year,
        rebalance_every=args.rebalance_every,
        fee_bps=args.fee_bps,
        fixed_fee=args.fixed_fee,
        seed=args.seed,
    )

    # Print summary
    print("\nResults:")
    print("-" * 30)
    print(f"Mean P&L:     {result.summary['mean']:.6f}")
    print(f"Std Dev:      {result.summary['std']:.6f}")
    print(f"Sharpe (Ann): {result.summary['sharpe_annualized']:.4f}")
    print(f"P5:           {result.summary['p5']:.6f}")
    print(f"P50:          {result.summary['p50']:.6f}")
    print(f"P95:          {result.summary['p95']:.6f}")

    print("\nNotes:")
    print("-" * 30)
    for key, value in result.notes.items():
        print(f"{key}: {value}")

    # Print summary as JSON
    print("\nSummary (JSON):")
    print("-" * 30)
    print(json.dumps(result.summary, indent=2))

    # Create plots directory and save histogram
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "hedge_pnl.png")

    plot_pnl_histogram(result.pnl_paths, result.summary, args, plot_path)


if __name__ == "__main__":
    main()
