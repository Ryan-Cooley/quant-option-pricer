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
        "--rebalance-every",
        type=str,
        default="1,2,5,10",
        help=(
            "Rebalance frequency(s) (in steps). Use comma-separated list for "
            "multiple values, e.g., '1,2,5,10'"
        ),
    )

    # Fee parameters
    parser.add_argument(
        "--fee-bps", type=float, default=0.0, help="Proportional fee (basis points)"
    )
    parser.add_argument(
        "--fixed-fee", type=float, default=0.0, help="Fixed fee per trade"
    )

    # Output parameters
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output CSV path for metrics (e.g., plots/hedge_metrics.csv)",
    )
    parser.add_argument(
        "--units",
        choices=["s0", "premium"],
        default="s0",
        help="Denominator for TE/Cost: 's0' (default) or 'premium'",
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def plot_pnl_histogram(pnl_paths, summary, args, all_results, output_path):
    """Create and save P&L histogram with metrics table."""
    plt.figure(figsize=(12, 8))

    # Get first result for metadata
    first_result = all_results[list(all_results.keys())[0]] if all_results else None

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

    # Add metrics table on the right side
    if all_results:
        # Prepare table data for all results
        table_data = []
        for dt in sorted(all_results.keys()):
            result = all_results[dt]
            if "metrics" in result.notes and dt in result.notes["metrics"]:
                metric_data = result.notes["metrics"][dt]
                te_bps = metric_data["te_bps"]
                cost_bps = metric_data["cost_bps"]
                table_data.append([f"Δt={dt}", f"{te_bps:.1f}", f"{cost_bps:.1f}"])

        if table_data:
            # Create table with dynamic labels based on units
            unit_label = "S0" if args.units == "s0" else "premium"
            table = plt.table(
                cellText=table_data,
                colLabels=[
                    "Δt (steps)",
                    f"TE (bps of {unit_label})",
                    f"Cost (bps of {unit_label})",
                ],
                cellLoc="center",
                loc="upper left",
                bbox=[1.02, 0.0, 0.35, 0.4],
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # Right-align numbers
            for i in range(len(table_data)):
                table[(i + 1, 1)]._text.set_horizontalalignment("right")
                table[(i + 1, 2)]._text.set_horizontalalignment("right")

            # Style headers
            for j in range(3):
                table[(0, j)].set_facecolor("#E6E6E6")
                table[(0, j)]._text.set_weight("bold")

    # Add metadata stamp
    if first_result:
        meta_string = (
            f"seed={args.seed}, Δt={list(all_results.keys())}, "
            f"n_paths={args.n_paths}, steps={first_result.notes['steps']}, "
            f"fee_bps={args.fee_bps}, units={args.units}"
        )
        plt.gcf().text(
            0.01,
            0.01,
            meta_string,
            fontsize=7,
            alpha=0.7,
            transform=plt.gcf().transFigure,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved P&L histogram →", output_path)


def plot_frontier_scatter(all_results, args, plots_dir):
    """Create and save frontier scatter plot (Cost vs TE)."""
    plt.figure(figsize=(8, 6))

    # Extract data for scatter plot
    te_values = []
    cost_values = []
    dt_labels = []

    for dt in sorted(all_results.keys()):
        result = all_results[dt]
        if "metrics" in result.notes and dt in result.notes["metrics"]:
            metrics = result.notes["metrics"][dt]
            te_values.append(metrics["te_bps"])
            cost_values.append(metrics["cost_bps"])
            dt_labels.append(str(dt))

    # Create scatter plot
    plt.scatter(te_values, cost_values, s=100, alpha=0.7, edgecolors="black")

    # Add dotted line connecting the points
    plt.plot(te_values, cost_values, "--", alpha=0.5, color="gray", linewidth=1)

    # Annotate points with Δt values
    for i, (te, cost, dt) in enumerate(zip(te_values, cost_values, dt_labels)):
        plt.annotate(
            f"Δt={dt}",
            (te, cost),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    # Labels and title with dynamic units
    unit_label = "S0" if args.units == "s0" else "premium"
    plt.xlabel(f"Tracking Error (bps of {unit_label})")
    plt.ylabel(f"Cost (bps of {unit_label})")
    plt.title("Hedging Cost–Error Frontier")

    # Grid
    plt.grid(True, alpha=0.3)

    # Add metadata stamp
    meta_string = (
        f"seed={args.seed}, Δt={list(all_results.keys())}, "
        f"n_paths={args.n_paths}, fee_bps={args.fee_bps}, units={args.units}"
    )
    plt.gcf().text(
        0.01, 0.01, meta_string, fontsize=7, alpha=0.7, transform=plt.gcf().transFigure
    )

    # Layout and save
    plt.tight_layout()
    frontier_path = os.path.join(plots_dir, "hedge_frontier.png")
    plt.savefig(frontier_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved frontier →", frontier_path)


def main():
    """Main function."""
    args = parse_args()

    # Parse rebalance frequencies
    try:
        rebalance_freqs = [int(x.strip()) for x in args.rebalance_every.split(",")]
    except ValueError:
        print(
            "Error: --rebalance-every must be comma-separated integers "
            "(e.g., '1,2,5,10')"
        )
        sys.exit(1)

    print("Delta-Hedging P&L Simulator")
    print("=" * 50)
    print(
        f"Parameters: S₀={args.S0}, K={args.K}, T={args.T}, r={args.r}, "
        f"σ={args.sigma}"
    )
    print(f"Option: {args.option_type}")
    print(f"Simulation: {args.n_paths:,} paths, {args.steps_per_year} steps/year")
    print(f"Rebalancing: Every {', '.join(map(str, rebalance_freqs))} steps")
    print(f"Fees: {args.fee_bps} bps + ${args.fixed_fee}")
    print()

    # Run simulations for each rebalance frequency
    print("Running delta-hedge simulations...")
    all_results = {}
    all_metrics = {}

    for rebalance_every in rebalance_freqs:
        result = simulate_delta_hedge(
            S0=args.S0,
            K=args.K,
            T=args.T,
            r=args.r,
            sigma=args.sigma,
            option_type=args.option_type,
            n_paths=args.n_paths,
            steps_per_year=args.steps_per_year,
            rebalance_every=rebalance_every,
            fee_bps=args.fee_bps,
            fixed_fee=args.fixed_fee,
            seed=args.seed,
            units=args.units,
        )
        all_results[rebalance_every] = result
        all_metrics[rebalance_every] = result.notes["metrics"][rebalance_every]

    # Print summary for first result (use as representative)
    first_result = all_results[rebalance_freqs[0]]
    print("\nResults (representative):")
    print("-" * 30)
    print(f"Mean P&L:     {first_result.summary['mean']:.6f}")
    print(f"Std Dev:      {first_result.summary['std']:.6f}")
    print(f"Sharpe (Ann): {first_result.summary['sharpe_annualized']:.4f}")
    print(f"P5:           {first_result.summary['p5']:.6f}")
    print(f"P50:          {first_result.summary['p50']:.6f}")
    print(f"P95:          {first_result.summary['p95']:.6f}")

    # Print metrics summary for all frequencies
    print("\nMetrics:")
    print("-" * 30)
    frontier_summary = []
    for dt in sorted(rebalance_freqs):
        metric_data = all_metrics[dt]
        te_bps = metric_data["te_bps"]
        cost_bps = metric_data["cost_bps"]
        print(f"Δt={dt}: TE={te_bps:.1f} bps, Cost={cost_bps:.1f} bps")
        frontier_summary.append(f"{dt}→TE={te_bps:.1f},Cost={cost_bps:.1f}")

    print("\nFrontier:", " | ".join(frontier_summary))
    print(
        "\nTrade-off: Smaller Δt → lower TE, higher cost; "
        "larger Δt → higher TE, lower cost."
    )

    print("\nNotes:")
    print("-" * 30)
    for key, value in first_result.notes.items():
        if key != "metrics":  # Skip metrics as they're printed above
            print(f"{key}: {value}")

    # Print summary as JSON
    print("\nSummary (JSON):")
    print("-" * 30)
    print(json.dumps(first_result.summary, indent=2))

    # Write CSV if requested
    if args.out_csv:
        import csv
        from datetime import datetime

        # Ensure parent directory exists
        csv_dir = os.path.dirname(args.out_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        # Write CSV with metrics for all Δt
        with open(args.out_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Header
            writer.writerow(
                [
                    "dt",
                    "te_bps",
                    "cost_bps",
                    "seed",
                    "n_paths",
                    "steps",
                    "fee_bps",
                    "S0",
                    "K",
                    "T",
                    "r",
                    "sigma",
                    "option_type",
                    "timestamp",
                ]
            )

            # Data rows
            timestamp = datetime.now().isoformat()
            for dt in sorted(rebalance_freqs):
                result = all_results[dt]
                metrics = all_metrics[dt]
                writer.writerow(
                    [
                        dt,
                        metrics["te_bps"],
                        metrics["cost_bps"],
                        args.seed,
                        args.n_paths,
                        result.notes["steps"],
                        args.fee_bps,
                        args.S0,
                        args.K,
                        args.T,
                        args.r,
                        args.sigma,
                        args.option_type,
                        timestamp,
                    ]
                )

        print("Saved metrics →", args.out_csv)

    # Create plots directory and save histogram
    # Use environment variable for test plots directory if set
    plots_dir = os.environ.get("TEST_PLOTS_DIR", os.path.join(os.path.dirname(__file__), "..", "plots"))
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "hedge_pnl.png")

    # Use first result for plotting (representative)
    plot_pnl_histogram(
        first_result.pnl_paths, first_result.summary, args, all_results, plot_path
    )

    # Create frontier scatter plot
    plot_frontier_scatter(all_results, args, plots_dir)


if __name__ == "__main__":
    main()
