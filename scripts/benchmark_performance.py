#!/usr/bin/env python3
"""
Performance Benchmark: NumPy vs Numba Monte Carlo Option Pricing

Compare performance of pure NumPy vs Numba JIT-compiled Monte Carlo pricing.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from src.quant.pricing import mc_euro_price  # noqa: E402


def time_numpy_call_pricer(
    n_paths: int, steps: int, *, shared_normals=None
) -> tuple[float, float]:
    """
    Time the NumPy-based Monte Carlo call option pricer.

    Args:
        n_paths: Number of Monte Carlo paths
        steps: Number of time steps per path
        shared_normals: Optional pre-generated normal random numbers

    Returns:
        Tuple of (time in seconds, computed price)
    """
    # Test parameters
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    if shared_normals is None:
        # Generate normals internally (original behavior)
        # Warm up
        _ = mc_euro_price(S0, K, T, r, sigma, n_paths=1000, steps_per_year=steps)

        # Time the actual computation
        start_time = time.perf_counter()
        price = mc_euro_price(S0, K, T, r, sigma, n_paths=n_paths, steps_per_year=steps)
        end_time = time.perf_counter()
    else:
        # Use shared normals for fair comparison
        dt = T / steps
        drift = (r - 0.5 * sigma * sigma) * dt
        diffusion = sigma * np.sqrt(dt)

        # Time the actual computation
        start_time = time.perf_counter()

        # Vectorized log-price evolution
        logS = np.log(S0) + np.cumsum(drift + diffusion * shared_normals, axis=0)
        ST = np.exp(logS[-1])

        # Calculate payoff and discount
        payoff = np.maximum(ST - K, 0)
        price = np.exp(-r * T) * np.mean(payoff)

        end_time = time.perf_counter()

    return end_time - start_time, price


def time_numba_call_pricer(
    n_paths: int, steps: int, *, shared_normals=None
) -> tuple[float, float]:
    """
    Time the Numba JIT-compiled Monte Carlo call option pricer.

    Args:
        n_paths: Number of Monte Carlo paths
        steps: Number of time steps per path
        shared_normals: Optional pre-generated normal random numbers

    Returns:
        Tuple of (time in seconds, computed price), or (np.nan, np.nan) if Numba is not
        available
    """
    try:
        from numba import njit, prange
        import math
    except ImportError:
        warnings.warn("Numba not available, skipping Numba benchmark")
        return np.nan, np.nan

    # Define optimized Numba JIT-compiled kernel
    @njit(parallel=True, fastmath=True, cache=True)
    def mc_call_numba_kernel(S0, K, T, r, sigma, normals):
        steps, n_paths = normals.shape
        dt = T / steps
        drift = (r - 0.5 * sigma * sigma) * dt
        vol = sigma * math.sqrt(dt)
        payoffs = np.empty(n_paths, dtype=np.float64)
        for j in prange(n_paths):
            s = S0
            for t in range(steps):
                s *= math.exp(drift + vol * normals[t, j])
            p = s - K
            payoffs[j] = p if p > 0.0 else 0.0
        # Numba supports np.mean in nopython mode
        return math.exp(-r * T) * payoffs.mean()

    # Test parameters
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    try:
        if shared_normals is None:
            # Generate normals internally (original behavior)
            # Warm up with tiny array to compile
            warmup_normals = np.random.normal(size=(32, 1024)).astype(np.float64)
            _ = mc_call_numba_kernel(S0, K, T, r, sigma, warmup_normals)

            # Generate normals for actual computation
            normals = np.random.normal(size=(steps, n_paths)).astype(np.float64)
        else:
            # Use shared normals for fair comparison
            normals = shared_normals

        # Time the actual computation
        start_time = time.perf_counter()
        price = mc_call_numba_kernel(S0, K, T, r, sigma, normals)
        end_time = time.perf_counter()

        return end_time - start_time, price

    except Exception as e:
        warnings.warn(f"Numba JIT compilation failed: {e}")
        return np.nan, np.nan


def run_bench(path_counts=(10_000, 100_000, 500_000, 1_000_000), steps=252, repeats=3):
    """
    Run performance benchmarks for different path counts.

    Args:
        path_counts: Tuple of path counts to test
        steps: Number of time steps per path
        repeats: Number of repetitions per test

    Returns:
        DataFrame with columns ['engine', 'n_paths', 'seconds']
    """
    results = []

    # Explicit warm-up for Numba kernel with matching specialization
    print("Warming up Numba kernel...")
    warmup_normals = np.random.normal(size=(steps, 8192)).astype(np.float64)
    _ = time_numba_call_pricer(8192, steps, shared_normals=warmup_normals)

    print("\nStage Timing Analysis:")
    print("=" * 50)

    for n_paths in path_counts:
        print("\nTesting", f"{n_paths:,}", "paths...")

        # Stage 1: RNG generation timing
        rng_start = time.perf_counter()
        Z = np.random.normal(size=(steps, n_paths)).astype(np.float64)
        rng_end = time.perf_counter()
        rng_time = rng_end - rng_start

        # Verify shapes and dtypes
        print(
            f"  Normals: shape={Z.shape}, dtype={Z.dtype}, "
            f"C-contiguous={Z.flags['C_CONTIGUOUS']}"
        )

        # Stage 2: NumPy computation timing
        numpy_times = []
        numpy_prices = []
        for _ in range(repeats):
            t, price = time_numpy_call_pricer(n_paths, steps, shared_normals=Z)
            numpy_times.append(t)
            numpy_prices.append(price)

        avg_numpy_time = np.mean(numpy_times)
        avg_numpy_price = np.mean(numpy_prices)

        # Stage 3: Numba computation timing
        numba_times = []
        numba_prices = []
        for _ in range(repeats):
            t, price = time_numba_call_pricer(n_paths, steps, shared_normals=Z)
            if not np.isnan(t):
                numba_times.append(t)
                numba_prices.append(price)

        if numba_times:
            avg_numba_time = np.mean(numba_times)
            avg_numba_price = np.mean(numba_prices)

            # Correctness check
            price_diff = abs(avg_numpy_price - avg_numba_price)
            print(
                f"  Prices: NumPy={avg_numpy_price:.6f}, Numba={avg_numba_price:.6f}, "
                f"diff={price_diff:.6f}"
            )
            assert (
                price_diff < 5e-4
            ), f"Price difference {price_diff:.6f} exceeds tolerance 5e-4"

            # Stage timing table
            print(f"  Timing (avg of {repeats} runs):")
            print(f"    RNG generation: {rng_time:.6f}s")
            print(f"    NumPy compute:  {avg_numpy_time:.6f}s")
            print(f"    Numba compute:  {avg_numba_time:.6f}s")

            results.append(
                {"engine": "numpy", "n_paths": n_paths, "seconds": avg_numpy_time}
            )
            results.append(
                {"engine": "numba", "n_paths": n_paths, "seconds": avg_numba_time}
            )
        else:
            print("  Prices: NumPy=", f"{avg_numpy_price:.6f}", ", Numba=N/A")
            print(f"  Timing (avg of {repeats} runs):")
            print(f"    RNG generation: {rng_time:.6f}s")
            print(f"    NumPy compute:  {avg_numpy_time:.6f}s")
            print("    Numba compute:  N/A")

            results.append(
                {"engine": "numpy", "n_paths": n_paths, "seconds": avg_numpy_time}
            )
            results.append({"engine": "numba", "n_paths": n_paths, "seconds": np.nan})

    return pd.DataFrame(results)


def emit_csv_and_snippet(
    df, csv_path="benchmarks/benchmarks.csv", md_path="benchmarks/README_snippet.md"
):
    """
    Write benchmark results to CSV and generate Markdown table.

    Args:
        df: DataFrame with benchmark results
        csv_path: Path to save CSV file
        md_path: Path to save Markdown snippet
    """
    # Create benchmarks directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Save CSV
    df.to_csv(csv_path, index=False)
    print("Saved CSV →", csv_path)

    # Generate Markdown table
    md_content = "# Performance Benchmarks\n\n"
    md_content += "Monte Carlo option pricing performance comparison.\n\n"
    md_content += "| Engine | Paths | Seconds |\n"
    md_content += "|---|---:|---:|\n"

    for _, row in df.iterrows():
        if pd.isna(row["seconds"]):
            md_content += f"| {row['engine']} | {row['n_paths']:,} | N/A |\n"
        else:
            md_content += (
                f"| {row['engine']} | {row['n_paths']:,} | {row['seconds']:.3f} |\n"
            )

    # Save Markdown
    with open(md_path, "w") as f:
        f.write(md_content)
    print("Saved Markdown →", md_path)

    # Print table to stdout
    print("\n" + md_content)


def create_benchmark_plot(df, plot_path="benchmarks/benchmarks.png"):
    """
    Create a log-log plot of benchmark results.

    Args:
        df: DataFrame with benchmark results
        plot_path: Path to save the plot
    """
    # Create benchmarks directory if it doesn't exist
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Filter out NaN values
    df_clean = df.dropna()

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot each engine
    for engine in df_clean["engine"].unique():
        engine_data = df_clean[df_clean["engine"] == engine]
        plt.loglog(
            engine_data["n_paths"],
            engine_data["seconds"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=engine.capitalize(),
        )

    # Customize plot
    plt.xlabel("Number of Paths", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Monte Carlo Option Pricing Performance: NumPy vs Numba", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    # Format x-axis ticks
    plt.xticks([1e4, 1e5, 1e6], ["10K", "100K", "1M"])

    # Add performance annotations
    for engine in df_clean["engine"].unique():
        engine_data = df_clean[df_clean["engine"] == engine]
        if len(engine_data) >= 2:
            # Calculate speedup at largest path count
            largest_paths = engine_data["n_paths"].max()
            largest_time = engine_data[engine_data["n_paths"] == largest_paths][
                "seconds"
            ].iloc[0]

            # Find corresponding NumPy time for comparison
            if engine == "numba":
                numpy_time = df_clean[
                    (df_clean["engine"] == "numpy")
                    & (df_clean["n_paths"] == largest_paths)
                ]["seconds"].iloc[0]
                speedup = numpy_time / largest_time
                plt.annotate(
                    f"{speedup:.1f}x faster",
                    xy=(largest_paths, largest_time),
                    xytext=(largest_paths * 1.5, largest_time * 0.7),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=10,
                    ha="left",
                )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot → {os.path.abspath(plot_path)}")
    plt.close()


def main():
    """Main benchmark function."""
    print("Performance Benchmark: NumPy vs Numba")
    print("=" * 50)

    # Run benchmarks
    df = run_bench()

    # Emit results
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "benchmarks", "benchmarks.csv"
    )
    md_path = os.path.join(
        os.path.dirname(__file__), "..", "benchmarks", "README_snippet.md"
    )
    plot_path = os.path.join(
        os.path.dirname(__file__), "..", "benchmarks", "benchmarks.png"
    )

    emit_csv_and_snippet(df, csv_path, md_path)
    create_benchmark_plot(df, plot_path)

    print("\nBenchmark complete!")
    return 0


if __name__ == "__main__":
    exit(main())
