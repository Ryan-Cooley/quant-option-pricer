#!/usr/bin/env python3
"""
Performance Benchmark: Numba vs Regular Python Monte Carlo
"""

import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

def simulate_gbm_python(S0, r, sigma, T, steps, n_paths, seed):
    """Regular Python implementation of GBM simulation."""
    np.random.seed(seed)
    dt = T / steps
    payoff_sum = 0.0
    for i in range(n_paths):
        logS = 0.0
        for j in range(steps):
            z = np.random.normal()
            logS += (r - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * z
        ST = S0 * np.exp(logS)
        payoff = max(ST - 150, 0.0)  # Fixed strike for benchmark
        payoff_sum += payoff
    return np.exp(-r * T) * (payoff_sum / n_paths)

@njit
def simulate_gbm_numba(S0, r, sigma, T, steps, n_paths, seed):
    """Numba-accelerated implementation of GBM simulation."""
    np.random.seed(seed)
    dt = T / steps
    payoff_sum = 0.0
    for i in range(n_paths):
        logS = 0.0
        for j in range(steps):
            z = np.random.normal()
            logS += (r - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * z
        ST = S0 * np.exp(logS)
        payoff = max(ST - 150, 0.0)  # Fixed strike for benchmark
        payoff_sum += payoff
    return np.exp(-r * T) * (payoff_sum / n_paths)

def benchmark_performance():
    """Run performance benchmarks comparing Python vs Numba."""
    
    # Test parameters
    S0, r, sigma, T = 150, 0.01, 0.2, 1.0
    steps = 252
    path_counts = [1000, 5000, 10000, 50000, 100000]
    
    print("Performance Benchmark: Python vs Numba Monte Carlo")
    print("=" * 60)
    print(f"Parameters: S0={S0}, r={r}, σ={sigma}, T={T}, steps={steps}")
    print()
    
    python_times = []
    numba_times = []
    speedups = []
    
    for n_paths in path_counts:
        print(f"Testing {n_paths:,} paths...")
        
        # Benchmark Python implementation
        start_time = time.time()
        python_result = simulate_gbm_python(S0, r, sigma, T, steps, n_paths, 42)
        python_time = time.time() - start_time
        python_times.append(python_time)
        
        # Benchmark Numba implementation
        start_time = time.time()
        numba_result = simulate_gbm_numba(S0, r, sigma, T, steps, n_paths, 42)
        numba_time = time.time() - start_time
        numba_times.append(numba_time)
        
        # Calculate speedup
        speedup = python_time / numba_time
        speedups.append(speedup)
        
        print(f"  Python: {python_time:.4f}s, Result: {python_result:.4f}")
        print(f"  Numba:  {numba_time:.4f}s, Result: {numba_result:.4f}")
        print(f"  Speedup: {speedup:.1f}x")
        print()
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Timing comparison
    plt.subplot(1, 2, 1)
    plt.loglog(path_counts, python_times, 'o-', label='Python', linewidth=2, markersize=8)
    plt.loglog(path_counts, numba_times, 's-', label='Numba', linewidth=2, markersize=8)
    plt.xlabel('Number of Paths')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Speedup
    plt.subplot(1, 2, 2)
    plt.semilogx(path_counts, speedups, 'o-', color='green', linewidth=2, markersize=8)
    plt.xlabel('Number of Paths')
    plt.ylabel('Speedup (Python/Numba)')
    plt.title('Numba Speedup')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("Summary:")
    print(f"Average speedup: {np.mean(speedups):.1f}x")
    print(f"Max speedup: {np.max(speedups):.1f}x")
    print(f"Min speedup: {np.min(speedups):.1f}x")
    print(f"Benchmark plot saved as: benchmark_results.png")

if __name__ == "__main__":
    benchmark_performance() 