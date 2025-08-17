#!/usr/bin/env python3
"""
Quickstart Demo Script

Demonstrates the three main features of quant-option-pricer:
1. Discrete Delta Hedging - P&L simulation with rebalancing
2. Implied Volatility - Round-trip BS price → IV → price  
3. IV Surface - CSV-driven volatility surface generation
"""

import sys
import os
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print("=" * 60)

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=".."
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def test_iv_roundtrip():
    """Test IV solver with round-trip validation."""
    print(f"\n{'='*60}")
    print("Testing Implied Volatility Round-Trip")
    print("=" * 60)

    try:
        from src.quant.pricing import bs_price
        from src.quant.iv import implied_vol_bs

        # Test parameters
        S, K, T, r, sigma_true = 100.0, 100.0, 0.5, 0.02, 0.25

        # Step 1: Compute BS price
        price = bs_price(S, K, T, r, sigma_true, "call")
        print(f"Original BS price: {price:.6f}")

        # Step 2: Solve for implied volatility
        sigma_implied = implied_vol_bs(price, S, K, T, r, "call")
        print(f"Implied volatility: {sigma_implied:.6f}")
        print(f"True volatility:   {sigma_true:.6f}")
        print(f"IV error:          {abs(sigma_implied - sigma_true):.2e}")

        # Step 3: Round-trip: recompute price from IV
        price_roundtrip = bs_price(S, K, T, r, sigma_implied, "call")
        print(f"Round-trip price:  {price_roundtrip:.6f}")
        print(f"Price error:       {abs(price_roundtrip - price):.2e}")

        return True
    except Exception as e:
        print(f"Error in IV round-trip test: {e}")
        return False


def main():
    """Run the quickstart demo."""
    print("🚀 Quant-Option-Pricer Quickstart Demo")
    print("=" * 60)

    # Ensure we're in the right directory
    os.chdir(os.path.dirname(__file__))

    # 1. Delta Hedging Demo
    success1 = run_command(
        "python scripts/run_hedge.py --S0 100 --K 100 --T 0.25 --r 0.01 --sigma 0.2 --n-paths 5000 --rebalance-every 1 --fee-bps 1.0",
        "Delta Hedging P&L Simulation",
    )

    # 2. IV Round-Trip Test
    success2 = test_iv_roundtrip()

    # 3. IV Surface Generation
    success3 = run_command(
        "python scripts/calibrate_surface.py", "IV Surface Generation"
    )

    # Summary
    print(f"\n{'='*60}")
    print("📊 DEMO SUMMARY")
    print("=" * 60)

    artifacts = []
    if success1 and Path("../plots/hedge_pnl.png").exists():
        artifacts.append("✅ plots/hedge_pnl.png")
    if success3 and Path("../plots/iv_surface.png").exists():
        artifacts.append("✅ plots/iv_surface.png")
    if Path("../benchmarks/benchmarks.csv").exists():
        artifacts.append("✅ benchmarks/benchmarks.csv")

    print("Generated artifacts:")
    for artifact in artifacts:
        print(f"  {artifact}")

    print(f"\nDelta Hedging: {'✅' if success1 else '❌'}")
    print(f"IV Round-Trip: {'✅' if success2 else '❌'}")
    print(f"IV Surface:    {'✅' if success3 else '❌'}")

    if all([success1, success2, success3]):
        print("\n🎉 All demos completed successfully!")
    else:
        print("\n⚠️  Some demos failed. Check the output above.")


if __name__ == "__main__":
    main()
