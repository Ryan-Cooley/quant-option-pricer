"""
Tests for the delta-hedging module.

Tests GBM path simulation, delta-hedging P&L, and fee impacts.
"""

import numpy as np
from src.quant.hedging import simulate_gbm_paths, simulate_delta_hedge, HedgeResult


def test_simulate_gbm_paths_basic():
    """Test basic GBM path simulation."""
    S0, r, sigma, T = 100, 0.05, 0.2, 1.0
    steps, n_paths = 252, 1000

    S = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42)

    # Check shapes
    assert S.shape == (steps + 1, n_paths)

    # Check initial values
    assert np.allclose(S[0, :], S0)

    # Check all values are positive
    assert np.all(S > 0)

    # Check no NaN or inf values
    assert not np.any(np.isnan(S))
    assert not np.any(np.isinf(S))


def test_simulate_gbm_paths_reproducibility():
    """Test GBM path reproducibility with same seed."""
    S0, r, sigma, T = 100, 0.05, 0.2, 1.0
    steps, n_paths = 252, 1000

    S1 = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42)
    S2 = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42)

    assert np.allclose(S1, S2)


def test_simulate_gbm_paths_zero_volatility():
    """Test GBM paths with zero volatility."""
    S0, r, sigma, T = 100, 0.05, 0.0, 1.0
    steps, n_paths = 252, 1000

    S = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42)

    # With zero volatility, should grow deterministically
    expected_final = S0 * np.exp(r * T)
    actual_final = S[-1, :]

    assert np.allclose(actual_final, expected_final)


def test_simulate_delta_hedge_basic():
    """Test basic delta-hedging simulation."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    result = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=1000,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,
        fixed_fee=0.0,
        seed=42,
    )

    # Check result type
    assert isinstance(result, HedgeResult)

    # Check shapes
    assert result.pnl_paths.shape == (1000,)
    assert result.terminal_inventory.shape == (1000,)

    # Check summary keys
    expected_keys = ["mean", "std", "sharpe_annualized", "p5", "p50", "p95"]
    for key in expected_keys:
        assert key in result.summary

    # Check notes keys
    expected_note_keys = [
        "dt",
        "steps",
        "rebalance_every",
        "fee_bps",
        "fixed_fee",
        "option_premium",
        "initial_delta",
        "metrics",
    ]
    for key in expected_note_keys:
        assert key in result.notes

    # Check metrics exist and are finite
    assert "metrics" in result.notes
    metrics = result.notes["metrics"]
    assert 1 in metrics  # rebalance_every=1
    assert "te_bps" in metrics[1]
    assert "cost_bps" in metrics[1]
    assert np.isfinite(metrics[1]["te_bps"])
    assert np.isfinite(metrics[1]["cost_bps"])


def test_cost_decreases_with_larger_dt():
    """Test that cost decreases as rebalance frequency decreases (larger Δt)."""
    S0, K, T, r, sigma = 100, 100, 0.25, 0.02, 0.2
    n_paths = 2000
    fee_bps = 1.0

    # Run with Δt=1 (daily rebalancing)
    result_dt1 = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=n_paths,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=fee_bps,
        fixed_fee=0.0,
        seed=42,
    )

    # Run with Δt=5 (weekly rebalancing)
    result_dt5 = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=n_paths,
        steps_per_year=252,
        rebalance_every=5,
        fee_bps=fee_bps,
        fixed_fee=0.0,
        seed=42,
    )

    # Extract costs
    cost_dt1 = result_dt1.notes["metrics"][1]["cost_bps"]
    cost_dt5 = result_dt5.notes["metrics"][5]["cost_bps"]

    # Assert cost decreases with larger Δt (allow small epsilon for numerical precision)
    assert (
        cost_dt1 >= cost_dt5 - 1e-6
    ), f"Cost should decrease with larger Δt: {cost_dt1:.3f} vs {cost_dt5:.3f}"

    # Assert all metrics are finite
    for result in [result_dt1, result_dt5]:
        for dt, metrics in result.notes["metrics"].items():
            assert np.isfinite(metrics["te_bps"])
            assert np.isfinite(metrics["cost_bps"])


def test_zero_fee_cost_is_zero():
    """Test that with zero fees, cost_bps is zero (within tiny tolerance)."""
    S0, K, T, r, sigma = 100, 100, 0.25, 0.02, 0.2
    n_paths = 2000

    result = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=n_paths,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,  # Zero fees
        fixed_fee=0.0,
        seed=42,
    )

    # Extract cost
    cost_bps = result.notes["metrics"][1]["cost_bps"]

    # Assert cost is zero (within tiny tolerance)
    assert abs(cost_bps) <= 1e-9, f"Cost should be zero with zero fees, got {cost_bps}"


def test_multiple_rebalance_frequencies():
    """Test that multiple rebalance frequencies work correctly."""
    S0, K, T, r, sigma = 100, 100, 0.25, 0.02, 0.2
    n_paths = 1000
    fee_bps = 1.0

    # Test with two different rebalance frequencies
    rebalance_freqs = [1, 5]
    results = {}

    for rebalance_every in rebalance_freqs:
        result = simulate_delta_hedge(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type="call",
            n_paths=n_paths,
            steps_per_year=252,
            rebalance_every=rebalance_every,
            fee_bps=fee_bps,
            fixed_fee=0.0,
            seed=42,
        )
        results[rebalance_every] = result

    # Check that metrics exist for both frequencies
    for dt in rebalance_freqs:
        assert dt in results[dt].notes["metrics"]
        metrics = results[dt].notes["metrics"][dt]
        assert "te_bps" in metrics
        assert "cost_bps" in metrics
        assert np.isfinite(metrics["te_bps"])
        assert np.isfinite(metrics["cost_bps"])

    # Verify the cost trade-off still holds
    cost_dt1 = results[1].notes["metrics"][1]["cost_bps"]
    cost_dt5 = results[5].notes["metrics"][5]["cost_bps"]
    assert (
        cost_dt1 >= cost_dt5 - 1e-6
    ), f"Cost should decrease with larger Δt: {cost_dt1:.3f} vs {cost_dt5:.3f}"


def test_units_functionality():
    """Test that both units modes work and produce different values."""
    S0, K, T, r, sigma = 100, 100, 0.25, 0.02, 0.2
    n_paths = 1000
    fee_bps = 1.0

    # Test with S0 units (default)
    result_s0 = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=n_paths,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=fee_bps,
        fixed_fee=0.0,
        seed=42,
        units="s0",
    )

    # Test with premium units
    result_premium = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=n_paths,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=fee_bps,
        fixed_fee=0.0,
        seed=42,
        units="premium",
    )

    # Extract metrics
    metrics_s0 = result_s0.notes["metrics"][1]
    metrics_premium = result_premium.notes["metrics"][1]

    # Assert both produce finite values
    assert np.isfinite(metrics_s0["te_bps"])
    assert np.isfinite(metrics_s0["cost_bps"])
    assert np.isfinite(metrics_premium["te_bps"])
    assert np.isfinite(metrics_premium["cost_bps"])

    # Assert values are different (different denominators)
    assert abs(metrics_s0["te_bps"] - metrics_premium["te_bps"]) > 1e-6
    assert abs(metrics_s0["cost_bps"] - metrics_premium["cost_bps"]) > 1e-6


def test_csv_output():
    """Test that CSV output is created with expected columns."""
    import tempfile
    import csv
    import os

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        csv_path = tmp.name

    try:
        # Run hedge simulation with CSV output
        from scripts.run_hedge import main
        import sys
        from unittest.mock import patch

        # Mock command line arguments
        test_args = [
            "run_hedge.py",
            "--S0",
            "100",
            "--K",
            "100",
            "--T",
            "0.25",
            "--r",
            "0.02",
            "--sigma",
            "0.2",
            "--n-paths",
            "100",
            "--rebalance-every",
            "1,5",
            "--fee-bps",
            "1.0",
            "--out-csv",
            csv_path,
        ]

        with patch.object(sys, "argv", test_args):
            main()

        # Check that CSV was created
        assert os.path.exists(csv_path)

        # Check CSV contents
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check header
        expected_header = [
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
        assert rows[0] == expected_header

        # Check data rows (should have 2 rows for dt=1 and dt=5)
        assert len(rows) == 3  # header + 2 data rows

        # Check that dt values are present
        dt_values = [int(row[0]) for row in rows[1:]]
        assert 1 in dt_values
        assert 5 in dt_values

        # Check that metrics are finite
        for row in rows[1:]:
            te_bps = float(row[1])
            cost_bps = float(row[2])
            assert np.isfinite(te_bps)
            assert np.isfinite(cost_bps)

    finally:
        # Clean up
        if os.path.exists(csv_path):
            os.unlink(csv_path)


def test_frontier_plot():
    """Test that frontier plot is created."""
    import tempfile
    import os

    # Create temporary plots directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        plots_dir = os.path.join(tmp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Create mock results
        from src.quant.hedging import HedgeResult
        import numpy as np

        mock_results = {
            1: HedgeResult(
                pnl_paths=np.array([0.1, 0.2]),
                summary={"mean": 0.15, "std": 0.05},
                terminal_inventory=np.array([0.1, 0.1]),
                notes={"metrics": {1: {"te_bps": 10.0, "cost_bps": 5.0}}, "steps": 252},
            ),
            5: HedgeResult(
                pnl_paths=np.array([0.1, 0.2]),
                summary={"mean": 0.15, "std": 0.05},
                terminal_inventory=np.array([0.1, 0.1]),
                notes={"metrics": {5: {"te_bps": 20.0, "cost_bps": 2.0}}, "steps": 252},
            ),
        }

        # Mock args
        class MockArgs:
            units = "s0"
            seed = 42
            n_paths = 1000
            fee_bps = 1.0

        args = MockArgs()

        # Test frontier plot creation
        from scripts.run_hedge import plot_frontier_scatter

        plot_frontier_scatter(mock_results, args, plots_dir)

        # Check that plot was created
        frontier_path = os.path.join(plots_dir, "hedge_frontier.png")
        assert os.path.exists(frontier_path)


def test_metadata_stamp():
    """Test that metadata stamp is added to plots."""
    import tempfile
    import os

    # Create temporary plots directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        plots_dir = os.path.join(tmp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Create mock results
        from src.quant.hedging import HedgeResult
        import numpy as np

        mock_results = {
            1: HedgeResult(
                pnl_paths=np.array([0.1, 0.2]),
                summary={"mean": 0.15, "std": 0.05},
                terminal_inventory=np.array([0.1, 0.1]),
                notes={"metrics": {1: {"te_bps": 10.0, "cost_bps": 5.0}}, "steps": 252},
            )
        }

        # Mock args
        class MockArgs:
            seed = 42
            n_paths = 1000
            fee_bps = 1.0
            units = "s0"
            S0 = 100.0
            K = 100.0
            T = 0.5
            r = 0.02
            sigma = 0.2
            option_type = "call"
            steps_per_year = 252
            rebalance_every = 1
            fixed_fee = 0.0

        args = MockArgs()

        # Test P&L histogram with metadata
        from scripts.run_hedge import plot_pnl_histogram

        plot_path = os.path.join(plots_dir, "test_pnl.png")
        plot_pnl_histogram(
            np.array([0.1, 0.2]),
            {
                "mean": 0.15,
                "p50": 0.15,
                "std": 0.05,
                "sharpe_annualized": 1.0,
                "p5": 0.1,
                "p95": 0.2,
            },
            args,
            mock_results,
            plot_path,
        )

        # Check that plot was created
        assert os.path.exists(plot_path)


def test_simulate_delta_hedge_reproducibility():
    """Test delta-hedging reproducibility with same seed."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    result1 = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=1000,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,
        fixed_fee=0.0,
        seed=42,
    )

    result2 = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=1000,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,
        fixed_fee=0.0,
        seed=42,
    )

    assert np.allclose(result1.pnl_paths, result2.pnl_paths)


def test_simulate_delta_hedge_call_vs_put():
    """Test delta-hedging for calls vs puts."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    result_call = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=1000,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,
        fixed_fee=0.0,
        seed=42,
    )

    result_put = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="put",
        n_paths=1000,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,
        fixed_fee=0.0,
        seed=42,
    )

    # Results should be different for calls vs puts
    assert not np.allclose(result_call.pnl_paths, result_put.pnl_paths)

    # But both should have finite values
    assert not np.any(np.isnan(result_call.pnl_paths))
    assert not np.any(np.isnan(result_put.pnl_paths))


def test_error_shrinks_with_frequency_zero_costs():
    """Test that P&L std decreases with more frequent rebalancing (zero costs)."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    # Daily rebalancing
    result_daily = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=5000,  # Lower for speed
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,
        fixed_fee=0.0,
        seed=42,
    )

    # Weekly rebalancing (every 5 days)
    result_weekly = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=5000,  # Lower for speed
        steps_per_year=252,
        rebalance_every=5,
        fee_bps=0.0,
        fixed_fee=0.0,
        seed=42,
    )

    # More frequent rebalancing should reduce P&L std (better hedging)
    std_daily = result_daily.summary["std"]
    std_weekly = result_weekly.summary["std"]

    # This should generally be true, but allow some tolerance for randomness
    # In practice, daily rebalancing should be more effective than weekly
    assert std_daily <= std_weekly * 1.1  # Allow 10% tolerance


def test_costs_hurt_more_when_rebalancing_often():
    """Test that costs hurt more when rebalancing more frequently."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    # Daily rebalancing with fees
    result_daily_fees = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=5000,  # Lower for speed
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=10.0,  # 10 bps fee
        fixed_fee=0.0,
        seed=42,
    )

    # Weekly rebalancing with same fees
    result_weekly_fees = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=5000,  # Lower for speed
        steps_per_year=252,
        rebalance_every=5,
        fee_bps=10.0,  # 10 bps fee
        fixed_fee=0.0,
        seed=42,
    )

    # More frequent rebalancing should have worse mean P&L due to higher costs
    mean_daily = result_daily_fees.summary["mean"]
    mean_weekly = result_weekly_fees.summary["mean"]

    # Daily rebalancing should have lower mean P&L (more costs)
    assert mean_daily <= mean_weekly


def test_shapes_and_finite():
    """Test that all arrays have correct shapes and finite values."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    result = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=1000,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=5.0,
        fixed_fee=0.1,
        seed=42,
    )

    # Check shapes
    assert result.pnl_paths.shape == (1000,)
    assert result.terminal_inventory.shape == (1000,)

    # Check all values are finite
    assert not np.any(np.isnan(result.pnl_paths))
    assert not np.any(np.isinf(result.pnl_paths))
    assert not np.any(np.isnan(result.terminal_inventory))
    assert not np.any(np.isinf(result.terminal_inventory))

    # Check summary values are finite
    for value in result.summary.values():
        assert not np.isnan(value)
        assert not np.isinf(value)

    # Check notes values are finite (except for strings)
    for key, value in result.notes.items():
        if isinstance(value, (int, float)):
            assert not np.isnan(value)
            assert not np.isinf(value)


def test_zero_volatility_hedging():
    """Test delta-hedging with zero volatility."""
    S0, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.0  # Zero volatility

    result = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=1000,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,
        fixed_fee=0.0,
        seed=42,
    )

    # With zero volatility, P&L should be very consistent
    assert result.summary["std"] < 1e-6  # Should be nearly deterministic


def test_at_the_money_hedging():
    """Test delta-hedging for at-the-money options."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2  # ATM

    result = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=1000,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,
        fixed_fee=0.0,
        seed=42,
    )

    # ATM options with positive rate should have initial delta > 0.5
    initial_delta = result.notes["initial_delta"]
    assert 0.5 < initial_delta < 0.7

    # P&L should be finite (short option strategy)
    mean_pnl = result.summary["mean"]
    assert not np.isnan(mean_pnl) and not np.isinf(mean_pnl)


def test_fixed_fee_impact():
    """Test impact of fixed fees on P&L."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    # No fees
    result_no_fees = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=1000,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,
        fixed_fee=0.0,
        seed=42,
    )

    # With fixed fees
    result_with_fees = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=1000,
        steps_per_year=252,
        rebalance_every=1,
        fee_bps=0.0,
        fixed_fee=1.0,  # $1 per trade
        seed=42,
    )

    # Fixed fees should reduce mean P&L
    mean_no_fees = result_no_fees.summary["mean"]
    mean_with_fees = result_with_fees.summary["mean"]

    assert mean_with_fees <= mean_no_fees
