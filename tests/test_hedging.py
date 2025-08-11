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
    ]
    for key in expected_note_keys:
        assert key in result.notes

    # Check no NaN or inf values
    assert not np.any(np.isnan(result.pnl_paths))
    assert not np.any(np.isinf(result.pnl_paths))


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
