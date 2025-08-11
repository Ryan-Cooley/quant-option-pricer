import numpy as np
import pytest
from quant_option import (
    compute_var_cvar,
    download_log_returns,
    annualized_vol,
)
from src.quant.pricing import (
    bs_price as black_scholes,
    bs_delta,
    bs_vega,
    mc_euro_price as monte_carlo_price,
)


def test_black_scholes_call():
    # Known value for S0=100, K=100, r=0, sigma=0.2, T=1
    price = black_scholes(100, 100, 1.0, 0.0, 0.2, "call")
    assert pytest.approx(price, rel=1e-3) == 7.9656


def test_bs_delta():
    delta = bs_delta(100, 100, 1.0, 0.0, 0.2, "call")
    assert pytest.approx(delta, rel=1e-3) == 0.5398


def test_bs_vega():
    vega = bs_vega(100, 100, 1.0, 0.0, 0.2)
    assert pytest.approx(vega, rel=1e-3) == 39.695


def test_compute_var_cvar():
    # With alpha=0.2, the 20th percentile is -6.0, so VaR=6.0, CVaR=10.0 (mean of [-10])
    pnl = np.array([-10, -5, 0, 5, 10])
    var, cvar = compute_var_cvar(pnl, alpha=0.2)
    assert pytest.approx(var, rel=1e-6) == 6.0
    assert pytest.approx(cvar, rel=1e-6) == 10.0


def test_annualized_vol_from_test_data():
    returns = download_log_returns(ticker="DUMMY", csv_path="tests/test_data.csv")
    sigma = annualized_vol(returns, trading_days=252)
    # Based on the test_data.csv values
    expected_sigma = 0.0734
    assert pytest.approx(sigma, abs=1e-4) == expected_sigma


def test_mc_price_reproducibility():
    "Ensures that two MC runs with the same seed yield the same price."
    price1 = monte_carlo_price(100, 100, 1.0, 0.05, 0.2, n_paths=1000, seed=42, option_type="call")
    price2 = monte_carlo_price(100, 100, 1.0, 0.05, 0.2, n_paths=1000, seed=42, option_type="call")
    assert price1 == price2


def test_mc_convergence_to_bs():
    """
    Tests that the MC price is reasonably close to the BS price with enough paths.
    A 5% tolerance is loose, but acceptable for a stochastic test.
    """
    S0, K, r, sigma, T = 100, 105, 0.05, 0.2, 1.0
    bs_price = black_scholes(S0, K, T, r, sigma, "call")
    mc_price = monte_carlo_price(S0, K, T, r, sigma, n_paths=100_000, seed=42, option_type="call")
    assert pytest.approx(mc_price, rel=0.05) == bs_price


def test_zero_volatility():
    """
    Sanity check: Zero volatility should result in deterministic price movement.
    With zero volatility, the option price should equal the discounted intrinsic value.
    """
    S0, K, r, sigma, T = 100, 105, 0.05, 0.0, 1.0
    # With zero volatility, price should be max(0, S0*exp(r*T) - K) * exp(-r*T)
    expected_price = max(0, S0 * np.exp(r * T) - K) * np.exp(-r * T)
    bs_price = black_scholes(S0, K, T, r, sigma, "call")
    mc_price = monte_carlo_price(S0, K, T, r, sigma, n_paths=1000, seed=42, option_type="call")

    # Both should be very close to expected
    assert pytest.approx(bs_price, rel=1e-10) == expected_price
    assert pytest.approx(mc_price, rel=1e-10) == expected_price


def test_zero_time_to_maturity():
    """
    Sanity check: Zero time to maturity should result in immediate payoff.
    Option price should equal max(0, S0 - K) for calls.
    """
    S0, K, r, sigma, T = 100, 105, 0.05, 0.2, 0.0
    expected_payoff = max(0, S0 - K)

    bs_price = black_scholes(S0, K, T, r, sigma, "call")
    mc_price = monte_carlo_price(S0, K, T, r, sigma, n_paths=1000, seed=42, option_type="call")

    # Both should equal the immediate payoff
    assert pytest.approx(bs_price, rel=1e-10) == expected_payoff
    assert pytest.approx(mc_price, rel=1e-10) == expected_payoff


def test_at_the_money_zero_volatility():
    """
    Additional sanity check: At-the-money call with zero volatility and zero rate
    should have zero value (no time value, no intrinsic value).
    """
    S0, K, r, sigma, T = 100, 100, 0.0, 0.0, 1.0
    bs_price = black_scholes(S0, K, r, sigma, T, "call")
    mc_price = monte_carlo_price(S0, K, r, sigma, T, 252, 1000, 42, "call")

    assert pytest.approx(bs_price, rel=1e-10) == 0.0
    assert pytest.approx(mc_price, rel=1e-10) == 0.0
