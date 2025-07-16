import numpy as np
import pytest
from quant_option import (
    black_scholes_call,
    bs_delta,
    bs_vega,
    compute_var_cvar,
    monte_carlo_price,
    download_log_returns,
    annualized_vol,
)


def test_black_scholes_call():
    # Known value for S0=100, K=100, r=0, sigma=0.2, T=1
    price = black_scholes_call(100, 100, 0.0, 0.2, 1.0)
    assert pytest.approx(price, rel=1e-3) == 7.9656


def test_bs_delta():
    delta = bs_delta(100, 100, 0.0, 0.2, 1.0)
    assert pytest.approx(delta, rel=1e-3) == 0.5398


def test_bs_vega():
    vega = bs_vega(100, 100, 0.0, 0.2, 1.0)
    assert pytest.approx(vega, rel=1e-3) == 39.695


def test_compute_var_cvar():
    # With alpha=0.2, the 20th percentile is -6.0, so VaR=6.0, CVaR=10.0 (mean of [-10])
    pnl = np.array([-10, -5, 0, 5, 10])
    var, cvar = compute_var_cvar(pnl, alpha=0.2)
    assert pytest.approx(var, rel=1e-6) == 6.0
    assert pytest.approx(cvar, rel=1e-6) == 10.0


def test_annualized_vol_from_test_data():
    returns = download_log_returns(
        ticker="DUMMY", csv_path="tests/test_data.csv"
    )
    sigma = annualized_vol(returns, trading_days=252)
    # Based on the test_data.csv values
    expected_sigma = 0.0734
    assert pytest.approx(sigma, abs=1e-4) == expected_sigma


def test_mc_price_reproducibility():
    "Ensures that two MC runs with the same seed yield the same price."
    price1 = monte_carlo_price(100, 100, 0.05, 0.2, 1.0, 252, 1000, 42)
    price2 = monte_carlo_price(100, 100, 0.05, 0.2, 1.0, 252, 1000, 42)
    assert price1 == price2


def test_mc_convergence_to_bs():
    """
    Tests that the MC price is reasonably close to the BS price with enough paths.
    A 5% tolerance is loose, but acceptable for a stochastic test.
    """
    S0, K, r, sigma, T = 100, 105, 0.05, 0.2, 1.0
    bs_price = black_scholes_call(S0, K, r, sigma, T)
    mc_price = monte_carlo_price(S0, K, r, sigma, T, 252, 100_000, 42)
    assert pytest.approx(mc_price, rel=0.05) == bs_price 