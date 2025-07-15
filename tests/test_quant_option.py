import numpy as np
import pytest
from quant_option import black_scholes_call, bs_delta, bs_vega, compute_var_cvar

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
    assert pytest.approx(var,  rel=1e-6) == 6.0
    assert pytest.approx(cvar, rel=1e-6) == 10.0 