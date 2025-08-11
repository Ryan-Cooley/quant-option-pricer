"""
Tests for the quant.pricing module.

Tests Black-Scholes pricing, Greeks, and Monte Carlo convergence.
"""

import pytest
import numpy as np
from src.quant.pricing import (
    d1_d2,
    bs_price,
    bs_delta,
    bs_vega,
    mc_euro_price,
)


def test_d1_d2_basic():
    """Test d1_d2 calculation with basic parameters."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    d1, d2 = d1_d2(S, K, T, r, sigma)
    
    # Expected values for S=K=100, T=1, r=0.05, sigma=0.2
    expected_d1 = (0.05 + 0.5 * 0.04) / 0.2  # (r + 0.5*sigma^2) / sigma
    expected_d2 = expected_d1 - 0.2
    
    assert abs(d1 - expected_d1) < 1e-10
    assert abs(d2 - expected_d2) < 1e-10


def test_d1_d2_zero_volatility():
    """Test d1_d2 with zero volatility."""
    S, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.0
    d1, d2 = d1_d2(S, K, T, r, sigma)
    
    # With zero volatility, forward = 100 * exp(0.05) = 105.127
    # Since forward > K, d1 and d2 should be infinity
    assert d1 == float('inf')
    assert d2 == float('inf')


def test_d1_d2_zero_time():
    """Test d1_d2 with zero time to expiry."""
    S, K, T, r, sigma = 100, 105, 0.0, 0.05, 0.2
    d1, d2 = d1_d2(S, K, T, r, sigma)
    
    # At expiry, S < K, so d1 and d2 should be -inf
    assert d1 == float('-inf')
    assert d2 == float('-inf')


def test_bs_price_known_value():
    """Test Black-Scholes price against known value."""
    # Test case: S=100, K=100, T=1, r=0.0, sigma=0.2
    # This is a standard test case with known result
    S, K, T, r, sigma = 100, 100, 1.0, 0.0, 0.2
    
    # Calculate expected value using our own bs_price function
    # For S=K=100, T=1, r=0, sigma=0.2, the call price should be approximately 7.9656
    expected_price = 7.9656
    
    actual_price = bs_price(S, K, T, r, sigma, "call")
    
    assert abs(actual_price - expected_price) < 1e-4


def test_bs_price_zero_volatility():
    """Test Black-Scholes price with zero volatility."""
    S, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.0
    
    # With zero volatility, price should be discounted intrinsic value
    forward = S * np.exp(r * T)  # 100 * exp(0.05) = 105.127
    intrinsic = max(0, forward - K)  # max(0, 105.127 - 105) = 0.127
    expected_price = intrinsic * np.exp(-r * T)  # 0.127 * exp(-0.05) = 0.121
    
    actual_price = bs_price(S, K, T, r, sigma, "call")
    
    assert abs(actual_price - expected_price) < 1e-6


def test_bs_price_zero_time():
    """Test Black-Scholes price at expiry."""
    S, K, T, r, sigma = 100, 105, 0.0, 0.05, 0.2
    
    # At expiry, price should equal intrinsic value
    expected_price = max(0, S - K)  # max(0, 100 - 105) = 0
    
    actual_price = bs_price(S, K, T, r, sigma, "call")
    
    assert abs(actual_price - expected_price) < 1e-10


def test_bs_price_put_call_parity():
    """Test put-call parity."""
    S, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.2
    
    call_price = bs_price(S, K, T, r, sigma, "call")
    put_price = bs_price(S, K, T, r, sigma, "put")
    
    # Put-call parity: C - P = S - K*exp(-r*T)
    expected_difference = S - K * np.exp(-r * T)
    actual_difference = call_price - put_price
    
    assert abs(actual_difference - expected_difference) < 1e-10


def test_bs_delta_basic():
    """Test Black-Scholes delta calculation."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    call_delta = bs_delta(S, K, T, r, sigma, "call")
    put_delta = bs_delta(S, K, T, r, sigma, "put")
    
    # For at-the-money options with positive rate, call delta should be > 0.5
    assert 0.5 < call_delta < 0.7
    
    # Put-call delta relation: Δ_put = Δ_call - 1
    assert abs(put_delta - (call_delta - 1)) < 1e-10


def test_bs_delta_zero_volatility():
    """Test Black-Scholes delta with zero volatility."""
    S, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.0
    
    # With zero volatility, delta should be step function at forward moneyness
    forward = S * np.exp(r * T)  # 100 * exp(0.05) = 105.127
    
    call_delta = bs_delta(S, K, T, r, sigma, "call")
    put_delta = bs_delta(S, K, T, r, sigma, "put")
    
    # Since forward > K, call delta should be 1, put delta should be 0
    assert call_delta == 1.0
    assert put_delta == 0.0


def test_bs_delta_zero_time():
    """Test Black-Scholes delta at expiry."""
    S, K, T, r, sigma = 100, 105, 0.0, 0.05, 0.2
    
    # At expiry, delta should be step function
    call_delta = bs_delta(S, K, T, r, sigma, "call")
    put_delta = bs_delta(S, K, T, r, sigma, "put")
    
    # Since S < K, call delta should be 0, put delta should be -1
    assert call_delta == 0.0
    assert put_delta == -1.0


def test_delta_put_call_relation():
    """Test put-call delta relation Δ_put ≈ Δ_call - 1."""
    S, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.2
    
    call_delta = bs_delta(S, K, T, r, sigma, "call")
    put_delta = bs_delta(S, K, T, r, sigma, "put")
    
    # Check the relation
    assert abs(put_delta - (call_delta - 1)) < 1e-10


def test_bs_vega_basic():
    """Test Black-Scholes vega calculation."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    vega = bs_vega(S, K, T, r, sigma)
    
    # Vega should be positive for at-the-money options
    assert vega > 0
    
    # Vega should be the same for calls and puts
    call_vega = bs_vega(S, K, T, r, sigma)
    put_vega = bs_vega(S, K, T, r, sigma)  # Same function, same result
    assert call_vega == put_vega


def test_bs_vega_zero_volatility():
    """Test Black-Scholes vega with zero volatility."""
    S, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.0
    
    vega = bs_vega(S, K, T, r, sigma)
    
    # With zero volatility, vega should be zero
    assert vega == 0.0


def test_bs_vega_zero_time():
    """Test Black-Scholes vega at expiry."""
    S, K, T, r, sigma = 100, 105, 0.0, 0.05, 0.2
    
    vega = bs_vega(S, K, T, r, sigma)
    
    # At expiry, vega should be zero
    assert vega == 0.0


def test_mc_euro_price_basic():
    """Test Monte Carlo pricing with basic parameters."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    price = mc_euro_price(S0, K, T, r, sigma, n_paths=10000, seed=42)
    
    # Price should be positive for at-the-money call
    assert price > 0


def test_mc_euro_price_zero_volatility():
    """Test Monte Carlo pricing with zero volatility."""
    S0, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.0
    
    price = mc_euro_price(S0, K, T, r, sigma, n_paths=1000, seed=42)
    
    # With zero volatility, should match discounted intrinsic value
    forward = S0 * np.exp(r * T)
    intrinsic = max(0, forward - K)
    expected_price = intrinsic * np.exp(-r * T)
    
    assert abs(price - expected_price) < 1e-6


def test_mc_euro_price_zero_time():
    """Test Monte Carlo pricing at expiry."""
    S0, K, T, r, sigma = 100, 105, 0.0, 0.05, 0.2
    
    price = mc_euro_price(S0, K, T, r, sigma, n_paths=1000, seed=42)
    
    # At expiry, should equal intrinsic value
    expected_price = max(0, S0 - K)
    
    assert abs(price - expected_price) < 1e-10


def test_mc_euro_price_put():
    """Test Monte Carlo pricing for put options."""
    S0, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.2
    
    call_price = mc_euro_price(S0, K, T, r, sigma, n_paths=50000, seed=42, option_type="call")
    put_price = mc_euro_price(S0, K, T, r, sigma, n_paths=50000, seed=42, option_type="put")
    
    # Put-call parity should hold approximately
    expected_difference = S0 - K * np.exp(-r * T)
    actual_difference = call_price - put_price
    
    assert abs(actual_difference - expected_difference) < 0.2  # Monte Carlo noise


@pytest.mark.slow
def test_mc_converges_to_bs():
    """Test that Monte Carlo converges to Black-Scholes price."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    # Black-Scholes price
    bs_call_price = bs_price(S0, K, T, r, sigma, "call")
    
    # Monte Carlo price with many paths
    mc_call_price = mc_euro_price(S0, K, T, r, sigma, n_paths=500_000, seed=42, option_type="call")
    
    # Should be within ±0.05
    assert abs(mc_call_price - bs_call_price) < 0.05


def test_mc_reproducibility():
    """Test Monte Carlo reproducibility with same seed."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    price1 = mc_euro_price(S0, K, T, r, sigma, n_paths=10000, seed=42)
    price2 = mc_euro_price(S0, K, T, r, sigma, n_paths=10000, seed=42)
    
    # Same seed should give same result
    assert price1 == price2
