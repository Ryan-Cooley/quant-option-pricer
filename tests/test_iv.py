"""
Tests for the implied volatility module.

Tests Black-Scholes implied volatility calculation with robust root-finding.
"""

import pytest
import numpy as np
from src.quant.iv import (
    _bounds_call_put,
    implied_vol_bs,
)
from src.quant.pricing import bs_price


def test_bounds_call_put_basic():
    """Test no-arbitrage bounds calculation."""
    S, K, T, r = 100, 100, 1.0, 0.05

    # Call option bounds
    lower_call, upper_call = _bounds_call_put(S, K, T, r, "call")
    expected_lower_call = max(0, S - K * np.exp(-r * T))
    expected_upper_call = S

    assert abs(lower_call - expected_lower_call) < 1e-10
    assert abs(upper_call - expected_upper_call) < 1e-10

    # Put option bounds
    lower_put, upper_put = _bounds_call_put(S, K, T, r, "put")
    expected_lower_put = max(0, K * np.exp(-r * T) - S)
    expected_upper_put = K * np.exp(-r * T)

    assert abs(lower_put - expected_lower_put) < 1e-10
    assert abs(upper_put - expected_upper_put) < 1e-10


def test_bounds_call_put_zero_time():
    """Test bounds at expiry."""
    S, K, T, r = 100, 105, 0.0, 0.05

    # Call option at expiry
    lower_call, upper_call = _bounds_call_put(S, K, T, r, "call")
    expected_lower_call = max(0, S - K)  # 0 since S < K
    expected_upper_call = S

    assert abs(lower_call - expected_lower_call) < 1e-10
    assert abs(upper_call - expected_upper_call) < 1e-10

    # Put option at expiry
    lower_put, upper_put = _bounds_call_put(S, K, T, r, "put")
    expected_lower_put = max(0, K - S)  # K - S since S < K
    expected_upper_put = K

    assert abs(lower_put - expected_lower_put) < 1e-10
    assert abs(upper_put - expected_upper_put) < 1e-10


def test_round_trip_price_iv():
    """Test price → IV → price round trip."""
    S, K, T, r, sigma_true = 100, 100, 0.5, 0.01, 0.25

    # Calculate price with true volatility
    price = bs_price(S, K, T, r, sigma_true, "call")

    # Calculate implied volatility
    sigma_implied = implied_vol_bs(price, S, K, T, r, "call")

    # Calculate price with implied volatility
    price_reconstructed = bs_price(S, K, T, r, sigma_implied, "call")

    # Check round-trip accuracy
    assert abs(price - price_reconstructed) < 1e-6
    assert abs(sigma_true - sigma_implied) < 0.1


def test_put_call_both_work():
    """Test that both put and call implied volatility work correctly."""
    S, K, T, r, sigma_true = 100, 105, 1.0, 0.05, 0.2

    # Test call option
    call_price = bs_price(S, K, T, r, sigma_true, "call")
    call_iv = implied_vol_bs(call_price, S, K, T, r, "call")
    assert abs(sigma_true - call_iv) < 0.1

    # Test put option
    put_price = bs_price(S, K, T, r, sigma_true, "put")
    put_iv = implied_vol_bs(put_price, S, K, T, r, "put")
    assert abs(sigma_true - put_iv) < 0.1


def test_bounds_violation_raises():
    """Test that price below intrinsic raises ValueError."""
    S, K, T, r = 100, 105, 1.0, 0.05

    # Calculate intrinsic value
    lower_call, _ = _bounds_call_put(S, K, T, r, "call")

    # Try to find IV for price below intrinsic
    price_below_intrinsic = lower_call - 1.0

    with pytest.raises(ValueError, match="violates no-arbitrage bounds"):
        implied_vol_bs(price_below_intrinsic, S, K, T, r, "call")


def test_price_above_upper_bound_raises():
    """Test that price above upper bound raises ValueError."""
    S, K, T, r = 100, 105, 1.0, 0.05

    # Calculate upper bound
    _, upper_call = _bounds_call_put(S, K, T, r, "call")

    # Try to find IV for price above upper bound
    price_above_upper = upper_call + 1.0

    with pytest.raises(ValueError, match="violates no-arbitrage bounds"):
        implied_vol_bs(price_above_upper, S, K, T, r, "call")


def test_zero_time_returns_zero():
    """Test that T <= 0 returns 0.0."""
    S, K, r = 100, 100, 0.05

    # T = 0
    iv = implied_vol_bs(0.0, S, K, 0.0, r, "call")
    assert iv == 0.0

    # T < 0
    iv = implied_vol_bs(0.0, S, K, -0.1, r, "call")
    assert iv == 0.0


def test_intrinsic_price_returns_zero():
    """Test that price at intrinsic value returns 0.0."""
    S, K, T, r = 100, 105, 1.0, 0.05

    # Calculate intrinsic value
    lower_call, _ = _bounds_call_put(S, K, T, r, "call")

    # Price at intrinsic should return 0.0
    iv = implied_vol_bs(lower_call, S, K, T, r, "call")
    assert iv == 0.0

    # Price very close to intrinsic should also return 0.0
    iv = implied_vol_bs(lower_call + 1e-13, S, K, T, r, "call")
    assert iv == 0.0


def test_high_volatility():
    """Test implied volatility calculation for high volatility."""
    S, K, T, r, sigma_true = 100, 100, 1.0, 0.05, 1.0  # 100% volatility

    price = bs_price(S, K, T, r, sigma_true, "call")
    sigma_implied = implied_vol_bs(price, S, K, T, r, "call")

    assert abs(sigma_true - sigma_implied) < 0.5


def test_low_volatility():
    """Test implied volatility calculation for low volatility."""
    S, K, T, r, sigma_true = 100, 100, 1.0, 0.05, 0.01  # 1% volatility

    price = bs_price(S, K, T, r, sigma_true, "call")
    sigma_implied = implied_vol_bs(price, S, K, T, r, "call")

    assert abs(sigma_true - sigma_implied) < 0.1


def test_different_moneyness():
    """Test implied volatility for different moneyness levels."""
    S, K, T, r, sigma_true = 100, 100, 1.0, 0.05, 0.2

    # ATM
    price_atm = bs_price(S, K, T, r, sigma_true, "call")
    iv_atm = implied_vol_bs(price_atm, S, K, T, r, "call")
    assert abs(sigma_true - iv_atm) < 0.1

    # ITM
    price_itm = bs_price(S, K * 0.9, T, r, sigma_true, "call")
    iv_itm = implied_vol_bs(price_itm, S, K * 0.9, T, r, "call")
    assert abs(sigma_true - iv_itm) < 0.5

    # OTM
    price_otm = bs_price(S, K * 1.1, T, r, sigma_true, "call")
    iv_otm = implied_vol_bs(price_otm, S, K * 1.1, T, r, "call")
    assert abs(sigma_true - iv_otm) < 0.1


def test_put_call_parity_iv():
    """Test that put-call parity holds for implied volatilities."""
    S, K, T, r, sigma_true = 100, 105, 1.0, 0.05, 0.2

    # Calculate prices
    call_price = bs_price(S, K, T, r, sigma_true, "call")
    put_price = bs_price(S, K, T, r, sigma_true, "put")

    # Calculate implied volatilities
    call_iv = implied_vol_bs(call_price, S, K, T, r, "call")
    put_iv = implied_vol_bs(put_price, S, K, T, r, "put")

    # Put-call parity should hold for IVs
    assert abs(call_iv - put_iv) < 1e-6


def test_custom_bracket():
    """Test implied volatility with custom bracket."""
    S, K, T, r, sigma_true = 100, 100, 1.0, 0.05, 0.3

    price = bs_price(S, K, T, r, sigma_true, "call")

    # Use custom bracket
    custom_bracket = (0.1, 2.0)
    sigma_implied = implied_vol_bs(price, S, K, T, r, "call", bracket=custom_bracket)

    assert abs(sigma_true - sigma_implied) < 0.1


def test_invalid_bracket_raises():
    """Test that invalid bracket raises ValueError."""
    S, K, T, r = 100, 100, 1.0, 0.05
    price = 10.0  # Valid price

    # Invalid brackets
    invalid_brackets = [
        (0.0, 1.0),  # a <= 0
        (-1.0, 1.0),  # a <= 0
        (1.0, 0.0),  # b <= 0
        (1.0, 1.0),  # a >= b
        (2.0, 1.0),  # a > b
    ]

    for bracket in invalid_brackets:
        with pytest.raises(ValueError, match="Invalid bracket"):
            implied_vol_bs(price, S, K, T, r, "call", bracket=bracket)


def test_reproducibility():
    """Test that implied volatility calculation is reproducible."""
    S, K, T, r = 100, 100, 1.0, 0.05
    price = 10.0

    # Calculate IV twice
    iv1 = implied_vol_bs(price, S, K, T, r, "call")
    iv2 = implied_vol_bs(price, S, K, T, r, "call")

    assert abs(iv1 - iv2) < 1e-10
