"""
Black-Scholes and Monte Carlo option pricing functions.

Core pricing engine with analytical Black-Scholes formulas and
Monte Carlo simulation for European options.
"""

import math
import numpy as np
from typing import Literal, Tuple

OptionType = Literal["call", "put"]


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function using math.erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """
    Calculate d1 and d2 parameters for Black-Scholes formula.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Tuple of (d1, d2)
    """
    if T <= 0:
        return (
            float("inf") if S > K else float("-inf") if S < K else 0.0,
            float("inf") if S > K else float("-inf") if S < K else 0.0,
        )

    if sigma <= 0:
        # Zero volatility case
        forward = S * math.exp(r * T)
        if forward > K:
            d1 = float("inf")
            d2 = float("inf")
        elif forward < K:
            d1 = float("-inf")
            d2 = float("-inf")
        else:
            d1 = 0.0
            d2 = 0.0
        return d1, d2

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """
    Black-Scholes option price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"

    Returns:
        Option price
    """
    if T <= 0:
        # At expiry, return intrinsic value
        if option_type == "call":
            return max(0, S - K)
        else:  # put
            return max(0, K - S)

    if sigma <= 0:
        # Zero volatility case
        forward = S * math.exp(r * T)
        if option_type == "call":
            intrinsic = max(0, forward - K)
        else:  # put
            intrinsic = max(0, K - forward)
        return intrinsic * math.exp(-r * T)

    d1, d2 = d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        price = S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    else:  # put
        price = K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)

    return price


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """
    Black-Scholes delta (first derivative w.r.t. stock price).

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"

    Returns:
        Delta value
    """
    if T <= 0:
        # At expiry
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:  # put
            return -1.0 if S < K else 0.0

    if sigma <= 0:
        # Zero volatility case
        forward = S * math.exp(r * T)
        if option_type == "call":
            return 1.0 if forward > K else 0.0
        else:  # put
            return -1.0 if forward < K else 0.0

    d1, _ = d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        delta = normal_cdf(d1)
    else:  # put
        delta = normal_cdf(d1) - 1.0

    return delta


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes vega (first derivative w.r.t. volatility).
    Same for calls and puts.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Vega value
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1, _ = d1_d2(S, K, T, r, sigma)
    vega = S * normal_pdf(d1) * math.sqrt(T)
    return vega


def mc_euro_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    steps_per_year: int = 252,
    seed: int = 42,
    option_type: OptionType = "call",
) -> float:
    """
    Monte Carlo European option price using log-Euler discretization.

    Args:
        S0: Initial stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        n_paths: Number of simulation paths
        steps_per_year: Time steps per year
        seed: Random seed
        option_type: "call" or "put"

    Returns:
        Option price
    """
    if T <= 0:
        # At expiry, return intrinsic value
        if option_type == "call":
            return max(0, S0 - K)
        else:  # put
            return max(0, K - S0)

    if sigma <= 0:
        # Zero volatility case
        forward = S0 * math.exp(r * T)
        if option_type == "call":
            intrinsic = max(0, forward - K)
        else:  # put
            intrinsic = max(0, K - forward)
        return intrinsic * math.exp(-r * T)

    # Set random seed
    np.random.seed(seed)

    # Simulation parameters
    steps = int(T * steps_per_year)
    dt = T / steps

    # Generate random numbers for all paths and steps
    Z = np.random.normal(0, 1, (n_paths, steps))

    # Log-Euler discretization
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)

    # Simulate log-returns
    log_returns = drift + diffusion * Z
    log_paths = np.cumsum(log_returns, axis=1)

    # Calculate final stock prices
    ST = S0 * np.exp(log_paths[:, -1])

    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:  # put
        payoffs = np.maximum(K - ST, 0)

    # Discount and average
    price = math.exp(-r * T) * np.mean(payoffs)

    return price
