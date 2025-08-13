from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .pricing import bs_price, bs_delta


@dataclass
class HedgeResult:
    pnl_paths: np.ndarray  # (n_paths,)
    summary: dict  # mean, std, sharpe_annualized, p5,p50,p95
    terminal_inventory: np.ndarray  # (n_paths,)
    notes: dict  # dt, steps, fees, etc.


def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42) -> np.ndarray:
    """Return S with shape (steps+1, n_paths) via log-Euler."""
    np.random.seed(seed)

    dt = T / steps

    # Generate random numbers for all paths and steps
    Z = np.random.normal(0, 1, (n_paths, steps))

    # Log-Euler discretization
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    # Simulate log-returns
    log_returns = drift + diffusion * Z

    # Initialize price paths
    S = np.zeros((steps + 1, n_paths))
    S[0, :] = S0

    # Build price paths
    for t in range(1, steps + 1):
        S[t, :] = S[t - 1, :] * np.exp(log_returns[:, t - 1])

    return S


def simulate_delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    steps_per_year: int = 252,
    rebalance_every: int = 1,  # in steps; 1=daily for 252
    fee_bps: float = 0.0,  # proportional fee on traded notional
    fixed_fee: float = 0.0,  # per nonzero trade
    seed: int = 42,
    units: str = "s0",  # denominator for TE/Cost: "s0" or "premium"
) -> HedgeResult:
    """
    Short 1 option at t0 for BS price; replicate with Δ_t shares.
    Rebalance on grid; fees applied; cash accrues at r. Return P&L at T.
    """
    # Calculate simulation parameters
    steps = max(1, int(round(T * steps_per_year)))
    dt = T / steps

    # Simulate price paths
    S = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed)

    # Initialize arrays
    cash = np.zeros((steps + 1, n_paths))
    inventory = np.zeros((steps + 1, n_paths))
    delta = np.zeros((steps + 1, n_paths))

    # Initial setup: short option, receive premium
    option_premium = bs_price(S0, K, T, r, sigma, option_type)
    cash[0, :] = option_premium
    delta[0, :] = bs_delta(S0, K, T, r, sigma, option_type)

    # Initial trade to establish delta position
    initial_shares = delta[0, :]
    inventory[0, :] = initial_shares

    # Calculate initial fees
    initial_notional = np.abs(initial_shares) * S0
    initial_prop_fees = initial_notional * fee_bps / 10000
    initial_fixed_fees = fixed_fee * (initial_shares != 0)
    initial_total_fees = initial_prop_fees + initial_fixed_fees
    cash[0, :] -= initial_total_fees

    # Main hedging loop
    for t in range(1, steps + 1):
        # Cash accrues interest
        cash[t, :] = cash[t - 1, :] * np.exp(r * dt)
        inventory[t, :] = inventory[t - 1, :]
        delta[t, :] = delta[t - 1, :]

        # Rebalance if needed
        if t % rebalance_every == 0:
            # Calculate remaining time to expiry
            tau = T - t * dt

            # Calculate target delta at current prices
            target_delta = np.zeros(n_paths)
            for i in range(n_paths):
                if tau > 0:
                    target_delta[i] = bs_delta(S[t, i], K, tau, r, sigma, option_type)
                else:
                    # At expiry, delta is step function
                    if option_type == "call":
                        target_delta[i] = 1.0 if S[t, i] > K else 0.0
                    else:  # put
                        target_delta[i] = -1.0 if S[t, i] < K else 0.0

            # Calculate required trade
            d_shares = target_delta - inventory[t, :]

            # Execute trade
            inventory[t, :] = target_delta
            delta[t, :] = target_delta

            # Calculate fees
            notional = np.abs(d_shares) * S[t, :]
            prop_fees = notional * fee_bps / 10000
            fixed_fees = fixed_fee * (d_shares != 0)
            total_fees = prop_fees + fixed_fees

            # Update cash (pay for shares, pay fees)
            cash[t, :] -= d_shares * S[t, :] + total_fees

    # Calculate option payoff at expiry
    if option_type == "call":
        option_payoff = np.maximum(S[-1, :] - K, 0)
    else:  # put
        option_payoff = np.maximum(K - S[-1, :], 0)

    # Final P&L calculation
    # P&L = cash at expiry + value of inventory - option payoff
    pnl_paths = cash[-1, :] + inventory[-1, :] * S[-1, :] - option_payoff

    # Calculate summary statistics
    mean_pnl = np.mean(pnl_paths)
    std_pnl = np.std(pnl_paths)

    # Annualized Sharpe ratio (assuming daily rebalancing)
    if std_pnl > 0:
        sharpe_annualized = mean_pnl / std_pnl * np.sqrt(steps_per_year / T)
    else:
        sharpe_annualized = 0.0

    # Percentiles
    p5 = np.percentile(pnl_paths, 5)
    p50 = np.percentile(pnl_paths, 50)
    p95 = np.percentile(pnl_paths, 95)

    summary = {
        "mean": mean_pnl,
        "std": std_pnl,
        "sharpe_annualized": sharpe_annualized,
        "p5": p5,
        "p50": p50,
        "p95": p95,
    }

    # Calculate tracking error and cost metrics
    # Choose denominator based on units parameter
    denom = S0 if units == "s0" else option_premium

    # Tracking Error: RMSE between P&L and its mean (measures hedging effectiveness)
    te_bps = np.sqrt(np.mean((pnl_paths - mean_pnl) ** 2)) / denom * 10000

    # Cost: turnover-based proportional fees
    fee_rate = fee_bps / 10000
    cost_cash = abs(delta[0, 0]) * S[0, 0] * fee_rate  # Initial position cost

    # Add rebalancing costs
    for i in range(1, len(delta)):
        if i % rebalance_every == 0:
            # Use first path for deterministic cost
            d_delta = delta[i, 0] - delta[i - 1, 0]
            cost_cash += abs(d_delta) * S[i, 0] * fee_rate

    cost_bps = 10000 * (cost_cash / denom)

    metrics = {
        rebalance_every: {
            "te_bps": te_bps,
            "cost_bps": cost_bps,
        }
    }

    notes = {
        "dt": dt,
        "steps": steps,
        "rebalance_every": rebalance_every,
        "fee_bps": fee_bps,
        "fixed_fee": fixed_fee,
        "option_premium": option_premium,
        "initial_delta": delta[0, 0],
        "metrics": metrics,
    }

    return HedgeResult(
        pnl_paths=pnl_paths,
        summary=summary,
        terminal_inventory=inventory[-1, :],
        notes=notes,
    )
