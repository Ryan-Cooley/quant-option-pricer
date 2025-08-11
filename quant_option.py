#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import datetime
from numba import njit
from pandas_datareader import data as pdr
from typing import Optional, Literal

OptionType = Literal["call", "put"]


# ── Data & Volatility ──────────────────────────────────────────────────────────
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads Close prices for the given ticker and date range.
    Tries Stooq first, then yfinance as fallback.
    Forward-fills missing data, sets Date as index, and returns a DataFrame with a
    'Close' column. Prints/logs which data source was used. Raises ValueError if
    ticker is invalid or no data is returned.
    """
    # Try Stooq first
    try:
        df = pdr.get_data_stooq(ticker, start=start, end=end)
        if not df.empty:
            print(f"Fetched data for {ticker} from Stooq.")
            df = df[["Close"]]
            df = df.ffill()
            df.index.name = "Date"
            # stooq returns data in descending order
            df = df.sort_index()
            return df
        else:
            print("Stooq returned empty data for ticker.")
    except Exception as e:
        print(f"Stooq failed for {ticker}: {e}")
    # Fallback to yfinance
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(
                f"No data found for ticker '{ticker}' in the given date range \
                from yfinance."
            )
        # Handle multi-index columns (new yfinance default)
        if isinstance(data.columns, pd.MultiIndex):
            if ("Close", ticker) in data.columns:
                close = data[("Close", ticker)]
            else:
                raise ValueError(
                    f"'Close' column not found for ticker '{ticker}' in yfinance data."
                )
        else:
            if "Close" in data.columns:
                close = data["Close"]
            else:
                raise ValueError("'Close' column not found in yfinance data.")
        df = pd.DataFrame({"Close": close}).ffill()
        df.index.name = "Date"
        print(f"Fetched data for {ticker} from yfinance.")
        return df
    except Exception as e:
        print(f"yfinance failed for {ticker}: {e}")
    raise ValueError("Failed to fetch data for ticker from all available sources.")


def period_to_dates(period: str):
    """
    Convert a period string like '1y', '6mo', '3d' to start and end dates.
    """
    end = datetime.datetime.today()
    if period.endswith("y"):
        start = end - pd.DateOffset(years=int(period[:-1]))
    elif period.endswith("mo"):
        start = end - pd.DateOffset(months=int(period[:-2]))
    elif period.endswith("d"):
        start = end - pd.DateOffset(days=int(period[:-1]))
    else:
        raise ValueError(f"Unknown period format: {period}")
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def download_log_returns(
    ticker: str, period: str = "1y", csv_path: Optional[str] = None, retries: int = 3
):
    """
    Download Close prices and compute daily log-returns.
    If csv_path is provided, read from CSV instead of hitting data sources.
    Tries Stooq first, then yfinance as fallback.
    """
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=True, index_col="Date")
    else:
        start, end = period_to_dates(period)
        df = fetch_data(ticker, start, end)
    price_col = "Close"
    df["LogReturn"] = np.log(df[price_col] / df[price_col].shift(1))
    return df["LogReturn"].dropna()


def annualized_vol(returns: np.ndarray, trading_days: int = 252):
    """Estimate annualized volatility from daily log-returns."""
    return returns.std(ddof=1) * np.sqrt(trading_days)


def plot_and_save_returns(returns, ticker, out_dir):
    """Plot and save daily log-returns as a PNG."""
    plt.figure()
    returns.plot(title=f"{ticker} Daily Log-Returns")
    plt.xlabel("Date")
    plt.ylabel("Log-Return")
    plt.tight_layout()
    out = os.path.join(out_dir, f"{ticker}_returns.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved returns plot → {out}")


# ── Black–Scholes & Greeks ──────────────────────────────────────────────────────
def black_scholes(S0, K, r, sigma, T, option_type: OptionType):
    """Analytical Black-Scholes price for a European option."""
    if T == 0:
        return max(0, S0 - K) if option_type == "call" else max(0, K - S0)
    if sigma == 0:
        val = (
            S0 * np.exp(r * T) - K if option_type == "call" else K - S0 * np.exp(r * T)
        )
        return max(0, val) * np.exp(-r * T)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S0, K, r, sigma, T, option_type: OptionType):
    """Analytical Delta of a European option."""
    S0, K, r, sigma, T = np.broadcast_arrays(S0, K, r, sigma, T)
    mask = (sigma == 0) | (T == 0)
    out = np.empty_like(S0, dtype=float)

    if option_type == "call":
        out[mask] = np.where(S0[mask] > K[mask], 1.0, 0.0)
    else:  # put
        out[mask] = np.where(S0[mask] < K[mask], -1.0, 0.0)

    if np.any(~mask):
        S0_, K_, r_, sigma_, T_ = (
            S0[~mask],
            K[~mask],
            r[~mask],
            sigma[~mask],
            T[~mask],
        )
        d1 = (np.log(S0_ / K_) + (r_ + 0.5 * sigma_**2) * T_) / (sigma_ * np.sqrt(T_))
        if option_type == "call":
            out[~mask] = norm.cdf(d1)
        else:  # put
            out[~mask] = norm.cdf(d1) - 1.0
    return out.item() if out.shape == () else out


def bs_vega(S0, K, r, sigma, T):
    """Analytical Vega of a European option (same for call/put)."""
    S0, K, r, sigma, T = np.broadcast_arrays(S0, K, r, sigma, T)
    mask = (sigma == 0) | (T == 0)
    out = np.empty_like(S0, dtype=float)
    out[mask] = 0.0
    if np.any(~mask):
        S0_, K_, r_, sigma_, T_ = (
            S0[~mask],
            K[~mask],
            r[~mask],
            sigma[~mask],
            T[~mask],
        )
        d1 = (np.log(S0_ / K_) + (r_ + 0.5 * sigma_**2) * T_) / (sigma_ * np.sqrt(T_))
        out[~mask] = S0_ * norm.pdf(d1) * np.sqrt(T_)
    return out.item() if out.shape == () else out


# ── Monte Carlo with Numba ─────────────────────────────────────────────────────
@njit
def simulate_gbm_numba(S0, K, r, sigma, T, steps, n_paths, seed, option_type_is_call):
    """
    Simulate n_paths of GBM and return the discounted mean payoff.
    Uses log-Euler discretization. Numba-accelerated for speed.
    """
    np.random.seed(seed)
    dt = T / steps
    payoff_sum = 0.0
    random_numbers = np.random.normal(0.0, 1.0, (n_paths, steps))
    for i in range(n_paths):
        logS = 0.0
        for j in range(steps):
            z = random_numbers[i, j]
            logS += (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        ST = S0 * np.exp(logS)
        if option_type_is_call:
            payoff = max(ST - K, 0.0)
        else:
            payoff = max(K - ST, 0.0)
        payoff_sum += payoff
    return np.exp(-r * T) * (payoff_sum / n_paths)


def monte_carlo_price(
    S0, K, r, sigma, T, steps, n_paths, seed, option_type: OptionType
):
    """Wrapper for the Numba-accelerated MC pricer."""
    return simulate_gbm_numba(
        S0, K, r, sigma, T, steps, n_paths, seed, option_type == "call"
    )


def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed):
    """Return array of simulated terminal prices (n_paths,)."""
    np.random.seed(seed)
    dt = T / steps
    logS = np.log(S0) + np.cumsum(
        (r - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * np.random.randn(n_paths, steps),
        axis=1,
    )
    return np.exp(logS[:, -1])


def mc_greek_bump(func, param_name, bump, *args, **kwargs):
    """Estimate a Greek via central finite difference."""
    # Create a dictionary of the function's arguments
    arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
    params = dict(zip(arg_names, args))
    params.update(kwargs)

    # Bump up
    up_params = params.copy()
    up_params[param_name] += bump
    up = func(**up_params)

    # Bump down
    down_params = params.copy()
    down_params[param_name] -= bump
    down = func(**down_params)

    return (up - down) / (2 * bump)


def compute_var_cvar(pnl, alpha=0.05):
    """Compute VaR and CVaR at confidence level alpha."""
    var_level = np.percentile(pnl, alpha * 100)
    var = -var_level
    tail = pnl[pnl <= var_level]
    cvar = -tail.mean() if len(tail) > 0 else var
    return var, cvar


def enhanced_risk_analysis(pnl, payoff, bs_price, option_type="call", alpha=0.05):
    """
    Enhanced risk analysis with comprehensive metrics.

    Returns a dictionary with detailed risk metrics and explanations.
    """
    var, cvar = compute_var_cvar(pnl, alpha)

    # Calculate additional risk metrics
    prob_worthless = (payoff == 0).mean()
    expected_pnl = pnl.mean()
    pnl_std = pnl.std()
    max_loss = pnl.min()
    max_gain = pnl.max()

    # Multiple confidence levels
    confidence_levels = [0.01, 0.05, 0.10, 0.25]
    var_cvar_multiple = {}
    for conf_level in confidence_levels:
        var_conf, cvar_conf = compute_var_cvar(pnl, conf_level)
        var_cvar_multiple[conf_level] = (var_conf, cvar_conf)

    # Explanation of VaR/CVaR behavior
    explanation = {
        "var_equals_cvar": var == cvar,
        "prob_worthless_greater_than_alpha": prob_worthless > alpha,
        "max_loss_equals_bs_price": abs(max_loss + bs_price) < 1e-6,
        "reason": "At-the-money options have ~50% probability of expiring worthless, "
        "making the 5th percentile loss equal to the maximum loss (option premium).",
    }

    return {
        "var": var,
        "cvar": cvar,
        "prob_worthless": prob_worthless,
        "expected_pnl": expected_pnl,
        "pnl_std": pnl_std,
        "max_loss": max_loss,
        "max_gain": max_gain,
        "var_cvar_multiple": var_cvar_multiple,
        "explanation": explanation,
    }


def print_enhanced_risk_analysis(
    risk_metrics, option_type="call", S0=150, K=150, r=0.01, sigma=0.25, T=1.0
):
    """Print formatted enhanced risk analysis results."""
    print("=== ENHANCED RISK ANALYSIS ===")
    print(f"Option Type: {option_type.title()}")
    print(f"Parameters: S₀={S0}, K={K}, r={r:.1%}, σ={sigma:.1%}, T={T}")
    print()

    print("Risk Metrics (95% confidence):")
    print(f"VaR (5%): ${risk_metrics['var']:.4f}")
    print(f"CVaR (5%): ${risk_metrics['cvar']:.4f}")
    print()

    print("Detailed Risk Analysis:")
    print(f"Probability of Expiring Worthless: {risk_metrics['prob_worthless']:.1%}")
    print(f"Expected P&L: ${risk_metrics['expected_pnl']:.4f}")
    print(f"P&L Standard Deviation: ${risk_metrics['pnl_std']:.4f}")
    print(f"Maximum Loss: ${risk_metrics['max_loss']:.4f}")
    print(f"Maximum Gain: ${risk_metrics['max_gain']:.4f}")
    print()

    print("VaR/CVaR at Different Confidence Levels:")
    for alpha, (var_alpha, cvar_alpha) in risk_metrics["var_cvar_multiple"].items():
        print(f"  {alpha*100:.0f}%: VaR=${var_alpha:.4f}, CVaR=${cvar_alpha:.4f}")
    print()

    if risk_metrics["explanation"]["var_equals_cvar"]:
        print("⚠️  VaR = CVaR Explanation:")
        print(f"   {risk_metrics['explanation']['reason']}")
        print(
            "   This is mathematically correct but not very informative for "
            "risk analysis."
        )
        print(
            "   Consider using out-of-the-money options or different confidence levels."
        )
        print()


def plot_enhanced_risk_analysis(pnl, payoff, risk_metrics, option_type="call"):
    """Create enhanced risk analysis plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # P&L distribution
    ax1.hist(pnl, bins=50, color="skyblue", edgecolor="k", alpha=0.7, density=True)
    ax1.axvline(
        -risk_metrics["var"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"VaR (5%): ${-risk_metrics['var']:.2f}",
    )
    ax1.axvline(
        -risk_metrics["cvar"],
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"CVaR (5%): ${-risk_metrics['cvar']:.2f}",
    )
    ax1.axvline(
        risk_metrics["expected_pnl"],
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean: ${risk_metrics['expected_pnl']:.2f}",
    )
    ax1.set_xlabel("P&L at Expiry")
    ax1.set_ylabel("Density")
    ax1.set_title(f"P&L Distribution for {option_type.title()} Option")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Payoff distribution
    ax2.hist(
        payoff, bins=30, color="lightgreen", edgecolor="k", alpha=0.7, density=True
    )
    ax2.axvline(
        payoff.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean Payoff: ${payoff.mean():.2f}",
    )
    ax2.set_xlabel("Payoff at Expiry")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Payoff Distribution for {option_type.title()} Option")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Cumulative P&L distribution
    sorted_pnl = np.sort(pnl)
    cumulative_prob = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
    ax3.plot(sorted_pnl, cumulative_prob, linewidth=2, color="blue")
    ax3.axhline(0.05, color="red", linestyle="--", label="5% threshold")
    ax3.axvline(
        -risk_metrics["var"],
        color="red",
        linestyle=":",
        label=f"VaR: ${-risk_metrics['var']:.2f}",
    )
    ax3.set_xlabel("P&L")
    ax3.set_ylabel("Cumulative Probability")
    ax3.set_title("Cumulative P&L Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Explanation of VaR/CVaR behavior
    ax4.text(0.1, 0.8, "Why VaR = CVaR = BS Price:", fontsize=12, fontweight="bold")
    ax4.text(
        0.1,
        0.7,
        f'• ~{risk_metrics["prob_worthless"]:.1%} of paths expire worthless',
        fontsize=10,
    )
    ax4.text(0.1, 0.6, "• 5th percentile loss = maximum loss", fontsize=10)
    ax4.text(0.1, 0.5, f'• VaR (5%) = -${risk_metrics["var"]:.2f}', fontsize=10)
    ax4.text(0.1, 0.4, f'• CVaR (5%) = -${risk_metrics["cvar"]:.2f}', fontsize=10)
    ax4.text(0.1, 0.3, "• This is mathematically correct", fontsize=10)
    ax4.text(0.1, 0.2, "• but not very informative", fontsize=10)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")
    ax4.set_title("VaR/CVaR Explanation")

    plt.tight_layout()
    return fig


def plot_pnl_histogram(pnl, var, cvar, out_dir, option_type: OptionType):
    """Plot histogram of P&L with VaR and CVaR marked."""
    plt.figure(figsize=(8, 5))
    plt.hist(pnl, bins=50, color="skyblue", edgecolor="k", alpha=0.7)
    plt.axvline(-var, color="red", linestyle="--", label=f"VaR (5%): {-var:.2f}")
    plt.axvline(-cvar, color="purple", linestyle=":", label=f"CVaR (5%): {-cvar:.2f}")
    plt.xlabel("P&L at Expiry")
    plt.ylabel("Frequency")
    plt.title(f"P&L Distribution for {option_type.title()} Option (MC)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "pnl_histogram.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved P&L histogram with VaR/CVaR → {out}")


def plot_greek_surface(greek_fn, var_name, grid_S, grid_sigma, args, out_dir):
    """3D surface of a given analytic Greek over (S, sigma)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    S_grid, sigma_grid = np.meshgrid(grid_S, grid_sigma)
    if var_name == "Vega":
        Z = greek_fn(S_grid, args.K, args.r, sigma_grid, args.T)
    else:
        Z = greek_fn(S_grid, args.K, args.r, sigma_grid, args.T, args.option_type)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S_grid, sigma_grid, Z, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Spot (S₀)")
    ax.set_ylabel("Volatility (σ)")
    ax.set_zlabel(var_name)
    plt.title(f"{var_name} Surface for {args.option_type.title()} Option")
    plt.tight_layout()
    out = os.path.join(out_dir, f"{var_name.lower()}_surface.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {var_name} surface → {out}")


# ── Convergence Plot ────────────────────────────────────────────────────────────
def plot_convergence(
    S0, K, r, sigma, T, steps, path_counts, seed, out_dir, option_type: OptionType
):
    """Plot MC price convergence vs. Black-Scholes analytical price."""
    bs = black_scholes(S0, K, r, sigma, T, option_type)
    estimates = [
        monte_carlo_price(S0, K, r, sigma, T, steps, n, seed + i, option_type)
        for i, n in enumerate(path_counts)
    ]
    print("\n--- Convergence Analysis ---")
    print(f"Black-Scholes {option_type} price: {bs:.4f}")
    for n_paths, estimate in zip(path_counts, estimates):
        error = abs(estimate - bs) / bs * 100 if bs != 0 else 0
        print(f"{n_paths:,} paths: {estimate:.4f} (error: {error:.2f}%)")

    plt.figure(figsize=(10, 6))
    plt.plot(path_counts, estimates, "o-", label="MC estimate")
    plt.hlines(
        bs,
        path_counts[0],
        path_counts[-1],
        linestyles="--",
        label=f"Black–Scholes: {bs:.4f}",
        colors="red",
    )
    plt.xscale("log")
    plt.xlabel("Number of paths")
    plt.ylabel("Option price")
    plt.title(
        f"MC Convergence for {option_type.title()} (S₀={S0}, K={K}, σ={sigma:.1%})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "convergence_v2.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved convergence plot → {out}")


# ── CLI & Main ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Quant Option Pricer w/ Greeks: MC vs Black–Scholes 🎯"
    )
    p.add_argument(
        "--ticker", type=str, default="AAPL", help="Ticker for volatility estimation"
    )
    p.add_argument(
        "--option-type",
        type=str,
        default="call",
        choices=["call", "put"],
        help="Type of option to price",
    )
    p.add_argument("--S0", type=float, default=150.0, help="Spot price")
    p.add_argument("--K", type=float, default=150.0, help="Strike price")
    p.add_argument("--r", type=float, default=0.01, help="Risk-free rate")
    p.add_argument("--T", type=float, default=1.0, help="Time to expiry (years)")
    p.add_argument("--steps", type=int, default=252, help="Time steps per path")
    p.add_argument("--paths", type=int, default=100_000, help="Number of MC paths")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--outdir", type=str, default="plots", help="Directory to save PNGs")
    p.add_argument("--csv", type=str, default=None, help="Local CSV path as fallback")
    p.add_argument(
        "--greeks",
        action="store_true",
        default=True,
        help="Compute and print Delta/Vega (analytic & MC, default: on)",
    )
    p.add_argument(
        "--no-greeks",
        dest="greeks",
        action="store_false",
        help="Disable Greek calculation",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Volatility estimation
    try:
        returns = download_log_returns(args.ticker, csv_path=args.csv)
        sigma = annualized_vol(returns)
        print(f"Est. annualized σ for {args.ticker}: {sigma:.2%}")
        plot_and_save_returns(returns, args.ticker, args.outdir)
    except Exception as e:
        print(f"Error fetching returns: {e}")
        return

    # 2. Pricing
    bs_price = black_scholes(args.S0, args.K, args.r, sigma, args.T, args.option_type)
    mc_price = monte_carlo_price(
        args.S0,
        args.K,
        args.r,
        sigma,
        args.T,
        args.steps,
        args.paths,
        args.seed,
        args.option_type,
    )
    print(f"\n--- Pricing Results ({args.option_type.title()}) ---")
    print(f"Black–Scholes price:    {bs_price:.4f}")
    print(f"Monte Carlo price ({args.paths:,} paths): {mc_price:.4f}")

    # 3. Greeks (optional)
    if args.greeks:
        print("\n--- Greeks ---")
        analytic_delta = bs_delta(
            args.S0, args.K, args.r, sigma, args.T, args.option_type
        )
        analytic_vega = bs_vega(args.S0, args.K, args.r, sigma, args.T)

        eps_S = args.S0 * 1e-4 if args.S0 != 0 else 1e-4
        mc_delta = mc_greek_bump(
            monte_carlo_price,
            "S0",
            eps_S,
            S0=args.S0,
            K=args.K,
            r=args.r,
            sigma=sigma,
            T=args.T,
            steps=args.steps,
            n_paths=args.paths,
            seed=args.seed,
            option_type=args.option_type,
        )

        eps_sigma = sigma * 1e-4 if sigma != 0 else 1e-4
        mc_vega = mc_greek_bump(
            monte_carlo_price,
            "sigma",
            eps_sigma,
            S0=args.S0,
            K=args.K,
            r=args.r,
            sigma=sigma,
            T=args.T,
            steps=args.steps,
            n_paths=args.paths,
            seed=args.seed,
            option_type=args.option_type,
        )
        print(f"Analytic Δ:       {analytic_delta:.4f}")
        print(f"MC Δ estimate:    {mc_delta:.4f}")
        print(f"Analytic Vega:    {analytic_vega:.4f}")
        print(f"MC Vega estimate: {mc_vega:.4f}")

    # 4. Convergence plot
    path_counts = sorted(set([1_000, 5_000, 10_000, 50_000, 100_000, args.paths]))
    plot_convergence(
        args.S0,
        args.K,
        args.r,
        sigma,
        args.T,
        args.steps,
        path_counts,
        args.seed,
        args.outdir,
        args.option_type,
    )

    # 5. Enhanced Risk Analysis
    ST = simulate_gbm_paths(
        args.S0, args.r, sigma, args.T, args.steps, args.paths, args.seed
    )
    if args.option_type == "call":
        payoff = np.maximum(ST - args.K, 0)
    else:
        payoff = np.maximum(args.K - ST, 0)

    pnl = np.exp(-args.r * args.T) * payoff - bs_price

    # Enhanced risk analysis
    risk_metrics = enhanced_risk_analysis(pnl, payoff, bs_price, args.option_type)
    print_enhanced_risk_analysis(
        risk_metrics, args.option_type, args.S0, args.K, args.r, sigma, args.T
    )

    # Create enhanced risk analysis plots
    plot_enhanced_risk_analysis(pnl, payoff, risk_metrics, args.option_type)
    plt.savefig(
        os.path.join(args.outdir, "enhanced_risk_analysis.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Also create the original P&L histogram for compatibility
    plot_pnl_histogram(
        pnl, risk_metrics["var"], risk_metrics["cvar"], args.outdir, args.option_type
    )

    # 6. Greek surface plots
    if args.greeks:
        grid_S = np.linspace(0.8 * args.S0, 1.2 * args.S0, 50)
        grid_sigma = np.linspace(0.5 * sigma, 1.5 * sigma, 50)
        plot_greek_surface(bs_delta, "Delta", grid_S, grid_sigma, args, args.outdir)
        plot_greek_surface(bs_vega, "Vega", grid_S, grid_sigma, args, args.outdir)

    print("\nAll results and plots saved. Project complete!")


if __name__ == "__main__":
    main()
