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
from typing import Optional


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
def black_scholes_call(S0, K, r, sigma, T):
    """Analytical Black-Scholes price for a European call option."""
    # Handle zero time case
    if T == 0:
        # With zero time, price is immediate payoff
        return max(0, S0 - K)

    # Handle zero volatility case
    if sigma == 0:
        # With zero volatility, price is discounted intrinsic value
        return max(0, S0 * np.exp(r * T) - K) * np.exp(-r * T)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S0, K, r, sigma, T):
    """Analytical Delta of a European call option (Black-Scholes).
    Supports scalar or array inputs."""
    S0, K, r, sigma, T = np.broadcast_arrays(S0, K, r, sigma, T)
    mask = (sigma == 0) | (T == 0)
    out = np.empty_like(S0, dtype=float)
    # Delta is 1 if in-the-money, 0 if out-of-the-money, for zero vol/time
    out[mask] = np.where(S0[mask] > K[mask], 1.0, 0.0)
    # For the rest, use Black-Scholes formula
    if np.any(~mask):
        S0_ = S0[~mask]
        K_ = K[~mask]
        r_ = r[~mask]
        sigma_ = sigma[~mask]
        T_ = T[~mask]
        d1 = (np.log(S0_ / K_) + (r_ + 0.5 * sigma_**2) * T_) / (sigma_ * np.sqrt(T_))
        out[~mask] = norm.cdf(d1)
    return out.item() if out.shape == () else out


def bs_vega(S0, K, r, sigma, T):
    """Analytical Vega of a European call option (Black-Scholes), per 1 vol unit.
    Supports scalar or array inputs."""
    S0, K, r, sigma, T = np.broadcast_arrays(S0, K, r, sigma, T)
    mask = (sigma == 0) | (T == 0)
    out = np.empty_like(S0, dtype=float)
    out[mask] = 0.0
    if np.any(~mask):
        S0_ = S0[~mask]
        K_ = K[~mask]
        r_ = r[~mask]
        sigma_ = sigma[~mask]
        T_ = T[~mask]
        d1 = (np.log(S0_ / K_) + (r_ + 0.5 * sigma_**2) * T_) / (sigma_ * np.sqrt(T_))
        out[~mask] = S0_ * norm.pdf(d1) * np.sqrt(T_)
    return out.item() if out.shape == () else out


# ── Monte Carlo with Numba ─────────────────────────────────────────────────────
@njit
def simulate_gbm_numba(S0, K, r, sigma, T, steps, n_paths, seed):
    """
    Simulate n_paths of GBM and return the discounted mean payoff for a European call.
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
        payoff = max(ST - K, 0.0)
        payoff_sum += payoff
    return np.exp(-r * T) * (payoff_sum / n_paths)


def monte_carlo_price(S0, K, r, sigma, T, steps, n_paths, seed):
    """Wrapper for the Numba-accelerated MC pricer."""
    return simulate_gbm_numba(S0, K, r, sigma, T, steps, n_paths, seed)


def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed):
    """Return array of simulated terminal prices (n_paths,)."""
    np.random.seed(seed)
    dt = T / steps
    logS = np.log(S0) + np.cumsum(
        (r - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * np.random.randn(n_paths, steps),
        axis=1,
    )
    ST = np.exp(logS[:, -1])
    return ST


def mc_greek_bump(func, param_name, bump, *args, **kwargs):
    """
    Estimate a Greek via central finite difference: (f(x+ε) - f(x-ε)) / (2ε).
    param_name: parameter to bump (e.g., 'S0', 'sigma')
    bump: bump size
    args: positional arguments to func
    kwargs: keyword arguments to func
    """
    params = dict(zip(func.__code__.co_varnames, args))
    up_args = [
        params[name] + bump if name == param_name else val
        for name, val in zip(func.__code__.co_varnames, args)
    ]
    down_args = [
        params[name] - bump if name == param_name else val
        for name, val in zip(func.__code__.co_varnames, args)
    ]
    up = func(*up_args, **kwargs)
    down = func(*down_args, **kwargs)
    return (up - down) / (2 * bump)


def compute_var_cvar(pnl, alpha=0.05):
    """
    Compute VaR and CVaR at confidence level alpha using percentile definitions.
    VaR = –P&L at the alpha-quantile
    CVaR = –mean(P&L below that quantile)
    """
    var_level = np.percentile(pnl, alpha * 100)
    var = -var_level
    tail = pnl[pnl <= var_level]
    cvar = -tail.mean() if len(tail) > 0 else var
    return var, cvar


def plot_pnl_histogram(pnl, var, cvar, out_dir):
    """Plot histogram of P&L with VaR and CVaR marked."""
    plt.figure(figsize=(8, 5))
    plt.hist(pnl, bins=50, color="skyblue", edgecolor="k", alpha=0.7)
    plt.axvline(-var, color="red", linestyle="--", label=f"VaR (5%): {-var:.2f}")
    plt.axvline(-cvar, color="purple", linestyle=":", label=f"CVaR (5%): {-cvar:.2f}")
    plt.xlabel("P&L at Expiry")
    plt.ylabel("Frequency")
    plt.title("P&L Distribution at Expiry (MC)")
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
    Z = greek_fn(S_grid, args.K, args.r, sigma_grid, args.T)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S_grid, sigma_grid, Z, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Spot (S₀)")
    ax.set_ylabel("Volatility (σ)")
    ax.set_zlabel(var_name)
    plt.title(f"{var_name} Surface")
    plt.tight_layout()
    out = os.path.join(out_dir, f"{var_name.lower()}_surface.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {var_name} surface → {out}")


# ── Convergence Plot ────────────────────────────────────────────────────────────
def plot_convergence(
    S0,
    K,
    r,
    sigma,
    T,
    steps,
    path_counts,
    seed,
    out_dir,
    plot_filename="convergence.png",
):
    """Plot MC price convergence vs. Black-Scholes analytical price."""
    bs = black_scholes_call(S0, K, r, sigma, T)
    estimates = [
        monte_carlo_price(S0, K, r, sigma, T, steps, n, seed + i)
        for i, n in enumerate(path_counts)
    ]
    # Print convergence details for debugging
    print("\n--- Convergence Analysis ---")
    print(f"Black-Scholes price: {bs:.4f}")
    for i, (n_paths, estimate) in enumerate(zip(path_counts, estimates)):
        error = abs(estimate - bs) / bs * 100
        print(f"{n_paths:,} paths: {estimate:.4f} (error: {error:.2f}%)")
    plt.figure(figsize=(10, 6))
    plt.plot(
        path_counts,
        estimates,
        "o-",
        label="MC estimate",
        linewidth=2,
        markersize=8,
    )
    plt.hlines(
        bs,
        path_counts[0],
        path_counts[-1],
        linestyles="--",
        label=(f"Black–Scholes: {bs:.4f}"),
        colors="red",
        linewidth=2,
    )
    plt.xscale("log")
    plt.xlabel("Number of paths")
    plt.ylabel("Option price")
    plt.title(f"MC Convergence vs Black–Scholes (S₀={S0}, K={K}, σ={sigma:.1%})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, plot_filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved convergence plot → {}".format(out))


def compute_key_metrics(pnl, delta, var, cvar):
    """
    Aggregate key risk and sensitivity metrics for display.
    """
    mean_pnl = np.mean(pnl)
    typical_delta = np.mean(delta) if hasattr(delta, "__len__") else delta
    return {
        "Mean P&L": mean_pnl,
        "VaR (5%)": var,
        "CVaR (5%)": cvar,
        "Δ per $1 move": typical_delta,
    }


# ── CLI & Main ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Quant Option Pricer w/ Greeks: MC vs Black–Scholes 🎯"
    )
    p.add_argument(
        "--ticker", type=str, default="AAPL", help="Ticker for volatility estimation"
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
    except Exception as e:
        print(f"Error fetching returns: {e}")
        return
    plot_and_save_returns(returns, args.ticker, args.outdir)
    sigma = annualized_vol(returns)
    print(f"Est. annualized σ for {args.ticker}: {sigma:.2%}")

    # 2. Pricing
    bs_price = black_scholes_call(args.S0, args.K, args.r, sigma, args.T)
    mc_price = monte_carlo_price(
        args.S0, args.K, args.r, sigma, args.T, args.steps, args.paths, args.seed
    )
    print(f"Black–Scholes call price:    {bs_price:.4f}")
    print(f"Monte Carlo call price ({args.paths:,} paths): {mc_price:.4f}")

    # 3. Greeks (optional)
    if args.greeks:
        print("\n--- Greeks ---")
        analytic_delta = bs_delta(args.S0, args.K, args.r, sigma, args.T)
        analytic_vega = bs_vega(args.S0, args.K, args.r, sigma, args.T)
        # MC Greeks via bump
        eps_S = args.S0 * 1e-4 if args.S0 != 0 else 1e-4
        eps_sigma = sigma * 1e-4 if sigma != 0 else 1e-4
        mc_delta = mc_greek_bump(
            monte_carlo_price,
            "S0",
            eps_S,
            args.S0,
            args.K,
            args.r,
            sigma,
            args.T,
            args.steps,
            args.paths,
            args.seed,
        )
        mc_vega = mc_greek_bump(
            monte_carlo_price,
            "sigma",
            eps_sigma,
            args.S0,
            args.K,
            args.r,
            sigma,
            args.T,
            args.steps,
            args.paths,
            args.seed,
        )
        print(f"Analytic Δ:       {analytic_delta:.4f}")
        print(f"MC Δ estimate:    {mc_delta:.4f}")
        print(f"Analytic Vega:    {analytic_vega:.4f}")
        print(f"MC Vega estimate: {mc_vega:.4f}")

    # 4. Convergence plot
    # Use a wide range of path counts for better convergence visualization
    path_counts = sorted(
        set([1_000, 5_000, 10_000, 50_000, 100_000, args.paths])
    )  # deduplicate
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
        plot_filename="convergence_v2.png",
    )

    # 5. VaR/CVaR and P&L histogram
    ST = simulate_gbm_paths(
        args.S0, args.r, sigma, args.T, args.steps, args.paths, args.seed
    )
    pnl = np.exp(-args.r * args.T) * np.maximum(ST - args.K, 0) - bs_price
    var, cvar = compute_var_cvar(pnl)
    print(f"VaR (5%):         {var:.4f}")
    print(f"CVaR (5%):        {cvar:.4f}")
    plot_pnl_histogram(pnl, var, cvar, args.outdir)

    # --- Key Results Table ---
    # Use MC Delta if available, else analytic
    delta_for_table = (
        mc_delta
        if "mc_delta" in locals()
        else bs_delta(args.S0, args.K, args.r, sigma, args.T)
    )
    key_metrics = compute_key_metrics(pnl, delta_for_table, var, cvar)
    print("\n--- Key Results ---")
    for k, v in key_metrics.items():
        print(f"{k:18s} {v:,.4f}")

    # 6. Greek surface plots
    grid_S = np.linspace(0.8 * args.S0, 1.2 * args.S0, 50)
    grid_sigma = np.linspace(0.5 * sigma, 1.5 * sigma, 50)
    plot_greek_surface(bs_delta, "Delta", grid_S, grid_sigma, args, args.outdir)
    plot_greek_surface(bs_vega, "Vega", grid_S, grid_sigma, args, args.outdir)

    print("\nAll results and plots saved. Project complete!")


if __name__ == "__main__":
    main()
