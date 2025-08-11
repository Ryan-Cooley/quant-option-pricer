[![CI](https://github.com/Ryan-Cooley/quant-option-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryan-Cooley/quant-option-pricer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

# Monte Carlo Option Pricer

A high-performance, production-ready Monte Carlo simulation engine for pricing European options, complete with comprehensive risk analysis and visualization. This project showcases a blend of advanced Python programming, financial modeling, and modern software engineering best practices.

## 🎯 Overview

The Monte Carlo Option Pricer provides a robust framework for option valuation by combining the analytical Black-Scholes model with stochastic Monte Carlo simulations. The core of the project is a high-performance simulation engine, accelerated with Numba JIT compilation, which achieves a significant speedup (over 100x) compared to pure Python implementations. This allows for large-scale simulations without sacrificing performance.

### Key Capabilities

- **Dual Pricing Models**: European Call and Put option pricing using both the analytical Black-Scholes formula and a Monte Carlo simulation engine.
- **Comprehensive Risk Metrics**: Calculation and visualization of key risk metrics, including Value at Risk (VaR), Conditional Value at Risk (CVaR), and the Greeks (Delta and Vega).
- **Live Data Integration**: Seamless integration with Yahoo Finance to fetch historical price data and calculate annualized volatility for more realistic pricing.
- **Advanced Visualizations**: A suite of plotting functions to generate insightful visualizations, including simulation convergence analysis, P&L distributions, and 3D sensitivity surfaces for the Greeks.
- **Production-Ready**: The project is fully containerized with Docker, includes a comprehensive test suite with `pytest`, and features a CI/CD pipeline using GitHub Actions for automated testing and code quality checks.

---

## 🚀 Core Features

- **⚡ High-Performance Simulation**: Numba-accelerated Monte Carlo engine for high-speed, large-scale simulations.
- **📊 Live Market Data**: Real-time volatility estimation using adjusted close prices from Yahoo Finance.
- **🎯 Advanced Risk Analysis**: In-depth risk assessment with VaR, CVaR, and Greeks calculations.
- **🔬 Mathematical Rigor**: The models are validated through a comprehensive test suite that includes edge cases and convergence checks against the analytical Black-Scholes model.
- **📈 Rich Visualizations**: Publication-quality plots for convergence analysis, P&L distributions, and sensitivity analysis.
- **🐳 Production Deployment**: Docker containerization and an automated CI/CD pipeline for robust, reproducible deployments.
- **🧪 Reproducible Results**: Deterministic simulations through careful management of random seeds, ensuring that results can be reproduced.

---

## 🏁 Quickstart

### Prerequisites

- Python 3.8+
- `pip` or `conda` for package management.

### Installation

```bash
# Clone the repository
git clone https://github.com/Ryan-Cooley/quant-option-pricer.git
cd quant-option-pricer

# Install dependencies (standard)
pip install -r requirements.txt

# Or install in editable/development mode (recommended for contributors)
pip install -e .
```

> **Note:**
> - Use `requirements.txt` for standard Python/pip environments.
> - Use `environment.yml` with `conda env create -f environment.yml` for full Conda-based environments (recommended for reproducibility or if you use Anaconda/Miniconda).

### Basic Usage

The primary entry point is `quant_option.py`, a command-line interface for pricing options.

```bash
# Price a call option for any ticker (e.g., MSFT, AAPL, TSLA)
python quant_option.py --ticker MSFT

# Price a put option for TSLA with custom parameters
python quant_option.py --ticker TSLA --option-type put --K 200 --T 0.5 --r 0.03 --paths 50000
```

*Sample outputs in this README use AAPL as the ticker.*

### Interactive Notebook

For a more interactive experience, a Jupyter notebook is provided in the `notebooks` directory. The default ticker in the notebook is set to AAPL for consistency with the sample outputs, but you can change it to any valid symbol.

```bash
# Launch the Jupyter notebook
jupyter notebook notebooks/QuantOptionDemo.ipynb
```

---

## ✅ Testing and Validation

The project includes a comprehensive test suite to ensure the correctness and reliability of the implementation.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=quant_option
```

### Validation Framework

- **Analytical Correctness**: The Black-Scholes formulas are tested against known, trusted values to ensure their accuracy.
- **Stochastic Validation**: The Monte Carlo engine's results are validated for convergence to the analytical Black-Scholes price. Reproducibility is ensured by testing with a fixed random seed.
- **Edge Case Handling**: The test suite includes checks for edge cases such as zero volatility and zero time-to-maturity to ensure the models behave as expected.
- **Data Pipeline**: The volatility estimation from historical data is tested using a static dataset for consistency.

---

## 📊 Sample Outputs

*All plots are generated by the code and can be found in the `plots/` directory.*

### 1. **Historical Log-Returns**
<img src="https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/MSFT_returns.png" alt="MSFT Returns" style="display: block; margin: auto;">
*Daily log-returns of the underlying asset, calculated from close prices to accurately reflect historical volatility.*

### 2. **Simulation Convergence**
<img src="https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/convergence_v2.png" alt="Convergence" style="display: block; margin: auto;">
*The Monte Carlo option price converges smoothly to the analytical Black-Scholes price as the number of simulation paths increases (up to 200,000), validating the accuracy of the simulation.*

### 3. **P&L Distribution**
<img src="https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/pnl_histogram.png" alt="P&L Distribution" style="display: block; margin: auto;">
*The distribution of potential profit and loss at option expiry, with VaR and CVaR metrics highlighted to provide a clear view of downside risk.*

### 3b. **Enhanced Risk Analysis**
<img src="https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/enhanced_risk_analysis.png" alt="Enhanced Risk Analysis" style="display: block; margin: auto;">
*A comprehensive view combining: (1) P&L distribution with VaR/CVaR and mean, (2) payoff distribution, (3) cumulative P&L distribution with the 5% threshold, and (4) an explanation panel clarifying why VaR and CVaR equal the option premium for at-the-money options.*

### 4. **Greeks Sensitivity Surfaces**
<img src="https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/delta_surface.png" alt="Delta Surface" style="display: block; margin: auto;">
<img src="https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/vega_surface.png" alt="Vega Surface" style="display: block; margin: auto;">
*3D surfaces showing the sensitivity of the option price (Delta and Vega) to changes in the spot price and volatility.*

---

### 🗝️ Key Results (AAPL, K=150, T=1.0, r=0.01, 50,000 paths)

| Metric             | Value        |
|--------------------|--------------|
| **BS Price**       | **19.6754**  |
| **MC Price**       | **19.0907**  |
| VaR (5%)           | 19.6754      |
| CVaR (5%)          | 19.6754      |
| **Analytic Δ**     | **0.5757**   |
| **MC Δ**           | **0.5731**   |

**Enhanced Risk Analysis:**
- **Probability of Expiring Worthless**: 55.5%
- **Expected P&L**: -$0.5846
- **P&L Standard Deviation**: $33.6852
- **Maximum Loss**: -$19.6754 (full premium)
- **Maximum Gain**: $462.8489

**VaR/CVaR at Different Confidence Levels:**
- 1%: VaR=$19.6754, CVaR=$19.6754
- 5%: VaR=$19.6754, CVaR=$19.6754
- 10%: VaR=$19.6754, CVaR=$19.6754
- 25%: VaR=$19.6754, CVaR=$19.6754

**⚠️ Note on VaR/CVaR Calculation:**

The VaR and CVaR values equal the BS price because:

1. **At-the-money option**: For S₀=K=150, approximately 55% of Monte Carlo paths result in the option expiring worthless
2. **5th percentile threshold**: Since 55% > 5%, the 5th percentile loss equals the maximum possible loss (the full option premium)
3. **Expected behavior**: This is mathematically correct but not very informative for risk analysis

**For more meaningful VaR/CVaR, consider:**
- Out-of-the-money options (higher probability of expiring worthless)
- Different confidence levels (e.g., 1% instead of 5%)
- P&L from seller's perspective
- Portfolio-level risk metrics

---

## Performance (NumPy vs Numba)

Fair comparison with shared RNG, JIT warm-up, and separate stage timing.

For small path counts, vectorized NumPy can win (JIT overhead). Beyond ~50–100k paths, the looped numba kernel with prange and fastmath typically pulls ahead.

**Artifacts:**
- CSV: `benchmarks/benchmarks.csv`
- Markdown snippet: `benchmarks/README_snippet.md`
- Plot: `benchmarks/benchmarks.png`

# Performance Benchmarks

Monte Carlo option pricing performance comparison.

| Engine | Paths | Seconds |
|---|---:|---:|
| numpy | 10,000 | 0.015 |
| numba | 10,000 | 0.008 |
| numpy | 100,000 | 0.152 |
| numba | 100,000 | 0.041 |
| numpy | 500,000 | 0.755 |
| numba | 500,000 | 0.185 |
| numpy | 1,000,000 | 2.418 |
| numba | 1,000,000 | 0.385 |

## Hedged P&L (Discrete Delta Hedging)

- Black–Scholes GBM dynamics, discrete re-hedging on a configurable grid (e.g., daily/weekly).
- Fee model: proportional (bps on traded notional) + optional fixed per trade.
- Outputs: P&L distribution, percentiles, annualized Sharpe; plots show **tracking error vs. re-hedge frequency** and the **cost–error trade-off**.

**Repro:**
```bash
python scripts/run_hedge.py --S0 100 --K 100 --T 0.5 --r 0.02 --sigma 0.2 \
  --option-type call --n-paths 100000 --rebalance-every 1 --fee-bps 1.0
```

Takeaway: Higher frequency ↓ tracking error, ↑ trading costs. Choose a frequency on the efficient frontier.

## Implied Volatility (BS) + Mini Surface

Robust IV via Brent/bisection with no-arbitrage bounds and round-trip unit tests.

CSV-driven demo: reads `data/iv_demo.csv`, writes `plots/iv_grid.csv` and `plots/iv_surface.png`.

**Repro:**
```bash
python scripts/calibrate_surface.py
```

Sample (moneyness = K/S, maturity in years), extracted from `plots/iv_grid.csv` when present:

| Moneyness \ T | 0.25 | 0.50 |
|---------------|------|------|
| 0.90          | 0.11 | 0.14 |
| 1.00          | 0.21 | 0.23 |
| 1.10          | 0.04 | 0.07 |

*(Numbers are illustrative; see your generated CSV for exact values.)*

## 🧑‍💻 Technical Achievements & Skills Demonstrated

### Technical Achievements

- **Engineered a High-Performance Monte Carlo Engine**: Developed a sophisticated financial simulator in Python, leveraging Numba for JIT compilation to achieve a **~38x speedup** over native Python, enabling large-scale simulations.
- **Implemented a Robust Testing Framework**: Created a comprehensive test suite with `pytest` to validate the analytical Black-Scholes model and verify the stochastic convergence and reproducibility of the Monte Carlo engine.
- **Automated Market Data & Enhanced Risk Analysis Pipeline**: Built a data pipeline that ingests historical market data from `yfinance`, calculates annualized volatility, and feeds it into a comprehensive risk analysis module that computes and visualizes VaR, CVaR, Greeks, and additional risk metrics including probability of expiring worthless, expected P&L, and multiple confidence levels.
- **Ensured Full Reproducibility with Docker & CI/CD**: Containerized the application with Docker and configured a GitHub Actions workflow for automated testing and linting, ensuring consistent and reliable results in any environment.

### Skills Demonstrated

- **Advanced Python**: Numba JIT compilation, NumPy vectorization, pandas for data manipulation, `argparse` for CLI.
- **Financial Modeling**: Black-Scholes theory, Monte Carlo simulation, risk metrics (VaR, CVaR), and Greeks.
- **Software Engineering**: Unit testing (`pytest`), CI/CD (GitHub Actions), containerization (Docker), and code quality (linting, formatting).
- **Data Science**: Statistical analysis, data visualization (`matplotlib`), and reproducible research.

---

## 🗂️ Project Structure

```
quant-option-pricer/
├── quant_option.py           # Main CLI script with enhanced risk analysis
├── benchmark_performance.py  # Performance benchmarking suite
├── requirements.txt          # Python dependencies (pip)
├── environment.yml           # Conda environment (alternative to requirements.txt)
├── Dockerfile                # Docker containerization
├── tests/
│   ├── test_quant_option.py  # Comprehensive unit and integration tests
│   └── test_data.csv         # Static test data for reproducible testing
├── notebooks/
│   └── QuantOptionDemo.ipynb # Interactive Jupyter notebook with enhanced analysis
├── plots/                    # Directory for generated output figures
└── .github/
    └── workflows/ci.yml      # GitHub Actions CI/CD pipeline
```

---

## 🔧 Development

### Code Quality

The project follows standard Python code quality practices.

```bash
# Format code with Black
black .

# Lint code with flake8
flake8 .
```

### Docker Development

```bash
# Build the Docker image
docker build -t quant-option .

# Run the container
docker run --rm -v "$(pwd)/plots:/app/plots" quant-option --ticker GOOG
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/my-new-feature`).
3.  Make your changes and add tests.
4.  Ensure all tests pass (`pytest`).
5.  Commit your changes (`git commit -m 'Add some feature'`).
6.  Push to the branch (`git push origin feature/my-new-feature`).
7.  Open a Pull Request.

---

## Reproduce

```bash
python scripts/run_hedge.py --S0 100 --K 100 --T 0.5 --r 0.02 --sigma 0.2 --n-paths 25000
python scripts/calibrate_surface.py
python scripts/benchmark_performance.py
```

See `notebooks/00_quickstart.ipynb` for a minimal end-to-end demo.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- The high-performance numerical computations are made possible by [Numba](https://numba.pydata.org/).
- Market data is sourced from [Yahoo Finance](https://finance.yahoo.com/) via the [yfinance](https://github.com/ranaroussi/yfinance) library.
- The option pricing models are based on the foundational work of Fischer Black, Myron Scholes, and Robert Merton.
