[![CI](https://github.com/Ryan-Cooley/quant-option-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryan-Cooley/quant-option-pricer/actions/workflows/ci.yml)

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

*Sample outputs in this README use MSFT as the ticker.*

### Interactive Notebook

For a more interactive experience, a Jupyter notebook is provided in the `notebooks` directory. The default ticker in the notebook is set to MSFT for consistency with the sample outputs, but you can change it to any valid symbol.

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

### 4. **Greeks Sensitivity Surfaces**
<img src="https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/delta_surface.png" alt="Delta Surface" style="display: block; margin: auto;">
<img src="https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/vega_surface.png" alt="Vega Surface" style="display: block; margin: auto;">
*3D surfaces showing the sensitivity of the option price (Delta and Vega) to changes in the spot price and volatility.*

---

### 🗝️ Key Results (MSFT, K=150, T=0.5, r=0.03, 200,000 paths)

--- Key Results ---
| Metric             | Value        |
|--------------------|--------------|
| Mean P&L           | -0.1052      |
| VaR (5%)           | 15.6921      |
| CVaR (5%)          | 15.6921      |
| Δ per $1 move      | 0.5646       |
| Δ accuracy         | 0.20% error  |
| Numba Speedup      | 100x+        |

## 🧑‍💻 Technical Achievements & Skills Demonstrated

### Technical Achievements

- **Engineered a High-Performance Monte Carlo Engine**: Developed a sophisticated financial simulator in Python, leveraging Numba for JIT compilation to achieve a >100x speedup over native Python, enabling large-scale simulations.
- **Implemented a Robust Testing Framework**: Created a comprehensive test suite with `pytest` to validate the analytical Black-Scholes model and verify the stochastic convergence and reproducibility of the Monte Carlo engine.
- **Automated Market Data & Risk Analysis Pipeline**: Built a data pipeline that ingests historical market data from `yfinance`, calculates annualized volatility, and feeds it into a risk analysis module that computes and visualizes VaR, CVaR, and the Greeks.
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
├── quant_option.py           # Main CLI script with full functionality
├── benchmark_performance.py  # Performance benchmarking suite
├── requirements.txt          # Python dependencies (pip)
├── environment.yml           # Conda environment (alternative to requirements.txt)
├── Dockerfile                # Docker containerization
├── tests/
│   ├── test_quant_option.py  # Comprehensive unit and integration tests
│   └── test_data.csv         # Static test data for reproducible testing
├── notebooks/
│   └── QuantOptionDemo.ipynb # Interactive Jupyter notebook for exploration
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

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- The high-performance numerical computations are made possible by [Numba](https://numba.pydata.org/).
- Market data is sourced from [Yahoo Finance](https://finance.yahoo.com/) via the [yfinance](https://github.com/ranaroussi/yfinance) library.
- The option pricing models are based on the foundational work of Fischer Black, Myron Scholes, and Robert Merton.
