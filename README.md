# Quant Option Pricer

A high-performance, production-grade Python project for pricing European call options using both Monte Carlo simulation (with Numba acceleration) and the Black-Scholes analytical formula. Incorporates real historical volatility from Yahoo Finance, risk metrics (VaR, CVaR), Greeks, convergence diagnostics, and interactive Jupyter notebook exploration.

---

## 🚀 Features

- **Monte Carlo Pricing**: Simulate up to 100,000+ GBM paths in seconds using Numba JIT.
- **Black-Scholes Comparison**: Analytical benchmark for European call options.
- **Live Volatility Feed**: Downloads 1 year of data from Yahoo Finance, computes annualized volatility.
- **Risk Metrics**: Computes and plots Value-at-Risk (VaR) and Conditional VaR (CVaR) from MC P&L.
- **Greeks**: Analytic and MC Delta/Vega, plus 3D surface plots.
- **Convergence Diagnostics**: Plots MC estimate vs. Black-Scholes price across log-spaced path counts.
- **Reproducible Outputs**: All figures saved under `plots/`.
- **Interactive Notebook**: Jupyter notebook with widgets for parameter exploration.
- **CI/CD & Docker**: Automated tests, linting, and Docker reproducibility.

---

## 🏁 Quickstart

### 1. Clone and Install
```bash
git clone https://github.com/Ryan-Cooley/quant-option-pricer.git
cd quant-option-pricer
pip install -r requirements.txt
```

### 2. Run Volatility Analysis & Option Pricer
```bash
python quant_option.py
```

### 3. Run the Interactive Notebook
```bash
pip install jupyter ipywidgets
jupyter notebook notebooks/QuantOptionDemo.ipynb
```

### 4. Run Performance Benchmark
```bash
python benchmark_performance.py
```

### 5. Run Tests & Linting
```bash
pytest tests
flake8 quant_option.py
black --check quant_option.py
```

### 6. Run in Docker
```bash
docker build -t quant-option .
docker run --rm -v "$PWD/plots:/app/plots" quant-option
```

---

## 📊 Plot Explanations

The project generates several key visualizations that provide deep insights into option pricing and risk analysis:

### 1. **Historical Log-Returns** (`plots/AAPL_returns.png`)
**What it shows**: Daily log-returns of the underlying asset (e.g., AAPL) over the past year.
**Financial significance**: 
- Log-returns are approximately normally distributed under the Black-Scholes model
- Volatility clustering and fat tails can be observed in real market data
- Used to estimate annualized volatility: σ = std(returns) × √252
- Helps validate the GBM assumption for option pricing

### 2. **Monte Carlo Convergence** (`plots/convergence.png`)
**What it shows**: MC option price estimates vs. number of simulated paths, compared to the analytical Black-Scholes price.
**Financial significance**:
- Demonstrates the Law of Large Numbers in action
- Shows how MC estimates converge to the true theoretical price
- Helps determine the minimum number of paths needed for accurate pricing
- Validates the MC implementation against the analytical benchmark
- Log-scale x-axis shows convergence across multiple orders of magnitude

### 3. **P&L Distribution with Risk Metrics** (`plots/pnl_histogram.png`)
**What it shows**: Histogram of simulated P&L at expiry, with VaR and CVaR marked.
**Financial significance**:
- **P&L Distribution**: Shows the range of possible outcomes for the option position
- **VaR (5%)**: The maximum expected loss with 95% confidence (red dashed line)
- **CVaR (5%)**: The expected loss given that we're in the worst 5% of scenarios (purple dotted line)
- Helps quantify downside risk and determine position sizing
- Demonstrates the asymmetric payoff structure of call options

### 4. **Delta Surface** (`plots/delta_surface.png`)
**What it shows**: 3D surface of Delta (∂V/∂S) across different spot prices and volatilities.
**Financial significance**:
- **Delta**: Measures the rate of change of option value with respect to underlying price
- **Surface behavior**: 
  - Near-the-money options have Delta ≈ 0.5
  - Deep in-the-money: Delta approaches 1.0
  - Deep out-of-the-money: Delta approaches 0.0
- **Volatility effect**: Higher volatility makes Delta less sensitive to spot price changes
- Essential for dynamic hedging and risk management

### 5. **Vega Surface** (`plots/vega_surface.png`)
**What it shows**: 3D surface of Vega (∂V/∂σ) across different spot prices and volatilities.
**Financial significance**:
- **Vega**: Measures the rate of change of option value with respect to volatility
- **Surface behavior**:
  - Maximum Vega occurs near-the-money
  - Vega decreases as option moves deep in/out-of-the-money
  - Vega increases with time to expiry
- **Volatility smile**: Real markets show Vega varies with strike, not just spot
- Critical for volatility trading and vega hedging strategies

### 6. **Performance Benchmark** (`benchmark_results.png`)
**What it shows**: Speed comparison between Numba-accelerated and pure Python implementations.
**Technical significance**:
- Demonstrates the performance benefits of Numba JIT compilation
- Shows scalability of the MC implementation
- Helps optimize computational resources for production use
- Validates that speed improvements don't compromise accuracy

---

## 🗂️ Project Structure

```
quant-option-pricer/
├── quant_option.py           # Main CLI script
├── benchmark_performance.py  # Performance benchmark
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker reproducibility
├── tests/
│   └── test_quant_option.py  # Unit tests
├── notebooks/
│   └── QuantOptionDemo.ipynb # Interactive notebook
├── plots/                    # Output figures
└── .github/
    └── workflows/ci.yml      # GitHub Actions CI
```

---

## 📈 Example Outputs
- `plots/AAPL_returns.png` — Historical log-returns
- `plots/convergence.png` — MC convergence to Black-Scholes
- `plots/pnl_histogram.png` — P&L histogram with VaR/CVaR
- `plots/delta_surface.png`, `plots/vega_surface.png` — Greek surfaces
- `benchmark_results.png` — Numba vs Python speedup

---

## 🧑‍💻 Resume/Portfolio Bullets
- Engineered a Monte Carlo simulator in Python to price European call options, achieving convergence within 1% of the Black-Scholes analytical price over 100k simulated paths.
- Automated live-volatility ingestion via yfinance, calculating annualized volatility from 1 year of log returns and dynamically feeding it into the option-pricing model.
- Visualized convergence, risk metrics (VaR/CVaR), and Greek surfaces with Matplotlib, producing publication-quality plots.
- Optimized vectorized NumPy operations and Numba JIT to simulate 100k asset paths in under 1 second on a standard laptop.
- Documented project structure, usage, and testing in a professional README; enabled full reproducibility via Docker and CI.

---

## 🤝 Contributing
Pull requests and issues are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
MIT License
