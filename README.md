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
