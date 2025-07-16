# Monte Carlo Option Pricer

![CI Workflow Status](https://github.com/Ryan-Cooley/quant-option-pricer/workflows/CI/badge.svg)

This project grew out of a curiosity about how complex systems behave under uncertainty and how we can build tools to explore those questions at scale. The Monte Carlo Option Pricer is a fast, reliable simulation engine that models financial options and serves as a testbed for understanding risk, performance, and system robustness in real-world scenarios. By combining modern Python techniques with a focus on reproducibility and automation, it invites scientists, engineers, and curious minds alike to experiment, learn, and trust the results. The project is fully automated and production-ready, designed to run smoothly and transparently in any environment.

---

## 🚀 Core Features

- **Flexible Simulation Engine**: Models complex systems and their behavior under uncertainty, using both analytical and simulation-based approaches.
- **Live Data Integration**: Ingests real-world data to keep experiments grounded and relevant.
- **Risk Modeling & Performance Metrics**: Provides clear insights into system reliability, variability, and sensitivity to changing conditions.
- **Advanced Visualizations**: Generates intuitive plots that help users see how systems respond to different scenarios.
- **Reproducibility & Automation**: Automated testing, containerization, and continuous integration ensure results are reliable and easy to share.

---

## 🏁 Quickstart

### 1. Clone and Install

```bash
git clone https://github.com/Ryan-Cooley/quant-option-pricer.git
cd quant-option-pricer
pip install -r requirements.txt
```

### 2. Run the Option Pricer

```bash
python quant_option.py --ticker AAPL --S0 150 --K 155
```

### 3. Run the Full Test Suite

This command runs all unit and integration tests to ensure the models are functioning correctly.

```bash
PYTHONPATH=. pytest
```

### 4. Run the Performance Benchmark

```bash
python benchmark_performance.py
```

### 5. Run in Docker

For guaranteed reproducibility, build and run the project in a container.

```bash
docker build -t quant-option .
docker run --rm -v "$PWD/plots:/app/plots" quant-option
```

---

## ✅ Testing & Validation

A key feature of this project is its emphasis on correctness and reliability, enforced by a comprehensive test suite using `pytest`.

- **Analytical Correctness**: The Black-Scholes formula and its Greeks are tested against known, trusted values to ensure their implementation is accurate.
- **Stochastic Validation**: The Monte Carlo simulator is tested for:
    - **Reproducibility**: Guarantees that the same random seed produces the exact same pricing result.
    - **Convergence**: Verifies that the MC price converges to the analytical Black-Scholes price within a tight tolerance as the number of simulation paths increases.
- **Data Pipeline Integrity**: The volatility calculation is tested using a static, local data file to ensure the data processing logic is sound and independent of live data sources.

This rigorous testing framework ensures that the results produced by the models are not just fast, but also accurate and reliable.

---

## 📊 Plot Explanations.

The project generates several key visualizations that provide insight into system behavior and reliability:

### 1. **Historical Log-Returns** (`https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/AAPL_returns.png`)
Shows how the underlying asset’s value changes day to day, capturing the natural variability and occasional surprises in real-world data. This helps set realistic expectations for system performance.

### 2. **Simulation Convergence** (`https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/convergence_v2.png`)
Demonstrates how repeated simulations become more reliable as more data is gathered, illustrating the importance of scale and repetition in understanding complex systems.

### 3. **P&L Distribution** (`https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/pnl_histogram.png`)
Visualizes the range of possible outcomes for a system, highlighting not just the average result but also the likelihood of rare, extreme events. This is key for understanding risk and robustness.

### 4. **System Sensitivity Surfaces** (`https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/delta_surface.png`, `https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/vega_surface.png`)
Shows how the system responds to changes in key parameters, making it easy to spot which factors have the biggest impact on outcomes.

### 5. **Performance Benchmark** (`https://raw.githubusercontent.com/Ryan-Cooley/quant-option-pricer/main/plots/benchmark_results.png`)
Compares the speed and efficiency of different computational approaches, demonstrating the value of performance optimization in large-scale experiments.

---

## 🧑‍💻 Resume & Portfolio Highlights

- **Engineered a High-Performance Monte Carlo Engine**: Developed a financial simulator in Python to price European options, leveraging Numba for JIT compilation to achieve a >100x speedup over pure Python for 100,000+ path simulations.
- **Validated Models with a Robust Test Suite**: Implemented a comprehensive `pytest` framework to validate the correctness of the analytical Black-Scholes formulas and to verify the stochastic convergence and reproducibility of the Monte Carlo engine.
- **Automated Market Data & Risk Analysis Pipeline**: Built a data pipeline that ingests historical market data from `yfinance`, calculates annualized volatility, and feeds it into a risk analysis module that computes and visualizes VaR, CVaR, and the Greeks.
- **Ensured Full Reproducibility with Docker & CI/CD**: Containerized the entire application with Docker and configured a GitHub Actions workflow for automated testing and linting, ensuring consistent and reliable results across any environment.

---

## 🗂️ Project Structure

```
quant-option-pricer/
├── quant_option.py           # Main CLI script
├── benchmark_performance.py  # Performance benchmark
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker reproducibility
├── tests/
│   ├── test_quant_option.py  # Unit and integration tests
│   └── test_data.csv         # Static data for testing
├── notebooks/
│   └── QuantOptionDemo.ipynb # Interactive notebook
├── plots/                    # Output figures
└── .github/
    └── workflows/ci.yml      # GitHub Actions CI
```

---

## 🤝 Contributing

Pull requests and issues are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT License