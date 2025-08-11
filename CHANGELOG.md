# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-08-11

### Added
- Discrete delta-hedging simulator (`src/quant/hedging.py`) + CLI (`scripts/run_hedge.py`) and tests.
- Black–Scholes implied volatility solver (`src/quant/iv.py`) with no-arbitrage checks and round-trip tests.
- CSV-driven IV surface tool (`scripts/calibrate_surface.py`) + demo data + smoke tests.
- Performance benchmark suite (`scripts/benchmark_performance.py`) with shared RNG, warm-up, stage timing; outputs CSV, Markdown, PNG.

### Changed
- Refactored core pricing into `src/quant/pricing.py`; updated CLI/notebooks/tests; improved docs.

### CI
- Slow tests excluded by default; convergence/benchmarks marked `slow`.
