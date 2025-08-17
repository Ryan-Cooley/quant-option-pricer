# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-08-16

### Added
- Dotted line connecting frontier points for better cost-error trade-off visualization
- Enhanced frontier plot with 4 default rebalance frequencies (1,2,5,10 steps)

### Changed
- Updated default `--rebalance-every` from "1" to "1,2,5,10" for meaningful frontier plots
- Updated CI workflow to use new default rebalance frequencies
- Updated README repro commands and flags table to reflect new defaults

### Fixed
- Code quality: Black formatting and Flake8 compliance across all files
- CI artifacts now show proper 4-point frontier plots

## [0.3.0] - 2025-08-16

### Added
- Hedging frontier analysis with CSV logging (`--out-csv`)
- Units toggle for TE/Cost denominators (`--units {s0,premium}`)
- Frontier scatter plot (`plots/hedge_frontier.png`)
- Metadata stamps on plots with simulation parameters
- Comprehensive test coverage for all new features

### Changed
- Enhanced IV surface visualization with dual-panel layout
- Improved plot aesthetics and readability

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
