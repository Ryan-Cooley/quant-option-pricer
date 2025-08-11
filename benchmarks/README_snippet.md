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
