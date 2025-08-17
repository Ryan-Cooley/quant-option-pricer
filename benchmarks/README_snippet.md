# Performance Benchmarks

Monte Carlo option pricing performance comparison.

| Engine | Paths | Seconds |
|---|---:|---:|
| numpy | 10,000 | 0.014 |
| numba | 10,000 | 0.007 |
| numpy | 100,000 | 0.149 |
| numba | 100,000 | 0.043 |
| numpy | 500,000 | 0.747 |
| numba | 500,000 | 0.206 |
| numpy | 1,000,000 | 2.498 |
| numba | 1,000,000 | 0.385 |
