# Performance Benchmarks

Monte Carlo option pricing performance comparison.

| Engine | Paths | Seconds |
|---|---:|---:|
| numpy | 10,000 | 0.014 |
| numba | 10,000 | 0.007 |
| numpy | 100,000 | 0.152 |
| numba | 100,000 | 0.042 |
| numpy | 500,000 | 0.756 |
| numba | 500,000 | 0.191 |
| numpy | 1,000,000 | 1.991 |
| numba | 1,000,000 | 0.426 |
