"""
Smoke tests for performance benchmark functionality.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from scripts.benchmark_performance import (  # noqa: E402
    time_numpy_call_pricer,
    time_numba_call_pricer,
    run_bench,
    emit_csv_and_snippet,
)


def test_time_numpy_small():
    """Test NumPy timing function with small dataset."""
    t, price = time_numpy_call_pricer(10_000, 64)
    assert float(t) > 0
    assert not np.isnan(t)
    assert price > 0


def test_time_numpy_medium():
    """Test NumPy timing function with medium dataset."""
    t, price = time_numpy_call_pricer(50_000, 128)
    assert float(t) > 0
    assert not np.isnan(t)
    assert price > 0


@pytest.mark.slow
def test_time_numpy_large():
    """Test NumPy timing function with large dataset (marked as slow)."""
    t, price = time_numpy_call_pricer(100_000, 252)
    assert float(t) > 0
    assert not np.isnan(t)
    assert price > 0


def test_time_numba_small():
    """Test Numba timing function with small dataset."""
    t, price = time_numba_call_pricer(10_000, 64)
    # Numba might not be available, so we just check it's not negative
    assert t >= 0 or np.isnan(t)
    assert price > 0 or np.isnan(price)


def test_run_bench_small():
    """Test benchmark runner with small path counts."""
    df = run_bench(path_counts=(10_000, 20_000), repeats=2)

    # Check DataFrame structure
    assert "engine" in df.columns
    assert "n_paths" in df.columns
    assert "seconds" in df.columns

    # Check that we have results
    assert len(df) > 0

    # Check that NumPy results are present and positive
    numpy_results = df[df["engine"] == "numpy"]
    assert len(numpy_results) > 0
    assert all(numpy_results["seconds"] > 0)


@pytest.mark.slow
def test_run_bench_medium():
    """Test benchmark runner with medium path counts (marked as slow)."""
    df = run_bench(path_counts=(50_000, 100_000), repeats=2)

    # Check DataFrame structure
    assert "engine" in df.columns
    assert "n_paths" in df.columns
    assert "seconds" in df.columns

    # Check that we have results
    assert len(df) > 0

    # Check that NumPy results are present and positive
    numpy_results = df[df["engine"] == "numpy"]
    assert len(numpy_results) > 0
    assert all(numpy_results["seconds"] > 0)


def test_emit_csv_and_snippet():
    """Test CSV and Markdown emission functionality."""
    # Create test data
    test_data = [
        {"engine": "numpy", "n_paths": 10000, "seconds": 0.123},
        {"engine": "numba", "n_paths": 10000, "seconds": 0.018},
        {"engine": "numpy", "n_paths": 50000, "seconds": 0.456},
        {"engine": "numba", "n_paths": 50000, "seconds": np.nan},
    ]
    df = pd.DataFrame(test_data)

    # Test emission
    csv_path = "./test_benchmarks.csv"
    md_path = "./test_README_snippet.md"

    try:
        emit_csv_and_snippet(df, csv_path, md_path)

        # Check that files were created
        assert os.path.exists(csv_path)
        assert os.path.exists(md_path)

        # Check CSV content
        df_read = pd.read_csv(csv_path)
        assert len(df_read) == len(df)
        assert "engine" in df_read.columns
        assert "n_paths" in df_read.columns
        assert "seconds" in df_read.columns

        # Check Markdown content
        with open(md_path, "r") as f:
            md_content = f.read()

        assert "| Engine | Paths | Seconds |" in md_content
        assert "| numpy | 10,000 | 0.123 |" in md_content
        assert "| numba | 50,000 | N/A |" in md_content

    finally:
        # Clean up test files
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists(md_path):
            os.remove(md_path)
