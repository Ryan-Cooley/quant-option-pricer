"""
Smoke test for IV surface calibration functionality.
"""

import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from scripts.calibrate_surface import compute_ivs  # noqa: E402


def test_compute_ivs_basic():
    """Test basic IV computation on a small dataset."""
    # Create test data
    test_data = {
        "S": [100, 100, 100],
        "K": [100, 105, 95],
        "T": [0.5, 0.5, 0.5],
        "option_type": ["call", "call", "call"],
        "mid_price": [7.15, 2.45, 12.50],
        "r": [0.01, 0.01, 0.01],
    }

    df = pd.DataFrame(test_data)

    # Compute IVs
    result = compute_ivs(df)

    # Check that result has iv column
    assert "iv" in result.columns

    # Check that all IVs are positive
    assert all(result["iv"] > 0)

    # Check that IVs are reasonable (between 0.05 and 1.0)
    assert all(0.05 <= iv <= 1.0 for iv in result["iv"])

    # Check that original data is preserved
    for col in ["S", "K", "T", "option_type", "mid_price", "r"]:
        assert all(result[col] == df[col])


def test_compute_ivs_with_invalid_data():
    """Test IV computation handles invalid data gracefully."""
    # Create test data with one invalid row
    test_data = {
        "S": [100, 100, 100],
        "K": [100, 105, 95],
        "T": [0.5, 0.5, 0.5],
        "option_type": ["call", "call", "call"],
        "mid_price": [7.15, 1000.0, 12.50],  # Invalid price in middle
        "r": [0.01, 0.01, 0.01],
    }

    df = pd.DataFrame(test_data)

    # Compute IVs
    result = compute_ivs(df)

    # Check that result has iv column
    assert "iv" in result.columns

    # Check that invalid row has NaN
    assert pd.isna(result.iloc[1]["iv"])

    # Check that valid rows have positive IVs
    assert result.iloc[0]["iv"] > 0
    assert result.iloc[2]["iv"] > 0


def test_compute_ivs_put_options():
    """Test IV computation works for put options."""
    # Create test data with put options
    test_data = {
        "S": [100, 100],
        "K": [100, 105],
        "T": [0.5, 0.5],
        "option_type": ["put", "put"],
        "mid_price": [3.50, 8.20],
        "r": [0.01, 0.01],
    }

    df = pd.DataFrame(test_data)

    # Compute IVs
    result = compute_ivs(df)

    # Check that result has iv column
    assert "iv" in result.columns

    # Check that all IVs are positive
    assert all(result["iv"] > 0)

    # Check that IVs are reasonable
    assert all(0.05 <= iv <= 1.0 for iv in result["iv"])
