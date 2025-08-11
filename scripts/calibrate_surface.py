#!/usr/bin/env python3
"""
IV Surface Calibration Tool

Read option data from CSV, compute implied volatilities, and generate surface plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from src.quant.iv import implied_vol_bs  # noqa: E402


def compute_ivs(df):
    """
    Compute implied volatilities for each row in the DataFrame.

    Args:
        df: DataFrame with columns S, K, T, option_type, mid_price, r

    Returns:
        DataFrame with added 'iv' column
    """
    df = df.copy()
    ivs = []

    for _, row in df.iterrows():
        try:
            iv = implied_vol_bs(
                price=row["mid_price"],
                S=row["S"],
                K=row["K"],
                T=row["T"],
                r=row["r"],
                option_type=row["option_type"],
            )
            ivs.append(iv)
        except Exception as e:
            print(f"Warning: Could not compute IV for row {row.name}: {e}")
            ivs.append(np.nan)

    df["iv"] = ivs
    return df


def create_iv_surface(df, output_path):
    """
    Create IV surface heatmap and save to file.

    Args:
        df: DataFrame with iv, K, S, T columns
        output_path: Path to save the plot
    """
    # Calculate moneyness
    df["moneyness"] = df["K"] / df["S"]

    # Create pivot table
    pivot = df.pivot_table(values="iv", index="moneyness", columns="T", aggfunc="mean")

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Create heatmap
    im = plt.imshow(
        pivot.values,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
    )

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Implied Volatility", rotation=270, labelpad=20)

    # Set labels and title
    plt.xlabel("Time to Maturity (years)")
    plt.ylabel("Moneyness (K/S)")
    plt.title("Implied Volatility Surface")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if not np.isnan(value):
                plt.text(
                    pivot.columns[j],
                    pivot.index[i],
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.3 else "black",
                    fontsize=10,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved IV surface plot → {output_path}")


def print_sample_table(df, n_rows=3, n_cols=2):
    """
    Print a sample table of the IV data.

    Args:
        df: DataFrame with IV data
        n_rows: Number of rows to show
        n_cols: Number of columns to show
    """
    # Select sample columns
    sample_cols = ["S", "K", "T", "option_type", "mid_price", "iv"]
    sample_df = df[sample_cols].head(n_rows)

    print("\nSample IV Data:")
    print("=" * 80)
    print(sample_df.to_string(index=False, float_format="%.3f"))
    print("=" * 80)


def main():
    """Main function."""
    # Input and output paths
    input_file = os.path.join(os.path.dirname(__file__), "..", "data", "iv_demo.csv")
    output_csv = os.path.join(os.path.dirname(__file__), "..", "plots", "iv_grid.csv")
    output_plot = os.path.join(
        os.path.dirname(__file__), "..", "plots", "iv_surface.png"
    )

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("IV Surface Calibration Tool")
    print("=" * 50)
    print(f"Reading data from: {input_file}")

    # Read the data
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} option records")
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return 1
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 1

    # Compute implied volatilities
    print("Computing implied volatilities...")
    df_with_iv = compute_ivs(df)

    # Filter out rows where IV computation failed
    valid_df = df_with_iv.dropna(subset=["iv"])
    if len(valid_df) < len(df):
        print(f"Warning: {len(df) - len(valid_df)} rows had IV computation issues")

    # Save the results
    print(f"Saving results to: {output_csv}")
    valid_df.to_csv(output_csv, index=False)

    # Create IV surface plot
    print("Creating IV surface plot...")
    create_iv_surface(valid_df, output_plot)

    # Print sample table
    print_sample_table(valid_df)

    print("\nCalibration complete!")
    return 0


if __name__ == "__main__":
    exit(main())
