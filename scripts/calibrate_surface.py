#!/usr/bin/env python3
"""
IV Surface Calibration Tool

Read option data from CSV, compute implied volatilities, and generate surface plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    Create clean, professional IV surface visualization.

    Args:
        df: DataFrame with iv, K, S, T columns
        output_path: Path to save the plot
    """
    # Calculate moneyness
    df["moneyness"] = df["K"] / df["S"]

    # Create pivot table
    pivot = df.pivot_table(values="iv", index="moneyness", columns="T", aggfunc="mean")

    # Create single figure with clean 3D surface
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for smooth interpolation
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values
    
    # Create smooth 3D surface with elegant styling
    surf = ax.plot_surface(X, Y, Z, 
                          cmap='plasma',           # Modern, professional colormap
                          alpha=0.85,
                          linewidth=0, 
                          antialiased=True,
                          edgecolor='none',
                          shade=True)
    
    # Add clean wireframe for structure without clutter
    ax.plot_wireframe(X, Y, Z, 
                     color='white', 
                     alpha=0.6, 
                     linewidth=0.8,
                     linestyle='-')
    
    # Add market data points as clean spheres
    for i, row in df.iterrows():
        ax.scatter(row['T'], row['moneyness'], row['iv'], 
                  color='red', s=60, alpha=0.9, edgecolors='darkred', linewidth=1)
    
    # Professional colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=25, pad=0.1)
    cbar.set_label('Implied Volatility (σ)', rotation=270, labelpad=20, fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Clean, professional styling
    ax.set_xlabel('Time to Maturity (years)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Moneyness (K/S)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_zlabel('Implied Volatility (σ)', fontsize=13, fontweight='bold', labelpad=10)
    
    # Set optimal viewing angle for clarity
    ax.view_init(elev=30, azim=225)
    
    # Clean background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Professional title with clean spacing
    ax.set_title('Implied Volatility Surface', 
                fontsize=16, fontweight='bold', pad=30)
    
    # Add subtle data summary in clean box
    iv_min, iv_max = np.nanmin(Z), np.nanmax(Z)
    iv_mean = np.nanmean(Z)
    
    textstr = f'Market Data Summary:\nIV Range: {iv_min:.1%} - {iv_max:.1%}\nAverage IV: {iv_mean:.1%}\nData Points: {len(df)}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props, family='monospace')
    
    # Set clean axis limits
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min() * 0.95, Z.max() * 1.05)
    
    # High-quality output
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()

    print(f"Saved clean IV surface plot → {output_path}")


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
