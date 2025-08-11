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
    Create enhanced IV surface visualization and save to file.

    Args:
        df: DataFrame with iv, K, S, T columns
        output_path: Path to save the plot
    """
    # Calculate moneyness
    df["moneyness"] = df["K"] / df["S"]

    # Create pivot table
    pivot = df.pivot_table(values="iv", index="moneyness", columns="T", aggfunc="mean")

    # Create figure with subplots for both 2D and 3D views
    fig = plt.figure(figsize=(18, 8))
    
    # Set style
    plt.style.use('default')
    
    # Create subplots with more space between them
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    
    # === Left subplot: Enhanced 2D Heatmap ===
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create contour-filled heatmap with more levels
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values
    
    # Create smooth contour plot
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 20)
    contour = ax1.contourf(X, Y, Z, levels=levels, cmap='RdYlBu_r', alpha=0.9)
    
    # Add contour lines
    contour_lines = ax1.contour(X, Y, Z, levels=10, colors='black', alpha=0.4, linewidths=0.5)
    ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Enhance colorbar
    cbar1 = plt.colorbar(contour, ax=ax1, shrink=0.8)
    cbar1.set_label('Implied Volatility (σ)', rotation=270, labelpad=20, fontsize=12)
    cbar1.ax.tick_params(labelsize=10)
    
    # Add data points
    for i, row in df.iterrows():
        ax1.plot(row['T'], row['moneyness'], 'ko', markersize=6, alpha=0.8)
        ax1.annotate(f"{row['iv']:.2f}", 
                    (row['T'], row['moneyness']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color='white', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Styling
    ax1.set_xlabel('Time to Maturity (years)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Moneyness (K/S)', fontsize=12, fontweight='bold')
    ax1.set_title('Implied Volatility Surface\n(Contour Map)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # === Right subplot: 3D Surface ===
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    
    # Create meshgrid for 3D surface
    X_3d, Y_3d = np.meshgrid(pivot.columns, pivot.index)
    Z_3d = pivot.values
    
    # Create 3D surface
    surf = ax2.plot_surface(X_3d, Y_3d, Z_3d, 
                           cmap='RdYlBu_r', 
                           alpha=0.9,
                           linewidth=0, 
                           antialiased=True,
                           edgecolor='none')
    
    # Add wireframe overlay
    ax2.plot_wireframe(X_3d, Y_3d, Z_3d, color='black', alpha=0.3, linewidth=0.5)
    
    # Add colorbar for 3D plot
    cbar2 = fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=20)
    cbar2.set_label('Implied Volatility (σ)', rotation=270, labelpad=15, fontsize=11)
    
    # Styling 3D plot
    ax2.set_xlabel('Time to Maturity (years)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Moneyness (K/S)', fontsize=11, fontweight='bold')
    ax2.set_zlabel('Implied Volatility (σ)', fontsize=11, fontweight='bold')
    ax2.set_title('Implied Volatility Surface\n(3D View)', fontsize=14, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax2.view_init(elev=25, azim=45)
    
    # Add subtle background color
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.grid(True, alpha=0.3)
    
    # Add dividing line down the center
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.1, 0.9], transform=fig.transFigure, 
                              color='gray', linewidth=2, alpha=0.7, linestyle='--'))
    
    # Overall figure styling
    fig.suptitle('Options Implied Volatility Analysis', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved enhanced IV surface plot → {output_path}")


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
