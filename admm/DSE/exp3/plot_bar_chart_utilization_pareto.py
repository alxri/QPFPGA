#!/usr/bin/env python3
"""
Global Pareto Utilization Plotter (Fixed)
1. Loads CSV and crawls Vivado folders to find BRAM usage (creating the missing column).
2. Identifies designs that are Pareto-optimal.
3. Extracts and plots detailed FPGA resource utilization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
CSV_PATH = Path("exp3_tile_size_sweep_results.csv")
BITSTREAM_ROOT = Path("../generated_bitstreams")
OUTPUT_DIR = Path("global_pareto_utilization_plots")

# Map Vivado report labels to clean plot labels
RESOURCE_MAP = {
    "CLB LUTs": "LUTs",
    "CLB Registers": "FFs",
    "Block RAM Tile": "BRAM",
    "URAM": "URAM",
    "DSPs": "DSPs"
}

def parse_bram_from_rpt(rpt_path: Path):
    """Helper to get just the BRAM % for Pareto calculations."""
    try:
        with rpt_path.open("r") as fh:
            for line in fh:
                if "Block RAM Tile" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if parts:
                        try:
                            return float(parts[-1])
                        except:
                            continue
    except:
        return None
    return None

def parse_utilization_rpt(rpt_path: Path):
    """Extracts all Utilization % metrics from the Vivado .rpt file."""
    stats = {}
    if not rpt_path.exists():
        return None
    try:
        with open(rpt_path, "r") as f:
            for line in f:
                for rpt_label, clean_label in RESOURCE_MAP.items():
                    if f"| {rpt_label}" in line:
                        parts = [p.strip() for p in line.split("|") if p.strip()]
                        if len(parts) >= 5:
                            try:
                                stats[clean_label] = float(parts[-1])
                            except ValueError:
                                continue
    except Exception as e:
        print(f"Error parsing {rpt_path}: {e}")
        return None
    return stats

def get_config_report_path(cfg):
    """Utility to find the .rpt file for a specific config."""
    rpt_path = BITSTREAM_ROOT / cfg / "vivado_build" / "vivado_build.runs" / "impl_1" / "design_1_wrapper_utilization_placed.rpt"
    if not rpt_path.exists():
        matches = list(BITSTREAM_ROOT.glob(f"*{cfg}*/vivado_build/vivado_build.runs/impl_1/*.rpt"))
        if matches:
            rpt_path = matches[0]
    return rpt_path if rpt_path.exists() else None

def get_pareto_configs(df):
    """Identifies the unique set of Pareto-optimal configs across all densities."""
    all_pareto_configs = set()
    
    for density in df["density"].unique():
        df_d = df[df["density"] == density].copy()
        df_d["solve_s"] = df_d["solve_ms"] / 1000.0
        
        # Take min latency for each unique config (collapsing multi-solve noise)
        points = df_d.groupby("config", as_index=False).agg({
            "solve_s": "min", 
            "bram_util": "first"
        }).sort_values("bram_util")

        # Compute Pareto front: faster designs at similar/lower resource cost
        best_latency = float("inf")
        for _, row in points.iterrows():
            if row["solve_s"] < best_latency:
                all_pareto_configs.add(row["config"])
                best_latency = row["solve_s"]
                
    return list(all_pareto_configs)

def plot_resources(config_label, stats, out_dir):
    resources = list(stats.keys())
    values = list(stats.values())
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']
    
    x = np.arange(len(resources))
    
    # 1. Define very thin width and spacing
    bar_width = 0.15   
    bar_spacing = 0.25 
    x_positions = x * bar_spacing
    
    plt.figure(figsize=(7, 4)) 
    bars = plt.bar(x_positions, values, width=bar_width, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    # 2. THE SECRET: Add large padding to the x-axis limits
    # This prevents Matplotlib from stretching the bars to fill the window
    plt.xlim(-0.2, (len(resources) * bar_spacing))
    
    plt.xticks(x_positions, resources, fontsize=9)
    plt.ylim(0, 115)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.3)
    
    plt.ylabel("Utilization (%)", fontweight='bold', fontsize=11)
    plt.title(f"Hardware Resource Utilization: {config_label}", fontweight='bold', fontsize=12, pad=15)
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_dir / f"util_{config_label}.png", dpi=200)
    plt.close()

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    print("Loading data and resolving BRAM utilization...")
    df = pd.read_csv(CSV_PATH)
    
    # --- FIX: Create the bram_util column by checking the reports ---
    def resolve_bram(cfg):
        path = get_config_report_path(cfg)
        return parse_bram_from_rpt(path) if path else None

    # Apply the lookup to create the missing column
    df["bram_util"] = df["config"].apply(resolve_bram)
    df["bram_util"] = pd.to_numeric(df["bram_util"], errors="coerce")
    
    # Drop rows where we couldn't find a hardware report
    df = df.dropna(subset=["bram_util"])
    
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Identifying Global Pareto Configurations...")
    pareto_configs = get_pareto_configs(df)
    print(f"Found {len(pareto_configs)} unique Pareto-optimal designs.")

    for cfg in pareto_configs:
        rpt_path = get_config_report_path(cfg)
        if rpt_path:
            stats = parse_utilization_rpt(rpt_path)
            if stats:
                print(f"  Generating plot for: {cfg}")
                plot_resources(cfg, stats, OUTPUT_DIR)
        else:
            print(f"  Warning: No utilization report found for {cfg}")

if __name__ == "__main__":
    main()