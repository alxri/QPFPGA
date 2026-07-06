#!/usr/bin/env python3
"""
Global Utilization Plotter
1. Loads CSV to find all unique configurations.
2. Crawls Vivado folders to find utilization reports.
3. Extracts and plots detailed FPGA resource utilization for ALL configs.
   (Separates LUTs into Logic and Memory)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
CSV_PATH = Path("exp3_tile_size_sweep_results.csv")
BITSTREAM_ROOT = Path("../generated_bitstreams")
OUTPUT_DIR = Path("tmp")

# Map Vivado report labels (exactly as they appear after stripping spaces) to clean plot labels
RESOURCE_MAP = {
    "LUT as Logic": "LUTs (Logic)",
    "LUT as Memory": "LUTs (Memory)",
    "CLB Registers": "FFs",
    "Block RAM Tile": "BRAM",
    "URAM": "URAM",
    "DSPs": "DSPs"
}

def parse_utilization_rpt(rpt_path: Path):
    """Extracts all Utilization % metrics from the Vivado .rpt file."""
    stats = {}
    if not rpt_path.exists():
        return None
    try:
        with open(rpt_path, "r") as f:
            for line in f:
                # Skip borders and empty lines
                if not line.startswith("|") or "+--" in line:
                    continue
                
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 5:
                    row_name = parts[0]
                    # Check if the stripped row name matches our map
                    if row_name in RESOURCE_MAP:
                        clean_label = RESOURCE_MAP[row_name]
                        try:
                            # The last column in this table is the 'Util%'
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

def plot_resources(config_label, stats, out_dir):
    # Ensure we plot in a consistent order based on RESOURCE_MAP
    resources = []
    values = []
    for clean_label in RESOURCE_MAP.values():
        if clean_label in stats:
            resources.append(clean_label)
            values.append(stats[clean_label])
            
    # Expanded color palette to account for extra categories
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']
    
    x = np.arange(len(resources))
    
    # 1. Define very thin width and spacing
    bar_width = 0.15   
    bar_spacing = 0.25 
    x_positions = x * bar_spacing
    
    plt.figure(figsize=(7, 4)) 
    bars = plt.bar(x_positions, values, width=bar_width, color=colors[:len(resources)], alpha=0.85, edgecolor='black', linewidth=0.8)
    
    # 2. Add large padding to the x-axis limits to prevent bar stretching
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
    plt.savefig(out_dir / f"util_{config_label}.png", dpi=300)
    plt.close()

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    
    # Ensure tmp directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Identifying All Configurations...")
    all_configs = df["config"].unique()
    print(f"Found {len(all_configs)} unique designs.")

    for cfg in all_configs:
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