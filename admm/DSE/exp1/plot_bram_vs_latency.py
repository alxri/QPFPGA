#!/usr/bin/env python3
"""Plot NUM_PEs vs latency with separate colors for adaptive_rho values.

Usage:
    python DSE/exp1/plot_pe_vs_latency.py
    python DSE/exp1/plot_pe_vs_latency.py --csv DSE/exp1/exp1_pe_sweep_results.csv --out DSE/exp1/pe_vs_latency.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from pathlib import Path as _Path
import re
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot NUM_PEs vs latency from a DSE CSV with adaptive_rho coloring."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("exp1_pe_sweep_results.csv"),
        help="Path to CSV (default: DSE/exp1_pe_sweep_results.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("pe_vs_latency.png"),
        help="Output image path (default: DSE/pe_vs_latency.png)",
    )
    parser.add_argument(
        "--metric",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation metric for latency per (pes, adaptive_rho).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    required_cols = {"pes", "adaptive_rho", "solve_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"CSV must contain columns: {sorted(required_cols)}. Missing: {missing_str}"
        )

    # Keep only successful solves with valid latency.
    if "status" in df.columns:
        status = df["status"]
        filtered = None

        status_num = pd.to_numeric(status, errors="coerce")
        if status_num.notna().any():
            # In these experiment CSVs, status 0 denotes success.
            filtered = df[status_num == 0]
        else:
            status_text = status.astype(str).str.strip().str.lower()
            success_tokens = {"ok", "success", "optimal", "solved", "0"}
            failure_tokens = {
                "fail",
                "failed",
                "error",
                "infeasible",
                "timeout",
            }
            filtered = df[status_text.isin(success_tokens)]
            # If no explicit success token exists, drop only known failures.
            if filtered.empty:
                filtered = df[~status_text.isin(failure_tokens)]

        if not filtered.empty:
            df = filtered

    df = df.copy()
    df["solve_ms"] = pd.to_numeric(df["solve_ms"], errors="coerce")
    df = df.dropna(subset=["solve_ms", "pes", "adaptive_rho"])

    if df.empty:
        raise ValueError("No valid rows left after filtering.")

    agg_fn = "mean" if args.metric == "mean" else "median"
    # Ensure consistent font across all text elements
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    plt.style.use("seaborn-v0_8-whitegrid")

    color_map = {0: "#1f77b4"}

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if "density" in df.columns:
        densities = sorted(df["density"].astype(str).unique())
    else:
        densities = [None]

    for density in densities:
        if density is None:
            df_density = df
            out_path = args.out
            title_suffix = ""
        else:
            df_density = df[df["density"].astype(str) == density]
            out_path = args.out.with_name(f"{args.out.stem}_{density}{args.out.suffix}")
            # Map density name to approximate nnz per column
            DENSITY_NNZ = {"low": 1, "med": 8, "medium": 8, "high": 32}
            nnz = DENSITY_NNZ.get(density, None)
            if nnz is not None:
                title_suffix = f"({density} density: {nnz} nnz/col)"
            else:
                title_suffix = f"({density} density)"

        # Only plot adaptive_rho = 0 as requested.
        df_density = df_density[pd.to_numeric(df_density["adaptive_rho"], errors="coerce") == 0]

        # Resolve BRAM utilization per config (from Vivado report), add as new column
        def parse_bram_from_rpt(rpt_path: _Path):
            try:
                with rpt_path.open("r") as fh:
                    for line in fh:
                        if "Block RAM Tile" in line:
                            parts = [p.strip() for p in line.split("|") if p.strip()]
                            if parts:
                                try:
                                    return float(parts[-1])
                                except Exception:
                                    continue
            except Exception:
                return None
            return None

        base_bitstreams = _Path(__file__).parents[1] / "generated_bitstreams"

        rpt_cache = {}

        def bram_for_config(cfg_label: str):
            if cfg_label in rpt_cache:
                return rpt_cache[cfg_label]
            # expected path: generated_bitstreams/{cfg_label}/vivado_build/vivado_build.runs/impl_1/design_1_wrapper_utilization_placed.rpt
            candidate = base_bitstreams / cfg_label / "vivado_build" / "vivado_build.runs" / "impl_1" / "design_1_wrapper_utilization_placed.rpt"
            if candidate.exists():
                val = parse_bram_from_rpt(candidate)
                rpt_cache[cfg_label] = val
                return val
            # fallback: try glob match for folder containing cfg_label
            for d in base_bitstreams.glob(f"*{cfg_label}*"):
                candidate = d / "vivado_build" / "vivado_build.runs" / "impl_1" / "design_1_wrapper_utilization_placed.rpt"
                if candidate.exists():
                    val = parse_bram_from_rpt(candidate)
                    rpt_cache[cfg_label] = val
                    return val
            # fallback: check for BRAM_% column in original df (if present)
            if "BRAM_%" in df_density.columns:
                try:
                    val = float(df_density[df_density["config"] == cfg_label]["BRAM_%"].iloc[0])
                    rpt_cache[cfg_label] = val
                    return val
                except Exception:
                    pass
            rpt_cache[cfg_label] = None
            return None

        df_density = df_density.copy()
        df_density["bram_util"] = df_density["config"].apply(lambda c: bram_for_config(str(c)))
        df_density["bram_util"] = pd.to_numeric(df_density["bram_util"], errors="coerce")
        # Drop rows without resolved BRAM utilization
        df_density = df_density.dropna(subset=["bram_util"]) 

        # Build a mapping from bram_util to representative NUM_PEs for annotation
        try:
            pes_map = (
                df_density.groupby("bram_util")["pes"]
                .agg(lambda s: int(pd.to_numeric(s, errors="coerce").mode().iat[0]) if not s.mode().empty else int(s.iloc[0]))
                .to_dict()
            )
        except Exception:
            pes_map = { }

        # Aggregate by BRAM utilization instead of NUM_PEs
        grouped = (
            df_density.groupby(["adaptive_rho", "bram_util"], as_index=False)["solve_ms"]
            .agg(agg_fn)
            .sort_values(["adaptive_rho", "bram_util"])
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        for rho, grp in grouped.groupby("adaptive_rho"):
            rho_int = int(rho)
            ax.plot(
                grp["bram_util"],
                grp["solve_ms"],
                marker="o",
                linewidth=2,
                color=color_map.get(rho_int, None),
            )

            # Annotate each point with NUM_PEs label using a plain line (no arrowhead).
            # Place labels left or right depending on x relative to group median to reduce overlap.
            # Use previous left/right placement to reduce overlap. For specific PEs (16,20,30,40)
            # place the label directly above the point.
            median_x = grp["bram_util"].median()
            for xb, yb in zip(grp["bram_util"], grp["solve_ms"]):
                pes_val = pes_map.get(xb)
                if pes_val is None:
                    continue
                special_above = int(pes_val) in {16, 20, 30, 40, 50, 60}
                if special_above:
                    dx, dy = 0, 8
                    ha = "center"
                    va = "bottom"
                    # nudge NUM_PEs=20 slightly to the right
                    if int(pes_val) == 16:
                        dx = -10
                    if int(pes_val) == 20:
                        dx = 18
                else:
                    if xb > median_x:
                        dx, dy = -40, 6
                        ha = "right"
                        va = "center"
                    else:
                        dx, dy = 10, 6
                        ha = "left"
                        va = "center"
                # additional small nudge if not special_above and pes==20
                if not special_above and int(pes_val) == 20:
                    try:
                        dx = int(dx) + 15
                    except Exception:
                        pass

                ax.annotate(
                    f"PEs={int(pes_val)}",
                    xy=(xb, yb),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    ha=ha,
                    va=va,
                    arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": color_map.get(rho_int, None)} if not special_above else None,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7},
                )

        # Set x-axis ticks in 5% steps across the data range
        try:
            xmin = float(df_density["bram_util"].min())
            xmax = float(df_density["bram_util"].max())
            if np.isfinite(xmin) and np.isfinite(xmax):
                start = max(0, int(np.floor(xmin / 5.0)) * 5)
                end = min(100, int(np.ceil(xmax / 5.0)) * 5)
                ticks = np.arange(start, end + 1, 5)
                ax.set_xticks(ticks)
        except Exception:
            pass

        ax.set_xlabel("BRAM Utilization (%)")
        ax.set_ylabel(f"Latency [ms]")
        ax.set_yscale("linear")
        ax.set_title(f"Number of PEs vs Latency {title_suffix}", fontweight="bold")
        # Format y-axis labels without scientific notation
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        # Add grid for readability (major ticks only, slightly thicker)
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.7)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
