#!/usr/bin/env python3
"""Plot BRAM utilization vs latency for exp2, split by density.

The plot uses the generated bitstream Vivado utilization reports to recover
BRAM utilization (%) for each config and labels each point with its reshape
factor.

Usage:
    python DSE/exp2/plot_bram_vs_latency.py
    python DSE/exp2/plot_bram_vs_latency.py --csv DSE/exp2/exp2_reshape_sweep_results.csv --out DSE/exp2/bram_vs_latency.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot BRAM utilization vs latency from the exp2 CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("exp2_reshape_sweep_results.csv"),
        help="Path to CSV (default: DSE/exp2/exp2_reshape_sweep_results.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("bram_vs_latency.png"),
        help="Output image path (default: DSE/exp2/bram_vs_latency.png)",
    )
    parser.add_argument(
        "--metric",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation metric for latency per (reshape_factor, bram_util).",
    )
    return parser


def parse_bram_from_rpt(rpt_path: Path):
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


def main() -> None:
    args = build_parser().parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    required_cols = {"config", "reshape_factor", "solve_ms", "density"}
    missing = required_cols - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"CSV must contain columns: {sorted(required_cols)}. Missing: {missing_str}"
        )

    if "status" in df.columns:
        status = df["status"]
        filtered = None

        status_num = pd.to_numeric(status, errors="coerce")
        if status_num.notna().any():
            filtered = df[status_num == 0]
        else:
            status_text = status.astype(str).str.strip().str.lower()
            success_tokens = {"ok", "success", "optimal", "solved", "0"}
            failure_tokens = {"fail", "failed", "error", "infeasible", "timeout"}
            filtered = df[status_text.isin(success_tokens)]
            if filtered.empty:
                filtered = df[~status_text.isin(failure_tokens)]

        if filtered is not None and not filtered.empty:
            df = filtered

    df = df.copy()
    df["solve_ms"] = pd.to_numeric(df["solve_ms"], errors="coerce")
    df["reshape_factor"] = pd.to_numeric(df["reshape_factor"], errors="coerce")
    df = df.dropna(subset=["solve_ms", "reshape_factor", "config", "density"])

    if df.empty:
        raise ValueError("No valid rows left after filtering.")

    agg_fn = "mean" if args.metric == "mean" else "median"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    plt.style.use("seaborn-v0_8-whitegrid")

    color_map = {0: "#1f77b4"}
    density_nnz = {"low": 1, "med": 8, "medium": 8, "high": 32}

    bitstream_root = Path(__file__).resolve().parents[1] / "generated_bitstreams"
    rpt_cache = {}

    def bram_for_config(cfg_label: str):
        if cfg_label in rpt_cache:
            return rpt_cache[cfg_label]

        candidate = (
            bitstream_root
            / cfg_label
            / "vivado_build"
            / "vivado_build.runs"
            / "impl_1"
            / "design_1_wrapper_utilization_placed.rpt"
        )
        if candidate.exists():
            val = parse_bram_from_rpt(candidate)
            rpt_cache[cfg_label] = val
            return val

        for d in bitstream_root.glob(f"*{cfg_label}*"):
            candidate = (
                d
                / "vivado_build"
                / "vivado_build.runs"
                / "impl_1"
                / "design_1_wrapper_utilization_placed.rpt"
            )
            if candidate.exists():
                val = parse_bram_from_rpt(candidate)
                rpt_cache[cfg_label] = val
                return val

        rpt_cache[cfg_label] = None
        return None

    df["bram_util"] = df["config"].apply(lambda c: bram_for_config(str(c)))
    df["bram_util"] = pd.to_numeric(df["bram_util"], errors="coerce")
    df = df.dropna(subset=["bram_util"])

    if df.empty:
        raise ValueError("No rows with resolved BRAM utilization were found.")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    densities = sorted(df["density"].astype(str).unique())

    for density in densities:
        df_density = df[df["density"].astype(str) == density].copy()
        if df_density.empty:
            continue

        out_path = args.out.with_name(f"{args.out.stem}_{density}{args.out.suffix}")
        nnz = density_nnz.get(density)
        if nnz is not None:
            title_suffix = f" ({density} density: {nnz} nnz/col)"
        else:
            title_suffix = f" ({density} density)"

        grouped = (
            df_density.groupby(["reshape_factor", "bram_util"], as_index=False)["solve_ms"]
            .agg(agg_fn)
            .sort_values(["reshape_factor", "bram_util"])
        )

        # Representative reshape factor per BRAM point.
        reshape_map = (
            df_density.groupby("bram_util")["reshape_factor"]
            .agg(lambda s: int(pd.to_numeric(s, errors="coerce").mode().iat[0]) if not s.mode().empty else int(s.iloc[0]))
            .to_dict()
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        for reshape_factor, grp in grouped.groupby("reshape_factor"):
            reshape_int = int(reshape_factor)
            ax.plot(
                grp["bram_util"],
                grp["solve_ms"],
                marker="o",
                linewidth=2,
                color=color_map.get(0, None),
            )

            median_x = grp["bram_util"].median()
            for xb, yb in zip(grp["bram_util"], grp["solve_ms"]):
                reshape_val = reshape_map.get(xb)
                if reshape_val is None:
                    continue

                special_above = int(reshape_val) in {1, 2, 4, 8}
                if special_above:
                    dx, dy = 0, 8
                    ha = "center"
                    va = "bottom"
                else:
                    if xb > median_x:
                        dx, dy = -40, 6
                        ha = "right"
                        va = "center"
                    else:
                        dx, dy = 10, 6
                        ha = "left"
                        va = "center"

                ax.annotate(
                    f"RF={int(reshape_val)}",
                    xy=(xb, yb),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    ha=ha,
                    va=va,
                    arrowprops={
                        "arrowstyle": "-",
                        "linewidth": 0.6,
                        "color": color_map.get(0, None),
                    }
                    if not special_above
                    else None,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7},
                )

        try:
            xmin = float(df_density["bram_util"].min())
            xmax = float(df_density["bram_util"].max())
            if np.isfinite(xmin) and np.isfinite(xmax):
                start = max(0, int(np.floor(xmin / 5.0)) * 5)
                end = min(100, int(np.ceil(xmax / 5.0)) * 5)
                ax.set_xticks(np.arange(start, end + 1, 5))
                ax.set_xlim(start, end)
        except Exception:
            pass

        ax.set_xlabel("BRAM Utilization (%)")
        ax.set_ylabel("Latency [ms]")
        ax.set_yscale("linear")
        ax.set_title(f"Reshape Factor vs Latency {title_suffix}", fontweight="bold")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.7)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
