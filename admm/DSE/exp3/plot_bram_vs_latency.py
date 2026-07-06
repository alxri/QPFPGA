# python plot_bram_vs_latency.py --label-offsets pareto_label_offsets_by_density.json --legend-offsets legend_offsets.json

#!/usr/bin/env python3
"""Plot BRAM utilization vs latency for exp3, split by density.

Usage:
    python DSE/exp3/plot_bram_vs_latency_exp3.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from adjustText import adjust_text  # <-- IMPORT THE NEW LIBRARY

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot BRAM utilization vs latency from the exp3 CSV.")
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("exp3_tile_size_sweep_results.csv"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("bram_vs_latency.png"))
    parser.add_argument("--metric", choices=["mean", "median"], default="mean")
    parser.add_argument(
        "--label-offsets",
        type=Path,
        default=None,
        help=("Optional JSON file with per-label offsets. "
              "Format: {\"config_label\": [dx, dy], ...} where dx/dy are in points."),
    )
    parser.add_argument(
        "--legend-offsets",
        type=Path,
        default=None,
        help=(
            'Optional JSON file with per-plot legend positions. '
            'Format: {"output_stem": {"loc": "upper left", "bbox_anchor": [x, y]}, ... } '
            'Coordinates for bbox_anchor are in axes fraction (0..1).'
        ),
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

    # Filter by success status
    if "status" in df.columns:
        status = df["status"]
        status_num = pd.to_numeric(status, errors="coerce")
        if status_num.notna().any():
            filtered = df[status_num.isin([0, 1])]
        else:
            status_text = status.astype(str).str.strip().str.lower()
            success_tokens = {"ok", "success", "optimal", "solved", "0", "1"}
            filtered = df[status_text.isin(success_tokens)]
        if filtered is not None and not filtered.empty:
            df = filtered

    df = df.copy()
    
    # Extract reshape factor
    df["reshape"] = df["config"].str.extract(r"reshape(\d+)").astype(float)
    df["reshape"] = df["reshape"].fillna(1) 
    
    df["solve_ms"] = pd.to_numeric(df["solve_ms"], errors="coerce")
    df["tile_size"] = pd.to_numeric(df["tile_size"], errors="coerce")
    df["pes"] = pd.to_numeric(df["pes"], errors="coerce")
    
    df = df.dropna(subset=["solve_ms", "tile_size", "pes", "config", "density"])
    df["solve_s"] = df["solve_ms"] / 1000.0

    # Load optional label offsets JSON. Supported formats:
    # 1) flat: {"config_label": [dx, dy], ...}
    # 2) per-density: {"low": {"config_label": [dx,dy], ...}, "high": {...}}
    # Offsets are interpreted as pixels for finer control.
    label_offsets_global: dict[str, list[float]] = {}
    label_offsets_by_density: dict[str, dict[str, list[float]]] = {}
    if args.label_offsets is not None:
        import json

        try:
            with open(args.label_offsets, "r") as lh:
                raw = json.load(lh)

            # Detect format
            if isinstance(raw, dict):
                # If values are lists of length 2, treat as flat global mapping
                sample_vals = list(raw.values())[:5]
                if all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in sample_vals):
                    for k, v in raw.items():
                        try:
                            dx, dy = float(v[0]), float(v[1])
                            label_offsets_global[str(k)] = [dx, dy]
                        except Exception:
                            print(f"Warning: invalid global label offset for {k}, expected [dx,dy]. Skipping.")
                else:
                    # Assume per-density mapping
                    for dens_k, dens_v in raw.items():
                        if not isinstance(dens_v, dict):
                            continue
                        label_offsets_by_density[ str(dens_k) ] = {}
                        for k, v in dens_v.items():
                            try:
                                dx, dy = float(v[0]), float(v[1])
                                label_offsets_by_density[str(dens_k)][str(k)] = [dx, dy]
                            except Exception:
                                print(f"Warning: invalid label offset for {k} under density {dens_k}. Skipping.")
            else:
                print(f"Warning: label offsets file {args.label_offsets} has unexpected format. Ignoring.")

        except FileNotFoundError:
            print(f"Label offsets file not found: {args.label_offsets}. Continuing without offsets.")
        except Exception as e:
            print(f"Error reading label offsets {args.label_offsets}: {e}. Continuing without offsets.")

    # Load optional legend offsets JSON (per-output file stem)
    legend_offsets: dict[str, dict] = {}
    if args.legend_offsets is not None:
        import json

        try:
            with open(args.legend_offsets, "r") as lh:
                raw_leg = json.load(lh)

            if isinstance(raw_leg, dict):
                for k, v in raw_leg.items():
                    if not isinstance(v, dict):
                        print(f"Warning: legend offset for {k} is not an object, skipping")
                        continue
                    entry: dict = {}
                    loc = v.get("loc")
                    if isinstance(loc, str):
                        entry["loc"] = loc
                    bbox = v.get("bbox_anchor")
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
                        try:
                            entry["bbox_anchor"] = (float(bbox[0]), float(bbox[1]))
                        except Exception:
                            print(f"Warning: invalid bbox_anchor for {k}, expected two numbers. Skipping bbox.")
                    legend_offsets[str(k)] = entry
            else:
                print(f"Warning: legend offsets file {args.legend_offsets} has unexpected format. Ignoring.")
        except FileNotFoundError:
            print(f"Legend offsets file not found: {args.legend_offsets}. Continuing without legend offsets.")
        except Exception as e:
            print(f"Error reading legend offsets {args.legend_offsets}: {e}. Continuing without legend offsets.")

    agg_fn = "mean" if args.metric == "mean" else "median"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    plt.style.use("seaborn-v0_8-whitegrid")

    density_nnz = {"low": 1, "med": 8, "medium": 8, "high": 32}

    bitstream_root = Path(__file__).resolve().parents[1] / "generated_bitstreams"
    rpt_cache = {}

    def bram_for_config(cfg_label: str):
        if cfg_label in rpt_cache: return rpt_cache[cfg_label]
        for d in [bitstream_root / cfg_label] + list(bitstream_root.glob(f"*{cfg_label}*")):
            candidate = d / "vivado_build" / "vivado_build.runs" / "impl_1" / "design_1_wrapper_utilization_placed.rpt"
            if candidate.exists():
                val = parse_bram_from_rpt(candidate)
                rpt_cache[cfg_label] = val
                return val
        rpt_cache[cfg_label] = None
        return None

    df["bram_util"] = df["config"].apply(lambda c: bram_for_config(str(c)))
    df["bram_util"] = pd.to_numeric(df["bram_util"], errors="coerce")
    df = df.dropna(subset=["bram_util"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    densities = sorted(df["density"].astype(str).unique())

    # Latency thresholds (seconds) for filtered plots per density
    latency_thresholds = {"high": 40.0, "medium": 8.0, "med": 8.0, "low": 5.0}

    def series_label(tile_size: float, reshape: float) -> str:
        return f"{int(tile_size)}x{int(tile_size)}, RF={int(reshape)}"

    def build_series_color_map(df_density: pd.DataFrame) -> dict[str, str]:
        labels = (
            df_density[["tile_size", "reshape"]]
            .drop_duplicates()
            .sort_values(["tile_size", "reshape"])
            .apply(lambda row: series_label(row["tile_size"], row["reshape"]), axis=1)
            .tolist()
        )
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
        return {label: cycle[i % len(cycle)] for i, label in enumerate(labels)}

    def render_density_plot(
        df_density: pd.DataFrame,
        out_path: Path,
        density_label: str,
        title_suffix: str,
        series_color_map: dict[str, str],
        legend_loc: str = "upper right",
    ) -> None:
        grouped = (
            df_density.groupby(["tile_size", "reshape", "bram_util"], as_index=False)
            .agg({"solve_s": agg_fn, "pes": "first"})
            .sort_values(["tile_size", "reshape", "bram_util"])
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        texts = []  # <-- LIST TO HOLD ALL LABELS

        for (t_size, rs), grp in grouped.groupby(["tile_size", "reshape"]):
            grp = grp.sort_values("bram_util")
            label_name = series_label(t_size, rs)
            color = series_color_map.get(label_name)

            ax.plot(
                grp["bram_util"],
                grp["solve_s"],
                marker="o",
                linewidth=2,
                label=label_name,
                color=color,
            )

            # Create text objects but don't place them permanently yet
            for xb, yb, p in zip(grp["bram_util"], grp["solve_s"], grp["pes"]):
                t = ax.text(
                    xb, yb, f"PEs={int(p)}",
                    fontsize=8, ha="center", va="center",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8, "ec": color}
                )
                texts.append(t)

        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.7))

        try:
            xmin, xmax = float(df_density["bram_util"].min()), float(df_density["bram_util"].max())
            if np.isfinite(xmin) and np.isfinite(xmax):
                start, end = max(0, int(np.floor(xmin / 5.0)) * 5), min(100, int(np.ceil(xmax / 5.0)) * 5)
                ax.set_xticks(np.arange(start, end + 1, 5))
                ax.set_xlim(start, end)
        except Exception:
            pass

        ax.set_xlabel("BRAM Utilization (%)", fontweight="bold")
        ax.set_ylabel("Latency [s]", fontweight="bold")
        ax.set_title(f"Tile Size, PEs, Reshape Factor vs Latency {title_suffix}", fontweight="bold", pad=15)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.7)
        # Allow per-output legend overrides from legend_offsets JSON (keyed by output stem)
        legend_cfg = legend_offsets.get(out_path.stem, {}) if isinstance(legend_offsets, dict) else {}
        cfg_loc = legend_cfg.get("loc", legend_loc)
        bbox = legend_cfg.get("bbox_anchor")
        if bbox is not None:
            ax.legend(
                title="",
                loc=cfg_loc,
                bbox_to_anchor=tuple(bbox),
                bbox_transform=ax.transAxes,
                prop={"size": 9, "weight": "normal"},
                title_fontproperties={"weight": "bold"},
                framealpha=0.85,
            )
        else:
            ax.legend(
                title="",
                loc=cfg_loc,
                prop={"size": 9, "weight": "normal"},
                title_fontproperties={"weight": "bold"},
                framealpha=0.85,
            )

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    def compute_pareto_from_min_latency(df_points: pd.DataFrame) -> pd.DataFrame:
        """Compute Pareto-optimal points by taking the minimum latency per BRAM utilization,
        then selecting the non-dominated set when sorting by increasing BRAM util.
        Returns a DataFrame of pareto points (columns include bram_util and solve_s).
        """
        if df_points.empty:
            return df_points.copy()

        # pick the lowest-latency row for each bram_util
        idx = df_points.groupby("bram_util")["solve_s"].idxmin()
        candidates = df_points.loc[idx].drop_duplicates(subset=["bram_util"]).copy()
        candidates = candidates.sort_values("bram_util").reset_index(drop=True)

        pareto_rows = []
        best_latency = float("inf")
        for _, row in candidates.iterrows():
            lat = float(row["solve_s"])
            if lat < best_latency:
                pareto_rows.append(row)
                best_latency = lat

        if pareto_rows:
            return pd.DataFrame(pareto_rows).reset_index(drop=True)
        return candidates.iloc[0:0].copy()

    def render_pareto_plot(
        df_density: pd.DataFrame,
        out_path: Path,
        density_label: str,
        title_suffix: str,
        legend_loc: str = "upper right",
    ) -> None:
        """Render a Pareto plot: all points (light), pareto curve (highlighted), and mark pareto vs non-pareto."""
        # Use the grouped points to collapse duplicates
        points = (
            df_density.groupby(["tile_size", "reshape", "bram_util", "pes", "config"], as_index=False)
            .agg({"solve_s": agg_fn})
            .sort_values("bram_util")
        )

        pareto = compute_pareto_from_min_latency(points)

        fig, ax = plt.subplots(figsize=(9, 6))

        # Plot all points as faint gray markers (non-pareto default)
        ax.scatter(points["bram_util"], points["solve_s"], color="#bbbbbb", s=40, zorder=1, label="All points")

        # Identify pareto indices in points
        if not pareto.empty:
            # plot pareto curve (connected) — no triangle markers
            ax.plot(pareto["bram_util"], pareto["solve_s"], color="#d62728", linewidth=2, zorder=3, label="Pareto Optimal Solutions")

            # mark non-pareto points that lie above the frontier as 'dominated'
            # Any point with solve_s > interpolated frontier at same or lower bram_util is dominated.
            # Simplify: a point is non-pareto if there exists a pareto point with bram_util <= and solve_s <=.
            dominated_mask = []
            pareto_list = pareto[["bram_util", "solve_s"]].values.tolist()
            for _, r in points.iterrows():
                dominated = False
                for bu, lat in pareto_list:
                    if (bu <= r["bram_util"]) and (lat <= r["solve_s"]):
                        # If strictly better in at least one, it's dominated; allow equality to mark dominated
                        if (bu < r["bram_util"]) or (lat < r["solve_s"]):
                            dominated = True
                            break
                dominated_mask.append(dominated)
            points["dominated"] = dominated_mask

            # Plot dominated points differently (blue outline)
            dom = points[points["dominated"]]
            non_dom = points[~points["dominated"]]
            if not dom.empty:
                ax.scatter(dom["bram_util"], dom["solve_s"], facecolors="none", edgecolors="#1f77b4", s=60, linewidths=1.2, zorder=2, label="Non Optimal Solutions")
            if not non_dom.empty:
                # Non-dominated (but not on pareto curve) — plot as smaller filled markers
                ax.scatter(non_dom["bram_util"], non_dom["solve_s"], color="#2ca02c", s=50, zorder=2, label="Optimal Solution")

            # Annotate pareto points. If a per-label offset exists in `label_offsets`, use it as pixels
            # for fine-grained manual control.
            for xb, yb, cfg in zip(pareto["bram_util"], pareto["solve_s"], pareto.get("config", pareto.index)):
                cfg_key = str(cfg)

                # Prefer a per-density user-provided offset; fall back to global offsets
                user_offset = None
                if density_label in label_offsets_by_density and cfg_key in label_offsets_by_density[density_label]:
                    user_offset = label_offsets_by_density[density_label][cfg_key]
                elif cfg_key in label_offsets_global:
                    user_offset = label_offsets_global[cfg_key]

                if user_offset is not None:
                    try:
                        # We use 'offset points' to be DPI independent
                        dx_pts, dy_pts = float(user_offset[0]), float(user_offset[1])
                        
                        ax.annotate(
                            cfg_key,
                            xy=(xb, yb),
                            xytext=(dx_pts, dy_pts),
                            textcoords="offset points", 
                            fontsize=10,
                            color="#800000",
                            ha="center", # Fixed anchor prevents massive jumping
                            va="center", # Fixed anchor prevents massive jumping
                        )
                        continue
                    except Exception:
                        pass

                # Default: place left-and-below, but adjust if it would go outside the axes
                xoff_pts, yoff_pts = -8, -8
                ha, va = "right", "top"

                # Compute display coordinates and ensure annotation stays inside axes
                disp_pt = ax.transData.transform((xb, yb))
                px_per_point = fig.dpi / 72.0
                disp_offset = np.array([xoff_pts * px_per_point, yoff_pts * px_per_point])
                annot_disp = disp_pt + disp_offset

                # If left placement would be outside axes, switch to right placement
                if not ax.bbox.contains(annot_disp[0], annot_disp[1]):
                    xoff_pts, yoff_pts = 6, -8
                    ha, va = "left", "top"
                    disp_offset = np.array([xoff_pts * px_per_point, yoff_pts * px_per_point])
                    annot_disp = disp_pt + disp_offset

                # If still outside, clamp into the bbox
                if not ax.bbox.contains(annot_disp[0], annot_disp[1]):
                    x_clamped = min(max(annot_disp[0], ax.bbox.x0 + 4), ax.bbox.x1 - 4)
                    y_clamped = min(max(annot_disp[1], ax.bbox.y0 + 4), ax.bbox.y1 - 4)
                    delta_px = np.array([x_clamped, y_clamped]) - disp_pt
                    xoff_pts = delta_px[0] / px_per_point
                    yoff_pts = delta_px[1] / px_per_point
                    ha = "left" if xoff_pts >= 0 else "right"
                    va = "bottom" if yoff_pts > 0 else "top"

                ax.annotate(
                    cfg_key,
                    xy=(xb, yb),
                    xytext=(xoff_pts, yoff_pts),
                    textcoords="offset points",
                    fontsize=10,
                    color="#800000",
                    ha=ha,
                    va=va,
                )

        ax.set_xlabel("BRAM Utilization (%)", fontweight="bold")
        ax.set_ylabel("Latency [s]", fontweight="bold")
        ax.set_title(f"Pareto Solutions {title_suffix}", fontweight="bold", pad=15)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.7)
        legend_cfg = legend_offsets.get(out_path.stem, {}) if isinstance(legend_offsets, dict) else {}
        cfg_loc = legend_cfg.get("loc", legend_loc)
        bbox = legend_cfg.get("bbox_anchor")
        if bbox is not None:
            ax.legend(
                title="",
                loc=cfg_loc,
                bbox_to_anchor=tuple(bbox),
                bbox_transform=ax.transAxes,
                prop={"size": 9},
                framealpha=0.85,
            )
        else:
            ax.legend(
                title="",
                loc=cfg_loc,
                prop={"size": 9},
                framealpha=0.85,
            )

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Pareto plot: {out_path}")

    for density in densities:
        df_density = df[df["density"].astype(str) == density].copy()
        if df_density.empty: continue

        series_color_map = build_series_color_map(df_density)

        out_path = args.out.with_name(f"{args.out.stem}_{density}{args.out.suffix}")
        nnz = density_nnz.get(density)
        title_suffix = f" ({density} density: {nnz} nnz/col)" if nnz is not None else f" ({density})"
        render_density_plot(df_density, out_path, density, title_suffix, series_color_map)

        # If this density has a configured latency threshold, produce an additional filtered plot
        if density in latency_thresholds:
            thr = float(latency_thresholds[density])
            df_under = df_density[df_density["solve_s"] <= thr].copy()
            if not df_under.empty:
                # 1) original-format under-x plot
                out_path_under = args.out.with_name(f"{args.out.stem}_{density}_under{int(thr)}s{args.out.suffix}")
                render_density_plot(
                    df_under,
                    out_path_under,
                    density,
                    f" ({density} density: {nnz} nnz/col)",
                    series_color_map,
                )

                # 2) pareto plot for the same under-x dataset
                out_path_under_pareto = args.out.with_name(f"{args.out.stem}_{density}_under{int(thr)}s_pareto{args.out.suffix}")
                render_pareto_plot(
                            df_under,
                            out_path_under_pareto,
                            density,
                            f" ({density} density: {nnz} nnz/col)",
                        )
            else:
                print(f"Skipped {density} under-{int(thr)}s plot because no rows met the latency threshold.")

if __name__ == "__main__":
    main()