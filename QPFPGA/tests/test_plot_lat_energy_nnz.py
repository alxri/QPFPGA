import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "osqp_benchmarks"
    / "results_final"
    / "plot_lat_energy_nnz.py"
)

spec = importlib.util.spec_from_file_location("plot_lat_energy_nnz", MODULE_PATH)
plot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_module)


def test_sanitize_plottable_series_filters_invalid_values_for_log_scale():
    x = pd.Series([10, 20, 30], dtype=float)
    y = pd.Series([1.0, 0.0, np.nan], dtype=float)

    cleaned_x, cleaned_y = plot_module.sanitize_plottable_series(x, y, log_scale=True)

    assert cleaned_x.tolist() == [10.0]
    assert cleaned_y.tolist() == [1.0]
