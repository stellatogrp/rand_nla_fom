"""Reduce ``results.csv`` and ``histories.csv`` to the tables the figures read.

``curves.csv``
    Percentiles over instances of each per-iteration metric, per
    configuration, against three abscissae: the outer iteration, the
    cumulative number of inner iterations, and the elapsed time.  After an
    instance has stopped its final value is carried forward, since the
    iterate no longer changes; ``active`` counts the instances still running.
    A history that records only some iterations (gradient descent) is
    sampled as a step function of the iteration number.
``summary.csv``
    One row per configuration with the moments of the scalar measurements
    and the outcome of every theorem check.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from ._io import read_columns, read_rows, write_rows

CURVES = "curves.csv"
SUMMARY = "summary.csv"
PERCENTILES = (10, 25, 50, 75, 90)
METRICS = (
    "suboptimality",
    "suboptimality_z",
    "feasibility",
    "z_norm",
    "dist",
    "inner",
    "bound_surrogate",
    "bound_exact",
    "residual_exact",
)
AXES = {"outer": "iteration", "inner": "cumulative_inner", "time": "elapsed"}
SCALARS = (
    "k_delta",
    "outer_iterations",
    "total_inner",
    "total_checks",
    "gradient_evaluations",
    "max_inner_per_outer",
    "mean_inner_per_outer",
    "runtime_seconds",
    "setup_seconds",
    "parameter_seconds",
    "calibration_seconds",
    "truth_seconds",
    "final_suboptimality",
    "final_feasibility",
    "final_z_norm",
    "thm43_bound",
    "thm42_bound",
    "thm31_bound",
)
nan = float("nan")


def config_key(row: dict) -> tuple:
    """Identify a configuration; ``sigma`` is rounded so floats compare."""
    sigma = row["sigma"]
    sigma = (
        nan
        if sigma is None or (isinstance(sigma, float) and math.isnan(sigma))
        else round(float(sigma), 10)
    )
    return (str(row["method"]), str(row["sweep"]), sigma, int(row["b"]))


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _grid_outer(max_iteration: int, dense: int = 200, points: int = 300) -> np.ndarray:
    if max_iteration <= dense + points:
        return np.arange(max_iteration + 1)
    sparse = np.unique(np.round(np.geomspace(dense, max_iteration, points)).astype(int))
    return np.unique(np.concatenate([np.arange(dense), sparse, [max_iteration]]))


def _grid_log(maximum: float, points: int = 150) -> np.ndarray:
    if not np.isfinite(maximum) or maximum <= 1:
        return np.array([maximum]) if np.isfinite(maximum) else np.array([])
    grid = np.geomspace(1.0, maximum, points)
    return np.unique(np.concatenate([grid, [maximum]]))


def _sample(x_instance: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Values of one instance on ``grid``, carrying the final value forward.

    The value at a grid point is that of the last record whose abscissa is
    at most the grid point, which also serves histories that record only
    some iterations (gradient descent); before the first record it is nan.
    """
    index = np.searchsorted(x_instance, grid, side="right") - 1
    return np.where(index >= 0, values[np.clip(index, 0, len(values) - 1)], nan)


def _flags(directory: Path) -> dict[tuple, dict]:
    """Per configuration: whether it is a grid point and which value is optimal."""
    flags: dict[tuple, dict] = {}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in read_rows(directory / "results.csv"):
        groups[config_key(row)].append(row)
    for key, rows in groups.items():
        flags[key] = dict(
            sigma_is_optimal=any(bool(r.get("sigma_is_optimal")) for r in rows),
            b_optimal_fraction=float(
                np.mean([bool(r.get("b_is_optimal")) for r in rows])
            ),
            b_exact_optimal_fraction=float(
                np.mean([bool(r.get("b_exact_is_optimal")) for r in rows])
            ),
            in_grid=any(bool(r.get("in_grid", True)) for r in rows),
        )
    return flags


def aggregate_curves(directory: Path) -> list[dict]:
    table = read_columns(directory / "histories.csv")
    if not table:
        return []
    flags = _flags(directory)
    keys = np.array(
        [
            f"{m}|{s}|{'nan' if math.isnan(sig) else round(float(sig), 10)}|{int(b)}"
            for m, s, sig, b in zip(
                table["method"], table["sweep"], table["sigma"], table["b"], strict=True
            )
        ],
        dtype=object,
    )
    seeds = table["seed"]
    rows: list[dict] = []
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        method, sweep, sigma_text, b_text = key.split("|")
        sigma = nan if sigma_text == "nan" else float(sigma_text)
        identity = dict(method=method, sweep=sweep, sigma=sigma, b=int(b_text))
        identity.update(
            flags.get(
                config_key(identity),
                dict(
                    sigma_is_optimal=False,
                    b_optimal_fraction=0.0,
                    b_exact_optimal_fraction=0.0,
                    in_grid=True,
                ),
            )
        )
        instances = []
        for seed in np.unique(seeds[mask]):
            pick = mask & (seeds == seed)
            order = np.argsort(table["iteration"][pick])
            instances.append(
                {
                    name: table[name][pick][order]
                    for name in list(METRICS) + list(AXES.values())
                }
            )
        for axis, column in AXES.items():
            if axis == "outer":
                grid = _grid_outer(int(max(inst[column][-1] for inst in instances)))
            else:
                grid = _grid_log(float(max(inst[column][-1] for inst in instances)))
            if grid.size == 0:
                continue
            ends = np.array([inst[column][-1] for inst in instances])
            active = (ends[None, :] >= grid[:, None]).sum(axis=1)
            for metric in METRICS:
                stacked = np.stack(
                    [
                        _sample(
                            inst[column],
                            np.abs(inst[metric])
                            if metric.startswith("subopt")
                            else inst[metric],
                            grid,
                        )
                        for inst in instances
                    ],
                    axis=1,
                )
                if np.all(np.isnan(stacked)):
                    continue
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    percentiles = np.nanpercentile(stacked, PERCENTILES, axis=1)
                    means = np.nanmean(stacked, axis=1)
                    counts = np.sum(~np.isnan(stacked), axis=1)
                for i, x in enumerate(grid):
                    if counts[i] == 0:
                        continue
                    rows.append(
                        dict(
                            **identity,
                            axis=axis,
                            metric=metric,
                            x=float(x),
                            n=int(counts[i]),
                            active=int(active[i]),
                            mean=float(means[i]),
                            **{
                                f"p{p}": float(percentiles[j, i])
                                for j, p in enumerate(PERCENTILES)
                            },
                        )
                    )
    return rows


def _moments(values: list[float]) -> dict:
    array = np.array([v for v in values if not _is_nan(v)], dtype=float)
    if array.size == 0:
        return dict(mean=nan, sd=nan, median=nan, p10=nan, p90=nan, min=nan, max=nan)
    # Infinite values (the theorem bounds at sigma = 1) give infinite moments
    # and a nan standard deviation; the arithmetic on them is deliberate.
    with np.errstate(invalid="ignore"):
        return dict(
            mean=float(array.mean()),
            sd=float(array.std(ddof=1)) if array.size > 1 else 0.0,
            median=float(np.median(array)),
            p10=float(np.percentile(array, 10)),
            p90=float(np.percentile(array, 90)),
            min=float(array.min()),
            max=float(array.max()),
        )


def aggregate_summary(directory: Path) -> list[dict]:
    results = read_rows(directory / "results.csv")
    if not results:
        return []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in results:
        groups[config_key(row)].append(row)
    summary: list[dict] = []
    for key in sorted(
        groups, key=lambda k: (k[0], k[1], k[2] if not _is_nan(k[2]) else -1, k[3])
    ):
        rows = groups[key]
        method, sweep, sigma, b = key
        stops = [str(r.get("stopped_by", "")) for r in rows]
        entry = dict(
            method=method,
            sweep=sweep,
            sigma=sigma,
            b=b,
            sigma_is_optimal=any(bool(r.get("sigma_is_optimal")) for r in rows),
            b_optimal_fraction=float(
                np.mean([bool(r.get("b_is_optimal")) for r in rows])
            ),
            b_exact_optimal_fraction=float(
                np.mean([bool(r.get("b_exact_is_optimal")) for r in rows])
            ),
            in_grid=any(bool(r.get("in_grid", True)) for r in rows),
            n_runs=len(rows),
            n_converged=sum(bool(r.get("converged")) for r in rows),
            n_max_total_inner=stops.count("max_total_inner"),
            n_time_limit=stops.count("time_limit"),
            n_max_outer=stops.count("max_outer"),
        )
        for name in SCALARS:
            values = [r.get(name, nan) for r in rows]
            if name == "final_suboptimality":
                values = [abs(v) if not _is_nan(v) else v for v in values]
            for stat, value in _moments(values).items():
                entry[f"{name}_{stat}"] = value

        def column(name: str, rows: list[dict] = rows) -> np.ndarray:
            return np.array([r.get(name, nan) for r in rows], dtype=float)

        thm31 = column("thm31_ratio")
        lemma32 = column("lemma32_min_ratio")
        thm43 = column("thm43_ratio")
        thm42 = column("thm42_ratio")
        violations = column("per_step_violations")
        violations_exact = column("per_step_violations_exact")
        with np.errstate(all="ignore"):
            entry.update(
                thm31_ratio_max=float(np.nanmax(thm31))
                if np.any(~np.isnan(thm31))
                else nan,
                thm31_holds=int(np.sum(thm31 <= 1.0)),
                lemma32_min_ratio_min=float(np.nanmin(lemma32))
                if np.any(~np.isnan(lemma32))
                else nan,
                lemma32_holds=int(np.sum(lemma32 >= 1.0 - 1e-9)),
                lemma32_checked_total=int(np.nansum(column("lemma32_checked"))),
                thm43_ratio_median=float(np.nanmedian(thm43))
                if np.any(~np.isnan(thm43))
                else nan,
                thm43_ratio_max=float(np.nanmax(thm43))
                if np.any(~np.isnan(thm43))
                else nan,
                thm43_holds=int(np.sum(thm43 <= 1.0)),
                thm42_ratio_median=float(np.nanmedian(thm42))
                if np.any(~np.isnan(thm42))
                else nan,
                thm42_ratio_max=float(np.nanmax(thm42))
                if np.any(~np.isnan(thm42))
                else nan,
                thm42_holds=int(np.sum(thm42 <= 1.0)),
                per_step_violations_total=float(np.nansum(violations))
                if np.any(~np.isnan(violations))
                else nan,
                per_step_max_ratio_max=float(np.nanmax(column("per_step_max_ratio")))
                if np.any(~np.isnan(column("per_step_max_ratio")))
                else nan,
                per_step_violations_exact_total=float(np.nansum(violations_exact))
                if np.any(~np.isnan(violations_exact))
                else nan,
                kappa_empirical_median=float(np.nanmedian(column("kappa_empirical")))
                if np.any(~np.isnan(column("kappa_empirical")))
                else nan,
                rate_bound_median=float(np.nanmedian(column("rate_bound")))
                if np.any(~np.isnan(column("rate_bound")))
                else nan,
                rate_observed_max_max=float(np.nanmax(column("rate_observed_max")))
                if np.any(~np.isnan(column("rate_observed_max")))
                else nan,
                rate_observed_median_median=float(
                    np.nanmedian(column("rate_observed_median"))
                )
                if np.any(~np.isnan(column("rate_observed_median")))
                else nan,
            )
        summary.append(entry)
    return summary


def aggregate_study(directory: Path) -> tuple[Path, Path]:
    """Write ``curves.csv`` and ``summary.csv`` next to the raw tables."""
    directory = Path(directory)
    curves = aggregate_curves(directory)
    summary = aggregate_summary(directory)
    if curves:
        write_rows(directory / CURVES, curves)
    if summary:
        write_rows(directory / SUMMARY, summary)
    return directory / CURVES, directory / SUMMARY


def main() -> None:
    import argparse

    from .study import DEFAULT_RESULTS_DIR, STUDIES

    parser = argparse.ArgumentParser(description="rebuild curves.csv and summary.csv")
    parser.add_argument(
        "studies", nargs="*", default=list(STUDIES), choices=list(STUDIES) + [[]]
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    for study in args.studies or list(STUDIES):
        print(aggregate_study(args.results_dir / study))


if __name__ == "__main__":
    main()
