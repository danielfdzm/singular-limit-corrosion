"""Regression, finite-element, and boundary analysis for the paper.

This module collects the short numerical-analysis routines used throughout the
experiments.  It depends only on the core solver and has no mesh-study
dependency.
"""

import math

import numpy as np

from .model import build_benchmarks
from .mesh import build_problem
from .diagnostics import (
    as_bool,
    bottom_trace,
    boundary_error_to_reference,
    evaluate_fe,
    evaluate_fe_unstructured,
    fractional_boundary_norm_from_midpoints,
)
from .storage import load_saved_solution, read_csv

# Plot styling, CSV utilities, and regression helpers

BLUE = "blue"
RED = "red"
BLACK = "#000000"
GREEN = "#006400"
GRAY = "#666666"

PLOT_TITLE_SIZE = 16
PLOT_TITLE_PAD = 8
PLOT_LABEL_SIZE = 12
PLOT_TICK_SIZE = 10
PLOT_LEGEND_SIZE = 11
PLOT_MARKER_SIZE = 6.0
PLOT_LINE_WIDTH = 2.0
PLOT_GUIDE_WIDTH = 1.8


def finite_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def fit_loglog(x, y):
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size < 2:
        return float("nan"), float("nan")
    design = np.column_stack([np.ones_like(x), np.log(x)])
    beta = np.linalg.lstsq(design, np.log(y), rcond=None)[0]
    return float(beta[1]), float(beta[0])


def bootstrap_loglog(
    x,
    y,
    samples=1000,
    seed=20260511,
):
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    slope, intercept = fit_loglog(x, y)
    if x.size < 2:
        grid = x.copy() if x.size else np.array([1.0], dtype=float)
        nan_grid = np.full_like(grid, np.nan, dtype=float)
        return {
            "x": x,
            "y": y,
            "grid": grid,
            "fit": nan_grid,
            "band_lo": nan_grid,
            "band_hi": nan_grid,
            "slope": slope,
            "intercept": intercept,
            "slope_lo": float("nan"),
            "slope_hi": float("nan"),
            "bootstrap_samples": 0,
        }
    grid = np.geomspace(float(np.min(x)), float(np.max(x)), 160)
    rng = np.random.default_rng(seed)
    slopes = []
    predictions = []
    for _ in range(samples):
        idx = rng.integers(0, x.size, size=x.size)
        if np.unique(x[idx]).size < 2:
            continue
        s, b = fit_loglog(x[idx], y[idx])
        if not (math.isfinite(s) and math.isfinite(b)):
            continue
        slopes.append(s)
        predictions.append(np.exp(b + s * np.log(grid)))
    if slopes:
        slope_ci = np.percentile(np.asarray(slopes), [2.5, 97.5])
    else:
        slope_ci = np.array([float("nan"), float("nan")])
    if predictions:
        pred = np.vstack(predictions)
        band = np.percentile(pred, [2.5, 97.5], axis=0)
    else:
        band = np.vstack([np.full_like(grid, np.nan), np.full_like(grid, np.nan)])
    fit_values = np.exp(intercept + slope * np.log(grid)) if math.isfinite(slope) else np.full_like(grid, np.nan)
    return {
        "x": x,
        "y": y,
        "grid": grid,
        "fit": fit_values,
        "band_lo": band[0],
        "band_hi": band[1],
        "slope": slope,
        "intercept": intercept,
        "slope_lo": float(slope_ci[0]),
        "slope_hi": float(slope_ci[1]),
        "bootstrap_samples": int(len(slopes)),
    }


def regression_row(name, fit, predicted):
    return {
        "diagnostic": name,
        "slope": float(fit["slope"]),
        "slope_ci_low": float(fit["slope_lo"]),
        "slope_ci_high": float(fit["slope_hi"]),
        "predicted_slope": predicted,
        "bootstrap_samples": int(fit["bootstrap_samples"]),
    }


def add_fit(
    ax,
    fit,
    color,
    data_label,
    fit_label,
    marker="o",
):
    x = np.asarray(fit["x"], dtype=float)
    y = np.asarray(fit["y"], dtype=float)
    grid = np.asarray(fit["grid"], dtype=float)
    ax.loglog(x, y, marker=marker, linestyle="", color=color, markersize=PLOT_MARKER_SIZE, label=data_label)
    ax.loglog(grid, np.asarray(fit["fit"], dtype=float), color=color, linewidth=PLOT_LINE_WIDTH, label=fit_label)
    ax.fill_between(
        grid,
        np.asarray(fit["band_lo"], dtype=float),
        np.asarray(fit["band_hi"], dtype=float),
        color=color,
        alpha=0.16,
        linewidth=0.0,
    )


def add_slope_guide(
    ax,
    x,
    y,
    slope,
    label,
    color=GRAY,
):
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 1:
        return
    x = x[mask]
    y = y[mask]
    anchor = int(np.nanargmin(x))
    x0 = float(x[anchor])
    y0 = float(y[anchor])
    grid = np.geomspace(float(np.min(x)), float(np.max(x)), 100)
    ax.loglog(grid, y0 * (grid / x0) ** slope, "--", color=color, linewidth=PLOT_GUIDE_WIDTH, label=label)


def add_connected_points(
    ax,
    x,
    y,
    color,
    label,
    marker="o",
):
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 1:
        return
    xs = np.asarray(x[mask], dtype=float)
    ys = np.asarray(y[mask], dtype=float)
    order = np.argsort(xs)
    ax.loglog(
        xs[order],
        ys[order],
        marker=marker,
        linestyle="-",
        color=color,
        linewidth=PLOT_LINE_WIDTH,
        markersize=PLOT_MARKER_SIZE,
        label=label,
    )


# Finite-element traces, gradients, and reference errors

def boundary_hquarter_error_to_equilibrium(problem, u):
    edges = problem.bottom_edges
    n0 = edges[:, 0]
    n1 = edges[:, 1]
    x_mid = 0.5 * (problem.nodes[n0, 0] + problem.nodes[n1, 0])
    values = 0.5 * (u[n0] + u[n1]) - problem.bottom_edge_values
    return fractional_boundary_norm_from_midpoints(x_mid, problem.bottom_lengths, values, s=0.25)


def boundary_hquarter_error_to_reference_general(
    problem,
    u,
    ref_problem,
    ref_u,
):
    x, values = bottom_trace(problem, u)
    ref_x, ref_values = bottom_trace(ref_problem, ref_u)
    grid = np.unique(np.concatenate([x, ref_x]))
    if grid.size < 2:
        return 0.0
    x_mid = 0.5 * (grid[:-1] + grid[1:])
    weights = grid[1:] - grid[:-1]
    diff = np.interp(x_mid, x, values) - np.interp(x_mid, ref_x, ref_values)
    return fractional_boundary_norm_from_midpoints(x_mid, weights, diff, s=0.25)


def evaluate_fe_gradient(problem, u, points):
    xs = points[:, 0]
    ys = points[:, 1]
    i = np.clip(np.searchsorted(problem.x, xs, side="right") - 1, 0, problem.x.size - 2)
    j = np.clip(np.searchsorted(problem.y, ys, side="right") - 1, 0, problem.y.size - 2)
    x0 = problem.x[i]
    x1 = problem.x[i + 1]
    y0 = problem.y[j]
    y1 = problem.y[j + 1]
    xi = (xs - x0) / (x1 - x0)
    eta = (ys - y0) / (y1 - y0)
    nx = problem.x.size
    n00 = j * nx + i
    n10 = j * nx + (i + 1)
    n01 = (j + 1) * nx + i
    n11 = (j + 1) * nx + (i + 1)
    grad = np.empty((points.shape[0], 2), dtype=float)
    lower = eta <= xi
    if np.any(lower):
        grad[lower, 0] = (u[n10[lower]] - u[n00[lower]]) / (x1[lower] - x0[lower])
        grad[lower, 1] = (u[n11[lower]] - u[n10[lower]]) / (y1[lower] - y0[lower])
    upper = ~lower
    if np.any(upper):
        grad[upper, 0] = (u[n11[upper]] - u[n01[upper]]) / (x1[upper] - x0[upper])
        grad[upper, 1] = (u[n01[upper]] - u[n00[upper]]) / (y1[upper] - y0[upper])
    return grad


def full_domain_points(parameters, nx=241, ny=121):
    xs = np.linspace(0.5 * parameters.lx / nx, parameters.lx - 0.5 * parameters.lx / nx, nx)
    ys = np.linspace(0.5 * parameters.ly / ny, parameters.ly - 0.5 * parameters.ly / ny, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def weighted_error_to_reference(problem, u, ref_problem, ref_u, kappa, points):
    grad = evaluate_fe_gradient(problem, u, points)
    ref_grad = evaluate_fe_gradient(ref_problem, ref_u, points)
    grad_sq = problem.physics.lx * problem.physics.ly * float(np.mean(np.sum((grad - ref_grad) ** 2, axis=1)))
    boundary = boundary_error_to_reference(problem, u, ref_problem, ref_u)
    return math.sqrt(max(grad_sq + boundary * boundary / kappa, 0.0))


def evaluate_fe_general(problem, u, points):
    if problem.mesh_strategy != "local_point_graded":
        return evaluate_fe(problem, u, points)
    return evaluate_fe_unstructured(problem, u, points)


def boundary_error_to_reference_general(
    problem,
    u,
    ref_problem,
    ref_u,
    num_points=801,
):
    x_eval = np.linspace(0.0, problem.physics.lx, num_points)
    x = problem.nodes[problem.bottom_nodes, 0]
    ref_x = ref_problem.nodes[ref_problem.bottom_nodes, 0]
    val = np.interp(x_eval, x, u[problem.bottom_nodes])
    ref_val = np.interp(x_eval, ref_x, ref_u[ref_problem.bottom_nodes])
    return float(math.sqrt(np.trapezoid((val - ref_val) ** 2, x_eval)))


def bulk_error_to_reference_general(
    problem,
    u,
    ref_problem,
    ref_u,
    points,
):
    val = evaluate_fe_general(problem, u, points)
    ref_val = evaluate_fe_general(ref_problem, ref_u, points)
    if not (np.all(np.isfinite(val)) and np.all(np.isfinite(ref_val))):
        raise RuntimeError("Interior sample points fell outside an unstructured triangulation.")
    return float(
        math.sqrt(
            (0.5 * problem.physics.lx) * (0.5 * problem.physics.ly)
            * float(np.mean((val - ref_val) ** 2))
        )
    )


# Boundary diagnostics and interior-window summaries

def boundary_hquarter_values_from_rows(
    problem,
    rows,
    data_dir,
):
    values = []
    for row in rows:
        kappa = finite_float(row.get("kappa"))
        if not math.isfinite(kappa):
            values.append(float("nan"))
            continue
        solution = load_saved_solution(problem, kappa, data_dir)
        if solution is None:
            solution = load_saved_solution(problem, kappa, data_dir)
        if solution is None:
            values.append(float("nan"))
            continue
        values.append(boundary_hquarter_error_to_equilibrium(problem, solution))
    return np.asarray(values, dtype=float)


def candidate_hmins_from_rows(parameters, rows):
    candidates = []
    for key in ("h_min", "reference_h_min"):
        for row in rows:
            hmin = finite_float(row.get(key))
            if math.isfinite(hmin):
                candidates.append(hmin)
    candidates.extend(
        [
            parameters.suite_hmin,
            parameters.single_suite_hmin,
            parameters.exact_refinement_cap,
            2.5e-7,
            1.0e-7,
            7.5e-8,
            5.0e-8,
            2.0e-8,
            1.0e-8,
        ]
    )
    unique = []
    for hmin in candidates:
        if math.isfinite(hmin) and all(not math.isclose(hmin, old, rel_tol=0.0, abs_tol=1.0e-14) for old in unique):
            unique.append(hmin)
    return unique


def build_matching_graded_problem(
    parameters,
    benchmark,
    rows,
):
    target_nodes = int(finite_float(rows[0].get("num_nodes"))) if rows else -1
    fallback = None
    for hmin in candidate_hmins_from_rows(parameters, rows):
        problem = build_problem(parameters, benchmark, "graded", hmin=hmin)
        fallback = problem
        if target_nodes > 0 and int(problem.nodes.shape[0]) == target_nodes:
            return problem
    if fallback is None:
        fallback = build_problem(parameters, benchmark, "graded", hmin=parameters.suite_hmin)
    return fallback


def load_single_problem(parameters, data_dir):
    benchmarks = build_benchmarks(parameters)
    benchmark = benchmarks["single_junction_mixed"]
    problem = build_problem(parameters, benchmark, "graded", hmin=parameters.single_suite_hmin)
    reference_path = data_dir / "u0_single_junction_mixed.npz"
    with np.load(reference_path) as archive:
        u0 = np.asarray(archive["potential"], dtype=float)
    return benchmark, problem, u0


def compact_points(parameters, box, nx=141, ny=71):
    x0, x1, y0, y1 = box
    xs = np.linspace(x0 * parameters.lx, x1 * parameters.lx, nx)
    ys = np.linspace(y0 * parameters.ly, y1 * parameters.ly, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def box_area(parameters, box):
    x0, x1, y0, y1 = box
    return (x1 - x0) * parameters.lx * (y1 - y0) * parameters.ly


def compute_interior_window_summary(parameters, data_dir):
    convergence_rows = read_csv(data_dir / "convergence_summary.csv")
    _, problem, u0 = load_single_problem(parameters, data_dir)
    boxes = [
        ("near-boundary", (0.10, 0.90, 0.10, 0.90)),
        ("middle", (0.25, 0.75, 0.25, 0.75)),
        ("core", (0.40, 0.60, 0.40, 0.60)),
    ]
    rows = []
    missing_solution_kappas = []
    for row in convergence_rows:
        if row.get("benchmark") != "single_junction_mixed" or not as_bool(row.get("certified", False)):
            continue
        kappa = finite_float(row["kappa"])
        if not (1.3e-7 <= kappa <= 1.0e-4):
            continue
        solution = load_saved_solution(problem, kappa, data_dir)
        if solution is None:
            missing_solution_kappas.append(kappa)
            continue
        for label, box in boxes:
            points = compact_points(parameters, box)
            values = evaluate_fe(problem, solution, points)
            ref = evaluate_fe(problem, u0, points)
            grad_values = evaluate_fe_gradient(problem, solution, points)
            grad_ref = evaluate_fe_gradient(problem, u0, points)
            pointwise = np.abs(values - ref)
            grad_pointwise = np.sqrt(np.sum((grad_values - grad_ref) ** 2, axis=1))
            err = math.sqrt(box_area(parameters, box) * float(np.mean((values - ref) ** 2)))
            rows.append(
                {
                    "box": label,
                    "x0": box[0],
                    "x1": box[1],
                    "y0": box[2],
                    "y1": box[3],
                    "kappa": kappa,
                    "kappa_log_kappa": kappa * abs(math.log(kappa)),
                    "bulk_error_l2": err,
                    "bulk_error_l2_sq": err * err,
                    "bulk_error_c0": float(np.max(pointwise)),
                    "bulk_error_c1": float(np.max(grad_pointwise)),
                }
            )
    if missing_solution_kappas:
        missing = ", ".join(f"{kappa:g}" for kappa in sorted(set(missing_solution_kappas), reverse=True))
        raise FileNotFoundError(
            "Missing saved single-junction solution files for certified dashboard kappas: "
            f"{missing}. Regenerate those NPZ files or remove the rows from the certified dashboard window."
        )
    rows.sort(key=lambda r: (str(r["box"]), -float(r["kappa"])))
    return rows
