"""Error measures, certification checks, and shared plotting styles."""

import math

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


GAUSS_XI = np.array(
    [
        0.06943184420297371,
        0.33000947820757187,
        0.6699905217924281,
        0.9305681557970262,
    ]
)
GAUSS_W = np.array(
    [
        0.17392742256872693,
        0.32607257743127307,
        0.32607257743127307,
        0.17392742256872693,
    ]
)

DIAGNOSTIC_KAPPA_MAX = 1.0e-4
DIAGNOSTIC_KAPPA_MIN = 1.0e-7
DIAGNOSTIC_BLUE = "blue"
GUIDE_RED = "red"
DIAGNOSTIC_BLACK = "#000000"
DIAGNOSTIC_GREEN = "#006400"
POTENTIAL_CMAP = plt.get_cmap("RdBu_r")
PLOT_TITLE_SIZE = 16
PLOT_TITLE_PAD = 8
PLOT_LABEL_SIZE = 12
PLOT_TICK_SIZE = 10
PLOT_LEGEND_SIZE = 11
PLOT_MARKER_SIZE = 6.0
PLOT_LINE_WIDTH = 2.0
PLOT_GUIDE_WIDTH = 1.8


def boundary_l2_error_sq(u, problem):
    ref = problem.bottom_edge_values
    n0 = problem.bottom_edges[:, 0]
    n1 = problem.bottom_edges[:, 1]
    v0 = u[n0] - ref
    v1 = u[n1] - ref
    return float(np.sum(problem.bottom_lengths * (v0 * v0 + v0 * v1 + v1 * v1) / 3.0))


def boundary_hhalf_norm(u, problem):
    bottom = u[problem.bottom_nodes]
    x = problem.x
    edges = list(zip(x[:-1], x[1:], bottom[:-1], bottom[1:]))

    l2_sq = float(
        np.sum(
            (x[1:] - x[:-1])
            * (bottom[:-1] * bottom[:-1] + bottom[:-1] * bottom[1:] + bottom[1:] * bottom[1:])
            / 3.0
        )
    )

    seminorm = 0.0
    for i, (xa0, xa1, ua0, ua1) in enumerate(edges):
        da = ua1 - ua0
        seminorm += da * da
        la = xa1 - xa0
        for j in range(i + 1, len(edges)):
            xb0, xb1, ub0, ub1 = edges[j]
            lb = xb1 - xb0
            pair_val = 0.0
            for si, wi in zip(GAUSS_XI, GAUSS_W):
                xs = xa0 + la * si
                us = (1.0 - si) * ua0 + si * ua1
                for tj, wj in zip(GAUSS_XI, GAUSS_W):
                    xt = xb0 + lb * tj
                    ut = (1.0 - tj) * ub0 + tj * ub1
                    pair_val += wi * wj * (us - ut) ** 2 / (xs - xt) ** 2
            seminorm += 2.0 * la * lb * pair_val

    return float(math.sqrt(max(l2_sq + seminorm, 0.0)))


def normalized_energy(energy, kappa):
    return energy / abs(math.log(kappa)) if kappa < 1.0 else energy


def cathode_anode_junction_count(benchmark):
    count = 0
    intervals = sorted(benchmark.intervals, key=lambda item: item[0])
    for left, right in zip(intervals, intervals[1:]):
        if left[2] != right[2] and math.isclose(left[1], right[0], rel_tol=0.0, abs_tol=1.0e-14):
            count += 1
    return count


def exact_log_energy_constant(benchmark, parameters):
    jump_count = cathode_anode_junction_count(benchmark)
    jump_size = parameters.phi_c - parameters.phi_a
    return jump_count * jump_size * jump_size / (2.0 * math.pi)


def exact_constant_ratio(energy, kappa, benchmark, parameters):
    constant = exact_log_energy_constant(benchmark, parameters)
    if constant <= 0.0 or kappa >= 1.0:
        return float("nan")
    return normalized_energy(energy, kappa) / constant


def max_butler_volmer_exponent_argument(u, parameters):
    args = (
        parameters.c1 * (u - parameters.phi_c),
        -parameters.c2 * (u - parameters.phi_c),
        parameters.a2 * (u - parameters.phi_a),
        -parameters.a1 * (u - parameters.phi_a),
    )
    return float(max(np.max(np.abs(arg)) for arg in args))


def certified_residual_threshold(parameters, initial_residual=None):
    base = parameters.certified_residual_factor * parameters.newton_tol
    if initial_residual is None or not math.isfinite(initial_residual):
        return base
    return base * max(1.0, float(initial_residual))


def is_certified_info(info, parameters):
    threshold = float(
        info.get(
            "certified_threshold",
            certified_residual_threshold(parameters, float(info.get("initial_residual_inf", float("nan")))),
        )
    )
    return float(info["residual_inf"]) <= threshold


def is_admissible_row(row, parameters, tol=1.0e-8):
    if "solution_min" not in row or "solution_max" not in row:
        return True
    try:
        solution_min = float(row["solution_min"])
        solution_max = float(row["solution_max"])
    except (TypeError, ValueError):
        return False
    return solution_min >= parameters.phi_a - tol and solution_max <= parameters.phi_c + tol


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def certified_rows(rows, parameters):
    return [
        row
        for row in rows
        if float(row["kappa"]) < 1.0 and as_bool(row.get("certified", True))
        and is_admissible_row(row, parameters)
    ]


def bold_legend_frame(legend):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_edgecolor(DIAGNOSTIC_BLACK)
    frame.set_linewidth(1.4)
    frame.set_alpha(1.0)


def style_plot_axes(ax):
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=PLOT_TICK_SIZE)


def set_panel_title(ax, title):
    ax.set_title(title, fontsize=PLOT_TITLE_SIZE, pad=PLOT_TITLE_PAD)


def theorem_diagnostic_rows(rows, parameters):
    return [
        row
        for row in certified_rows(rows, parameters)
        if DIAGNOSTIC_KAPPA_MIN <= float(row["kappa"]) <= DIAGNOSTIC_KAPPA_MAX
    ]


def anchored_scale(kappa_arr, value_arr, reference_arr):
    if not kappa_arr.size:
        return float("nan")
    idx = int(np.nanargmin(kappa_arr))
    return float(value_arr[idx] / reference_arr[idx])


def bulk_sample_points(parameters, nx=121, ny=61):
    xs = np.linspace(parameters.bulk_box_x0 * parameters.lx, parameters.bulk_box_x1 * parameters.lx, nx)
    ys = np.linspace(parameters.bulk_box_y0 * parameters.ly, parameters.bulk_box_y1 * parameters.ly, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def bulk_area(parameters):
    return (
        (parameters.bulk_box_x1 - parameters.bulk_box_x0)
        * (parameters.bulk_box_y1 - parameters.bulk_box_y0)
        * parameters.lx
        * parameters.ly
    )


def evaluate_fe_unstructured(problem, u, points):
    triangulation = mtri.Triangulation(problem.nodes[:, 0], problem.nodes[:, 1], problem.triangles)
    interpolator = mtri.LinearTriInterpolator(triangulation, u)
    values = interpolator(points[:, 0], points[:, 1])
    return np.asarray(values.filled(np.nan), dtype=float)


def evaluate_fe(problem, u, points):
    if problem.benchmark.geometry != "rectangle":
        return evaluate_fe_unstructured(problem, u, points)

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

    values = np.empty(points.shape[0], dtype=float)
    lower = eta <= xi
    upper = ~lower

    if np.any(lower):
        lam0 = 1.0 - xi[lower]
        lam1 = xi[lower] - eta[lower]
        lam2 = eta[lower]
        values[lower] = lam0 * u[n00[lower]] + lam1 * u[n10[lower]] + lam2 * u[n11[lower]]

    if np.any(upper):
        lam0 = 1.0 - eta[upper]
        lam1 = xi[upper]
        lam2 = eta[upper] - xi[upper]
        values[upper] = lam0 * u[n00[upper]] + lam1 * u[n11[upper]] + lam2 * u[n01[upper]]

    return values


def bulk_error_to_reference(
    problem,
    u,
    ref_problem,
    u_ref,
    points,
):
    val = evaluate_fe(problem, u, points)
    ref_val = evaluate_fe(ref_problem, u_ref, points)
    return float(math.sqrt(bulk_area(problem.physics) * np.mean((val - ref_val) ** 2)))


def boundary_error_to_reference(
    problem,
    u,
    ref_problem,
    u_ref,
    num_points=801,
):
    x_eval = np.linspace(0.0, problem.physics.lx, num_points)
    val = np.interp(x_eval, problem.x, u[problem.bottom_nodes])
    ref_val = np.interp(x_eval, ref_problem.x, u_ref[ref_problem.bottom_nodes])
    return float(math.sqrt(np.trapezoid((val - ref_val) ** 2, x_eval)))


def fractional_boundary_norm_from_midpoints(
    x_mid,
    weights,
    values,
    s=0.25,
    chunk=512,
):
    mask = np.isfinite(x_mid) & np.isfinite(weights) & np.isfinite(values) & (weights > 0.0)
    x_mid = np.asarray(x_mid[mask], dtype=float)
    weights = np.asarray(weights[mask], dtype=float)
    values = np.asarray(values[mask], dtype=float)
    if x_mid.size == 0:
        return float("nan")
    order = np.argsort(x_mid)
    x_mid = x_mid[order]
    weights = weights[order]
    values = values[order]
    l2_sq = float(np.sum(weights * values * values))
    exponent = 1.0 + 2.0 * s
    seminorm = 0.0
    for start in range(0, x_mid.size, chunk):
        stop = min(start + chunk, x_mid.size)
        dist = np.abs(x_mid[start:stop, None] - x_mid[None, :])
        diff = values[start:stop, None] - values[None, :]
        block_weights = weights[start:stop, None] * weights[None, :]
        block = np.zeros_like(dist)
        valid = dist > 0.0
        block[valid] = block_weights[valid] * diff[valid] ** 2 / dist[valid] ** exponent
        seminorm += float(np.sum(block))
    return float(math.sqrt(max(l2_sq + seminorm, 0.0)))


def bottom_trace(problem, u):
    x = np.asarray(problem.nodes[problem.bottom_nodes, 0], dtype=float)
    values = np.asarray(u[problem.bottom_nodes], dtype=float)
    order = np.argsort(x)
    return x[order], values[order]


def boundary_hquarter_error_to_reference(
    problem,
    u,
    ref_problem,
    u_ref,
):
    x, values = bottom_trace(problem, u)
    ref_x, ref_values = bottom_trace(ref_problem, u_ref)
    grid = np.unique(np.concatenate([x, ref_x]))
    if grid.size < 2:
        return 0.0
    x_mid = 0.5 * (grid[:-1] + grid[1:])
    weights = grid[1:] - grid[:-1]
    diff = np.interp(x_mid, x, values) - np.interp(x_mid, ref_x, ref_values)
    return fractional_boundary_norm_from_midpoints(x_mid, weights, diff, s=0.25)


def triangle_gradient_magnitude(problem, u):
    tri = problem.triangles
    pts = problem.nodes[tri]
    vals = u[tri]
    x0, y0 = pts[:, 0, 0], pts[:, 0, 1]
    x1, y1 = pts[:, 1, 0], pts[:, 1, 1]
    x2, y2 = pts[:, 2, 0], pts[:, 2, 1]
    u0, u1, u2 = vals[:, 0], vals[:, 1], vals[:, 2]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    grad_x = (u0 * (y1 - y2) + u1 * (y2 - y0) + u2 * (y0 - y1)) / det
    grad_y = (u0 * (x2 - x1) + u1 * (x0 - x2) + u2 * (x1 - x0)) / det
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)
