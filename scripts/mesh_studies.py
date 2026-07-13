"""Local mesh construction, convergence studies, and mesh figures."""

import math

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from scipy import sparse

from .numerical_analysis import (
    BLACK,
    BLUE,
    GRAY,
    GREEN,
    PLOT_LABEL_SIZE,
    PLOT_LEGEND_SIZE,
    PLOT_LINE_WIDTH,
    PLOT_MARKER_SIZE,
    PLOT_TICK_SIZE,
    RED,
    add_fit,
    add_slope_guide,
    bootstrap_loglog,
    boundary_error_to_reference_general,
    boundary_hquarter_error_to_reference_general,
    bulk_error_to_reference_general,
    compact_points,
    finite_float,
    full_domain_points,
    regression_row,
    weighted_error_to_reference,
)
from .model import (
    Problem,
    benchmark_junction_points,
    build_benchmarks,
    corrugated_stress_physics,
    corrugated_top_height,
    interval_label_for_point,
    reactive_value,
)
from .mesh import build_problem, triangle_stiffness
from .diagnostics import (
    as_bool,
    bold_legend_frame,
    boundary_l2_error_sq,
    certified_residual_threshold,
    exact_constant_ratio,
    exact_log_energy_constant,
    normalized_energy,
    set_panel_title,
    style_plot_axes,
)
from .nonlinear_solver import solve_with_continuation
from .storage import read_csv, write_csv

# Local junction meshes

def mesh_edge_segments_from_grid(
    x,
    y,
    parameters,
    benchmark,
):
    nx = x.size
    ny = y.size
    xx, yy = np.meshgrid(x, y, indexing="xy")
    if benchmark.geometry == "corrugated_top":
        height = corrugated_top_height(xx, parameters)
        nodes = np.column_stack([xx.ravel(), (yy * height / parameters.ly).ravel()])
    else:
        nodes = np.column_stack([xx.ravel(), yy.ravel()])

    edges = []
    for j in range(ny):
        row = j * nx
        edges.extend((row + i, row + i + 1) for i in range(nx - 1))
    for j in range(ny - 1):
        row = j * nx
        next_row = (j + 1) * nx
        edges.extend((row + i, next_row + i) for i in range(nx))
    for j in range(ny - 1):
        row = j * nx
        next_row = (j + 1) * nx
        edges.extend((row + i, next_row + i + 1) for i in range(nx - 1))
    points = 1.0e3 * nodes
    return points[np.asarray(edges, dtype=int)]


def unique_reference_points(points):
    return np.unique(np.round(np.asarray(points, dtype=float), 12), axis=0)


def local_junction_reference_points(
    parameters,
    anchors,
    radius,
    radial_layers,
    angular_layers,
    power,
    background_x_count,
    background_y_count,
):
    points = [
        (float(x), float(y))
        for x in np.linspace(0.0, parameters.lx, background_x_count)
        for y in np.linspace(0.0, parameters.ly, background_y_count)
    ]
    for anchor in anchors:
        points.append((anchor, 0.0))
        for m in range(1, radial_layers + 1):
            radius_m = radius * (m / radial_layers) ** power
            for theta in np.linspace(0.0, np.pi, angular_layers + 1):
                xval = anchor + radius_m * np.cos(theta)
                yval = radius_m * np.sin(theta)
                if 0.0 <= xval <= parameters.lx and 0.0 <= yval <= parameters.ly:
                    points.append((float(xval), float(yval)))
    return unique_reference_points(points)


def adaptive_local_junction_reference_points(
    parameters,
    anchors,
    hmin,
    radius,
    angular_layers,
    power,
    background_x_count,
    background_y_count,
):
    points = [
        (float(x), float(y))
        for x in np.linspace(0.0, parameters.lx, background_x_count)
        for y in np.linspace(0.0, parameters.ly, background_y_count)
    ]
    for anchor in anchors:
        points.append((anchor, 0.0))
        m = 1
        while True:
            radius_m = hmin * (m**power)
            if radius_m > radius:
                break
            for theta in np.linspace(0.0, np.pi, angular_layers + 1):
                xval = anchor + radius_m * np.cos(theta)
                yval = radius_m * np.sin(theta)
                if 0.0 <= xval <= parameters.lx and 0.0 <= yval <= parameters.ly:
                    points.append((float(xval), float(yval)))
            m += 1
    return unique_reference_points(points)


def physical_points_from_reference(points, parameters):
    height = corrugated_top_height(points[:, 0], parameters)
    mapped_y = points[:, 1] * height / parameters.ly
    return np.column_stack([points[:, 0], mapped_y])


def physical_points_for_benchmark(points, parameters, benchmark):
    if benchmark.geometry == "rectangle":
        return points.copy()
    if benchmark.geometry == "corrugated_top":
        return physical_points_from_reference(points, parameters)
    raise ValueError(f"Unknown geometry: {benchmark.geometry}")


def triangulated_edge_segments_from_reference_points(points, parameters):
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
    triangles = np.asarray(triangulation.triangles, dtype=int)
    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    physical_points = 1.0e3 * physical_points_from_reference(points, parameters)
    return physical_points[edges]


def build_local_junction_problem(
    parameters,
    benchmark,
    hmin,
    mesh_strategy="local_point_graded",
):
    reference_points = adaptive_local_junction_reference_points(
        parameters,
        benchmark_junction_points(benchmark),
        hmin,
        radius=4.0e-3,
        angular_layers=16,
        power=2.0,
        background_x_count=25,
        background_y_count=10,
    )
    triangulation = mtri.Triangulation(reference_points[:, 0], reference_points[:, 1])
    nodes = physical_points_for_benchmark(reference_points, parameters, benchmark)
    triangles = np.asarray(triangulation.triangles, dtype=int)
    rows = []
    cols = []
    data = []
    for tri in triangles:
        local = triangle_stiffness(nodes[tri])
        for a in range(3):
            for b in range(3):
                rows.append(int(tri[a]))
                cols.append(int(tri[b]))
                data.append(float(local[a, b]))
    stiffness = sparse.coo_matrix((data, (rows, cols)), shape=(nodes.shape[0], nodes.shape[0])).tocsr()

    bottom_nodes = np.flatnonzero(np.isclose(reference_points[:, 1], 0.0, atol=1.0e-13))
    bottom_nodes = bottom_nodes[np.argsort(reference_points[bottom_nodes, 0])]
    bottom_edges = np.column_stack([bottom_nodes[:-1], bottom_nodes[1:]])
    bottom_lengths = np.linalg.norm(nodes[bottom_edges[:, 1]] - nodes[bottom_edges[:, 0]], axis=1)
    bottom_x = reference_points[bottom_nodes, 0]
    midpoints = 0.5 * (bottom_x[:-1] + bottom_x[1:])
    edge_labels = np.array(
        [interval_label_for_point(float(midpoint), benchmark.intervals) for midpoint in midpoints],
        dtype="<U1",
    )
    bottom_edge_labels = np.where(edge_labels == "c", 0, 1).astype(int)
    bottom_edge_values = np.array([reactive_value(label, parameters) for label in edge_labels], dtype=float)
    bottom_node_values = np.empty(bottom_nodes.size, dtype=float)
    bottom_node_values[0] = bottom_edge_values[0]
    bottom_node_values[-1] = bottom_edge_values[-1]
    if bottom_nodes.size > 2:
        bottom_node_values[1:-1] = 0.5 * (bottom_edge_values[:-1] + bottom_edge_values[1:])

    return Problem(
        benchmark=benchmark,
        physics=parameters,
        mesh_strategy=mesh_strategy,
        x=bottom_x,
        y=np.unique(reference_points[:, 1]),
        nodes=nodes,
        triangles=triangles,
        stiffness=stiffness,
        bottom_nodes=bottom_nodes,
        bottom_edges=bottom_edges,
        bottom_lengths=bottom_lengths,
        bottom_edge_labels=bottom_edge_labels,
        bottom_edge_values=bottom_edge_values,
        bottom_node_values=bottom_node_values,
    )


# Mesh-convergence and Newton-scaling studies

def run_mesh_rate_two_axis(parameters, output_dir, data_dir):
    benchmarks = build_benchmarks(parameters)
    benchmark = benchmarks["single_junction_mixed"]
    kappas = (1.0e-3, 1.0e-4, 1.0e-5)
    ratios = (0.05, 0.10, 0.25, 0.50, 1.00)
    reference_factor = 0.0125
    points = full_domain_points(parameters)
    rows = []

    for kappa in kappas:
        ref_hmin = reference_factor * kappa
        ref_problem = build_problem(parameters, benchmark, "graded", hmin=ref_hmin)
        print(f"[mesh-rate] kappa={kappa:.1e} reference hmin={ref_hmin:.2e} N={ref_problem.nodes.shape[0]}")
        ref_solution, ref_info = solve_with_continuation(ref_problem, kappa)
        for ratio in ratios:
            hmin = ratio * kappa
            problem = build_problem(parameters, benchmark, "graded", hmin=hmin)
            print(f"[mesh-rate] kappa={kappa:.1e} ratio={ratio:.2f} hmin={hmin:.2e} N={problem.nodes.shape[0]}")
            solution, info = solve_with_continuation(problem, kappa)
            weighted = weighted_error_to_reference(problem, solution, ref_problem, ref_solution, kappa, points)
            residual = float(info["residual_inf"])
            threshold = float(info.get("certified_threshold", 1.0e-9))
            rows.append(
                {
                    "benchmark": benchmark.name,
                    "kappa": kappa,
                    "h_min": hmin,
                    "h_min_over_kappa": ratio,
                    "num_nodes": int(problem.nodes.shape[0]),
                    "num_triangles": int(problem.triangles.shape[0]),
                    "reference_h_min": ref_hmin,
                    "reference_nodes": int(ref_problem.nodes.shape[0]),
                    "weighted_error_kappa": weighted,
                    "scaled_prefactor": weighted / math.sqrt(hmin * abs(math.log(kappa))),
                    "iterations": float(info["iterations"]),
                    "residual_inf": residual,
                    "initial_residual_inf": float(info.get("initial_residual_inf", float("nan"))),
                    "certified_threshold": threshold,
                    "certified": residual <= threshold,
                    "solver": str(info["solver"]),
                    "reference_residual_inf": float(ref_info["residual_inf"]),
                }
            )
    rows.sort(key=lambda r: (float(r["kappa"]), float(r["h_min_over_kappa"])), reverse=True)
    write_csv(rows, data_dir / "mesh_rate_two_axis_summary.csv")
    plot_mesh_rate_two_axis(rows, output_dir / "mesh_rate_two_axis.pdf", data_dir)
    return rows


def plot_mesh_rate_two_axis(rows, outfile, data_dir):
    kappas = sorted({finite_float(r["kappa"]) for r in rows}, reverse=True)
    ratios = sorted({finite_float(r["h_min_over_kappa"]) for r in rows})
    fig = plt.figure(figsize=(10.6, 7.0))
    gs = fig.add_gridspec(3, 2, left=0.075, right=0.985, bottom=0.085, top=0.935, hspace=0.48, wspace=0.34)
    left_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    right_ax = fig.add_subplot(gs[:, 1])
    summary_rows = []
    for ax, kappa in zip(left_axes, kappas):
        selected_all = [r for r in rows if math.isclose(finite_float(r["kappa"]), kappa)]
        selected = [r for r in selected_all if as_bool(r.get("certified", False))]
        failed = [r for r in selected_all if not as_bool(r.get("certified", False))]
        if failed:
            failed_x = np.array([finite_float(r["h_min"]) for r in failed])
            failed_y = np.array([finite_float(r["weighted_error_kappa"]) for r in failed])
            ax.loglog(
                failed_x,
                failed_y,
                marker="x",
                linestyle="",
                color=GRAY,
                markersize=5.5,
                label="uncertified",
            )
        x = np.array([finite_float(r["h_min"]) for r in selected])
        y = np.array([finite_float(r["weighted_error_kappa"]) for r in selected])
        if x.size >= 3:
            fit = bootstrap_loglog(x, y, samples=500)
            add_fit(
                ax,
                fit,
                color=BLUE,
                data_label="certified",
                fit_label="regression",
                marker="o",
            )
            add_slope_guide(ax, x, y, 0.5, "slope 1/2 guide")
        else:
            fit = bootstrap_loglog(np.array([], dtype=float), np.array([], dtype=float), samples=0)
            if x.size:
                ax.loglog(
                    x,
                    y,
                    marker="o",
                    linestyle="",
                    color=BLUE,
                    markersize=PLOT_MARKER_SIZE,
                    label="certified",
                )
        ax.grid(False)
        ax.set_title(rf"$\kappa={kappa:g}$, $p={float(fit['slope']):.2f}$")
        ax.set_xlabel(r"$h_{\min}$")
        ax.set_ylabel(r"$\|\phi_{\kappa,h}-\phi_{\kappa,h_*}\|_\kappa$")
        summary_rows.append(regression_row(f"mesh_rate_kappa_{kappa:g}", fit, 0.5))

    palette = [BLUE, RED, BLACK, GREEN, GRAY]
    for idx, ratio in enumerate(ratios):
        selected = sorted(
            [
                r
                for r in rows
                if math.isclose(finite_float(r["h_min_over_kappa"]), ratio)
                and as_bool(r.get("certified", False))
            ],
            key=lambda r: finite_float(r["kappa"]),
            reverse=True,
        )
        if not selected:
            continue
        k = np.array([finite_float(r["kappa"]) for r in selected])
        pref = np.array([finite_float(r["scaled_prefactor"]) for r in selected])
        right_ax.semilogx(
            k,
            pref,
            "o-",
            color=palette[idx % len(palette)],
            linewidth=1.7,
            markersize=5,
            label=rf"$h_{{\min}}/\kappa={ratio:g}$",
        )
    right_ax.invert_xaxis()
    right_ax.grid(False)
    right_ax.set_xlabel(r"$\kappa$")
    right_ax.set_ylabel(r"$\|\cdot\|_\kappa/(h_{\min}^{1/2}|\log\kappa|^{1/2})$")
    right_ax.set_title("Scaled prefactor across conductivities")
    right_ax.legend(fontsize=8, frameon=True)
    fig.savefig(outfile)
    plt.close(fig)
    write_csv(summary_rows, data_dir / "mesh_rate_two_axis_regressions.csv")


def run_mesh_comparison_case(parameters, kappa):
    benchmark = build_benchmarks(parameters)["single_junction_mixed"]
    constant = exact_log_energy_constant(benchmark, parameters)
    ref_hmin = max(parameters.exact_refinement_cap, 0.05 * kappa)
    graded_hmin = max(parameters.exact_refinement_cap, 0.25 * kappa)
    ref_problem = build_local_junction_problem(parameters, benchmark, ref_hmin, mesh_strategy="local_point_graded")
    print(f"[newton-scaling] kappa={kappa:.1e} local reference hmin={ref_hmin:.2e} N={ref_problem.nodes.shape[0]}")
    ref_solution, ref_info = solve_with_continuation(ref_problem, kappa)
    ref_energy = float(ref_info["energy"])

    graded_problem = build_local_junction_problem(parameters, benchmark, graded_hmin, mesh_strategy="local_point_graded")
    graded_n = int(graded_problem.nodes.shape[0])
    cases = [
        ("graded_proposed", graded_problem),
        ("uniform_matched", build_problem(parameters, benchmark, "uniform", target_nodes=graded_n)),
        ("uniform_refined", build_problem(parameters, benchmark, "uniform", target_nodes=4 * graded_n)),
    ]
    sample_points = compact_points(parameters, (0.25, 0.75, 0.25, 0.75))
    rows = []
    for strategy, problem in cases:
        print(f"[newton-scaling] kappa={kappa:.1e} strategy={strategy} N={problem.nodes.shape[0]}")
        solution, info = solve_with_continuation(problem, kappa)
        energy = float(info["energy"])
        normalized = normalized_energy(energy, kappa)
        rows.append(
            {
                "benchmark": benchmark.name,
                "mesh_strategy": strategy,
                "kappa": kappa,
                "num_nodes": int(problem.nodes.shape[0]),
                "num_triangles": int(problem.triangles.shape[0]),
                "boundary_error_to_reference": boundary_error_to_reference_general(problem, solution, ref_problem, ref_solution),
                "boundary_hquarter_to_reference": boundary_hquarter_error_to_reference_general(
                    problem,
                    solution,
                    ref_problem,
                    ref_solution,
                ),
                "bulk_error_to_reference": bulk_error_to_reference_general(problem, solution, ref_problem, ref_solution, sample_points),
                "boundary_l2_to_phi0_sq": boundary_l2_error_sq(solution, problem),
                "energy": energy,
                "energy_minus_reference": energy - ref_energy,
                "normalized_energy": normalized,
                "exact_constant_ratio": exact_constant_ratio(energy, kappa, benchmark, parameters) if constant > 0.0 else float("nan"),
                "h_min": graded_hmin if strategy == "graded_proposed" else float("nan"),
                "reference_h_min": ref_hmin,
                "mesh_family": "local_point",
                "iterations": float(info["iterations"]),
                "residual_inf": float(info["residual_inf"]),
                "initial_residual_inf": float(info.get("initial_residual_inf", float("nan"))),
                "certified_threshold": float(info.get("certified_threshold", float("nan"))),
                "certified": float(info["residual_inf"]) <= float(info.get("certified_threshold", 1.0e-9)),
                "solver": str(info["solver"]),
                "boundary_efficiency": 0.0,
                "bulk_efficiency": 0.0,
                "reference_nodes": int(ref_problem.nodes.shape[0]),
            }
        )
    for row in rows:
        n = math.sqrt(float(row["num_nodes"]))
        row["boundary_efficiency"] = finite_float(row["boundary_error_to_reference"]) * n
        row["bulk_efficiency"] = finite_float(row["bulk_error_to_reference"]) * n
    return rows


def load_or_compute_newton_scaling(parameters, data_dir, run_missing):
    path = data_dir / "mesh_comparison_refined_single_summary.csv"
    rows = read_csv(path) if path.exists() else []
    targets = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
    existing = {
        (round(finite_float(r["kappa"]), 14), str(r["mesh_strategy"]))
        for r in rows
        if str(r.get("mesh_family", "")) == "local_point"
        and math.isfinite(finite_float(r.get("boundary_hquarter_to_reference")))
    }
    missing_kappas = [
        k for k in targets
        if any((round(k, 14), s) not in existing for s in ("graded_proposed", "uniform_matched", "uniform_refined"))
    ]
    if run_missing:
        new_rows = []
        for kappa in missing_kappas:
            new_rows.extend(run_mesh_comparison_case(parameters, kappa))
        if new_rows:
            rows = [
                r for r in rows
                if finite_float(r["kappa"]) not in {finite_float(nr["kappa"]) for nr in new_rows}
            ] + new_rows
            rows.sort(key=lambda r: (finite_float(r["kappa"]), str(r["mesh_strategy"])), reverse=True)
            write_csv(rows, path)
    return rows


def run_newton_uniform_tenfold_case(parameters, kappa):
    benchmark = build_benchmarks(parameters)["single_junction_mixed"]
    graded_hmin = max(parameters.exact_refinement_cap, 0.25 * kappa)
    graded_problem = build_local_junction_problem(parameters, benchmark, graded_hmin, mesh_strategy="local_point_graded")
    target_nodes = 10 * int(graded_problem.nodes.shape[0])
    problem = build_problem(parameters, benchmark, "uniform", target_nodes=target_nodes)
    print(
        "[newton-scaling] "
        f"kappa={kappa:.1e} strategy=uniform_tenfold target={target_nodes} N={problem.nodes.shape[0]}"
    )
    _, info = solve_with_continuation(problem, kappa)
    threshold = float(info.get("certified_threshold", certified_residual_threshold(parameters)))
    return {
        "strategy": "uniform_tenfold",
        "kappa": kappa,
        "num_nodes": int(problem.nodes.shape[0]),
        "iterations": float(info["iterations"]),
        "residual_inf": float(info["residual_inf"]),
        "certified": float(info["residual_inf"]) <= threshold,
        "solver": str(info["solver"]),
        "mesh_family": "local_point",
    }


def load_or_compute_tenfold_newton(parameters, data_dir, run_missing):
    path = data_dir / "newton_uniform_tenfold_summary.csv"
    all_rows = read_csv(path) if path.exists() else []
    rows = [row for row in all_rows if str(row.get("mesh_family", "")) == "local_point"]
    needs_rewrite = len(rows) != len(all_rows)
    targets = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
    existing = {
        round(finite_float(row["kappa"]), 14)
        for row in rows
        if str(row.get("mesh_family", "")) == "local_point"
    }
    missing = [kappa for kappa in targets if round(kappa, 14) not in existing]
    if run_missing and missing:
        rows.extend(run_newton_uniform_tenfold_case(parameters, kappa) for kappa in missing)
        rows.sort(key=lambda r: finite_float(r["kappa"]), reverse=True)
        needs_rewrite = True
    if needs_rewrite:
        write_csv(rows, path)
    return rows


def build_newton_scaling_summary(data_dir, mesh_rows):
    targets = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
    summary = []
    mesh_rate_path = data_dir / "mesh_rate_two_axis_summary.csv"
    if mesh_rate_path.exists():
        for row in read_csv(mesh_rate_path):
            kappa = finite_float(row["kappa"])
            ratio = finite_float(row["h_min_over_kappa"])
            if any(math.isclose(kappa, target) for target in targets) and math.isclose(ratio, 0.10):
                summary.append(
                    {
                        "strategy": "graded_certified",
                        "kappa": kappa,
                        "num_nodes": int(float(row["num_nodes"])),
                        "iterations": finite_float(row["iterations"]),
                        "residual_inf": finite_float(row["residual_inf"]),
                        "certified": as_bool(row.get("certified", False)),
                        "solver": row.get("solver", ""),
                    }
                )
    exact_path = data_dir / "exact_constant_refinement_summary.csv"
    if exact_path.exists():
        for row in read_csv(exact_path):
            kappa = finite_float(row["kappa"])
            if math.isclose(kappa, 1.0e-6):
                summary.append(
                    {
                        "strategy": "graded_certified",
                        "kappa": kappa,
                        "num_nodes": int(float(row["num_nodes"])),
                        "iterations": finite_float(row["iterations"]),
                        "residual_inf": finite_float(row["residual_inf"]),
                        "certified": as_bool(row.get("certified", False)),
                        "solver": row.get("solver", ""),
                    }
                )
    for row in mesh_rows:
        kappa = finite_float(row["kappa"])
        strategy = str(row["mesh_strategy"])
        if not any(math.isclose(kappa, target) for target in targets):
            continue
        if strategy not in {"graded_proposed", "uniform_matched", "uniform_refined"}:
            continue
        summary.append(
            {
                "strategy": "graded_certified" if strategy == "graded_proposed" else strategy,
                "kappa": kappa,
                "num_nodes": int(float(row["num_nodes"])),
                "iterations": finite_float(row["iterations"]),
                "residual_inf": finite_float(row["residual_inf"]),
                "certified": as_bool(row.get("certified", False)),
                "solver": row.get("solver", ""),
            }
        )
    tenfold_path = data_dir / "newton_uniform_tenfold_summary.csv"
    if tenfold_path.exists():
        for row in read_csv(tenfold_path):
            kappa = finite_float(row["kappa"])
            if not any(math.isclose(kappa, target) for target in targets):
                continue
            summary.append(
                {
                    "strategy": "uniform_tenfold",
                    "kappa": kappa,
                    "num_nodes": int(float(row["num_nodes"])),
                    "iterations": finite_float(row["iterations"]),
                    "residual_inf": finite_float(row["residual_inf"]),
                    "certified": as_bool(row.get("certified", False)),
                    "solver": row.get("solver", ""),
                }
            )
    # Prefer one row per strategy/kappa if duplicate sources are present.
    dedup = {}
    for row in summary:
        dedup[(str(row["strategy"]), round(float(row["kappa"]), 14))] = row
    summary = list(dedup.values())
    summary.sort(key=lambda r: (float(r["kappa"]), str(r["strategy"])), reverse=True)
    write_csv(summary, data_dir / "newton_scaling_summary.csv")
    return summary


def plot_newton_scaling(rows, output_dir):
    strategies = [
        ("graded_certified", "Graded", BLUE, "o", True),
        ("uniform_matched", "Uniform $N$", RED, "s", True),
        ("uniform_refined", "Uniform $4N$", BLACK, "D", True),
        ("uniform_tenfold", "Uniform $10N$", GREEN, "^", False),
    ]
    selected = rows
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15))
    fig.subplots_adjust(left=0.08, right=0.992, bottom=0.165, top=0.895, wspace=0.24)
    for strategy, label, color, marker, show_iterations in strategies:
        rs = sorted(
            [row for row in selected if row["strategy"] == strategy],
            key=lambda r: finite_float(r["kappa"]),
            reverse=True,
        )
        if not rs:
            continue
        kappa = np.array([finite_float(r["kappa"]) for r in rs])
        iterations = np.array([finite_float(r["iterations"]) for r in rs])
        residual = np.array([finite_float(r["residual_inf"]) for r in rs])
        if show_iterations:
            axes[0].semilogx(
                kappa,
                iterations,
                marker=marker,
                color=color,
                linewidth=PLOT_LINE_WIDTH,
                markersize=PLOT_MARKER_SIZE,
                label=label,
            )
        axes[1].loglog(
            kappa,
            residual,
            marker=marker,
            color=color,
            linewidth=PLOT_LINE_WIDTH,
            markersize=PLOT_MARKER_SIZE,
            label=label,
        )
    for ax in axes:
        ax.invert_xaxis()
        style_plot_axes(ax)
        ax.set_xlabel(r"$\kappa$", fontsize=PLOT_LABEL_SIZE)
    axes[0].set_ylabel("nonlinear iterations", fontsize=PLOT_LABEL_SIZE)
    set_panel_title(axes[0], "Iteration count")
    axes[1].set_ylabel(r"$\|\nabla I(\lambda_h)\|_{\ell^\infty}$", fontsize=PLOT_LABEL_SIZE)
    set_panel_title(axes[1], "Residual certificate")
    legend0 = axes[0].legend(fontsize=PLOT_LEGEND_SIZE, frameon=True)
    legend1 = axes[1].legend(
        fontsize=PLOT_LEGEND_SIZE,
        frameon=True,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.24),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.6,
        labelspacing=0.35,
    )
    bold_legend_frame(legend0)
    bold_legend_frame(legend1)
    fig.savefig(output_dir / "newton_scaling_mesh_strategies.pdf")
    plt.close(fig)


# Mesh visualization

def add_mesh_collection(
    ax,
    segments,
    xlim=None,
    ylim=None,
    linewidth=0.12,
):
    selected = segments
    if xlim is not None and ylim is not None:
        mid = selected.mean(axis=1)
        pad_x = 0.08 * (xlim[1] - xlim[0])
        pad_y = 0.08 * (ylim[1] - ylim[0])
        selected = selected[
            (xlim[0] - pad_x <= mid[:, 0])
            & (mid[:, 0] <= xlim[1] + pad_x)
            & (ylim[0] - pad_y <= mid[:, 1])
            & (mid[:, 1] <= ylim[1] + pad_y)
        ]
    collection = LineCollection(selected, colors="#4a4a4a", linewidths=linewidth, alpha=0.95, zorder=1)
    collection.set_rasterized(True)
    ax.add_collection(collection)


def draw_reactive_segments(ax, benchmark, linewidth=2.0):
    for start, end, label in benchmark.intervals:
        color = BLUE if label == "c" else RED
        ax.plot(
            [1.0e3 * start, 1.0e3 * end],
            [0.0, 0.0],
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
            clip_on=False,
            zorder=7,
        )


def draw_corrugated_outline(
    ax,
    parameters,
    linewidth=1.0,
):
    top_x = np.linspace(0.0, parameters.lx, 600)
    top_y = corrugated_top_height(top_x, parameters)
    right_height = float(corrugated_top_height(np.array([parameters.lx]), parameters)[0])
    ax.plot(1.0e3 * top_x, 1.0e3 * top_y, color=BLACK, linewidth=linewidth)
    ax.plot(
        [1.0e3 * parameters.lx, 1.0e3 * parameters.lx],
        [0.0, 1.0e3 * right_height],
        color=BLACK,
        linewidth=linewidth,
    )


def add_arrowed_axes(
    ax,
    x_arrow_end,
    y_arrow_end,
):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate(
        "",
        xy=(x_arrow_end, 0.0),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=BLACK, linewidth=0.8, shrinkA=0, shrinkB=0),
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(0.0, y_arrow_end),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=BLACK, linewidth=0.8, shrinkA=0, shrinkB=0),
        clip_on=False,
    )
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False)


def plot_mesh_strategy_triptych(parameters, output_dir):
    stress_parameters = corrugated_stress_physics(parameters)
    benchmark = build_benchmarks(stress_parameters)["corrugated_four_junction_mixed"]
    grading_radius = 1.6e-3
    radial_layers = 7
    angular_layers = 11
    grading_power = 2.0
    graded_points = local_junction_reference_points(
        stress_parameters,
        benchmark_junction_points(benchmark),
        grading_radius,
        radial_layers,
        angular_layers,
        grading_power,
        background_x_count=16,
        background_y_count=6,
    )
    uniform_x = np.linspace(0.0, stress_parameters.lx, 29)
    uniform_y = np.linspace(0.0, stress_parameters.ly, 15)
    if graded_points.shape[0] != uniform_x.size * uniform_y.size:
        raise RuntimeError("Uniform and graded visualization meshes must have the same node count.")
    panels = [
        ("Uniform", mesh_edge_segments_from_grid(uniform_x, uniform_y, stress_parameters, benchmark)),
        ("Graded", triangulated_edge_segments_from_reference_points(graded_points, stress_parameters)),
    ]

    x_arrow_end = 22.0
    y_arrow_end = 13.0

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15))
    fig.subplots_adjust(left=0.08, right=0.992, bottom=0.165, top=0.895, wspace=0.24)
    for ax, (title, segments) in zip(axes, panels):
        add_mesh_collection(ax, segments, linewidth=0.20)
        draw_corrugated_outline(ax, stress_parameters, linewidth=1.0)
        ax.set_xlim(0.0, x_arrow_end)
        ax.set_ylim(0.0, y_arrow_end)
        ax.set_aspect("equal", adjustable="box")
        set_panel_title(ax, title)
        ax.set_xlabel(r"$x$", fontsize=PLOT_LABEL_SIZE)
        ax.set_ylabel(r"$y$", fontsize=PLOT_LABEL_SIZE)
        ax.set_xticks(np.arange(0.0, 20.1, 2.5))
        ax.set_yticks(np.arange(0.0, 12.1, 2.0))
        ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE)
        add_arrowed_axes(ax, x_arrow_end=x_arrow_end, y_arrow_end=y_arrow_end)
        draw_reactive_segments(ax, benchmark, linewidth=5.2)
    axes[1].set_ylabel("")
    fig.savefig(output_dir / "mesh_strategy_triptych.pdf", dpi=300)
    plt.close(fig)
