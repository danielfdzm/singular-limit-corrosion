"""Build the paper-facing comparison tables and diagnostic figures.

This file makes theorem comparisons, practical electrochemical parameter
studies, and the final figures from computed finite-element data.
"""

import argparse
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .mesh_studies import (
    build_newton_scaling_summary,
    load_or_compute_newton_scaling,
    load_or_compute_tenfold_newton,
    plot_mesh_strategy_triptych,
    plot_newton_scaling,
    run_mesh_rate_two_axis,
)
from .numerical_analysis import (
    BLACK,
    BLUE,
    GREEN,
    PLOT_GUIDE_WIDTH,
    PLOT_LABEL_SIZE,
    PLOT_LEGEND_SIZE,
    PLOT_LINE_WIDTH,
    PLOT_MARKER_SIZE,
    PLOT_TICK_SIZE,
    PLOT_TITLE_PAD,
    PLOT_TITLE_SIZE,
    RED,
    add_connected_points,
    add_fit,
    add_slope_guide,
    bootstrap_loglog,
    boundary_hquarter_values_from_rows,
    build_matching_graded_problem,
    compute_interior_window_summary,
    finite_float,
    regression_row,
)
from .model import (
    Parameters,
    build_benchmarks,
    corrugated_stress_physics,
    seawater_dcc_physics,
    stainless_zro2_physics,
)
from .mesh import build_problem, solve_mixed_reference
from .nonlinear_solver import energy_gradient_hessian, solve_problem
from .diagnostics import (
    anchored_scale,
    as_bool,
    bold_legend_frame,
    boundary_l2_error_sq,
    certified_residual_threshold,
    exact_log_energy_constant,
    is_admissible_row,
    is_certified_info,
    max_butler_volmer_exponent_argument,
    normalized_energy,
    set_panel_title,
    style_plot_axes,
)
from .storage import load_saved_solution, read_csv, save_solution, write_csv
from .solution_plots import plot_mesh_comparison_refined

def create_dashboard(parameters, output_dir, data_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_rows = read_csv(data_dir / "convergence_summary.csv")
    single_rows = [
        row
        for row in dashboard_rows
        if row.get("benchmark") == "single_junction_mixed"
        and as_bool(row.get("certified", False))
        and 1.3e-7 <= finite_float(row["kappa"]) <= 1.0e-4
    ]
    interior_rows = compute_interior_window_summary(parameters, data_dir)
    write_csv(interior_rows, data_dir / "interior_rate_window_summary.csv")
    middle_rows = [row for row in interior_rows if row["box"] == "middle"]
    energy_rows = [
        row
        for row in read_csv(data_dir / "exact_constant_refinement_summary.csv")
        if as_bool(row.get("certified", False))
        and is_admissible_row(row, parameters)
        and finite_float(row["kappa"]) < 1.0
        and finite_float(row["exact_constant_ratio"]) < 1.0
    ]

    boundary_x = np.array([finite_float(row["kappa"]) * abs(math.log(finite_float(row["kappa"]))) for row in single_rows])
    boundary_y = np.array([finite_float(row["boundary_error_l2_sq"]) for row in single_rows])
    interior_x = np.array([finite_float(row["kappa_log_kappa"]) for row in middle_rows])
    interior_y = np.array([finite_float(row["bulk_error_l2_sq"]) for row in middle_rows])
    interior_c0_y = np.array([finite_float(row["bulk_error_c0"]) for row in middle_rows])
    interior_c1_y = np.array([finite_float(row["bulk_error_c1"]) for row in middle_rows])
    remainder_x = np.array(
        [
            math.log(abs(math.log(finite_float(row["kappa"])))) / abs(math.log(finite_float(row["kappa"])))
            for row in energy_rows
        ]
    )
    remainder_y = np.array([1.0 - finite_float(row["exact_constant_ratio"]) for row in energy_rows])

    boundary_fit = bootstrap_loglog(boundary_x, boundary_y)
    interior_fit = bootstrap_loglog(interior_x, interior_y)
    interior_c0_fit = bootstrap_loglog(interior_x, interior_c0_y)
    interior_c1_fit = bootstrap_loglog(interior_x, interior_c1_y)
    remainder_fit = bootstrap_loglog(remainder_x, remainder_y)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    fig.subplots_adjust(left=0.08, right=0.992, bottom=0.19, top=0.86, wspace=0.28)

    add_connected_points(axes[0], boundary_x, boundary_y, color=BLUE, label="numerical values")
    add_slope_guide(axes[0], boundary_x, boundary_y, 1.0, "slope 1", color=RED)
    set_panel_title(axes[0], r"$L^2$ boundary convergence")
    axes[0].set_xlabel(r"$\kappa|\log\kappa|$", fontsize=PLOT_LABEL_SIZE)
    axes[0].set_ylabel(r"$\|\phi_{\kappa,h}-\Phi_0\|_{L^2(\Gamma_*)}^2$", fontsize=PLOT_LABEL_SIZE)

    add_connected_points(axes[1], interior_x, interior_c0_y, color=GREEN, label=r"$C^0$ norm")
    add_connected_points(axes[1], interior_x, interior_c1_y, color=BLUE, label=r"$C^1$ norm", marker="^")
    add_slope_guide(axes[1], interior_x, interior_c0_y, 1.0, "slope 1", color=RED)
    add_slope_guide(axes[1], interior_x, interior_c1_y, 1.0, "_nolegend_", color=RED)
    set_panel_title(axes[1], "Interior rate")
    axes[1].set_xlabel(r"$\kappa|\log\kappa|$", fontsize=PLOT_LABEL_SIZE)
    axes[1].set_ylabel("interior error diagnostics", fontsize=PLOT_LABEL_SIZE)

    for ax in axes:
        ax.invert_xaxis()
        style_plot_axes(ax)
    for idx, ax in enumerate(axes):
        legend_fontsize = PLOT_LEGEND_SIZE
        legend_kwargs = {"fontsize": legend_fontsize, "frameon": True}
        if idx == 1:
            legend_kwargs.update(
                {
                    "loc": "upper right",
                    "ncol": 1,
                    "columnspacing": 0.85,
                    "handlelength": 1.55,
                    "labelspacing": 0.35,
                }
            )
        else:
            legend_kwargs["loc"] = "best"
        legend = ax.legend(**legend_kwargs)
        bold_legend_frame(legend)
    fig.savefig(output_dir / "theorem_dashboard.pdf")
    plt.close(fig)

    plot_interior_windows(interior_rows, output_dir / "interior_rate_windows.pdf")
    regression_rows = [
        regression_row("boundary_L2_sq_vs_kappa_log_kappa", boundary_fit, 1.0),
        regression_row("interior_EK_sq_vs_kappa_log_kappa", interior_fit, 1.0),
        regression_row("interior_C0_vs_kappa_log_kappa", interior_c0_fit, 0.5),
        regression_row("interior_C1_vs_kappa_log_kappa", interior_c1_fit, 0.5),
        regression_row("energy_remainder_vs_loglog_over_log", remainder_fit, 1.0),
    ]
    write_csv(regression_rows, data_dir / "theorem_dashboard_regressions.csv")
    return {
        "boundary": boundary_fit,
        "interior": interior_fit,
        "interior_c0": interior_c0_fit,
        "interior_c1": interior_c1_fit,
        "remainder": remainder_fit,
        "regression_rows": regression_rows,
    }


def plot_interior_windows(rows, outfile):
    styles = {
        "near-boundary": (BLUE, "o", r"$[.10,.90]\times[.10,.90]$"),
        "middle": (RED, "s", r"$[.25,.75]\times[.25,.75]$"),
        "core": (BLACK, "D", r"$[.40,.60]\times[.40,.60]$"),
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.05), constrained_layout=True)
    for label, (color, marker, tex_label) in styles.items():
        selected = [row for row in rows if row["box"] == label]
        if not selected:
            continue
        x = np.array([finite_float(row["kappa_log_kappa"]) for row in selected])
        y = np.array([finite_float(row["bulk_error_l2_sq"]) for row in selected])
        fit = bootstrap_loglog(x, y)
        add_fit(ax, fit, color=color, data_label=tex_label, fit_label=f"{label} regression", marker=marker)
    ax.grid(False)
    ax.set_xlabel(r"$\kappa|\log\kappa|$")
    ax.set_ylabel(r"$E_K(\kappa)^2$")
    ax.set_title("Interior-rate regression across compact subsets")
    ax.legend(fontsize=8, frameon=True)
    fig.savefig(outfile)
    plt.close(fig)


def plot_geometric_stability(parameters, output_dir, data_dir):
    datasets = [
        (
            "multi_exact_constant_refinement_summary.csv",
            "Multi rectangle, $N=3$",
            BLUE,
            "o",
        ),
        (
            "corrugated_stress_summary.csv",
            "Corrugated, $N=4$",
            RED,
            "s",
        ),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.05), constrained_layout=True)
    summary_rows = []
    for filename, label, color, marker in datasets:
        rows = [
            row
            for row in read_csv(data_dir / filename)
            if as_bool(row.get("certified", False)) and 1.0e-7 <= finite_float(row["kappa"]) <= 1.0e-4
            and is_admissible_row(row, parameters)
        ]
        rows.sort(key=lambda r: finite_float(r["kappa"]), reverse=True)
        kappa = np.array([finite_float(row["kappa"]) for row in rows])
        ratio = np.array([finite_float(row["exact_constant_ratio"]) for row in rows])
        ax.semilogx(kappa, ratio, marker=marker, color=color, linewidth=1.9, markersize=5.4, label=label)
        finite = ratio[np.isfinite(ratio)]
        if finite.size:
            summary_rows.append(
                {
                    "benchmark": label,
                    "num_points": int(finite.size),
                    "smallest_kappa": float(np.min(kappa[np.isfinite(ratio)])),
                    "ratio_at_smallest_kappa": float(ratio[np.nanargmin(kappa)]),
                    "min_ratio": float(np.min(finite)),
                    "max_ratio": float(np.max(finite)),
                }
            )
    ax.axhline(1.0, color=RED, linestyle="--", linewidth=PLOT_GUIDE_WIDTH, label="exact constant")
    ax.invert_xaxis()
    style_plot_axes(ax)
    ax.set_xlabel(r"$\kappa$", fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel(r"$R_E(\kappa)=J_{\kappa,h}/(C_N|\log\kappa|)$", fontsize=PLOT_LABEL_SIZE)
    set_panel_title(ax, "Geometric stability of the logarithmic constant")
    legend = ax.legend(fontsize=PLOT_LEGEND_SIZE, frameon=True)
    bold_legend_frame(legend)
    fig.savefig(output_dir / "geometry_stability_comparison.pdf")
    plt.close(fig)
    write_csv(summary_rows, data_dir / "geometry_stability_summary.csv")


def plot_corrugated_csv_diagnostics(
    parameters,
    data_dir,
    output_dir,
):
    datasets = [
        (
            "corrugated_stress_summary.csv",
            "corrugated_four_junction_diagnostics.pdf",
            "Corrugated four-junction mixed benchmark",
        ),
        (
            "corrugated_stress_corrugated_six_junction_summary.csv",
            "corrugated_six_junction_diagnostics.pdf",
            "Corrugated six-junction mixed benchmark",
        ),
    ]
    summary_rows = []
    for csv_name, plot_name, title in datasets:
        path = data_dir / csv_name
        if not path.exists():
            continue
        all_rows = [
            row
            for row in read_csv(path)
            if finite_float(row.get("kappa")) < 1.0
            and as_bool(row.get("certified", False))
            and is_admissible_row(row, parameters)
            and math.isfinite(finite_float(row.get("exact_constant_ratio")))
        ]
        all_rows.sort(key=lambda row: finite_float(row["kappa"]), reverse=True)
        excluded = [
            row
            for row in read_csv(path)
            if finite_float(row.get("kappa")) < 1.0
            and as_bool(row.get("certified", False))
            and not is_admissible_row(row, parameters)
        ]
        if not all_rows:
            continue

        kappa = np.array([finite_float(row["kappa"]) for row in all_rows])
        boundary = np.array([finite_float(row["boundary_error_l2_sq"]) for row in all_rows])
        ratio = np.array([finite_float(row["exact_constant_ratio"]) for row in all_rows])
        ref = kappa * np.abs(np.log(kappa))
        scale = anchored_scale(kappa, boundary, ref)

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
        axes[0].loglog(
            kappa,
            boundary,
            "o-",
            color=BLUE,
            linewidth=PLOT_LINE_WIDTH,
            markersize=PLOT_MARKER_SIZE,
            label="data",
        )
        axes[0].loglog(
            kappa,
            scale * ref,
            "--",
            color=RED,
            linewidth=PLOT_GUIDE_WIDTH,
            label=r"$C\kappa|\log\kappa|$",
        )
        axes[0].set_xlim(float(np.max(kappa)), float(np.min(kappa)))
        axes[0].set_xlabel(r"$\kappa$", fontsize=PLOT_LABEL_SIZE)
        axes[0].set_ylabel(r"$\|\phi_\kappa-\Phi_0\|_{L^2(\Gamma_D)}^2$", fontsize=PLOT_LABEL_SIZE)
        set_panel_title(axes[0], r"$L^2$ boundary convergence")
        style_plot_axes(axes[0])
        legend0 = axes[0].legend(fontsize=PLOT_LEGEND_SIZE, frameon=True)
        bold_legend_frame(legend0)

        axes[1].semilogx(
            kappa,
            ratio,
            "o-",
            color=BLUE,
            linewidth=PLOT_LINE_WIDTH,
            markersize=PLOT_MARKER_SIZE,
            label="data",
        )
        axes[1].axhline(1.0, color=RED, linestyle="--", linewidth=PLOT_GUIDE_WIDTH, label="asymptotic target")
        axes[1].set_xlim(float(np.max(kappa)), float(np.min(kappa)))
        axes[1].set_xlabel(r"$\kappa$", fontsize=PLOT_LABEL_SIZE)
        axes[1].set_ylabel(r"$J_\kappa/(C_N|\log\kappa|)$", fontsize=PLOT_LABEL_SIZE)
        set_panel_title(axes[1], "Exact-constant ratio")
        style_plot_axes(axes[1])
        legend1 = axes[1].legend(
            fontsize=PLOT_LEGEND_SIZE,
            frameon=True,
            loc="lower right",
            bbox_to_anchor=(0.99, 0.02),
        )
        bold_legend_frame(legend1)

        fig.savefig(output_dir / plot_name)
        plt.close(fig)

        summary_rows.append(
            {
                "dataset": csv_name,
                "num_admissible_certified": len(all_rows),
                "num_excluded_nonadmissible_certified": len(excluded),
                "smallest_admissible_kappa": float(np.min(kappa)),
                "ratio_at_smallest_admissible_kappa": float(ratio[np.argmin(kappa)]),
            }
        )
    write_csv(summary_rows, data_dir / "corrugated_admissible_diagnostics_summary.csv")

PRACTICAL_CORRUGATED_KAPPAS = (
    1.0,
    1.0e-1,
    1.0e-2,
    1.0e-3,
    1.0e-4,
    3.0e-5,
    1.0e-5,
    3.0e-6,
    1.0e-6,
    5.0e-7,
    3.0e-7,
)


def practical_source_specs(parameters):
    return [
        {
            "key": "seawater_dcc",
            "label": "Seawater DCC",
            "parameters": seawater_dcc_physics(parameters),
            "outfile": "practical_seawater_dcc_corrugated_diagnostics.pdf",
        },
        {
            "key": "stainless_zro2",
            "label": "ZrO2-coated stainless steel",
            "parameters": stainless_zro2_physics(parameters),
            "outfile": "practical_stainless_zro2_corrugated_diagnostics.pdf",
        },
    ]


def practical_variant_benchmark(source_key, benchmark):
    return replace(
        benchmark,
        name=f"practical_{source_key}_{benchmark.name}",
        file_tag=f"practical_{source_key}_{benchmark.file_tag}",
        title=f"{benchmark.title} ({source_key})",
    )


def practical_summary_path(data_dir, source_key, benchmark):
    return data_dir / f"practical_{source_key}_{benchmark.file_tag}_summary.csv"


def run_practical_corrugated_case(
    source_key,
    source_label,
    parameters,
    benchmark,
    data_dir,
):
    kappas = tuple(sorted(set(PRACTICAL_CORRUGATED_KAPPAS), reverse=True))
    target_min = min(kappa for kappa in kappas if kappa < 1.0)
    hmin = max(parameters.exact_refinement_cap, min(2.5e-7, 0.25 * target_min))
    variant = practical_variant_benchmark(source_key, benchmark)
    problem = build_problem(parameters, variant, "graded", hmin=hmin)
    reference = solve_mixed_reference(problem)
    previous = reference.copy()
    rows = []

    for kappa in kappas:
        saved_solution = load_saved_solution(problem, kappa, data_dir)
        if saved_solution is not None:
            print(f"[practical:{source_key}:{benchmark.file_tag}] kappa={kappa:.1e} (saved)")
            solution = saved_solution
            energy, grad, _ = energy_gradient_hessian(solution, problem, kappa, with_hessian=False)
            info = {
                "iterations": 0.0,
                "residual_inf": float(np.linalg.norm(grad, ord=np.inf)),
                "initial_residual_inf": float("nan"),
                "certified_threshold": certified_residual_threshold(parameters),
                "accepted_step": 0.0,
                "energy": float(energy),
                "solver": "cached",
            }
            if not is_certified_info(info, parameters):
                print(f"[practical:{source_key}:{benchmark.file_tag}] kappa={kappa:.1e} polishing saved solution")
                solution, info = solve_problem(problem, kappa, solution)
                save_solution(problem, kappa, solution, data_dir)
        else:
            print(f"[practical:{source_key}:{benchmark.file_tag}] kappa={kappa:.1e}")
            solution, info = solve_problem(problem, kappa, previous)
            save_solution(problem, kappa, solution, data_dir)

        previous = solution
        energy = float(info["energy"])
        normalized = normalized_energy(energy, kappa)
        constant = exact_log_energy_constant(variant, parameters)
        boundary_error_sq = boundary_l2_error_sq(solution, problem)
        max_exp_arg = max_butler_volmer_exponent_argument(solution, parameters)
        row = {
            "source_key": source_key,
            "source_label": source_label,
            "benchmark": variant.name,
            "base_benchmark": benchmark.name,
            "junction_count": int(round(constant * 2.0 * math.pi / ((parameters.phi_c - parameters.phi_a) ** 2))),
            "geometry": benchmark.geometry,
            "mesh_strategy": "graded",
            "kappa": kappa,
            "h_min": hmin,
            "num_nodes": int(problem.nodes.shape[0]),
            "num_triangles": int(problem.triangles.shape[0]),
            "phi_a": parameters.phi_a,
            "phi_c": parameters.phi_c,
            "ia0": parameters.ia0,
            "ic0": parameters.ic0,
            "a1": parameters.a1,
            "a2": parameters.a2,
            "c1": parameters.c1,
            "c2": parameters.c2,
            "energy": energy,
            "normalized_energy": normalized,
            "exact_energy_constant": constant,
            "exact_constant_ratio": normalized / constant if constant > 0.0 and kappa < 1.0 else float("nan"),
            "boundary_error_l2_sq": boundary_error_sq,
            "boundary_rate_ratio": boundary_error_sq / (kappa * abs(math.log(kappa))) if kappa < 1.0 else float("nan"),
            "iterations": float(info["iterations"]),
            "residual_inf": float(info["residual_inf"]),
            "initial_residual_inf": float(info.get("initial_residual_inf", float("nan"))),
            "certified_threshold": float(info.get("certified_threshold", certified_residual_threshold(parameters))),
            "accepted_step": float(info["accepted_step"]),
            "solver": str(info["solver"]),
            "certified": is_certified_info(info, parameters),
            "solution_min": float(np.min(solution)),
            "solution_max": float(np.max(solution)),
            "max_bv_exponent_argument": max_exp_arg,
            "exponential_clamp_active": max_exp_arg >= 80.0,
        }
        rows.append(row)

    rows.sort(key=lambda row: finite_float(row["kappa"]), reverse=True)
    write_csv(rows, practical_summary_path(data_dir, source_key, benchmark))
    return rows


def load_or_run_practical_corrugated_cases(
    parameters,
    data_dir,
    run_missing,
):
    out = {}
    for spec in practical_source_specs(parameters):
        source_key = str(spec["key"])
        source_label = str(spec["label"])
        source_parameters = spec["parameters"]
        source_benchmarks = build_benchmarks(source_parameters)
        out[source_key] = {}
        for base_name in ("corrugated_four_junction_mixed", "corrugated_six_junction_mixed"):
            benchmark = source_benchmarks[base_name]
            path = practical_summary_path(data_dir, source_key, benchmark)
            if path.exists():
                rows = read_csv(path)
            elif run_missing:
                rows = run_practical_corrugated_case(source_key, source_label, source_parameters, benchmark, data_dir)
            else:
                rows = []
            out[source_key][base_name] = rows
    return out


def plot_practical_parameter_diagnostics(
    parameters,
    practical_rows,
    output_dir,
    data_dir,
):
    three_panel_scale = 14.2 / 10.4
    title_size = PLOT_TITLE_SIZE * three_panel_scale
    title_pad = PLOT_TITLE_PAD * three_panel_scale
    label_size = PLOT_LABEL_SIZE * three_panel_scale
    tick_size = PLOT_TICK_SIZE * three_panel_scale
    legend_size = PLOT_LEGEND_SIZE * three_panel_scale
    marker_size = PLOT_MARKER_SIZE * three_panel_scale
    line_width = PLOT_LINE_WIDTH * three_panel_scale
    guide_width = PLOT_GUIDE_WIDTH * three_panel_scale

    specs = {str(spec["key"]): spec for spec in practical_source_specs(parameters)}
    colors = {
        "corrugated_four_junction_mixed": BLUE,
        "corrugated_six_junction_mixed": GREEN,
    }
    markers = {
        "corrugated_four_junction_mixed": "o",
        "corrugated_six_junction_mixed": "s",
    }
    labels = {
        "corrugated_four_junction_mixed": "$N=4$",
        "corrugated_six_junction_mixed": "$N=6$",
    }

    for source_key, by_benchmark in practical_rows.items():
        if not any(by_benchmark.values()):
            continue
        spec = specs[source_key]
        source_parameters = spec["parameters"]
        source_benchmarks = build_benchmarks(source_parameters)
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.15), constrained_layout=True)
        guide_label_used = False
        hquarter_guide_label_used = False

        for base_name, rows in by_benchmark.items():
            filtered = [
                row
                for row in rows
                if finite_float(row.get("kappa")) < 1.0
                and as_bool(row.get("certified", False))
                and is_admissible_row(row, source_parameters)
            ]
            filtered.sort(key=lambda row: finite_float(row["kappa"]), reverse=True)
            if not filtered:
                continue
            kappa = np.array([finite_float(row["kappa"]) for row in filtered], dtype=float)
            boundary = np.array([finite_float(row["boundary_error_l2_sq"]) for row in filtered], dtype=float)
            ratio = np.array([finite_float(row["exact_constant_ratio"]) for row in filtered], dtype=float)
            ref = kappa * np.abs(np.log(kappa))
            scale = anchored_scale(kappa, boundary, ref)
            benchmark = practical_variant_benchmark(source_key, source_benchmarks[base_name])
            problem = build_matching_graded_problem(source_parameters, benchmark, filtered)
            hquarter = boundary_hquarter_values_from_rows(problem, filtered, data_dir)
            hquarter_ref = np.power(kappa, 0.25) * np.sqrt(np.abs(np.log(kappa)))
            hquarter_scale = anchored_scale(kappa, hquarter, hquarter_ref)

            axes[0].loglog(
                kappa,
                boundary,
                marker=markers[base_name],
                color=colors[base_name],
                linewidth=line_width,
                markersize=marker_size,
                label=labels[base_name],
            )
            axes[0].loglog(
                kappa,
                scale * ref,
                "--",
                color=RED,
                linewidth=guide_width,
                label=r"$C\kappa|\log\kappa|$" if not guide_label_used else "_nolegend_",
            )
            guide_label_used = True

            axes[1].semilogx(
                kappa,
                hquarter,
                marker=markers[base_name],
                color=colors[base_name],
                linewidth=line_width,
                markersize=marker_size,
                label=labels[base_name],
            )
            axes[1].loglog(
                kappa,
                hquarter_scale * hquarter_ref,
                "--",
                color=RED,
                linewidth=guide_width,
                label=(
                    r"$C\kappa^{1/4}|\log\kappa|^{1/2}$"
                    if not hquarter_guide_label_used
                    else "_nolegend_"
                ),
            )
            hquarter_guide_label_used = True

            axes[2].semilogx(
                kappa,
                ratio,
                marker=markers[base_name],
                color=colors[base_name],
                linewidth=line_width,
                markersize=marker_size,
                label=labels[base_name],
            )

        axes[2].axhline(1.0, color=RED, linestyle="--", linewidth=guide_width, label="asymptotic target")
        for idx, ax in enumerate(axes):
            ax.invert_xaxis()
            ax.grid(False)
            ax.tick_params(axis="both", which="major", labelsize=tick_size)
            ax.set_xlabel(r"$\kappa$", fontsize=label_size)
            legend_kwargs = {"fontsize": legend_size, "frameon": True}
            if idx in (0, 1):
                handles, legend_labels = ax.get_legend_handles_labels()
                order = []
                guide_label = (
                    r"$C\kappa|\log\kappa|$"
                    if idx == 0
                    else r"$C\kappa^{1/4}|\log\kappa|^{1/2}$"
                )
                for wanted in ("$N=4$", "$N=6$", guide_label):
                    order.extend(i for i, label in enumerate(legend_labels) if label == wanted)
                if order:
                    legend_kwargs["handles"] = [handles[i] for i in order]
                    legend_kwargs["labels"] = [legend_labels[i] for i in order]
            if idx == 2:
                legend_kwargs.update({"loc": "lower right"})
            legend = ax.legend(**legend_kwargs)
            bold_legend_frame(legend)
        axes[0].set_ylabel(r"$\|\phi_\kappa-\Phi_0\|_{L^2(\Gamma_D)}^2$", fontsize=label_size)
        axes[0].set_title(r"$L^2$ boundary convergence", fontsize=title_size, pad=title_pad)
        axes[1].set_ylabel(r"$\|\phi_\kappa-\Phi_0\|_{H^{1/4}(\Gamma_D)}$", fontsize=label_size)
        axes[1].set_title(r"$H^{1/4}$ boundary convergence", fontsize=title_size, pad=title_pad)
        axes[2].set_ylabel(r"$J_\kappa/(C_N|\log\kappa|)$", fontsize=label_size)
        axes[2].set_title("Exact-constant ratio", fontsize=title_size, pad=title_pad)
        fig.savefig(output_dir / str(spec["outfile"]))
        plt.close(fig)


def run_practical_parameter_experiments(
    parameters,
    output_dir,
    data_dir,
    run_missing,
):
    rows = load_or_run_practical_corrugated_cases(parameters, data_dir, run_missing)
    plot_practical_parameter_diagnostics(parameters, rows, output_dir, data_dir)

def strip_width_columns(data_dir):
    path = data_dir / "convergence_summary.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    stripped = []
    for row in rows:
        row = dict(row)
        row.pop("transition_width", None)
        row.pop("width_over_kappa", None)
        stripped.append(row)
    write_csv(stripped, path)


def require_postprocess_inputs(data_dir):
    required = (
        "convergence_summary.csv",
        "u0_single_junction_mixed.npz",
        "exact_constant_refinement_summary.csv",
        "multi_exact_constant_refinement_summary.csv",
        "corrugated_stress_summary.csv",
        "corrugated_stress_corrugated_six_junction_summary.csv",
    )
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        formatted = "\n".join(f"  - {data_dir / name}" for name in missing)
        raise FileNotFoundError(
            "Postprocessing requires precomputed numerical summaries and saved solutions. "
            "Generate the upstream experiment data first.\n"
            f"Missing:\n{formatted}"
        )


def run_postprocess(args):
    parameters = Parameters()
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    require_postprocess_inputs(data_dir)
    strip_width_columns(data_dir)
    create_dashboard(parameters, output_dir, data_dir)
    plot_geometric_stability(parameters, output_dir, data_dir)
    plot_corrugated_csv_diagnostics(corrugated_stress_physics(parameters), data_dir, output_dir)
    run_practical_parameter_experiments(parameters, output_dir, data_dir, args.practical_parameter_experiments)
    plot_mesh_strategy_triptych(parameters, output_dir)
    rows = load_or_compute_newton_scaling(parameters, data_dir, args.run_missing_newton)
    if args.mesh_rate_two_axis:
        run_mesh_rate_two_axis(parameters, output_dir, data_dir)
    plot_mesh_comparison_refined(rows, output_dir / "mesh_comparison_refined_single.pdf")
    load_or_compute_tenfold_newton(parameters, data_dir, args.run_missing_newton)
    newton_rows = build_newton_scaling_summary(data_dir, rows)
    plot_newton_scaling(newton_rows, output_dir)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="paper_figures/generated")
    parser.add_argument("--data-dir", default="data/computed")
    parser.add_argument("--mesh-rate-two-axis", action="store_true")
    parser.add_argument("--run-missing-newton", action="store_true")
    parser.add_argument("--practical-parameter-experiments", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run_postprocess(parse_args(argv))




if __name__ == "__main__":
    main()
