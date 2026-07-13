"""Figures produced from finite-element solutions and convergence diagnostics.

This module contains the paper's field, boundary, mesh-comparison,
three-dimensional, and geometry-overview plots.
"""

import math

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .diagnostics import (
    DIAGNOSTIC_BLUE,
    DIAGNOSTIC_GREEN,
    DIAGNOSTIC_KAPPA_MAX,
    DIAGNOSTIC_KAPPA_MIN,
    GUIDE_RED,
    PLOT_GUIDE_WIDTH,
    PLOT_LABEL_SIZE,
    PLOT_LEGEND_SIZE,
    PLOT_LINE_WIDTH,
    PLOT_MARKER_SIZE,
    PLOT_TICK_SIZE,
    PLOT_TITLE_PAD,
    PLOT_TITLE_SIZE,
    POTENTIAL_CMAP,
    anchored_scale,
    as_bool,
    bold_legend_frame,
    exact_log_energy_constant,
    is_admissible_row,
    set_panel_title,
    style_plot_axes,
    theorem_diagnostic_rows,
    triangle_gradient_magnitude,
)
from .model import (
    corrugated_top_height,
    reactive_value,
)


def domain_outline(problem):
    x_bottom = problem.x
    x_top = x_bottom[::-1]
    y_bottom = np.zeros_like(x_bottom)
    if problem.benchmark.geometry == "rectangle":
        y_top = np.full_like(x_top, problem.physics.ly)
    elif problem.benchmark.geometry == "corrugated_top":
        y_top = corrugated_top_height(x_top, problem.physics)
    else:
        raise ValueError(f"Unknown geometry: {problem.benchmark.geometry}")
    outline_x = np.concatenate([x_bottom, x_top, [x_bottom[0]]])
    outline_y = np.concatenate([y_bottom, y_top, [0.0]])
    return outline_x, outline_y


def style_solution_axes(ax, problem):
    outline_x, outline_y = domain_outline(problem)
    xmin = float(np.min(outline_x))
    xmax = float(np.max(outline_x))
    ymin = float(np.min(outline_y))
    ymax = float(np.max(outline_y))
    xpad = 0.02 * max(xmax - xmin, np.finfo(float).eps)
    ypad = 0.04 * max(ymax - ymin, np.finfo(float).eps)

    ax.plot(
        outline_x,
        outline_y,
        color="black",
        linewidth=1.4,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )
    ax.set_aspect("equal")
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_snapshot(problem, u, outfile):
    triangulation = mtri.Triangulation(problem.nodes[:, 0], problem.nodes[:, 1], problem.triangles)
    parameters = problem.physics
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    fig.subplots_adjust(left=0.12, right=0.84, bottom=0.18, top=0.95)
    tpc = ax.tripcolor(
        triangulation,
        u,
        shading="gouraud",
        cmap=POTENTIAL_CMAP,
        vmin=parameters.phi_a,
        vmax=parameters.phi_c,
    )
    style_solution_axes(ax, problem)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    cbar = fig.colorbar(tpc, cax=cax, ticks=np.linspace(parameters.phi_a, parameters.phi_c, 5))
    cbar.set_label("potential")
    fig.savefig(outfile)
    plt.close(fig)


def plot_corrugated_snapshots(problem, solutions, outfile):
    kappas = sorted(solutions.keys(), reverse=True)
    triangulation = mtri.Triangulation(problem.nodes[:, 0], problem.nodes[:, 1], problem.triangles)
    parameters = problem.physics
    fig, axes = plt.subplots(1, len(kappas), figsize=(5.2 * len(kappas) + 1.4, 3.9))
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.08, top=0.88, wspace=0.12)
    if len(kappas) == 1:
        axes = np.array([axes])
    tpc = None
    for ax, kappa in zip(axes, kappas):
        tpc = ax.tripcolor(
            triangulation,
            solutions[kappa],
            shading="gouraud",
            cmap=POTENTIAL_CMAP,
            vmin=parameters.phi_a,
            vmax=parameters.phi_c,
        )
        style_solution_axes(ax, problem)
        ax.set_title(rf"$\kappa={kappa:g}$", fontsize=PLOT_TITLE_SIZE, pad=PLOT_TITLE_PAD)
    if tpc is None:
        raise RuntimeError("No solution was supplied for the snapshot plot.")
    cax = fig.add_axes([0.925, 0.16, 0.014, 0.68])
    cbar = fig.colorbar(tpc, cax=cax, ticks=np.linspace(parameters.phi_a, parameters.phi_c, 5))
    cbar.set_label("potential", fontsize=PLOT_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=PLOT_TICK_SIZE)
    fig.savefig(outfile)
    plt.close(fig)


def plot_step_profile(ax, benchmark, parameters):
    breaks = [benchmark.intervals[0][0]]
    values = []
    for _, end, label in benchmark.intervals:
        breaks.append(end)
        values.append(reactive_value(label, parameters))
    ax.step(
        breaks,
        values + [values[-1]],
        where="post",
        color="black",
        linewidth=1.8,
        linestyle="--",
        label=r"$\Phi_0$",
    )


def plot_boundary_traces(problem, traces, outfile):
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    for kappa, trace in sorted(traces.items(), reverse=True):
        ax.plot(problem.x, trace, linewidth=2.0, label=rf"$\kappa={kappa:g}$")
    plot_step_profile(ax, problem.benchmark, problem.physics)
    ax.set_xlabel("x on the reactive boundary [m]")
    ax.set_ylabel("potential")
    ax.set_title("Reactive boundary traces")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(outfile)
    plt.close(fig)


def plot_corrugated_diagnostics(
    rows,
    parameters,
    outfile,
):
    filtered = [
        row
        for row in rows
        if float(row["kappa"]) < 1.0 and as_bool(row.get("certified", False))
        and is_admissible_row(row, parameters)
    ]
    kappa_arr = np.array([float(row["kappa"]) for row in filtered], dtype=float)
    boundary_arr = np.array([float(row["boundary_error_l2_sq"]) for row in filtered], dtype=float)
    ref_arr = kappa_arr * np.abs(np.log(kappa_arr))
    ratio_arr = np.array([float(row["exact_constant_ratio"]) for row in filtered], dtype=float)
    scale = anchored_scale(kappa_arr, boundary_arr, ref_arr)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].loglog(
        kappa_arr,
        boundary_arr,
        "o-",
        color=DIAGNOSTIC_BLUE,
        linewidth=PLOT_LINE_WIDTH,
        markersize=PLOT_MARKER_SIZE,
        label="data",
    )
    if kappa_arr.size:
        axes[0].loglog(
            kappa_arr,
            scale * ref_arr,
            "--",
            color=GUIDE_RED,
            linewidth=PLOT_GUIDE_WIDTH,
            label=r"$C\kappa|\log\kappa|$",
        )
        axes[0].set_xlim(float(np.max(kappa_arr)), float(np.min(kappa_arr)))
    axes[0].set_xlabel(r"$\kappa$", fontsize=PLOT_LABEL_SIZE)
    axes[0].set_ylabel(r"$\|\phi_\kappa-\Phi_0\|_{L^2(\Gamma_D)}^2$", fontsize=PLOT_LABEL_SIZE)
    set_panel_title(axes[0], "Boundary convergence")
    style_plot_axes(axes[0])
    legend0 = axes[0].legend(fontsize=PLOT_LEGEND_SIZE, frameon=True)
    bold_legend_frame(legend0)

    axes[1].semilogx(
        kappa_arr,
        ratio_arr,
        "o-",
        color=DIAGNOSTIC_BLUE,
        linewidth=PLOT_LINE_WIDTH,
        markersize=PLOT_MARKER_SIZE,
        label="data",
    )
    if kappa_arr.size:
        axes[1].axhline(1.0, color=GUIDE_RED, linestyle="--", linewidth=PLOT_GUIDE_WIDTH, label="predicted limit")
        axes[1].set_xlim(float(np.max(kappa_arr)), float(np.min(kappa_arr)))
    axes[1].set_xlabel(r"$\kappa$", fontsize=PLOT_LABEL_SIZE)
    axes[1].set_ylabel(r"$J_\kappa/(A_\Sigma|\log\kappa|)$", fontsize=PLOT_LABEL_SIZE)
    set_panel_title(axes[1], "Exact-constant ratio")
    style_plot_axes(axes[1])
    legend1 = axes[1].legend(fontsize=PLOT_LEGEND_SIZE, frameon=True)
    bold_legend_frame(legend1)
    fig.savefig(outfile)
    plt.close(fig)


def plot_boundary_convergence(
    rows,
    benchmark,
    parameters,
    outfile,
):
    filtered = theorem_diagnostic_rows(rows, parameters)
    kappa_arr = np.array([float(row["kappa"]) for row in filtered], dtype=float)
    error_arr = np.array([float(row["boundary_error_l2_sq"]) for row in filtered], dtype=float)
    ref_arr = kappa_arr * np.abs(np.log(kappa_arr))
    scale = anchored_scale(kappa_arr, error_arr, ref_arr)

    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    ax.loglog(
        kappa_arr,
        error_arr,
        "o-",
        color=DIAGNOSTIC_BLUE,
        linewidth=2.0,
        markersize=6.0,
        label="boundary error",
    )
    if kappa_arr.size:
        ax.loglog(
            kappa_arr,
            scale * ref_arr,
            "--",
            color=GUIDE_RED,
            linewidth=1.8,
            label=rf"guide $C\,\kappa|\log\kappa|$",
        )
        ax.set_xlim(DIAGNOSTIC_KAPPA_MAX, DIAGNOSTIC_KAPPA_MIN)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\|\phi_\kappa-\Phi_0\|_{L^2(\Gamma_D)}^2$")
    ax.set_title(f"Boundary convergence: {benchmark.title}")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.savefig(outfile)
    plt.close(fig)
    if not kappa_arr.size:
        return float("nan")
    ratio = error_arr / ref_arr
    return float(np.max(ratio))


def plot_bulk_error(
    rows,
    benchmark,
    parameters,
    outfile,
):
    filtered = theorem_diagnostic_rows(rows, parameters)
    kappa_arr = np.array([float(row["kappa"]) for row in filtered], dtype=float)
    error_arr = np.array([float(row["bulk_error_l2_K"]) for row in filtered], dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    ax.loglog(kappa_arr, error_arr, "o-", color=DIAGNOSTIC_BLUE, linewidth=2.0, markersize=6.0)
    if kappa_arr.size:
        ax.set_xlim(DIAGNOSTIC_KAPPA_MAX, DIAGNOSTIC_KAPPA_MIN)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\|\phi_\kappa-u_0^{\mathrm{mix}}\|_{L^2(K)}$")
    ax.set_title(f"Interior error on $K$: {benchmark.title}")
    ax.grid(alpha=0.25, which="both")
    fig.savefig(outfile)
    plt.close(fig)


def plot_energy_scaling(
    rows,
    benchmark,
    parameters,
    outfile,
):
    filtered = theorem_diagnostic_rows(rows, parameters)
    kappa_arr = np.array([float(row["kappa"]) for row in filtered], dtype=float)
    energy_arr = np.array([float(row["energy"]) for row in filtered], dtype=float)
    norm_arr = np.array([float(row["normalized_energy"]) for row in filtered], dtype=float)
    log_arr = np.abs(np.log(kappa_arr))
    scale = anchored_scale(kappa_arr, energy_arr, log_arr)
    exact_constant = exact_log_energy_constant(benchmark, parameters)
    exact_ratio_arr = norm_arr / exact_constant if exact_constant > 0.0 else np.full_like(norm_arr, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), constrained_layout=True)

    axes[0].plot(
        log_arr,
        energy_arr,
        "o-",
        color=DIAGNOSTIC_BLUE,
        linewidth=2.0,
        markersize=6.0,
        label="computed energy",
    )
    if log_arr.size:
        axes[0].plot(
            log_arr,
            scale * log_arr,
            "--",
            color=GUIDE_RED,
            linewidth=1.8,
            label=rf"guide $C|\log\kappa|$",
        )
        axes[0].set_xlim(abs(math.log(DIAGNOSTIC_KAPPA_MAX)), abs(math.log(DIAGNOSTIC_KAPPA_MIN)))
    axes[0].set_xlabel(r"$|\log\kappa|$")
    axes[0].set_ylabel(r"$J_\kappa(\phi_{\kappa,h})$")
    axes[0].set_title("Raw energy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].semilogx(kappa_arr, norm_arr, "o-", color=DIAGNOSTIC_BLUE, linewidth=2.0, markersize=6.0)
    if kappa_arr.size:
        axes[1].set_xlim(DIAGNOSTIC_KAPPA_MAX, DIAGNOSTIC_KAPPA_MIN)
    axes[1].set_xlabel(r"$\kappa$")
    axes[1].set_ylabel(r"$J_\kappa(\phi_{\kappa,h})/|\log\kappa|$")
    axes[1].set_title("Normalized energy")
    axes[1].grid(alpha=0.25, which="both")

    axes[2].semilogx(kappa_arr, exact_ratio_arr, "o-", color=DIAGNOSTIC_BLUE, linewidth=2.0, markersize=6.0)
    if kappa_arr.size:
        axes[2].axhline(1.0, color=GUIDE_RED, linestyle="--", linewidth=1.8, label="exact constant")
        axes[2].set_xlim(DIAGNOSTIC_KAPPA_MAX, DIAGNOSTIC_KAPPA_MIN)
    axes[2].set_xlabel(r"$\kappa$")
    axes[2].set_ylabel(r"$J_\kappa/(A_\Sigma|\log\kappa|)$")
    axes[2].set_title("Exact constant ratio")
    axes[2].grid(alpha=0.25, which="both")
    if kappa_arr.size:
        axes[2].legend(fontsize=9)

    fig.suptitle(f"Energy scaling: {benchmark.title}", fontsize=12)
    fig.savefig(outfile)
    plt.close(fig)


def plot_mesh_comparison(rows, outfile):
    kappas = sorted({float(row["kappa"]) for row in rows}, reverse=True)
    strategies = ["uniform", "graded"]
    labels = [
        rf"$10^{{{int(round(math.log10(kappa)))}}}$" if kappa < 1.0 else rf"${kappa:g}$"
        for kappa in kappas
    ]
    x = np.arange(len(kappas))
    width = 0.32

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), constrained_layout=True)
    for offset, strategy in zip([-0.5 * width, 0.5 * width], strategies):
        boundary_vals = []
        bulk_vals = []
        node_counts = []
        for kappa in kappas:
            row = next(
                item for item in rows if float(item["kappa"]) == kappa and item["mesh_strategy"] == strategy
            )
            boundary_vals.append(float(row["boundary_error_to_reference"]))
            bulk_vals.append(float(row["bulk_error_to_reference"]))
            node_counts.append(int(row["num_nodes"]))

        bars0 = axes[0].bar(x + offset, boundary_vals, width=width, label=strategy.title())
        bars1 = axes[1].bar(x + offset, bulk_vals, width=width, label=strategy.title())
        for bar, n_nodes in zip(bars0, node_counts):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"N={n_nodes}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
        for bar, n_nodes in zip(bars1, node_counts):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"N={n_nodes}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("Boundary error to reference")
    axes[1].set_ylabel(r"$L^2(K)$ error to reference")
    axes[1].set_xlabel(r"$\kappa$")
    axes[0].set_xticks(x, labels)
    axes[1].set_xticks(x, labels)
    axes[0].grid(alpha=0.25, axis="y", which="both")
    axes[1].grid(alpha=0.25, axis="y", which="both")
    axes[0].legend()
    axes[1].legend()
    axes[0].set_title("Single-junction mesh comparison")
    fig.savefig(outfile)
    plt.close(fig)


def plot_exact_constant_refinement(rows, parameters, outfile):
    filtered = [
        row
        for row in rows
        if as_bool(row.get("certified", False)) and is_admissible_row(row, parameters)
    ]
    kappa_arr = np.array([float(row["kappa"]) for row in filtered], dtype=float)
    ratio_arr = np.array([float(row["exact_constant_ratio"]) for row in filtered], dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    ax.semilogx(
        kappa_arr,
        ratio_arr,
        "o-",
        color=DIAGNOSTIC_BLUE,
        linewidth=2.0,
        markersize=6.0,
        label=r"computed $J_\kappa/(A_\Sigma|\log\kappa|)$",
    )
    if kappa_arr.size:
        ax.axhline(1.0, color=GUIDE_RED, linestyle="--", linewidth=1.8, label="predicted limit")
        ax.set_xlim(float(np.max(kappa_arr)), float(np.min(kappa_arr)))
    ax.set_xlabel(r"$\kappa$ with $h_{\min}\simeq 0.05\kappa$")
    ax.set_ylabel(r"$J_\kappa/(A_\Sigma|\log\kappa|)$")
    ax.set_title("Exact-constant refinement path")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.savefig(outfile)
    plt.close(fig)


def plot_mesh_comparison_refined(rows, outfile):
    three_panel_scale = 14.2 / 10.4
    title_size = PLOT_TITLE_SIZE * three_panel_scale
    title_pad = PLOT_TITLE_PAD * three_panel_scale
    label_size = PLOT_LABEL_SIZE * three_panel_scale
    tick_size = PLOT_TICK_SIZE * three_panel_scale
    legend_size = PLOT_LEGEND_SIZE * three_panel_scale
    marker_size = PLOT_MARKER_SIZE * three_panel_scale
    line_width = PLOT_LINE_WIDTH * three_panel_scale

    def numeric_value(row, key):
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            return float("nan")
        return value if math.isfinite(value) else float("nan")

    strategies = ("uniform_matched", "uniform_refined", "graded_proposed")
    labels = {
        "uniform_matched": r"Uniform ($N$ nodes)",
        "uniform_refined": r"Uniform ($4N$ nodes)",
        "graded_proposed": r"Graded ($N$ nodes)",
    }
    colors = {
        "uniform_matched": DIAGNOSTIC_BLUE,
        "uniform_refined": GUIDE_RED,
        "graded_proposed": "#000000",
    }
    markers = {
        "uniform_matched": "s",
        "uniform_refined": "D",
        "graded_proposed": "o",
    }
    linestyles = {
        "uniform_matched": (0, (5, 2)),
        "uniform_refined": (0, (1, 1.6)),
        "graded_proposed": "-",
    }

    by_strategy = {s: [] for s in strategies}
    for row in rows:
        key = str(row["mesh_strategy"])
        if key in by_strategy:
            by_strategy[key].append(row)
    for s in strategies:
        by_strategy[s].sort(key=lambda r: float(r["kappa"]), reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.15), constrained_layout=True)
    target_panel_width_to_height = 1.253

    for s in strategies:
        rs = by_strategy[s]
        if not rs:
            continue
        kap = np.array([float(r["kappa"]) for r in rs], dtype=float)
        bdry = np.array([float(r["boundary_error_to_reference"]) for r in rs], dtype=float)
        hquarter = np.array([numeric_value(r, "boundary_hquarter_to_reference") for r in rs], dtype=float)
        bulk = np.array([float(r["bulk_error_to_reference"]) for r in rs], dtype=float)
        axes[0].loglog(
            kap,
            np.maximum(bdry, np.finfo(float).tiny),
            marker=markers[s],
            color=colors[s],
            linestyle=linestyles[s],
            linewidth=line_width,
            markersize=marker_size,
            label=labels[s],
            markerfacecolor="white" if s != "graded_proposed" else colors[s],
            markeredgewidth=1.6 * three_panel_scale,
        )
        axes[1].loglog(
            kap,
            np.maximum(hquarter, np.finfo(float).tiny),
            marker=markers[s],
            color=colors[s],
            linestyle=linestyles[s],
            linewidth=line_width,
            markersize=marker_size,
            label=labels[s],
            markerfacecolor="white" if s != "graded_proposed" else colors[s],
            markeredgewidth=1.6 * three_panel_scale,
        )
        axes[2].loglog(
            kap,
            np.maximum(bulk, np.finfo(float).tiny),
            marker=markers[s],
            color=colors[s],
            linestyle=linestyles[s],
            linewidth=line_width,
            markersize=marker_size,
            label=labels[s],
            markerfacecolor="white" if s != "graded_proposed" else colors[s],
            markeredgewidth=1.6 * three_panel_scale,
        )

    for ax in axes:
        ax.set_box_aspect(1.0 / target_panel_width_to_height)
        ax.invert_xaxis()
        ax.grid(False)
        ax.tick_params(axis="both", which="major", labelsize=tick_size)
        ax.set_xlabel(r"$\kappa$", fontsize=label_size, labelpad=7)
    axes[0].set_ylabel(r"$\|\phi_{\kappa,h}-\phi_{\kappa,h_*}\|_{L^2(\Gamma_*)}$", fontsize=label_size)
    axes[0].set_title(r"$L^2$ error to reference", fontsize=title_size, pad=title_pad)
    axes[1].set_ylabel(r"$\|\phi_{\kappa,h}-\phi_{\kappa,h_*}\|_{H^{1/4}(\Gamma_*)}$", fontsize=label_size)
    axes[1].set_title(r"$H^{1/4}$ error to reference", fontsize=title_size, pad=title_pad)
    legend1 = axes[1].legend(
        fontsize=legend_size,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        columnspacing=0.9 * three_panel_scale,
        handlelength=1.6 * three_panel_scale,
        borderaxespad=0.0,
        frameon=True,
    )
    legend1.set_in_layout(False)
    axes[2].set_ylabel(r"$\|\phi_{\kappa,h}-\phi_{\kappa,h_*}\|_{L^2(K)}$", fontsize=label_size)
    axes[2].set_title(r"Interior $L^2$ error on $K$", fontsize=title_size, pad=title_pad)
    bold_legend_frame(legend1)

    fig.savefig(outfile, bbox_inches="tight", bbox_extra_artists=(legend1,))
    plt.close(fig)


def style_3d_axis(ax):
    ax.grid(False)
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    ax.tick_params(axis="both", which="major", labelsize=8, pad=1)


def kappa_tick_label(kappa):
    if np.isclose(kappa, 1.0):
        return r"$1$"
    exponent = int(round(math.log10(kappa)))
    mantissa = kappa / (10.0**exponent)
    if np.isclose(mantissa, 1.0):
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:g}\times10^{{{exponent}}}$"


def plot_kappa_evolution_3d(problem, solutions, outfile):
    target_kappas = (1.0e-3, 1.0e-5, 1.0e-7)
    selected = []
    for target in target_kappas:
        if target in solutions:
            selected.append((target, solutions[target]))
            continue
        # Fall back to the closest available kappa in log scale.
        if not solutions:
            continue
        log_target = math.log10(target)
        nearest = min(
            solutions.keys(),
            key=lambda k: abs(math.log10(k) - log_target),
        )
        selected.append((nearest, solutions[nearest]))
    if len(selected) < 1:
        raise ValueError("At least one saved solution is required for the 3D evolution plot.")

    parameters = problem.physics
    cmap = plt.get_cmap("jet")
    norm = matplotlib.colors.Normalize(vmin=parameters.phi_a, vmax=parameters.phi_c)
    triangulation = mtri.Triangulation(
        1.0e3 * problem.nodes[:, 0], 1.0e3 * problem.nodes[:, 1], problem.triangles
    )

    panel_count = len(selected)
    fig = plt.figure(figsize=(4.8 * panel_count + 2.0, 5.1))
    gs = fig.add_gridspec(
        1,
        panel_count,
        left=0.045,
        right=0.835,
        bottom=0.20,
        top=0.965,
        wspace=0.18,
    )

    base_z = parameters.phi_a - 0.04
    step_lift = 0.010
    for col, (kappa, values) in enumerate(selected):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        surface = ax.plot_trisurf(
            triangulation,
            values,
            cmap=cmap,
            norm=norm,
            linewidth=0.0,
            antialiased=True,
            shade=False,
            alpha=0.98,
        )
        surface.set_rasterized(True)
        ax.tricontour(
            triangulation,
            values,
            levels=np.linspace(parameters.phi_a, parameters.phi_c, 9),
            zdir="z",
            offset=base_z,
            cmap=cmap,
            norm=norm,
            linewidths=0.65,
            alpha=0.95,
        )
        step_x = 1.0e3 * problem.nodes[problem.bottom_nodes, 0]
        step_y = np.zeros(problem.bottom_nodes.size)
        step_z = problem.bottom_node_values + step_lift
        step_underlay, = ax.plot(
            step_x,
            step_y,
            step_z,
            color="white",
            linewidth=6.2,
            solid_capstyle="round",
            zorder=1000,
        )
        step_line, = ax.plot(
            step_x,
            step_y,
            step_z,
            color="black",
            linewidth=3.4,
            solid_capstyle="round",
            zorder=1001,
        )
        step_line.set_path_effects([pe.Stroke(linewidth=5.0, foreground="white"), pe.Normal()])
        step_underlay.set_path_effects([pe.Stroke(linewidth=7.4, foreground="white"), pe.Normal()])
        ax.set_xlim(0.0, 1.0e3 * parameters.lx)
        ax.set_ylim(0.0, 1.0e3 * parameters.ly)
        ax.set_zlim(base_z, parameters.phi_c + 0.025)
        ax.view_init(elev=27, azim=-62)
        ax.set_box_aspect((1.75, 0.92, 0.78))
        style_3d_axis(ax)
        ax.set_xlabel(r"$x$ [mm]", labelpad=10, fontsize=PLOT_LABEL_SIZE + 3)
        ax.set_ylabel(r"$y$ [mm]", labelpad=10, fontsize=PLOT_LABEL_SIZE + 3)
        ax.set_zlabel(r"$\phi_{\kappa,h}$", labelpad=10, fontsize=PLOT_LABEL_SIZE + 3)
        for axis_name in ("x", "y", "z"):
            ax.tick_params(axis=axis_name, which="major", labelsize=PLOT_TICK_SIZE + 2, pad=3)
        ax.set_title(
            rf"$\kappa={kappa_tick_label(kappa).strip('$')}$",
            y=-0.20,
            fontsize=PLOT_TITLE_SIZE + 1,
        )

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cax = fig.add_axes([0.915, 0.25, 0.026, 0.56])
    cbar = fig.colorbar(mappable, cax=cax, ticks=np.linspace(parameters.phi_a, parameters.phi_c, 5))
    cbar.ax.set_title("potential", fontsize=PLOT_TITLE_SIZE, pad=12)
    cbar.ax.tick_params(labelsize=PLOT_TICK_SIZE + 2, pad=5)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_junction_microscope(problem, u, kappa, outfile):
    parameters = problem.physics
    triangulation = mtri.Triangulation(problem.nodes[:, 0], problem.nodes[:, 1], problem.triangles)
    grad_mag = triangle_gradient_magnitude(problem, u)
    log_grad = np.log10(np.maximum(grad_mag, 1.0e-12))
    x_junction = 0.5 * parameters.lx

    local_x = 1.0e3 * (problem.nodes[:, 0] - x_junction)
    local_y = 1.0e3 * problem.nodes[:, 1]
    local_triangulation = mtri.Triangulation(local_x, local_y, problem.triangles)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    tpc = axes[0].tripcolor(
        triangulation,
        u,
        shading="gouraud",
        cmap=POTENTIAL_CMAP,
        vmin=parameters.phi_a,
        vmax=parameters.phi_c,
    )
    style_solution_axes(axes[0], problem)
    axes[0].set_title(rf"Full field, $\kappa={kappa:g}$", fontsize=15)
    cbar0 = fig.colorbar(tpc, ax=axes[0], fraction=0.045, pad=0.018, ticks=np.linspace(parameters.phi_a, parameters.phi_c, 5))
    cbar0.set_label("potential")

    gpc = axes[1].tripcolor(
        local_triangulation,
        facecolors=log_grad,
        shading="flat",
        cmap=POTENTIAL_CMAP,
    )
    levels = np.linspace(parameters.phi_a, parameters.phi_c, 9)
    axes[1].tricontour(local_triangulation, u, levels=levels, colors="white", linewidths=0.45, alpha=0.70)
    axes[1].axvline(0.0, color=DIAGNOSTIC_GREEN, linewidth=1.8, alpha=0.95)
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-1.75, 1.75)
    axes[1].set_ylim(0.0, 2.2)
    axes[1].set_xlabel(r"$10^3(x-x_j)$")
    axes[1].set_ylabel(r"$10^3 y$")
    axes[1].set_title("Junction-gradient microscope", fontsize=15)
    axes[1].set_facecolor(DIAGNOSTIC_BLUE)
    axes[1].tick_params(labelsize=9)
    cbar1 = fig.colorbar(gpc, ax=axes[1], fraction=0.045, pad=0.018)
    cbar1.set_label(r"$\log_{10}|\nabla\phi_{\kappa,h}|$")
    fig.savefig(outfile)
    plt.close(fig)
