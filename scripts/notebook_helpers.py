"""Small, reader-facing helpers for the numerical-experiments notebook.

The notebook deliberately contains the mathematics rather than repository and
file-management code.  This module provides the few high-level operations it
needs: loading one coherent data source, displaying a paper figure, preparing
compact tables, and optionally launching a fresh computation in process.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import IFrame, Markdown, display

from .diagnostics import (
    DIAGNOSTIC_BLUE,
    GUIDE_RED,
    PLOT_LABEL_SIZE,
    PLOT_LEGEND_SIZE,
    PLOT_LINE_WIDTH,
    bold_legend_frame,
    set_panel_title,
    style_plot_axes,
)


INCLUDED_SUMMARIES = Path("data/summaries")
INCLUDED_METADATA = Path("data/metadata")
INCLUDED_TRACES = INCLUDED_SUMMARIES / "single_junction_traces.csv"
COMPUTED_DATA = Path("data/computed")
INCLUDED_FIGURES = Path("paper_figures")
GENERATED_FIGURES = Path("paper_figures/generated")


SUMMARY_FILES = {
    "convergence": "convergence_summary.csv",
    "exact_single": "exact_constant_refinement_summary.csv",
    "exact_multi": "multi_exact_constant_refinement_summary.csv",
    "corrugated_four": "corrugated_stress_summary.csv",
    "corrugated_six": "corrugated_stress_corrugated_six_junction_summary.csv",
    "corrugated_retention": "corrugated_admissible_diagnostics_summary.csv",
    "practical_four": "practical_stainless_zro2_corrugated_four_junction_summary.csv",
    "practical_six": "practical_stainless_zro2_corrugated_six_junction_summary.csv",
    "mesh_comparison": "mesh_comparison_refined_single_summary.csv",
    "regressions": "theorem_dashboard_regressions.csv",
}


FIGURE_TITLES = {
    "kappa_evolution_3d_single": "Single-junction evolution",
    "theorem_dashboard": "Boundary and interior convergence",
    "corrugated_six_junction_diagnostics": "Corrugated six-junction diagnostics",
    "practical_stainless_zro2_corrugated_diagnostics": "Practical-parameter robustness",
    "mesh_strategy_triptych": "Meshes near the reactive junctions",
    "mesh_comparison_refined_single": "Quantitative mesh comparison",
}


def choose_source(recompute):
    """Choose one data source for everything shown in the notebook."""
    return "computed" if recompute else "included"

def show_table(table):
    """Display a dataframe as a Markdown table so its formulas are rendered."""

    def format_value(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            size = abs(value)
            if size != 0.0 and (size < 1.0e-3 or size >= 1.0e4):
                return f"{value:.3e}"
            return f"{value:.6g}"
        return str(value).replace("|", r"\|").replace("\n", "<br>")

    columns = [str(name) for name in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    display(Markdown("\n".join(lines)))


def baseline_parameter_table():
    """Baseline electrochemical and geometric parameters."""
    return pd.DataFrame(
        [
            (r"$L_x$", "0.02 m", "rectangle length"),
            (r"$L_y$", "0.01 m", "rectangle height"),
            (r"$\phi_c$", "0.2", "cathodic equilibrium potential"),
            (r"$\phi_a$", "-0.2", "anodic equilibrium potential"),
            (r"$i_{c,0}$", r"$3.0\times10^{-4}$", "cathodic exchange-current scale"),
            (r"$i_{a,0}$", r"$3.0\times10^{-2}$", "anodic exchange-current scale"),
            (r"$C_1=A_2$", "83.13", "positive exponential slopes"),
            (r"$C_2=A_1$", "35.63", "negative exponential slopes"),
        ],
        columns=["symbol", "value", "role"],
    )


def benchmark_table(source):
    """Summarize the four geometries without assembling their stiffness matrices."""
    experiment = load_metadata("experiment_metadata.json", source)
    corrugated_four = load_metadata("corrugated_stress_metadata.json", source)
    corrugated_six = load_metadata(
        "corrugated_stress_corrugated_six_junction_metadata.json", source
    )
    rectangle = experiment["benchmarks"]
    rows = [
        {
            "benchmark": "single rectangle",
            "geometry": "rectangle",
            "junctions N": 1,
            "physics": "baseline",
            "h_min": 2.0e-8,
            "nodes": rectangle["single_junction_mixed"]["num_nodes"],
            "triangles": rectangle["single_junction_mixed"]["num_triangles"],
        },
        {
            "benchmark": "multi rectangle",
            "geometry": "rectangle",
            "junctions N": 3,
            "physics": "baseline",
            "h_min": 5.0e-7,
            "nodes": rectangle["multi_junction_mixed"]["num_nodes"],
            "triangles": rectangle["multi_junction_mixed"]["num_triangles"],
        },
        {
            "benchmark": "corrugated four-junction",
            "geometry": "corrugated top",
            "junctions N": corrugated_four["junction_count"],
            "physics": "balanced stress test",
            "h_min": corrugated_four["hmin"],
            "nodes": corrugated_four["num_nodes"],
            "triangles": corrugated_four["num_triangles"],
        },
        {
            "benchmark": "corrugated six-junction",
            "geometry": "corrugated top",
            "junctions N": corrugated_six["junction_count"],
            "physics": "balanced stress test",
            "h_min": corrugated_six["hmin"],
            "nodes": corrugated_six["num_nodes"],
            "triangles": corrugated_six["num_triangles"],
        },
    ]
    return pd.DataFrame(rows)


def load_summary(name, source):
    """Load a named table from the selected source."""
    validate_source(source)
    if name not in SUMMARY_FILES:
        available = ", ".join(sorted(SUMMARY_FILES))
        raise KeyError(f"Unknown summary {name!r}; choose one of {available}.")
    filename = SUMMARY_FILES[name]
    folder = INCLUDED_SUMMARIES if source == "included" else COMPUTED_DATA
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(
            f"The {source} data do not contain {filename}. "
            "Run the complete recomputation before selecting computed results."
        )
    return pd.read_csv(path)


def single_trace_diagnostics(source):
    """Solver diagnostics for the two conductivities in the worked example."""
    rows = load_summary("convergence", source)
    rows = rows[
        (rows["benchmark"] == "single_junction_mixed")
        & rows["kappa"].isin([1.0e-2, 1.0e-4])
    ].copy()
    rows = rows.sort_values("kappa", ascending=False)
    return rows[["kappa", "num_nodes", "iterations", "residual_inf"]].rename(
        columns={
            "kappa": r"$\kappa$",
            "num_nodes": "nodes",
            "iterations": "nonlinear iterations",
            "residual_inf": r"$\ell^\infty$ residual",
        }
    ).reset_index(drop=True)


def plot_single_junction_traces(source):
    """Plot the included or freshly computed bottom-boundary traces."""
    traces = single_junction_traces(source)
    display(Markdown("#### Single-junction bottom trace"))
    fig, ax = plt.subplots(figsize=(10.4, 3.8), constrained_layout=True)
    x_mm = 1.0e3 * traces["x_m"].to_numpy()
    ax.plot(
        x_mm,
        traces["phi_limit"],
        color="black",
        linewidth=PLOT_LINE_WIDTH,
        label=r"limit $\Phi_0$",
    )
    ax.plot(
        x_mm,
        traces["phi_kappa_1e_minus_2"],
        color=DIAGNOSTIC_BLUE,
        linewidth=PLOT_LINE_WIDTH,
        label=r"$\kappa=10^{-2}$",
    )
    ax.plot(
        x_mm,
        traces["phi_kappa_1e_minus_4"],
        color=GUIDE_RED,
        linewidth=PLOT_LINE_WIDTH,
        label=r"$\kappa=10^{-4}$",
    )
    ax.axvline(10.0, color="0.55", linestyle="--", linewidth=1.1)
    ax.set_xlabel(r"bottom-boundary coordinate $x$ (mm)", fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel(r"potential $\phi_{\kappa,h}(x,0)$", fontsize=PLOT_LABEL_SIZE)
    set_panel_title(ax, "The reactive transition sharpens at the junction")
    style_plot_axes(ax)
    legend = ax.legend(fontsize=PLOT_LEGEND_SIZE, frameon=True)
    bold_legend_frame(legend)
    display(fig)
    plt.close(fig)


def single_convergence_table(source):
    """Representative certified rows in the theorem-facing window."""
    rows = load_summary("convergence", source)
    rows = rows[
        (rows["benchmark"] == "single_junction_mixed")
        & truth_values(rows["certified"])
        & (rows["kappa"] >= 1.3e-7)
        & (rows["kappa"] <= 1.0e-4)
    ].copy()
    wanted = np.array([1.0e-4, 1.0e-5, 1.0e-6, 4.0e-7, 2.0e-7, 1.3e-7])
    keep = np.array([np.any(np.isclose(value, wanted)) for value in rows["kappa"]])
    rows = rows[keep].sort_values("kappa", ascending=False)
    return rows[
        ["kappa", "boundary_error_l2_sq", "bulk_error_l2_K", "residual_inf"]
    ].rename(
        columns={
            "kappa": r"$\kappa$",
            "boundary_error_l2_sq": r"boundary $L^2$ error squared",
            "bulk_error_l2_K": r"interior $L^2(K)$ error",
            "residual_inf": r"$\ell^\infty$ residual",
        }
    ).reset_index(drop=True)


def regression_table(source):
    """Observed log--log slopes and bootstrap intervals."""
    rows = load_summary("regressions", source).copy()
    labels = {
        "boundary_L2_sq_vs_kappa_log_kappa": r"boundary $L^2$ error squared",
        "interior_EK_sq_vs_kappa_log_kappa": r"interior $L^2(K)$ error squared",
        "interior_C0_vs_kappa_log_kappa": r"interior $C^0(K)$ error",
        "interior_C1_vs_kappa_log_kappa": r"interior $C^1(K)$ error",
        "energy_remainder_vs_loglog_over_log": "energy-ratio remainder",
    }
    rows["diagnostic"] = rows["diagnostic"].map(labels).fillna(rows["diagnostic"])
    return rows[
        ["diagnostic", "slope", "slope_ci_low", "slope_ci_high", "predicted_slope"]
    ].rename(
        columns={
            "slope": "observed slope",
            "slope_ci_low": "95% CI low",
            "slope_ci_high": "95% CI high",
            "predicted_slope": "reference slope",
        }
    )


def exact_constant_table(source):
    """Single- and three-junction proportional-refinement results."""
    single = load_summary("exact_single", source).copy()
    multi = load_summary("exact_multi", source).copy()
    single["junctions N"] = 1
    multi["junctions N"] = 3
    rows = pd.concat([single, multi], ignore_index=True)
    rows = rows.sort_values(["junctions N", "kappa"], ascending=[True, False])
    return rows[
        [
            "junctions N",
            "kappa",
            "h_min_over_kappa",
            "num_nodes",
            "exact_energy_constant",
            "exact_constant_ratio",
            "residual_inf",
        ]
    ].rename(
        columns={
            "kappa": r"$\kappa$",
            "h_min_over_kappa": r"$h_{\min}/\kappa$",
            "num_nodes": "nodes",
            "exact_energy_constant": r"$C_N$",
            "exact_constant_ratio": r"$R_E(\kappa)$",
            "residual_inf": r"$\ell^\infty$ residual",
        }
    ).reset_index(drop=True)


def corrugated_table(source):
    """Smallest admissible conductivities for the six-junction geometry."""
    rows = admissible_rows(load_summary("corrugated_six", source))
    rows = rows[rows["kappa"] < 1.0].sort_values("kappa").head(6)
    return rows[
        ["kappa", "boundary_error_l2_sq", "exact_constant_ratio", "residual_inf"]
    ].rename(
        columns={
            "kappa": r"$\kappa$",
            "boundary_error_l2_sq": r"boundary $L^2$ error squared",
            "exact_constant_ratio": r"$R_E(\kappa)$",
            "residual_inf": r"$\ell^\infty$ residual",
        }
    ).reset_index(drop=True)


def practical_parameter_table(source):
    """Parameter ratios used in the ZrO2-coated stainless-steel check."""
    row = load_summary("practical_four", source).iloc[0]
    values = [
        (r"$\phi_a$", row["phi_a"], "anodic equilibrium potential"),
        (r"$\phi_c$", row["phi_c"], "cathodic equilibrium potential"),
        (r"$i_{a,0}/i_{c,0}$", row["ia0"] / row["ic0"], "exchange-current ratio"),
        (r"$A_1=A_2$", row["a1"], "anodic exponential slopes"),
        (r"$C_1=C_2$", row["c1"], "cathodic exponential slopes"),
    ]
    return pd.DataFrame(values, columns=["symbol", "value", "role"])


def practical_summary(source):
    """Smallest retained and next excluded practical-parameter rows."""
    output = []
    for junctions, key in ((4, "practical_four"), (6, "practical_six")):
        all_rows = load_summary(key, source).sort_values("kappa")
        retained = admissible_rows(all_rows)
        retained = retained[retained["kappa"] < 1.0].sort_values("kappa")
        best = retained.iloc[0]
        smaller = all_rows[all_rows["kappa"] < best["kappa"]]
        next_row = smaller.iloc[-1] if not smaller.empty else None
        output.append(
            {
                "junctions N": junctions,
                "smallest retained kappa": best["kappa"],
                "energy ratio at retained point": best["exact_constant_ratio"],
                "next excluded kappa": np.nan if next_row is None else next_row["kappa"],
                "next residual": np.nan if next_row is None else next_row["residual_inf"],
            }
        )
    return pd.DataFrame(output)


def mesh_comparison_table(source, kappa=1.0e-5):
    """Compare the three mesh strategies at one conductivity."""
    rows = load_summary("mesh_comparison", source)
    rows = rows[np.isclose(rows["kappa"], kappa)].copy()
    order = {"graded_proposed": 0, "uniform_matched": 1, "uniform_refined": 2}
    rows["order"] = rows["mesh_strategy"].map(order)
    rows = rows.sort_values("order")
    rows["mesh_strategy"] = rows["mesh_strategy"].map(
        {
            "graded_proposed": "graded (N)",
            "uniform_matched": "uniform (N)",
            "uniform_refined": "uniform (4N)",
        }
    )
    return rows[
        [
            "mesh_strategy",
            "num_nodes",
            "boundary_error_to_reference",
            "boundary_hquarter_to_reference",
            "bulk_error_to_reference",
            "exact_constant_ratio",
        ]
    ].rename(
        columns={
            "mesh_strategy": "mesh",
            "num_nodes": "nodes",
            "boundary_error_to_reference": r"boundary $L^2$ error",
            "boundary_hquarter_to_reference": r"boundary $H^{1/4}$ error",
            "bulk_error_to_reference": r"interior $L^2(K)$ error",
            "exact_constant_ratio": r"$R_E(\kappa)$",
        }
    ).reset_index(drop=True)


def show_figure(stem, source, height=520):
    """Display a figure from the selected source."""
    validate_source(source)
    if stem not in FIGURE_TITLES:
        available = ", ".join(sorted(FIGURE_TITLES))
        raise KeyError(f"Unknown figure {stem!r}; choose one of {available}.")
    display(Markdown(f"#### {FIGURE_TITLES[stem]}"))
    if source == "included":
        pdf_path = INCLUDED_FIGURES / f"{stem}.pdf"
    else:
        pdf_path = GENERATED_FIGURES / f"{stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"The {source} figure {stem}.pdf is missing. Run the complete recomputation first."
        )
    display(IFrame(src=str(pdf_path), width="100%", height=height))


def recompute_all():
    """Recompute every table and figure used by the notebook."""
    from scripts.reproduce_paper import reproduce

    reproduce("paper-figures")


def validate_source(source):
    if source not in {"included", "computed"}:
        raise ValueError("source must be either 'included' or 'computed'.")


def load_metadata(filename, source):
    validate_source(source)
    folder = INCLUDED_METADATA if source == "included" else COMPUTED_DATA
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(
            f"The {source} data do not contain {filename}. "
            "Run the complete recomputation before selecting computed results."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def truth_values(values):
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def admissible_rows(rows):
    mask = truth_values(rows["certified"])
    phi_a = rows["phi_a"] if "phi_a" in rows else -0.2
    phi_c = rows["phi_c"] if "phi_c" in rows else 0.2
    if {"solution_min", "solution_max"}.issubset(rows.columns):
        mask &= rows["solution_min"] >= phi_a - 1.0e-8
        mask &= rows["solution_max"] <= phi_c + 1.0e-8
    if "exponential_clamp_active" in rows:
        mask &= ~truth_values(rows["exponential_clamp_active"])
    return rows[mask].copy()


def single_junction_traces(source):
    validate_source(source)
    if source == "included":
        return pd.read_csv(INCLUDED_TRACES)

    solution_2 = np.load(COMPUTED_DATA / "solution_single_junction_mixed_graded_0_01.npz")
    solution_4 = np.load(COMPUTED_DATA / "solution_single_junction_mixed_graded_1em04.npz")
    reference = np.load(COMPUTED_DATA / "u0_single_junction_mixed.npz")
    x = np.asarray(solution_2["x"])
    bottom_count = x.size
    return pd.DataFrame(
        {
            "x_m": x,
            "phi_limit": np.asarray(reference["potential"][:bottom_count]),
            "phi_kappa_1e_minus_2": np.asarray(solution_2["potential"][:bottom_count]),
            "phi_kappa_1e_minus_4": np.asarray(solution_4["potential"][:bottom_count]),
        }
    )
