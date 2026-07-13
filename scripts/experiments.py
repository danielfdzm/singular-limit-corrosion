"""Steps used to run the numerical experiments.

The functions here assemble the benchmark, corrugated-geometry, mesh,
refinement, and cached-data studies used for the paper.
"""

import math

import numpy as np

from .diagnostics import (
    as_bool,
    boundary_error_to_reference,
    boundary_hhalf_norm,
    boundary_hquarter_error_to_reference,
    boundary_l2_error_sq,
    bulk_error_to_reference,
    bulk_sample_points,
    cathode_anode_junction_count,
    certified_residual_threshold,
    exact_constant_ratio,
    exact_log_energy_constant,
    is_certified_info,
    max_butler_volmer_exponent_argument,
    normalized_energy,
)
from .mesh import build_problem, solve_mixed_reference
from .model import corrugated_top_height
from .nonlinear_solver import (
    energy_gradient_hessian,
    polish_with_unconstrained_newton,
    solve_with_continuation,
    solve_problem,
)
from .solution_plots import (
    plot_boundary_convergence,
    plot_boundary_traces,
    plot_bulk_error,
    plot_corrugated_diagnostics,
    plot_corrugated_snapshots,
    plot_energy_scaling,
    plot_exact_constant_refinement,
    plot_junction_microscope,
    plot_kappa_evolution_3d,
    plot_mesh_comparison,
    plot_mesh_comparison_refined,
    plot_snapshot,
)
from .storage import (
    load_saved_solution,
    read_csv,
    save_reference,
    save_solution,
    write_csv,
    write_json,
)


def augment_exact_energy_columns(
    rows,
    parameters,
    benchmarks,
):
    for row in rows:
        benchmark = benchmarks.get(str(row.get("benchmark", "")))
        if benchmark is None:
            continue
        kappa = float(row["kappa"])
        energy = float(row["energy"])
        constant = exact_log_energy_constant(benchmark, parameters)
        row["exact_energy_constant"] = constant
        row["exact_constant_ratio"] = (
            exact_constant_ratio(energy, kappa, benchmark, parameters) if constant > 0.0 else float("nan")
        )


def postprocess_convergence_csv(
    parameters,
    benchmarks,
    selected,
    mesh_strategy,
    output_dir,
    data_dir,
):
    csv_file = data_dir / "convergence_summary.csv"
    rows = read_csv(csv_file)
    augment_exact_energy_columns(rows, parameters, benchmarks)
    write_csv(rows, csv_file)

    for benchmark in selected:
        bench_rows = [
            row
            for row in rows
            if row.get("benchmark") == benchmark.name
            and row.get("mesh_strategy", mesh_strategy) == mesh_strategy
            and row.get("physics_variant", "asymmetric") == "asymmetric"
        ]
        bench_rows.sort(key=lambda item: float(item["kappa"]), reverse=True)
        plot_boundary_convergence(
            bench_rows,
            benchmark,
            parameters,
            output_dir / f"boundary_convergence_{benchmark.file_tag}.pdf",
        )
        plot_bulk_error(bench_rows, benchmark, parameters, output_dir / f"bulk_error_to_u0_{benchmark.file_tag}.pdf")
        plot_energy_scaling(bench_rows, benchmark, parameters, output_dir / f"energy_scaling_{benchmark.file_tag}.pdf")


def run_flashy_experiments(
    parameters,
    benchmarks,
    output_dir,
    data_dir,
):
    benchmark = benchmarks["single_junction_mixed"]
    problem = build_problem(parameters, benchmark, "graded", hmin=suite_hmin_for_benchmark(parameters, benchmark))
    requested_kappas = [
        1.0, 1.0e-1, 1.0e-2, 1.0e-4, 5.0e-5, 3.0e-5, 1.0e-5,
        5.0e-6, 1.0e-6, 4.0e-7, 3.0e-7, 2.5e-7, 2.0e-7,
        1.8e-7, 1.6e-7, 1.5e-7, 1.3e-7,
    ]
    convergence_path = data_dir / "convergence_summary.csv"
    if convergence_path.exists():
        rows = read_csv(convergence_path)
        certified = [
            float(row["kappa"])
            for row in rows
            if row.get("benchmark") == benchmark.name
            and as_bool(row.get("certified", False))
            and not as_bool(row.get("exponential_clamp_active", False))
        ]
        if certified:
            requested_kappas = sorted(set(certified), reverse=True)
    # Always include the three reference snapshots used in the 3D evolution plot,
    # even when they did not pass the certification filter, so the figure can be
    # reconstructed from saved solutions.
    snapshot_kappas = (1.0e-3, 1.0e-5, 1.0e-7)
    requested_kappas = sorted(set(requested_kappas) | set(snapshot_kappas), reverse=True)
    solutions = {}
    for kappa in requested_kappas:
        solution = load_saved_solution(problem, kappa, data_dir)
        if solution is not None:
            solutions[kappa] = solution
    if not solutions:
        raise FileNotFoundError(
            "No saved single-junction graded solutions were found. Run the graded benchmark suite first."
        )

    plot_kappa_evolution_3d(problem, solutions, output_dir / "kappa_evolution_3d_single.pdf")
    microscope_kappa = min(solutions)
    plot_junction_microscope(
        problem,
        solutions[microscope_kappa],
        microscope_kappa,
        output_dir / "junction_microscope_single.pdf",
    )
    metadata = {
        "benchmark": benchmark.name,
        "mesh_strategy": "graded",
        "num_nodes": int(problem.nodes.shape[0]),
        "num_triangles": int(problem.triangles.shape[0]),
        "kappas": [float(kappa) for kappa in sorted(solutions.keys(), reverse=True)],
        "evolution_plot": "kappa_evolution_3d_single.pdf",
        "microscope_kappa": float(microscope_kappa),
    }
    write_json(metadata, data_dir / "flashy_experiment_metadata.json")
    return metadata


def matching_uniform_node_count(parameters, benchmark):
    """Use about as many uniform-mesh nodes as the graded mesh has."""
    graded_problem = build_problem(parameters, benchmark, "graded", hmin=suite_hmin_for_benchmark(parameters, benchmark))
    return int(graded_problem.nodes.shape[0])


def suite_hmin_for_benchmark(parameters, benchmark):
    if benchmark.name == "single_junction_mixed":
        return parameters.single_suite_hmin
    return parameters.suite_hmin


def run_benchmark_suite(
    parameters,
    benchmark,
    mesh_strategy,
    output_dir,
    data_dir,
    kappa_override,
):
    if mesh_strategy == "graded":
        problem = build_problem(parameters, benchmark, "graded", hmin=suite_hmin_for_benchmark(parameters, benchmark))
    elif mesh_strategy == "uniform":
        problem = build_problem(
            parameters,
            benchmark,
            "uniform",
            target_nodes=matching_uniform_node_count(parameters, benchmark),
        )
    else:
        raise ValueError(f"Unsupported suite mesh strategy: {mesh_strategy}")

    kappas = tuple(sorted(set(kappa_override or benchmark.continuation_kappas), reverse=True))
    trace_kappas = set(benchmark.trace_kappas)
    snapshot_kappas = set(benchmark.snapshot_kappas)
    snapshot_names = {
        1.0: "kappa1.pdf",
        1.0e-1: "kappa0_1.pdf",
        1.0e-2: "kappa0_01.pdf",
        1.0e-4: "kappa0_0001.pdf",
        1.0e-5: "kappa0_00001.pdf",
        2.0e-7: "kappa2em07.pdf",
    }

    reference = solve_mixed_reference(problem)
    reference_file = save_reference(problem, reference, data_dir)
    sample_points = bulk_sample_points(parameters)

    rows = []
    traces_for_plot = {}

    previous = reference.copy()
    for kappa in kappas:
        print(f"[solve:{benchmark.name}:{mesh_strategy}] kappa={kappa:.1e}")
        u, info = solve_problem(problem, kappa, previous)
        previous = u
        boundary_error_sq = boundary_l2_error_sq(u, problem)
        max_exp_arg = max_butler_volmer_exponent_argument(u, parameters)
        energy_value = float(info["energy"])
        normalized_value = normalized_energy(energy_value, kappa)
        exact_constant = exact_log_energy_constant(benchmark, parameters)

        row = {
            "benchmark": benchmark.name,
            "mesh_strategy": mesh_strategy,
            "physics_variant": "asymmetric",
            "kappa": kappa,
            "num_nodes": int(problem.nodes.shape[0]),
            "num_triangles": int(problem.triangles.shape[0]),
            "energy": energy_value,
            "normalized_energy": normalized_value,
            "exact_energy_constant": exact_constant,
            "exact_constant_ratio": normalized_value / exact_constant if exact_constant > 0.0 and kappa < 1.0 else float("nan"),
            "boundary_error_l2_sq": boundary_error_sq,
            "bulk_error_l2_K": bulk_error_to_reference(problem, u, problem, reference, sample_points),
            "boundary_hhalf_norm": boundary_hhalf_norm(u, problem),
            "boundary_rate_ratio": boundary_error_sq / (kappa * abs(math.log(kappa))) if kappa < 1.0 else float("nan"),
            "iterations": float(info["iterations"]),
            "residual_inf": float(info["residual_inf"]),
            "initial_residual_inf": float(info.get("initial_residual_inf", float("nan"))),
            "certified_threshold": float(info.get("certified_threshold", certified_residual_threshold(parameters))),
            "accepted_step": float(info["accepted_step"]),
            "solver": str(info["solver"]),
            "certified": is_certified_info(info, parameters),
            "solution_min": float(np.min(u)),
            "solution_max": float(np.max(u)),
            "max_bv_exponent_argument": max_exp_arg,
            "exponential_clamp_active": max_exp_arg >= 80.0,
        }
        rows.append(row)
        save_solution(problem, kappa, u, data_dir)

        if benchmark.name == "single_junction_mixed" and mesh_strategy == "graded":
            if kappa in snapshot_kappas and kappa in snapshot_names:
                plot_snapshot(problem, u, output_dir / snapshot_names[kappa])
            if kappa in trace_kappas:
                traces_for_plot[kappa] = u[problem.bottom_nodes].copy()

    rows.sort(key=lambda item: float(item["kappa"]), reverse=True)
    slope = plot_boundary_convergence(
        rows,
        benchmark,
        parameters,
        output_dir / f"boundary_convergence_{benchmark.file_tag}.pdf",
    )
    plot_bulk_error(rows, benchmark, parameters, output_dir / f"bulk_error_to_u0_{benchmark.file_tag}.pdf")
    plot_energy_scaling(rows, benchmark, parameters, output_dir / f"energy_scaling_{benchmark.file_tag}.pdf")

    if benchmark.name == "single_junction_mixed" and mesh_strategy == "graded":
        plot_boundary_traces(problem, traces_for_plot, output_dir / "bottom_boundary_plot.pdf")

    metadata = {
        "benchmark": benchmark.name,
        "mesh_strategy": mesh_strategy,
        "physics_variant": "asymmetric",
        "num_nodes": int(problem.nodes.shape[0]),
        "num_triangles": int(problem.triangles.shape[0]),
        "reference_file": reference_file,
        "boundary_upper_scale": slope,
        "num_certified_points": int(sum(bool(row["certified"]) for row in rows)),
    }
    return rows, metadata


def run_corrugated_stress_test(
    parameters,
    benchmark,
    output_dir,
    data_dir,
    kappa_override,
):
    kappas = tuple(sorted(set(kappa_override or benchmark.continuation_kappas), reverse=True))
    hmin = min(3.0e-7, 0.2 * min(kappas))
    problem = build_problem(parameters, benchmark, "graded", hmin=hmin)
    reference = solve_mixed_reference(problem)
    snapshot_kappas = set(benchmark.snapshot_kappas)
    previous = reference.copy()
    rows = []
    snapshots = {}
    if benchmark.name == "corrugated_four_junction_mixed":
        summary_path = data_dir / "corrugated_stress_summary.csv"
        metadata_path = data_dir / "corrugated_stress_metadata.json"
    else:
        summary_path = data_dir / f"corrugated_stress_{benchmark.file_tag}_summary.csv"
        metadata_path = data_dir / f"corrugated_stress_{benchmark.file_tag}_metadata.json"

    for kappa in kappas:
        saved_solution = load_saved_solution(problem, kappa, data_dir)
        if saved_solution is not None:
            print(f"[corrugated:{benchmark.name}] kappa={kappa:.1e} (saved)")
            solution = saved_solution
            saved_energy, saved_gradient, _ = energy_gradient_hessian(solution, problem, kappa, with_hessian=False)
            info = {
                "iterations": 0.0,
                "residual_inf": float(np.linalg.norm(saved_gradient, ord=np.inf)),
                "accepted_step": 0.0,
                "energy": float(saved_energy),
                "solver": "cached",
            }
            if not is_certified_info(info, parameters):
                print(f"[corrugated:{benchmark.name}] kappa={kappa:.1e} polishing saved solution")
                solution, info = solve_problem(problem, kappa, solution)
                if not is_certified_info(info, parameters):
                    best_solution = solution
                    best_info = info
                    solution_unconstrained, info_unconstrained = polish_with_unconstrained_newton(
                        problem, kappa, solution
                    )
                    if float(info_unconstrained["residual_inf"]) < float(best_info["residual_inf"]):
                        best_solution = solution_unconstrained
                        best_info = info_unconstrained
                    if not is_certified_info(best_info, parameters):
                        solution_unconstrained, info_unconstrained = polish_with_unconstrained_newton(
                            problem, kappa, previous
                        )
                        if float(info_unconstrained["residual_inf"]) < float(best_info["residual_inf"]):
                            best_solution = solution_unconstrained
                            best_info = info_unconstrained
                    solution = best_solution
                    info = best_info
                save_solution(problem, kappa, solution, data_dir)
        else:
            print(f"[corrugated:{benchmark.name}] kappa={kappa:.1e}")
            solution, info = solve_problem(problem, kappa, previous)
            if not is_certified_info(info, parameters):
                best_solution = solution
                best_info = info
                for seed in (solution, previous):
                    solution_unconstrained, info_unconstrained = polish_with_unconstrained_newton(
                        problem, kappa, seed
                    )
                    if float(info_unconstrained["residual_inf"]) < float(best_info["residual_inf"]):
                        best_solution = solution_unconstrained
                        best_info = info_unconstrained
                    if is_certified_info(best_info, parameters):
                        break
                solution = best_solution
                info = best_info
            save_solution(problem, kappa, solution, data_dir)
        previous = solution
        energy = float(info["energy"])
        normalized_value = normalized_energy(energy, kappa)
        exact_constant = exact_log_energy_constant(benchmark, parameters)
        max_exp_arg = max_butler_volmer_exponent_argument(solution, parameters)
        row = {
            "benchmark": benchmark.name,
            "geometry": benchmark.geometry,
            "mesh_strategy": "graded",
            "kappa": kappa,
            "num_nodes": int(problem.nodes.shape[0]),
            "num_triangles": int(problem.triangles.shape[0]),
            "energy": energy,
            "normalized_energy": normalized_value,
            "exact_energy_constant": exact_constant,
            "exact_constant_ratio": normalized_value / exact_constant if exact_constant > 0.0 and kappa < 1.0 else float("nan"),
            "boundary_error_l2_sq": boundary_l2_error_sq(solution, problem),
            "boundary_rate_ratio": (
                boundary_l2_error_sq(solution, problem) / (kappa * abs(math.log(kappa)))
                if kappa < 1.0
                else float("nan")
            ),
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
        if kappa in snapshot_kappas:
            snapshots[kappa] = solution.copy()

    rows.sort(key=lambda item: float(item["kappa"]), reverse=True)
    write_csv(rows, summary_path)
    plot_corrugated_snapshots(problem, snapshots, output_dir / f"{benchmark.file_tag}_snapshots.pdf")
    plot_corrugated_diagnostics(rows, parameters, output_dir / f"{benchmark.file_tag}_diagnostics.pdf")
    metadata = {
        "benchmark": benchmark.name,
        "geometry": benchmark.geometry,
        "physics_variant": "balanced",
        "butler_volmer": {
            "ic0": parameters.ic0,
            "ia0": parameters.ia0,
            "c1": parameters.c1,
            "c2": parameters.c2,
            "a1": parameters.a1,
            "a2": parameters.a2,
        },
        "hmin": hmin,
        "num_nodes": int(problem.nodes.shape[0]),
        "num_triangles": int(problem.triangles.shape[0]),
        "junction_count": cathode_anode_junction_count(benchmark),
        "num_certified_points": int(sum(as_bool(row["certified"]) for row in rows)),
        "height_min": float(np.min(corrugated_top_height(problem.x, parameters))),
        "height_max": float(np.max(corrugated_top_height(problem.x, parameters))),
    }
    write_json(metadata, metadata_path)
    return rows, metadata


def run_mesh_comparison(
    parameters,
    benchmark,
    output_dir,
    kappa_override,
):
    kappas = tuple(kappa_override or (1.0e-3, 1.0e-5))
    sample_points = bulk_sample_points(parameters)
    rows = []
    metadata = {"benchmark": benchmark.name, "cases": []}

    for kappa in kappas:
        reference_problem = build_problem(
            parameters,
            benchmark,
            "graded",
            hmin=max(parameters.suite_hmin, parameters.compare_ref_factor * kappa),
        )
        reference_initial = solve_mixed_reference(reference_problem)
        reference_solution, _ = solve_problem(reference_problem, kappa, reference_initial)

        graded_problem = build_problem(parameters, benchmark, "graded", hmin=max(parameters.suite_hmin, kappa))
        graded_initial = solve_mixed_reference(graded_problem)
        graded_solution, graded_info = solve_problem(graded_problem, kappa, graded_initial)

        uniform_problem = build_problem(
            parameters,
            benchmark,
            "uniform",
            target_nodes=int(graded_problem.nodes.shape[0]),
        )
        uniform_initial = solve_mixed_reference(uniform_problem)
        uniform_solution, uniform_info = solve_problem(uniform_problem, kappa, uniform_initial)

        for strategy, problem, solution, info in (
            ("uniform", uniform_problem, uniform_solution, uniform_info),
            ("graded", graded_problem, graded_solution, graded_info),
        ):
            row = {
                "benchmark": benchmark.name,
                "mesh_strategy": strategy,
                "kappa": kappa,
                "num_nodes": int(problem.nodes.shape[0]),
                "num_triangles": int(problem.triangles.shape[0]),
                "boundary_error_to_reference": boundary_error_to_reference(
                    problem, solution, reference_problem, reference_solution
                ),
                "bulk_error_to_reference": bulk_error_to_reference(
                    problem, solution, reference_problem, reference_solution, sample_points
                ),
                "energy": float(info["energy"]),
                "iterations": float(info["iterations"]),
                "residual_inf": float(info["residual_inf"]),
            }
            rows.append(row)

        metadata["cases"].append(
            {
                "kappa": kappa,
                "reference_nodes": int(reference_problem.nodes.shape[0]),
                "graded_nodes": int(graded_problem.nodes.shape[0]),
                "uniform_nodes": int(uniform_problem.nodes.shape[0]),
            }
        )

    rows.sort(key=lambda item: (float(item["kappa"]), str(item["mesh_strategy"])), reverse=True)
    plot_mesh_comparison(rows, output_dir / "mesh_comparison_single.pdf")
    return rows, metadata


def run_mesh_comparison_refined(
    parameters,
    benchmark,
    output_dir,
    kappa_override,
):
    r"""Three-strategy mesh comparison.

    For every target conductivity, the nonlinear problem is solved on
      (i)   a uniform mesh with the same node count as the proposed graded mesh,
      (ii)  a uniform mesh that has been uniformly refined to four times that
            node count, and
      (iii) the proposed junction-graded mesh with $h_{\min}$ proportional to $\kappa$.
    Errors are measured against an over-resolved graded reference solution
    (with $h_{\min}=0.05\kappa$) computed at the same target conductivity, and
    every nonlinear minimizer is reached through a continuation chain in $\kappa$.
    """
    kappas = tuple(sorted(set(kappa_override or (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)), reverse=True))
    sample_points = bulk_sample_points(parameters)
    constant = exact_log_energy_constant(benchmark, parameters)
    REF_FACTOR = 0.05
    GRADED_FACTOR = 0.25
    UNIFORM_REFINED_MULT = 4

    rows = []
    cases = []

    for kappa in kappas:
        ref_hmin = max(parameters.exact_refinement_cap, REF_FACTOR * kappa)
        ref_problem = build_problem(parameters, benchmark, "graded", hmin=ref_hmin)
        print(
            f"[mesh-compare-refined] kappa={kappa:.1e} reference hmin={ref_hmin:.2e} "
            f"N={ref_problem.nodes.shape[0]}"
        )
        ref_solution, ref_info = solve_with_continuation(ref_problem, kappa)

        graded_hmin = max(parameters.exact_refinement_cap, GRADED_FACTOR * kappa)
        graded_problem = build_problem(parameters, benchmark, "graded", hmin=graded_hmin)
        graded_n = int(graded_problem.nodes.shape[0])
        print(f"[mesh-compare-refined] kappa={kappa:.1e} graded hmin={graded_hmin:.2e} N={graded_n}")
        graded_sol, graded_info = solve_with_continuation(graded_problem, kappa)

        uniform_match_problem = build_problem(parameters, benchmark, "uniform", target_nodes=graded_n)
        print(
            f"[mesh-compare-refined] kappa={kappa:.1e} uniform-matched "
            f"N={uniform_match_problem.nodes.shape[0]}"
        )
        uniform_match_sol, uniform_match_info = solve_with_continuation(
            uniform_match_problem, kappa
        )

        uniform_refined_problem = build_problem(
            parameters,
            benchmark,
            "uniform",
            target_nodes=UNIFORM_REFINED_MULT * graded_n,
        )
        print(
            f"[mesh-compare-refined] kappa={kappa:.1e} uniform-refined "
            f"N={uniform_refined_problem.nodes.shape[0]}"
        )
        uniform_refined_sol, uniform_refined_info = solve_with_continuation(
            uniform_refined_problem, kappa
        )

        ref_n = int(ref_problem.nodes.shape[0])
        ref_energy = float(ref_info["energy"])
        for strategy_label, problem, sol, info in (
            ("uniform_matched", uniform_match_problem, uniform_match_sol, uniform_match_info),
            ("uniform_refined", uniform_refined_problem, uniform_refined_sol, uniform_refined_info),
            ("graded_proposed", graded_problem, graded_sol, graded_info),
        ):
            energy = float(info["energy"])
            normalized = normalized_energy(energy, kappa)
            bdry_err = boundary_error_to_reference(problem, sol, ref_problem, ref_solution)
            bulk_err = bulk_error_to_reference(
                problem, sol, ref_problem, ref_solution, sample_points
            )
            n = int(problem.nodes.shape[0])
            row = {
                "benchmark": benchmark.name,
                "mesh_strategy": strategy_label,
                "kappa": kappa,
                "num_nodes": n,
                "num_triangles": int(problem.triangles.shape[0]),
                "boundary_error_to_reference": bdry_err,
                "boundary_hquarter_to_reference": boundary_hquarter_error_to_reference(
                    problem,
                    sol,
                    ref_problem,
                    ref_solution,
                ),
                "bulk_error_to_reference": bulk_err,
                "boundary_l2_to_phi0_sq": boundary_l2_error_sq(sol, problem),
                "energy": energy,
                "energy_minus_reference": energy - ref_energy,
                "normalized_energy": normalized,
                "exact_constant_ratio": (
                    normalized / constant if constant > 0.0 and kappa < 1.0 else float("nan")
                ),
                "iterations": float(info["iterations"]),
                "residual_inf": float(info["residual_inf"]),
                "certified": is_certified_info(info, parameters),
                "solver": str(info["solver"]),
                "boundary_efficiency": bdry_err * math.sqrt(n),
                "bulk_efficiency": bulk_err * math.sqrt(n),
                "reference_nodes": ref_n,
            }
            rows.append(row)

        cases.append(
            {
                "kappa": kappa,
                "ref_hmin": ref_hmin,
                "ref_nodes": ref_n,
                "ref_energy": ref_energy,
                "graded_hmin": graded_hmin,
                "graded_nodes": graded_n,
                "uniform_matched_nodes": int(uniform_match_problem.nodes.shape[0]),
                "uniform_refined_nodes": int(uniform_refined_problem.nodes.shape[0]),
            }
        )

    rows.sort(key=lambda item: (float(item["kappa"]), str(item["mesh_strategy"])), reverse=True)

    metadata = {
        "benchmark": benchmark.name,
        "ref_factor": REF_FACTOR,
        "graded_factor": GRADED_FACTOR,
        "uniform_refined_mult": UNIFORM_REFINED_MULT,
        "cases": cases,
    }

    plot_mesh_comparison_refined(
        rows, output_dir / f"mesh_comparison_refined_{benchmark.file_tag}.pdf"
    )
    return rows, metadata


def run_exact_constant_refinement(
    parameters,
    benchmark,
    output_dir,
    data_dir,
    kappa_override,
):
    if benchmark.name not in {"single_junction_mixed", "multi_junction_mixed"}:
        raise ValueError("The exact-constant refinement study is implemented for rectangular benchmarks.")

    targets = tuple(kappa_override or (1.0e-4, 1.0e-5, 1.0e-6, 1.3e-7))
    rows = []
    metadata = {
        "benchmark": benchmark.name,
        "mesh_rule": "h_min = max(exact_refinement_cap, exact_refinement_factor * kappa) at the target point",
        "exact_refinement_factor": parameters.exact_refinement_factor,
        "exact_refinement_cap": parameters.exact_refinement_cap,
        "targets": [],
    }

    for target in targets:
        hmin = max(parameters.exact_refinement_cap, parameters.exact_refinement_factor * target)
        problem = build_problem(parameters, benchmark, "graded", hmin=hmin)
        reference = solve_mixed_reference(problem)
        previous = reference.copy()
        levels = sorted(
            {level for level in benchmark.continuation_kappas if level >= target} | {target},
            reverse=True,
        )
        solution = previous
        info = None
        for level in levels:
            print(f"[exact-constant:{benchmark.name}] target={target:.1e}, solve={level:.1e}, hmin={hmin:.1e}")
            solution, info = solve_problem(problem, level, previous)
            previous = solution

        if info is None:
            raise RuntimeError("The continuation loop did not run.")
        energy = float(info["energy"])
        constant = exact_log_energy_constant(benchmark, parameters)
        max_exp_arg = max_butler_volmer_exponent_argument(solution, parameters)
        row = {
            "benchmark": benchmark.name,
            "kappa": target,
            "h_min": hmin,
            "h_min_over_kappa": hmin / target,
            "num_nodes": int(problem.nodes.shape[0]),
            "num_triangles": int(problem.triangles.shape[0]),
            "continuation_levels": len(levels),
            "energy": energy,
            "normalized_energy": normalized_energy(energy, target),
            "exact_energy_constant": constant,
            "exact_constant_ratio": exact_constant_ratio(energy, target, benchmark, parameters),
            "boundary_error_l2_sq": boundary_l2_error_sq(solution, problem),
            "iterations": float(info["iterations"]),
            "residual_inf": float(info["residual_inf"]),
            "solver": str(info["solver"]),
            "certified": is_certified_info(info, parameters),
            "solution_min": float(np.min(solution)),
            "solution_max": float(np.max(solution)),
            "max_bv_exponent_argument": max_exp_arg,
            "exponential_clamp_active": max_exp_arg >= 80.0,
        }
        rows.append(row)
        metadata["targets"].append(
            {
                "kappa": target,
                "h_min": hmin,
                "h_min_over_kappa": hmin / target,
                "num_nodes": int(problem.nodes.shape[0]),
                "num_triangles": int(problem.triangles.shape[0]),
                "continuation_levels": len(levels),
            }
        )

    rows.sort(key=lambda item: float(item["kappa"]), reverse=True)
    if benchmark.name == "single_junction_mixed":
        summary_name = "exact_constant_refinement_summary.csv"
        metadata_name = "exact_constant_refinement_metadata.json"
    else:
        summary_name = f"{benchmark.file_tag}_exact_constant_refinement_summary.csv"
        metadata_name = f"{benchmark.file_tag}_exact_constant_refinement_metadata.json"
    write_csv(rows, data_dir / summary_name)
    write_json(metadata, data_dir / metadata_name)
    plot_exact_constant_refinement(rows, parameters, output_dir / f"exact_constant_refinement_{benchmark.file_tag}.pdf")
    return rows, metadata


def run_spatial_convergence(
    parameters,
    benchmark,
    data_dir,
    kappa_fixed=1.0,
):
    """Spatial convergence study: Error vs h at fixed kappa.

    Solves on a sequence of uniformly refined meshes and computes errors
    against an overresolved graded reference solution.
    """
    # Build overresolved reference
    ref_problem = build_problem(parameters, benchmark, "graded", hmin=parameters.suite_hmin)
    ref_initial = solve_mixed_reference(ref_problem)
    ref_solution, _ = solve_problem(ref_problem, kappa_fixed, ref_initial)
    sample_points = bulk_sample_points(parameters)

    # Sequence of uniform mesh sizes (target node counts)
    target_nodes_list = [45, 153, 435, 1035, 2701, 5151]
    rows = []

    for target in target_nodes_list:
        problem = build_problem(parameters, benchmark, "uniform", target_nodes=target)
        actual_nodes = int(problem.nodes.shape[0])
        hmax = float(np.max(np.diff(problem.x)))
        initial = solve_mixed_reference(problem)
        u, info = solve_problem(problem, kappa_fixed, initial)

        bdry_err = boundary_error_to_reference(problem, u, ref_problem, ref_solution)
        bulk_err = bulk_error_to_reference(problem, u, ref_problem, ref_solution, sample_points)

        row = {
            "num_nodes": actual_nodes,
            "h_max": hmax,
            "boundary_error": bdry_err,
            "bulk_error": bulk_err,
            "energy": float(info["energy"]),
            "iterations": float(info["iterations"]),
            "residual_inf": float(info["residual_inf"]),
        }
        rows.append(row)
        print(f"[spatial:{benchmark.name}] N={actual_nodes}, h={hmax:.4e}, "
              f"bdry_err={bdry_err:.4e}, bulk_err={bulk_err:.4e}")

    # Compute convergence rates
    for i in range(1, len(rows)):
        h_prev = float(rows[i - 1]["h_max"])
        h_curr = float(rows[i]["h_max"])
        if h_prev > 0 and h_curr > 0 and h_prev != h_curr:
            log_ratio = math.log(h_curr / h_prev)
            bdry_prev = float(rows[i - 1]["boundary_error"])
            bdry_curr = float(rows[i]["boundary_error"])
            bulk_prev = float(rows[i - 1]["bulk_error"])
            bulk_curr = float(rows[i]["bulk_error"])
            if bdry_prev > 0 and bdry_curr > 0:
                rows[i]["boundary_rate"] = math.log(bdry_curr / bdry_prev) / log_ratio
            else:
                rows[i]["boundary_rate"] = float("nan")
            if bulk_prev > 0 and bulk_curr > 0:
                rows[i]["bulk_rate"] = math.log(bulk_curr / bulk_prev) / log_ratio
            else:
                rows[i]["bulk_rate"] = float("nan")
        else:
            rows[i]["boundary_rate"] = float("nan")
            rows[i]["bulk_rate"] = float("nan")
    rows[0]["boundary_rate"] = float("nan")
    rows[0]["bulk_rate"] = float("nan")

    # Write CSV
    write_csv(rows, data_dir / "spatial_convergence_summary.csv")

    # Write metadata
    meta = {
        "benchmark": benchmark.name,
        "kappa_fixed": kappa_fixed,
        "reference_nodes": int(ref_problem.nodes.shape[0]),
        "reference_triangles": int(ref_problem.triangles.shape[0]),
        "reference_hmin": float(parameters.suite_hmin),
        "reference_growth": float(parameters.mesh_growth_x),
    }
    write_json(meta, data_dir / "spatial_convergence_metadata.json")

    print(f"[done] spatial convergence data written to {data_dir}")
    return rows
