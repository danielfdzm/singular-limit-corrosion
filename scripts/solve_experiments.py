"""Run the paper's numerical experiments from the command line."""

import argparse
from pathlib import Path

from .diagnostics import certified_residual_threshold
from .experiments import (
    postprocess_convergence_csv,
    run_benchmark_suite,
    run_corrugated_stress_test,
    run_exact_constant_refinement,
    run_flashy_experiments,
    run_mesh_comparison,
    run_mesh_comparison_refined,
    run_spatial_convergence,
)
from .model import Parameters, build_benchmarks, corrugated_stress_physics
from .storage import write_csv, write_json


def parse_kappa_list(text):
    cleaned = text.strip()
    if not cleaned:
        return None
    values = tuple(float(chunk) for chunk in cleaned.split(",") if chunk.strip())
    if not values:
        return None
    return tuple(sorted(set(values), reverse=True))


def parse_args(argv=None):
    """Parse command-line options, or an explicit list supplied by a notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=[
            "single_junction_mixed",
            "multi_junction_mixed",
            "corrugated_four_junction_mixed",
            "corrugated_six_junction_mixed",
            "all",
        ],
        default="all",
        help="Benchmark to run.",
    )
    parser.add_argument(
        "--mesh-strategy",
        choices=["graded", "uniform", "compare"],
        default="graded",
        help="Mesh strategy for the main suite, or `compare` for the single-junction mesh study.",
    )
    parser.add_argument(
        "--mesh-comparison-refined",
        action="store_true",
        help=(
            "Run the three-strategy mesh comparison: uniform-matched, uniform-refined "
            "(4x nodes), and the proposed junction-graded mesh, all measured against "
            "an over-resolved graded reference."
        ),
    )
    parser.add_argument(
        "--spatial-convergence",
        action="store_true",
        help="Run spatial convergence study (Error vs h at fixed kappa).",
    )
    parser.add_argument(
        "--postprocess-csv",
        action="store_true",
        help="Refresh theorem-facing plots and exact-constant diagnostics from convergence_summary.csv.",
    )
    parser.add_argument(
        "--exact-constant-refinement",
        action="store_true",
        help="Run the rectangular proportional refinement path for the exact logarithmic constant.",
    )
    parser.add_argument(
        "--corrugated-stress-test",
        action="store_true",
        help="Run the corrugated multi-junction stress test.",
    )
    parser.add_argument(
        "--flashy-experiments",
        action="store_true",
        help="Generate high-impact visual diagnostics from saved certified solutions.",
    )
    parser.add_argument(
        "--kappa-list",
        default="",
        help="Optional comma-separated list of kappa values overriding the defaults.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/generated",
        help="Directory where the PDF plots will be written.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/computed",
        help="Directory where CSV/JSON/NPZ outputs will be written.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the selected studies; ``argv`` makes this callable from a notebook."""
    args = parse_args(argv)
    parameters = Parameters()
    benchmarks = build_benchmarks(parameters)
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    kappa_override = parse_kappa_list(args.kappa_list)
    selected = (
        [benchmark for benchmark in benchmarks.values() if benchmark.geometry == "rectangle"]
        if args.benchmark == "all"
        else [benchmarks[args.benchmark]]
    )

    if args.flashy_experiments:
        metadata = run_flashy_experiments(parameters, benchmarks, output_dir, data_dir)
        print(f"[done] flashy experiment kappas: {metadata['kappas']}")
        print(f"[done] plots written to {output_dir}")
        print(f"[done] data written to {data_dir}")
        return

    if args.corrugated_stress_test or args.benchmark in {
        "corrugated_four_junction_mixed",
        "corrugated_six_junction_mixed",
    }:
        stress_parameters = corrugated_stress_physics(parameters)
        stress_benchmarks = build_benchmarks(stress_parameters)
        if args.benchmark in {"corrugated_four_junction_mixed", "corrugated_six_junction_mixed"}:
            selected_stress = [stress_benchmarks[args.benchmark]]
        else:
            selected_stress = [
                stress_benchmarks["corrugated_four_junction_mixed"],
                stress_benchmarks["corrugated_six_junction_mixed"],
            ]
        total_rows = 0
        for benchmark in selected_stress:
            rows, metadata = run_corrugated_stress_test(
                stress_parameters,
                benchmark,
                output_dir,
                data_dir,
                kappa_override,
            )
            total_rows += len(rows)
            print(
                f"[done] {metadata['benchmark']} rows: {len(rows)}; "
                f"certified: {metadata['num_certified_points']}"
            )
        print(f"[done] corrugated stress-test rows: {total_rows}")
        print(f"[done] plots written to {output_dir}")
        print(f"[done] data written to {data_dir}")
        return

    if args.postprocess_csv:
        postprocess_convergence_csv(parameters, benchmarks, selected, args.mesh_strategy, output_dir, data_dir)
        print(f"[done] refreshed plots from {data_dir / 'convergence_summary.csv'}")
        return

    if args.exact_constant_refinement:
        benchmark = benchmarks.get(
            args.benchmark if args.benchmark != "all" else "single_junction_mixed"
        )
        rows, metadata = run_exact_constant_refinement(parameters, benchmark, output_dir, data_dir, kappa_override)
        print(f"[done] exact-constant refinement rows: {len(rows)}")
        print(f"[done] plots written to {output_dir}")
        print(f"[done] data written to {data_dir}")
        return

    if args.spatial_convergence:
        benchmark = benchmarks.get(
            args.benchmark if args.benchmark != "all" else "single_junction_mixed"
        )
        run_spatial_convergence(parameters, benchmark, data_dir)
        return

    if args.mesh_strategy == "compare":
        if args.benchmark == "all":
            raise ValueError("The mesh comparison study is implemented only for a single benchmark.")
        benchmark = benchmarks[args.benchmark]
        rows, metadata = run_mesh_comparison(parameters, benchmark, output_dir, kappa_override)
        write_csv(rows, data_dir / "mesh_comparison_summary.csv")
        write_json(metadata, data_dir / "mesh_comparison_metadata.json")
        print(f"[done] plots written to {output_dir}")
        print(f"[done] data written to {data_dir}")
        return

    if args.mesh_comparison_refined:
        benchmark = benchmarks[args.benchmark if args.benchmark != "all" else "single_junction_mixed"]
        rows, metadata = run_mesh_comparison_refined(
            parameters, benchmark, output_dir, kappa_override
        )
        write_csv(rows, data_dir / f"mesh_comparison_refined_{benchmark.file_tag}_summary.csv")
        write_json(
            metadata, data_dir / f"mesh_comparison_refined_{benchmark.file_tag}_metadata.json"
        )
        print(f"[done] refined-mesh comparison rows: {len(rows)}")
        print(f"[done] plots written to {output_dir}")
        print(f"[done] data written to {data_dir}")
        return

    all_rows = []
    metadata = {
        "domain": {"lx": parameters.lx, "ly": parameters.ly},
        "equilibrium_potentials": {"phi_c": parameters.phi_c, "phi_a": parameters.phi_a},
        "butler_volmer": {
            "ic0": parameters.ic0,
            "ia0": parameters.ia0,
            "c1": parameters.c1,
            "c2": parameters.c2,
            "a1": parameters.a1,
            "a2": parameters.a2,
        },
        "mesh_strategy": args.mesh_strategy,
        "certification_threshold": certified_residual_threshold(parameters),
        "benchmarks": {},
    }

    for benchmark in selected:
        rows, bench_meta = run_benchmark_suite(
            parameters,
            benchmark,
            args.mesh_strategy,
            output_dir,
            data_dir,
            kappa_override,
        )
        all_rows.extend(rows)
        metadata["benchmarks"][benchmark.name] = bench_meta

    write_csv(all_rows, data_dir / "convergence_summary.csv")
    write_json(metadata, data_dir / "experiment_metadata.json")
    print(f"[done] plots written to {output_dir}")
    print(f"[done] data written to {data_dir}")


if __name__ == "__main__":
    main()
