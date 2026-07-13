"""Run one experiment or reproduce all paper figures."""

import argparse

from . import postprocess
from . import solve_experiments


def experiment_groups(figures, data):
    """Return the solver options needed for each experiment group."""
    common = ["--output-dir", figures, "--data-dir", data]

    rectangular = (
        "rectangular convergence experiments",
        solve_experiments.main,
        ["--benchmark", "all", "--mesh-strategy", "graded"] + common,
    )
    exact_single = (
        "single-junction logarithmic constant",
        solve_experiments.main,
        ["--benchmark", "single_junction_mixed", "--exact-constant-refinement"] + common,
    )
    exact_multi = (
        "multi-junction logarithmic constant",
        solve_experiments.main,
        [
            "--benchmark",
            "multi_junction_mixed",
            "--exact-constant-refinement",
            "--kappa-list",
            "1e-4,1e-5,1e-6,1.3e-7",
        ]
        + common,
    )
    corrugated = (
        "corrugated-boundary experiments",
        solve_experiments.main,
        ["--corrugated-stress-test"] + common,
    )
    three_dimensional = (
        "three-dimensional solution view",
        solve_experiments.main,
        ["--flashy-experiments"] + common,
    )
    final_figures = (
        "comparison tables and final figures",
        postprocess.main,
        [
            "--run-missing-newton",
            "--mesh-rate-two-axis",
            "--practical-parameter-experiments",
        ]
        + common,
    )
    short_check = (
        "two-value single-junction check",
        solve_experiments.main,
        [
            "--benchmark",
            "single_junction_mixed",
            "--mesh-strategy",
            "graded",
            "--kappa-list",
            "1e-2,1e-4",
        ]
        + common,
    )

    complete_run = [
        rectangular,
        exact_single,
        exact_multi,
        corrugated,
        three_dimensional,
        final_figures,
    ]
    return {
        "rectangular": [rectangular],
        "exact-single": [exact_single],
        "exact-multi": [exact_multi],
        "corrugated": [corrugated],
        "flashy": [three_dimensional],
        "postprocess": [final_figures],
        "paper-figures": complete_run,
        "all": complete_run,
        "short-check": [short_check],
        "smoke": [short_check],
    }


def reproduce(
    target="paper-figures",
    figures="paper_figures/generated",
    data="data/computed",
    dry_run=False,
):
    groups = experiment_groups(figures, data)
    if target not in groups:
        raise ValueError("Unknown experiment group: " + target)

    steps = groups[target]
    for number, (label, run, options) in enumerate(steps, start=1):
        print(f"[{number}/{len(steps)}] {label}")
        if not dry_run:
            run(options)


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=tuple(experiment_groups("paper_figures/generated", "data/computed")),
        help="Experiment group to run.",
    )
    parser.add_argument(
        "--figures",
        default="paper_figures/generated",
        help="Folder for newly computed figures.",
    )
    parser.add_argument(
        "--data",
        default="data/computed",
        help="Folder for newly computed tables and finite-element fields.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List stages without running them.")
    return parser.parse_args(arguments)


def main(arguments=None):
    args = parse_args(arguments)
    reproduce(args.target, args.figures, args.data, args.dry_run)


if __name__ == "__main__":
    main()
