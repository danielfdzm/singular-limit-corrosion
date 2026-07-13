"""Physical parameters, benchmark geometries, and Butler--Volmer laws."""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

# These classes simply keep related parameters and arrays together.

@dataclass
class Parameters:
    lx: float = 0.02
    ly: float = 0.01
    phi_c: float = 0.2
    phi_a: float = -0.2
    ic0: float = 3.0e-4
    ia0: float = 3.0e-2
    c1: float = 83.13
    c2: float = 35.63
    a1: float = 35.63
    a2: float = 83.13
    suite_hmin: float = 5.0e-7
    single_suite_hmin: float = 2.0e-8
    mesh_growth_x: float = 1.18
    mesh_growth_y: float = 1.18
    newton_tol: float = 1.0e-10
    newton_step_tol: float = 1.0e-12
    newton_maxit: int = 50
    certified_residual_factor: float = 10.0
    line_search_min: float = 2.0 ** -20
    line_search_c1: float = 1.0e-4
    compare_ref_factor: float = 0.25
    exact_refinement_factor: float = 0.05
    exact_refinement_cap: float = 1.0e-8
    bulk_box_x0: float = 0.25
    bulk_box_x1: float = 0.75
    bulk_box_y0: float = 0.25
    bulk_box_y1: float = 0.75


@dataclass
class Benchmark:
    name: str
    file_tag: str
    title: str
    intervals: tuple[tuple[float, float, str], ...]
    snapshot_kappas: tuple[float, ...]
    trace_kappas: tuple[float, ...]
    continuation_kappas: tuple[float, ...]
    convergence_kappas: tuple[float, ...]
    geometry: str = "rectangle"


@dataclass
class Problem:
    benchmark: Benchmark
    physics: Parameters
    mesh_strategy: str
    x: np.ndarray
    y: np.ndarray
    nodes: np.ndarray
    triangles: np.ndarray
    stiffness: sparse.csr_matrix
    bottom_nodes: np.ndarray
    bottom_edges: np.ndarray
    bottom_lengths: np.ndarray
    bottom_edge_labels: np.ndarray
    bottom_edge_values: np.ndarray
    bottom_node_values: np.ndarray


def build_benchmarks(parameters):
    lx = parameters.lx
    corrugated_four = Benchmark(
        name="corrugated_four_junction_mixed",
        file_tag="corrugated_four_junction",
        title="Corrugated four-junction mixed benchmark",
        intervals=(
            (0.00 * lx, 0.18 * lx, "c"),
            (0.18 * lx, 0.39 * lx, "a"),
            (0.39 * lx, 0.60 * lx, "c"),
            (0.60 * lx, 0.81 * lx, "a"),
            (0.81 * lx, 1.00 * lx, "c"),
        ),
        snapshot_kappas=(1.0e-2, 1.0e-4, 1.0e-5, 1.0e-7),
        trace_kappas=(),
        continuation_kappas=(
            1.0, 1.0e-1, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4,
            1.0e-4, 5.0e-5, 3.0e-5, 1.0e-5, 5.0e-6, 3.0e-6,
            1.0e-6, 4.0e-7, 3.0e-7, 2.5e-7, 2.0e-7, 1.8e-7,
            1.6e-7, 1.5e-7, 1.4e-7, 1.3e-7, 1.2e-7, 1.1e-7,
            1.0e-7,
        ),
        convergence_kappas=(
            1.0e-2, 1.0e-4, 5.0e-5, 3.0e-5, 1.0e-5, 5.0e-6,
            1.0e-6, 4.0e-7, 3.0e-7, 2.5e-7, 2.0e-7, 1.8e-7,
            1.6e-7, 1.5e-7, 1.3e-7,
        ),
        geometry="corrugated_top",
    )
    corrugated_six = Benchmark(
        name="corrugated_six_junction_mixed",
        file_tag="corrugated_six_junction",
        title="Corrugated six-junction mixed benchmark",
        intervals=(
            (0.00 * lx, 0.12 * lx, "c"),
            (0.12 * lx, 0.27 * lx, "a"),
            (0.27 * lx, 0.40 * lx, "c"),
            (0.40 * lx, 0.55 * lx, "a"),
            (0.55 * lx, 0.70 * lx, "c"),
            (0.70 * lx, 0.84 * lx, "a"),
            (0.84 * lx, 1.00 * lx, "c"),
        ),
        snapshot_kappas=(1.0e-2, 1.0e-4, 1.0e-5, 1.0e-7),
        trace_kappas=(),
        continuation_kappas=(
            1.0, 1.0e-1, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4,
            1.0e-4, 5.0e-5, 3.0e-5, 1.0e-5, 5.0e-6, 3.0e-6,
            1.0e-6, 4.0e-7, 3.0e-7, 2.5e-7, 2.0e-7, 1.8e-7,
            1.6e-7, 1.5e-7, 1.4e-7, 1.3e-7, 1.2e-7, 1.1e-7,
            1.0e-7,
        ),
        convergence_kappas=(
            1.0e-2, 1.0e-4, 5.0e-5, 3.0e-5, 1.0e-5, 5.0e-6,
            1.0e-6, 4.0e-7, 3.0e-7, 2.5e-7, 2.0e-7, 1.8e-7,
            1.6e-7, 1.5e-7, 1.3e-7,
        ),
        geometry="corrugated_top",
    )
    benchmarks = {
        "single_junction_mixed": Benchmark(
            name="single_junction_mixed",
            file_tag="single",
            title="Single-junction mixed benchmark",
            intervals=(
                (0.0, 0.5 * lx, "c"),
                (0.5 * lx, lx, "a"),
            ),
            snapshot_kappas=(1.0, 1.0e-1, 1.0e-2, 1.0e-4, 1.0e-5, 2.0e-7),
            trace_kappas=(1.0, 1.0e-1, 1.0e-2, 1.0e-4, 1.0e-5, 2.0e-7),
            continuation_kappas=(
                1.0, 1.0e-1, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4,
                5.0e-5, 3.0e-5, 1.0e-5, 5.0e-6, 3.0e-6, 1.0e-6,
                4.0e-7, 3.0e-7, 2.5e-7, 2.0e-7, 1.8e-7, 1.6e-7,
                1.5e-7, 1.4e-7, 1.3e-7, 1.2e-7, 1.1e-7, 1.0e-7,
            ),
            convergence_kappas=(
                1.0e-2, 1.0e-4, 5.0e-5, 3.0e-5, 1.0e-5, 5.0e-6,
                1.0e-6, 4.0e-7, 3.0e-7, 2.5e-7, 2.0e-7, 1.8e-7,
                1.6e-7, 1.5e-7, 1.3e-7,
            ),
        ),
        "multi_junction_mixed": Benchmark(
            name="multi_junction_mixed",
            file_tag="multi",
            title="Multi-junction mixed benchmark",
            intervals=(
                (0.0, 0.25 * lx, "c"),
                (0.25 * lx, 0.50 * lx, "a"),
                (0.50 * lx, 0.75 * lx, "c"),
                (0.75 * lx, lx, "a"),
            ),
            snapshot_kappas=(),
            trace_kappas=(),
            continuation_kappas=(
                1.0, 1.0e-1, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4,
                5.0e-5, 1.0e-5, 5.0e-6, 1.0e-6,
                5.0e-7, 1.0e-7, 5.0e-8, 1.0e-8,
            ),
            convergence_kappas=(
                1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8,
            ),
        ),
    }
    benchmarks["corrugated_four_junction_mixed"] = corrugated_four
    benchmarks["corrugated_six_junction_mixed"] = corrugated_six
    return benchmarks


def corrugated_stress_physics(parameters):
    """Use balanced kinetics for the corrugated four-junction stress test."""
    exponent = 0.5 * (parameters.c1 + parameters.c2)
    exchange_current = math.sqrt(parameters.ic0 * parameters.ia0)
    return replace(
        parameters,
        ic0=exchange_current,
        ia0=exchange_current,
        c1=exponent,
        c2=exponent,
        a1=exponent,
        a2=exponent,
    )


def seawater_dcc_physics(parameters):
    """Parameters from a seawater differential-concentration corrosion model.

    The source reports iron and oxygen equilibrium potentials, exchange current
    densities, and natural-log Tafel denominators. The common exchange-current
    magnitude is absorbed into the nondimensional conductivity, so only the
    reported exchange-current ratio is retained in ``ia0/ic0``.
    """
    iron_exchange = 7.7e-7
    oxygen_exchange = 7.1e-5
    return replace(
        parameters,
        phi_a=-0.76,
        phi_c=0.189,
        ia0=iron_exchange / oxygen_exchange,
        ic0=1.0,
        a1=1.0 / 0.41,
        a2=1.0 / 0.41,
        c1=1.0 / 0.18,
        c2=1.0 / 0.18,
    )


def stainless_zro2_physics(parameters):
    """Parameters fitted for one-layer ZrO2-coated stainless steel in NaCl.

    The source reports Tafel slopes in volts per decade. We convert them to
    exponential coefficients using ln(10)/beta and retain the reported
    exchange-current ratio after absorbing the common scale into kappa.
    """
    anodic_exchange = 1.637e-8
    cathodic_exchange = 1.61e-8
    return replace(
        parameters,
        phi_a=-0.58,
        phi_c=-0.07,
        ia0=anodic_exchange / cathodic_exchange,
        ic0=1.0,
        a1=math.log(10.0) / 0.243,
        a2=math.log(10.0) / 0.243,
        c1=math.log(10.0) / 0.181,
        c2=math.log(10.0) / 0.181,
    )


def safe_exp(x):
    return np.exp(np.clip(x, -80.0, 80.0))


def i_c(phi, parameters):
    return parameters.ic0 * (
        safe_exp(parameters.c1 * (phi - parameters.phi_c)) - safe_exp(-parameters.c2 * (phi - parameters.phi_c))
    )


def i_a(phi, parameters):
    return parameters.ia0 * (
        safe_exp(parameters.a2 * (phi - parameters.phi_a)) - safe_exp(-parameters.a1 * (phi - parameters.phi_a))
    )


def i_c_prime(phi, parameters):
    return parameters.ic0 * (
        parameters.c1 * safe_exp(parameters.c1 * (phi - parameters.phi_c))
        + parameters.c2 * safe_exp(-parameters.c2 * (phi - parameters.phi_c))
    )


def i_a_prime(phi, parameters):
    return parameters.ia0 * (
        parameters.a2 * safe_exp(parameters.a2 * (phi - parameters.phi_a))
        + parameters.a1 * safe_exp(-parameters.a1 * (phi - parameters.phi_a))
    )


def i_c_primitive(phi, parameters):
    dc = parameters.ic0 / parameters.c1 + parameters.ic0 / parameters.c2
    return (
        parameters.ic0 / parameters.c1 * safe_exp(parameters.c1 * (phi - parameters.phi_c))
        + parameters.ic0 / parameters.c2 * safe_exp(-parameters.c2 * (phi - parameters.phi_c))
        - dc
    )


def i_a_primitive(phi, parameters):
    da = parameters.ia0 / parameters.a1 + parameters.ia0 / parameters.a2
    return (
        parameters.ia0 / parameters.a1 * safe_exp(-parameters.a1 * (phi - parameters.phi_a))
        + parameters.ia0 / parameters.a2 * safe_exp(parameters.a2 * (phi - parameters.phi_a))
        - da
    )


def format_kappa_tag(kappa):
    if kappa >= 1.0e-3:
        text = f"{kappa:.3f}".rstrip("0").rstrip(".")
    else:
        mantissa, exponent = f"{kappa:.3e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        text = f"{mantissa}e{int(exponent):+03d}"
    return text.replace("-", "m").replace(".", "_").replace("+", "")


def reactive_value(label, parameters):
    return parameters.phi_c if label == "c" else parameters.phi_a


def benchmark_junction_points(benchmark):
    points = []
    intervals = benchmark.intervals
    for idx in range(1, len(intervals)):
        left = intervals[idx - 1]
        right = intervals[idx]
        if left[2] != right[2]:
            points.append(left[1])
    return points


def interval_label_for_point(xval, intervals):
    tol = 1.0e-12
    for idx, (start, end, label) in enumerate(intervals):
        if xval < start - tol:
            continue
        if xval < end - tol:
            return label
        if idx == len(intervals) - 1 and xval <= end + tol:
            return label
    raise ValueError(f"Point x={xval} does not belong to any interval.")


def corrugated_top_height(x, parameters):
    return parameters.ly * (
        1.0
        + 0.22 * np.sin(2.0 * np.pi * x / parameters.lx)
        + 0.10 * np.sin(4.0 * np.pi * x / parameters.lx + 0.5)
    )
