"""Read and write numerical fields, tables, and experiment metadata."""

import csv
import json
import math

import numpy as np

from .model import format_kappa_tag


def save_solution(problem, kappa, u, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = format_kappa_tag(kappa)
    parameters = problem.physics
    np.savez_compressed(
        out_dir / f"solution_{problem.benchmark.name}_{problem.mesh_strategy}_{tag}.npz",
        benchmark=problem.benchmark.name,
        mesh_strategy=problem.mesh_strategy,
        kappa=kappa,
        phi_c=parameters.phi_c,
        phi_a=parameters.phi_a,
        ic0=parameters.ic0,
        ia0=parameters.ia0,
        c1=parameters.c1,
        c2=parameters.c2,
        a1=parameters.a1,
        a2=parameters.a2,
        x=problem.x,
        y=problem.y,
        nodes=problem.nodes,
        triangles=problem.triangles,
        potential=u,
    )


def scalar_archive_value(archive, key):
    value = archive[key]
    if np.asarray(value).shape == ():
        return value.item()
    return value


def load_saved_solution_from_path(problem, kappa, path):
    if not path.exists():
        return None
    with np.load(path) as archive:
        for key, expected in (
            ("benchmark", problem.benchmark.name),
            ("mesh_strategy", problem.mesh_strategy),
        ):
            if key not in archive or str(scalar_archive_value(archive, key)) != expected:
                return None
        if "kappa" not in archive or not math.isclose(
            float(scalar_archive_value(archive, "kappa")),
            kappa,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            return None
        nodes = archive["nodes"]
        if nodes.shape != problem.nodes.shape or not np.allclose(nodes, problem.nodes):
            return None
        parameters = problem.physics
        physics_keys = ("phi_c", "phi_a", "ic0", "ia0", "c1", "c2", "a1", "a2")
        physics_key_present = [key in archive for key in physics_keys]
        if any(physics_key_present) and not all(physics_key_present):
            return None
        physics_keys_present = all(physics_key_present)
        for key, expected in (
            ("phi_c", parameters.phi_c),
            ("phi_a", parameters.phi_a),
            ("ic0", parameters.ic0),
            ("ia0", parameters.ia0),
            ("c1", parameters.c1),
            ("c2", parameters.c2),
            ("a1", parameters.a1),
            ("a2", parameters.a2),
        ):
            if physics_keys_present and not np.isclose(float(archive[key]), expected):
                return None
        return np.asarray(archive["potential"], dtype=float)


def load_saved_solution(problem, kappa, data_dir):
    primary = data_dir / f"solution_{problem.benchmark.name}_{problem.mesh_strategy}_{format_kappa_tag(kappa)}.npz"
    solution = load_saved_solution_from_path(problem, kappa, primary)
    if solution is not None:
        return solution
    pattern = f"solution_{problem.benchmark.name}_{problem.mesh_strategy}_*.npz"
    for candidate in sorted(data_dir.glob(pattern)):
        if candidate == primary:
            continue
        solution = load_saved_solution_from_path(problem, kappa, candidate)
        if solution is not None:
            return solution
    return None


def save_reference(problem, u_ref, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if problem.mesh_strategy == "graded":
        filename = f"u0_{problem.benchmark.name}.npz"
    else:
        filename = f"u0_{problem.benchmark.name}_{problem.mesh_strategy}.npz"
    np.savez_compressed(
        out_dir / filename,
        benchmark=problem.benchmark.name,
        mesh_strategy=problem.mesh_strategy,
        x=problem.x,
        y=problem.y,
        nodes=problem.nodes,
        triangles=problem.triangles,
        potential=u_ref,
    )
    return filename


def write_csv(rows, outfile):
    if not rows:
        return
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with outfile.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(infile):
    with infile.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(data, outfile):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
