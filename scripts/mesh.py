"""Structured finite-element meshes and the mixed harmonic reference problem."""

import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .model import (
    Problem,
    benchmark_junction_points,
    corrugated_top_height,
    interval_label_for_point,
    reactive_value,
)


def deduplicate_sorted(points, tol=1.0e-12):
    sorted_points = sorted(float(p) for p in points)
    unique = []
    for point in sorted_points:
        if not unique or abs(point - unique[-1]) > tol * max(1.0, abs(point), abs(unique[-1])):
            unique.append(point)
    return np.array(unique, dtype=float)


def graded_coords(length, hmin, growth):
    coords = [0.0]
    step = hmin
    while coords[-1] + step < length:
        coords.append(coords[-1] + step)
        step *= growth
    if coords[-1] < length:
        coords.append(length)
    return np.array(coords, dtype=float)


def graded_coords_from_anchors(length, anchors, hmin, growth):
    if not anchors:
        return graded_coords(length, hmin, growth)

    points = [0.0, length]
    for anchor in anchors:
        points.append(anchor)
        distance = 0.0
        step = hmin
        while True:
            distance += step
            added = False
            left = anchor - distance
            right = anchor + distance
            if left > 0.0:
                points.append(left)
                added = True
            if right < length:
                points.append(right)
                added = True
            if not added:
                break
            step *= growth
    return deduplicate_sorted(points)


def uniform_counts_for_target_nodes(target_nodes):
    best_nx = 17
    best_ny = 9
    best_diff = math.inf
    for q in range(2, 2000):
        nseg_x = 4 * q
        nseg_y = 2 * q
        nodes = (nseg_x + 1) * (nseg_y + 1)
        diff = abs(nodes - target_nodes)
        if diff < best_diff:
            best_diff = diff
            best_nx = nseg_x + 1
            best_ny = nseg_y + 1
        if nodes > max(target_nodes * 1.5, target_nodes + 1000) and best_diff < math.inf:
            break
    return best_nx, best_ny


def triangle_stiffness(coords):
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * abs(det)
    b = np.array([y1 - y2, y2 - y0, y0 - y1])
    c = np.array([x2 - x1, x0 - x2, x1 - x0])
    return (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)


def physical_nodes_from_grid(
    xx,
    yy,
    parameters,
    benchmark,
):
    if benchmark.geometry == "rectangle":
        return np.column_stack([xx.ravel(), yy.ravel()])
    if benchmark.geometry == "corrugated_top":
        height = corrugated_top_height(xx, parameters)
        mapped_y = yy * height / parameters.ly
        return np.column_stack([xx.ravel(), mapped_y.ravel()])
    raise ValueError(f"Unknown geometry: {benchmark.geometry}")


def build_problem(
    parameters,
    benchmark,
    mesh_strategy,
    hmin=None,
    target_nodes=None,
):
    if mesh_strategy == "graded":
        step = parameters.suite_hmin if hmin is None else hmin
        x = graded_coords_from_anchors(parameters.lx, benchmark_junction_points(benchmark), step, parameters.mesh_growth_x)
        y = graded_coords(parameters.ly, step, parameters.mesh_growth_y)
    elif mesh_strategy == "uniform":
        if target_nodes is None:
            target_nodes = 5000
        nx, ny = uniform_counts_for_target_nodes(target_nodes)
        x = np.linspace(0.0, parameters.lx, nx)
        y = np.linspace(0.0, parameters.ly, ny)
    else:
        raise ValueError(f"Unknown mesh strategy: {mesh_strategy}")

    nx = x.size
    ny = y.size
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = physical_nodes_from_grid(xx, yy, parameters, benchmark)

    triangles = []
    rows = []
    cols = []
    data = []

    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = j * nx + i + 1
            n01 = (j + 1) * nx + i
            n11 = (j + 1) * nx + i + 1
            cell_tris = ([n00, n10, n11], [n00, n11, n01])
            for tri in cell_tris:
                tri_arr = np.array(tri, dtype=int)
                triangles.append(tri)
                local = triangle_stiffness(nodes[tri_arr])
                for a in range(3):
                    for b in range(3):
                        rows.append(tri[a])
                        cols.append(tri[b])
                        data.append(local[a, b])

    stiffness = sparse.coo_matrix((data, (rows, cols)), shape=(nodes.shape[0], nodes.shape[0])).tocsr()

    bottom_nodes = np.arange(nx, dtype=int)
    bottom_edges = np.column_stack([bottom_nodes[:-1], bottom_nodes[1:]])
    bottom_lengths = np.linalg.norm(nodes[bottom_edges[:, 1]] - nodes[bottom_edges[:, 0]], axis=1)
    midpoints = 0.5 * (x[:-1] + x[1:])

    edge_labels = np.array(
        [interval_label_for_point(midpoint, benchmark.intervals) for midpoint in midpoints],
        dtype="<U1",
    )
    bottom_edge_labels = np.where(edge_labels == "c", 0, 1).astype(int)
    bottom_edge_values = np.array([reactive_value(label, parameters) for label in edge_labels], dtype=float)

    bottom_node_values = np.empty_like(x)
    bottom_node_values[0] = bottom_edge_values[0]
    bottom_node_values[-1] = bottom_edge_values[-1]
    if x.size > 2:
        bottom_node_values[1:-1] = 0.5 * (bottom_edge_values[:-1] + bottom_edge_values[1:])

    return Problem(
        benchmark=benchmark,
        physics=parameters,
        mesh_strategy=mesh_strategy,
        x=x,
        y=y,
        nodes=nodes,
        triangles=np.array(triangles, dtype=int),
        stiffness=stiffness,
        bottom_nodes=bottom_nodes,
        bottom_edges=bottom_edges,
        bottom_lengths=bottom_lengths,
        bottom_edge_labels=bottom_edge_labels,
        bottom_edge_values=bottom_edge_values,
        bottom_node_values=bottom_node_values,
    )


def solve_mixed_reference(problem):
    fixed = problem.bottom_nodes
    fixed_values = problem.bottom_node_values
    n = problem.nodes.shape[0]
    free_mask = np.ones(n, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)

    u = np.zeros(n, dtype=float)
    u[fixed] = fixed_values

    kff = problem.stiffness[free][:, free]
    kfc = problem.stiffness[free][:, fixed]
    rhs = -kfc @ fixed_values
    u[free] = spsolve(kff, rhs)
    return u
