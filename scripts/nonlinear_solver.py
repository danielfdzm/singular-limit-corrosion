"""Energy assembly and nonlinear Newton/L-BFGS solution algorithms."""

import math

import numpy as np
from scipy import optimize, sparse
from scipy.sparse.linalg import spsolve

from .mesh import solve_mixed_reference
from .diagnostics import (
    GAUSS_W,
    GAUSS_XI,
    certified_residual_threshold,
    is_certified_info,
)
from .model import (
    i_a,
    i_a_prime,
    i_a_primitive,
    i_c,
    i_c_prime,
    i_c_primitive,
)


def edge_hessian_entries(
    n0,
    n1,
    coeff,
    nshape,
    rows,
    cols,
    data,
):
    c00 = coeff * nshape[0] * nshape[0]
    c01 = coeff * nshape[0] * nshape[1]
    c11 = coeff * nshape[1] * nshape[1]

    rows.extend(n0.tolist())
    cols.extend(n0.tolist())
    data.extend(c00.tolist())

    rows.extend(n0.tolist())
    cols.extend(n1.tolist())
    data.extend(c01.tolist())

    rows.extend(n1.tolist())
    cols.extend(n0.tolist())
    data.extend(c01.tolist())

    rows.extend(n1.tolist())
    cols.extend(n1.tolist())
    data.extend(c11.tolist())


def boundary_energy_gradient_hessian(
    u,
    problem,
    kappa,
    with_hessian,
):
    parameters = problem.physics
    n = u.size
    grad = np.zeros(n, dtype=float)
    rows = []
    cols = []
    data = []
    energy = 0.0

    n0 = problem.bottom_edges[:, 0]
    n1 = problem.bottom_edges[:, 1]
    u0 = u[n0]
    u1 = u[n1]
    lengths = problem.bottom_lengths

    cathode_mask = problem.bottom_edge_labels == 0
    anode_mask = ~cathode_mask

    for xi, w in zip(GAUSS_XI, GAUSS_W):
        nshape = np.array([1.0 - xi, xi])
        uq = nshape[0] * u0 + nshape[1] * u1

        if np.any(cathode_mask):
            ic_val = i_c(uq[cathode_mask], parameters)
            ic_prim = i_c_primitive(uq[cathode_mask], parameters)
            weight = lengths[cathode_mask] * w / kappa
            energy += np.sum(weight * ic_prim)
            grad[n0[cathode_mask]] += weight * ic_val * nshape[0]
            grad[n1[cathode_mask]] += weight * ic_val * nshape[1]
            if with_hessian:
                coeff = weight * i_c_prime(uq[cathode_mask], parameters)
                edge_hessian_entries(
                    n0[cathode_mask], n1[cathode_mask], coeff, nshape, rows, cols, data
                )

        if np.any(anode_mask):
            ia_val = i_a(uq[anode_mask], parameters)
            ia_prim = i_a_primitive(uq[anode_mask], parameters)
            weight = lengths[anode_mask] * w / kappa
            energy += np.sum(weight * ia_prim)
            grad[n0[anode_mask]] += weight * ia_val * nshape[0]
            grad[n1[anode_mask]] += weight * ia_val * nshape[1]
            if with_hessian:
                coeff = weight * i_a_prime(uq[anode_mask], parameters)
                edge_hessian_entries(
                    n0[anode_mask], n1[anode_mask], coeff, nshape, rows, cols, data
                )

    hessian = None
    if with_hessian:
        hessian = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return energy, grad, hessian


def energy_gradient_hessian(
    u,
    problem,
    kappa,
    with_hessian=True,
):
    bulk_energy = 0.5 * float(u @ (problem.stiffness @ u))
    bulk_grad = problem.stiffness @ u
    boundary_energy, boundary_grad, boundary_hessian = boundary_energy_gradient_hessian(
        u, problem, kappa, with_hessian
    )
    energy = bulk_energy + boundary_energy
    grad = bulk_grad + boundary_grad
    hessian = None
    if with_hessian:
        hessian = problem.stiffness + boundary_hessian
    return energy, grad, hessian


def projected_gradient_fallback(
    u,
    energy,
    grad,
    hessian,
    problem,
    kappa,
):
    parameters = problem.physics
    max_diag = float(np.max(hessian.diagonal()))
    alpha = min(1.0, 1.0 / max(max_diag, 1.0))

    while alpha >= parameters.line_search_min:
        candidate = np.clip(u - alpha * grad, parameters.phi_a, parameters.phi_c)
        cand_energy, cand_grad, cand_hessian = energy_gradient_hessian(
            candidate, problem, kappa, with_hessian=True
        )
        if cand_energy < energy:
            return candidate, cand_energy, cand_grad, cand_hessian, alpha
        alpha *= 0.5
    return None


def solve_with_lbfgsb(
    problem,
    kappa,
    initial,
):
    parameters = problem.physics
    bounds = [(parameters.phi_a, parameters.phi_c)] * initial.size

    def objective(vec):
        energy, grad, _ = energy_gradient_hessian(vec, problem, kappa, with_hessian=False)
        return float(energy), grad

    result = optimize.minimize(
        fun=objective,
        x0=np.clip(initial.copy(), parameters.phi_a, parameters.phi_c),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 4000,
            "maxls": 50,
            "ftol": 1.0e-14,
            "gtol": parameters.newton_tol,
        },
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"L-BFGS-B failed for kappa={kappa:.3e}: {result.message}")

    u = np.clip(result.x, parameters.phi_a, parameters.phi_c)
    energy, grad, _ = energy_gradient_hessian(u, problem, kappa, with_hessian=False)
    _, initial_grad, _ = energy_gradient_hessian(initial, problem, kappa, with_hessian=False)
    initial_residual = float(np.linalg.norm(initial_grad, ord=np.inf))
    info = {
        "iterations": float(result.nit),
        "residual_inf": float(np.linalg.norm(grad, ord=np.inf)),
        "initial_residual_inf": initial_residual,
        "certified_threshold": certified_residual_threshold(parameters, initial_residual),
        "accepted_step": -1.0,
        "energy": float(energy),
        "solver": "L-BFGS-B",
    }
    return u, info


def polish_with_newton(
    problem,
    kappa,
    initial,
    maxit=25,
):
    parameters = problem.physics
    u = np.clip(initial.copy(), parameters.phi_a, parameters.phi_c)
    energy, grad, hessian = energy_gradient_hessian(u, problem, kappa, with_hessian=True)
    initial_residual = float(np.linalg.norm(grad, ord=np.inf))
    accepted_step = 0.0
    it_used = 0

    for it in range(1, maxit + 1):
        it_used = it
        residual = float(np.linalg.norm(grad, ord=np.inf))
        if residual < parameters.newton_tol:
            break

        step = spsolve(hessian, -grad)
        step_norm = float(np.linalg.norm(step, ord=np.inf))
        if step_norm < parameters.newton_step_tol:
            break

        directional = float(grad @ step)
        alpha = 1.0
        improved = False

        while alpha >= parameters.line_search_min:
            candidate = np.clip(u + alpha * step, parameters.phi_a, parameters.phi_c)
            trial_energy, trial_grad, trial_hessian = energy_gradient_hessian(
                candidate, problem, kappa, with_hessian=True
            )
            if trial_energy <= energy + parameters.line_search_c1 * alpha * directional:
                u = candidate
                energy = trial_energy
                grad = trial_grad
                hessian = trial_hessian
                accepted_step = alpha
                improved = True
                break
            alpha *= 0.5

        if not improved:
            break

    residual = float(np.linalg.norm(grad, ord=np.inf))
    info = {
        "iterations": float(it_used),
        "residual_inf": residual,
        "initial_residual_inf": initial_residual,
        "certified_threshold": certified_residual_threshold(parameters, initial_residual),
        "accepted_step": accepted_step,
        "energy": float(energy),
        "solver": "L-BFGS-B+Newton",
    }
    return u, info


def polish_with_unconstrained_newton(
    problem,
    kappa,
    initial,
    maxit=40,
):
    parameters = problem.physics
    u = initial.copy()
    energy, grad, hessian = energy_gradient_hessian(u, problem, kappa, with_hessian=True)
    initial_residual = float(np.linalg.norm(grad, ord=np.inf))
    accepted_step = 0.0
    it_used = 0

    for it in range(1, maxit + 1):
        it_used = it
        residual = float(np.linalg.norm(grad, ord=np.inf))
        if residual < parameters.newton_tol:
            break

        step = spsolve(hessian, -grad)
        step_norm = float(np.linalg.norm(step, ord=np.inf))
        if step_norm < parameters.newton_step_tol:
            break

        directional = float(grad @ step)
        alpha = 1.0
        improved = False

        while alpha >= parameters.line_search_min:
            candidate = u + alpha * step
            trial_energy, trial_grad, trial_hessian = energy_gradient_hessian(
                candidate, problem, kappa, with_hessian=True
            )
            if (
                math.isfinite(trial_energy)
                and trial_energy <= energy + parameters.line_search_c1 * alpha * directional
            ):
                u = candidate
                energy = trial_energy
                grad = trial_grad
                hessian = trial_hessian
                accepted_step = alpha
                improved = True
                break
            alpha *= 0.5

        if not improved:
            break

    residual = float(np.linalg.norm(grad, ord=np.inf))
    info = {
        "iterations": float(it_used),
        "residual_inf": residual,
        "initial_residual_inf": initial_residual,
        "certified_threshold": certified_residual_threshold(parameters, initial_residual),
        "accepted_step": accepted_step,
        "energy": float(energy),
        "solver": "Newton-unconstrained",
    }
    return u, info


def solve_problem(
    problem,
    kappa,
    initial=None,
):
    parameters = problem.physics
    if initial is None:
        avg = 0.5 * (parameters.phi_c + parameters.phi_a)
        u = np.full(problem.nodes.shape[0], avg, dtype=float)
    else:
        u = np.clip(initial.copy(), parameters.phi_a, parameters.phi_c)

    energy, grad, hessian = energy_gradient_hessian(u, problem, kappa, with_hessian=True)
    initial_residual = float(np.linalg.norm(grad, ord=np.inf))
    residual = float(np.linalg.norm(grad, ord=np.inf))
    accepted_step = 0.0
    it_used = 0

    for it in range(1, parameters.newton_maxit + 1):
        it_used = it
        residual = float(np.linalg.norm(grad, ord=np.inf))
        if residual < parameters.newton_tol:
            break

        step = spsolve(hessian, -grad)
        step_norm = float(np.linalg.norm(step, ord=np.inf))
        if step_norm < parameters.newton_step_tol:
            break

        directional = float(grad @ step)
        alpha = 1.0
        trial_u = None
        trial_energy = math.inf
        trial_grad = None
        trial_hessian = None

        while alpha >= parameters.line_search_min:
            candidate = np.clip(u + alpha * step, parameters.phi_a, parameters.phi_c)
            trial_energy, trial_grad, trial_hessian = energy_gradient_hessian(
                candidate, problem, kappa, with_hessian=True
            )
            if trial_energy <= energy + parameters.line_search_c1 * alpha * directional:
                trial_u = candidate
                break
            alpha *= 0.5

        if trial_u is None:
            fallback = projected_gradient_fallback(
                u=u,
                energy=energy,
                grad=grad,
                hessian=hessian,
                problem=problem,
                kappa=kappa,
            )
            if fallback is None:
                return solve_with_lbfgsb(problem, kappa, u)
            trial_u, trial_energy, trial_grad, trial_hessian, alpha = fallback

        u = trial_u
        energy = trial_energy
        grad = trial_grad
        hessian = trial_hessian
        accepted_step = alpha

    residual = float(np.linalg.norm(grad, ord=np.inf))
    info = {
        "iterations": float(it_used),
        "residual_inf": residual,
        "initial_residual_inf": initial_residual,
        "certified_threshold": certified_residual_threshold(parameters, initial_residual),
        "accepted_step": accepted_step,
        "energy": float(energy),
        "solver": "Newton",
    }
    if residual > 10.0 * parameters.newton_tol:
        u_lbfgs, info_lbfgs = solve_with_lbfgsb(problem, kappa, u)
        if is_certified_info(info_lbfgs, parameters):
            return u_lbfgs, info_lbfgs
        u_polished, info_polished = polish_with_newton(problem, kappa, u_lbfgs)
        if is_certified_info(info_polished, parameters):
            return u_polished, info_polished
        if float(info_polished["residual_inf"]) < float(info_lbfgs["residual_inf"]):
            return u_polished, info_polished
        return u_lbfgs, info_lbfgs
    return u, info


def continuation_kappas(target_kappa):
    """Conductivity values used to reach a small target gradually."""
    base = (
        1.0, 0.3, 0.1, 0.03, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4,
        1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7,
    )
    levels = [kappa for kappa in base if kappa > target_kappa * (1.0 + 1.0e-12)]
    levels.append(target_kappa)
    return tuple(levels)


def solve_with_continuation(problem, target_kappa):
    """Solve successively from large conductivity down to the target value."""
    previous = solve_mixed_reference(problem)
    solution = previous
    info = None
    for kappa in continuation_kappas(target_kappa):
        solution, info = solve_problem(problem, kappa, previous)
        previous = solution
    if info is None:
        raise RuntimeError("The continuation solve did not run.")
    return solution, info
