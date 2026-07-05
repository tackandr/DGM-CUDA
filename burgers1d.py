"""
1D Burgers equation solver using the Discontinuous Galerkin method.

Solves the inviscid Burgers equation:

    u_t + (u^2/2)_x = 0

on a periodic domain [x_left, x_right] using M elements of polynomial order N.
The solution is stored as an M x (N+1) array, where index [m, i] holds the
nodal value at the i-th GLL node of element m.

A connectivity matrix (EToE, EToF) records the neighboring element and face
index for every face of every element.  The LIFT operator uses this information
to gather the correct exterior (plus-side) nodal values when computing the
numerical flux at each interface.

Public API
----------
build_mesh          -- uniform physical node coordinates, shape (M, N+1)
build_operators     -- V, D, LIFT matrices for the reference element
build_connectivity  -- EToE / EToF connectivity arrays, shape (M, 2)
burgers_godunov_flux -- Godunov upwind flux for Burgers' equation
numerical_flux      -- alias for burgers_godunov_flux (backward compat)
exact_sine_solution -- pre-shock analytical solution for u(x,0) = sin(pi x)
burgers_rhs         -- DG right-hand side, shape (M, N+1)
limit_solution      -- TVB minmod limiter for troubled cells
rk4_step            -- one classical RK4 advance
simulate            -- full time-integration driver
"""

from functools import lru_cache

import numpy as np

from dgm1d import (
    legendreGL,
    Vandermonde1D,
    GradVandermonde1D,
)

from mesh1d import build_mesh, build_operators, build_connectivity  # noqa: F401

from limiters1d import (                                               # noqa: E402
    minmod3 as _minmod3_lib,
    tvb_minmod as _tvb_minmod_lib,
    build_limiter_data as _build_limiter_data_lib,
    limit_scalar as limit_solution,
)


# ---------------------------------------------------------------------------
# Numerical flux
# ---------------------------------------------------------------------------

def burgers_godunov_flux(u_minus, u_plus):
    """Godunov upwind numerical flux for Burgers' equation.

    For the convex Burgers flux f(u) = u^2 / 2, the Godunov flux equals the
    upwind state selected by the local Riemann problem:

      * shocks (u- > u+): use the state indicated by the shock speed
      * rarefactions (u- <= u+): use the left state if both are positive,
        the right state if both are negative, and zero when the fan straddles 0

    Operates element-wise; inputs may be scalars or arrays of any shape.
    """
    u_minus = np.asarray(u_minus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    flux_minus = 0.5 * u_minus**2
    flux_plus = 0.5 * u_plus**2

    shock = u_minus > u_plus
    shock_speed = 0.5 * (u_minus + u_plus)
    shock_flux = np.where(shock_speed >= 0.0, flux_minus, flux_plus)

    rarefaction_flux = np.where(
        u_plus <= 0.0,
        flux_plus,
        np.where(u_minus >= 0.0, flux_minus, 0.0),
    )
    return np.where(shock, shock_flux, rarefaction_flux)


# Backward-compatible alias
numerical_flux = burgers_godunov_flux


def exact_sine_solution(x, t, tol=1e-13, max_iter=50):
    """Return the analytical pre-shock Burgers solution for u(x,0)=sin(pi x).

    For inviscid Burgers, characteristics give the implicit relation

        u(x, t) = sin(pi * (x - t * u(x, t)))

    which is single-valued only before the first shock forms at t = 1 / pi.

    Parameters
    ----------
    x : array_like
        Evaluation points.
    t : float
        Time at which to evaluate the pre-shock solution.
    tol : float, optional
        Newton-iteration stopping tolerance on the update size.
    max_iter : int, optional
        Maximum number of Newton iterations.
    """
    if t < 0.0:
        raise ValueError("t must be non-negative")
    if t >= 1.0 / np.pi:
        raise ValueError("exact_sine_solution is only defined before shock formation (t < 1/pi)")

    x = np.asarray(x, dtype=float)
    if t == 0.0:
        return np.sin(np.pi * x)

    u = np.sin(np.pi * x)
    for _ in range(max_iter):
        phase = np.pi * (x - t * u)
        residual = u - np.sin(phase)
        jacobian = 1.0 + np.pi * t * np.cos(phase)
        delta = residual / jacobian
        u -= delta
        if np.max(np.abs(delta)) < tol:
            return u

    raise RuntimeError("Newton iteration did not converge for the analytical sine-wave solution")


def _quadrature_order(N):
    """Return a quadrature order exact for the Burgers volume and mean terms."""
    # For degree-N states, f(u)=u^2/2 reaches degree 2N and its weak-form volume
    # term multiplies another degree-N basis polynomial derivative, so 2N+3
    # points keep both the stiffness assembly and limiter averages exact.
    return max(2 * N + 3, 8)


@lru_cache(maxsize=16)
def _build_weak_form_operators(N):
    """Return exact reference-element weak-form operators for degree N."""
    r = legendreGL(N)
    V = Vandermonde1D(N, r)
    V_inv = np.linalg.solve(V, np.eye(V.shape[0]))

    q_order = _quadrature_order(N)
    q_nodes, q_weights = np.polynomial.legendre.leggauss(q_order)
    Vq = Vandermonde1D(N, q_nodes)
    Vrq = GradVandermonde1D(N, q_nodes)

    interp = Vq @ V_inv
    grad_interp = Vrq @ V_inv
    weighted_interp = q_weights[:, np.newaxis] * interp

    mass = interp.T @ weighted_interp
    inv_mass = np.linalg.solve(mass, np.eye(mass.shape[0]))
    stiffness = weighted_interp.T @ grad_interp
    return inv_mass, stiffness


# ---------------------------------------------------------------------------
# DG right-hand side
# ---------------------------------------------------------------------------

def burgers_rhs(u, D, LIFT, J, EToE, EToF):
    """Compute the DG right-hand side for the inviscid 1D Burgers equation.

    Uses a weak-form discretisation on each element k:

        J_k * M * du_k/dt = S^T f(u_k) - [F*_left, 0, ..., 0, F*_right]

    where M is the reference-element mass matrix, S is the stiffness matrix and
    f(u) = u^2/2. The reference operators are integrated with sufficiently high
    Gauss-Legendre quadrature so the semi-discrete residual is not contaminated
    by under-integrated nonlinear volume terms. Interface fluxes use a consistent
    left-right orientation so that the same single-valued F* is seen by both
    elements sharing an interface; the left boundary contribution carries a
    minus sign because the weak-form boundary term is evaluated with the outward
    normal on the left face.

    Parameters
    ----------
    u : ndarray, shape (M, N+1)
        Current nodal values.
    D : ndarray, shape (N+1, N+1)
        Reference-element differentiation matrix, used only to validate N.
    LIFT : ndarray, shape (N+1, 2)
        Reference-element lift operator, used only to validate N.
    J : ndarray, shape (M,)
        Per-element Jacobian (physical element width / 2).
    EToE : ndarray, shape (M, 2), int
        Element-to-element connectivity (from build_connectivity).
    EToF : ndarray, shape (M, 2), int
        Element-to-face connectivity (from build_connectivity).

    Returns
    -------
    rhs : ndarray, shape (M, N+1)
    """
    if D.shape[0] != u.shape[1] or LIFT.shape != (u.shape[1], 2):
        raise ValueError("D and LIFT must match the element polynomial order carried by u")

    N = u.shape[1] - 1
    inv_mass, stiffness = _build_weak_form_operators(N)

    flux = 0.5 * u**2
    boundary = np.zeros_like(u)

    left_neigh = EToE[:, 0]
    right_neigh = EToE[:, 1]
    boundary[:, 0] = -numerical_flux(u[left_neigh, N], u[:, 0])  # exterior-left, then interior-left
    boundary[:, N] = numerical_flux(u[:, N], u[right_neigh, 0])

    residual = flux @ stiffness - boundary
    return (residual @ inv_mass.T) / J[:, np.newaxis]


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------

def rk4_step(u, dt, D, LIFT, J, EToE, EToF):
    """Advance u by one classical 4th-order Runge-Kutta step.

    Parameters
    ----------
    u : ndarray, shape (M, N+1)
    dt : float
    D, LIFT, J, EToE, EToF : see burgers_rhs

    Returns
    -------
    u_new : ndarray, shape (M, N+1)
    """
    def rhs(v):
        return burgers_rhs(v, D, LIFT, J, EToE, EToF)

    u0 = limit_solution(u, J)
    k1 = rhs(u0)
    u1 = limit_solution(u0 + 0.5 * dt * k1, J)
    k2 = rhs(u1)
    u2 = limit_solution(u0 + 0.5 * dt * k2, J)
    k3 = rhs(u2)
    u3 = limit_solution(u0 + dt * k3, J)
    k4 = rhs(u3)
    return limit_solution(u0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), J)


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def simulate(M, N, t_final, cfl=0.5, x_left=0.0, x_right=2.0, u_init=None):
    """Run the 1D inviscid Burgers equation simulation.

    Time steps are chosen adaptively via the CFL condition

        dt = cfl * h / (max|u| * (N+1)^2)

    where h = (x_right - x_left) / M is the element width.  Periodic
    boundary conditions are imposed through the connectivity matrix.

    Parameters
    ----------
    M : int
        Number of elements.
    N : int
        Polynomial order; each element has N+1 GLL nodes.
    t_final : float
        Final simulation time.
    cfl : float, optional
        CFL number (default 0.5).
    x_left, x_right : float, optional
        Domain boundaries (default [0, 2]).
    u_init : callable or None, optional
        Initial condition as a function of x.  Receives an (M, N+1) array of
        node coordinates and must return an array of the same shape.
        Defaults to u(x) = sin(pi * x).

    Returns
    -------
    x : ndarray, shape (M, N+1)
        Node coordinates.
    u : ndarray, shape (M, N+1)
        Solution at time t.
    t : float
        Actual final time reached (equals t_final barring floating-point rounding).
    """
    x, r = build_mesh(M, N, x_left, x_right)
    _, D, LIFT = build_operators(N, r)
    EToE, EToF = build_connectivity(M)

    h = (x_right - x_left) / M        # uniform element width
    J = np.full(M, 0.5 * h)           # Jacobian: (b-a)/2 per element

    if u_init is None:
        u_init = lambda xv: np.sin(np.pi * xv)
    u = u_init(x).astype(float)

    t = 0.0
    # Small floor prevents division by zero when u is identically zero
    _MIN_SPEED = 1e-14
    while t < t_final:
        max_speed = np.max(np.abs(u)) + _MIN_SPEED
        dt = cfl * h / (max_speed * (N + 1) ** 2)
        dt = min(dt, t_final - t)
        if dt <= 0.0:
            break
        u = rk4_step(u, dt, D, LIFT, J, EToE, EToF)
        t += dt

    return x, u, t
