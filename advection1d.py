"""
1D variable-coefficient advection solver using the Discontinuous Galerkin method.

Equation:
    u_t + (a(x, t) * u)_x = 0

where a(x, t) is a given velocity field provided as a callable.  The
velocity may vary in both space and time, making the equation nonlinear
in the sense of having non-constant wave speed (variable-coefficient
hyperbolic conservation law).

Public API
----------
advection_rhs_1d       -- DG right-hand side, shape (M, N+1)
rk4_step_advection_1d  -- one classical RK4 advance (time-dependent RHS)
simulate_advection1d   -- full time-integration driver
"""

from functools import lru_cache

import numpy as np

from dgm1d import GradVandermonde1D, Vandermonde1D, legendreGL
from limiters1d import limit_scalar
from mesh1d import build_connectivity, build_mesh, build_operators


# ---------------------------------------------------------------------------
# Reference-element operators (cached per polynomial order)
# ---------------------------------------------------------------------------

def _quadrature_order(N):
    """Return a quadrature order exact for degree-2N flux products."""
    return max(2 * N + 3, 8)


@lru_cache(maxsize=16)
def _build_weak_form_operators(N):
    """Return exact reference-element weak-form operators for degree N.

    Builds the inverse mass matrix and stiffness matrix using over-integrated
    Gauss-Legendre quadrature so that the weak-form volume integrals of degree
    up to 2N are computed exactly.

    Parameters
    ----------
    N : int
        Polynomial order.

    Returns
    -------
    inv_mass : ndarray, shape (N+1, N+1)
    stiffness : ndarray, shape (N+1, N+1)
        stiffness[i, j] = ∫_{-1}^{1} φ_i(r) * dφ_j/dr dr
    """
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
# Numerical flux
# ---------------------------------------------------------------------------

def _upwind_flux_1d(u_minus, u_plus, an):
    """Scalar upwind numerical flux at a 1D interface.

    Selects the upwind state based on the sign of the normal velocity ``an``:
    if ``an > 0`` (flow goes right) the left state ``u_minus`` is used; if
    ``an < 0`` the right state ``u_plus`` is used.  Written in the unified form

        F*(u-, u+, a) = 0.5 * [a*(u- + u+) - |a|*(u+ - u-)]

    Parameters
    ----------
    u_minus : array_like
        Interior (left) state at the face.
    u_plus : array_like
        Exterior (right) state at the face.
    an : array_like
        Normal velocity at the face (positive = rightward).

    Returns
    -------
    ndarray
        Upwind numerical flux.
    """
    return 0.5 * (an * (u_minus + u_plus) - np.abs(an) * (u_plus - u_minus))


# ---------------------------------------------------------------------------
# DG right-hand side
# ---------------------------------------------------------------------------

def advection_rhs_1d(
    u,
    x,
    t,
    velocity_fn,
    stiffness,
    inv_mass,
    J,
    EToE,
    EToF,
    bc_type="periodic",
    u_bc=None,
):
    """Compute the DG right-hand side for 1D variable-coefficient advection.

    Discretises u_t + (a(x,t)*u)_x = 0 using a weak-form nodal DG method.
    The volume flux f = a(x,t)*u is evaluated at all element nodes and
    integrated against the test-function gradient via the pre-built stiffness
    matrix.  Interface fluxes use an upwind scheme keyed to the sign of the
    local velocity.

    Parameters
    ----------
    u : ndarray, shape (M, N+1)
        Current nodal values.
    x : ndarray, shape (M, N+1)
        Physical node coordinates.
    t : float
        Current time.
    velocity_fn : callable(x, t) -> ndarray
        Velocity field a(x, t).  Called with an array ``x`` and scalar ``t``;
        must return an array of the same shape as ``x``.
    stiffness : ndarray, shape (N+1, N+1)
        Reference-element stiffness matrix (from ``_build_weak_form_operators``).
    inv_mass : ndarray, shape (N+1, N+1)
        Reference-element inverse mass matrix.
    J : ndarray, shape (M,)
        Per-element Jacobians (half element width h/2).
    EToE : ndarray, shape (M, 2), int
        Element-to-element connectivity (from ``build_connectivity``).
    EToF : ndarray, shape (M, 2), int
        Element-to-face connectivity.
    bc_type : {'periodic', 'transmissive', 'dirichlet'}
        Boundary condition type.
    u_bc : callable(x, t) -> float, optional
        Boundary value function; required for ``bc_type='dirichlet'``.

    Returns
    -------
    rhs : ndarray, shape (M, N+1)
    """
    M, Np1 = u.shape
    N = Np1 - 1

    # Evaluate velocity field at all nodes: shape (M, N+1)
    a = np.asarray(velocity_fn(x, t), dtype=float)

    # Volume flux f(u, x, t) = a(x, t) * u, shape (M, N+1)
    flux = a * u

    # Gather exterior states using periodic element connectivity
    left_neigh = EToE[:, 0]    # (M,)  left neighbour element indices
    right_neigh = EToE[:, 1]   # (M,)  right neighbour element indices

    u_left_ext = u[left_neigh, N].copy()    # exterior state at each left face
    u_right_ext = u[right_neigh, 0].copy()  # exterior state at each right face

    # Override exterior states at physical domain boundaries for non-periodic BC
    if bc_type != "periodic":
        if bc_type == "transmissive":
            # Zero-gradient outflow: ghost cell = interior face value
            u_left_ext[0] = u[0, 0]
            u_right_ext[-1] = u[-1, N]
        elif bc_type == "dirichlet":
            if u_bc is None:
                raise ValueError("u_bc must be provided for bc_type='dirichlet'")
            u_left_ext[0] = float(u_bc(x[0, 0], t))
            u_right_ext[-1] = float(u_bc(x[-1, N], t))
        else:
            raise ValueError(
                f"Unsupported bc_type '{bc_type}'. "
                "Supported types: 'periodic', 'transmissive', 'dirichlet'."
            )

    # Velocity at left and right face nodes of each element
    a_left = a[:, 0]   # (M,)
    a_right = a[:, N]  # (M,)

    # Boundary term: [F* - f_interior] contributed to the two boundary nodes
    # Left face (node 0): outward normal is -1, so flux contribution is negative
    # Right face (node N): outward normal is +1
    boundary = np.zeros_like(u)
    boundary[:, 0] = -_upwind_flux_1d(u_left_ext, u[:, 0], a_left)
    boundary[:, N] = _upwind_flux_1d(u[:, N], u_right_ext, a_right)

    # Weak-form residual: volume integral minus boundary correction
    # (flux @ stiffness)[m, i] = ∫ f(u_m) dφ_i/dr dr  (volume term)
    residual = flux @ stiffness - boundary
    return (residual @ inv_mass.T) / J[:, np.newaxis]


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------

def rk4_step_advection_1d(u, t, dt, rhs_fn):
    """Advance one classical 4th-order Runge-Kutta step.

    Handles a time-dependent right-hand side by evaluating ``rhs_fn`` at the
    correct RK stage times (t, t+dt/2, t+dt/2, t+dt).

    Parameters
    ----------
    u : ndarray, shape (M, N+1)
        Current nodal values.
    t : float
        Current time.
    dt : float
        Time step.
    rhs_fn : callable(u, t) -> ndarray
        Right-hand side function returning an array of the same shape as ``u``.

    Returns
    -------
    u_new : ndarray, shape (M, N+1)
    """
    k1 = rhs_fn(u, t)
    k2 = rhs_fn(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_fn(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_fn(u + dt * k3, t + dt)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def simulate_advection1d(
    M,
    N,
    t_final,
    velocity,
    cfl=0.5,
    x_left=0.0,
    x_right=1.0,
    u_init=None,
    bc_type="periodic",
    u_bc=None,
    limiter_type=None,
):
    """Run a 1D variable-coefficient advection simulation.

    Solves u_t + (a(x, t)*u)_x = 0 on [x_left, x_right] using a nodal DG
    scheme of polynomial order N on M uniform elements.  Time steps are chosen
    adaptively via the CFL condition

        dt = cfl * h / (max|a(x,t)| * (N+1)^2)

    evaluated with the velocity field at the current time.

    Parameters
    ----------
    M : int
        Number of elements.
    N : int
        Polynomial order; each element has N+1 GLL nodes.
    t_final : float
        Final simulation time.
    velocity : callable(x, t) -> ndarray
        Velocity field a(x, t).  Called with an (M, N+1) array ``x`` and a
        scalar time ``t``; must return an array of the same shape.
    cfl : float, optional
        CFL number (default 0.5).
    x_left, x_right : float, optional
        Domain boundaries (default [0, 1]).
    u_init : callable or None, optional
        Initial condition as a function of x.  Receives an (M, N+1) array of
        node coordinates and must return an array of the same shape.
        Defaults to u(x) = sin(2*pi*x).
    bc_type : {'periodic', 'transmissive', 'dirichlet'}, optional
        Boundary condition type (default 'periodic').
    u_bc : callable(x, t) -> float, optional
        Boundary value function; required for bc_type='dirichlet'.
    limiter_type : {None, 'tvb'}, optional
        Limiter applied after each RK4 step (default None).  'tvb' applies the
        TVB minmod limiter from limiters1d to suppress Gibbs oscillations near
        discontinuities.

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

    h = (x_right - x_left) / M
    J = np.full(M, 0.5 * h)

    inv_mass, stiffness = _build_weak_form_operators(N)

    if u_init is None:
        u_init = lambda xv: np.sin(2 * np.pi * xv)
    u = np.asarray(u_init(x), dtype=float)

    _SUPPORTED_LIMITERS = (None, "tvb")
    if limiter_type not in _SUPPORTED_LIMITERS:
        raise ValueError(
            f"Unsupported limiter_type '{limiter_type}'. "
            f"Supported types: {', '.join(str(s) for s in _SUPPORTED_LIMITERS)}."
        )

    def rhs(v, t_curr):
        return advection_rhs_1d(
            v, x, t_curr, velocity,
            stiffness, inv_mass, J,
            EToE, EToF,
            bc_type=bc_type,
            u_bc=u_bc,
        )

    _MIN_SPEED = 1e-14
    t = 0.0
    while t < t_final:
        a_now = np.asarray(velocity(x, t), dtype=float)
        max_speed = np.max(np.abs(a_now)) + _MIN_SPEED
        dt = cfl * h / (max_speed * (N + 1) ** 2)
        dt = min(dt, t_final - t)
        if dt <= 0.0:
            break
        u = rk4_step_advection_1d(u, t, dt, rhs)
        if limiter_type == "tvb":
            u = limit_scalar(u, J, bc_type=bc_type)
        t += dt

    return x, u, t
