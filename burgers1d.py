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
numerical_flux      -- local Lax-Friedrichs flux for Burgers' equation
burgers_rhs         -- DG right-hand side, shape (M, N+1)
rk4_step            -- one classical RK4 advance
simulate            -- full time-integration driver
"""

import numpy as np

from dgm1d import legendreGL, Vandermonde1D, DMatrix1D, Lift1D


# ---------------------------------------------------------------------------
# Mesh and operator construction
# ---------------------------------------------------------------------------

def build_mesh(M, N, x_left=0.0, x_right=2.0):
    """Return the physical node coordinates for a uniform mesh.

    The domain [x_left, x_right] is divided into M equal elements.  Within
    each element the N+1 Gauss-Lobatto-Legendre (GLL) nodes are mapped from
    the reference element [-1, 1] to the physical element [a, b].

    Parameters
    ----------
    M : int
        Number of elements.
    N : int
        Polynomial order; each element has N+1 nodes.
    x_left, x_right : float
        Domain boundaries.

    Returns
    -------
    x : ndarray, shape (M, N+1)
        Physical coordinates.  x[m, i] is the i-th node of element m.
    r : ndarray, shape (N+1,)
        GLL reference nodes on [-1, 1].
    """
    r = legendreGL(N)
    boundaries = np.linspace(x_left, x_right, M + 1)
    x = np.empty((M, N + 1))
    for m in range(M):
        a, b = boundaries[m], boundaries[m + 1]
        x[m] = 0.5 * (b - a) * r + 0.5 * (a + b)
    return x, r


def build_operators(N, r):
    """Return the DG operators for the reference element of order N.

    Parameters
    ----------
    N : int
        Polynomial order.
    r : ndarray, shape (N+1,)
        GLL reference nodes (e.g. from legendreGL(N)).

    Returns
    -------
    V : ndarray, shape (N+1, N+1)
        Vandermonde matrix.
    D : ndarray, shape (N+1, N+1)
        Nodal differentiation matrix.  Applying D @ u_ref gives the derivative
        of the interpolating polynomial at the reference nodes.
    LIFT : ndarray, shape (N+1, 2)
        Lift operator mapping two boundary flux values to N+1 nodal residuals.
    """
    V = Vandermonde1D(N, r)
    D = DMatrix1D(N, r)
    LIFT = Lift1D(N, V)
    return V, D, LIFT


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

def build_connectivity(M):
    """Return element-to-element and element-to-face connectivity arrays.

    For a 1D periodic mesh of M elements, each element has two faces:
      face 0 = left face,  face 1 = right face.

    The connectivity is:
      Left  neighbor of element m : element (m-1) % M, face 1 (its right face)
      Right neighbor of element m : element (m+1) % M, face 0 (its left face)

    Parameters
    ----------
    M : int
        Number of elements.

    Returns
    -------
    EToE : ndarray, shape (M, 2), dtype int
        EToE[m, f] is the index of the element neighboring element m at face f.
    EToF : ndarray, shape (M, 2), dtype int
        EToF[m, f] is the face index on the neighboring element EToE[m, f]
        that is shared with element m at face f.
    """
    idx = np.arange(M)
    EToE = np.stack([(idx - 1) % M, (idx + 1) % M], axis=1).astype(int)
    EToF = np.zeros((M, 2), dtype=int)
    EToF[:, 0] = 1  # left face's neighbor contributes its right face
    # EToF[:, 1] = 0 already (right face's neighbor contributes its left face)
    return EToE, EToF


# ---------------------------------------------------------------------------
# Numerical flux
# ---------------------------------------------------------------------------

def numerical_flux(u_minus, u_plus):
    """Local Lax-Friedrichs (Rusanov) numerical flux for Burgers' equation.

    Computes f*(u-, u+) = (f(u-) + f(u+)) / 2 - alpha/2 * (u+ - u-)
    where f(u) = u^2/2 and alpha = max(|u-|, |u+|).

    Operates element-wise; inputs may be scalars or arrays of any shape.

    Parameters
    ----------
    u_minus : array_like
        Interior (minus-side) nodal values.
    u_plus : array_like
        Exterior (plus-side) nodal values.

    Returns
    -------
    f_star : ndarray
        Numerical flux values, same shape as inputs.
    """
    u_minus = np.asarray(u_minus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    alpha = np.maximum(np.abs(u_minus), np.abs(u_plus))
    return 0.5 * (0.5 * u_minus**2 + 0.5 * u_plus**2) - 0.5 * alpha * (u_plus - u_minus)


# ---------------------------------------------------------------------------
# DG right-hand side
# ---------------------------------------------------------------------------

def burgers_rhs(u, D, LIFT, J, EToE, EToF):
    """Compute the DG right-hand side for the inviscid 1D Burgers equation.

    Uses the strong form on each element k:

        J_k * du_k/dt = D @ f(u_k)
                        - LIFT @ [n_left  * (F*_left  - f(u_k[0])),
                                  n_right * (F*_right - f(u_k[N]))]

    where f(u) = u^2/2, n_left = -1, n_right = +1 are outward normals.
    Interface fluxes use a consistent left-right orientation so that the same
    single-valued F* is seen by both elements sharing an interface (conservation
    form).  The connectivity arrays EToE / EToF identify the neighboring element
    and face for each face of each element.

    Parameters
    ----------
    u : ndarray, shape (M, N+1)
        Current nodal values.
    D : ndarray, shape (N+1, N+1)
        Reference-element differentiation matrix.
    LIFT : ndarray, shape (N+1, 2)
        Lift operator.
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
    N = u.shape[1] - 1
    fmask = np.array([0, N])     # local node index for each face (left=0, right=N)
    normals = np.array([-1.0, 1.0])  # outward normals at left and right faces

    # Physical flux at all nodes
    flux = 0.5 * u**2            # shape (M, N+1)

    # --- Volume term ---
    # vol[m] = D @ flux[m]  =>  vol = flux @ D.T   shape (M, N+1)
    vol = flux @ D.T

    # --- Surface term ---
    # Boundary (interior) values of the current element at each face, shape (M, 2)
    u_int = np.stack([u[:, fmask[0]], u[:, fmask[1]]], axis=1)

    # Exterior values gathered from neighboring elements, shape (M, 2)
    u_ext = np.empty_like(u_int)
    for f in range(2):
        neigh_elem = EToE[:, f]            # (M,) indices of neighboring elements
        neigh_node = fmask[EToF[:, f]]     # (M,) local node on the neighbor's face
        u_ext[:, f] = u[neigh_elem, neigh_node]

    # Compute single-valued interface fluxes using a consistent L-R orientation:
    #   Face 0 (left):  minus-side = left-neighbor's right boundary (u_ext[:,0])
    #                   plus-side  = current element's left boundary  (u_int[:,0])
    #   Face 1 (right): minus-side = current element's right boundary (u_int[:,1])
    #                   plus-side  = right-neighbor's left boundary   (u_ext[:,1])
    # The same F* is therefore used by both elements sharing an interface,
    # which is the requirement for the scheme to be in conservation form.
    f_star = np.stack([
        numerical_flux(u_ext[:, 0], u_int[:, 0]),   # face 0: L=neighbour, R=current
        numerical_flux(u_int[:, 1], u_ext[:, 1]),   # face 1: L=current,   R=neighbour
    ], axis=1)  # shape (M, 2)

    # Interior flux at the boundary nodes, shape (M, 2)
    f_interior = 0.5 * u_int**2

    # Flux difference scaled by outward normal: n * (F* - f_interior), shape (M, 2)
    flux_diff = normals * (f_star - f_interior)

    # Distribute boundary corrections to interior DOFs via LIFT
    # LIFT is (N+1, 2); flux_diff.T is (2, M) → result is (N+1, M), then transpose
    surf = (LIFT @ flux_diff.T).T          # shape (M, N+1)

    return (vol - surf) / J[:, np.newaxis]


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

    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


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
