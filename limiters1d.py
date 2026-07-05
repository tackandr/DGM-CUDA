"""
Limiter library for 1D DG solvers.

Provides TVB minmod limiters for both scalar fields and conserved-variable
systems (e.g. the 1D Euler equations).  The scalar limiter is extracted
directly from the original burgers1d.py implementation; the system limiter
adds a characteristic-variable rotation step to suppress spurious oscillations
at multi-wave discontinuities.

Public API
----------
minmod3            -- element-wise minmod of three arrays
tvb_minmod         -- TVB-relaxed three-argument minmod
build_limiter_data -- cached reference-node and mean-weight data
limit_scalar       -- TVB minmod limiter for a scalar (M, N+1) field
limit_system       -- TVB minmod limiter for a system (nvar, M, N+1) field
                      using characteristic-variable rotation
"""

from functools import lru_cache

import numpy as np

from dgm1d import Vandermonde1D, legendreGL


# ---------------------------------------------------------------------------
# Internal quadrature order
# ---------------------------------------------------------------------------

def _quadrature_order(N):
    return max(2 * N + 3, 8)


# ---------------------------------------------------------------------------
# Minmod primitives
# ---------------------------------------------------------------------------

def minmod3(a, b, c):
    """Return the minmod of three same-shaped arrays.

    The minmod function returns the value with the smallest magnitude when all
    arguments have the same sign, and zero otherwise.

    Parameters
    ----------
    a, b, c : array_like
        Input arrays; must be broadcast-compatible.

    Returns
    -------
    ndarray
        Element-wise minmod result.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    same_sign = (np.sign(a) == np.sign(b)) & (np.sign(a) == np.sign(c))
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(same_sign, np.sign(a) * magnitude, 0.0)


def tvb_minmod(a, b, c, threshold):
    """Return the TVB-relaxed minmod of three same-shaped arrays.

    The TVB modification skips limiting when |a| <= threshold, preserving
    smooth extrema whose boundary deviations are naturally O(dx^2).

    Parameters
    ----------
    a, b, c : array_like
        Input arrays.
    threshold : array_like
        Per-element TVB threshold (typically tvb_constant * dx^2).

    Returns
    -------
    ndarray
        Element-wise TVB minmod result.
    """
    return np.where(np.abs(a) <= threshold, a, minmod3(a, b, c))


# ---------------------------------------------------------------------------
# Reference-element data (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def build_limiter_data(N):
    """Return reference-node and cell-average data used by the limiter.

    Parameters
    ----------
    N : int
        Polynomial order.

    Returns
    -------
    r : ndarray, shape (N+1,)
        GLL reference nodes.
    mean_weights : ndarray, shape (N+1,)
        Quadrature weights for computing the cell average from nodal values.
        The cell average is u @ mean_weights.
    """
    r = legendreGL(N)
    V = Vandermonde1D(N, r)
    V_inv = np.linalg.solve(V, np.eye(V.shape[0]))

    q_order = _quadrature_order(N)
    q_nodes, q_weights = np.polynomial.legendre.leggauss(q_order)
    Vq = Vandermonde1D(N, q_nodes)
    interp = Vq @ V_inv
    mean_weights = 0.5 * (q_weights @ interp)
    return r, mean_weights


# ---------------------------------------------------------------------------
# Scalar limiter
# ---------------------------------------------------------------------------

def limit_scalar(u, J, tvb_constant=50.0, bc_type="periodic"):
    """Apply a TVB minmod limiter to a scalar DG field.

    Preserves the cell average and, when a cell is flagged as troubled,
    replaces the local polynomial with a bounded linear profile built from
    limited edge slopes.  The TVB threshold prevents clipping smooth extrema
    before the first shock.

    Parameters
    ----------
    u : ndarray, shape (M, N+1)
        Current nodal values.
    J : array_like, shape (M,)
        Per-element Jacobians.  The physical element width is dx = 2J.
    tvb_constant : float, optional
        TVB constant (default 50).
    bc_type : {'periodic', 'transmissive'}, optional
        Boundary condition type (default 'periodic').  For 'transmissive',
        the ghost cell averages at the domain boundaries are set equal to
        the boundary element averages (zero-gradient extrapolation) so
        that the periodic np.roll is not applied across the domain edges.

    Returns
    -------
    ndarray, shape (M, N+1)
        Limited nodal values.
    """
    N = u.shape[1] - 1
    if N <= 1:
        return u

    r, mean_weights = build_limiter_data(N)
    cell_average = u @ mean_weights
    avg_left = np.roll(cell_average, 1)
    avg_right = np.roll(cell_average, -1)
    if bc_type == "transmissive":
        avg_left[0] = cell_average[0]
        avg_right[-1] = cell_average[-1]

    left_slope = cell_average - u[:, 0]
    right_slope = u[:, N] - cell_average
    forward_diff = avg_right - cell_average
    backward_diff = cell_average - avg_left
    dx = 2.0 * np.asarray(J, dtype=float)
    tvb_threshold = tvb_constant * dx * dx

    limited_left = tvb_minmod(left_slope, forward_diff, backward_diff, tvb_threshold)
    limited_right = tvb_minmod(right_slope, forward_diff, backward_diff, tvb_threshold)
    limited_slope = 0.5 * (limited_left + limited_right)

    troubled = (np.abs(limited_left - left_slope) > 1e-12) | (np.abs(limited_right - right_slope) > 1e-12)
    if not np.any(troubled):
        return u

    limited = u.copy()
    limited[troubled] = cell_average[troubled, np.newaxis] + limited_slope[troubled, np.newaxis] * r[np.newaxis, :]
    return limited


# ---------------------------------------------------------------------------
# System limiter (characteristic-variable)
# ---------------------------------------------------------------------------

def _roe_eigenvectors_euler(rho_avg, u_avg, H_avg, c_avg, gamma):
    """Return the right and left eigenvector matrices of the 1D Euler Roe matrix.

    For a given Roe-averaged state (rho_avg, u_avg, H_avg, c_avg), returns
    R (right eigenvectors as columns) and L = R^{-1} (left eigenvectors as rows).

    Parameters
    ----------
    rho_avg, u_avg, H_avg, c_avg : float
        Roe-averaged density, velocity, total enthalpy, sound speed.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    R : ndarray, shape (3, 3)
    L : ndarray, shape (3, 3)
    """
    g1 = gamma - 1.0
    b1 = g1 / (2.0 * c_avg**2)
    b2 = u_avg * b1

    # Right eigenvectors (columns): [1-wave, 2-wave(entropy), 3-wave]
    R = np.array([
        [1.0,           1.0,           1.0          ],
        [u_avg - c_avg, u_avg,         u_avg + c_avg],
        [H_avg - u_avg * c_avg, 0.5 * u_avg**2, H_avg + u_avg * c_avg],
    ])

    # Left eigenvectors (rows) = R^{-1}
    L = np.array([
        [0.5 * (b1 * u_avg**2 + u_avg / c_avg),
         -0.5 * (b2 + 1.0 / c_avg),
         0.5 * b1],
        [1.0 - b1 * u_avg**2,
         b2,
         -b1],
        [0.5 * (b1 * u_avg**2 - u_avg / c_avg),
         -0.5 * (b2 - 1.0 / c_avg),
         0.5 * b1],
    ])
    return R, L


def limit_system(U, J, gamma=1.4, tvb_constant=50.0, bc_type="periodic"):
    """Apply a TVB minmod limiter to a conserved-variable system.

    For each troubled element (detected component-wise), rotates the solution
    to the local characteristic space, applies the scalar TVB limiter to each
    characteristic variable independently, then rotates back.  The local
    characteristic directions are derived from Roe-averaged states between
    each element and its neighbours.

    Parameters
    ----------
    U : ndarray, shape (nvar, M, N+1)
        Conserved-variable nodal values.  For the 1D Euler equations nvar=3
        and the variables are [rho, rho*u, E] in physical units.
    J : array_like, shape (M,)
        Per-element Jacobians.  Physical element width is dx = 2J.
    gamma : float, optional
        Ratio of specific heats (used to build Roe eigenvectors when nvar==3).
        Ignored for nvar != 3 (component-wise limiting falls back to
        limit_scalar applied independently to each variable).
    tvb_constant : float, optional
        TVB constant (default 50).
    bc_type : {'periodic', 'transmissive'}, optional
        Boundary condition type (default 'periodic').  For 'transmissive',
        ghost cell averages at the domain boundaries equal the boundary
        element averages (zero-gradient extrapolation).

    Returns
    -------
    ndarray, shape (nvar, M, N+1)
        Limited nodal values.
    """
    nvar, M, Np1 = U.shape
    N = Np1 - 1
    if N <= 1:
        return U

    if nvar != 3:
        # Fallback: apply scalar limiter to each component independently
        return np.stack([limit_scalar(U[v], J, tvb_constant, bc_type) for v in range(nvar)])

    r, mean_weights = build_limiter_data(N)
    J = np.asarray(J, dtype=float)
    dx = 2.0 * J
    tvb_threshold = tvb_constant * dx * dx

    # Cell averages for all variables, shape (nvar, M)
    cell_avg = U @ mean_weights          # (nvar, M)
    avg_L = np.roll(cell_avg, 1, axis=1)
    avg_R = np.roll(cell_avg, -1, axis=1)
    if bc_type == "transmissive":
        avg_L[:, 0] = cell_avg[:, 0]
        avg_R[:, -1] = cell_avg[:, -1]

    # Detect troubled cells using component-wise slope checking
    left_slope = cell_avg - U[:, :, 0]   # (nvar, M)
    right_slope = U[:, :, N] - cell_avg  # (nvar, M)
    forward_diff = avg_R - cell_avg      # (nvar, M)
    backward_diff = cell_avg - avg_L     # (nvar, M)

    limited_left_comp = tvb_minmod(left_slope, forward_diff, backward_diff, tvb_threshold[np.newaxis, :])
    limited_right_comp = tvb_minmod(right_slope, forward_diff, backward_diff, tvb_threshold[np.newaxis, :])

    troubled = np.any(
        (np.abs(limited_left_comp - left_slope) > 1e-12) |
        (np.abs(limited_right_comp - right_slope) > 1e-12),
        axis=0,
    )  # shape (M,)

    if not np.any(troubled):
        return U

    limited = U.copy()
    troubled_idx = np.where(troubled)[0]

    for m in troubled_idx:
        # Build Roe-averaged state between left and right neighbours for this element
        # Use cell averages of the left and right neighbours
        if bc_type == "transmissive":
            m_l = 0 if m == 0 else m - 1
            m_r = M - 1 if m == M - 1 else m + 1
        else:
            m_l = (m - 1) % M
            m_r = (m + 1) % M
        U_l = cell_avg[:, m_l]
        U_r = cell_avg[:, m_r]

        rho_l, rhou_l, E_l = U_l
        rho_r, rhou_r, E_r = U_r

        # Enforce positivity for averaging
        rho_l = max(rho_l, 1e-14)
        rho_r = max(rho_r, 1e-14)
        u_l = rhou_l / rho_l
        u_r = rhou_r / rho_r
        p_l = max((gamma - 1.0) * (E_l - 0.5 * rhou_l * u_l), 1e-14)
        p_r = max((gamma - 1.0) * (E_r - 0.5 * rhou_r * u_r), 1e-14)
        H_l = (E_l + p_l) / rho_l
        H_r = (E_r + p_r) / rho_r

        sqrt_rho_l = np.sqrt(rho_l)
        sqrt_rho_r = np.sqrt(rho_r)
        denom = sqrt_rho_l + sqrt_rho_r

        u_avg = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) / denom
        H_avg = (sqrt_rho_l * H_l + sqrt_rho_r * H_r) / denom
        rho_avg = sqrt_rho_l * sqrt_rho_r

        c2 = max((gamma - 1.0) * (H_avg - 0.5 * u_avg**2), 1e-14)
        c_avg = np.sqrt(c2)

        R, L = _roe_eigenvectors_euler(rho_avg, u_avg, H_avg, c_avg, gamma)

        # Current element nodal values and average (shape (3, N+1) and (3,))
        u_elem = U[:, m, :]           # (3, N+1)
        u_mean = cell_avg[:, m]       # (3,)

        # Rotate to characteristic space: w = L @ u
        # For the nodal values: w_nodes[i, :] = L[i, :] @ u_elem
        w_elem = L @ u_elem           # (3, N+1)
        w_mean = L @ u_mean           # (3,)

        # Neighbour means in characteristic space
        w_avg_l = L @ avg_L[:, m]
        w_avg_r = L @ avg_R[:, m]

        # Apply scalar TVB limiter to each characteristic variable
        thr = tvb_threshold[m]
        w_limited = np.empty_like(w_elem)
        for k in range(3):
            ls = w_mean[k] - w_elem[k, 0]
            rs = w_elem[k, N] - w_mean[k]
            fwd = w_avg_r[k] - w_mean[k]
            bwd = w_mean[k] - w_avg_l[k]
            ll = tvb_minmod(ls, fwd, bwd, thr)
            lr = tvb_minmod(rs, fwd, bwd, thr)
            lslope = 0.5 * (ll + lr)
            w_limited[k] = w_mean[k] + lslope * r

        # Rotate back to physical space
        limited[:, m, :] = R @ w_limited

    return limited
