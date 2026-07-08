"""
2D linear advection solver on triangular DG elements.

Equation:
    u_t + a_x u_x + a_y u_y = 0
"""

from functools import lru_cache

import numpy as np

from dgm2d import Vandermonde2D, simplex_nodes_2d
from mesh2d import setup_2d

# Small fraction of limiter tolerance used to regularize denominator divisions.
EPS_TOL_RATIO = 0.01


def upwind_flux(u_minus, u_plus, an):
    """Scalar upwind numerical flux for normal speed an."""
    return 0.5 * (an * (u_minus + u_plus) - np.abs(an) * (u_plus - u_minus))


def central_flux(u_minus, u_plus, an):
    """Scalar central numerical flux for normal speed an."""
    return 0.5 * an * (u_minus + u_plus)


def advection_rhs(
    u,
    ax,
    ay,
    Dr,
    Ds,
    LIFT,
    rx,
    sx,
    ry,
    sy,
    J,
    nx,
    ny,
    Fscale,
    EToE,
    EToF,
    fmask,
    bc_type="periodic",
    u_bc=None,
    flux_type="upwind",
):
    """Return DG residual for 2D linear advection."""
    K, Np = u.shape
    Nfaces = 3
    Nfp = len(fmask[0])

    ur = u @ Dr.T
    us = u @ Ds.T

    ux = rx * ur + sx * us
    uy = ry * ur + sy * us
    rhs = -(ax * ux + ay * uy)

    face_flux_term = np.zeros((K, Nfaces * Nfp), dtype=float)

    for f in range(Nfaces):
        ids = fmask[f]
        sl = slice(f * Nfp, (f + 1) * Nfp)

        uM = u[:, ids]
        nxf = nx[:, sl]
        nyf = ny[:, sl]
        an = ax * nxf + ay * nyf

        uP = np.empty_like(uM)
        for k in range(K):
            kn = EToE[k, f]
            fn = EToF[k, f]
            if (kn == k and fn == f) and bc_type != "periodic":
                if bc_type == "transmissive":
                    uP[k] = uM[k]
                elif bc_type == "dirichlet":
                    if u_bc is None:
                        raise ValueError("u_bc must be provided for dirichlet bc")
                    uP[k] = u_bc[k, ids]
                else:
                    raise ValueError("Unsupported bc_type")
            else:
                # Opposite orientation across a shared edge for CCW triangles
                uP[k] = u[kn, fmask[fn][::-1]]

        if flux_type == "upwind":
            flux = upwind_flux(uM, uP, an)
        elif flux_type == "central":
            flux = central_flux(uM, uP, an)
        else:
            raise ValueError(
                f"Unsupported flux_type '{flux_type}'. Supported types: 'upwind', 'central'."
            )
        face_flux_term[:, sl] = Fscale[:, sl] * (flux - an * uM)

    rhs -= (LIFT @ face_flux_term.T).T
    return rhs


def rk4_step_advection(u, dt, rhs_fn):
    """Advance one classical RK4 step."""
    k1 = rhs_fn(u)
    k2 = rhs_fn(u + 0.5 * dt * k1)
    k3 = rhs_fn(u + 0.5 * dt * k2)
    k4 = rhs_fn(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _stable_dt(ax, ay, Fscale, N, cfl):
    speed_n = np.sqrt(ax * ax + ay * ay)
    speed_n = max(speed_n, 1e-14)
    scale = np.max(Fscale)
    return cfl / (speed_n * scale * (N + 1) ** 2)


@lru_cache(maxsize=16)
def _triangle_mean_weights(N):
    """Return nodal weights that produce per-element cell averages."""
    # Use an over-integrating quadrature to accurately recover mean weights for
    # curved nodal distributions across practical DG orders.
    q_order = max(2 * N + 4, 12)
    r, s = simplex_nodes_2d(N)
    xg, wg = np.polynomial.legendre.leggauss(q_order)
    yg, vg = np.polynomial.legendre.leggauss(q_order)

    xi = 0.5 * (xg + 1.0)
    wx_ref = 0.5 * wg
    eta_ref = 0.5 * (yg + 1.0)
    wy_ref = 0.5 * vg

    rq = []
    sq = []
    wq = []
    for ix, x in enumerate(xi):
        eta = (1.0 - x) * eta_ref
        wy = (1.0 - x) * wy_ref
        for iy, y in enumerate(eta):
            rq.append(2.0 * x - 1.0)
            sq.append(2.0 * y - 1.0)
            wq.append(wx_ref[ix] * wy[iy] * 4.0)

    rq = np.asarray(rq)
    sq = np.asarray(sq)
    wq = np.asarray(wq)

    V = Vandermonde2D(N, r, s)
    Vq = Vandermonde2D(N, rq, sq)
    interp = Vq @ np.linalg.solve(V, np.eye(V.shape[0]))

    # Reference triangle area is 2.
    return 0.5 * (wq @ interp)


def bound_preserving_limiter_2d(u, EToE, mean_weights, tol=1e-12):
    """Apply a Zhang-Shu-style scaling limiter using neighbor cell-average bounds."""
    cell_average = u @ mean_weights
    neigh_average = cell_average[EToE]
    lower = np.minimum(cell_average, np.min(neigh_average, axis=1))
    upper = np.maximum(cell_average, np.max(neigh_average, axis=1))

    umin = np.min(u, axis=1)
    umax = np.max(u, axis=1)
    theta = np.ones_like(cell_average)
    # Keep the denominator strictly positive while tying epsilon to tol scale.
    eps = max(1e-14, EPS_TOL_RATIO * tol)

    over = umax > upper + tol
    theta[over] = np.minimum(
        theta[over],
        (upper[over] - cell_average[over]) / np.maximum(umax[over] - cell_average[over], eps),
    )
    under = umin < lower - tol
    theta[under] = np.minimum(
        theta[under],
        (cell_average[under] - lower[under]) / np.maximum(cell_average[under] - umin[under], eps),
    )

    theta = np.clip(theta, 0.0, 1.0)
    return cell_average[:, np.newaxis] + theta[:, np.newaxis] * (u - cell_average[:, np.newaxis])


def _phi_minmod(r):
    """Minmod flux-limiter function: φ(r) = max(0, min(1, r)).

    Reduces the nodal deviation to be no larger than the inter-cell difference.
    Equivalent to the Barth–Jespersen bound-preserving limiter.
    """
    return np.maximum(0.0, np.minimum(1.0, r))


def _phi_superbee(r):
    """Superbee flux-limiter function: φ(r) = max(0, max(min(2r,1), min(r,2))).

    Allows deviations up to twice the inter-cell difference before limiting, so it
    is less diffusive than minmod while retaining the TVD property in 1-D and
    reducing oscillations in 2-D.
    """
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))


def tvd_limiter_2d(u, EToE, mean_weights, phi_fn):
    """Apply a TVD slope limiter to a 2D DG field.

    Computes a per-element scaling factor θ ∈ [0, 1] by comparing the maximum
    nodal deviation from the cell average against the largest supportive
    inter-cell average difference (positive or negative sides treated separately).
    The limiter function ``phi_fn`` maps the ratio r = Δ / dev to a scale in
    [0, 1] and controls how aggressively the deviation is reduced.

    Parameters
    ----------
    u : ndarray, shape (K, Np)
        Current nodal values.
    EToE : ndarray, shape (K, 3)
        Element-to-element neighbour connectivity.
    mean_weights : ndarray, shape (Np,)
        Quadrature weights for computing cell averages from nodal values.
    phi_fn : callable(r : ndarray) -> ndarray
        Scalar TVD limiter function (e.g. ``_phi_minmod`` or ``_phi_superbee``).

    Returns
    -------
    ndarray, shape (K, Np)
        Limited nodal values with the same cell averages.
    """
    avg = u @ mean_weights                          # (K,)
    kn_avgs = avg[EToE]                             # (K, 3)
    delta_max = np.max(kn_avgs - avg[:, np.newaxis], axis=1)   # most positive nbr diff
    delta_min = np.min(kn_avgs - avg[:, np.newaxis], axis=1)   # most negative nbr diff

    dev = u - avg[:, np.newaxis]                    # (K, Np) nodal deviations
    dev_max = np.max(dev, axis=1)                   # (K,) largest overshoot
    dev_min = np.min(dev, axis=1)                   # (K,) largest undershoot

    eps = 1e-14

    # Positive-deviation side: compare against the most supportive (largest) nbr diff.
    mask_pos = dev_max > eps
    r_pos = np.where(mask_pos, delta_max / np.maximum(dev_max, eps), 1.0)
    theta_pos = np.where(mask_pos, np.minimum(1.0, phi_fn(r_pos)), 1.0)

    # Negative-deviation side: compare against the most supportive (most negative) nbr diff.
    mask_neg = dev_min < -eps
    r_neg = np.where(mask_neg, delta_min / np.minimum(dev_min, -eps), 1.0)
    theta_neg = np.where(mask_neg, np.minimum(1.0, phi_fn(r_neg)), 1.0)

    theta = np.clip(np.minimum(theta_pos, theta_neg), 0.0, 1.0)
    return avg[:, np.newaxis] + theta[:, np.newaxis] * dev


def simulate_advection_2d(
    N,
    nx,
    ny,
    t_final,
    ax=1.0,
    ay=0.3,
    cfl=0.25,
    periodic=True,
    u_init=None,
    flux_type="upwind",
    limiter_type=None,
):
    """Run a 2D advection simulation on a uniform triangular mesh."""
    data = setup_2d(N, nx, ny, periodic=periodic)

    x = data["x"]
    y = data["y"]
    if u_init is None:
        u_init = lambda xv, yv: np.sin(2 * np.pi * xv) * np.sin(2 * np.pi * yv)

    u = np.asarray(u_init(x, y), dtype=float)

    def rhs(v):
        return advection_rhs(
            v,
            ax,
            ay,
            data["Dr"],
            data["Ds"],
            data["LIFT"],
            data["rx"],
            data["sx"],
            data["ry"],
            data["sy"],
            data["J"],
            data["nx"],
            data["ny"],
            data["Fscale"],
            data["EToE"],
            data["EToF"],
            data["fmask"],
            bc_type="periodic" if periodic else "transmissive",
            flux_type=flux_type,
        )

    t = 0.0
    dt0 = _stable_dt(ax, ay, data["Fscale"], N, cfl)

    _TVD_PHI = {
        "minmod": _phi_minmod,
        "superbee": _phi_superbee,
    }
    _SUPPORTED = ("None", "bound_preserving", "minmod", "superbee")

    if limiter_type is None:
        mean_weights = None
        phi_fn = None
    elif limiter_type == "bound_preserving":
        mean_weights = _triangle_mean_weights(N)
        phi_fn = None
    elif limiter_type in _TVD_PHI:
        mean_weights = _triangle_mean_weights(N)
        phi_fn = _TVD_PHI[limiter_type]
    else:
        raise ValueError(
            f"Unsupported limiter_type '{limiter_type}'. Supported types: {', '.join(_SUPPORTED)}."
        )

    while t < t_final:
        dt = min(dt0, t_final - t)
        if dt <= 0:
            break
        u = rk4_step_advection(u, dt, rhs)
        if limiter_type == "bound_preserving":
            u = bound_preserving_limiter_2d(u, data["EToE"], mean_weights)
        elif phi_fn is not None:
            u = tvd_limiter_2d(u, data["EToE"], mean_weights, phi_fn)
        t += dt

    return data, u, t
