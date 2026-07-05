"""
1D inviscid Euler equations solver using the Discontinuous Galerkin method.

Solves the 1D Euler equations:

    U_t + F(U)_x = 0

where U = [rho, rho*u, E] and F(U) = [rho*u, rho*u^2 + p, u*(E+p)],
p = (gamma-1)*(E - 0.5*rho*u^2).

The solution is stored as an array of shape (3, M, N+1), where index
[v, m, i] holds the value of conserved variable v at the i-th GLL node of
element m.

Public API
----------
build_mesh          -- (re-exported from mesh1d)
build_operators     -- (re-exported from mesh1d)
build_connectivity  -- (re-exported from mesh1d)
euler_pressure           -- pressure from conserved variables
euler_physical_flux      -- physical flux vector F(U)
euler_wave_speeds        -- min/max wave speeds for a state
euler_lax_friedrichs_flux -- Local Lax-Friedrichs (Rusanov) flux
euler_roe_flux            -- Roe linearised flux with Harten entropy fix
euler_hllc_flux           -- HLLC three-wave approximate Riemann flux
sod_ic              -- Sod shock-tube initial conditions
sod_exact_solution  -- exact five-region solution to the Sod problem
euler_rhs           -- DG right-hand side for the Euler equations
limit_euler         -- component-wise TVB minmod limiter for Euler variables
euler_cfl_dt        -- CFL-based time-step estimate
rk4_step_euler      -- one classical RK4 advance with limiting
simulate_euler      -- full time-integration driver
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from mesh1d import build_mesh, build_operators, build_connectivity  # noqa: F401
from limiters1d import build_limiter_data, limit_scalar, limit_system

# Re-export mesh / operator builders and flux functions for users who import
# everything from here
__all__ = [
    "build_mesh",
    "build_operators",
    "build_connectivity",
    "euler_pressure",
    "euler_physical_flux",
    "euler_wave_speeds",
    "euler_lax_friedrichs_flux",
    "euler_roe_flux",
    "euler_hllc_flux",
    "sod_ic",
    "sod_exact_solution",
    "euler_rhs",
    "limit_euler",
    "euler_cfl_dt",
    "rk4_step_euler",
    "simulate_euler",
]


# ---------------------------------------------------------------------------
# Euler flux functions
# ---------------------------------------------------------------------------

_PRESSURE_FLOOR = 1e-14
_DENSITY_FLOOR = 1e-14


def euler_pressure(U, gamma):
    """Compute pressure from Euler conserved variables.

    Parameters
    ----------
    U : ndarray, shape (3, ...)
        Conserved variables [rho, rho*u, E] along axis 0.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    p : ndarray, shape (...)
        Thermodynamic pressure, floored at a small positive value.
    """
    rho = np.maximum(U[0], _DENSITY_FLOOR)
    rhou = U[1]
    E = U[2]
    p = (gamma - 1.0) * (E - 0.5 * rhou**2 / rho)
    return np.maximum(p, _PRESSURE_FLOOR)


def euler_physical_flux(U, gamma):
    """Compute the physical Euler flux vector F(U).

    For U = [rho, rho*u, E], the flux is:
        F = [rho*u,  rho*u^2 + p,  u*(E + p)]

    Parameters
    ----------
    U : ndarray, shape (3, ...)
        Conserved variables along axis 0.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    F : ndarray, shape (3, ...)
        Physical flux vector.
    """
    rho = np.maximum(U[0], _DENSITY_FLOOR)
    rhou = U[1]
    E = U[2]
    u = rhou / rho
    p = euler_pressure(U, gamma)
    return np.array([
        rhou,
        rhou * u + p,
        u * (E + p),
    ])


def euler_wave_speeds(U, gamma):
    """Compute the minimum and maximum wave speeds for Euler conserved state U.

    Returns (S_min, S_max) = (u - c, u + c) where c = sqrt(gamma * p / rho).

    Parameters
    ----------
    U : ndarray, shape (3, ...)
        Conserved variables along axis 0.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    S_min : ndarray, shape (...)
    S_max : ndarray, shape (...)
    """
    rho = np.maximum(U[0], _DENSITY_FLOOR)
    u = U[1] / rho
    p = euler_pressure(U, gamma)
    c = np.sqrt(np.maximum(gamma * p / rho, _PRESSURE_FLOOR))
    return u - c, u + c


def euler_lax_friedrichs_flux(U_L, U_R, gamma):
    """Local Lax-Friedrichs (Rusanov) numerical flux for the Euler equations.

    Uses the largest local wave speed to add scalar diffusion:

        F* = 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)

    where alpha = max(|u_L| + c_L, |u_R| + c_R).

    Parameters
    ----------
    U_L, U_R : ndarray, shape (3,) or (3, N)
        Left and right conserved-variable states.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    F_star : ndarray, same shape as U_L
        Numerical flux vector.
    """
    F_L = euler_physical_flux(U_L, gamma)
    F_R = euler_physical_flux(U_R, gamma)

    sL_min, sL_max = euler_wave_speeds(U_L, gamma)
    sR_min, sR_max = euler_wave_speeds(U_R, gamma)
    alpha = np.maximum(np.abs(sL_min), np.maximum(np.abs(sL_max),
                       np.maximum(np.abs(sR_min), np.abs(sR_max))))

    return 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)


def euler_roe_flux(U_L, U_R, gamma):
    """Roe linearised approximate Riemann flux for the 1D Euler equations.

    Computes Roe-averaged state (density-weighted velocity and total enthalpy),
    constructs eigendecomposition of the Roe matrix, and applies Harten's
    entropy fix to avoid expansion shocks near sonic points.

    Parameters
    ----------
    U_L, U_R : ndarray, shape (3,) or (3, N)
        Left and right conserved-variable states.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    F_star : ndarray, same shape as U_L
        Numerical flux vector.
    """
    # Handle both 1-D (shape (3,)) and 2-D (shape (3, N)) inputs uniformly
    scalar_input = U_L.ndim == 1
    if scalar_input:
        U_L = U_L[:, np.newaxis]
        U_R = U_R[:, np.newaxis]

    F_L = euler_physical_flux(U_L, gamma)
    F_R = euler_physical_flux(U_R, gamma)

    rho_L = np.maximum(U_L[0], _DENSITY_FLOOR)
    rho_R = np.maximum(U_R[0], _DENSITY_FLOOR)
    u_L = U_L[1] / rho_L
    u_R = U_R[1] / rho_R
    p_L = euler_pressure(U_L, gamma)
    p_R = euler_pressure(U_R, gamma)
    H_L = (U_L[2] + p_L) / rho_L
    H_R = (U_R[2] + p_R) / rho_R

    sqrt_L = np.sqrt(rho_L)
    sqrt_R = np.sqrt(rho_R)
    denom = sqrt_L + sqrt_R

    u_roe = (sqrt_L * u_L + sqrt_R * u_R) / denom
    H_roe = (sqrt_L * H_L + sqrt_R * H_R) / denom

    c2_roe = np.maximum((gamma - 1.0) * (H_roe - 0.5 * u_roe**2), _PRESSURE_FLOOR)
    c_roe = np.sqrt(c2_roe)

    # Wave speeds
    lam1 = u_roe - c_roe
    lam2 = u_roe
    lam3 = u_roe + c_roe

    # Harten's entropy fix: broaden near-zero eigenvalues
    eps = np.maximum(0.1 * c_roe, 1e-14)

    def _fix(lam, lam_L, lam_R):
        delta = np.maximum(np.abs(lam_R - lam_L), eps)
        return np.where(np.abs(lam) < delta, (lam**2 + delta**2) / (2.0 * delta), np.abs(lam))

    s_L1, s_L3 = euler_wave_speeds(U_L, gamma)
    s_R1, s_R3 = euler_wave_speeds(U_R, gamma)
    abs_lam1 = _fix(lam1, s_L1, s_R1)
    abs_lam2 = np.abs(lam2)
    abs_lam3 = _fix(lam3, s_L3, s_R3)

    # Wave strengths (project dU onto left eigenvectors)
    rho_roe = sqrt_L * sqrt_R
    du = u_R - u_L
    dp = p_R - p_L

    a1 = (dp - rho_roe * c_roe * du) / (2.0 * c_roe**2)
    a3 = (dp + rho_roe * c_roe * du) / (2.0 * c_roe**2)
    a2 = U_R[0] - U_L[0] - dp / c_roe**2

    # Dissipation = sum_k |lam_k| * a_k * r_k  (right eigenvectors)
    dissipation = np.zeros_like(U_L)
    # Wave 1: r = [1, u_roe - c_roe, H_roe - u_roe*c_roe]
    dissipation[0] += abs_lam1 * a1
    dissipation[1] += abs_lam1 * a1 * (u_roe - c_roe)
    dissipation[2] += abs_lam1 * a1 * (H_roe - u_roe * c_roe)
    # Wave 2: r = [1, u_roe, 0.5*u_roe^2]
    dissipation[0] += abs_lam2 * a2
    dissipation[1] += abs_lam2 * a2 * u_roe
    dissipation[2] += abs_lam2 * a2 * 0.5 * u_roe**2
    # Wave 3: r = [1, u_roe + c_roe, H_roe + u_roe*c_roe]
    dissipation[0] += abs_lam3 * a3
    dissipation[1] += abs_lam3 * a3 * (u_roe + c_roe)
    dissipation[2] += abs_lam3 * a3 * (H_roe + u_roe * c_roe)

    F_star = 0.5 * (F_L + F_R) - 0.5 * dissipation

    if scalar_input:
        return F_star[:, 0]
    return F_star


def euler_hllc_flux(U_L, U_R, gamma):
    """HLLC approximate Riemann flux for the 1D Euler equations.

    Uses the three-wave HLLC structure to correctly resolve the contact
    discontinuity.  Wave speed estimates follow the Einfeldt (PVRS) approach.

    Parameters
    ----------
    U_L, U_R : ndarray, shape (3,) or (3, N)
        Left and right conserved-variable states.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    F_star : ndarray, same shape as U_L
        Numerical flux vector.
    """
    scalar_input = U_L.ndim == 1
    if scalar_input:
        U_L = U_L[:, np.newaxis]
        U_R = U_R[:, np.newaxis]

    F_L = euler_physical_flux(U_L, gamma)
    F_R = euler_physical_flux(U_R, gamma)

    rho_L = np.maximum(U_L[0], _DENSITY_FLOOR)
    rho_R = np.maximum(U_R[0], _DENSITY_FLOOR)
    u_L = U_L[1] / rho_L
    u_R = U_R[1] / rho_R
    p_L = euler_pressure(U_L, gamma)
    p_R = euler_pressure(U_R, gamma)
    E_L = U_L[2]
    E_R = U_R[2]

    c_L = np.sqrt(np.maximum(gamma * p_L / rho_L, _PRESSURE_FLOOR))
    c_R = np.sqrt(np.maximum(gamma * p_R / rho_R, _PRESSURE_FLOOR))

    # Einfeldt wave speed estimates
    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)

    # Contact wave speed (S*)
    num = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
    den = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    S_star = num / np.where(np.abs(den) > 1e-14, den, 1e-14)

    def _hllc_state(rho, u, E, p, S, U_side):
        """Build the HLLC intermediate state U*_K."""
        factor = rho * (S - u) / np.where(np.abs(S - S_star) > 1e-14, S - S_star, 1e-14)
        U_star = np.empty_like(U_side)
        U_star[0] = factor
        U_star[1] = factor * S_star
        U_star[2] = factor * (E / rho + (S_star - u) * (S_star + p / (rho * (S - u))))
        return U_star

    U_L_star = _hllc_state(rho_L, u_L, E_L, p_L, S_L, U_L)
    U_R_star = _hllc_state(rho_R, u_R, E_R, p_R, S_R, U_R)

    F_star = np.where(
        S_L[np.newaxis, :] >= 0,
        F_L,
        np.where(
            S_star[np.newaxis, :] >= 0,
            F_L + S_L * (U_L_star - U_L),
            np.where(
                S_R[np.newaxis, :] >= 0,
                F_R + S_R * (U_R_star - U_R),
                F_R,
            ),
        ),
    )

    if scalar_input:
        return F_star[:, 0]
    return F_star


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def sod_ic(x, gamma=1.4):
    """Return Sod shock-tube initial conditions in conserved-variable form.

    The initial state has a discontinuity at x = 0.5 (works for x in [0, 1]):

        Left  (x < 0.5): rho=1.0, u=0.0, p=1.0
        Right (x > 0.5): rho=0.125, u=0.0, p=0.1

    Parameters
    ----------
    x : ndarray, shape (M, N+1)
        Physical node coordinates.
    gamma : float, optional
        Ratio of specific heats (default 1.4).

    Returns
    -------
    U : ndarray, shape (3, M, N+1)
        Conserved variables [rho, rho*u, E] at each node.
    """
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.where(x < 0.5, 1.0, 0.1)
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return np.array([rho, rho * u, E])


def sod_exact_solution(x, t, gamma=1.4):
    """Compute the exact Sod shock-tube solution at time t.

    Solves the five-region structure:
        Region 1: left unperturbed state
        Region 2: left rarefaction fan
        Region 3: contact-discontinuity plateau (left side)
        Region 4: contact-discontinuity plateau (right side, same u and p)
        Region 5: right shock
        Region 6: right unperturbed state

    The solution is for the Sod initial data on an infinite domain with
    the discontinuity initially at x = 0.5.

    Parameters
    ----------
    x : ndarray
        Evaluation points.
    t : float
        Time at which to evaluate (must be > 0).
    gamma : float, optional
        Ratio of specific heats (default 1.4).

    Returns
    -------
    rho : ndarray
    u   : ndarray
    p   : ndarray
    """
    x = np.asarray(x, dtype=float)

    # Initial states
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1

    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)

    g1 = gamma - 1.0
    g2 = (gamma + 1.0) / (2.0 * gamma)
    g3 = g1 / (2.0 * gamma)
    g4 = 2.0 * gamma / g1
    g5 = 2.0 / g1
    g6 = 2.0 / (gamma + 1.0)
    g7 = g1 / (gamma + 1.0)

    # Solve for p_star using the Rankine-Hugoniot / isentropic relations
    def _f(p_star):
        # Left wave: rarefaction (p_star < p_L assumed for Sod)
        if p_star <= p_L:
            f_L = g5 * c_L * ((p_star / p_L) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
            f_L_deriv = (1.0 / (rho_L * c_L)) * (p_star / p_L) ** (-(gamma + 1.0) / (2.0 * gamma))
        else:
            A_L = 2.0 / ((gamma + 1.0) * rho_L)
            B_L = g1 / (gamma + 1.0) * p_L
            sqrtL = np.sqrt(A_L / (p_star + B_L))
            f_L = (p_star - p_L) * sqrtL
            f_L_deriv = sqrtL * (1.0 - (p_star - p_L) / (2.0 * (p_star + B_L)))

        # Right wave: shock
        A_R = 2.0 / ((gamma + 1.0) * rho_R)
        B_R = g1 / (gamma + 1.0) * p_R
        sqrtR = np.sqrt(A_R / (p_star + B_R))
        f_R = (p_star - p_R) * sqrtR
        f_R_deriv = sqrtR * (1.0 - (p_star - p_R) / (2.0 * (p_star + B_R)))

        return f_L + f_R + (u_R - u_L)

    p_star = brentq(_f, 1e-8, max(p_L, p_R) * 100.0, xtol=1e-12, rtol=1e-12)

    # Velocity in the star region (constant across contact)
    A_R = 2.0 / ((gamma + 1.0) * rho_R)
    B_R = g1 / (gamma + 1.0) * p_R
    sqrtR = np.sqrt(A_R / (p_star + B_R))
    f_R_val = (p_star - p_R) * sqrtR

    # Left rarefaction contribution to velocity
    f_L_val = g5 * c_L * ((p_star / p_L) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R_val - f_L_val)

    # Density in star regions
    rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)
    rho_star_R = rho_R * (p_star + B_R) / (p_R + (g1 / (gamma + 1.0)) * p_star)

    c_star_L = np.sqrt(gamma * p_star / rho_star_L)

    # Wave positions (relative to original discontinuity at x=0.5)
    x0 = 0.5

    # Left rarefaction: head and tail speeds
    S_HL = u_L - c_L                    # head of rarefaction (leftmost)
    S_TL = u_star - c_star_L            # tail of rarefaction (rightmost)

    # Right shock speed
    S_R_shock = u_R + c_R * np.sqrt(g2 * (p_star / p_R) + g3)

    # Contact speed
    S_contact = u_star

    xi = (x - x0) / (t + 1e-300)  # similarity variable (relative to x0)

    rho_out = np.empty_like(x)
    u_out = np.empty_like(x)
    p_out = np.empty_like(x)

    # Region 1: left unperturbed (xi < S_HL)
    mask1 = xi < S_HL
    rho_out[mask1] = rho_L
    u_out[mask1] = u_L
    p_out[mask1] = p_L

    # Region 2: rarefaction fan (S_HL <= xi <= S_TL)
    mask2 = (~mask1) & (xi <= S_TL)
    xi2 = xi[mask2]
    u_fan = 2.0 / (gamma + 1.0) * (c_L + xi2)
    u_out[mask2] = u_fan
    c_fan = c_L - g1 / 2.0 * (u_fan - u_L)  # from left Riemann invariant
    p_out[mask2] = p_L * (c_fan / c_L) ** (2.0 * gamma / g1)
    rho_out[mask2] = rho_L * (c_fan / c_L) ** (2.0 / g1)

    # Region 3: left star region (S_TL < xi < S_contact)
    mask3 = (~mask1) & (~mask2) & (xi < S_contact)
    rho_out[mask3] = rho_star_L
    u_out[mask3] = u_star
    p_out[mask3] = p_star

    # Region 4: right star region (S_contact <= xi < S_R_shock)
    mask4 = (xi >= S_contact) & (xi < S_R_shock)
    rho_out[mask4] = rho_star_R
    u_out[mask4] = u_star
    p_out[mask4] = p_star

    # Region 5: right unperturbed (xi >= S_R_shock)
    mask5 = xi >= S_R_shock
    rho_out[mask5] = rho_R
    u_out[mask5] = u_R
    p_out[mask5] = p_R

    return rho_out, u_out, p_out


# ---------------------------------------------------------------------------
# DG operators (cached, matching burgers1d pattern)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _build_weak_form_operators_euler(N):
    """Return exact reference-element weak-form operators for degree N."""
    from dgm1d import legendreGL, Vandermonde1D, GradVandermonde1D

    r = legendreGL(N)
    V = Vandermonde1D(N, r)
    V_inv = np.linalg.solve(V, np.eye(V.shape[0]))

    # For Euler, flux is degree 2N+1 (rho*u^2 quadratic in u); use 2N+3 points
    q_order = max(2 * N + 3, 8)
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

def euler_rhs(U, D, LIFT, J, EToE, EToF, gamma, flux_fn=None, bc_type="periodic"):
    """Compute the DG right-hand side for the inviscid 1D Euler equations.

    Uses a weak-form discretisation.  For each conserved variable v:

        J_k * M * dU_k^v/dt = S^T F^v(U_k) - [F*_left^v, 0, ..., 0, F*_right^v]

    Volume integrals use over-integrated Gauss quadrature to avoid aliasing.
    Interface fluxes use the supplied approximate Riemann solver ``flux_fn``.

    Parameters
    ----------
    U : ndarray, shape (3, M, N+1)
        Current conserved-variable nodal values.
    D : ndarray, shape (N+1, N+1)
        Reference-element differentiation matrix (used for shape validation).
    LIFT : ndarray, shape (N+1, 2)
        Reference-element lift operator (used for shape validation).
    J : ndarray, shape (M,)
        Per-element Jacobians.
    EToE : ndarray, shape (M, 2), int
        Element-to-element connectivity.
    EToF : ndarray, shape (M, 2), int
        Element-to-face connectivity.
    gamma : float
        Ratio of specific heats.
    flux_fn : callable, optional
        Numerical flux function with signature flux_fn(U_L, U_R, gamma) -> F*.
        Defaults to euler_roe_flux.
    bc_type : {'periodic', 'transmissive'}
        Boundary condition type.  'periodic' wraps around as in the Burgers
        solver; 'transmissive' (zero-gradient) extrapolates the boundary state.

    Returns
    -------
    rhs : ndarray, shape (3, M, N+1)
    """
    if flux_fn is None:
        flux_fn = euler_roe_flux

    nvar, M, Np1 = U.shape
    N = Np1 - 1

    if D.shape[0] != Np1 or LIFT.shape != (Np1, 2):
        raise ValueError("D and LIFT must match the element polynomial order")

    inv_mass, stiffness = _build_weak_form_operators_euler(N)

    # Volume term: for each element compute F(u) @ stiffness
    # F(U) shape: (3, M, N+1) — physical flux at each node
    F_vol = euler_physical_flux(U.reshape(3, -1), gamma).reshape(3, M, N+1)

    # Volume residual: (3, M, N+1)
    # vol_res[v, m, k] = sum_n F_vol[v,m,n] * stiffness[n,k]  (matches Burgers flux @ stiffness)
    vol_res = F_vol @ stiffness

    # Interface fluxes
    left_neigh = EToE[:, 0]    # shape (M,)
    right_neigh = EToE[:, 1]   # shape (M,)

    if bc_type == "transmissive":
        # Override boundary elements: ghost = boundary value (zero gradient)
        left_neigh = left_neigh.copy()
        right_neigh = right_neigh.copy()
        # No actual modification needed — we override the flux at domain boundaries below

    # Left interface of element m: exterior = right face of left_neigh
    U_ext_left = U[:, left_neigh, N]    # (3, M)
    U_int_left = U[:, :, 0]             # (3, M)

    # Right interface of element m: interior = right face, exterior = left face of right_neigh
    U_int_right = U[:, :, N]            # (3, M)
    U_ext_right = U[:, right_neigh, 0]  # (3, M)

    if bc_type == "transmissive":
        # For element 0: left ghost = element 0's own left boundary value
        U_ext_left[:, 0] = U_int_left[:, 0]
        # For element M-1: right ghost = element M-1's own right boundary value
        U_ext_right[:, M - 1] = U_int_right[:, M - 1]

    F_left = flux_fn(U_ext_left, U_int_left, gamma)    # (3, M)
    F_right = flux_fn(U_int_right, U_ext_right, gamma) # (3, M)

    # Boundary flux contributions: same sign convention as Burgers
    # Left face outward normal is -1, right face outward normal is +1
    boundary = np.zeros_like(U)
    boundary[:, :, 0] = -F_left     # (3, M)
    boundary[:, :, N] = F_right     # (3, M)

    # Residual in reference element: (3, M, N+1)
    res = vol_res - boundary

    # Apply inverse mass matrix and Jacobian scaling
    # rhs[v, m, :] = (res[v, m, :] @ inv_mass.T) / J[m]   (matches Burgers convention)
    rhs = (res @ inv_mass.T) / J[np.newaxis, :, np.newaxis]
    return rhs


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------

def limit_euler(U, J, gamma=1.4, tvb_constant=50.0, bc_type="periodic"):
    """Apply a TVB minmod limiter to the Euler conserved variables.

    Uses component-wise limiting: applies the scalar TVB limiter independently
    to each of the three conserved variables (rho, rho*u, E).  This is more
    robust near discontinuities than characteristic-variable limiting because it
    does not require computing Roe eigenvectors from potentially unphysical
    intermediate RK states.

    For characteristic-variable limiting, use limiters1d.limit_system directly.

    Parameters
    ----------
    U : ndarray, shape (3, M, N+1)
        Conserved-variable nodal values.
    J : array_like, shape (M,)
        Per-element Jacobians.
    gamma : float, optional
        Ratio of specific heats (unused; kept for API consistency).
    tvb_constant : float, optional
        TVB constant (default 50).
    bc_type : {'periodic', 'transmissive'}, optional
        Boundary condition type (default 'periodic').  For 'transmissive',
        ghost cell averages at the domain boundaries equal the boundary
        element averages so the limiter does not wrap across domain edges.

    Returns
    -------
    ndarray, shape (3, M, N+1)
        Limited nodal values.
    """
    return np.stack([limit_scalar(U[v], J, tvb_constant, bc_type) for v in range(3)])


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------

def euler_cfl_dt(U, h, N, cfl, gamma):
    """Compute the CFL-limited time step for the Euler equations.

    dt = cfl * h / (max_wave_speed * (N+1)^2)

    where max_wave_speed = max(|u| + c) over all nodes.

    Parameters
    ----------
    U : ndarray, shape (3, M, N+1)
        Current conserved-variable nodal values.
    h : float
        Uniform element width (physical).
    N : int
        Polynomial order.
    cfl : float
        CFL number.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    dt : float
    """
    U_flat = U.reshape(3, -1)
    _, S_max = euler_wave_speeds(U_flat, gamma)
    S_min, _ = euler_wave_speeds(U_flat, gamma)
    max_speed = float(np.max(np.maximum(np.abs(S_min), np.abs(S_max))))
    max_speed = max(max_speed, 1e-14)
    return cfl * h / (max_speed * (N + 1) ** 2)


def rk4_step_euler(U, dt, D, LIFT, J, EToE, EToF, gamma, flux_fn=None, bc_type="periodic"):
    """Advance the Euler solution by one classical 4th-order Runge-Kutta step.

    The TVB limiter is applied after each stage to suppress spurious oscillations.

    Parameters
    ----------
    U : ndarray, shape (3, M, N+1)
    dt : float
    D, LIFT, J, EToE, EToF : see euler_rhs
    gamma : float
    flux_fn : callable, optional
    bc_type : str, optional

    Returns
    -------
    U_new : ndarray, shape (3, M, N+1)
    """
    if flux_fn is None:
        flux_fn = euler_roe_flux

    def rhs(V):
        return euler_rhs(V, D, LIFT, J, EToE, EToF, gamma, flux_fn, bc_type)

    def limit(V):
        return limit_euler(V, J, gamma, bc_type=bc_type)

    U0 = limit(U)
    k1 = rhs(U0)
    U1 = limit(U0 + 0.5 * dt * k1)
    k2 = rhs(U1)
    U2 = limit(U0 + 0.5 * dt * k2)
    k3 = rhs(U2)
    U3 = limit(U0 + dt * k3)
    k4 = rhs(U3)
    return limit(U0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def simulate_euler(M, N, t_final, gamma=1.4, cfl=0.5, x_left=0.0, x_right=1.0,
                   U_init=None, flux_fn=None, bc_type="transmissive"):
    """Run the 1D inviscid Euler equations simulation.

    Parameters
    ----------
    M : int
        Number of elements.
    N : int
        Polynomial order; each element has N+1 GLL nodes.
    t_final : float
        Final simulation time.
    gamma : float, optional
        Ratio of specific heats (default 1.4).
    cfl : float, optional
        CFL number (default 0.5).
    x_left, x_right : float, optional
        Domain boundaries (default [0, 1]).
    U_init : callable or None, optional
        Initial condition.  Receives the (M, N+1) node coordinate array and
        must return a (3, M, N+1) array of conserved variables.
        Defaults to the Sod shock-tube initial condition.
    flux_fn : callable or None, optional
        Numerical flux function; defaults to euler_roe_flux.
    bc_type : {'transmissive', 'periodic'}, optional
        Boundary condition type (default 'transmissive').

    Returns
    -------
    x : ndarray, shape (M, N+1)
        Node coordinates.
    U : ndarray, shape (3, M, N+1)
        Solution at time t.
    t : float
        Actual final time reached.
    """
    if flux_fn is None:
        flux_fn = euler_roe_flux

    x, r = build_mesh(M, N, x_left, x_right)
    _, D, LIFT = build_operators(N, r)
    EToE, EToF = build_connectivity(M)

    h = (x_right - x_left) / M
    J = np.full(M, 0.5 * h)

    if U_init is None:
        U_init = lambda xv: sod_ic(xv, gamma)
    if callable(U_init):
        U = U_init(x).astype(float)
    else:
        U = np.asarray(U_init, dtype=float)

    t = 0.0
    while t < t_final:
        dt = euler_cfl_dt(U, h, N, cfl, gamma)
        dt = min(dt, t_final - t)
        if dt <= 0.0:
            break
        U = rk4_step_euler(U, dt, D, LIFT, J, EToE, EToF, gamma, flux_fn, bc_type)
        t += dt

    return x, U, t
