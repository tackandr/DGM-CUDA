"""
Numerical flux library for 1D DG solvers.

Provides Riemann solver–based interface fluxes for both the scalar Burgers
equation and the 1D inviscid Euler equations.

Public API
----------
Burgers
~~~~~~~
burgers_godunov_flux     -- exact Godunov upwind flux for f(u) = u²/2

Euler helpers
~~~~~~~~~~~~~
euler_pressure           -- pressure from conserved variables
euler_physical_flux      -- physical flux vector F(U) for Euler
euler_wave_speeds        -- min/max wave speeds for a state

Euler numerical fluxes
~~~~~~~~~~~~~~~~~~~~~~
euler_lax_friedrichs_flux -- Local Lax-Friedrichs (Rusanov) flux
euler_roe_flux            -- Roe linearised flux with Harten entropy fix
euler_hllc_flux           -- HLLC three-wave approximate Riemann flux
"""

import numpy as np


# ---------------------------------------------------------------------------
# Burgers flux
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


# ---------------------------------------------------------------------------
# Euler helpers
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


# ---------------------------------------------------------------------------
# Euler numerical fluxes
# ---------------------------------------------------------------------------

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
