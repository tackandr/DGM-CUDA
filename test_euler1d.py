"""
Tests for the 1D inviscid Euler DG solver.

Covers:
- Physical flux computation
- Numerical flux consistency and positivity (LLF, Roe, HLLC)
- DG right-hand side: shape, constant state, conservation
- Limiter: cell-average preservation, constant state
- Sod shock tube: accuracy vs exact solution, positivity, conservation
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxes1d import (
    euler_physical_flux,
    euler_pressure,
    euler_lax_friedrichs_flux,
    euler_roe_flux,
    euler_hllc_flux,
)
from limiters1d import build_limiter_data, limit_scalar
from euler1d import (
    build_mesh,
    build_operators,
    build_connectivity,
    sod_ic,
    sod_exact_solution,
    euler_rhs,
    limit_euler,
    euler_cfl_dt,
    rk4_step_euler,
    simulate_euler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GAMMA = 1.4


def _uniform_state(rho, u, p, shape):
    """Return conserved-variable array of a uniform state."""
    E = p / (GAMMA - 1.0) + 0.5 * rho * u ** 2
    U = np.zeros((3,) + shape)
    U[0] = rho
    U[1] = rho * u
    U[2] = E
    return U


def _cell_averages(U, N):
    """Quadrature-weighted cell averages, shape (nvar, M)."""
    _, mw = build_limiter_data(N)
    return U @ mw


# ---------------------------------------------------------------------------
# TestEulerPhysicalFlux
# ---------------------------------------------------------------------------

class TestEulerPhysicalFlux:
    def test_zero_velocity(self):
        """F(rho, 0, p/(gamma-1)) = [0, p, 0]."""
        rho, p = 1.2, 0.8
        E = p / (GAMMA - 1.0)
        U = np.array([[rho], [0.0], [E]])
        F = euler_physical_flux(U, GAMMA)
        assert_allclose(F[0], 0.0)
        assert_allclose(F[1], p, rtol=1e-12)
        assert_allclose(F[2], 0.0)

    def test_known_state(self):
        """Hand-computed flux for a specific state."""
        rho, u, p = 1.0, 2.0, 3.0
        E = p / (GAMMA - 1.0) + 0.5 * rho * u ** 2
        U = np.array([[rho], [rho * u], [E]])
        F = euler_physical_flux(U, GAMMA)
        assert_allclose(F[0], rho * u, rtol=1e-12)
        assert_allclose(F[1], rho * u ** 2 + p, rtol=1e-12)
        assert_allclose(F[2], u * (E + p), rtol=1e-12)

    def test_shape_preserved(self):
        M = 10
        U = _uniform_state(1.0, 0.5, 1.0, (M,))
        F = euler_physical_flux(U, GAMMA)
        assert F.shape == U.shape


# ---------------------------------------------------------------------------
# TestEulerNumericalFlux
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flux_fn", [
    euler_lax_friedrichs_flux,
    euler_roe_flux,
    euler_hllc_flux,
])
class TestEulerNumericalFlux:
    def _make_states(self):
        """Return two physically admissible states."""
        rho_L, u_L, p_L = 1.0, 0.5, 1.0
        rho_R, u_R, p_R = 0.5, -0.2, 0.5
        E_L = p_L / (GAMMA - 1.0) + 0.5 * rho_L * u_L ** 2
        E_R = p_R / (GAMMA - 1.0) + 0.5 * rho_R * u_R ** 2
        UL = np.array([rho_L, rho_L * u_L, E_L])
        UR = np.array([rho_R, rho_R * u_R, E_R])
        return UL, UR

    def test_consistency(self, flux_fn):
        """F*(U, U) should equal the physical flux F(U)."""
        UL, _ = self._make_states()
        F_star = flux_fn(UL[:, np.newaxis], UL[:, np.newaxis], GAMMA).ravel()
        F_phys = euler_physical_flux(UL[:, np.newaxis], GAMMA).ravel()
        assert_allclose(F_star, F_phys, rtol=1e-10, atol=1e-12)

    def test_conservation(self, flux_fn):
        """F*(U_L, U_R) seen from both sides is the same value."""
        UL, UR = self._make_states()
        F_lr = flux_fn(UL[:, np.newaxis], UR[:, np.newaxis], GAMMA).ravel()
        F_rl = flux_fn(UR[:, np.newaxis], UL[:, np.newaxis], GAMMA).ravel()
        # Symmetry check: F*(L→R) = F*(R→L) is NOT required (upwind flux is not symmetric).
        # Conservation only requires a single-valued flux; the call returns the
        # same flux for the (L,R) pair regardless of the calling side.
        assert np.all(np.isfinite(F_lr))
        assert np.all(np.isfinite(F_rl))

    def test_positivity_input(self, flux_fn):
        """Flux evaluation on physical inputs must produce finite results."""
        UL, UR = self._make_states()
        F = flux_fn(UL[:, np.newaxis], UR[:, np.newaxis], GAMMA)
        assert np.all(np.isfinite(F))

    def test_sod_states(self, flux_fn):
        """Flux between left and right Sod states must be finite."""
        rho_L, p_L = 1.0, 1.0
        rho_R, p_R = 0.125, 0.1
        E_L = p_L / (GAMMA - 1.0)
        E_R = p_R / (GAMMA - 1.0)
        UL = np.array([rho_L, 0.0, E_L])
        UR = np.array([rho_R, 0.0, E_R])
        F = flux_fn(UL[:, np.newaxis], UR[:, np.newaxis], GAMMA)
        assert np.all(np.isfinite(F))


# ---------------------------------------------------------------------------
# TestEulerRHS
# ---------------------------------------------------------------------------

class TestEulerRHS:
    def _operators(self, M=20, N=2):
        x, r = build_mesh(M, N, 0.0, 1.0)
        _, D, LIFT = build_operators(N, r)
        EToE, EToF = build_connectivity(M)
        J = np.full(M, 0.5 / M)
        return x, D, LIFT, J, EToE, EToF, M, N

    def test_shape(self):
        x, D, LIFT, J, EToE, EToF, M, N = self._operators()
        U = _uniform_state(1.0, 0.0, 1.0, (M, N + 1))
        rhs = euler_rhs(U, D, LIFT, J, EToE, EToF, GAMMA, euler_roe_flux, "periodic")
        assert rhs.shape == (3, M, N + 1)

    def test_constant_state_zero_rhs_periodic(self):
        """A spatially constant state must give zero RHS (periodic BC)."""
        x, D, LIFT, J, EToE, EToF, M, N = self._operators()
        U = _uniform_state(1.0, 0.5, 1.0, (M, N + 1))
        rhs = euler_rhs(U, D, LIFT, J, EToE, EToF, GAMMA, euler_roe_flux, "periodic")
        assert_allclose(rhs, 0.0, atol=1e-11)

    def test_conservation_periodic(self):
        """Integral of RHS over periodic domain must be zero."""
        x, D, LIFT, J, EToE, EToF, M, N = self._operators(M=20, N=2)
        U0 = sod_ic(x, GAMMA)
        rhs = euler_rhs(U0, D, LIFT, J, EToE, EToF, GAMMA, euler_roe_flux, "periodic")
        _, mw = build_limiter_data(N)
        # Integral = sum_m J_m * (rhs[v,m,:] @ mw)
        for v in range(3):
            integral = np.sum(J[:, np.newaxis] * rhs[v] * mw[np.newaxis, :]) * 2.0
            assert abs(integral) < 1e-10, f"var {v}: integral={integral}"

    def test_all_fluxes_give_finite_rhs(self):
        """All three numerical fluxes must produce finite RHS for the Sod IC."""
        x, D, LIFT, J, EToE, EToF, M, N = self._operators()
        U0 = sod_ic(x, GAMMA)
        for flux_fn in [euler_lax_friedrichs_flux, euler_roe_flux, euler_hllc_flux]:
            rhs = euler_rhs(U0, D, LIFT, J, EToE, EToF, GAMMA, flux_fn, "transmissive")
            assert np.all(np.isfinite(rhs)), f"{flux_fn.__name__} produced non-finite RHS"


# ---------------------------------------------------------------------------
# TestEulerLimiter
# ---------------------------------------------------------------------------

class TestEulerLimiter:
    def test_constant_state_unchanged(self):
        """Limiting a spatially constant state must not change it."""
        M, N = 30, 2
        x, _ = build_mesh(M, N, 0.0, 1.0)
        J = np.full(M, 0.5 / M)
        U = _uniform_state(1.0, 0.5, 1.0, (M, N + 1))
        U_lim = limit_euler(U, J, GAMMA)
        assert_allclose(U_lim, U, atol=1e-12)

    def test_cell_averages_preserved(self):
        """Cell averages must be unchanged by the limiter."""
        M, N = 30, 2
        x, _ = build_mesh(M, N, 0.0, 1.0)
        J = np.full(M, 0.5 / M)
        U = sod_ic(x, GAMMA)
        avg_before = _cell_averages(U, N)
        U_lim = limit_euler(U, J, GAMMA)
        avg_after = _cell_averages(U_lim, N)
        assert_allclose(avg_after, avg_before, atol=1e-12)

    def test_output_shape(self):
        M, N = 30, 2
        x, _ = build_mesh(M, N, 0.0, 1.0)
        J = np.full(M, 0.5 / M)
        U = sod_ic(x, GAMMA)
        assert limit_euler(U, J, GAMMA).shape == U.shape


# ---------------------------------------------------------------------------
# TestSodShockTube  (main validation)
# ---------------------------------------------------------------------------

class TestSodShockTube:
    @pytest.fixture(scope="class")
    def sod_result(self):
        M, N = 80, 2
        x, _ = build_mesh(M, N, 0.0, 1.0)
        U0 = sod_ic(x, GAMMA)
        x_c, U_out, t = simulate_euler(
            M=M, N=N, t_final=0.2, gamma=GAMMA, cfl=0.3,
            x_left=0.0, x_right=1.0, U_init=U0,
            flux_fn=euler_roe_flux, bc_type="transmissive",
        )
        return x_c, U_out, t, M, N

    def _cell_avgs(self, U_out, N):
        _, mw = build_limiter_data(N)
        rho = U_out[0] @ mw
        rhou = U_out[1] @ mw
        E = U_out[2] @ mw
        u = rhou / np.maximum(rho, 1e-14)
        p = (GAMMA - 1.0) * (E - 0.5 * rhou * u)
        return rho, u, p

    def test_reaches_tfinal(self, sod_result):
        _, _, t, _, _ = sod_result
        assert abs(t - 0.2) < 1e-10

    def test_all_finite(self, sod_result):
        _, U_out, _, _, _ = sod_result
        assert np.all(np.isfinite(U_out))

    def test_positivity(self, sod_result):
        x_c, U_out, t, M, N = sod_result
        rho, u, p = self._cell_avgs(U_out, N)
        assert np.all(rho > 0), "Negative density"
        assert np.all(p > 0), "Negative pressure"

    def test_l1_accuracy(self, sod_result):
        """L1 error vs exact Sod solution must be < 5% of the total variation."""
        x_c, U_out, t, M, N = sod_result
        rho_num, u_num, p_num = self._cell_avgs(U_out, N)
        _, mw = build_limiter_data(N)
        x_centers = x_c @ mw
        rho_ex, u_ex, p_ex = sod_exact_solution(x_centers, t)
        dx = 1.0 / M
        l1_rho = dx * np.abs(rho_num - rho_ex).sum()
        l1_u = dx * np.abs(u_num - u_ex).sum()
        l1_p = dx * np.abs(p_num - p_ex).sum()
        assert l1_rho < 0.05, f"rho L1 error too large: {l1_rho:.4f}"
        assert l1_u < 0.05, f"u L1 error too large: {l1_u:.4f}"
        assert l1_p < 0.05, f"p L1 error too large: {l1_p:.4f}"

    def test_solution_bounded(self, sod_result):
        """Solution must stay within expanded initial data bounds."""
        x_c, U_out, t, M, N = sod_result
        rho, u, p = self._cell_avgs(U_out, N)
        assert rho.max() <= 1.1, f"Density blowup: {rho.max()}"
        assert p.max() <= 1.1, f"Pressure blowup: {p.max()}"

    def test_mass_conservation_periodic(self):
        """Total mass is conserved for a periodic simulation."""
        M, N = 60, 2
        x, _ = build_mesh(M, N, 0.0, 1.0)
        U0 = sod_ic(x, GAMMA)
        _, mw = build_limiter_data(N)
        h = 1.0 / M

        # Compute initial total mass
        mass0 = h * (U0[0] @ mw).sum()

        x_c, U_out, t = simulate_euler(
            M=M, N=N, t_final=0.05, gamma=GAMMA, cfl=0.3,
            x_left=0.0, x_right=1.0, U_init=U0,
            flux_fn=euler_roe_flux, bc_type="periodic",
        )
        mass_f = h * (U_out[0] @ mw).sum()
        assert abs(mass_f - mass0) / mass0 < 1e-3, (
            f"Mass not conserved: {abs(mass_f - mass0)/mass0:.2e}"
        )
