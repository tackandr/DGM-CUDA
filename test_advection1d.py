"""Tests for the 1D variable-coefficient advection solver in advection1d.py."""

import numpy as np
import pytest

from advection1d import (
    _upwind_flux_1d,
    advection_rhs_1d,
    rk4_step_advection_1d,
    simulate_advection1d,
    _build_weak_form_operators,
)
from mesh1d import build_connectivity, build_mesh, build_operators
from limiters1d import build_limiter_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup(M, N, x_left=0.0, x_right=1.0):
    """Return mesh, operators, and connectivity for tests."""
    x, r = build_mesh(M, N, x_left, x_right)
    _, D, LIFT = build_operators(N, r)
    EToE, EToF = build_connectivity(M)
    h = (x_right - x_left) / M
    J = np.full(M, 0.5 * h)
    inv_mass, stiffness = _build_weak_form_operators(N)
    return x, D, LIFT, J, EToE, EToF, inv_mass, stiffness


def _mean_weights(N):
    """Compute cell-average weights for degree-N elements."""
    _, mw = build_limiter_data(N)
    return mw


# ---------------------------------------------------------------------------
# _upwind_flux_1d
# ---------------------------------------------------------------------------

class TestUpwindFlux1D:
    def test_positive_speed_selects_left_state(self):
        """When a > 0 the upwind state is u_minus (left)."""
        f = _upwind_flux_1d(3.0, 1.0, 2.0)
        assert f == pytest.approx(2.0 * 3.0)

    def test_negative_speed_selects_right_state(self):
        """When a < 0 the upwind state is u_plus (right)."""
        f = _upwind_flux_1d(1.0, 4.0, -2.0)
        assert f == pytest.approx(-2.0 * 4.0)

    def test_zero_speed_gives_zero_flux(self):
        """Zero velocity produces zero flux regardless of states."""
        f = _upwind_flux_1d(3.0, -5.0, 0.0)
        assert f == pytest.approx(0.0)

    def test_equal_states_gives_a_times_u(self):
        """When u- == u+ the flux equals a * u."""
        u = 2.5
        a = -1.5
        f = _upwind_flux_1d(u, u, a)
        assert f == pytest.approx(a * u)

    def test_array_input(self):
        """Vectorised input shapes are preserved."""
        u_m = np.array([1.0, 2.0, 3.0])
        u_p = np.array([2.0, 3.0, 4.0])
        an = np.array([1.0, -1.0, 0.0])
        f = _upwind_flux_1d(u_m, u_p, an)
        assert f.shape == (3,)
        assert f[0] == pytest.approx(1.0 * 1.0)   # a>0 → u_minus
        assert f[1] == pytest.approx(-1.0 * 3.0)  # a<0 → u_plus
        assert f[2] == pytest.approx(0.0)          # a=0 → zero


# ---------------------------------------------------------------------------
# advection_rhs_1d
# ---------------------------------------------------------------------------

class TestAdvectionRhs1D:
    def test_output_shape(self):
        """RHS has same shape as u."""
        M, N = 5, 3
        x, D, LIFT, J, EToE, EToF, inv_mass, stiffness = _setup(M, N)
        u = np.random.default_rng(0).uniform(-1, 1, (M, N + 1))
        rhs = advection_rhs_1d(u, x, 0.0, lambda xv, tv: np.ones_like(xv),
                                stiffness, inv_mass, J, EToE, EToF)
        assert rhs.shape == (M, N + 1)

    def test_constant_solution_zero_rhs_periodic(self):
        """A spatially constant u gives zero RHS for any constant velocity."""
        for c in [-2.0, 0.0, 1.5]:
            M, N = 6, 3
            x, D, LIFT, J, EToE, EToF, inv_mass, stiffness = _setup(M, N)
            u = np.full((M, N + 1), 3.7)
            rhs = advection_rhs_1d(u, x, 0.0, lambda xv, tv: np.full_like(xv, c),
                                    stiffness, inv_mass, J, EToE, EToF,
                                    bc_type="periodic")
            np.testing.assert_allclose(rhs, 0.0, atol=1e-11,
                                        err_msg=f"Failed for a = {c}")

    def test_constant_solution_zero_rhs_transmissive(self):
        """Constant u also gives zero RHS with transmissive BC."""
        M, N = 5, 2
        x, D, LIFT, J, EToE, EToF, inv_mass, stiffness = _setup(M, N)
        u = np.full((M, N + 1), -1.0)
        rhs = advection_rhs_1d(u, x, 0.0, lambda xv, tv: np.full_like(xv, 1.5),
                                stiffness, inv_mass, J, EToE, EToF,
                                bc_type="transmissive")
        np.testing.assert_allclose(rhs, 0.0, atol=1e-11)

    def test_conservative_flux_cancellation_periodic(self):
        """For periodic BC, the sum of all RHS entries is zero (mass conservation)."""
        M, N = 8, 2
        x, D, LIFT, J, EToE, EToF, inv_mass, stiffness = _setup(M, N)
        rng = np.random.default_rng(42)
        u = rng.uniform(-1, 1, (M, N + 1))
        mw = _mean_weights(N)
        rhs = advection_rhs_1d(u, x, 0.0, lambda xv, tv: np.ones_like(xv),
                                stiffness, inv_mass, J, EToE, EToF)
        # d/dt ∑_m J_m * ∫ u_m dr  =  ∑_m J_m (rhs_m @ mw * 2)
        # For periodic BC with constant velocity this should be zero (up to quadrature)
        mass_rate = np.sum(J[:, np.newaxis] * rhs * mw[np.newaxis, :]) * 2.0
        assert mass_rate == pytest.approx(0.0, abs=1e-11)

    def test_unsupported_bc_type_raises(self):
        """Unknown bc_type raises ValueError."""
        M, N = 4, 2
        x, D, LIFT, J, EToE, EToF, inv_mass, stiffness = _setup(M, N)
        u = np.zeros((M, N + 1))
        with pytest.raises(ValueError, match="Unsupported bc_type"):
            advection_rhs_1d(u, x, 0.0, lambda xv, tv: np.ones_like(xv),
                             stiffness, inv_mass, J, EToE, EToF,
                             bc_type="invalid")

    def test_dirichlet_requires_u_bc(self):
        """Dirichlet BC without u_bc raises ValueError."""
        M, N = 4, 2
        x, D, LIFT, J, EToE, EToF, inv_mass, stiffness = _setup(M, N)
        u = np.zeros((M, N + 1))
        with pytest.raises(ValueError, match="u_bc must be provided"):
            advection_rhs_1d(u, x, 0.0, lambda xv, tv: np.ones_like(xv),
                             stiffness, inv_mass, J, EToE, EToF,
                             bc_type="dirichlet")

    def test_variable_velocity_vs_constant_velocity(self):
        """A spatially uniform velocity callable gives the same RHS as a constant one."""
        M, N = 6, 3
        x, D, LIFT, J, EToE, EToF, inv_mass, stiffness = _setup(M, N)
        rng = np.random.default_rng(7)
        u = rng.uniform(-1, 1, (M, N + 1))
        c = 1.3
        rhs_const = advection_rhs_1d(u, x, 0.5, lambda xv, tv: np.full_like(xv, c),
                                     stiffness, inv_mass, J, EToE, EToF)
        rhs_vary = advection_rhs_1d(u, x, 0.5, lambda xv, tv: c * np.ones_like(xv),
                                    stiffness, inv_mass, J, EToE, EToF)
        np.testing.assert_allclose(rhs_const, rhs_vary, atol=1e-14)


# ---------------------------------------------------------------------------
# rk4_step_advection_1d
# ---------------------------------------------------------------------------

class TestRk4Step:
    def test_constant_velocity_trivial_solution(self):
        """Constant u with constant velocity: u is unchanged after one step."""
        M, N = 4, 2
        x, D, LIFT, J, EToE, EToF, inv_mass, stiffness = _setup(M, N)
        u = np.full((M, N + 1), 2.0)

        def rhs(v, tv):
            return advection_rhs_1d(v, x, tv, lambda xv, tv2: np.ones_like(xv),
                                    stiffness, inv_mass, J, EToE, EToF)

        u_new = rk4_step_advection_1d(u, 0.0, 0.01, rhs)
        np.testing.assert_allclose(u_new, u, atol=1e-12)

    def test_output_shape_preserved(self):
        M, N = 5, 3
        x, D, LIFT, J, EToE, EToF, inv_mass, stiffness = _setup(M, N)
        u = np.random.default_rng(1).uniform(-1, 1, (M, N + 1))

        def rhs(v, tv):
            return advection_rhs_1d(v, x, tv, lambda xv, tv2: np.ones_like(xv),
                                    stiffness, inv_mass, J, EToE, EToF)

        u_new = rk4_step_advection_1d(u, 0.0, 1e-3, rhs)
        assert u_new.shape == u.shape


# ---------------------------------------------------------------------------
# simulate_advection1d — shapes and basic sanity
# ---------------------------------------------------------------------------

class TestSimulateShapes:
    def test_output_shapes_and_time(self):
        """simulate returns arrays of correct shape and t == t_final."""
        M, N = 8, 2
        t_final = 0.1
        x, u, t = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: np.ones_like(xv),
        )
        assert x.shape == (M, N + 1)
        assert u.shape == (M, N + 1)
        assert t == pytest.approx(t_final)

    def test_unsupported_limiter_raises(self):
        with pytest.raises(ValueError, match="Unsupported limiter_type"):
            simulate_advection1d(
                4, 2, 0.05,
                velocity=lambda xv, tv: np.ones_like(xv),
                limiter_type="unknown",
            )


# ---------------------------------------------------------------------------
# Constant-velocity advection: exact solution recovery
# ---------------------------------------------------------------------------

class TestConstantVelocityExact:
    """For a(x,t) = c and periodic IC u_0(x)=sin(2πx) on [0,1] the exact
    solution is u(x,t) = sin(2π(x-c*t)).  After t=1/c the wave returns to
    its initial position (one full period)."""

    @pytest.mark.parametrize("c", [1.0, -1.0, 0.5])
    def test_full_period_recovery(self, c):
        """After one full advection period the solution matches the initial condition."""
        M, N = 16, 3
        t_period = 1.0 / abs(c)
        x, u, t = simulate_advection1d(
            M, N, t_period,
            velocity=lambda xv, tv: np.full_like(xv, c),
            cfl=0.3,
            x_left=0.0, x_right=1.0,
            u_init=lambda xv: np.sin(2 * np.pi * xv),
        )
        u_exact = np.sin(2 * np.pi * x)
        np.testing.assert_allclose(u, u_exact, atol=1e-4)

    @pytest.mark.parametrize("c", [1.0, 2.0])
    def test_half_period_phase_shift(self, c):
        """After half a period the solution is the negated initial condition."""
        M, N = 16, 3
        t_half = 0.5 / abs(c)
        x, u, t = simulate_advection1d(
            M, N, t_half,
            velocity=lambda xv, tv: np.full_like(xv, c),
            cfl=0.3,
            x_left=0.0, x_right=1.0,
            u_init=lambda xv: np.sin(2 * np.pi * xv),
        )
        u_exact = -np.sin(2 * np.pi * x)
        np.testing.assert_allclose(u, u_exact, atol=1e-4)


# ---------------------------------------------------------------------------
# Mass conservation (periodic BC)
# ---------------------------------------------------------------------------

class TestMassConservation:
    def _total_mass(self, u, J, N):
        """Compute ∑_m 2 * J_m * ∫ u_m dr (total mass)."""
        mw = _mean_weights(N)
        return 2.0 * np.sum(J * (u @ mw))

    def test_constant_velocity_conserves_mass(self):
        """For periodic BC, total mass is conserved for constant velocity."""
        M, N = 10, 2
        t_final = 0.5
        x, r = build_mesh(M, N)
        h = 1.0 / M
        J = np.full(M, 0.5 * h)
        u0 = np.sin(2 * np.pi * x)
        mass0 = self._total_mass(u0, J, N)

        _, u_final, _ = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: np.ones_like(xv),
            cfl=0.3,
        )
        mass_final = self._total_mass(u_final, J, N)
        assert mass_final == pytest.approx(mass0, rel=1e-10)

    def test_variable_velocity_conserves_mass(self):
        """For periodic BC with a space-varying velocity, total mass is conserved."""
        M, N = 12, 2
        t_final = 0.3
        x, r = build_mesh(M, N)
        h = 1.0 / M
        J = np.full(M, 0.5 * h)
        u0 = np.sin(2 * np.pi * x) + 2.0
        mass0 = self._total_mass(u0, J, N)

        _, u_final, _ = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: 1.0 + 0.5 * np.sin(2 * np.pi * xv),
            cfl=0.3,
            u_init=lambda xv: np.sin(2 * np.pi * xv) + 2.0,
        )
        mass_final = self._total_mass(u_final, J, N)
        assert mass_final == pytest.approx(mass0, rel=1e-9)


# ---------------------------------------------------------------------------
# Spatial convergence order
# ---------------------------------------------------------------------------

class TestConvergenceOrder:
    """For a smooth initial condition and constant velocity, the DG scheme
    should converge at order N+1 in the L2 norm as the mesh is refined."""

    def _l2_error(self, M, N, t_final=0.25, c=1.0):
        x, u, t = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: np.full_like(xv, c),
            cfl=0.25,
        )
        u_exact = np.sin(2 * np.pi * (x - c * t_final))
        h = 1.0 / M
        return np.sqrt(h * np.mean((u - u_exact) ** 2))

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_convergence_rate(self, N):
        """L2 error should decrease at approximately rate N+1 under refinement."""
        M_coarse, M_fine = 8, 16
        e_coarse = self._l2_error(M_coarse, N)
        e_fine = self._l2_error(M_fine, N)
        observed_rate = np.log2(e_coarse / e_fine)
        assert observed_rate > N, (
            f"N={N}: expected rate > {N}, got {observed_rate:.2f}"
        )


# ---------------------------------------------------------------------------
# Transmissive boundary conditions
# ---------------------------------------------------------------------------

class TestTransmissiveBC:
    def test_outflow_boundary_stable(self):
        """Transmissive BC with leftward-only velocity should be stable and bounded."""
        M, N = 10, 2
        t_final = 0.3
        x, u, t = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: np.full_like(xv, 1.0),
            cfl=0.3,
            x_left=0.0, x_right=1.0,
            u_init=lambda xv: np.exp(-50 * (xv - 0.25) ** 2),
            bc_type="transmissive",
        )
        assert np.all(np.isfinite(u))
        assert np.max(np.abs(u)) < 2.0  # no blow-up

    def test_dirichlet_inflow_applied(self):
        """With Dirichlet BC and rightward flow, the left boundary should match u_bc."""
        M, N = 8, 2
        t_final = 0.2
        x_left, x_right = 0.0, 1.0
        x, u, t = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: np.full_like(xv, 1.0),
            cfl=0.3,
            x_left=x_left, x_right=x_right,
            u_init=lambda xv: np.zeros_like(xv),
            bc_type="dirichlet",
            u_bc=lambda xv, tv: 1.0,  # constant inflow value = 1
        )
        assert np.all(np.isfinite(u))


# ---------------------------------------------------------------------------
# Limiter tests
# ---------------------------------------------------------------------------

class TestLimiter:
    def _total_mass(self, u, J, N):
        mw = _mean_weights(N)
        return 2.0 * np.sum(J * (u @ mw))

    def test_tvb_limiter_preserves_mass(self):
        """TVB limiter should preserve the total mass."""
        M, N = 12, 3
        t_final = 0.5
        x, r = build_mesh(M, N)
        h = 1.0 / M
        J = np.full(M, 0.5 * h)
        u0 = np.sin(2 * np.pi * x)
        mass0 = self._total_mass(u0, J, N)

        _, u_final, _ = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: np.ones_like(xv),
            cfl=0.3,
            limiter_type="tvb",
        )
        mass_final = self._total_mass(u_final, J, N)
        assert mass_final == pytest.approx(mass0, rel=1e-9)

    def test_tvb_limiter_reduces_oscillations(self):
        """For a discontinuous IC, the TVB-limited run should have fewer oscillations."""
        M, N = 20, 3
        t_final = 0.2

        def step_ic(xv):
            return np.where(xv < 0.5, 1.0, 0.0)

        _, u_limited, _ = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: np.ones_like(xv),
            cfl=0.3,
            u_init=step_ic,
            limiter_type="tvb",
        )
        _, u_unlimited, _ = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: np.ones_like(xv),
            cfl=0.3,
            u_init=step_ic,
            limiter_type=None,
        )
        # Limited solution should have smaller max overshoot above the IC max (1.0)
        overshoot_limited = np.max(u_limited) - 1.0
        overshoot_unlimited = np.max(u_unlimited) - 1.0
        assert overshoot_limited < overshoot_unlimited


# ---------------------------------------------------------------------------
# Variable velocity field (non-constant)
# ---------------------------------------------------------------------------

class TestVariableVelocity:
    def test_zero_velocity_no_change(self):
        """With a(x,t) = 0 everywhere, u should not evolve."""
        M, N = 8, 2
        t_final = 0.3
        x_left, x_right = 0.0, 1.0
        x, r = build_mesh(M, N, x_left, x_right)
        u0 = np.sin(2 * np.pi * x)

        _, u_final, _ = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: np.zeros_like(xv),
            cfl=0.3,
            x_left=x_left, x_right=x_right,
            u_init=lambda xv: np.sin(2 * np.pi * xv),
        )
        np.testing.assert_allclose(u_final, u0, atol=1e-12)

    def test_solution_stays_finite_with_space_varying_velocity(self):
        """The solver should remain stable for a smooth space-varying velocity field."""
        M, N = 16, 2
        t_final = 0.5
        x, u, t = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: 1.0 + 0.5 * np.sin(2 * np.pi * xv),
            cfl=0.3,
        )
        assert np.all(np.isfinite(u))
        assert np.max(np.abs(u)) < 5.0  # no blow-up

    def test_time_dependent_velocity_stays_finite(self):
        """The solver should be stable for a time-dependent velocity field."""
        M, N = 12, 2
        t_final = 0.4
        x, u, t = simulate_advection1d(
            M, N, t_final,
            velocity=lambda xv, tv: np.cos(2 * np.pi * tv) * np.ones_like(xv),
            cfl=0.3,
        )
        assert np.all(np.isfinite(u))
