"""Tests for the 2D variable-coefficient advection solver functions in advection2d.py."""

import numpy as np
import pytest

from advection2d import (
    advection_rhs,
    advection_rhs_vc_2d,
    rk4_step_advection_vc,
    simulate_advection_vc_2d,
    _triangle_mean_weights,
)
from mesh2d import setup_2d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rhs_vc(u, data, ax_fn, ay_fn, t=0.0, bc_type="periodic", flux_type="upwind"):
    """Convenience wrapper: evaluate advection_rhs_vc_2d from a data dict."""
    x, y = data["x"], data["y"]
    ax_f = np.asarray(ax_fn(x, y, t), dtype=float)
    ay_f = np.asarray(ay_fn(x, y, t), dtype=float)
    return advection_rhs_vc_2d(
        u, ax_f, ay_f,
        data["Dr"], data["Ds"], data["LIFT"],
        data["rx"], data["sx"], data["ry"], data["sy"],
        data["J"], data["nx"], data["ny"],
        data["Fscale"], data["EToE"], data["EToF"], data["fmask"],
        bc_type=bc_type,
        flux_type=flux_type,
        boundary_mask=data.get("boundary_mask"),
    )


def _rhs_const(u, data, ax, ay, bc_type="periodic", flux_type="upwind"):
    """Convenience wrapper: evaluate the existing constant-velocity advection_rhs."""
    return advection_rhs(
        u, ax, ay,
        data["Dr"], data["Ds"], data["LIFT"],
        data["rx"], data["sx"], data["ry"], data["sy"],
        data["J"], data["nx"], data["ny"],
        data["Fscale"], data["EToE"], data["EToF"], data["fmask"],
        bc_type=bc_type,
        flux_type=flux_type,
        boundary_mask=data.get("boundary_mask"),
    )


def _cell_masses(u, N, data):
    """Return total mass: sum_k 2 * J_k * avg_k * area_ref."""
    mw = _triangle_mean_weights(N)
    # J has shape (K, Np); for affine triangles it is constant per element.
    return 2.0 * np.sum(data["J"][:, 0] * (u @ mw))


# ---------------------------------------------------------------------------
# advection_rhs_vc_2d: constant callable equals constant-scalar advection_rhs
# ---------------------------------------------------------------------------

class TestRhsVCEqualsConstant:
    """A spatially uniform callable must reproduce the existing constant-coefficient RHS."""

    @pytest.mark.parametrize("ax,ay", [(1.0, 0.3), (-0.5, 0.7), (0.0, 1.0)])
    def test_periodic_matches_constant_rhs(self, ax, ay):
        N, nx_m, ny_m = 2, 4, 3
        data = setup_2d(N, nx_m, ny_m, periodic=True)
        rng = np.random.default_rng(0)
        u = rng.uniform(-1, 1, data["x"].shape)

        rhs_const = _rhs_const(u, data, ax, ay)
        rhs_vc = _rhs_vc(u, data,
                         lambda xv, yv, tv: np.full_like(xv, ax),
                         lambda xv, yv, tv: np.full_like(xv, ay))
        np.testing.assert_allclose(rhs_vc, rhs_const, atol=1e-12)

    def test_transmissive_matches_constant_rhs(self):
        ax, ay = 0.6, -0.4
        N, nx_m, ny_m = 2, 3, 3
        data = setup_2d(N, nx_m, ny_m, periodic=False)
        rng = np.random.default_rng(1)
        u = rng.uniform(-1, 1, data["x"].shape)

        rhs_const = _rhs_const(u, data, ax, ay, bc_type="transmissive")
        rhs_vc = _rhs_vc(u, data,
                         lambda xv, yv, tv: np.full_like(xv, ax),
                         lambda xv, yv, tv: np.full_like(xv, ay),
                         bc_type="transmissive")
        np.testing.assert_allclose(rhs_vc, rhs_const, atol=1e-12)

    def test_central_flux_matches(self):
        ax, ay = 1.0, 1.0
        N, nx_m, ny_m = 2, 4, 4
        data = setup_2d(N, nx_m, ny_m, periodic=True)
        rng = np.random.default_rng(2)
        u = rng.uniform(-1, 1, data["x"].shape)

        rhs_const = _rhs_const(u, data, ax, ay, flux_type="central")
        rhs_vc = _rhs_vc(u, data,
                         lambda xv, yv, tv: np.full_like(xv, ax),
                         lambda xv, yv, tv: np.full_like(xv, ay),
                         flux_type="central")
        np.testing.assert_allclose(rhs_vc, rhs_const, atol=1e-12)


# ---------------------------------------------------------------------------
# advection_rhs_vc_2d: zero RHS for constant solutions
# ---------------------------------------------------------------------------

class TestConstantSolutionZeroRhs:
    def test_periodic(self):
        N, nx_m, ny_m = 2, 4, 3
        data = setup_2d(N, nx_m, ny_m, periodic=True)
        u = np.full_like(data["x"], 3.0)
        rhs = _rhs_vc(u, data,
                      lambda xv, yv, tv: 1.5 * np.ones_like(xv),
                      lambda xv, yv, tv: -0.7 * np.ones_like(xv))
        np.testing.assert_allclose(rhs, 0.0, atol=1e-12)

    def test_transmissive(self):
        N, nx_m, ny_m = 2, 4, 3
        data = setup_2d(N, nx_m, ny_m, periodic=False)
        u = np.full_like(data["x"], -2.0)
        rhs = _rhs_vc(u, data,
                      lambda xv, yv, tv: np.ones_like(xv),
                      lambda xv, yv, tv: np.ones_like(xv),
                      bc_type="transmissive")
        np.testing.assert_allclose(rhs, 0.0, atol=1e-12)

    def test_variable_velocity_constant_u(self):
        """Constant u gives zero RHS even with spatially varying velocity."""
        N, nx_m, ny_m = 2, 4, 4
        data = setup_2d(N, nx_m, ny_m, periodic=True)
        u = np.full_like(data["x"], 1.0)
        rhs = _rhs_vc(u, data,
                      lambda xv, yv, tv: 1.0 + 0.5 * np.sin(2 * np.pi * xv),
                      lambda xv, yv, tv: np.cos(2 * np.pi * yv))
        np.testing.assert_allclose(rhs, 0.0, atol=1e-11)


# ---------------------------------------------------------------------------
# advection_rhs_vc_2d: error handling
# ---------------------------------------------------------------------------

class TestRhsVCErrors:
    def test_unsupported_bc_type_raises(self):
        N, nx_m, ny_m = 2, 3, 3
        data = setup_2d(N, nx_m, ny_m, periodic=False)
        u = np.zeros_like(data["x"])
        with pytest.raises(ValueError, match="Unsupported bc_type"):
            _rhs_vc(u, data,
                    lambda xv, yv, tv: np.ones_like(xv),
                    lambda xv, yv, tv: np.ones_like(xv),
                    bc_type="invalid")

    def test_unsupported_flux_type_raises(self):
        N, nx_m, ny_m = 2, 3, 3
        data = setup_2d(N, nx_m, ny_m, periodic=True)
        u = np.zeros_like(data["x"])
        with pytest.raises(ValueError, match="Unsupported flux_type"):
            _rhs_vc(u, data,
                    lambda xv, yv, tv: np.ones_like(xv),
                    lambda xv, yv, tv: np.ones_like(xv),
                    flux_type="bad")

    def test_dirichlet_requires_u_bc(self):
        N, nx_m, ny_m = 2, 3, 3
        data = setup_2d(N, nx_m, ny_m, periodic=False)
        u = np.zeros_like(data["x"])
        x, y = data["x"], data["y"]
        ax_f = np.ones_like(x)
        ay_f = np.zeros_like(y)
        with pytest.raises(ValueError, match="u_bc must be provided"):
            advection_rhs_vc_2d(
                u, ax_f, ay_f,
                data["Dr"], data["Ds"], data["LIFT"],
                data["rx"], data["sx"], data["ry"], data["sy"],
                data["J"], data["nx"], data["ny"],
                data["Fscale"], data["EToE"], data["EToF"], data["fmask"],
                bc_type="dirichlet",
            )


# ---------------------------------------------------------------------------
# rk4_step_advection_vc: basic checks
# ---------------------------------------------------------------------------

class TestRk4StepVC:
    def test_constant_u_unchanged(self):
        """Constant u with any uniform velocity stays unchanged after one step."""
        N, nx_m, ny_m = 2, 4, 3
        data = setup_2d(N, nx_m, ny_m, periodic=True)
        u = np.full_like(data["x"], 2.5)

        def rhs(v, tv):
            return _rhs_vc(v, data,
                           lambda xv, yv, s: np.full_like(xv, 1.0),
                           lambda xv, yv, s: np.full_like(xv, 0.5),
                           t=tv)

        u_new = rk4_step_advection_vc(u, 0.0, 0.01, rhs)
        np.testing.assert_allclose(u_new, u, atol=1e-12)

    def test_output_shape_preserved(self):
        N, nx_m, ny_m = 2, 4, 4
        data = setup_2d(N, nx_m, ny_m, periodic=True)
        u = np.random.default_rng(3).uniform(-1, 1, data["x"].shape)

        def rhs(v, tv):
            return _rhs_vc(v, data,
                           lambda xv, yv, s: np.ones_like(xv),
                           lambda xv, yv, s: np.zeros_like(xv),
                           t=tv)

        u_new = rk4_step_advection_vc(u, 0.0, 1e-3, rhs)
        assert u_new.shape == u.shape


# ---------------------------------------------------------------------------
# simulate_advection_vc_2d: shapes and time
# ---------------------------------------------------------------------------

class TestSimulateVCShapes:
    def test_output_shapes_and_time(self):
        N, nx_m, ny_m = 2, 4, 3
        t_final = 0.05
        data, u, t = simulate_advection_vc_2d(
            N, nx_m, ny_m, t_final,
            velocity_x=lambda xv, yv, tv: np.ones_like(xv),
            velocity_y=lambda xv, yv, tv: np.zeros_like(xv),
        )
        K = 2 * nx_m * ny_m
        Np = (N + 1) * (N + 2) // 2
        assert u.shape == (K, Np)
        assert data["x"].shape == (K, Np)
        assert t == pytest.approx(t_final)

    def test_unsupported_limiter_raises(self):
        with pytest.raises(ValueError, match="Unsupported limiter_type"):
            simulate_advection_vc_2d(
                2, 4, 3, 0.05,
                velocity_x=lambda xv, yv, tv: np.ones_like(xv),
                velocity_y=lambda xv, yv, tv: np.zeros_like(xv),
                limiter_type="bad",
            )


# ---------------------------------------------------------------------------
# simulate_advection_vc_2d: exact solution for constant velocity translation
# ---------------------------------------------------------------------------

class TestConstantVelocityExact:
    """For constant a_x, a_y and periodic IC u_0(x,y) = sin(2pi*x)*sin(2pi*y),
    the exact solution is u(x,y,t) = sin(2pi*(x-a_x*t))*sin(2pi*(y-a_y*t))."""

    @pytest.mark.parametrize("ax,ay", [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)])
    def test_short_time_accuracy(self, ax, ay):
        """After a short integration the error should be small for a smooth IC."""
        N, nx_m, ny_m = 3, 6, 6
        t_final = 0.1
        data, u, t = simulate_advection_vc_2d(
            N, nx_m, ny_m, t_final,
            velocity_x=lambda xv, yv, tv: np.full_like(xv, ax),
            velocity_y=lambda xv, yv, tv: np.full_like(xv, ay),
            cfl=0.2,
        )
        x, y = data["x"], data["y"]
        u_exact = np.sin(2 * np.pi * (x - ax * t)) * np.sin(2 * np.pi * (y - ay * t))
        np.testing.assert_allclose(u, u_exact, atol=1.5e-2)

    def test_vc_matches_constant_simulate(self):
        """VC solver with constant callable must match the constant-coefficient driver."""
        from advection2d import simulate_advection_2d
        ax, ay = 0.7, 0.4
        N, nx_m, ny_m = 2, 4, 4
        t_final = 0.08

        data_c, u_c, t_c = simulate_advection_2d(
            N, nx_m, ny_m, t_final, ax=ax, ay=ay, cfl=0.2
        )
        data_v, u_v, t_v = simulate_advection_vc_2d(
            N, nx_m, ny_m, t_final,
            velocity_x=lambda xv, yv, tv: np.full_like(xv, ax),
            velocity_y=lambda xv, yv, tv: np.full_like(xv, ay),
            cfl=0.2,
        )
        # Both should reach the same final time and produce the same solution
        assert t_c == pytest.approx(t_v, rel=1e-10)
        np.testing.assert_allclose(u_v, u_c, atol=1e-12)


# ---------------------------------------------------------------------------
# Mass conservation for periodic BC
# ---------------------------------------------------------------------------

class TestMassConservation:
    def _total_mass(self, u, N, data):
        mw = _triangle_mean_weights(N)
        return 2.0 * np.sum(data["J"][:, 0] * (u @ mw))

    def test_constant_velocity_conserves_mass(self):
        N, nx_m, ny_m = 2, 5, 4
        t_final = 0.2
        ax, ay = 1.0, 0.5
        data_init = setup_2d(N, nx_m, ny_m, periodic=True)
        u0 = np.sin(2 * np.pi * data_init["x"]) * np.sin(2 * np.pi * data_init["y"])
        mass0 = self._total_mass(u0, N, data_init)

        data, u, t = simulate_advection_vc_2d(
            N, nx_m, ny_m, t_final,
            velocity_x=lambda xv, yv, tv: np.full_like(xv, ax),
            velocity_y=lambda xv, yv, tv: np.full_like(xv, ay),
            cfl=0.2,
        )
        mass_f = self._total_mass(u, N, data)
        assert mass_f == pytest.approx(mass0, rel=1e-9)

    def test_divergence_free_variable_velocity_conserves_mass(self):
        """Divergence-free velocity a = (sin(2pi*y), -sin(2pi*x)) conserves mass."""
        N, nx_m, ny_m = 2, 5, 5
        t_final = 0.15
        data_init = setup_2d(N, nx_m, ny_m, periodic=True)
        u0 = np.sin(2 * np.pi * data_init["x"]) * np.cos(2 * np.pi * data_init["y"]) + 2.0
        mass0 = self._total_mass(u0, N, data_init)

        data, u, t = simulate_advection_vc_2d(
            N, nx_m, ny_m, t_final,
            velocity_x=lambda xv, yv, tv: np.sin(2 * np.pi * yv),
            velocity_y=lambda xv, yv, tv: -np.sin(2 * np.pi * xv),
            cfl=0.2,
            u_init=lambda xv, yv: np.sin(2 * np.pi * xv) * np.cos(2 * np.pi * yv) + 2.0,
        )
        mass_f = self._total_mass(u, N, data)
        assert mass_f == pytest.approx(mass0, rel=1e-8)


# ---------------------------------------------------------------------------
# Stability: solution stays bounded for general velocity fields
# ---------------------------------------------------------------------------

class TestStability:
    def test_space_varying_velocity_stays_finite(self):
        N, nx_m, ny_m = 2, 6, 6
        t_final = 0.3
        data, u, t = simulate_advection_vc_2d(
            N, nx_m, ny_m, t_final,
            velocity_x=lambda xv, yv, tv: 1.0 + 0.5 * np.sin(2 * np.pi * xv),
            velocity_y=lambda xv, yv, tv: 0.5 * np.cos(2 * np.pi * yv),
            cfl=0.2,
        )
        assert np.all(np.isfinite(u))
        assert np.max(np.abs(u)) < 10.0  # no blow-up

    def test_time_dependent_velocity_stays_finite(self):
        N, nx_m, ny_m = 2, 5, 5
        t_final = 0.3
        data, u, t = simulate_advection_vc_2d(
            N, nx_m, ny_m, t_final,
            velocity_x=lambda xv, yv, tv: np.cos(2 * np.pi * tv) * np.ones_like(xv),
            velocity_y=lambda xv, yv, tv: np.sin(2 * np.pi * tv) * np.ones_like(yv),
            cfl=0.2,
        )
        assert np.all(np.isfinite(u))

    def test_zero_velocity_no_evolution(self):
        """With zero velocity the solution should not change."""
        N, nx_m, ny_m = 2, 4, 4
        t_final = 0.2
        data_init = setup_2d(N, nx_m, ny_m, periodic=True)
        u0 = np.sin(2 * np.pi * data_init["x"]) * np.sin(2 * np.pi * data_init["y"])

        data, u, t = simulate_advection_vc_2d(
            N, nx_m, ny_m, t_final,
            velocity_x=lambda xv, yv, tv: np.zeros_like(xv),
            velocity_y=lambda xv, yv, tv: np.zeros_like(yv),
            cfl=0.2,
            u_init=lambda xv, yv: np.sin(2 * np.pi * xv) * np.sin(2 * np.pi * yv),
        )
        np.testing.assert_allclose(u, u0, atol=1e-12)


# ---------------------------------------------------------------------------
# Limiters
# ---------------------------------------------------------------------------

class TestLimiters:
    def _total_mass(self, u, N, data):
        mw = _triangle_mean_weights(N)
        return 2.0 * np.sum(data["J"][:, 0] * (u @ mw))

    @pytest.mark.parametrize("limiter_type", ["bound_preserving", "minmod", "van_leer"])
    def test_limiter_preserves_mass(self, limiter_type):
        N, nx_m, ny_m = 2, 5, 4
        t_final = 0.15
        data_init = setup_2d(N, nx_m, ny_m, periodic=True)
        u0 = np.sin(2 * np.pi * data_init["x"]) * np.sin(2 * np.pi * data_init["y"])
        mass0 = self._total_mass(u0, N, data_init)

        data, u, t = simulate_advection_vc_2d(
            N, nx_m, ny_m, t_final,
            velocity_x=lambda xv, yv, tv: np.full_like(xv, 1.0),
            velocity_y=lambda xv, yv, tv: np.full_like(xv, 0.5),
            cfl=0.2,
            limiter_type=limiter_type,
        )
        mass_f = self._total_mass(u, N, data)
        assert mass_f == pytest.approx(mass0, rel=1e-8)

    def test_transmissive_bc(self):
        """Non-periodic run with variable velocity should be stable."""
        N, nx_m, ny_m = 2, 4, 4
        t_final = 0.1
        data, u, t = simulate_advection_vc_2d(
            N, nx_m, ny_m, t_final,
            velocity_x=lambda xv, yv, tv: np.full_like(xv, 1.0),
            velocity_y=lambda xv, yv, tv: np.full_like(xv, 0.5),
            cfl=0.2,
            periodic=False,
        )
        assert np.all(np.isfinite(u))
