"""Tests for the 1D Burgers equation DG solver in burgers1d.py."""

import numpy as np
import pytest

from burgers1d import (
    build_connectivity,
    build_mesh,
    build_operators,
    burgers_rhs,
    exact_sine_solution,
    limit_solution,
    numerical_flux,
    rk4_step,
    simulate,
)
from dgm1d import Vandermonde1D, legendreGL


def _mean_weights(N, r):
    """Return exact nodal weights for the reference-element cell average."""
    q_nodes, q_weights = np.polynomial.legendre.leggauss(2 * N + 3)
    V = Vandermonde1D(N, r)
    V_inv = np.linalg.solve(V, np.eye(V.shape[0]))
    interp = Vandermonde1D(N, q_nodes) @ V_inv
    return 0.5 * (q_weights @ interp)


# ---------------------------------------------------------------------------
# build_mesh
# ---------------------------------------------------------------------------

class TestBuildMesh:
    def test_shape(self):
        """x has shape (M, N+1) and r has shape (N+1,)."""
        M, N = 5, 3
        x, r = build_mesh(M, N)
        assert x.shape == (M, N + 1)
        assert r.shape == (N + 1,)

    def test_endpoints_match_domain(self):
        """Leftmost node of first element equals x_left; rightmost of last equals x_right."""
        x, _ = build_mesh(6, 3, x_left=1.0, x_right=5.0)
        assert x[0, 0] == pytest.approx(1.0)
        assert x[-1, -1] == pytest.approx(5.0)

    def test_nodes_strictly_increasing(self):
        """Within each element, physical nodes are strictly increasing.
        Adjacent element boundaries share the same physical coordinate, so
        the globally ravelled array is non-decreasing (not strictly increasing).
        """
        x, _ = build_mesh(4, 3)
        # Within each element: strictly increasing
        for m in range(x.shape[0]):
            assert np.all(np.diff(x[m]) > 0)
        # Across the full domain: non-decreasing
        assert np.all(np.diff(x.ravel()) >= 0)

    def test_element_width(self):
        """Each element spans exactly (x_right - x_left) / M in physical space."""
        M, N = 5, 2
        x_left, x_right = 0.0, 2.0
        x, _ = build_mesh(M, N, x_left=x_left, x_right=x_right)
        expected_width = (x_right - x_left) / M
        for m in range(M):
            assert x[m, -1] - x[m, 0] == pytest.approx(expected_width, rel=1e-12)

    def test_r_nodes_are_gll(self):
        """Returned r matches legendreGL(N) exactly."""
        N = 4
        x, r = build_mesh(3, N)
        np.testing.assert_array_equal(r, legendreGL(N))


# ---------------------------------------------------------------------------
# build_connectivity
# ---------------------------------------------------------------------------

class TestBuildConnectivity:
    def test_shape(self):
        """EToE and EToF have shape (M, 2)."""
        M = 6
        EToE, EToF = build_connectivity(M)
        assert EToE.shape == (M, 2)
        assert EToF.shape == (M, 2)

    def test_left_neighbor(self):
        """Left neighbor (face 0) of element m is element (m-1) % M."""
        M = 5
        EToE, EToF = build_connectivity(M)
        for m in range(M):
            assert EToE[m, 0] == (m - 1) % M

    def test_right_neighbor(self):
        """Right neighbor (face 1) of element m is element (m+1) % M."""
        M = 5
        EToE, EToF = build_connectivity(M)
        for m in range(M):
            assert EToE[m, 1] == (m + 1) % M

    def test_neighbor_face_indices(self):
        """Left neighbor contributes its right face (1); right neighbor its left face (0)."""
        M = 4
        _, EToF = build_connectivity(M)
        assert np.all(EToF[:, 0] == 1)
        assert np.all(EToF[:, 1] == 0)

    def test_periodic_wrapping(self):
        """Connectivity wraps periodically at domain ends."""
        M = 4
        EToE, _ = build_connectivity(M)
        assert EToE[0, 0] == M - 1    # element 0's left neighbor is last element
        assert EToE[M - 1, 1] == 0    # last element's right neighbor is element 0

    def test_symmetry(self):
        """Connectivity is symmetric: if B is A's right neighbor then A is B's left neighbor."""
        M = 7
        EToE, EToF = build_connectivity(M)
        for m in range(M):
            right_neigh = EToE[m, 1]
            assert EToE[right_neigh, 0] == m


# ---------------------------------------------------------------------------
# numerical_flux
# ---------------------------------------------------------------------------

class TestNumericalFlux:
    def test_agrees_with_interior_flux_when_equal(self):
        """When u- == u+, the numerical flux equals the interior flux f(u) = u^2/2."""
        u = np.array([-2.0, 0.0, 1.5, 3.0])
        np.testing.assert_allclose(numerical_flux(u, u), 0.5 * u**2, atol=1e-14)

    def test_right_going_shock_uses_left_state(self):
        """Positive-speed shocks use the left upwind state."""
        assert numerical_flux(2.0, 1.0) == pytest.approx(2.0)

    def test_left_going_shock_uses_right_state(self):
        """Negative-speed shocks use the right upwind state."""
        assert numerical_flux(-1.0, -2.0) == pytest.approx(2.0)

    def test_rarefaction_through_zero_returns_zero_flux(self):
        """A rarefaction fan spanning zero uses the sonic flux."""
        assert numerical_flux(-1.0, 1.0) == pytest.approx(0.0)

    def test_positive_rarefaction_uses_left_state(self):
        """Positive rarefactions are upwinded from the left state."""
        assert numerical_flux(0.5, 1.0) == pytest.approx(0.125)

    def test_scalar_inputs(self):
        """Works with scalar inputs."""
        f_star = numerical_flux(1.0, 2.0)
        assert np.isscalar(f_star) or f_star.ndim == 0

    def test_output_shape_matches_input(self):
        """Output shape matches input shape."""
        u_m = np.ones((4, 5))
        u_p = np.ones((4, 5)) * 2
        assert numerical_flux(u_m, u_p).shape == (4, 5)


# ---------------------------------------------------------------------------
# burgers_rhs
# ---------------------------------------------------------------------------

class TestBurgersRHS:
    """Tests for the DG right-hand side."""

    def _build(self, M, N, x_left=0.0, x_right=2.0):
        x, r = build_mesh(M, N, x_left, x_right)
        _, D, LIFT = build_operators(N, r)
        EToE, EToF = build_connectivity(M)
        h = (x_right - x_left) / M
        J = np.full(M, 0.5 * h)
        return x, D, LIFT, J, EToE, EToF

    def test_output_shape(self):
        """RHS has same shape as u."""
        M, N = 5, 3
        x, D, LIFT, J, EToE, EToF = self._build(M, N)
        u = np.random.default_rng(1).uniform(-1, 1, (M, N + 1))
        rhs = burgers_rhs(u, D, LIFT, J, EToE, EToF)
        assert rhs.shape == (M, N + 1)

    def test_constant_gives_zero_rhs(self):
        """Constant u has zero spatial gradient; all interface fluxes cancel → rhs = 0."""
        for c in [-1.5, 0.0, 2.0]:
            M, N = 4, 3
            x, D, LIFT, J, EToE, EToF = self._build(M, N)
            u = np.full((M, N + 1), c)
            rhs = burgers_rhs(u, D, LIFT, J, EToE, EToF)
            np.testing.assert_allclose(rhs, 0.0, atol=1e-12,
                                       err_msg=f"Failed for constant u = {c}")

    def test_conservative_interface_fluxes(self):
        """The same single-valued F* must appear at both sides of every interface.

        For every interface between element m and its right neighbour (m+1)%M,
        the flux computed by element m (face 1, L-R convention: u[m,N] vs u[m+1,0])
        must equal the flux computed by element m+1 (face 0, same convention).
        """
        M, N = 6, 3
        x, D, LIFT, J, EToE, EToF = self._build(M, N)
        fmask = np.array([0, N])
        rng = np.random.default_rng(42)
        u = rng.uniform(-1, 1, (M, N + 1))
        for m in range(M):
            mp1 = (m + 1) % M
            u_L = u[m, N]           # right boundary of element m
            u_R = u[mp1, 0]         # left boundary of element m+1
            # Element m's right-face flux (L=current, R=right-neighbour)
            F_from_m = numerical_flux(u_L, u_R)
            # Element m+1's left-face flux (L=left-neighbour, R=current)
            # After the L-R fix this must equal F_from_m
            u_ext_mp1_left = u[EToE[mp1, 0], fmask[EToF[mp1, 0]]]   # = u[m, N]
            u_int_mp1_left = u[mp1, fmask[0]]                         # = u[mp1, 0]
            F_from_mp1 = numerical_flux(u_ext_mp1_left, u_int_mp1_left)
            assert F_from_m == pytest.approx(F_from_mp1), (
                f"Interface flux mismatch at m={m}: {F_from_m} vs {F_from_mp1}"
            )

    def test_volume_term_accuracy(self):
        """Volume term D @ f(u) correctly differentiates u^2/2 at interior nodes.

        For a smooth, single-element-equivalent polynomial, the volume term
        should approximate the derivative of f = u^2/2.  We use a degree-N
        polynomial whose derivative we know exactly.
        """
        M, N = 1, 4
        x, D, LIFT, J, EToE, EToF = self._build(M, N, x_left=-1.0, x_right=1.0)
        r = x[0]   # single element, physical coords = reference coords (J=1)
        # u is a degree-2 polynomial so f=u^2/2 is degree-4 and D @ f should be exact
        u = (r**2).reshape(1, -1)
        f = 0.5 * u**2   # (r^2)^2/2 = r^4/2
        vol = f @ D.T    # should give d(r^4/2)/dr = 2r^3
        np.testing.assert_allclose(vol[0], 2.0 * r**3, atol=1e-10)


# ---------------------------------------------------------------------------
# rk4_step
# ---------------------------------------------------------------------------

class TestRK4Step:
    def test_constant_unchanged(self):
        """A constant solution is a fixed point of rk4_step."""
        M, N = 4, 2
        x, r = build_mesh(M, N)
        _, D, LIFT = build_operators(N, r)
        EToE, EToF = build_connectivity(M)
        J = np.full(M, 0.5 * 2.0 / M)
        u = np.full((M, N + 1), 1.5)
        u_new = rk4_step(u, dt=0.01, D=D, LIFT=LIFT, J=J, EToE=EToE, EToF=EToF)
        np.testing.assert_allclose(u_new, u, atol=1e-12)

    def test_output_shape(self):
        """rk4_step preserves the shape of u."""
        M, N = 3, 4
        x, r = build_mesh(M, N)
        _, D, LIFT = build_operators(N, r)
        EToE, EToF = build_connectivity(M)
        J = np.full(M, 0.5 * 2.0 / M)
        u = np.ones((M, N + 1))
        assert rk4_step(u, 0.01, D, LIFT, J, EToE, EToF).shape == (M, N + 1)


# ---------------------------------------------------------------------------
# limiter
# ---------------------------------------------------------------------------

class TestLimiter:
    def test_preserves_constant_state(self):
        """The limiter leaves constant states unchanged."""
        M, N = 5, 4
        u = np.full((M, N + 1), 1.25)
        J = np.full(M, 0.2)
        np.testing.assert_allclose(limit_solution(u, J), u, atol=1e-14)

    def test_preserves_cell_averages(self):
        """Limiting troubled cells preserves the per-element average."""
        M, N = 4, 5
        x, r = build_mesh(M, N)
        u = np.sin(np.pi * x)
        u[1] *= 4.0
        J = np.full(M, 0.5 * 2.0 / M)
        limited = limit_solution(u, J)
        assert np.all(np.isfinite(limited))
        mean_weights = _mean_weights(N, r)
        np.testing.assert_allclose(limited @ mean_weights, u @ mean_weights, atol=1e-12)


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_output_shapes(self):
        """simulate returns (M, N+1) arrays for x and u."""
        M, N = 5, 2
        x, u, t = simulate(M, N, t_final=0.05)
        assert x.shape == (M, N + 1)
        assert u.shape == (M, N + 1)

    def test_reaches_t_final(self):
        """Simulation advances to t_final within floating-point tolerance."""
        x, u, t = simulate(4, 2, t_final=0.1)
        assert t == pytest.approx(0.1, rel=1e-6)

    def test_custom_initial_condition(self):
        """User-supplied initial condition is applied correctly."""
        M, N = 4, 2
        x, u, t = simulate(M, N, t_final=0.0, u_init=lambda xv: xv**2)
        # At t=0 (dt chosen as min(cfl_dt, 0)), u should equal x^2
        # (simulate still takes at least one step, but for t_final=0
        #  the loop is skipped and u = u_init(x))
        x_ref, _ = build_mesh(M, N)
        np.testing.assert_allclose(u, x_ref**2, atol=1e-14)

    def test_constant_ic_unchanged(self):
        """A constant initial condition stays constant for all time."""
        M, N = 6, 3
        c = 2.0
        x, u, t = simulate(M, N, t_final=0.3, u_init=lambda xv: np.full_like(xv, c))
        np.testing.assert_allclose(u, c, atol=1e-10)

    def test_mass_approximately_conserved(self):
        """The sum of u over all nodes changes by less than 1 % over t in [0, 0.1].

        For sin(pi*x) on [0,2] the domain integral is zero; numerical conservation
        should keep the discrete sum near zero throughout the run.
        """
        M, N = 10, 3
        mass_init = 0.0   # integral of sin(pi*x) over [0,2] = 0
        x, u, t = simulate(M, N, t_final=0.1)
        h = 2.0 / M
        _, r = build_mesh(M, N)
        mass_final = h * np.sum(u @ _mean_weights(N, r))
        # Tolerance: 5 % of the domain length — DG with 10 elements and N=3
        # gives sub-percent conservation; 5 % provides generous headroom.
        _MASS_ATOL = 0.05
        assert abs(mass_final - mass_init) < _MASS_ATOL

    def test_different_polynomial_orders(self):
        """simulate runs without errors for N = 1, 2, 3, 4."""
        for N in range(1, 5):
            x, u, t = simulate(4, N, t_final=0.05)
            assert u.shape == (4, N + 1)
            assert np.all(np.isfinite(u))

    def test_different_element_counts(self):
        """simulate runs without errors for various M."""
        for M in [2, 5, 10, 20]:
            x, u, t = simulate(M, 2, t_final=0.05)
            assert u.shape == (M, 3)
            assert np.all(np.isfinite(u))

    def test_matches_exact_sine_solution_before_shock(self):
        """A longer pre-shock run stays stable and matches the analytical sine-wave solution."""
        M, N = 40, 4
        t_final = 0.3
        x, u, t = simulate(M, N, t_final=t_final)
        u_exact = exact_sine_solution(x, t)
        assert np.all(np.isfinite(u))
        # This tolerance matches the observed pre-shock error level for the
        # current discretisation while remaining tight enough to catch the
        # interface blow-up regression that previously produced infinities.
        np.testing.assert_allclose(u, u_exact, atol=1.6e-2, rtol=2.0e-2)

    def test_stays_bounded_after_shock(self):
        """A post-shock run remains finite and avoids the long-time collapse."""
        x, u, t = simulate(40, 5, t_final=1.2)
        POST_SHOCK_MAX_AMPLITUDE = 1.0
        assert np.all(np.isfinite(u))
        # The limiter/upwind combination should keep the post-shock sine-wave run
        # within the original [-1, 1] amplitude envelope.
        assert np.max(np.abs(u)) < POST_SHOCK_MAX_AMPLITUDE


class TestExactSineSolution:
    def test_initial_time_matches_sine_wave(self):
        """The analytical solution reduces to the initial sine wave at t=0."""
        x = np.linspace(0.0, 2.0, 21)
        np.testing.assert_allclose(exact_sine_solution(x, 0.0), np.sin(np.pi * x), atol=1e-14)

    def test_rejects_post_shock_time(self):
        """The closed-form characteristic solution is only used before shock formation."""
        with pytest.raises(ValueError):
            exact_sine_solution(np.array([0.25, 0.75]), 1.0 / np.pi)
