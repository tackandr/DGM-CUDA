"""Tests for the 1D Discontinuous Galerkin operators in dgm1d.py."""

import numpy as np
import pytest
from numpy import linalg as LA

from dgm1d import (
    DMatrix1D,
    GradVandermonde1D,
    Jacobian,
    Vandermonde1D,
    legendreGL,
)


# ---------------------------------------------------------------------------
# legendreGL
# ---------------------------------------------------------------------------

class TestLegendreGL:
    def test_endpoints(self):
        """Returned nodes always include -1 and +1."""
        for N in range(1, 7):
            r = legendreGL(N)
            assert r[0] == pytest.approx(-1.0)
            assert r[-1] == pytest.approx(1.0)

    def test_length(self):
        """legendreGL(N) returns exactly N+1 nodes."""
        for N in range(1, 8):
            r = legendreGL(N)
            assert len(r) == N + 1

    def test_sorted(self):
        """Nodes are in strictly ascending order."""
        for N in range(1, 7):
            r = legendreGL(N)
            assert np.all(np.diff(r) > 0)

    def test_interior_in_range(self):
        """All interior nodes lie strictly inside (-1, 1)."""
        for N in range(2, 8):
            r = legendreGL(N)
            assert np.all(r[1:-1] > -1.0)
            assert np.all(r[1:-1] < 1.0)

    def test_symmetric(self):
        """Nodes are symmetric about the origin."""
        for N in range(1, 7):
            r = legendreGL(N)
            assert r == pytest.approx(-r[::-1])

    def test_n1(self):
        """For N=1 only the two endpoints are returned."""
        r = legendreGL(1)
        assert r == pytest.approx([-1.0, 1.0])


# ---------------------------------------------------------------------------
# Vandermonde1D
# ---------------------------------------------------------------------------

class TestVandermonde1D:
    def test_shape(self):
        """V has shape (len(r), N+1)."""
        for N in range(1, 6):
            r = legendreGL(N)
            V = Vandermonde1D(N, r)
            assert V.shape == (N + 1, N + 1)

    def test_inverse_consistency(self):
        """V @ inv(V) should equal the identity matrix."""
        for N in range(1, 6):
            r = legendreGL(N)
            V = Vandermonde1D(N, r)
            np.testing.assert_allclose(V @ LA.inv(V), np.eye(N + 1), atol=1e-13)

    def test_first_column(self):
        """The zeroth-degree (constant) normalised polynomial is sqrt(1/2)."""
        for N in range(1, 6):
            r = legendreGL(N)
            V = Vandermonde1D(N, r)
            expected = np.full(N + 1, np.sqrt(0.5))
            np.testing.assert_allclose(V[:, 0], expected, atol=1e-14)

    def test_invertible(self):
        """V must be invertible (non-zero determinant)."""
        for N in range(1, 7):
            r = legendreGL(N)
            V = Vandermonde1D(N, r)
            assert abs(LA.det(V)) > 1e-10


# ---------------------------------------------------------------------------
# GradVandermonde1D
# ---------------------------------------------------------------------------

class TestGradVandermonde1D:
    def test_shape(self):
        """Gradient Vandermonde has shape (len(r), N+1)."""
        for N in range(1, 6):
            r = legendreGL(N)
            Vr = GradVandermonde1D(N, r)
            assert Vr.shape == (N + 1, N + 1)

    def test_first_column_zero(self):
        """Derivative of the constant (degree-0) polynomial is 0 everywhere."""
        for N in range(1, 6):
            r = legendreGL(N)
            Vr = GradVandermonde1D(N, r)
            np.testing.assert_allclose(Vr[:, 0], 0.0, atol=1e-14)

    def test_degree1_derivative(self):
        """Derivative of normalised P_1(x)=sqrt(3/2)*x is sqrt(3/2) everywhere."""
        for N in range(1, 6):
            r = legendreGL(N)
            Vr = GradVandermonde1D(N, r)
            expected = np.full(N + 1, np.sqrt(3.0 / 2.0))
            np.testing.assert_allclose(Vr[:, 1], expected, atol=1e-13)


# ---------------------------------------------------------------------------
# DMatrix1D
# ---------------------------------------------------------------------------

class TestDMatrix1D:
    def _setup(self, N):
        r = legendreGL(N)
        V = Vandermonde1D(N, r)
        D = DMatrix1D(N, r, V)
        return r, V, D

    def test_shape(self):
        """D has shape (N+1, N+1)."""
        for N in range(1, 7):
            r, V, D = self._setup(N)
            assert D.shape == (N + 1, N + 1)

    def test_differentiate_constant(self):
        """D applied to a constant vector gives zero."""
        for N in range(1, 7):
            r, V, D = self._setup(N)
            f = np.ones(N + 1)
            df = D @ f
            np.testing.assert_allclose(df, 0.0, atol=1e-12)

    def test_differentiate_x(self):
        """D applied to f(x)=x gives df/dx=1 everywhere."""
        for N in range(1, 7):
            r, V, D = self._setup(N)
            f = r
            df = D @ f
            np.testing.assert_allclose(df, 1.0, atol=1e-12)

    def test_differentiate_x_squared(self):
        """D applied to f(x)=x^2 gives df/dx=2x at each node."""
        for N in range(2, 7):
            r, V, D = self._setup(N)
            f = r ** 2
            df = D @ f
            np.testing.assert_allclose(df, 2.0 * r, atol=1e-11)

    def test_differentiate_cubic(self):
        """D applied to f(x)=x^3 gives df/dx=3x^2 for N>=3."""
        for N in range(3, 7):
            r, V, D = self._setup(N)
            f = r ** 3
            df = D @ f
            np.testing.assert_allclose(df, 3.0 * r ** 2, atol=1e-10)

    def test_row_sum_zero(self):
        """Each row of D sums to zero (partition of unity property)."""
        for N in range(1, 7):
            r, V, D = self._setup(N)
            np.testing.assert_allclose(D.sum(axis=1), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

class TestJacobian:
    def test_uniform_element(self):
        """For a uniform element [a, b] the Jacobian is (b-a)/2 everywhere."""
        for N in range(1, 6):
            r = legendreGL(N)
            V = Vandermonde1D(N, r)
            D = DMatrix1D(N, r, V)
            a, b = -1.0, 3.0
            # Physical nodes: linear map from [-1,1] to [a,b]
            x = 0.5 * (b - a) * r + 0.5 * (a + b)
            J = Jacobian(x, D)
            expected = np.full(N + 1, 0.5 * (b - a))
            np.testing.assert_allclose(J, expected, atol=1e-12)

    def test_identity_element(self):
        """On the reference element [-1,1] the Jacobian is 1 everywhere."""
        for N in range(1, 6):
            r = legendreGL(N)
            V = Vandermonde1D(N, r)
            D = DMatrix1D(N, r, V)
            J = Jacobian(r, D)
            np.testing.assert_allclose(J, 1.0, atol=1e-12)

    def test_output_shape(self):
        """Jacobian has the same shape as the node vector."""
        N = 4
        r = legendreGL(N)
        V = Vandermonde1D(N, r)
        D = DMatrix1D(N, r, V)
        J = Jacobian(r, D)
        assert J.shape == r.shape
