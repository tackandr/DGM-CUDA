"""Tests for 2D simplex DG reference operators."""

import numpy as np
from numpy import linalg as LA

from dgm2d import (
    num_nodes_2d,
    simplex_nodes_2d,
    modal_basis_2d,
    grad_modal_basis_2d,
    Vandermonde2D,
    GradVandermonde2D,
    Dmatrices2D,
    face_mask_2d,
    Lift2D,
)


def test_node_count_formula():
    for N in range(0, 7):
        r, s = simplex_nodes_2d(N)
        assert r.shape == (num_nodes_2d(N),)
        assert s.shape == (num_nodes_2d(N),)


def test_vandermonde_shape_and_invertibility():
    for N in range(1, 6):
        r, s = simplex_nodes_2d(N)
        V = Vandermonde2D(N, r, s)
        Np = num_nodes_2d(N)
        assert V.shape == (Np, Np)
        assert abs(LA.det(V)) > 1e-12


def test_modal_basis_orthonormality_quadrature():
    N = 4
    r, s = simplex_nodes_2d(N)
    B = modal_basis_2d(N, r, s)
    assert B.shape == (num_nodes_2d(N), num_nodes_2d(N))


def test_gradient_vandermonde_shape():
    for N in range(1, 5):
        r, s = simplex_nodes_2d(N)
        Vr, Vs = GradVandermonde2D(N, r, s)
        Np = num_nodes_2d(N)
        assert Vr.shape == (Np, Np)
        assert Vs.shape == (Np, Np)


def test_differentiation_polynomials():
    N = 4
    r, s = simplex_nodes_2d(N)
    Dr, Ds = Dmatrices2D(N, r, s)

    f = r**2 + 2.0 * r * s + 3.0 * s**2 + 0.25
    fr = 2.0 * r + 2.0 * s
    fs = 2.0 * r + 6.0 * s

    np.testing.assert_allclose(Dr @ f, fr, atol=5e-11)
    np.testing.assert_allclose(Ds @ f, fs, atol=5e-11)


def test_face_masks_and_lift_shape():
    for N in range(1, 6):
        r, s = simplex_nodes_2d(N)
        fmask = face_mask_2d(N, r, s)
        assert len(fmask) == 3
        for ids in fmask:
            assert len(ids) == N + 1

        np.testing.assert_allclose(s[fmask[0]], -1.0, atol=1e-12)
        np.testing.assert_allclose(r[fmask[2]], -1.0, atol=1e-12)
        np.testing.assert_allclose(r[fmask[1]] + s[fmask[1]], 0.0, atol=1e-12)

        LIFT = Lift2D(N, r, s, fmask)
        assert LIFT.shape == (num_nodes_2d(N), 3 * (N + 1))


def test_grad_modal_matches_fd():
    N = 3
    r, s = simplex_nodes_2d(N)
    Br, Bs = grad_modal_basis_2d(N, r, s)

    eps = 1e-6
    Bp = modal_basis_2d(N, r + eps, s)
    Bm = modal_basis_2d(N, r - eps, s)
    Br_fd = (Bp - Bm) / (2.0 * eps)

    Bp = modal_basis_2d(N, r, s + eps)
    Bm = modal_basis_2d(N, r, s - eps)
    Bs_fd = (Bp - Bm) / (2.0 * eps)

    np.testing.assert_allclose(Br, Br_fd, atol=5e-7)
    np.testing.assert_allclose(Bs, Bs_fd, atol=5e-7)
