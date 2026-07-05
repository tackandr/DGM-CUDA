"""Tests for 2D triangular mesh/connectivity/geometry helpers."""

import numpy as np

from dgm2d import simplex_nodes_2d, Dmatrices2D, face_mask_2d
from mesh2d import (
    build_uniform_tri_mesh,
    build_connectivity_2d,
    build_geometry_2d,
    setup_2d,
)


def test_uniform_mesh_shapes():
    VX, VY, EToV = build_uniform_tri_mesh(4, 3)
    assert VX.shape == VY.shape
    assert EToV.shape == (2 * 4 * 3, 3)


def test_connectivity_shapes_and_ranges():
    VX, VY, EToV = build_uniform_tri_mesh(3, 2)
    EToE, EToF = build_connectivity_2d(EToV, VX, VY)
    K = EToV.shape[0]

    assert EToE.shape == (K, 3)
    assert EToF.shape == (K, 3)
    assert np.all((EToE >= 0) & (EToE < K))
    assert np.all((EToF >= 0) & (EToF < 3))


def test_connectivity_symmetry_for_interior_faces():
    VX, VY, EToV = build_uniform_tri_mesh(4, 3)
    EToE, EToF = build_connectivity_2d(EToV, VX, VY)

    K = EToV.shape[0]
    for k in range(K):
        for f in range(3):
            kn = EToE[k, f]
            fn = EToF[k, f]
            if kn != k or fn != f:
                assert EToE[kn, fn] == k
                assert EToF[kn, fn] == f


def test_periodic_connectivity_has_no_self_boundary_faces():
    nx, ny = 3, 2
    x_left, x_right = 0.0, 1.0
    y_bottom, y_top = 0.0, 1.0
    VX, VY, EToV = build_uniform_tri_mesh(nx, ny, x_left, x_right, y_bottom, y_top)
    EToE, EToF = build_connectivity_2d(
        EToV,
        VX,
        VY,
        periodic=True,
        domain=(x_left, x_right, y_bottom, y_top),
    )
    K = EToV.shape[0]
    self_faces = 0
    for k in range(K):
        for f in range(3):
            if EToE[k, f] == k and EToF[k, f] == f:
                self_faces += 1
    assert self_faces == 0


def test_geometry_affine_jacobian_positive_and_constant_per_element():
    N = 3
    VX, VY, EToV = build_uniform_tri_mesh(2, 2)
    r, s = simplex_nodes_2d(N)
    Dr, Ds = Dmatrices2D(N, r, s)
    fmask = face_mask_2d(N, r, s)

    x, y, rx, sx, ry, sy, J, nx, ny, sJ, Fscale = build_geometry_2d(
        VX, VY, EToV, r, s, Dr, Ds, fmask
    )

    assert np.all(J > 0.0)
    for k in range(J.shape[0]):
        np.testing.assert_allclose(J[k], J[k, 0], atol=1e-12)


def test_normals_unit_length_and_fscale_positive():
    data = setup_2d(N=2, nx=3, ny=3)
    nxv = data["nx"]
    nyv = data["ny"]
    Fscale = data["Fscale"]

    np.testing.assert_allclose(np.sqrt(nxv**2 + nyv**2), 1.0, atol=1e-12)
    assert np.all(Fscale > 0.0)
