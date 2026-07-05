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


def _bell_gradient_error(N, nx, ny):
    data = setup_2d(N=N, nx=nx, ny=ny, periodic=True)
    x = data["x"]
    y = data["y"]

    x0, y0 = 0.43, 0.31
    sigma = 0.18
    dx = x - x0
    dy = y - y0
    u = np.exp(-(dx**2 + dy**2) / sigma**2)
    ux_exact = -2.0 * dx * u / sigma**2
    uy_exact = -2.0 * dy * u / sigma**2

    ur = u @ data["Dr"].T
    us = u @ data["Ds"].T
    ux = data["rx"] * ur + data["sx"] * us
    uy = data["ry"] * ur + data["sy"] * us

    err_x = np.sqrt(np.mean((ux - ux_exact) ** 2))
    err_y = np.sqrt(np.mean((uy - uy_exact) ** 2))
    return err_x, err_y


def test_gradient_operators_bell_curve_error_decreases_with_refinement():
    expected_factor = 4.0
    max_refined_error = 0.08

    coarse_x, coarse_y = _bell_gradient_error(N=3, nx=4, ny=4)
    fine_x, fine_y = _bell_gradient_error(N=3, nx=8, ny=8)

    assert fine_x < coarse_x / expected_factor
    assert fine_y < coarse_y / expected_factor
    assert fine_x < max_refined_error
    assert fine_y < max_refined_error
