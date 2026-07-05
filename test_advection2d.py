"""Tests for the 2D DG linear advection solver."""

import numpy as np

from advection2d import advection_rhs, simulate_advection_2d
from mesh2d import setup_2d


def _rhs_with_data(u, data, ax=1.0, ay=0.4, bc_type="periodic"):
    return advection_rhs(
        u,
        ax,
        ay,
        data["Dr"],
        data["Ds"],
        data["LIFT"],
        data["rx"],
        data["sx"],
        data["ry"],
        data["sy"],
        data["J"],
        data["nx"],
        data["ny"],
        data["Fscale"],
        data["EToE"],
        data["EToF"],
        data["fmask"],
        bc_type=bc_type,
    )


def test_constant_solution_has_zero_rhs_periodic():
    data = setup_2d(N=2, nx=4, ny=3, periodic=True)
    u = np.full_like(data["x"], 2.5)
    rhs = _rhs_with_data(u, data, ax=0.7, ay=-0.2, bc_type="periodic")
    np.testing.assert_allclose(rhs, 0.0, atol=1e-12)


def test_constant_solution_has_zero_rhs_transmissive():
    data = setup_2d(N=2, nx=4, ny=3, periodic=False)
    u = np.full_like(data["x"], -1.0)
    rhs = _rhs_with_data(u, data, ax=1.1, ay=0.3, bc_type="transmissive")
    np.testing.assert_allclose(rhs, 0.0, atol=1e-12)


def test_simulate_returns_expected_shapes_and_time():
    N = 2
    nx, ny = 4, 3
    tfinal = 0.05
    data, u, t = simulate_advection_2d(N=N, nx=nx, ny=ny, t_final=tfinal, periodic=True)

    K = 2 * nx * ny
    Np = (N + 1) * (N + 2) // 2
    assert u.shape == (K, Np)
    assert data["x"].shape == (K, Np)
    assert t == tfinal


def test_semidiscrete_rhs_error_reduces_with_refinement():
    N = 2
    ax, ay = 0.7, 0.4

    def u_fn(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def rhs_exact(x, y):
        ux = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        uy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
        return -(ax * ux + ay * uy)

    errs = []
    for nx, ny in [(4, 4), (8, 8)]:
        data = setup_2d(N=N, nx=nx, ny=ny, periodic=False)
        u = u_fn(data["x"], data["y"])
        rhs = _rhs_with_data(u, data, ax=ax, ay=ay, bc_type="transmissive")
        err = np.sqrt(np.mean((rhs - rhs_exact(data["x"], data["y"])) ** 2))
        errs.append(err)

    assert errs[1] < errs[0]
