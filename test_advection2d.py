"""Tests for the 2D DG linear advection solver."""

import numpy as np

from advection2d import advection_rhs, simulate_advection_2d
from mesh2d import setup_2d


def _rhs_with_data(u, data, ax=1.0, ay=0.4, bc_type="periodic", flux_type="upwind"):
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
        flux_type=flux_type,
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


def test_x_only_advection_stable():
    """X-only advection (ax=1, ay=0) with periodic IC should stay bounded at t=0.5.

    After t=0.5 the sine wave should have shifted half a period, so the
    solution should be close to the *negative* of the initial condition
    (sin(2π(x-0.5)) = -sin(2πx)).  This test isolates any instability that
    arises purely from x-direction advection.
    """
    N = 2
    nx, ny = 4, 4
    t_final = 0.5

    def u_init(x, y):
        return np.sin(2 * np.pi * x)

    data, u, t = simulate_advection_2d(
        N=N, nx=nx, ny=ny, t_final=t_final, ax=1.0, ay=0.0,
        cfl=0.1, periodic=True, u_init=u_init,
    )
    u_exact = -np.sin(2 * np.pi * data["x"])

    l2_init = np.sqrt(np.mean(np.sin(2 * np.pi * data["x"]) ** 2))
    l2_err = np.sqrt(np.mean((u - u_exact) ** 2))
    # Solution should not blow up: error relative to the initial norm < 1
    assert l2_err / l2_init < 1.0, (
        f"x-only advection unstable: relative L2 error = {l2_err/l2_init:.3f}"
    )


def test_y_only_advection_stable():
    """Y-only advection (ax=0, ay=1) with periodic IC should stay bounded at t=0.5.

    After t=0.5 the sine wave should have shifted half a period, so the
    solution should be close to the *negative* of the initial condition
    (sin(2π(y-0.5)) = -sin(2πy)).  This test isolates any instability that
    arises purely from y-direction advection.
    """
    N = 2
    nx, ny = 4, 4
    t_final = 0.5

    def u_init(x, y):
        return np.sin(2 * np.pi * y)

    data, u, t = simulate_advection_2d(
        N=N, nx=nx, ny=ny, t_final=t_final, ax=0.0, ay=1.0,
        cfl=0.1, periodic=True, u_init=u_init,
    )
    u_exact = -np.sin(2 * np.pi * data["y"])

    l2_init = np.sqrt(np.mean(np.sin(2 * np.pi * data["y"]) ** 2))
    l2_err = np.sqrt(np.mean((u - u_exact) ** 2))
    # Solution should not blow up: error relative to the initial norm < 1
    assert l2_err / l2_init < 1.0, (
        f"y-only advection unstable: relative L2 error = {l2_err/l2_init:.3f}"
    )


def test_x_only_advection_central_flux_bounded():
    N = 2
    nx, ny = 4, 4
    t_final = 0.5

    def u_init(x, y):
        return np.sin(2 * np.pi * x)

    data, u, t = simulate_advection_2d(
        N=N, nx=nx, ny=ny, t_final=t_final, ax=1.0, ay=0.0,
        cfl=0.1, periodic=True, u_init=u_init, flux_type="central",
    )
    assert np.isclose(data["VX"].min(), 0.0)
    assert np.isclose(data["VX"].max(), 1.0)
    u_exact = -np.sin(2 * np.pi * data["x"])

    l2_init = np.sqrt(np.mean(np.sin(2 * np.pi * data["x"]) ** 2))
    l2_err = np.sqrt(np.mean((u - u_exact) ** 2))
    assert l2_err / l2_init < 0.5, (
        f"x-only central advection unstable: relative L2 error = {l2_err/l2_init:.3f}"
    )
    assert np.max(np.abs(u)) < 1.5


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
