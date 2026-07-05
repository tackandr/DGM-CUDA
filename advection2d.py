"""
2D linear advection solver on triangular DG elements.

Equation:
    u_t + a_x u_x + a_y u_y = 0
"""

import numpy as np

from mesh2d import setup_2d


def upwind_flux(u_minus, u_plus, an):
    """Scalar upwind numerical flux for normal speed an."""
    return 0.5 * (an * (u_minus + u_plus) - np.abs(an) * (u_plus - u_minus))


def advection_rhs(
    u,
    ax,
    ay,
    Dr,
    Ds,
    LIFT,
    rx,
    sx,
    ry,
    sy,
    J,
    nx,
    ny,
    Fscale,
    EToE,
    EToF,
    fmask,
    bc_type="periodic",
    u_bc=None,
):
    """Return DG residual for 2D linear advection."""
    K, Np = u.shape
    Nfaces = 3
    Nfp = len(fmask[0])

    ur = u @ Dr.T
    us = u @ Ds.T

    ux = rx * ur + sx * us
    uy = ry * ur + sy * us
    rhs = -(ax * ux + ay * uy)

    face_flux_term = np.zeros((K, Nfaces * Nfp), dtype=float)

    for f in range(Nfaces):
        ids = fmask[f]
        sl = slice(f * Nfp, (f + 1) * Nfp)

        uM = u[:, ids]
        nxf = nx[:, sl]
        nyf = ny[:, sl]
        an = ax * nxf + ay * nyf

        uP = np.empty_like(uM)
        for k in range(K):
            kn = EToE[k, f]
            fn = EToF[k, f]
            if (kn == k and fn == f) and bc_type != "periodic":
                if bc_type == "transmissive":
                    uP[k] = uM[k]
                elif bc_type == "dirichlet":
                    if u_bc is None:
                        raise ValueError("u_bc must be provided for dirichlet bc")
                    uP[k] = u_bc[k, ids]
                else:
                    raise ValueError("Unsupported bc_type")
            else:
                # Opposite orientation across a shared edge for CCW triangles
                uP[k] = u[kn, fmask[fn][::-1]]

        flux = upwind_flux(uM, uP, an)
        face_flux_term[:, sl] = Fscale[:, sl] * (flux - an * uM)

    rhs += (LIFT @ face_flux_term.T).T
    return rhs


def rk4_step_advection(u, dt, rhs_fn):
    """Advance one classical RK4 step."""
    k1 = rhs_fn(u)
    k2 = rhs_fn(u + 0.5 * dt * k1)
    k3 = rhs_fn(u + 0.5 * dt * k2)
    k4 = rhs_fn(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _stable_dt(ax, ay, Fscale, N, cfl):
    speed_n = np.sqrt(ax * ax + ay * ay)
    speed_n = max(speed_n, 1e-14)
    scale = np.max(Fscale)
    return cfl / (speed_n * scale * (N + 1) ** 2)


def simulate_advection_2d(
    N,
    nx,
    ny,
    t_final,
    ax=1.0,
    ay=0.3,
    cfl=0.25,
    periodic=True,
    u_init=None,
):
    """Run a 2D advection simulation on a uniform triangular mesh."""
    data = setup_2d(N, nx, ny, periodic=periodic)

    x = data["x"]
    y = data["y"]
    if u_init is None:
        u_init = lambda xv, yv: np.sin(2 * np.pi * xv) * np.sin(2 * np.pi * yv)

    u = np.asarray(u_init(x, y), dtype=float)

    def rhs(v):
        return advection_rhs(
            v,
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
            bc_type="periodic" if periodic else "transmissive",
        )

    t = 0.0
    dt0 = _stable_dt(ax, ay, data["Fscale"], N, cfl)

    while t < t_final:
        dt = min(dt0, t_final - t)
        if dt <= 0:
            break
        u = rk4_step_advection(u, dt, rhs)
        t += dt

    return data, u, t
