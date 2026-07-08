"""
Mesh, connectivity, and geometric-factor helpers for 2D triangular DG solvers.

Public API
----------
build_uniform_tri_mesh  -- structured triangular mesh (VX, VY, EToV)
build_connectivity_2d   -- element-to-element / element-to-face connectivity
build_operators_2d      -- reference-element operator bundle
build_geometry_2d       -- mapped nodes and geometric factors
setup_2d                -- one-shot setup helper for future 2D solvers
"""

import numpy as np

from dgm2d import build_reference_operators_2d


def build_uniform_tri_mesh(nx, ny, x_left=0.0, x_right=1.0, y_bottom=0.0, y_top=1.0):
    """Return a structured triangular mesh over a rectangular domain.

    Each Cartesian cell is split into two triangles with consistent CCW ordering.
    """
    xg = np.linspace(x_left, x_right, nx + 1)
    yg = np.linspace(y_bottom, y_top, ny + 1)

    X, Y = np.meshgrid(xg, yg, indexing="xy")
    VX = X.ravel()
    VY = Y.ravel()

    def vid(i, j):
        return j * (nx + 1) + i

    elems = []
    for j in range(ny):
        for i in range(nx):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            elems.append([v00, v10, v11])  # lower-right triangle
            elems.append([v00, v11, v01])  # upper-left triangle

    EToV = np.asarray(elems, dtype=int)
    return VX, VY, EToV


def _edge_vertices(EToV):
    """Return local face->vertex pairs for each element."""
    # Faces match reference face ordering used in dgm2d.face_mask_2d:
    # f0: v0-v1, f1: v1-v2, f2: v2-v0
    return np.stack([
        EToV[:, [0, 1]],
        EToV[:, [1, 2]],
        EToV[:, [2, 0]],
    ], axis=1)


def _canon_point(x, y, periodic, domain, tol=1e-12):
    if not periodic:
        return x, y

    x_left, x_right, y_bottom, y_top = domain
    Lx = x_right - x_left
    Ly = y_top - y_bottom

    xc = ((x - x_left) % Lx) + x_left
    yc = ((y - y_bottom) % Ly) + y_bottom

    if np.isclose(xc, x_right, atol=tol):
        xc = x_left
    if np.isclose(yc, y_top, atol=tol):
        yc = y_bottom
    return xc, yc


def build_connectivity_2d(EToV, VX, VY, periodic=False, domain=None, tol=1e-12):
    """Return element-to-element and element-to-face connectivity arrays.

    Boundary faces are self-connected by default; with periodic=True they are
    wrapped using domain box coordinates (x_left, x_right, y_bottom, y_top).
    """
    K = EToV.shape[0]
    Nfaces = 3
    EToE = np.tile(np.arange(K)[:, None], (1, Nfaces))
    EToF = np.tile(np.arange(Nfaces)[None, :], (K, 1))

    if periodic and domain is None:
        raise ValueError("domain must be provided when periodic=True")

    face_verts = _edge_vertices(EToV)
    edge_map = {}

    for k in range(K):
        for f in range(Nfaces):
            va, vb = face_verts[k, f]
            xm = 0.5 * (VX[va] + VX[vb])
            ym = 0.5 * (VY[va] + VY[vb])
            xm_c, ym_c = _canon_point(xm, ym, periodic, domain, tol)
            key = (round(xm_c, 12), round(ym_c, 12))

            if key in edge_map:
                k2, f2 = edge_map.pop(key)
                EToE[k, f] = k2
                EToF[k, f] = f2
                EToE[k2, f2] = k
                EToF[k2, f2] = f
            else:
                edge_map[key] = (k, f)

    return EToE.astype(int), EToF.astype(int)


def build_operators_2d(N):
    """Return reference-element operators for 2D triangles."""
    return build_reference_operators_2d(N)


def _map_nodes_affine(r, s, x0, y0, x1, y1, x2, y2):
    # barycentric shape functions on reference triangle
    l0 = -0.5 * (r + s)
    l1 = 0.5 * (1.0 + r)
    l2 = 0.5 * (1.0 + s)
    x = l0 * x0 + l1 * x1 + l2 * x2
    y = l0 * y0 + l1 * y1 + l2 * y2
    return x, y


def build_geometry_2d(VX, VY, EToV, r, s, Dr, Ds, fmask):
    """Return mapped nodal coordinates and geometric factors for each element."""
    K = EToV.shape[0]
    Np = r.size
    Nfaces = 3
    Nfp = len(fmask[0])

    x = np.empty((K, Np), dtype=float)
    y = np.empty((K, Np), dtype=float)

    for k in range(K):
        v0, v1, v2 = EToV[k]
        x[k], y[k] = _map_nodes_affine(
            r,
            s,
            VX[v0], VY[v0],
            VX[v1], VY[v1],
            VX[v2], VY[v2],
        )

    xr = x @ Dr.T
    xs = x @ Ds.T
    yr = y @ Dr.T
    ys = y @ Ds.T

    J = xr * ys - xs * yr
    rx = ys / J
    sx = -yr / J
    ry = -xs / J
    sy = xr / J

    nx = np.empty((K, Nfaces * Nfp), dtype=float)
    ny = np.empty((K, Nfaces * Nfp), dtype=float)
    sJ = np.empty((K, Nfaces * Nfp), dtype=float)
    Fscale = np.empty((K, Nfaces * Nfp), dtype=float)

    for f in range(Nfaces):
        ids = fmask[f]
        sl = slice(f * Nfp, (f + 1) * Nfp)

        if f == 0:
            nxf = yr[:, ids]
            nyf = -xr[:, ids]
        elif f == 1:
            nxf = ys[:, ids] - yr[:, ids]
            nyf = xr[:, ids] - xs[:, ids]
        else:
            nxf = -ys[:, ids]
            nyf = xs[:, ids]

        sjf = np.sqrt(nxf**2 + nyf**2)
        nxf = nxf / sjf
        nyf = nyf / sjf

        nx[:, sl] = nxf
        ny[:, sl] = nyf
        sJ[:, sl] = sjf
        Fscale[:, sl] = sjf / J[:, ids]

    return x, y, rx, sx, ry, sy, J, nx, ny, sJ, Fscale


def setup_2d(N, nx, ny, x_left=0.0, x_right=1.0, y_bottom=0.0, y_top=1.0, periodic=False):
    """Build a complete 2D DG setup bundle for triangular elements."""
    VX, VY, EToV = build_uniform_tri_mesh(nx, ny, x_left, x_right, y_bottom, y_top)
    domain = (x_left, x_right, y_bottom, y_top)
    EToE, EToF = build_connectivity_2d(EToV, VX, VY, periodic=periodic, domain=domain)

    r, s, V, Vr, Vs, Dr, Ds, fmask, LIFT = build_operators_2d(N)
    x, y, rx, sx, ry, sy, J, nxv, nyv, sJ, Fscale = build_geometry_2d(VX, VY, EToV, r, s, Dr, Ds, fmask)

    return {
        "VX": VX,
        "VY": VY,
        "EToV": EToV,
        "EToE": EToE,
        "EToF": EToF,
        "r": r,
        "s": s,
        "V": V,
        "Vr": Vr,
        "Vs": Vs,
        "Dr": Dr,
        "Ds": Ds,
        "fmask": fmask,
        "LIFT": LIFT,
        "x": x,
        "y": y,
        "rx": rx,
        "sx": sx,
        "ry": ry,
        "sy": sy,
        "J": J,
        "nx": nxv,
        "ny": nyv,
        "sJ": sJ,
        "Fscale": Fscale,
    }
