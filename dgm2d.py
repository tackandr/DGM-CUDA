"""
2D simplex (triangle) DG reference-element operators.

Implements a nodal DG operator stack on the reference triangle with vertices
(-1,-1), (1,-1), (-1,1), following the Hesthaven & Warburton simplex layout:

    simplex_nodes_2d     -- nodal points, shape (Np,), (Np,)
    modal_basis_2d       -- orthonormal modal basis at points
    grad_modal_basis_2d  -- gradients of modal basis wrt (r,s)
    Vandermonde2D        -- nodal Vandermonde matrix
    GradVandermonde2D    -- nodal gradient Vandermonde matrices
    Dmatrices2D          -- nodal differentiation operators Dr, Ds
    face_mask_2d         -- indices of nodes on the 3 faces
    Lift2D               -- triangle lift matrix
"""

from functools import lru_cache

import numpy as np
from numpy import linalg as LA

from dgm1d import Vandermonde1D, legendreGL


def num_nodes_2d(N):
    """Return number of nodes for total-degree N polynomial space on triangle."""
    return (N + 1) * (N + 2) // 2


def monomial_powers_2d(N):
    """Return monomial powers (i,j) with i+j<=N in a stable order."""
    powers = []
    for n in range(N + 1):
        for j in range(n + 1):
            i = n - j
            powers.append((i, j))
    return powers


def _warp_lagrange(N, t):
    """Lagrange-interpolated displacement from equispaced to GLL nodes at t.

    Evaluates the degree-N polynomial that interpolates (gll_i - eq_i) at
    the N+1 equispaced nodes eq_i = -1 + 2*i/N on [-1, 1], where gll_i are
    the corresponding GLL nodes.  The result is the "un-normalised" warp
    from H&W Section 6.3 used by the warp/blend simplex node construction.

    Parameters
    ----------
    N : int
        Polynomial order.
    t : array_like
        Evaluation positions.

    Returns
    -------
    warp : ndarray, same shape as t
        Displacement values.
    """
    gll = legendreGL(N)
    eq = np.linspace(-1.0, 1.0, N + 1)
    delta = gll - eq  # vanishes at both endpoints

    t = np.asarray(t, dtype=float).ravel()
    warp = np.zeros(len(t))
    for i in range(N + 1):
        li = np.ones(len(t))
        for j in range(N + 1):
            if i != j:
                li *= (t - eq[j]) / (eq[i] - eq[j])
        warp += delta[i] * li
    return warp


def simplex_nodes_2d(N):
    """Return order-N nodal points on the reference triangle.

    Uses a warp/blend construction (Hesthaven & Warburton, Sec. 6.3): starts
    from equispaced barycentric lattice nodes and applies a smooth warp so
    that all three edges carry the 1D Gauss-Lobatto-Legendre (GLL) node
    distribution matching ``legendreGL(N)``.

    Reference triangle has vertices v0=(-1,-1), v1=(1,-1), v2=(-1,1).

    Face labelling (same as ``face_mask_2d``):
      F0 (L2=0, s=-1)  : from v0 to v1, tangential parameter t = L1 - L0
      F1 (L0=0, r+s=0) : from v1 to v2, tangential parameter t = L2 - L1
      F2 (L1=0, r=-1)  : from v2 to v0, tangential parameter t = L0 - L2
    """
    if N < 0:
        raise ValueError("N must be non-negative")
    if N == 0:
        return np.array([-1.0]), np.array([-1.0])

    # Equispaced barycentric nodes
    L0_list, L1_list, L2_list = [], [], []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            l1 = i / N
            l2 = j / N
            l0 = 1.0 - l1 - l2
            L0_list.append(l0)
            L1_list.append(l1)
            L2_list.append(l2)

    L0 = np.asarray(L0_list)
    L1 = np.asarray(L1_list)
    L2 = np.asarray(L2_list)

    # Equispaced reference coordinates
    r0 = -L0 + L1 - L2
    s0 = -L0 - L1 + L2

    # Tangential parameters along each face
    t_F0 = L1 - L0
    t_F1 = L2 - L1
    t_F2 = L0 - L2

    # Blending factors: equal to (1 - t^2) when a node lies exactly on face k
    blend_F0 = 4.0 * L0 * L1
    blend_F1 = 4.0 * L1 * L2
    blend_F2 = 4.0 * L0 * L2

    # Normalised warp: warp_unnorm / (1 - t^2).
    # At vertices |t|=1 the blend is zero so the singularity is removable;
    # we safely return 0 there.  The threshold guards against round-off in t.
    _VERTEX_TOL = 1.0 - 1e-10

    def _warp_norm(t_arr):
        t_arr = np.asarray(t_arr)
        w = _warp_lagrange(N, t_arr)
        denom = 1.0 - t_arr ** 2
        safe = np.abs(t_arr) < _VERTEX_TOL
        return np.where(safe, w / np.where(safe, denom, 1.0), 0.0)

    W0 = blend_F0 * _warp_norm(t_F0)
    W1 = blend_F1 * _warp_norm(t_F1)
    W2 = blend_F2 * _warp_norm(t_F2)

    # (r, s) displacement contributed by each face:
    #   F0 -> (+W0, 0), F1 -> (-W1, +W1), F2 -> (0, -W2)
    return r0 + W0 - W1, s0 + W1 - W2


def _monomial_values(r, s, powers):
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    out = np.empty((r.size, len(powers)), dtype=float)
    for k, (i, j) in enumerate(powers):
        out[:, k] = (r ** i) * (s ** j)
    return out


def _monomial_grads(r, s, powers):
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    vr = np.empty((r.size, len(powers)), dtype=float)
    vs = np.empty((r.size, len(powers)), dtype=float)
    for k, (i, j) in enumerate(powers):
        if i == 0:
            vr[:, k] = 0.0
        else:
            vr[:, k] = i * (r ** (i - 1)) * (s ** j)
        if j == 0:
            vs[:, k] = 0.0
        else:
            vs[:, k] = j * (r ** i) * (s ** (j - 1))
    return vr, vs


def _triangle_quadrature(order):
    """Tensor-product Gauss quadrature mapped to the reference triangle."""
    order = max(int(order), 2)
    xg, wg = np.polynomial.legendre.leggauss(order)
    yg, vg = np.polynomial.legendre.leggauss(order)

    # map [-1,1] -> [0,1]
    xi = 0.5 * (xg + 1.0)
    wx = 0.5 * wg
    eta_ref = 0.5 * (yg + 1.0)
    wy_ref = 0.5 * vg

    r_pts = []
    s_pts = []
    w_pts = []

    for ix, x in enumerate(xi):
        # eta in [0, 1-x]
        eta = (1.0 - x) * eta_ref
        wy = (1.0 - x) * wy_ref
        for iy, y in enumerate(eta):
            # (x,y) in unit simplex -> (r,s) in reference simplex
            r = 2.0 * x - 1.0
            s = 2.0 * y - 1.0
            # Jacobian unit-simplex -> reference-simplex equals 4
            w = wx[ix] * wy[iy] * 4.0
            r_pts.append(r)
            s_pts.append(s)
            w_pts.append(w)

    return np.asarray(r_pts), np.asarray(s_pts), np.asarray(w_pts)


@lru_cache(maxsize=16)
def _orthonormal_transform_2d(N):
    """Return powers and transform matrix mapping monomials -> orthonormal basis."""
    powers = monomial_powers_2d(N)
    rq, sq, wq = _triangle_quadrature(max(2 * N + 4, 12))
    Mq = _monomial_values(rq, sq, powers)

    # Gram matrix in monomial basis
    G = (Mq.T * wq) @ Mq
    G = 0.5 * (G + G.T)

    L = LA.cholesky(G)
    C = LA.solve(L.T, np.eye(L.shape[0]))
    return powers, C


def modal_basis_2d(N, r, s):
    """Return orthonormal modal basis values on the reference triangle."""
    powers, C = _orthonormal_transform_2d(N)
    M = _monomial_values(r, s, powers)
    return M @ C


def grad_modal_basis_2d(N, r, s):
    """Return gradients of orthonormal modal basis wrt (r,s)."""
    powers, C = _orthonormal_transform_2d(N)
    Mr, Ms = _monomial_grads(r, s, powers)
    return Mr @ C, Ms @ C


def Vandermonde2D(N, r, s):
    """Return Vandermonde matrix on reference triangle nodes/points."""
    return modal_basis_2d(N, r, s)


def GradVandermonde2D(N, r, s):
    """Return gradient Vandermonde matrices Vr, Vs."""
    return grad_modal_basis_2d(N, r, s)


def Dmatrices2D(N, r, s):
    """Return nodal differentiation matrices Dr and Ds."""
    V = Vandermonde2D(N, r, s)
    Vr, Vs = GradVandermonde2D(N, r, s)
    Dr = LA.solve(V.T, Vr.T).T
    Ds = LA.solve(V.T, Vs.T).T
    return Dr, Ds


def face_mask_2d(N, r, s, tol=1e-12):
    """Return node indices on the three faces of the reference triangle.

    Face ordering:
      0: s = -1   (v0 -> v1)
      1: r + s = 0 (v1 -> v2)
      2: r = -1   (v2 -> v0)
    """
    r = np.asarray(r)
    s = np.asarray(s)

    f0 = np.where(np.abs(s + 1.0) < tol)[0]
    f1 = np.where(np.abs(r + s) < tol)[0]
    f2 = np.where(np.abs(r + 1.0) < tol)[0]

    # orient consistently with local face ordering
    f0 = f0[np.argsort(r[f0])]         # v0(-1,-1) -> v1(1,-1)
    f1 = f1[np.argsort(s[f1])]         # v1(1,-1) -> v2(-1,1)
    f2 = f2[np.argsort(-s[f2])]        # v2(-1,1) -> v0(-1,-1)

    expected = N + 1
    if not (len(f0) == len(f1) == len(f2) == expected):
        raise RuntimeError("Face-node extraction failed; unexpected node counts per face")

    return [f0.astype(int), f1.astype(int), f2.astype(int)]


def _face_coordinate_t(face_id, r_face, s_face):
    if face_id == 0:
        return r_face
    if face_id == 1:
        return s_face
    return -s_face


def Lift2D(N, r, s, fmask=None):
    """Return triangle lift matrix mapping face flux dofs to volume residuals.

    Returns
    -------
    LIFT : ndarray, shape (Np, 3*(N+1))
    """
    V = Vandermonde2D(N, r, s)
    Np = V.shape[0]
    Nfaces = 3
    Nfp = N + 1

    if fmask is None:
        fmask = face_mask_2d(N, r, s)

    Emat = np.zeros((Np, Nfaces * Nfp), dtype=float)

    for f in range(Nfaces):
        ids = fmask[f]
        t = _face_coordinate_t(f, r[ids], s[ids])
        V1d = Vandermonde1D(N, t)
        M_edge = LA.inv(V1d @ V1d.T)
        Emat[np.ix_(ids, np.arange(f * Nfp, (f + 1) * Nfp))] = M_edge

    return V @ (V.T @ Emat)


def build_reference_operators_2d(N):
    """Convenience builder returning all reference-element data for order N."""
    r, s = simplex_nodes_2d(N)
    V = Vandermonde2D(N, r, s)
    Vr, Vs = GradVandermonde2D(N, r, s)
    Dr, Ds = Dmatrices2D(N, r, s)
    fmask = face_mask_2d(N, r, s)
    LIFT = Lift2D(N, r, s, fmask)
    return r, s, V, Vr, Vs, Dr, Ds, fmask, LIFT
