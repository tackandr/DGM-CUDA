"""
Mesh, operator and connectivity helpers for 1D DG solvers.

Provides the three building-block routines needed to set up any 1D DG
simulation on a uniform mesh:

    build_mesh          -- physical node coordinates, shape (M, N+1)
    build_operators     -- V, D, LIFT matrices for the reference element
    build_connectivity  -- EToE / EToF connectivity arrays, shape (M, 2)
"""

import numpy as np

from dgm1d import (
    legendreGL,
    Vandermonde1D,
    DMatrix1D,
    Lift1D,
)


def build_mesh(M, N, x_left=0.0, x_right=2.0):
    """Return the physical node coordinates for a uniform mesh.

    The domain [x_left, x_right] is divided into M equal elements.  Within
    each element the N+1 Gauss-Lobatto-Legendre (GLL) nodes are mapped from
    the reference element [-1, 1] to the physical element [a, b].

    Parameters
    ----------
    M : int
        Number of elements.
    N : int
        Polynomial order; each element has N+1 nodes.
    x_left, x_right : float
        Domain boundaries.

    Returns
    -------
    x : ndarray, shape (M, N+1)
        Physical coordinates.  x[m, i] is the i-th node of element m.
    r : ndarray, shape (N+1,)
        GLL reference nodes on [-1, 1].
    """
    r = legendreGL(N)
    boundaries = np.linspace(x_left, x_right, M + 1)
    x = np.empty((M, N + 1))
    for m in range(M):
        a, b = boundaries[m], boundaries[m + 1]
        x[m] = 0.5 * (b - a) * r + 0.5 * (a + b)
    return x, r


def build_operators(N, r):
    """Return the DG operators for the reference element of order N.

    Parameters
    ----------
    N : int
        Polynomial order.
    r : ndarray, shape (N+1,)
        GLL reference nodes (e.g. from legendreGL(N)).

    Returns
    -------
    V : ndarray, shape (N+1, N+1)
        Vandermonde matrix.
    D : ndarray, shape (N+1, N+1)
        Nodal differentiation matrix.  Applying D @ u_ref gives the derivative
        of the interpolating polynomial at the reference nodes.
    LIFT : ndarray, shape (N+1, 2)
        Lift operator mapping two boundary flux values to N+1 nodal residuals.
    """
    V = Vandermonde1D(N, r)
    D = DMatrix1D(N, r)
    LIFT = Lift1D(N, V)
    return V, D, LIFT


def build_connectivity(M):
    """Return element-to-element and element-to-face connectivity arrays.

    For a 1D periodic mesh of M elements, each element has two faces:
      face 0 = left face,  face 1 = right face.

    The connectivity is:
      Left  neighbor of element m : element (m-1) % M, face 1 (its right face)
      Right neighbor of element m : element (m+1) % M, face 0 (its left face)

    Parameters
    ----------
    M : int
        Number of elements.

    Returns
    -------
    EToE : ndarray, shape (M, 2), dtype int
        EToE[m, f] is the index of the element neighboring element m at face f.
    EToF : ndarray, shape (M, 2), dtype int
        EToF[m, f] is the face index on the neighboring element EToE[m, f]
        that is shared with element m at face f.
    """
    idx = np.arange(M)
    EToE = np.stack([(idx - 1) % M, (idx + 1) % M], axis=1).astype(int)
    EToF = np.zeros((M, 2), dtype=int)
    EToF[:, 0] = 1  # left face's neighbor contributes its right face
    # EToF[:, 1] = 0 already (right face's neighbor contributes its left face)
    return EToE, EToF
