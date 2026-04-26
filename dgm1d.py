"""
1D Discontinuous Galerkin Method operators.

Provides functions to build the core operators needed for a 1D nodal DG
discretisation on a reference element [-1, 1]:

    legendreGL        -- Gauss-Lobatto-Legendre quadrature nodes
    Vandermonde1D     -- Vandermonde matrix of normalised Legendre polynomials
    GradVandermonde1D -- Gradient (derivative) of the Vandermonde matrix
    DMatrix1D         -- Nodal differentiation matrix
    Jacobian          -- Mapping Jacobian from reference to physical element
"""

import numpy as np
from numpy import linalg as LA
import scipy.special as sps


def legendreGL(N):
    """Return the N+1 Gauss-Lobatto-Legendre (GLL) nodes of order N.

    The GLL nodes consist of the two endpoints ±1 together with the N-1
    interior Gauss-Legendre nodes of order N-1, giving N+1 nodes in total.
    These nodes make the Vandermonde matrix square and invertible for degree-N
    polynomial approximation.

    Parameters
    ----------
    N : int
        Polynomial order.  The returned array contains N+1 nodes.

    Returns
    -------
    r : ndarray, shape (N+1,)
        GLL nodes in ascending order on [-1, 1].
    """
    if N == 1:
        return np.array([-1.0, 1.0])
    r_interior, _ = np.polynomial.legendre.leggauss(N - 1)
    return np.concatenate([[-1.0], r_interior, [1.0]])


def Vandermonde1D(N, r):
    """Return the Vandermonde matrix for degree-N normalised Legendre polynomials.

    Entry V[i, j] = P_j(r_i) * sqrt((2j+1)/2), where P_j is the Legendre
    polynomial of degree j.  The normalisation makes the polynomials
    orthonormal on [-1, 1] so that V^T V = I when r are the exact GL nodes.

    Parameters
    ----------
    N : int
        Polynomial order; the matrix has N+1 columns (degrees 0 … N).
    r : array_like, shape (N+1,)
        Evaluation nodes.

    Returns
    -------
    V : ndarray, shape (len(r), N+1)
        Vandermonde matrix.
    """
    r = np.asarray(r, dtype=float)
    # legendre_p_all returns shape (1, N+1, len(r))
    vals = sps.legendre_p_all(N, r)   # shape (1, N+1, len(r))
    norm = np.sqrt((2 * np.arange(N + 1) + 1) / 2.0)  # shape (N+1,)
    # V[i, j] = vals[0, j, i] * norm[j]
    V = (vals[0] * norm[:, np.newaxis]).T
    return V


def GradVandermonde1D(N, r):
    """Return the gradient of the Vandermonde matrix for the Nth order element.

    Entry Vr[i, j] = P_j'(r_i) * sqrt((2j+1)/2), the derivative of the
    normalised Legendre polynomial of degree j evaluated at node r_i.

    Parameters
    ----------
    N : int
        Polynomial order; the matrix has N+1 columns (degrees 0 … N).
    r : array_like, shape (N+1,)
        Evaluation nodes.

    Returns
    -------
    Vr : ndarray, shape (len(r), N+1)
        Gradient Vandermonde matrix.
    """
    r = np.asarray(r, dtype=float)
    # legendre_p_all with diff_n=1 returns shape (2, N+1, len(r))
    # index [0] = values, index [1] = first derivatives
    derivs = sps.legendre_p_all(N, r, diff_n=1)  # shape (2, N+1, len(r))
    norm = np.sqrt((2 * np.arange(N + 1) + 1) / 2.0)  # shape (N+1,)
    # Vr[i, j] = derivs[1, j, i] * norm[j]
    Vr = (derivs[1] * norm[:, np.newaxis]).T
    return Vr


def DMatrix1D(N, r, V):
    """Return the nodal differentiation matrix for the Nth order reference element.

    Computes D = Vr * V^{-1}, where Vr is the gradient Vandermonde matrix and
    V is the Vandermonde matrix.  Applying D to a vector of nodal values gives
    the derivative of the interpolating polynomial at those nodes.

    Parameters
    ----------
    N : int
        Polynomial order.
    r : array_like, shape (N+1,)
        GLL nodes (e.g. from legendreGL(N)).
    V : ndarray, shape (N+1, N+1)
        Vandermonde matrix (e.g. from Vandermonde1D(N, r)).

    Returns
    -------
    D : ndarray, shape (N+1, N+1)
        Differentiation matrix.
    """
    return np.matmul(GradVandermonde1D(N, r), LA.inv(V))


def Jacobian(x, Dr):
    """Return the Jacobian of the mapping from the reference to a physical element.

    Computes J = Dr @ x, where x is the vector of physical node coordinates
    and Dr is the differentiation matrix on the reference element.

    Parameters
    ----------
    x : array_like, shape (N+1,)
        Physical node coordinates.
    Dr : ndarray, shape (N+1, N+1)
        Differentiation matrix on the reference element.

    Returns
    -------
    J : ndarray, shape (N+1,)
        Jacobian dx/dr evaluated at each node.
    """
    return np.matmul(Dr, x)
