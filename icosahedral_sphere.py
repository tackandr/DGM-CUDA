"""
Icosahedral spherical mesh and simple scalar-advection helpers.

Public API
----------
build_icosahedral_sphere_mesh    -- refined spherical icosahedral triangle mesh
build_sphere_connectivity        -- face-neighbor connectivity on closed sphere
build_sphere_geometry            -- cell areas, centers, edge metrics and normals
build_edge_flux_integrals        -- edge-integrated flux weights for a tangent wind
constant_eastward_wind           -- constant-speed eastward tangent wind field
scalar_advection_rhs_sphere      -- first-order upwind FV RHS on spherical cells
simulate_scalar_advection_sphere -- SSP-RK3 scalar transport driver
"""

from functools import lru_cache

import numpy as np

from dgm2d import simplex_nodes_2d

_FACE_VERTEX_IDS = ((0, 1), (1, 2), (2, 0))


def _normalize_rows(v):
    v = np.asarray(v, dtype=float)
    norms = np.linalg.norm(v, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("all vectors must have positive length")
    return v / norms[:, np.newaxis]


def _base_icosahedron():
    golden_ratio = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = np.array(
        [
            (-1.0, golden_ratio, 0.0),
            (1.0, golden_ratio, 0.0),
            (-1.0, -golden_ratio, 0.0),
            (1.0, -golden_ratio, 0.0),
            (0.0, -1.0, golden_ratio),
            (0.0, 1.0, golden_ratio),
            (0.0, -1.0, -golden_ratio),
            (0.0, 1.0, -golden_ratio),
            (golden_ratio, 0.0, -1.0),
            (golden_ratio, 0.0, 1.0),
            (-golden_ratio, 0.0, -1.0),
            (-golden_ratio, 0.0, 1.0),
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=int,
    )
    vertices = _normalize_rows(vertices)
    return vertices, _orient_faces_outward(vertices, faces)


def _orient_faces_outward(vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int).copy()
    for k in range(faces.shape[0]):
        tri = vertices[faces[k]]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.dot(normal, tri.mean(axis=0)) < 0.0:
            faces[k] = [faces[k, 0], faces[k, 2], faces[k, 1]]
    return faces


def build_icosahedral_sphere_mesh(refinement_level=0, radius=1.0):
    """Return a spherical icosahedral triangle mesh."""
    refinement_level = int(refinement_level)
    radius = float(radius)
    if refinement_level < 0:
        raise ValueError("refinement_level must be non-negative")
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    vertices, faces = _base_icosahedron()

    for _ in range(refinement_level):
        vertex_list = vertices.tolist()
        midpoint_cache = {}
        refined_faces = []

        def midpoint_index(i, j):
            key = (min(i, j), max(i, j))
            if key in midpoint_cache:
                return midpoint_cache[key]
            midpoint = np.asarray(vertex_list[i]) + np.asarray(vertex_list[j])
            midpoint = midpoint / np.linalg.norm(midpoint)
            idx = len(vertex_list)
            vertex_list.append(midpoint.tolist())
            midpoint_cache[key] = idx
            return idx

        for a, b, c in faces:
            ab = midpoint_index(a, b)
            bc = midpoint_index(b, c)
            ca = midpoint_index(c, a)
            refined_faces.extend(
                [
                    [a, ab, ca],
                    [ab, b, bc],
                    [ca, bc, c],
                    [ab, bc, ca],
                ]
            )

        vertices = np.asarray(vertex_list, dtype=float)
        faces = _orient_faces_outward(vertices, np.asarray(refined_faces, dtype=int))

    return radius * vertices, faces


def build_sphere_connectivity(faces):
    """Return element-to-element / element-to-face connectivity for a closed sphere."""
    faces = np.asarray(faces, dtype=int)
    K = faces.shape[0]
    EToE = np.tile(np.arange(K)[:, np.newaxis], (1, 3))
    EToF = np.tile(np.arange(3)[np.newaxis, :], (K, 1))
    edge_map = {}

    for k, face in enumerate(faces):
        for f, (i, j) in enumerate(_FACE_VERTEX_IDS):
            key = tuple(sorted((face[i], face[j])))
            if key in edge_map:
                k2, f2 = edge_map.pop(key)
                EToE[k, f] = k2
                EToF[k, f] = f2
                EToE[k2, f2] = k
                EToF[k2, f2] = f
            else:
                edge_map[key] = (k, f)

    if edge_map:
        raise ValueError("faces do not define a closed triangular sphere mesh")

    return EToE.astype(int), EToF.astype(int)


def _spherical_triangle_area_unit(a, b, c):
    det = np.dot(a, np.cross(b, c))
    denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    return 2.0 * np.arctan2(det, denom)


def _great_circle_point(a, b, tau):
    alpha = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if alpha < 1e-14:
        return a.copy(), alpha
    point = (np.sin((1.0 - tau) * alpha) * a + np.sin(tau * alpha) * b) / np.sin(alpha)
    return point / np.linalg.norm(point), alpha


@lru_cache(maxsize=16)
def _simplex_nodes_to_barycentric(order):
    r, s = simplex_nodes_2d(int(order))
    l1 = 0.5 * (r + 1.0)
    l2 = 0.5 * (s + 1.0)
    l0 = 1.0 - l1 - l2
    return np.stack((l0, l1, l2), axis=1)


def build_sphere_geometry(vertices, faces, quadrature_order=6, simplex_order=1):
    """Return geometry and edge-integrated metric data for spherical triangles."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least 2")
    if simplex_order < 1:
        raise ValueError("simplex_order must be at least 1")

    radii = np.linalg.norm(vertices, axis=1)
    radius = float(np.mean(radii))
    if np.any(np.abs(radii - radius) > 1e-10 * max(radius, 1.0)):
        raise ValueError("all vertices must lie on the same sphere")

    unit_vertices = vertices / radius
    EToE, EToF = build_sphere_connectivity(faces)
    K = faces.shape[0]

    centers = np.empty((K, 3), dtype=float)
    areas = np.empty(K, dtype=float)
    # Build once so we can allocate and reuse mapped simplex nodes per element.
    simplex_bary = _simplex_nodes_to_barycentric(simplex_order)
    simplex_points = np.empty((K, simplex_bary.shape[0], 3), dtype=float)
    edge_midpoints = np.empty((K, 3, 3), dtype=float)
    edge_normals = np.empty((K, 3, 3), dtype=float)
    edge_lengths = np.empty((K, 3), dtype=float)

    for k, face in enumerate(faces):
        tri = unit_vertices[face]
        simplex_pts = simplex_bary @ tri
        simplex_pts = simplex_pts / np.linalg.norm(simplex_pts, axis=1, keepdims=True)
        simplex_points[k] = radius * simplex_pts
        center = simplex_pts.mean(axis=0)
        center = center / np.linalg.norm(center)
        centers[k] = radius * center
        areas[k] = radius**2 * _spherical_triangle_area_unit(tri[0], tri[1], tri[2])
        if areas[k] <= 0.0:
            raise ValueError(f"face {k} is not outward-oriented on the sphere")

        for f, (i, j) in enumerate(_FACE_VERTEX_IDS):
            a = tri[i]
            b = tri[j]
            midpoint, alpha = _great_circle_point(a, b, 0.5)
            normal = np.cross(a, b)
            normal = normal / np.linalg.norm(normal)
            if np.dot(normal, center) > 0.0:
                normal = -normal

            edge_midpoints[k, f] = radius * midpoint
            edge_normals[k, f] = normal
            edge_lengths[k, f] = radius * alpha

    return {
        "vertices": vertices,
        "unit_vertices": unit_vertices,
        "faces": faces,
        "EToE": EToE,
        "EToF": EToF,
        "radius": radius,
        "centers": centers,
        "areas": areas,
        "simplex_points": simplex_points,
        "edge_midpoints": edge_midpoints,
        "edge_normals": edge_normals,
        "edge_lengths": edge_lengths,
    }


def constant_eastward_wind(points, speed=1.0):
    """Return a constant-speed eastward tangent wind at points on the sphere."""
    pts = np.asarray(points, dtype=float)
    one_point = pts.ndim == 1
    pts2 = np.atleast_2d(pts)
    rho = np.hypot(pts2[:, 0], pts2[:, 1])
    wind = np.zeros_like(pts2)
    mask = rho > 1e-14
    wind[mask, 0] = -pts2[mask, 1] / rho[mask]
    wind[mask, 1] = pts2[mask, 0] / rho[mask]
    wind[~mask, 1] = 1.0
    wind *= float(speed)
    return wind[0] if one_point else wind


def build_edge_flux_integrals(geometry, wind_fn, quadrature_order=6):
    xi, wi = np.polynomial.legendre.leggauss(quadrature_order)
    tau_q = 0.5 * (xi + 1.0)
    weight_q = 0.5 * wi

    unit_vertices = geometry["unit_vertices"]
    faces = geometry["faces"]
    radius = geometry["radius"]
    centers = geometry["centers"] / radius
    edge_flux = np.empty((faces.shape[0], 3), dtype=float)

    for k, face in enumerate(faces):
        tri = unit_vertices[face]
        for f, (i, j) in enumerate(_FACE_VERTEX_IDS):
            a = tri[i]
            b = tri[j]
            alpha = geometry["edge_lengths"][k, f] / radius
            normal = geometry["edge_normals"][k, f]

            if alpha < 1e-14:
                edge_flux[k, f] = 0.0
                continue

            line_integral = 0.0
            sin_alpha = np.sin(alpha)
            for tau, weight in zip(tau_q, weight_q):
                point = (
                    np.sin((1.0 - tau) * alpha) * a + np.sin(tau * alpha) * b
                ) / sin_alpha
                point = radius * point / np.linalg.norm(point)
                line_integral += weight * np.dot(wind_fn(point), normal)

            edge_flux[k, f] = radius * alpha * line_integral

            if np.dot(normal, centers[k]) > 1e-12:
                raise RuntimeError(
                    f"edge normal orientation is inconsistent for element {k}, face {f}: "
                    f"dot={np.dot(normal, centers[k]):.3e}"
                )

    return edge_flux


def scalar_advection_rhs_sphere(q, areas, edge_flux, EToE):
    """Return a first-order upwind FV residual for scalar advection on the sphere."""
    q = np.asarray(q, dtype=float)
    rhs = np.zeros_like(q)
    K = q.shape[0]

    for k in range(K):
        flux_sum = np.zeros_like(q[k])
        for f in range(3):
            kn = EToE[k, f]
            flux = float(edge_flux[k, f])
            q_upwind = q[k] if flux >= 0.0 else q[kn]
            # Scalar face flux from this FV stencil is applied to each nodal value so
            # the nodal representation evolves with the same conservative cell fluxes.
            flux_sum += flux * q_upwind
        rhs[k] = -flux_sum / areas[k]

    return rhs


def _stable_dt_sphere(areas, edge_flux, cfl):
    denom = np.sum(np.abs(edge_flux), axis=1)
    safe = np.where(denom > 1e-14, areas / denom, np.inf)
    return float(cfl * np.min(safe))


def simulate_scalar_advection_sphere(
    refinement_level,
    t_final,
    speed=1.0,
    cfl=0.25,
    radius=1.0,
    q_init=None,
    quadrature_order=6,
    simplex_order=1,
):
    """Advect a scalar represented on simplex nodes of each spherical triangle."""
    vertices, faces = build_icosahedral_sphere_mesh(refinement_level=refinement_level, radius=radius)
    geometry = build_sphere_geometry(
        vertices,
        faces,
        quadrature_order=quadrature_order,
        simplex_order=simplex_order,
    )
    edge_flux = build_edge_flux_integrals(
        geometry,
        lambda p: constant_eastward_wind(p, speed=speed),
        quadrature_order=quadrature_order,
    )

    simplex_points = geometry["simplex_points"]
    if q_init is None:
        q_init = lambda pts: np.exp(-((pts[:, 0] - 1.0) ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2) / 0.15)

    K, Np = simplex_points.shape[0], simplex_points.shape[1]
    q_raw = np.asarray(q_init(simplex_points.reshape(K * Np, 3)), dtype=float)
    if q_raw.shape == (K, Np):
        q = q_raw
    elif q_raw.shape == (K * Np,):
        q = q_raw.reshape(K, Np)
    elif q_raw.shape == (K,):
        q = np.repeat(q_raw[:, np.newaxis], Np, axis=1)
    else:
        raise ValueError(
            "q_init must return shape (K*Np,), (K, Np), or (K,) for K elements and Np simplex nodes"
        )

    def rhs(v):
        return scalar_advection_rhs_sphere(v, geometry["areas"], edge_flux, geometry["EToE"])

    t = 0.0
    dt0 = _stable_dt_sphere(geometry["areas"], edge_flux, cfl)

    while t < t_final:
        dt = min(dt0, t_final - t)
        if dt <= 0.0:
            break
        k1 = rhs(q)
        q1 = q + dt * k1
        k2 = rhs(q1)
        q2 = 0.75 * q + 0.25 * (q1 + dt * k2)
        k3 = rhs(q2)
        q = (q + 2.0 * (q2 + dt * k3)) / 3.0
        t += dt

    geometry["edge_flux"] = edge_flux
    return geometry, q, t
