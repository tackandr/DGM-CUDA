import numpy as np

from icosahedral_sphere import (
    build_edge_flux_integrals,
    build_icosahedral_sphere_mesh,
    build_sphere_geometry,
    constant_eastward_wind,
    scalar_advection_rhs_sphere,
    simulate_scalar_advection_sphere,
)


def test_refined_icosahedral_mesh_has_expected_counts_and_radius():
    vertices, faces = build_icosahedral_sphere_mesh(refinement_level=1, radius=1.0)

    assert vertices.shape == (42, 3)
    assert faces.shape == (80, 3)
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0, atol=1e-12)


def test_constant_state_rhs_is_nearly_zero_for_eastward_flow():
    vertices, faces = build_icosahedral_sphere_mesh(refinement_level=1, radius=1.0)
    geometry = build_sphere_geometry(vertices, faces, quadrature_order=6)
    edge_flux = build_edge_flux_integrals(geometry, lambda p: constant_eastward_wind(p, speed=1.0), quadrature_order=6)

    q = np.ones(faces.shape[0], dtype=float)
    rhs = scalar_advection_rhs_sphere(q, geometry["areas"], edge_flux, geometry["EToE"])

    np.testing.assert_allclose(rhs, 0.0, atol=5e-8)


def test_geometry_supports_higher_order_simplex_nodes():
    vertices, faces = build_icosahedral_sphere_mesh(refinement_level=1, radius=1.0)
    geometry = build_sphere_geometry(vertices, faces, quadrature_order=6, simplex_order=3)

    assert geometry["centers"].shape == (faces.shape[0], 3)
    np.testing.assert_allclose(np.linalg.norm(geometry["centers"], axis=1), 1.0, atol=1e-12)
    assert geometry["simplex_points"].shape == (faces.shape[0], 10, 3)
    np.testing.assert_allclose(np.linalg.norm(geometry["simplex_points"], axis=2), 1.0, atol=1e-12)


def test_scalar_gaussian_advects_eastward_on_one_refinement_level():
    width = 0.35
    speed = 1.0
    t_final = 0.15

    def gaussian_from_lonlat(lon, lat):
        return np.exp(-((lon ** 2 + lat ** 2) / width**2))

    def q_init(points):
        lon = np.arctan2(points[:, 1], points[:, 0])
        lat = np.arcsin(points[:, 2] / np.linalg.norm(points, axis=1))
        return gaussian_from_lonlat(lon, lat)

    geometry, q, t = simulate_scalar_advection_sphere(
        refinement_level=1,
        t_final=t_final,
        speed=speed,
        cfl=0.25,
        radius=1.0,
        q_init=q_init,
        quadrature_order=6,
    )

    simplex_points = geometry["simplex_points"]
    lon = np.arctan2(simplex_points[:, :, 1], simplex_points[:, :, 0])
    lat = np.arcsin(simplex_points[:, :, 2] / np.linalg.norm(simplex_points, axis=2))
    angular_displacement = speed * t / np.maximum(np.cos(lat), 1e-12)
    lon_back = ((lon - angular_displacement + np.pi) % (2.0 * np.pi)) - np.pi
    q_exact = gaussian_from_lonlat(lon_back, lat)
    q0 = q_init(simplex_points.reshape(-1, 3)).reshape(q.shape)
    q_cell = np.mean(q, axis=1)
    q0_cell = np.mean(q0, axis=1)
    q_exact_cell = np.mean(q_exact, axis=1)
    centers = geometry["centers"]

    initial_mass = np.sum(q0_cell * geometry["areas"])
    final_mass = np.sum(q_cell * geometry["areas"])
    l2_err = np.sqrt(np.mean((q_cell - q_exact_cell) ** 2))

    lon0 = np.arctan2(
        np.sum(q0_cell * geometry["areas"] * centers[:, 1]),
        np.sum(q0_cell * geometry["areas"] * centers[:, 0]),
    )
    lon1 = np.arctan2(
        np.sum(q_cell * geometry["areas"] * centers[:, 1]),
        np.sum(q_cell * geometry["areas"] * centers[:, 0]),
    )

    assert abs(final_mass - initial_mass) < 1e-12
    assert np.min(q) >= -1e-12
    assert np.max(q) <= np.max(q0) + 1e-12
    assert l2_err < 0.08
    assert lon1 > lon0 + 0.05


def test_simulation_accepts_higher_order_simplex_nodes():
    geometry, q, t = simulate_scalar_advection_sphere(
        refinement_level=1,
        t_final=0.02,
        speed=1.0,
        cfl=0.25,
        radius=1.0,
        quadrature_order=6,
        simplex_order=3,
    )

    assert t > 0.0
    assert q.shape == (geometry["faces"].shape[0], geometry["simplex_points"].shape[1])
    np.testing.assert_allclose(np.linalg.norm(geometry["centers"], axis=1), 1.0, atol=1e-12)


def test_simulation_accepts_cell_average_q_init_and_broadcasts_to_simplex_nodes():
    simplex_nodes_per_elem = 3

    def q_init(points):
        return np.ones(points.shape[0] // simplex_nodes_per_elem)

    geometry, q, _ = simulate_scalar_advection_sphere(
        refinement_level=0,
        t_final=0.01,
        simplex_order=1,
        q_init=q_init,
    )

    assert q.shape[0] == geometry["faces"].shape[0]
    np.testing.assert_allclose(q[:, 0], q[:, 1])
    np.testing.assert_allclose(q[:, 1], q[:, 2])
