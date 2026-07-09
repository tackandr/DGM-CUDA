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

    centers = geometry["centers"]
    lon = np.arctan2(centers[:, 1], centers[:, 0])
    lat = np.arcsin(centers[:, 2] / np.linalg.norm(centers, axis=1))
    angular_displacement = speed * t / np.maximum(np.cos(lat), 1e-12)
    lon_back = ((lon - angular_displacement + np.pi) % (2.0 * np.pi)) - np.pi
    q_exact = gaussian_from_lonlat(lon_back, lat)
    q0 = q_init(centers)

    initial_mass = np.sum(q0 * geometry["areas"])
    final_mass = np.sum(q * geometry["areas"])
    l2_err = np.sqrt(np.mean((q - q_exact) ** 2))

    lon0 = np.arctan2(np.sum(q0 * geometry["areas"] * centers[:, 1]), np.sum(q0 * geometry["areas"] * centers[:, 0]))
    lon1 = np.arctan2(np.sum(q * geometry["areas"] * centers[:, 1]), np.sum(q * geometry["areas"] * centers[:, 0]))

    assert abs(final_mass - initial_mass) < 1e-12
    assert np.min(q) >= -1e-12
    assert np.max(q) <= np.max(q0) + 1e-12
    assert l2_err < 0.08
    assert lon1 > lon0 + 0.05
