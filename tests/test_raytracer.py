import numpy as np
import pytest

import CoordinateTransformations as ct
import Raytracer as rt
import RayTracerObject as rto
from Geometry.Rotations import rotx, rotz


def test_ray_tracer():
    # Create a simple scene with a plane and a sphere
    plane_transform = ct.AffineTransformation(
        system1="World",
        system2="plane",
        matrix=np.eye(4),
    )
    sphere_transform = ct.AffineTransformation(
        system1="World",
        system2="sphere",
        matrix=np.eye(4),
    )

    plane = rto.Plane(coordinate_system=plane_transform, name="plane", xExtent=5.0, yExtent=5.0)
    sphere = rto.Sphere(coordinate_system=sphere_transform, name="sphere", radius=2.0)

    scene = rt.RayTracerScene(objects=[plane, sphere])

    # Generate rays from a point source
    rng = np.random.default_rng(seed=42)
    rays = rt.Rays(
        origin=np.array([[0.0, 0.0, 10.0]] * 10),  # 10 rays originating from the same point
        direction=rng.normal(size=(10, 3)),  # Random directions
    )

    # Perform ray tracing
    intersection, intersection_hist = rt.perform_ray_tracing(rays, scene)

    print(intersection)  # Print the intersection points for debugging

    # Check that the intersection points have the expected shape
    assert intersection.shape == (10, 3)
    assert intersection_hist.shape == (len(scene.objects) + 1, 10, 3)


def analytic_plane_intersection(rays: rt.Rays) -> np.ndarray:
    """
    Closed-form intersection of each ray with the surface a Plane describes.

    Plane.surface_height is identically zero, so in the object's local frame the
    surface is the z = 0 plane and o_z + t * d_z = 0 fixes the ray parameter.

    Args:
        rays (rt.Rays): Rays in the plane's local frame, none parallel to z = 0.

    Returns:
        np.ndarray: The (n, 3) intersection points.
    """
    t = -rays.origin[:, 2] / rays.direction[:, 2]
    return rays.origin + t[:, np.newaxis] * rays.direction


def analytic_sphere_intersection(rays: rt.Rays, radius: float) -> np.ndarray:
    """
    Closed-form intersection of each ray with the surface a Sphere describes.

    Sphere.surface_height is the sagitta z = R - sqrt(R^2 - x^2 - y^2), so the
    surface is not a sphere centred on the local origin: it is the cap, lying at
    z <= R, of a sphere of radius R centred at local (0, 0, R). Substituting the
    ray into |p - centre|^2 = R^2 gives a quadratic; of its two roots only the
    one on that cap is a point of the modelled surface.

    Args:
        rays (rt.Rays): Rays in the sphere's local frame that cross the cap.
        radius (float): The sphere's radius.

    Returns:
        np.ndarray: The (n, 3) intersection points on the cap.
    """
    centre = np.array([0.0, 0.0, radius])
    origin_to_centre = rays.origin - centre

    a = np.sum(rays.direction * rays.direction, axis=-1)
    b = 2.0 * np.sum(origin_to_centre * rays.direction, axis=-1)
    c = np.sum(origin_to_centre * origin_to_centre, axis=-1) - radius**2

    discriminant = b**2 - 4.0 * a * c
    assert np.all(discriminant > 0.0), "Every test ray must cross the sphere."

    root = np.sqrt(discriminant)
    near = rays.origin + ((-b - root) / (2.0 * a))[:, np.newaxis] * rays.direction
    far = rays.origin + ((-b + root) / (2.0 * a))[:, np.newaxis] * rays.direction

    on_cap = near[:, 2] <= radius
    assert np.all(on_cap != (far[:, 2] <= radius)), (
        "Exactly one root per ray must lie on the cap for the comparison to be unambiguous."
    )
    return np.where(on_cap[:, np.newaxis], near, far)


def test_alternating_projections_matches_analytic_plane():
    """The iterative intersection must reproduce the closed-form line-plane solution."""
    plane = rto.Plane(
        coordinate_system=ct.AffineTransformation("plane", "World"),
        name="plane",
        xExtent=50.0,
        yExtent=50.0,
    )

    rng = np.random.default_rng(seed=7)
    num_rays = 200

    # Origins above the plane with a solid -z component in every direction, so
    # each ray crosses z = 0 once and none of them runs parallel to the surface.
    origins = np.column_stack(
        (
            rng.uniform(-3.0, 3.0, num_rays),
            rng.uniform(-3.0, 3.0, num_rays),
            rng.uniform(4.0, 12.0, num_rays),
        )
    )
    directions = np.column_stack(
        (
            rng.uniform(-0.6, 0.6, num_rays),
            rng.uniform(-0.6, 0.6, num_rays),
            -np.ones(num_rays),
        )
    )
    rays = rt.Rays(origin=origins, direction=directions)

    numeric = rt.alternating_projections_intersections(rays, plane)
    analytic = analytic_plane_intersection(rays)

    assert np.all(np.isfinite(numeric)), "Every ray was aimed inside the plane's extent."
    np.testing.assert_allclose(numeric, analytic, atol=1e-12)


def test_alternating_projections_matches_analytic_sphere():
    """The iterative intersection must reproduce the closed-form line-sphere solution."""
    radius = 5.0
    sphere = rto.Sphere(
        coordinate_system=ct.AffineTransformation("sphere", "World"),
        name="sphere",
        radius=radius,
    )

    rng = np.random.default_rng(seed=11)
    num_rays = 200

    # Aim each ray at a known point on the cap so an intersection is guaranteed.
    # The analytic solver still has to find that point on its own, and the square
    # root on the sampled radius spreads the targets evenly over the disc.
    target_radius = radius * np.sqrt(rng.uniform(0.0, 0.25, num_rays))
    azimuth = rng.uniform(0.0, 2.0 * np.pi, num_rays)
    targets = np.column_stack(
        (
            target_radius * np.cos(azimuth),
            target_radius * np.sin(azimuth),
            radius - np.sqrt(radius**2 - target_radius**2),
        )
    )

    # Starting well below the vertex means each ray enters through the cap and
    # leaves through the far hemisphere, so the root on the cap is unique.
    origins = np.column_stack(
        (
            rng.uniform(-2.0, 2.0, num_rays),
            rng.uniform(-2.0, 2.0, num_rays),
            rng.uniform(-12.0, -8.0, num_rays),
        )
    )
    rays = rt.Rays(origin=origins, direction=targets - origins)

    numeric = rt.alternating_projections_intersections(rays, sphere)
    analytic = analytic_sphere_intersection(rays, radius)

    # The analytic root selection has to agree with the points the rays were aimed at.
    np.testing.assert_allclose(analytic, targets, atol=1e-12)

    assert np.all(np.isfinite(numeric)), "Every ray was aimed at the sphere's cap."
    np.testing.assert_allclose(numeric, analytic, atol=1e-8)


def test_ray_transformation():
    """Rays should transform correctly between coordinate systems."""
    # Create a simple transformation: translate by (1, 2, 3) and rotate 90 degrees about z.
    # The matrix is applied from the right (p_world = p_local @ matrix), so the
    # rotation rows carry the local axes and the translation sits in the last row
    # rather than the last column.
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotz(np.pi / 2.0)
    transformation_matrix[3, :3] = np.array([1.0, 2.0, 3.0])

    affine_transformation = ct.AffineTransformation(
        system1="local", system2="world", matrix=transformation_matrix
    )

    # Create a ray in the local coordinate system
    local_ray = rt.Rays(origin=np.array([[0.0, 0.0, 0.0]]), direction=np.array([[1.0, 0.0, 0.0]]))
    # Transform the ray to the world coordinate system
    world_ray = local_ray.transform(affine_transformation)

    # Expected origin and direction after transformation
    expected_origin = np.array([[1, 2, 3]])
    expected_direction = np.array([[0, 1, 0]])  # Rotated

    np.testing.assert_allclose(world_ray.origin, expected_origin, atol=1e-12)
    np.testing.assert_allclose(world_ray.direction, expected_direction, atol=1e-12)


# --------------------------------------------------------------------------
# Reflection
# --------------------------------------------------------------------------


def test_reflect_matches_hand_computed_geometry():
    """Textbook cases: 45 degrees turns the ray by 90, normal incidence reverses it."""
    normal = np.array([[0.0, 0.0, 1.0]])
    origin = np.array([[0.0, 0.0, 5.0]])
    intersection = np.array([[0.0, 0.0, 0.0]])

    diagonal = rt.Rays(origin=origin.copy(), direction=np.array([[1.0, 0.0, -1.0]]))
    reflected = diagonal.reflect(intersection, normal)
    np.testing.assert_allclose(
        reflected.direction, np.array([[1.0, 0.0, 1.0]]) / np.sqrt(2.0), atol=1e-15
    )

    head_on = rt.Rays(origin=origin.copy(), direction=np.array([[0.0, 0.0, -1.0]]))
    np.testing.assert_allclose(
        head_on.reflect(intersection, normal).direction, np.array([[0.0, 0.0, 1.0]]), atol=1e-15
    )


def test_reflect_obeys_the_law_of_reflection():
    """For random rays and normals: the normal component flips, the rest is untouched."""
    rng = np.random.default_rng(seed=3)
    num_rays = 200

    normals = rng.normal(size=(num_rays, 3))
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    intersections = rng.normal(size=(num_rays, 3))

    rays = rt.Rays(origin=rng.normal(size=(num_rays, 3)), direction=rng.normal(size=(num_rays, 3)))
    incident = rays.direction
    reflected = rays.reflect(intersections, normals).direction

    # Reflection is an isometry, so the outgoing ray is still a unit vector.
    np.testing.assert_allclose(np.linalg.norm(reflected, axis=-1), 1.0, atol=1e-12)

    # The angle of incidence equals the angle of reflection: the component along
    # the normal changes sign while the tangential component is preserved.
    along_normal_in = np.sum(incident * normals, axis=-1, keepdims=True)
    along_normal_out = np.sum(reflected * normals, axis=-1, keepdims=True)
    np.testing.assert_allclose(along_normal_out, -along_normal_in, atol=1e-12)
    np.testing.assert_allclose(
        reflected - along_normal_out * normals, incident - along_normal_in * normals, atol=1e-12
    )

    # The outgoing ray stays in the plane of incidence spanned by d and n.
    np.testing.assert_allclose(
        np.sum(reflected * np.cross(incident, normals), axis=-1), 0.0, atol=1e-12
    )


def test_reflect_is_its_own_inverse():
    """Reflecting twice off the same surface restores the original direction."""
    rng = np.random.default_rng(seed=5)
    num_rays = 50

    normals = rng.normal(size=(num_rays, 3))
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    intersections = rng.normal(size=(num_rays, 3))

    rays = rt.Rays(origin=rng.normal(size=(num_rays, 3)), direction=rng.normal(size=(num_rays, 3)))
    once = rays.reflect(intersections, normals)
    twice = once.reflect(intersections, normals)

    np.testing.assert_allclose(twice.direction, rays.direction, atol=1e-12)


def test_reflect_restarts_the_ray_at_the_intersection():
    """The outgoing ray must leave from the surface, not from the old origin.

    perform_ray_tracing relies on this to propagate: the next surface intersects
    the reflected ray starting where the previous one was hit.
    """
    rays = rt.Rays(origin=np.array([[0.0, 0.0, 9.0]]), direction=np.array([[0.0, 0.0, -1.0]]))
    intersection = np.array([[0.25, -0.5, 0.0]])

    reflected = rays.reflect(intersection, np.array([[0.0, 0.0, 1.0]]))

    np.testing.assert_allclose(reflected.origin, intersection)


def test_reflect_normalises_the_surface_normal():
    """A normal given with arbitrary length must not change the outgoing ray."""
    rays = rt.Rays(origin=np.zeros((1, 3)), direction=np.array([[1.0, 2.0, -3.0]]))
    intersection = np.zeros((1, 3))

    unit = rays.reflect(intersection, np.array([[0.0, 0.0, 1.0]])).direction
    scaled = rays.reflect(intersection, np.array([[0.0, 0.0, 17.5]])).direction

    np.testing.assert_allclose(scaled, unit, atol=1e-15)


# --------------------------------------------------------------------------
# Refraction
# --------------------------------------------------------------------------


def test_refract_obeys_snells_law():
    """n1 * sin(theta_i) == n2 * sin(theta_t), with the ray staying in the plane of incidence."""
    n1, n2 = 1.0, 1.5
    incidence_angles = np.deg2rad(np.array([5.0, 20.0, 35.0, 50.0]))

    normals = np.tile(np.array([0.0, 0.0, 1.0]), (incidence_angles.size, 1))
    directions = np.column_stack(
        (
            np.zeros_like(incidence_angles),
            np.sin(incidence_angles),
            -np.cos(incidence_angles),
        )
    )
    rays = rt.Rays(origin=np.zeros((incidence_angles.size, 3)), direction=directions.copy())

    refracted = rays.refract(np.zeros_like(directions), normals, n1=n1, n2=n2).direction

    np.testing.assert_allclose(np.linalg.norm(refracted, axis=-1), 1.0, atol=1e-12)

    # The transmitted angle is measured against the continuing -z direction.
    transmission_angles = np.arccos(np.clip(-refracted[:, 2], -1.0, 1.0))
    np.testing.assert_allclose(
        n1 * np.sin(incidence_angles), n2 * np.sin(transmission_angles), atol=1e-12
    )

    # Refraction bends towards the normal but never across it.
    assert np.all(refracted[:, 1] > 0.0)
    np.testing.assert_allclose(refracted[:, 0], 0.0, atol=1e-12)
    assert np.all(transmission_angles < incidence_angles)


def test_refract_falls_back_to_reflection_beyond_the_critical_angle():
    """Past the critical angle no ray is transmitted, so the surface acts as a mirror."""
    n1, n2 = 1.5, 1.0
    critical_angle = np.arcsin(n2 / n1)
    incidence_angles = critical_angle + np.deg2rad(np.array([5.0, 15.0, 25.0]))

    normals = np.tile(np.array([0.0, 0.0, 1.0]), (incidence_angles.size, 1))
    directions = np.column_stack(
        (
            np.zeros_like(incidence_angles),
            np.sin(incidence_angles),
            -np.cos(incidence_angles),
        )
    )
    rays = rt.Rays(origin=np.zeros((incidence_angles.size, 3)), direction=directions.copy())
    surface = np.zeros((incidence_angles.size, 3))
    mirrored = rays.reflect(surface, normals).direction

    np.testing.assert_allclose(
        rays.refract(surface, normals, n1=n1, n2=n2).direction, mirrored, atol=1e-12
    )


def test_refract_is_indifferent_to_which_way_the_normal_points():
    """A surface normal facing away from the ray must give the same transmitted ray.

    Plane carries a normalDirection that can flip its normals, so refract has to
    resolve the sign itself rather than trusting the caller.
    """
    directions = np.array([[0.0, np.sin(0.4), -np.cos(0.4)]])
    surface = np.zeros((1, 3))

    facing = rt.Rays(origin=np.zeros((1, 3)), direction=directions.copy())
    away = rt.Rays(origin=np.zeros((1, 3)), direction=directions.copy())

    towards_ray = facing.refract(surface, np.array([[0.0, 0.0, 1.0]]), n1=1.0, n2=1.5)
    along_ray = away.refract(surface, np.array([[0.0, 0.0, -1.0]]), n1=1.0, n2=1.5)

    np.testing.assert_allclose(along_ray.direction, towards_ray.direction, atol=1e-15)


def test_refract_restarts_the_ray_at_the_intersection():
    """Like reflect, the transmitted ray leaves from the surface it just crossed."""
    rays = rt.Rays(origin=np.array([[0.0, 0.0, 9.0]]), direction=np.array([[0.0, 0.3, -1.0]]))
    intersection = np.array([[0.25, -0.5, 0.0]])

    refracted = rays.refract(intersection, np.array([[0.0, 0.0, 1.0]]), n1=1.0, n2=1.5)

    np.testing.assert_allclose(refracted.origin, intersection)


def test_refract_at_normal_incidence_passes_straight_through():
    """A ray along the normal is not bent, whatever the index change."""
    rays = rt.Rays(origin=np.zeros((1, 3)), direction=np.array([[0.0, 0.0, -1.0]]))

    refracted = rays.refract(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), n1=1.0, n2=1.5)

    np.testing.assert_allclose(refracted.direction, np.array([[0.0, 0.0, -1.0]]), atol=1e-15)


def test_refract_handles_bundles_of_any_size():
    """The row-wise dot must work for any ray count, not only for three rays."""
    for num_rays in (1, 2, 3, 7, 50):
        directions = np.tile([0.0, 0.4, -1.0], (num_rays, 1))
        rays = rt.Rays(origin=np.zeros((num_rays, 3)), direction=directions)
        normals = np.tile([0.0, 0.0, 1.0], (num_rays, 1))

        refracted = rays.refract(np.zeros((num_rays, 3)), normals, n1=1.0, n2=1.5)

        assert refracted.direction.shape == (num_rays, 3)
        np.testing.assert_allclose(np.linalg.norm(refracted.direction, axis=-1), 1.0, atol=1e-12)


# --------------------------------------------------------------------------
# Transformation of rays
# --------------------------------------------------------------------------


def test_transform_translation_moves_origins_but_leaves_directions():
    """A direction is a difference of points, so it must not pick up the translation."""
    translation = np.array([1.0, -2.0, 3.0])
    transformation = ct.AffineTransformation("A", "B")
    transformation.set_translation(translation)

    origins = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    rays = rt.Rays(origin=origins.copy(), direction=np.tile([0.0, 0.0, -1.0], (2, 1)))
    moved = rays.transform(transformation)

    np.testing.assert_allclose(moved.origin, origins + translation, atol=1e-15)
    np.testing.assert_allclose(moved.direction, rays.direction, atol=1e-15)


def test_transform_rotation_turns_origins_and_directions_alike():
    """A pure rotation acts on origins and directions the same way."""
    transformation = ct.AffineTransformation("A", "B")
    transformation.set_rotation(rotz(np.pi / 2.0))

    rays = rt.Rays(origin=np.array([[1.0, 0.0, 0.0]]), direction=np.array([[1.0, 0.0, 0.0]]))
    turned = rays.transform(transformation)

    np.testing.assert_allclose(turned.origin, np.array([[0.0, 1.0, 0.0]]), atol=1e-15)
    np.testing.assert_allclose(turned.direction, np.array([[0.0, 1.0, 0.0]]), atol=1e-15)


def test_transform_round_trip_restores_the_rays():
    """Transforming into a frame and back out again is the identity."""
    transformation = ct.AffineTransformation("A", "B")
    transformation.set_rotation(rotx(0.4))
    transformation.set_translation(np.array([1.0, -2.0, 3.0]))

    rng = np.random.default_rng(seed=13)
    rays = rt.Rays(origin=rng.normal(size=(20, 3)), direction=rng.normal(size=(20, 3)))

    there = rays.transform(transformation)
    back = there.transform(transformation.invert_transformation())

    np.testing.assert_allclose(back.origin, rays.origin, atol=1e-12)
    np.testing.assert_allclose(back.direction, rays.direction, atol=1e-12)


def test_transform_preserves_unit_directions_and_leaves_the_source_alone():
    """A rigid transform keeps directions unit length and does not mutate its input."""
    transformation = ct.AffineTransformation("A", "B")
    transformation.set_rotation(rotx(-0.9))
    transformation.set_translation(np.array([4.0, 5.0, 6.0]))

    rng = np.random.default_rng(seed=17)
    rays = rt.Rays(origin=rng.normal(size=(30, 3)), direction=rng.normal(size=(30, 3)))
    original_origin = rays.origin.copy()
    original_direction = rays.direction.copy()

    moved = rays.transform(transformation)

    np.testing.assert_allclose(np.linalg.norm(moved.direction, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(rays.origin, original_origin)
    np.testing.assert_allclose(rays.direction, original_direction)


# --------------------------------------------------------------------------
# Building and combining Rays
# --------------------------------------------------------------------------


def test_rays_normalise_their_directions_on_construction():
    """Rays are stored with unit directions whatever length they are given."""
    rays = rt.Rays(origin=np.zeros((2, 3)), direction=np.array([[0.0, 0.0, -7.0], [3.0, 4.0, 0.0]]))

    np.testing.assert_allclose(np.linalg.norm(rays.direction, axis=-1), 1.0, atol=1e-15)
    np.testing.assert_allclose(rays.direction[1], np.array([0.6, 0.8, 0.0]), atol=1e-15)


def test_combine_rays_concatenates_both_bundles_in_order():
    """Combining appends the second bundle after the first, keeping every ray."""
    first = rt.Rays(origin=np.zeros((2, 3)), direction=np.tile([0.0, 0.0, -1.0], (2, 1)))
    second = rt.Rays(origin=np.ones((3, 3)), direction=np.tile([1.0, 0.0, 0.0], (3, 1)))

    combined = first.combine_rays(second)

    assert len(combined) == len(first) + len(second)
    np.testing.assert_allclose(combined.origin[: len(first)], first.origin)
    np.testing.assert_allclose(combined.origin[len(first) :], second.origin)
    np.testing.assert_allclose(combined.direction[: len(first)], first.direction)
    np.testing.assert_allclose(combined.direction[len(first) :], second.direction)


def test_indexing_rays_keeps_the_bundle_shape():
    """A single ray picked out of a bundle is still an (1, 3) bundle."""
    rays = rt.Rays(origin=np.zeros((4, 3)), direction=np.tile([0.0, 0.0, -1.0], (4, 1)))

    assert len(rays) == 4
    assert rays[1].origin.shape == (1, 3)
    assert rays[1:3].origin.shape == (2, 3)


# --------------------------------------------------------------------------
# Scene coordinate graph
# --------------------------------------------------------------------------


def placed_object(name, tilt, position, radius=None):
    """Build a Plane (or Sphere, if a radius is given) placed relative to World."""
    transformation = ct.AffineTransformation(name, "World")
    transformation.set_rotation(rotx(tilt))
    transformation.set_translation(position)

    if radius is None:
        return rto.Plane(coordinate_system=transformation, name=name)
    return rto.Sphere(coordinate_system=transformation, name=name, radius=radius)


def test_scene_registers_every_object_coordinate_system():
    """Constructing a scene wires each object's frame into the transform graph."""
    first = placed_object("First", 0.3, np.array([0.0, 1.0, 2.0]))
    second = placed_object("Second", -0.2, np.array([1.0, 0.0, -3.0]), radius=9.0)
    scene = rt.RayTracerScene(objects=[first, second])

    assert sorted(scene.coordinate_systems.get_all_systems()) == ["First", "Second", "World"]
    assert scene.object_names == ["First", "Second"]


def test_scene_graph_composes_transformations_between_objects():
    """A hop between two object frames must agree with going through World by hand."""
    first = placed_object("First", 0.3, np.array([0.0, 1.0, 2.0]))
    second = placed_object("Second", -0.2, np.array([1.0, 0.0, -3.0]), radius=9.0)
    scene = rt.RayTracerScene(objects=[first, second])
    graph = scene.coordinate_systems

    rng = np.random.default_rng(seed=19)
    points = rng.normal(size=(25, 3))

    direct = graph.transform_points(points, From="First", To="Second")
    via_world = graph.transform_points(
        graph.transform_points(points, From="First", To="World"), From="World", To="Second"
    )
    np.testing.assert_allclose(direct, via_world, atol=1e-12)

    # Every edge is stored together with its inverse, so a round trip is exact.
    round_trip = graph.transform_points(
        graph.transform_points(points, From="First", To="World"), From="World", To="First"
    )
    np.testing.assert_allclose(round_trip, points, atol=1e-12)


def test_scene_graph_matches_the_objects_own_transformation():
    """The World -> object edge is exactly the inverse of the object's own frame."""
    obj = placed_object("Solo", 0.6, np.array([2.0, -1.0, 0.5]))
    scene = rt.RayTracerScene(objects=[obj])

    rng = np.random.default_rng(seed=23)
    local_points = rng.normal(size=(10, 3))

    expected = obj.get_coordinate_system().transform(local_points)
    np.testing.assert_allclose(
        scene.coordinate_systems.transform_points(local_points, From="Solo", To="World"),
        expected,
        atol=1e-12,
    )


def test_add_object_extends_the_graph():
    """add_object registers the new frame so later objects are reachable too."""
    scene = rt.RayTracerScene(objects=[placed_object("First", 0.0, np.zeros(3))])
    scene.add_object(placed_object("Late", 0.1, np.array([0.0, 0.0, 4.0])))

    assert scene.object_names == ["First", "Late"]
    assert "Late" in scene.coordinate_systems.get_all_systems()
    assert scene.get_object_by_name("Late").get_name() == "Late"

    point = np.array([[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        scene.coordinate_systems.transform_points(point, From="Late", To="First"),
        np.array([[0.0, 0.0, 4.0]]),
        atol=1e-12,
    )


def test_scene_rejects_duplicate_coordinate_systems():
    """Two objects sharing a frame name would make the graph ambiguous."""
    with pytest.raises(AssertionError):
        rt.RayTracerScene(
            objects=[
                placed_object("Same", 0.0, np.zeros(3)),
                placed_object("Same", 0.5, np.ones(3)),
            ]
        )


def test_get_object_by_name_reports_unknown_names():
    """Looking up a frame that is not in the scene is an error, not a silent None."""
    scene = rt.RayTracerScene(objects=[placed_object("First", 0.0, np.zeros(3))])

    with pytest.raises(ValueError, match="not found in the scene"):
        scene.get_object_by_name("Missing")
