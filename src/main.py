"""A point source folded off a flat mirror and focused by a spherical mirror.

The whole system lies in the world y-z plane, so every element is oriented by a
single rotation about x. The chief ray runs:

    source (0, 0, +10)
        | travelling -z
        v
    fold mirror (0, 0, 0), tilted 45 deg  --- travelling -y --->  spherical
                                                                  mirror
                                                                  (0, -12, 0)
                                                                      |
    detector  <---- travelling +y, tilted up by 2 * off-axis angle ----+

Conventions worth remembering while reading the numbers below:
  * An AffineTransformation("Object", "World") maps local points to world points
    as p_world = p_local @ R + t, so set_translation places the object's local
    origin (its vertex) in the world.
  * rotx(theta) sends the local +z axis to world (0, -sin(theta), cos(theta)),
    and a Plane's normal is its local +z, so the tilt fully fixes the normal.
  * A Sphere's local surface is a bowl whose centre of curvature sits at local
    (0, 0, radius): it is concave towards local +z, which must therefore face
    the incoming beam.
"""

import numpy as np

from CoordinateTransformations import AffineTransformation
from Geometry.Rotations import rotx
from Raytracer import Rays, RayTracerScene, perform_ray_tracing, plot_traced_rays
from RayTracerObject import Plane, Sphere

NUM_RAYS = 200
RANDOM_SEED = 42

# The source is isotropic, but only the cone that the fold mirror actually
# subtends is worth tracing, so the directions are sampled inside that cone.
SOURCE_POSITION = np.array([0.0, 0.0, 10.0])
SOURCE_AXIS = np.array([0.0, 0.0, -1.0])
SOURCE_HALF_ANGLE = np.deg2rad(4.6)

# A 45 deg fold turns the downward beam into a beam travelling along -y.
FOLD_MIRROR_POSITION = np.array([0.0, 0.0, 0.0])
FOLD_MIRROR_TILT = np.deg2rad(45.0)

# The spherical mirror faces +y (local +z towards the incoming beam) and is
# tilted a few degrees off axis so the focus clears the fold mirror on the way
# back instead of landing on top of it.
SPHERE_POSITION = np.array([0.0, -12.0, 0.0])
SPHERE_RADIUS = 24.0
SPHERE_OFF_AXIS_ANGLE = np.deg2rad(5.0)

# Mirror apertures, sized to catch the whole source cone with a little margin.
FOLD_MIRROR_EXTENTS = (1.2, 1.8)  # Stretched in local y by the 45 deg tilt
SPHERE_EXTENTS = (2.2, 2.2)
DETECTOR_EXTENTS = (1.0, 1.0)

# Concave mirror: f = R / 2. The object distance is the unfolded path length
# from the source to the spherical mirror, so the thin-mirror equation gives the
# image distance at which the detector has to sit.
FOCAL_LENGTH = SPHERE_RADIUS / 2.0
OBJECT_DISTANCE = np.linalg.norm(SOURCE_POSITION - FOLD_MIRROR_POSITION) + np.linalg.norm(
    FOLD_MIRROR_POSITION - SPHERE_POSITION
)
IMAGE_DISTANCE = 1.0 / (1.0 / FOCAL_LENGTH - 1.0 / OBJECT_DISTANCE)

# Reflecting a -y beam off a normal tilted by the off-axis angle turns it back
# along +y, rotated up by twice that angle.
FOCUSED_BEAM_DIRECTION = np.array(
    [0.0, np.cos(2 * SPHERE_OFF_AXIS_ANGLE), np.sin(2 * SPHERE_OFF_AXIS_ANGLE)]
)
DETECTOR_POSITION = SPHERE_POSITION + IMAGE_DISTANCE * FOCUSED_BEAM_DIRECTION


def point_source_rays(
    position: np.ndarray,
    axis: np.ndarray,
    half_angle: float,
    num_rays: int,
    rng: np.random.Generator,
) -> Rays:
    """
    Emit rays from a single point, uniformly distributed over a cone of directions.

    Uniform means uniform in solid angle (cos(theta) is sampled uniformly rather
    than theta), so the result is a slice of an isotropic point source and not a
    beam that bunches up along its axis.

    Args:
        position (np.ndarray): The (3,) world position all rays start from.
        axis (np.ndarray): The (3,) direction the cone is centred on.
        half_angle (float): Half opening angle of the cone, in radians.
        num_rays (int): Number of rays to emit.
        rng (np.random.Generator): Source of randomness.

    Returns:
        Rays: The emitted rays, all sharing the same origin.
    """
    axis = axis / np.linalg.norm(axis)

    cos_theta = rng.uniform(np.cos(half_angle), 1.0, size=num_rays)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=num_rays)

    # Complete the axis into an orthonormal basis so the cone can be built around
    # an arbitrary direction. The reference vector only has to be non-parallel.
    reference = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(reference, axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    directions = (
        (sin_theta * np.cos(phi))[:, np.newaxis] * u
        + (sin_theta * np.sin(phi))[:, np.newaxis] * v
        + cos_theta[:, np.newaxis] * axis
    )

    return Rays(origin=np.tile(position, (num_rays, 1)), direction=directions)


def placed_transformation(name: str, tilt: float, position: np.ndarray) -> AffineTransformation:
    """
    Build the local-to-world transformation for an element tilted about x.

    Args:
        name (str): Name of the element's coordinate system.
        tilt (float): Rotation about the world x-axis, in radians.
        position (np.ndarray): World position of the element's local origin.

    Returns:
        AffineTransformation: The transformation from the element to the world.
    """
    transformation = AffineTransformation(name, "World")
    transformation.set_rotation(rotx(tilt))
    transformation.set_translation(position)
    return transformation


def build_scene() -> RayTracerScene:
    """
    Assemble the fold mirror, the spherical mirror and the detector.

    Returns:
        RayTracerScene: The scene, with the objects in the order the rays hit them.
    """
    fold_mirror = Plane(
        coordinate_system=placed_transformation(
            "FoldMirror", FOLD_MIRROR_TILT, FOLD_MIRROR_POSITION
        ),
        name="FoldMirror",
        xExtent=FOLD_MIRROR_EXTENTS[0],
        yExtent=FOLD_MIRROR_EXTENTS[1],
    )

    # Facing +y means the local +z axis maps to world +y, which is a -90 deg
    # tilt, plus the small off-axis angle.
    spherical_mirror = Sphere(
        coordinate_system=placed_transformation(
            "SphericalMirror", SPHERE_OFF_AXIS_ANGLE - np.pi / 2, SPHERE_POSITION
        ),
        name="SphericalMirror",
        radius=SPHERE_RADIUS,
        xExtent=SPHERE_EXTENTS[0],
        yExtent=SPHERE_EXTENTS[1],
    )

    # The detector sits square to the focused beam, so its normal is tilted back
    # by the same 2 * off-axis angle the beam picked up.
    detector = Plane(
        coordinate_system=placed_transformation(
            "Detector", np.pi / 2 + 2 * SPHERE_OFF_AXIS_ANGLE, DETECTOR_POSITION
        ),
        name="Detector",
        xExtent=DETECTOR_EXTENTS[0],
        yExtent=DETECTOR_EXTENTS[1],
    )

    return RayTracerScene(objects=[fold_mirror, spherical_mirror, detector])


def report_spot(intersection_history: np.ndarray, scene: RayTracerScene) -> None:
    """
    Print how many rays survived each surface and how tightly they focus.

    Rays that miss an aperture come back as NaN and stay NaN for the rest of the
    trace, so counting the finite rows per surface shows where light is lost.

    Args:
        intersection_history (np.ndarray): The (1 + n_objects, n_rays, 3) trace.
        scene (RayTracerScene): The scene that was traced.
    """
    for index, object in enumerate(scene.objects, start=1):
        hits = np.isfinite(intersection_history[index]).all(axis=-1).sum()
        print(f"{object.get_name():>16}: {hits:5d} / {intersection_history.shape[1]} rays")

    # Measure the spot in the detector's own frame, where it is a flat x-y patch.
    detector_hits = scene.coordinate_systems.transform_points(
        intersection_history[-1], From="World", To="Detector"
    )
    detector_hits = detector_hits[np.isfinite(detector_hits).all(axis=-1)]

    rms_radius = np.sqrt(np.mean(detector_hits[:, 0] ** 2 + detector_hits[:, 1] ** 2))
    print(f"\nParaxial image distance: {IMAGE_DISTANCE:.3f} (f = {FOCAL_LENGTH:.1f})")
    print(f"Detector position:       {np.round(DETECTOR_POSITION, 3)}")
    print(f"RMS spot radius:         {rms_radius:.4f}")


def main() -> tuple[np.ndarray, RayTracerScene]:
    """Trace a point source off a fold mirror and a spherical mirror onto a detector."""
    scene = build_scene()

    rays = point_source_rays(
        SOURCE_POSITION,
        SOURCE_AXIS,
        SOURCE_HALF_ANGLE,
        NUM_RAYS,
        np.random.default_rng(seed=RANDOM_SEED),
    )

    _, intersection_history = perform_ray_tracing(rays, scene)

    report_spot(intersection_history, scene)
    plot_traced_rays(intersection_history, scene, reference_system="World")

    return intersection_history, scene


if __name__ == "__main__":
    intersection_history, scene = main()
