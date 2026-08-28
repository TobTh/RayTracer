from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import CoordinateTransformations as ct
from RayTracerObject import RayTracerObject


def perform_ray_tracing(
    rays: "Rays",
    scene: "RayTracerScene",
    RaysInSource: bool = False,
    Refract: bool = False,
    SourceSystem: str = "World",
    TargetSystem: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform ray tracing for a given set of rays and a scene.

    Args:
        rays (Rays): The initial rays to trace.
        scene (RayTracerScene): The scene containing objects to interact with.
        RaysInSource (bool): Whether the rays are in the source coordinate system.

    Returns:
        Rays: The final rays after tracing through the scene.
    """
    current_rays = rays
    intersection_history = np.array(
        [rays.origin]
    )  # One (n, 3) array of world-frame intersections per object

    if not TargetSystem:
        TargetSystem = scene.objects[-1].name  # Default to the last object's coordinate system

    for object_index, object in enumerate(scene.objects):
        if SourceSystem != "World" and object.name != SourceSystem:
            continue  # Skip objects that are not in the source system

        if object_index == 0:
            if RaysInSource:
                transformation = ct.AffineTransformation(
                    system1=object.name,
                    system2=object.name,
                    matrix=np.eye(4),
                )  # Identity transformation if rays are already in the source system
            else:
                transformation = scene.coordinate_systems.get_transformation(
                    From="World", To=object.name
                )
        else:
            transformation = scene.coordinate_systems.get_transformation(
                From=scene.objects[object_index - 1].name, To=object.name
            )

        transformed_rays = current_rays.transform(transformation)

        intersections = alternating_projections_intersections(transformed_rays, object)

        intersections_world = scene.coordinate_systems.transform_points(
            intersections, From=object.name, To="World"
        )

        intersection_history = np.concatenate(
            (intersection_history, intersections_world[None, :, :]), axis=0
        )  # Store the intersections for analysis

        # Calculate reflected and refracted rays based on the intersection normals

        normals = object.surface_normal(intersections)

        reflected_rays = transformed_rays.reflect(intersections, normals)

        # Combine reflected and refracted rays for the next iteration
        if Refract:
            refracted_rays = transformed_rays.refract(intersections, normals, n1=1.0, n2=1.5)
            current_rays = reflected_rays.combine_rays(refracted_rays)
        else:
            current_rays = reflected_rays

    # Stack into an (n, 3, n_objects) array: one 2D slice of intersections per object
    return intersections, intersection_history


def plot_traced_rays(
    intersection_history: np.ndarray, scene: "RayTracerScene", reference_system: str = "World"
) -> None:
    """
    Plot the traced rays and their intersection points in the specified reference system.

    Args:
        intersection_history (np.ndarray): The history of intersection points for each object.
        scene (RayTracerScene): The scene containing objects to interact with.
        reference_system (str): The coordinate system to plot in. Defaults to "World".
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the scene objects
    ax = scene.plot_scene(
        reference_system=reference_system, ax=ax, show_plot=False, show_normals=True
    )

    # Plot the intersection points for each object
    for object_index, object in enumerate(scene.objects):
        ax.scatter(
            intersection_history[object_index + 1, :, 0],
            intersection_history[object_index + 1, :, 1],
            intersection_history[object_index + 1, :, 2],
            label=f"Intersections with {object.name}",
            s=20,
        )

        # Draw the leg that brought the rays here, from wherever they came from:
        # the ray origins in slice 0 for the first object, the previous surface
        # otherwise.
        for ray_index in range(intersection_history.shape[1]):
            ax.plot(
                [
                    intersection_history[object_index, ray_index, 0],
                    intersection_history[object_index + 1, ray_index, 0],
                ],
                [
                    intersection_history[object_index, ray_index, 1],
                    intersection_history[object_index + 1, ray_index, 1],
                ],
                [
                    intersection_history[object_index, ray_index, 2],
                    intersection_history[object_index + 1, ray_index, 2],
                ],
                color="black",
                alpha=0.5,
                linewidth=1,
            )

    intersections_world = intersection_history[0, ...]
    ax.scatter(
        intersections_world[:, 0],
        intersections_world[:, 1],
        intersections_world[:, 2],
        label="Initial points",
        s=20,
    )

    ax.set_title(f"Traced Rays and Intersections in '{reference_system}' coordinates")
    ax.legend()
    plt.show()


def alternating_projections_intersections(rays: "Rays", object: "RayTracerObject") -> np.ndarray:
    """
    Find the closest intersection of rays with the scene objects using alternating projections.

    Args:
        rays (Rays): The rays to check for intersections.
        object (RayTracerObject): The object to intersect with.

    Returns:
        np.ndarray: The intersection points of the rays with the object.
    """

    intersections = intersect_rays_with_plane(
        rays, np.array([[0, 0, 1]]), np.array([[0, 0, 0]])
    )  # Example plane at z=0

    max_iterations = 100
    tolerance = 1e-8

    for _ in range(max_iterations):
        # Project the current estimate onto the object's surface: since the
        # surface is defined as z = surface_height(x, y), the surface point
        # corresponding to the current (x, y) estimate is simply
        # (x, y, surface_height(x, y)).
        surface_points = np.stack(
            (
                intersections[..., 0],
                intersections[..., 1],
                object.surface_height(intersections),
            ),
            axis=-1,
        )

        # Project the surface points back onto the corresponding ray
        # by finding the closest point on each ray to the surface point
        origin_to_surface = surface_points - rays.origin
        t = np.sum(origin_to_surface * rays.direction, axis=-1) / np.sum(
            rays.direction * rays.direction, axis=-1
        )

        ray_points = rays.origin + t[:, np.newaxis] * rays.direction

        # Check for convergence
        if np.max(np.linalg.norm(ray_points - intersections, axis=-1)) < tolerance:
            intersections = ray_points
            break

        intersections = ray_points

    return intersections


def intersect_rays_with_plane(
    rays: "Rays", plane_normal: np.ndarray, plane_point: np.ndarray
) -> np.ndarray:
    """
    Calculate the intersection points of rays with a plane.

    Args:
        rays (Rays): The rays to intersect with the plane.
        plane_normal (np.ndarray): The normal vector of the plane.
        plane_point (np.ndarray): A point on the plane.

    Returns:
        np.ndarray: The intersection points of the rays with the plane.
    """

    assert plane_normal.shape == (..., 3) or plane_normal.shape == (1, 3), (
        "Plane normal must be a 3D vector."
    )

    # Normalize the plane normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Calculate the dot product of the ray directions and the plane normal
    dot_product = np.dot(rays.direction, plane_normal.T)

    # Avoid division by zero for rays parallel to the plane
    parallel_mask = np.isclose(dot_product, 0)
    if np.any(parallel_mask):
        raise ValueError("Some rays are parallel to the plane and do not intersect.")

    # Calculate the distance from the ray origins to the intersection points
    d = np.dot(plane_point - rays.origin, plane_normal.T) / dot_product

    # Calculate the intersection points
    intersection_points: np.ndarray = rays.origin + d * rays.direction

    return intersection_points


@dataclass(slots=True)
class Rays:
    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        assert self.origin.shape == self.direction.shape
        # assert self.origin.ndim == 3 and self.origin.shape[1] == 3
        assert self.origin.dtype == np.float64
        assert self.direction.dtype == np.float64

        self.direction /= np.linalg.norm(
            self.direction, axis=-1, keepdims=True
        )  # Normalize directions

    def __len__(self) -> Any:
        return self.origin.shape[0]

    def __getitem__(self, idx: int) -> "Rays":
        o, d = self.origin[idx], self.direction[idx]
        if o.ndim == 1:  # int index → keep the (n, 3) invariant
            o, d = o[None, :], d[None, :]
        return Rays(o, d)

    def plot_rays(self, ax: Axes3D | None = None) -> None:
        """
        Plot the rays in 3D space.

        Args:
            ax (Axes3D | None): The matplotlib 3D axis to plot on. If None, a new
                figure and axis are created. Defaults to None.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            # add_subplot is typed as returning a plain Axes whatever the projection,
            # so the 3D projection has to be asserted here.
            ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))

        print(f"Plotting {len(self)} rays.")

        # Plot each ray as a line segment
        for i in range(len(self)):
            origin = self.origin[i]
            direction = self.direction[i]
            end_point = origin + direction  # Extend the ray for visualization

            ax.plot(
                [origin[0], end_point[0]],
                [origin[1], end_point[1]],
                [origin[2], end_point[2]],
                color="blue",
                label="Rays" if i == 0 else "",  # Only label the first ray for the legend
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    def transform(self, transformation: ct.AffineTransformation) -> "Rays":

        transformed_origin = transformation.transform(self.origin)
        transformed_direction = (
            transformation.transform(self.origin + self.direction) - transformed_origin
        )
        return Rays(transformed_origin, transformed_direction)

    def combine_rays(rays1: "Rays", rays2: "Rays") -> "Rays":
        """
        Combine two Rays objects into one.

        Args:
            rays1 (Rays): The first Rays object.
            rays2 (Rays): The second Rays object.

        Returns:
            Rays: A new Rays object containing the combined origins and directions.
        """
        combined_origin = np.concatenate((rays1.origin, rays2.origin), axis=0)
        combined_direction = np.concatenate((rays1.direction, rays2.direction), axis=0)
        return Rays(combined_origin, combined_direction)

    def reflect(self, intersection: np.ndarray, normal: np.ndarray) -> "Rays":
        """
        Reflect the rays off a surface with the given normal vectors.

        Args:
            normal (np.ndarray): An array of shape (n, 3) representing the normal vectors
                at the points of reflection. Each normal vector should be normalized.

        Returns:
            Rays: A new Rays object representing the reflected rays.
        """
        # Ensure that the normal vectors are normalized
        normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

        # Calculate the reflected direction using the reflection formula
        reflected_direction = (
            self.direction - 2 * np.sum(self.direction * normal, axis=-1, keepdims=True) * normal
        )

        return Rays(intersection, reflected_direction)

    def refract(self, intersection: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> "Rays":
        """
        Refract the rays through a surface with the given normal vectors.

        Args:
            intersection (np.ndarray): An (n, 3) array of points where the rays meet
                the surface. The refracted rays continue from there, exactly as the
                reflected ones do.
            normal (np.ndarray): An array of shape (n, 3) representing the normal vectors
                at the points of refraction. They need neither be normalized nor point
                towards the incoming rays; both are sorted out here.
            n1 (float): The refractive index of the medium from which the rays are coming.
            n2 (float): The refractive index of the medium into which the rays are entering.

        Returns:
            Rays: A new Rays object representing the refracted rays, starting at the
                surface. Rays that arrive beyond the critical angle carry no transmitted
                part, so they come back reflected instead.
        """

        assert self.origin.shape == normal.shape or normal.shape == (1, 3), (
            "Normal vectors must have the same shape as ray origins or be a single normal vector."
        )

        # Ensure that the normal vectors are normalized
        normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

        # Snell's law is written for a normal that opposes the incident ray, so the
        # ones pointing along it are flipped. Reflection is indifferent to that sign,
        # but the transmitted branch is not.
        cos_theta_i = -np.sum(self.direction * normal, axis=-1, keepdims=True)
        normal = np.where(cos_theta_i < 0.0, -normal, normal)
        cos_theta_i = np.abs(cos_theta_i)

        # Calculate the ratio of refractive indices
        eta = n1 / n2

        # Calculate the sine squared of the angle of refraction using Snell's law
        sin_theta_t_squared = eta**2 * (1 - cos_theta_i**2)

        # Check for total internal reflection
        total_internal_reflection = sin_theta_t_squared > 1.0

        # Calculate the cosine of the angle of refraction. Clipping keeps the square
        # root away from the negative values that total internal reflection produces;
        # those entries are discarded below anyway.
        cos_theta_t = np.sqrt(np.clip(1 - sin_theta_t_squared, 0.0, None))

        # Calculate the refracted direction using Snell's law
        refracted_direction = eta * self.direction + (eta * cos_theta_i - cos_theta_t) * normal

        # Beyond the critical angle the surface simply mirrors the ray. With the
        # normal now opposing it, that is the reflection formula written out.
        reflected_direction = self.direction + 2 * cos_theta_i * normal

        return Rays(
            intersection,
            np.where(total_internal_reflection, reflected_direction, refracted_direction),
        )


class RayTracerScene:
    def __init__(self, objects: list[RayTracerObject]):
        self.objects = objects
        self.coordinate_systems = ct.CoordinateTransformGraph()
        self.object_names = [obj.get_name() for obj in objects]

        for obj in objects:
            self.coordinate_systems.add_transformation(obj.get_coordinate_system())

    def add_object(self, obj: RayTracerObject) -> None:
        self.objects.append(obj)
        self.object_names.append(obj.get_name())
        self.coordinate_systems.add_transformation(obj.get_coordinate_system())

    def get_object_by_name(self, name: str) -> RayTracerObject:
        for obj in self.objects:
            if obj.get_name() == name:
                return obj
        raise ValueError(f"Object with name '{name}' not found in the scene.")

    def update_object(self, name: str, updated_object: RayTracerObject) -> None:
        self.get_object_by_name(name)
        self.coordinate_systems.add_transformation(updated_object.get_coordinate_system())

    def _sample_surface(self, obj: RayTracerObject, resolution: int) -> np.ndarray:
        """
        Sample an object's surface on a grid in its own local coordinate system.

        The grid spans the object's x/y extent; the z coordinate comes from the
        object's surface_height. Points off the surface come back as NaN, which
        matplotlib renders as a hole rather than a spurious patch.

        Returns:
            np.ndarray: (resolution, resolution, 3) array of local-frame points.
        """
        x = np.linspace(-obj.xExtent, obj.xExtent, resolution)
        y = np.linspace(-obj.yExtent, obj.yExtent, resolution)
        grid_x, grid_y = np.meshgrid(x, y)

        flat_grid = np.stack((grid_x, grid_y, np.zeros_like(grid_x)), axis=-1)
        grid_z = obj.surface_height(flat_grid)

        return np.stack((grid_x, grid_y, grid_z), axis=-1)

    def _to_reference_frame(self, points: np.ndarray, From: str, To: str) -> np.ndarray:
        """
        Transform a grid of points into the reference frame, preserving its shape.

        transform_points only accepts a single point or an (n, 3) array, so the
        grid is flattened for the transform and restored afterwards.
        """
        if From == To:
            return points

        flat_points = points.reshape(-1, 3)
        transformed = self.coordinate_systems.transform_points(flat_points, From=From, To=To)
        return transformed.reshape(points.shape)

    def plot_scene(
        self,
        reference_system: str = "",
        resolution: int = 40,
        show_normals: bool = False,
        normal_stride: int = 8,
        show_plot: bool = True,
        ax: Axes3D | None = None,
    ) -> Axes3D:
        """
        Plot every object of the scene as a surface in one common coordinate system.

        Args:
            reference_system (str): The frame to plot in. Defaults to the parent
                system of the first object (e.g. "World").
            resolution (int): Number of grid samples per axis, per object.
            show_normals (bool): Draw surface normals as arrows.
            normal_stride (int): Plot every n-th normal, to keep the plot readable.
            ax: An existing 3D axis to draw into. A new figure is created if omitted.

        Returns:
            The axis that was drawn into.
        """
        if not self.objects:
            raise ValueError("Cannot plot an empty scene.")

        if not reference_system:
            reference_system = self.objects[
                0
            ].name  # Default to the first object's coordinate system

        if reference_system not in self.coordinate_systems.get_all_systems():
            raise ValueError(
                f"Reference system '{reference_system}' is not part of the scene. "
                f"Known systems: {self.coordinate_systems.get_all_systems()}"
            )

        # create_figure = ax is None
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for index, obj in enumerate(self.objects):
            local_frame = obj.name
            color = color_cycle[index % len(color_cycle)]

            local_points = self._sample_surface(obj, resolution)
            points = self._to_reference_frame(local_points, From=local_frame, To=reference_system)

            ax.plot_surface(
                points[..., 0],
                points[..., 1],
                points[..., 2],
                color=color,
                alpha=0.6,
                linewidth=0,
                antialiased=True,
            )
            # plot_surface produces no legend handle of its own, so label a proxy line.
            ax.plot([], [], [], color=color, label=obj.get_name())

            if show_normals:
                self._plot_normals(
                    ax, obj, local_points, local_frame, reference_system, normal_stride
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Scene in '{reference_system}' coordinates")
        ax.set_aspect("equal")
        ax.legend()

        if show_plot:
            plt.show()

        return ax

    def _plot_normals(
        self,
        ax: Axes3D,
        obj: RayTracerObject,
        local_points: np.ndarray,
        local_frame: str,
        reference_system: str,
        stride: int,
    ) -> None:
        """
        Draw the object's surface normals as arrows in the reference frame.

        Normals are directions, so they must not pick up the translation part of
        the transformation. Transforming base and tip separately and taking the
        difference leaves only the rotation.
        """
        local_normals = obj.surface_normal(local_points)

        bases = self._to_reference_frame(local_points, From=local_frame, To=reference_system)
        tips = self._to_reference_frame(
            local_points + local_normals, From=local_frame, To=reference_system
        )
        directions = tips - bases

        bases = bases[::stride, ::stride]
        directions = directions[::stride, ::stride]

        ax.quiver(
            bases[..., 0],
            bases[..., 1],
            bases[..., 2],
            directions[..., 0],
            directions[..., 1],
            directions[..., 2],
            color="black",
            length=0.3,
            normalize=True,
            arrow_length_ratio=0.3,
            linewidth=1,
        )
