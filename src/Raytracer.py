import matplotlib.pyplot as plt
import numpy as np

import CoordinateTransformations as ct


class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)  # Normalize the direction

    def transform(self, transformation: ct.AffineTransformation) -> "Ray":
        transformed_origin = transformation.transform(self.origin)
        transformed_direction = (
            transformation.transform(self.origin + self.direction) - transformed_origin
        )
        return Ray(transformed_origin, transformed_direction)

    def __repr__(self):
        return f"Ray(origin={self.origin}, direction={self.direction})"

    def __str__(self):
        return f"Ray with origin {self.origin} and direction {self.direction}"

    def plot(self, ax=None, length=1.0, color="blue", label=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        # end_point = self.origin + self.direction * length
        ax.quiver(
            self.origin[0],
            self.origin[1],
            self.origin[2],
            self.direction[0],
            self.direction[1],
            self.direction[2],
            length=length,
            color=color,
            label=label,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Ray Visualization")
        ax.set_aspect("equal")

        if label:
            ax.legend()

        plt.show()


class RayTracerScene:
    def __init__(self, objects):
        self.objects = objects
        self.coordinate_systems = ct.CoordinateTransformGraph()
        self.object_names = [obj.get_name() for obj in objects]

        for obj in objects:
            self.coordinate_systems.add_transformation(obj.get_coordinate_system())

    def add_object(self, obj):
        self.objects.append(obj)
        self.object_names.append(obj.get_name())
        self.coordinate_systems.add_transformation(obj.get_coordinate_system())

    def get_object_by_name(self, name):
        for obj in self.objects:
            if obj.get_name() == name:
                return obj
        raise ValueError(f"Object with name '{name}' not found in the scene.")

    def update_object(self, name, updated_object):
        self.get_object_by_name(name)
        self.coordinate_systems.add_transformation(updated_object.get_coordinate_system())

    def _sample_surface(self, obj, resolution: int) -> np.ndarray:
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
        ax=None,
    ):
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
            reference_system = self.objects[0].get_coordinate_system().system1

        if reference_system not in self.coordinate_systems.get_all_systems():
            raise ValueError(
                f"Reference system '{reference_system}' is not part of the scene. "
                f"Known systems: {self.coordinate_systems.get_all_systems()}"
            )

        show_plot = ax is None
        if show_plot:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for index, obj in enumerate(self.objects):
            local_frame = obj.get_coordinate_system().system2
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

    def _plot_normals(self, ax, obj, local_points, local_frame, reference_system, stride):
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
