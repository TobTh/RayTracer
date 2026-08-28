from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

import CoordinateTransformations as ct


class RayTracerObject(ABC):
    def __init__(
        self,
        coordinate_system: ct.AffineTransformation,
        name: str = "Unnamed Object",
        normalDirection: float = 1.0,
        xExtent: float = 1.0,
        yExtent: float = 1.0,
    ):
        """
        Abstract base class for ray tracer objects."""
        self.coordinate_system = coordinate_system
        self.name = name
        self.normalDirection = normalDirection
        self.xExtent = xExtent
        self.yExtent = yExtent

    def surface_normal(self, point: ArrayLike) -> np.ndarray:
        points = self._as_points(point)
        return self._surface_normal(points)

    def surface_height(self, point: ArrayLike) -> np.ndarray:
        points = self._as_points(point)
        return self._surface_height(points)

    @staticmethod
    def _as_points(point: ArrayLike) -> np.ndarray:
        points = np.asarray(point, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("point must have shape (..., 3).")
        return points

    @abstractmethod
    def _surface_normal(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate the surface normal at a given point on the object.

        Args:
            point (np.ndarray): The point on the object's surface.

        Returns:
            np.ndarray: The surface normal vector at the given point.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _surface_height(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate the surface height at a given point on the object.

        Args:
            point (np.ndarray): The point on the object's surface.

        Returns:
            np.ndarray: The surface height at the given point.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def update_coordinate_system(self, new_coordinate_system: ct.AffineTransformation) -> None:
        """
        Update the object's coordinate system.

        Args:
            new_coordinate_system (ct.AffineTransformation): The new coordinate system to be set.
        """
        self.coordinate_system = new_coordinate_system

    def get_coordinate_system(self) -> ct.AffineTransformation:
        """
        Get the object's current coordinate system.

        Returns:
            ct.AffineTransformation: The current coordinate system of the object.
        """
        return self.coordinate_system

    def get_name(self) -> str:
        """
        Get the name of the object.

        Returns:
            str: The name of the object.
        """
        return self.name


class Plane(RayTracerObject):
    def __init__(
        self,
        coordinate_system: ct.AffineTransformation,
        name: str = "Plane",
        normalDirection: float = 1.0,
        xExtent: float = 1.0,
        yExtent: float = 1.0,
    ):
        super().__init__(coordinate_system, name, normalDirection, xExtent, yExtent)

    def _surface_normal(self, point: np.ndarray) -> np.ndarray:

        valid_mask = (abs(point[..., 0]) <= self.xExtent) & (abs(point[..., 1]) <= self.yExtent)

        normals = np.zeros_like(point)
        normals[..., 2] = (
            self.normalDirection
        )  # Normal vector points in the positive z-direction for a plane in its local coordinate system
        normals[~valid_mask, ...] = np.nan  # Set normals to NaN for points outside the valid extent
        return normals

    def _surface_height(self, point: np.ndarray) -> np.ndarray:

        # For a plane, the height is constant. Assuming the plane is at z=0 in its local coordinate system,

        valid_mask = (abs(point[..., 0]) <= self.xExtent) & (abs(point[..., 1]) <= self.yExtent)

        surface_height = np.zeros_like(
            point[..., 0]
        )  # Height is zero for a plane in its local coordinate system

        surface_height[~valid_mask, ...] = (
            np.nan
        )  # Set height to NaN for points outside the valid extent
        return surface_height


class Sphere(RayTracerObject):
    def __init__(
        self,
        coordinate_system: ct.AffineTransformation,
        radius: float,
        name: str = "Sphere",
        normalDirection: float = 1.0,
        xExtent: float = 1.0,
        yExtent: float = 1.0,
    ):
        super().__init__(coordinate_system, name, normalDirection, xExtent, yExtent)
        self.radius = radius

    def _surface_normal(self, point: np.ndarray) -> np.ndarray:
        # For a sphere centered at the origin in its local coordinate system, the normal at a point is the normalized vector from the center to that point.

        normals = np.stack(
            (point[..., 0], point[..., 1], self._surface_height(point) - self.radius), axis=-1
        )
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)  # Normalize the normals
        return (
            -normals * self.normalDirection
        )  # Adjust for normal direction, assuming the default snormal points inward for a sphere

    def _surface_height(self, point: np.ndarray) -> np.ndarray:
        # For a sphere centered at the origin in its local coordinate system, the height is given by the z-coordinate of the point.

        z = np.sqrt(self.radius**2 - point[..., 0] ** 2 - point[..., 1] ** 2)
        surface_height: np.ndarray = (
            self.radius - z
        )  # Height is the distance from the top of the sphere to the point's z-coordinate
        return surface_height  # Adjust for normal direction


class Paraboloid(RayTracerObject):
    def __init__(
        self,
        coordinate_system: ct.AffineTransformation,
        a: float,
        name: str = "Paraboloid",
        normalDirection: float = 1.0,
    ):
        super().__init__(coordinate_system, name, normalDirection)
        self.a = a  # The parameter 'a' defines the shape of the paraboloid

    def _surface_normal(self, point: np.ndarray) -> np.ndarray:
        # For a paraboloid defined by z = (x^2 + y^2) / (4a), the normal vector can be computed as follows:
        x = point[..., 0]
        y = point[..., 1]
        z = (x**2 + y**2) / (4 * self.a)

        # The gradient of the surface function gives the normal vector
        normals = np.stack((x / (2 * self.a), y / (2 * self.a), -np.ones_like(z)), axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)  # Normalize the normals
        return normals * self.normalDirection  # Adjust for normal direction

    def _surface_height(self, point: np.ndarray) -> np.ndarray:
        # For a paraboloid defined by z = (x^2 + y^2) / (4a), the height is given by this equation.
        x = point[..., 0]
        y = point[..., 1]
        height: np.ndarray = (x**2 + y**2) / (4 * self.a)
        return height
