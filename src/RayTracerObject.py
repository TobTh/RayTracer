from abc import ABC, abstractmethod

import numpy as np

import CoordinateTransformations as ct


class RayTracerObject(ABC):
    def __init__(self, coordinate_system: ct.AffineTransformation, name: str = "Unnamed Object"):
        self.coordinate_system = coordinate_system
        self.name = name

    @abstractmethod
    def surface_normal(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate the surface normal at a given point on the object.

        Args:
            point (np.ndarray): The point on the object's surface.

        Returns:
            np.ndarray: The surface normal vector at the given point.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def surface_height(self, point: np.ndarray) -> float:
        """
        Calculate the surface height at a given point on the object.

        Args:
            point (np.ndarray): The point on the object's surface.

        Returns:
            float: The surface height at the given point.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def update_coordinate_system(self, new_coordinate_system: ct.AffineTransformation):
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
