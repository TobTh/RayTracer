import numpy as np

from Raytracer import RayTracerScene
from src.CoordinateTransformations import AffineTransformation, CoordinateTransformGraph
from src.Geometry.Rotations import rotx, rotz
from src.RayTracerObject import Plane, Sphere


def main():
    # Example usage of the CoordinateTransformations module
    point = (1, 2, 3)

    # Create an affine transformation (for example, a translation)
    translation_vector = (5, -3, 2)
    rotation = rotz(np.pi / 4)  # Rotate 45 degrees around the z-axis

    trafo = AffineTransformation("Sys1", "Sys2")
    trafo.set_translation(translation_vector)
    trafo.set_rotation(rotation)

    coordTree = CoordinateTransformGraph()
    coordTree.add_transformation(trafo)

    print(f"Original point: {point}")
    transformed_point = coordTree.transform_points(point, To="Sys1", From="Sys2")
    print(f"Transformed point: {transformed_point}")

    trafoMat = coordTree.get_transformation(From="Sys1", To="Sys2").get_transformation_matrix()
    print(f"Transformation matrix from Sys1 to Sys2:\n{trafoMat}")

    inverse_trafoMat = coordTree.get_transformation(
        From="Sys2", To="Sys1"
    ).get_transformation_matrix()
    print(f"Inverse transformation matrix from Sys2 to Sys1:\n{inverse_trafoMat}")

    transformed_point_inv = coordTree.transform_points(transformed_point, From="Sys1", To="Sys2")
    print(f"Transformed point back to original: {transformed_point_inv}")


if __name__ == "__main__":
    # main()

    trafo1 = AffineTransformation("World", "Plane")
    trafo1.set_transformation_matrix(np.eye(4))  # Identity transformation for Plane
    plane = Plane(coordinate_system=trafo1, name="Plane")

    trafo2 = AffineTransformation("World", "Plane2")
    trafo2.set_translation(np.array([1.0, 0.0, 0.0]))  # Translate Plane2 by (1, 0, 0)
    trafo2.set_rotation(rotz(np.pi / 4))  # Rotate Plane2 by 45 degrees around the z-axis
    plane2 = Plane(coordinate_system=trafo2, name="Plane2")

    trafo3 = AffineTransformation("World", "Sphere")
    trafo3.set_translation(np.array([0.0, 0.0, 1.0]))  # Translate Sphere by (0, 0, 1)
    trafo3.set_rotation(rotx(np.pi / 4))  # Rotate Sphere by 45 degrees around the z-axis
    sphere = Sphere(coordinate_system=trafo3, name="Sphere", radius=1.0)

    scene = RayTracerScene(objects=[sphere])
    scene.plot_scene(show_normals=True, resolution=100, reference_system="Sphere")
