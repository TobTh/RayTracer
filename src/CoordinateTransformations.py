from typing import Annotated

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, validate_call
from pydantic.functional_validators import AfterValidator, BeforeValidator


def _to_ndarray(value) -> np.ndarray:
    # np.array (unlike np.asarray) always copies, so a caller's array is never
    # aliased into — and later mutated by — the transformation that received it.
    return np.array(value, dtype=float)


def _check_vector3(value: np.ndarray) -> np.ndarray:
    if value.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {value.shape}.")
    return value


def _check_points3(value: np.ndarray) -> np.ndarray:
    if value.ndim not in (1, 2) or value.shape[-1] != 3:
        raise ValueError(
            f"Expected a 3-vector or an (n, 3) array of 3-vectors, got shape {value.shape}."
        )
    return value


def _check_matrix3(value: np.ndarray) -> np.ndarray:
    if value.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got shape {value.shape}.")
    return value


def _check_transformation_matrix(value: np.ndarray) -> np.ndarray:
    if value.shape != (4, 4):
        raise ValueError(f"Transformation matrix must be 4x4, got shape {value.shape}.")
    if not np.allclose(value[:4, 3], [0, 0, 0, 1]):
        raise ValueError("Last column of transformation matrix must be [0, 0, 0, 1].")
    return value


Vector3 = Annotated[np.ndarray, BeforeValidator(_to_ndarray), AfterValidator(_check_vector3)]
Points3 = Annotated[np.ndarray, BeforeValidator(_to_ndarray), AfterValidator(_check_points3)]
Matrix3 = Annotated[np.ndarray, BeforeValidator(_to_ndarray), AfterValidator(_check_matrix3)]
TransformationMatrix = Annotated[
    np.ndarray, BeforeValidator(_to_ndarray), AfterValidator(_check_transformation_matrix)
]

_VALIDATE_CONFIG = ConfigDict(arbitrary_types_allowed=True)


class AffineTransformation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The transformation matrix is a 4x4 matrix representing the affine transformation
    # in homogeneous coordinates. It is applied from the right, meaning coordinates need
    # to be given as row vectors (nx3) for the multiplication to work correctly.
    system1: str
    system2: str
    matrix: TransformationMatrix = Field(default_factory=lambda: np.eye(4))

    def __init__(
        self, system1: str, system2: str, matrix: np.ndarray | None = None, **data
    ) -> None:
        super().__init__(
            system1=system1,
            system2=system2,
            matrix=matrix if matrix is not None else np.eye(4),
            **data,
        )

    @validate_call(config=_VALIDATE_CONFIG)
    def transform(self, points: Points3) -> np.ndarray:
        # A single point comes back as a single point, an (n, 3) array as an (n, 3) array.
        is_single_point = points.ndim == 1
        points = np.atleast_2d(points)

        homogeneous_points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
        transformed_points = (homogeneous_points @ self.matrix)[:, :3]

        return transformed_points[0] if is_single_point else transformed_points

    def get_transformation_matrix(self) -> np.ndarray:
        return self.matrix

    @validate_call(config=_VALIDATE_CONFIG)
    def set_transformation_matrix(self, matrix: TransformationMatrix) -> None:
        self.matrix = matrix

    @validate_call(config=_VALIDATE_CONFIG)
    def set_translation(self, translation_vector: Vector3) -> None:
        self.matrix[3, :3] = translation_vector

    @validate_call(config=_VALIDATE_CONFIG)
    def set_rotation(self, rotation_matrix: Matrix3) -> None:
        self.matrix[:3, :3] = rotation_matrix

    def invert_transformation(self) -> "AffineTransformation":
        inverse_matrix = np.linalg.inv(self.matrix)
        inverse_matrix[:, 3] = np.array([0, 0, 0, 1])

        return AffineTransformation(self.system2, self.system1, matrix=inverse_matrix)

    def concatenate(self, other: "AffineTransformation") -> "AffineTransformation":
        concatenated_matrix = self.matrix @ other.get_transformation_matrix()
        return AffineTransformation(self.system1, other.system2, matrix=concatenated_matrix)


class CoordinateTransformGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_transformation(self, transformation: AffineTransformation) -> None:

        assert not self.graph.has_edge(transformation.system1, transformation.system2), (
            f"Transformation from {transformation.system1} to {transformation.system2} already exists."
        )

        self.graph.add_edge(
            transformation.system1, transformation.system2, transformation=transformation
        )
        self.graph.add_edge(
            transformation.system2,
            transformation.system1,
            transformation=transformation.invert_transformation(),
        )

    def get_transformation(self, system1: str, system2: str, via: str = "") -> AffineTransformation:
        if not self.graph.has_node(system1) or not self.graph.has_node(system2):
            raise ValueError(
                f"One or both systems '{system1}' and '{system2}' are not in the graph."
            )

        try:
            if via and self.graph.has_node(via):
                path = nx.shortest_path(self.graph, source=system1, target=via, weight=None)
                path_to_via = nx.shortest_path(self.graph, source=system1, target=via, weight=None)
                if via not in path_to_via:
                    raise ValueError(
                        f"No transformation path found from {system1} to {system2} via {via}."
                    )
                path_from_via = nx.shortest_path(
                    self.graph, source=via, target=system2, weight=None
                )
                path = path_to_via[:-1] + path_from_via
            else:
                path = nx.shortest_path(self.graph, source=system1, target=system2)
        except nx.NetworkXNoPath:
            raise ValueError(f"No transformation path found from {system1} to {system2}.")

        transformation = AffineTransformation(system1, system1)  # Identity transformation
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            transformation = transformation.concatenate(edge_data["transformation"])

        return transformation

    def transform_points(
        self, points: Points3, From: str = "", To: str = "", Via: str = ""
    ) -> np.ndarray:
        transformation = self.get_transformation(From, To, Via)
        return transformation.transform(points)

    def get_all_systems(self) -> list[str]:
        return list(self.graph.nodes)

    def plot_graph(self) -> None:

        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=2000,
            font_size=10,
            font_weight="bold",
            arrowsize=20,
        )
        edge_labels = {
            (u, v): f"{data['transformation'].system1}->{data['transformation'].system2}"
            for u, v, data in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color="red")
        plt.title("Coordinate Transformation Graph")
        plt.show()

    def plot_coordinate_systems_3d(self) -> None:
        """Plot all coordinate systems in 3D space with their unit vectors."""

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        colors = ["red", "green", "blue"]
        labels = ["X", "Y", "Z"]

        # Plot each coordinate system
        for system in self.get_all_systems():
            # Get origin (position) from identity transformation
            try:
                transformation = self.get_transformation(system, system)
            except:  # noqa: E722
                # If system is not connected to itself, create identity
                transformation = AffineTransformation(system, system)

            origin = transformation.matrix[3, :3]

            # Plot the three unit vectors (X, Y, Z axes)
            for i, (color, label) in enumerate(zip(colors, labels)):
                unit_vector = np.zeros(3)
                unit_vector[i] = 1.0
                # Transform the unit vector to the target system's frame
                transformed_vector = transformation.transform(unit_vector)
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    transformed_vector[0] - origin[0],
                    transformed_vector[1] - origin[1],
                    transformed_vector[2] - origin[2],
                    color=color,
                    arrow_length_ratio=0.2,
                    linewidth=2,
                )

            # Label the origin
            ax.text(origin[0], origin[1], origin[2], system, fontsize=10, fontweight="bold")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Coordinate Systems")
        plt.show()
