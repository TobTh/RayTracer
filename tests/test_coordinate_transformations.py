import numpy as np
import pytest
from pydantic import ValidationError

from CoordinateTransformations import AffineTransformation, CoordinateTransformGraph

# Rotation by +90 degrees about the z axis, in the usual column-vector convention.
# Because AffineTransformation applies its matrix from the right (p' = p @ M),
# transforming a row vector with this block yields p @ RZ90.
RZ90 = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def valid_matrix(translation=(0.0, 0.0, 0.0), rotation=None) -> np.ndarray:
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[3, :3] = translation
    return matrix


def translation_between(system1, system2, translation) -> AffineTransformation:
    transformation = AffineTransformation(system1, system2)
    transformation.set_translation(translation)
    return transformation


def chain_graph() -> CoordinateTransformGraph:
    """A -> B -> C, a pure chain with no alternative routes."""
    graph = CoordinateTransformGraph()
    graph.add_transformation(translation_between("A", "B", [1.0, 0.0, 0.0]))
    graph.add_transformation(translation_between("B", "C", [0.0, 2.0, 0.0]))
    return graph


def inconsistent_diamond() -> CoordinateTransformGraph:
    """Two routes from A to D that deliberately disagree.

    A -> B -> D composes to a [1, 1, 0] shift, A -> C -> D to a [0, 0, 10] shift.
    Because the routes differ, the composed result reveals which one was walked,
    which is what makes `via` observable at all — a consistent graph would give
    the same answer either way.
    """
    graph = CoordinateTransformGraph()
    graph.add_transformation(translation_between("A", "B", [1.0, 0.0, 0.0]))
    graph.add_transformation(translation_between("B", "D", [0.0, 1.0, 0.0]))
    graph.add_transformation(translation_between("A", "C", [0.0, 0.0, 5.0]))
    graph.add_transformation(translation_between("C", "D", [0.0, 0.0, 5.0]))
    return graph


ORIGIN = np.array([[0.0, 0.0, 0.0]])


class TestConstruction:
    def test_defaults_to_identity(self):
        transformation = AffineTransformation("world", "camera")

        assert transformation.system1 == "world"
        assert transformation.system2 == "camera"
        np.testing.assert_array_equal(transformation.get_transformation_matrix(), np.eye(4))

    def test_accepts_explicit_matrix(self):
        matrix = valid_matrix(translation=(1.0, 2.0, 3.0))

        transformation = AffineTransformation("a", "b", matrix=matrix)

        np.testing.assert_array_equal(transformation.get_transformation_matrix(), matrix)

    def test_coerces_nested_lists_to_float_array(self):
        transformation = AffineTransformation("a", "b", matrix=valid_matrix().tolist())

        matrix = transformation.get_transformation_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float64

    def test_rejects_matrix_of_wrong_shape(self):
        with pytest.raises(ValidationError, match="must be 4x4"):
            AffineTransformation("a", "b", matrix=np.eye(3))

    def test_rejects_matrix_with_bad_last_column(self):
        matrix = valid_matrix()
        matrix[0, 3] = 0.5

        with pytest.raises(ValidationError, match=r"\[0, 0, 0, 1\]"):
            AffineTransformation("a", "b", matrix=matrix)

    def test_copies_the_matrix_it_is_given(self):
        matrix = valid_matrix()
        transformation = AffineTransformation("a", "b", matrix=matrix)

        transformation.set_translation([9.0, 9.0, 9.0])

        np.testing.assert_array_equal(matrix, np.eye(4))

    def test_later_edits_to_the_source_matrix_are_not_picked_up(self):
        matrix = valid_matrix()
        transformation = AffineTransformation("a", "b", matrix=matrix)

        matrix[3, :3] = [9.0, 9.0, 9.0]

        np.testing.assert_array_equal(transformation.get_transformation_matrix(), np.eye(4))


class TestTransform:
    def test_identity_leaves_points_unchanged(self):
        transformation = AffineTransformation("a", "b")
        points = np.array([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])

        np.testing.assert_allclose(transformation.transform(points), points)

    def test_translation_is_added_to_every_point(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_translation(np.array([1.0, 2.0, 3.0]))
        points = np.array([[4.0, 5.0, 6.0], [0.0, 0.0, 0.0]])

        np.testing.assert_allclose(
            transformation.transform(points),
            np.array([[5.0, 7.0, 9.0], [1.0, 2.0, 3.0]]),
        )

    def test_rotation_applies_matrix_from_the_right(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_rotation(RZ90)
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        np.testing.assert_allclose(transformation.transform(points), points @ RZ90, atol=1e-12)

    def test_rotation_then_translation(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_rotation(RZ90)
        transformation.set_translation([10.0, 0.0, 0.0])
        points = np.array([[1.0, 0.0, 0.0]])

        # p @ R + t
        np.testing.assert_allclose(
            transformation.transform(points),
            np.array([[10.0, -1.0, 0.0]]),
            atol=1e-12,
        )

    def test_preserves_input_shape(self):
        transformation = AffineTransformation("a", "b")
        points = np.zeros((7, 3))

        assert transformation.transform(points).shape == (7, 3)

    def test_accepts_a_single_point_and_returns_a_single_point(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_translation([1.0, 2.0, 3.0])

        result = transformation.transform(np.array([4.0, 5.0, 6.0]))

        assert result.shape == (3,)
        np.testing.assert_allclose(result, [5.0, 7.0, 9.0])

    def test_single_point_matches_the_batched_result(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_rotation(RZ90)
        transformation.set_translation([1.0, 2.0, 3.0])
        point = np.array([4.0, 5.0, 6.0])

        np.testing.assert_allclose(
            transformation.transform(point),
            transformation.transform(point.reshape(1, 3))[0],
        )

    def test_accepts_a_single_point_as_a_plain_list(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_translation([1.0, 1.0, 1.0])

        np.testing.assert_allclose(transformation.transform([0.0, 0.0, 0.0]), [1.0, 1.0, 1.0])

    def test_keeps_an_empty_batch_two_dimensional(self):
        transformation = AffineTransformation("a", "b")

        assert transformation.transform(np.zeros((0, 3))).shape == (0, 3)

    def test_accepts_list_of_points(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_translation([1.0, 1.0, 1.0])

        np.testing.assert_allclose(
            transformation.transform([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 1.0, 1.0]]),
        )

    def test_rejects_points_that_are_not_3d(self):
        transformation = AffineTransformation("a", "b")

        with pytest.raises(ValidationError, match="3-vector"):
            transformation.transform(np.zeros((2, 4)))

    def test_rejects_more_than_two_dimensions(self):
        transformation = AffineTransformation("a", "b")

        with pytest.raises(ValidationError, match="3-vector"):
            transformation.transform(np.zeros((2, 2, 3)))

    def test_does_not_mutate_input_points(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_translation([1.0, 2.0, 3.0])
        points = np.array([[4.0, 5.0, 6.0]])

        transformation.transform(points)

        np.testing.assert_array_equal(points, np.array([[4.0, 5.0, 6.0]]))


class TestSetters:
    def test_set_transformation_matrix_replaces_matrix(self):
        transformation = AffineTransformation("a", "b")
        matrix = valid_matrix(translation=(1.0, 2.0, 3.0), rotation=RZ90)

        transformation.set_transformation_matrix(matrix)

        np.testing.assert_array_equal(transformation.get_transformation_matrix(), matrix)

    def test_set_transformation_matrix_rejects_bad_last_column(self):
        transformation = AffineTransformation("a", "b")
        matrix = valid_matrix()
        matrix[3, 3] = 5.0

        with pytest.raises(ValidationError, match=r"\[0, 0, 0, 1\]"):
            transformation.set_transformation_matrix(matrix)

    def test_rejected_matrix_leaves_transformation_untouched(self):
        transformation = AffineTransformation("a", "b")

        with pytest.raises(ValidationError):
            transformation.set_transformation_matrix(np.zeros((2, 2)))

        np.testing.assert_array_equal(transformation.get_transformation_matrix(), np.eye(4))

    def test_set_translation_writes_bottom_row_only(self):
        transformation = AffineTransformation("a", "b")

        transformation.set_translation([1.0, 2.0, 3.0])

        matrix = transformation.get_transformation_matrix()
        np.testing.assert_array_equal(matrix[3, :3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(matrix[:3, :3], np.eye(3))
        np.testing.assert_array_equal(matrix[:, 3], [0.0, 0.0, 0.0, 1.0])

    def test_set_rotation_writes_upper_block_only(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_translation([1.0, 2.0, 3.0])

        transformation.set_rotation(RZ90)

        matrix = transformation.get_transformation_matrix()
        np.testing.assert_array_equal(matrix[:3, :3], RZ90)
        np.testing.assert_array_equal(matrix[3, :3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(matrix[:, 3], [0.0, 0.0, 0.0, 1.0])

    def test_set_translation_rejects_wrong_length(self):
        transformation = AffineTransformation("a", "b")

        with pytest.raises(ValidationError, match="3-vector"):
            transformation.set_translation([1.0, 2.0])

    def test_set_translation_rejects_a_batch_of_vectors(self):
        transformation = AffineTransformation("a", "b")

        with pytest.raises(ValidationError, match="3-vector"):
            transformation.set_translation(np.zeros((2, 3)))

    def test_set_translation_copies_the_vector_it_is_given(self):
        transformation = AffineTransformation("a", "b")
        translation = np.array([1.0, 2.0, 3.0])

        transformation.set_translation(translation)
        translation[0] = 99.0

        np.testing.assert_array_equal(
            transformation.get_transformation_matrix()[3, :3], [1.0, 2.0, 3.0]
        )

    def test_set_rotation_rejects_wrong_shape(self):
        transformation = AffineTransformation("a", "b")

        with pytest.raises(ValidationError, match="3x3"):
            transformation.set_rotation(np.eye(4))


class TestInvert:
    def test_swaps_the_coordinate_systems(self):
        transformation = AffineTransformation("world", "camera")

        inverted = transformation.invert_transformation()

        assert inverted.system1 == "camera"
        assert inverted.system2 == "world"

    def test_round_trip_recovers_original_points(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_rotation(RZ90)
        transformation.set_translation([1.0, 2.0, 3.0])
        points = np.array([[4.0, 5.0, 6.0], [-1.0, 0.0, 2.5]])

        inverted = transformation.invert_transformation()

        np.testing.assert_allclose(
            inverted.transform(transformation.transform(points)), points, atol=1e-12
        )

    def test_inverse_matrix_stays_a_valid_affine_matrix(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_rotation(RZ90)
        transformation.set_translation([1.0, 2.0, 3.0])

        inverse_matrix = transformation.invert_transformation().get_transformation_matrix()

        np.testing.assert_array_equal(inverse_matrix[:, 3], [0.0, 0.0, 0.0, 1.0])

    def test_does_not_modify_the_original(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_translation([1.0, 2.0, 3.0])
        before = transformation.get_transformation_matrix().copy()

        transformation.invert_transformation()

        np.testing.assert_array_equal(transformation.get_transformation_matrix(), before)


class TestConcatenate:
    def test_chains_the_coordinate_systems(self):
        first = AffineTransformation("world", "camera")
        second = AffineTransformation("camera", "screen")

        combined = first.concatenate(second)

        assert combined.system1 == "world"
        assert combined.system2 == "screen"

    def test_equals_applying_each_transformation_in_order(self):
        first = AffineTransformation("world", "camera")
        first.set_translation([1.0, 0.0, 0.0])
        second = AffineTransformation("camera", "screen")
        second.set_rotation(RZ90)
        points = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]])

        combined = first.concatenate(second)

        np.testing.assert_allclose(
            combined.transform(points),
            second.transform(first.transform(points)),
            atol=1e-12,
        )

    def test_concatenating_with_identity_is_a_no_op(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_rotation(RZ90)
        transformation.set_translation([1.0, 2.0, 3.0])
        identity = AffineTransformation("b", "c")

        combined = transformation.concatenate(identity)

        np.testing.assert_allclose(
            combined.get_transformation_matrix(),
            transformation.get_transformation_matrix(),
        )

    def test_concatenating_with_the_inverse_gives_the_identity(self):
        transformation = AffineTransformation("a", "b")
        transformation.set_rotation(RZ90)
        transformation.set_translation([1.0, 2.0, 3.0])

        combined = transformation.concatenate(transformation.invert_transformation())

        np.testing.assert_allclose(combined.get_transformation_matrix(), np.eye(4), atol=1e-12)


class TestGraphRegistration:
    def test_new_graph_has_no_systems(self):
        assert CoordinateTransformGraph().get_all_systems() == []

    def test_adding_a_transformation_registers_both_systems(self):
        graph = CoordinateTransformGraph()

        graph.add_transformation(translation_between("world", "camera", [1.0, 0.0, 0.0]))

        assert sorted(graph.get_all_systems()) == ["camera", "world"]

    def test_adding_a_transformation_makes_it_traversable_both_ways(self):
        graph = CoordinateTransformGraph()
        graph.add_transformation(translation_between("A", "B", [1.0, 2.0, 3.0]))

        forward = graph.get_transformation("A", "B")
        backward = graph.get_transformation("B", "A")

        np.testing.assert_allclose(forward.transform(ORIGIN), [[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(backward.transform(ORIGIN), [[-1.0, -2.0, -3.0]])

    def test_rejects_a_duplicate_transformation(self):
        graph = CoordinateTransformGraph()
        graph.add_transformation(translation_between("A", "B", [1.0, 0.0, 0.0]))

        with pytest.raises(AssertionError, match="already exists"):
            graph.add_transformation(translation_between("A", "B", [9.0, 9.0, 9.0]))

    def test_rejects_a_transformation_that_reverses_an_existing_one(self):
        graph = CoordinateTransformGraph()
        graph.add_transformation(translation_between("A", "B", [1.0, 0.0, 0.0]))

        # A -> B already installed the B -> A inverse edge, so this would silently
        # overwrite it with a conflicting transformation.
        with pytest.raises(AssertionError, match="already exists"):
            graph.add_transformation(translation_between("B", "A", [9.0, 9.0, 9.0]))

    def test_separate_components_can_coexist(self):
        graph = chain_graph()

        graph.add_transformation(translation_between("X", "Y", [5.0, 0.0, 0.0]))

        assert sorted(graph.get_all_systems()) == ["A", "B", "C", "X", "Y"]


class TestGraphLookup:
    def test_composes_transformations_along_a_multi_hop_path(self):
        graph = chain_graph()

        result = graph.get_transformation("A", "C").transform(ORIGIN)

        np.testing.assert_allclose(result, [[1.0, 2.0, 0.0]])

    def test_composes_the_reverse_path(self):
        graph = chain_graph()

        result = graph.get_transformation("C", "A").transform(ORIGIN)

        np.testing.assert_allclose(result, [[-1.0, -2.0, 0.0]])

    def test_result_is_labelled_with_the_requested_systems(self):
        graph = chain_graph()

        transformation = graph.get_transformation("A", "C")

        assert transformation.system1 == "A"
        assert transformation.system2 == "C"

    def test_a_system_to_itself_is_the_identity(self):
        graph = chain_graph()

        transformation = graph.get_transformation("B", "B")

        np.testing.assert_array_equal(transformation.get_transformation_matrix(), np.eye(4))

    def test_round_trip_through_the_graph_recovers_the_points(self):
        graph = chain_graph()
        points = np.array([[4.0, 5.0, 6.0], [-1.0, 0.0, 2.5]])

        there = graph.get_transformation("A", "C").transform(points)
        back = graph.get_transformation("C", "A").transform(there)

        np.testing.assert_allclose(back, points, atol=1e-12)

    def test_unknown_source_system_is_rejected(self):
        graph = chain_graph()

        with pytest.raises(ValueError, match="not in the graph"):
            graph.get_transformation("nope", "C")

    def test_unknown_target_system_is_rejected(self):
        graph = chain_graph()

        with pytest.raises(ValueError, match="not in the graph"):
            graph.get_transformation("A", "nope")

    def test_disconnected_systems_report_no_path(self):
        graph = chain_graph()
        graph.add_transformation(translation_between("X", "Y", [5.0, 0.0, 0.0]))

        with pytest.raises(ValueError, match="No transformation path"):
            graph.get_transformation("A", "Y")


class TestGraphVia:
    def test_via_routes_through_the_requested_system(self):
        graph = inconsistent_diamond()

        direct = graph.get_transformation("A", "D").transform(ORIGIN)
        via_c = graph.get_transformation("A", "D", via="C").transform(ORIGIN)

        np.testing.assert_allclose(direct, [[1.0, 1.0, 0.0]])
        np.testing.assert_allclose(via_c, [[0.0, 0.0, 10.0]])

    def test_via_on_the_route_already_taken_changes_nothing(self):
        graph = inconsistent_diamond()

        via_b = graph.get_transformation("A", "D", via="B").transform(ORIGIN)

        np.testing.assert_allclose(via_b, [[1.0, 1.0, 0.0]])

    def test_via_result_is_labelled_with_the_endpoints(self):
        graph = inconsistent_diamond()

        transformation = graph.get_transformation("A", "D", via="C")

        assert transformation.system1 == "A"
        assert transformation.system2 == "D"

    def test_via_a_detour_and_back_is_a_no_op(self):
        graph = chain_graph()
        # A -> C detouring through B is the only route anyway; going via A means
        # walking A -> A first, which must cancel out.
        via_a = graph.get_transformation("A", "C", via="A").transform(ORIGIN)

        np.testing.assert_allclose(via_a, [[1.0, 2.0, 0.0]], atol=1e-12)

    def test_via_an_unreachable_system_reports_no_path(self):
        graph = inconsistent_diamond()
        graph.add_transformation(translation_between("X", "Y", [5.0, 0.0, 0.0]))

        with pytest.raises(ValueError, match="No transformation path"):
            graph.get_transformation("A", "D", via="X")


class TestGraphTransformPoints:
    def test_matches_looking_up_the_transformation_directly(self):
        graph = chain_graph()
        points = np.array([[4.0, 5.0, 6.0], [0.0, 0.0, 0.0]])

        np.testing.assert_allclose(
            graph.transform_points(points, From="A", To="C"),
            graph.get_transformation("A", "C").transform(points),
        )

    def test_transforms_a_batch_of_points(self):
        graph = chain_graph()
        points = np.array([[4.0, 5.0, 6.0], [0.0, 0.0, 0.0]])

        np.testing.assert_allclose(
            graph.transform_points(points, From="A", To="C"),
            np.array([[5.0, 7.0, 6.0], [1.0, 2.0, 0.0]]),
        )

    def test_transforms_a_single_point(self):
        graph = chain_graph()

        result = graph.transform_points(np.array([4.0, 5.0, 6.0]), From="A", To="C")

        assert result.shape == (3,)
        np.testing.assert_allclose(result, [5.0, 7.0, 6.0])

    def test_honours_via(self):
        graph = inconsistent_diamond()

        np.testing.assert_allclose(
            graph.transform_points(ORIGIN, From="A", To="D", Via="C"),
            [[0.0, 0.0, 10.0]],
        )

    def test_missing_endpoints_are_rejected(self):
        graph = chain_graph()

        with pytest.raises(ValueError, match="not in the graph"):
            graph.transform_points(ORIGIN)
