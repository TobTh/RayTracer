import numpy as np

from Geometry.Rotations import rotx, roty, rotz


class TestRotations:
    def test_rotx(self):

        theta = np.pi / 4  # 45 degrees
        expected_matrix = np.array(
            [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]]
        )

        assert np.allclose(rotx(theta), expected_matrix), "rotx function failed the test."

    def test_roty(self):
        theta = np.pi / 4  # 45 degrees
        expected_matrix = np.array(
            [[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]]
        )

        assert np.allclose(roty(theta), expected_matrix), "roty function failed the test."

    def test_rotz(self):
        theta = np.pi / 4  # 45 degrees
        expected_matrix = np.array(
            [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )

        assert np.allclose(rotz(theta), expected_matrix), "rotz function failed the test."

    def test_rotations_inverse(self):
        theta = np.pi / 4  # 45 degrees

        # Test inverse property for rotx
        assert np.allclose(rotx(theta) @ rotx(-theta), np.eye(3)), (
            "Inverse property failed for rotx."
        )
        assert np.allclose(rotx(theta) @ rotx(theta).T, np.eye(3)), (
            "Inverse should equal transpose for rotx."
        )

        # Test inverse property for roty
        assert np.allclose(roty(theta) @ roty(-theta), np.eye(3)), (
            "Inverse property failed for roty."
        )
        assert np.allclose(roty(theta) @ roty(theta).T, np.eye(3)), (
            "Inverse should equal transpose for roty."
        )

        # Test inverse property for rotz
        assert np.allclose(rotz(theta) @ rotz(-theta), np.eye(3)), (
            "Inverse property failed for rotz."
        )
        assert np.allclose(rotz(theta) @ rotz(theta).T, np.eye(3)), (
            "Inverse should equal transpose for rotz."
        )

    def test_rotations_composition(self):
        theta1 = np.pi / 4  # 45 degrees
        theta2 = np.pi / 6  # 30 degrees

        # Test composition property for rotx
        assert np.allclose(rotx(theta1) @ rotx(theta2), rotx(theta1 + theta2)), (
            "Composition property failed for rotx."
        )

        # Test composition property for roty
        assert np.allclose(roty(theta1) @ roty(theta2), roty(theta1 + theta2)), (
            "Composition property failed for roty."
        )

        # Test composition property for rotz
        assert np.allclose(rotz(theta1) @ rotz(theta2), rotz(theta1 + theta2)), (
            "Composition property failed for rotz."
        )

    def test_rotations_orthogonality(self):
        theta = np.pi / 4  # 45 degrees

        # Test orthogonality for rotx
        assert np.allclose(rotx(theta).T @ rotx(theta), np.eye(3)), (
            "Orthogonality property failed for rotx."
        )

        # Test orthogonality for roty
        assert np.allclose(roty(theta).T @ roty(theta), np.eye(3)), (
            "Orthogonality property failed for roty."
        )

        # Test orthogonality for rotz
        assert np.allclose(rotz(theta).T @ rotz(theta), np.eye(3)), (
            "Orthogonality property failed for rotz."
        )

    def test_rotations_determinant(self):
        theta = np.pi / 4  # 45 degrees

        # Test determinant for rotx
        assert np.isclose(np.linalg.det(rotx(theta)), 1), "Determinant property failed for rotx."

        # Test determinant for roty
        assert np.isclose(np.linalg.det(roty(theta)), 1), "Determinant property failed for roty."

        # Test determinant for rotz
        assert np.isclose(np.linalg.det(rotz(theta)), 1), "Determinant property failed for rotz."

    def test_rotations_identity(self):
        # Test identity property for rotx
        assert np.allclose(rotx(0), np.eye(3)), "Identity property failed for rotx."

        # Test identity property for roty
        assert np.allclose(roty(0), np.eye(3)), "Identity property failed for roty."

        # Test identity property for rotz
        assert np.allclose(rotz(0), np.eye(3)), "Identity property failed for rotz."

    def test_rotations_composition_with_identity(self):
        theta = np.pi / 4  # 45 degrees

        # Test composition with identity for rotx
        assert np.allclose(rotx(theta) @ np.eye(3), rotx(theta)), (
            "Composition with identity failed for rotx."
        )

        # Test composition with identity for roty
        assert np.allclose(roty(theta) @ np.eye(3), roty(theta)), (
            "Composition with identity failed for roty."
        )

        # Test composition with identity for rotz
        assert np.allclose(rotz(theta) @ np.eye(3), rotz(theta)), (
            "Composition with identity failed for rotz."
        )

    def test_rotations_inverse_with_identity(self):
        theta = np.pi / 4  # 45 degrees

        # Test inverse with identity for rotx
        assert np.allclose(rotx(theta) @ rotx(-theta), np.eye(3)), (
            "Inverse with identity failed for rotx."
        )

        # Test inverse with identity for roty
        assert np.allclose(roty(theta) @ roty(-theta), np.eye(3)), (
            "Inverse with identity failed for roty."
        )

        # Test inverse with identity for rotz
        assert np.allclose(rotz(theta) @ rotz(-theta), np.eye(3)), (
            "Inverse with identity failed for rotz."
        )

    def test_rotations_combined(self):
        theta_x = np.pi / 4  # 45 degrees
        theta_y = np.pi / 6  # 30 degrees
        theta_z = np.pi / 3  # 60 degrees

        combined_rotation = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)

        # Test that the combined rotation is still a valid rotation matrix
        assert np.allclose(combined_rotation.T @ combined_rotation, np.eye(3)), (
            "Combined rotation is not orthogonal."
        )
        assert np.isclose(np.linalg.det(combined_rotation), 1), (
            "Combined rotation does not have determinant of 1."
        )

    def test_rotations_inverse_combined(self):
        theta_x = np.pi / 4  # 45 degrees
        theta_y = np.pi / 6  # 30 degrees
        theta_z = np.pi / 3  # 60 degrees

        combined_rotation = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)
        inverse_combined_rotation = rotx(-theta_x) @ roty(-theta_y) @ rotz(-theta_z)

        # Test that the inverse of the combined rotation is correct
        assert np.allclose(combined_rotation @ inverse_combined_rotation, np.eye(3)), (
            "Inverse of combined rotation is incorrect."
        )

    def test_rotations_properties(self):
        theta_x = np.pi / 4  # 45 degrees
        theta_y = np.pi / 6  # 30 degrees
        theta_z = np.pi / 3  # 60 degrees

        combined_rotation = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)

        # Test orthogonality
        assert np.allclose(combined_rotation.T @ combined_rotation, np.eye(3)), (
            "Combined rotation is not orthogonal."
        )

        # Test determinant
        assert np.isclose(np.linalg.det(combined_rotation), 1), (
            "Combined rotation does not have determinant of 1."
        )

    def test_apply_rotation_to_point(self):
        theta = np.pi / 4  # 45 degrees
        point = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        rotated_point_z = point @ rotz(theta)

        expected_point = np.array(
            [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        print(f"Rotated point: {rotated_point_z}, Expected point: {expected_point}")
        assert np.allclose(rotated_point_z, expected_point), "Applying Rz to point failed."

        rotated_point_y = point @ roty(theta)
        expected_point = np.array(
            [[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]]
        )
        print(f"Rotated point: {rotated_point_y}, Expected point: {expected_point}")
        assert np.allclose(rotated_point_y, expected_point), "Applying Ry to point failed."

        rotated_point_x = point @ rotx(theta)
        expected_point = np.array(
            [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]]
        )
        print(f"Rotated point: {rotated_point_x}, Expected point: {expected_point}")
        assert np.allclose(rotated_point_x, expected_point), "Applying Rx to point failed."

    def test_apply_rotation_to_point_transpose(self):
        theta = np.pi / 4  # 45 degrees
        point = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        rotated_point_z = point @ rotz(theta)
        rotated_point_z_transpose = rotz(-theta) @ point.T

        assert np.allclose(rotated_point_z_transpose.T, rotated_point_z), (
            "Applying Rz^T to point failed."
        )

        rotated_point_y = point @ roty(theta)
        rotated_point_y_transpose = roty(-theta) @ point.T
        assert np.allclose(rotated_point_y_transpose.T, rotated_point_y), (
            "Applying Ry^T to point failed."
        )

        rotated_point_x = point @ rotx(theta)
        rotated_point_x_transpose = rotx(-theta) @ point.T
        assert np.allclose(rotated_point_x_transpose.T, rotated_point_x), (
            "Applying Rx^T to point failed."
        )
