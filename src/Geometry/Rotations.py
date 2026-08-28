import numpy as np


def rotz(theta: float) -> np.ndarray:
    """
    Returns a rotation matrix for a rotation about the z-axis by an angle theta (in radians).

    Parameters:
    theta : float
        The angle of rotation in radians.

    Returns:
    numpy.ndarray
        A 3x3 rotation matrix representing the rotation about the z-axis.
    """

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def roty(theta: float) -> np.ndarray:
    """
    Returns a rotation matrix for a rotation about the y-axis by an angle theta (in radians).

    Parameters:
    theta : float
        The angle of rotation in radians.

    Returns:
    numpy.ndarray
        A 3x3 rotation matrix representing the rotation about the y-axis.
    """

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def rotx(theta: float) -> np.ndarray:
    """
    Returns a rotation matrix for a rotation about the x-axis by an angle theta (in radians).

    Parameters:
    theta : float
        The angle of rotation in radians.

    Returns:
    numpy.ndarray
        A 3x3 rotation matrix representing the rotation about the x-axis.
    """

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
