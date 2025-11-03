import numpy as np


def is_mostly_z_axis(rotation_vector, threshold=0.2):
    """
    Checks if a rotation vector is mostly around the z-axis.

    Parameters:
        rotation_vector (list or np.ndarray): A 3-element list or array representing the rotation vector [x, y, z].
        threshold (float): A value defining how small x and y should be relative to z to consider it mostly z-axis.

    Returns:
        bool: True if the vector is mostly around the z-axis, False otherwise.
    """
    rotation_vector = np.asarray(rotation_vector)

    if len(rotation_vector) != 3:
        raise ValueError("Rotation vector must have exactly 3 components.")

    xy_magnitude = np.linalg.norm(rotation_vector[:2])  # Magnitude of x and y components
    z_magnitude = abs(rotation_vector[2])  # Absolute value of z component

    return xy_magnitude < threshold * z_magnitude