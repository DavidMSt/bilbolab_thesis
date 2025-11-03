import math

import numpy as np

frodo_virtual_agent_colors = {
    'frodo1_v': [0.25, 0.4, 0.13],
    'frodo2_v': [0.62, 0.19, 0.74],
    'frodo3_v': [0.1, 0.6, 0.4],
    'frodo4_v': [0.7, 0.2, 0.1],
}


def vector_is_between(v, v1, v2):
    if np.cross(v1, v) * np.cross(v1, v2) >= 0 and np.cross(v2, v) * np.cross(v2, v1) >= 0:
        return True
    else:
        return False


def get_fov_vectors(psi, fov):
    v_ori = np.array([math.cos(psi), math.sin(psi)])
    alpha = fov / 2
    rotmat1 = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
    rotmat2 = np.array([[math.cos(-alpha), -math.sin(-alpha)], [math.sin(-alpha), math.cos(-alpha)]])
    v1 = rotmat1 @ v_ori
    v2 = rotmat2 @ v_ori

    return v1, v2


def is_in_fov(pos, psi, fov, radius, other_agent_pos):
    v1, v2 = get_fov_vectors(psi, fov)
    if vector_is_between(other_agent_pos - pos, v1, v2) and np.linalg.norm(other_agent_pos-pos) <= radius:
        return True
    else:
        return False
