import numpy as np

VECTOR_FRODO_TO_CAMERA = np.ndarray([1, 2, 3])
VECTOR_FRODO_TO_ARUCO = np.ndarray([1, 2, 3])
VECTOR_ORIGIN_TO_ARUCO_FRONT = np.ndarray([1, 2, 3])
VECTOR_ORIGIN_TO_ARUCO_BACK = np.ndarray([1, 2, 3])

ORIGIN_MARKER_FRONT_ID = 0
ORIGIN_MARKER_BACK_ID = 1
ORIGIN_MARKER_THICKNESS = 0.007

FRODO_MARKER_IDS = {
    'frodo1': {
        'front': 10,
        'back': 11
    },
    'frodo2': {
        'front': 12,
        'back': 13
    },
    'frodo3': {
        'front': 14,
        'back': 15
    },
    'frodo4': {
        'front': 16,
        'back': 17
    }
}
