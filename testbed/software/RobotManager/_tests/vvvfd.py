import math
import numpy as np

from robots.frodo.frodo_utilities import vector2LocalFrame

if __name__ == '__main__':
    vec = np.array([1, 2])

    psi = 2
    R_world_to_body = np.array([
        [math.cos(psi), math.sin(psi)],
        [-math.sin(psi), math.cos(psi)]
    ])

    vec_local = R_world_to_body @ vec

    vec_local_2 = vector2LocalFrame(vec, psi)

    print(vec_local)
    print(vec_local_2)
