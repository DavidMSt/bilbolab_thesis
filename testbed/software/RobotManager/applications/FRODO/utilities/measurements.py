import dataclasses
import math

import numpy as np
import qmt

from applications.FRODO.utilities.measurement_model import FRODO_MeasurementModel
from core.utils.states import State
from robots.frodo.frodo_utilities import vector2LocalFrame


@dataclasses.dataclass
class FRODO_State(State):
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    v: float = 0.0
    psi_dot: float = 0.0


# ======================================================================================================================
def _vector_is_between(v, v1, v2):
    if np.cross(v1, v) * np.cross(v1, v2) >= 0 and np.cross(v2, v) * np.cross(v2, v1) >= 0:
        return True
    else:
        return False


def _get_fov_vectors(psi, fov):
    v_ori = np.array([math.cos(psi), math.sin(psi)])
    alpha = fov / 2
    rotmat1 = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
    rotmat2 = np.array([[math.cos(-alpha), -math.sin(-alpha)], [math.sin(-alpha), math.cos(-alpha)]])
    v1 = rotmat1 @ v_ori
    v2 = rotmat2 @ v_ori

    return v1, v2


def agent_is_in_fov(agent_from_state: FRODO_State | np.ndarray,
                    agent_to_state: FRODO_State | np.ndarray,
                    agent_from_fov: float,
                    agent_from_min_distance: float = 0.0,
                    agent_from_max_distance: float | None = None) -> bool:
    if isinstance(agent_from_state, np.ndarray):
        agent_from_state = FRODO_State.fromarray(agent_from_state)

    if isinstance(agent_to_state, np.ndarray):
        agent_to_state = FRODO_State.fromarray(agent_to_state)

    agent_from_position = np.array([agent_from_state.x, agent_from_state.y])
    agent_to_position = np.array([agent_to_state.x, agent_to_state.y])
    distance = np.linalg.norm(agent_from_position - agent_to_position)

    if agent_from_max_distance is not None and distance > agent_from_max_distance:
        return False
    elif distance < agent_from_min_distance:
        return False

    v1, v2 = _get_fov_vectors(agent_from_state.psi, agent_from_fov)
    if not _vector_is_between(agent_to_position - agent_from_position, v1, v2):
        return False
    else:
        return True


def generate_ideal_measurement(agent_from_state: FRODO_State | np.ndarray,
                               agent_to_state: FRODO_State | np.ndarray) -> np.ndarray:
    if isinstance(agent_from_state, np.ndarray):
        agent_from_state = FRODO_State.fromarray(agent_from_state)

    if isinstance(agent_to_state, np.ndarray):
        agent_to_state = FRODO_State.fromarray(agent_to_state)

    position_agent_from = np.array([agent_from_state.x, agent_from_state.y])
    position_agent_to = np.array([agent_to_state.x, agent_to_state.y])

    position_relative_global = position_agent_to - position_agent_from
    position_relative_local = vector2LocalFrame(position_relative_global, agent_from_state.psi)
    psi_local = qmt.wrapToPi(agent_to_state.psi - agent_from_state.psi)

    return np.array([position_relative_local[0], position_relative_local[1], psi_local])


def generate_noisy_measurement(agent_from_state: FRODO_State | np.ndarray,
                               agent_to_state: FRODO_State | np.ndarray,
                               measurement_model: FRODO_MeasurementModel) -> tuple[np.ndarray, np.ndarray]:
    ideal_measurement = generate_ideal_measurement(agent_from_state, agent_to_state)

    covariance = measurement_model.get_covariance(ideal_measurement, agent_from_state.v, agent_from_state.psi_dot)
    bias_x = measurement_model.bias_x
    bias_y = measurement_model.bias_y
    bias_psi = measurement_model.bias_psi

    # Add bias
    measurement = ideal_measurement + np.array([bias_x, bias_y, bias_psi])

    # Add noise
    measurement += np.random.multivariate_normal(np.zeros(3), covariance)

    return measurement, covariance
