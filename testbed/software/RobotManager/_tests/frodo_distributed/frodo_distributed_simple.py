import dataclasses
from typing import Optional, Tuple

import numpy as np

USE_CI = True

INDEX_X = 0
INDEX_Y = 1
INDEX_PSI = 2


def ci_fuse(mean1: np.ndarray, cov1: np.ndarray,
            mean2: np.ndarray, cov2: np.ndarray,
            omega: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse two Gaussians N(mean1, cov1) and N(mean2, cov2) using Covariance Intersection.
    If omega is None, choose omega in [0,1] by a simple 1D line search minimizing trace of the fused covariance.
    Returns (fused_mean, fused_cov).
    """
    # Safeguards
    cov1 = 0.5 * (cov1 + cov1.T)
    cov2 = 0.5 * (cov2 + cov2.T)

    # Precompute information parameters
    Lam1 = np.linalg.inv(cov1)
    Lam2 = np.linalg.inv(cov2)
    eta1 = Lam1 @ mean1
    eta2 = Lam2 @ mean2

    if omega is None:
        # Coarse-but-robust 1D search over omega in [0,1]
        best_omega = 0.0
        best_trace = np.inf
        # Try a small grid; 0 and 1 are included
        for w in np.linspace(0.0, 1.0, 51):
            Lam = w * Lam1 + (1.0 - w) * Lam2
            try:
                P = np.linalg.inv(Lam)
            except np.linalg.LinAlgError:
                continue
            tr = np.trace(P)
            if tr < best_trace:
                best_trace = tr
                best_omega = w
        omega = best_omega

    Lam = omega * Lam1 + (1.0 - omega) * Lam2
    P = np.linalg.inv(Lam)
    eta = omega * eta1 + (1.0 - omega) * eta2
    m = P @ eta
    return m, P


@dataclasses.dataclass
class Agent:
    id: str
    state: np.ndarray  # [x, y, psi]
    state_estimated: np.ndarray
    covariance: np.ndarray

    def update_from_measured_state(self, measured_state: np.ndarray, measured_covariance: np.ndarray):

        if USE_CI:
            new_mean, new_cov = ci_fuse(self.state_estimated, self.covariance,
                                        measured_state, measured_covariance,
                                        omega=None)  # pick omega by trace-min
            self.state_estimated = new_mean
            self.covariance = new_cov
        else:
            K = self.covariance @ np.linalg.inv(self.covariance + measured_covariance)
            self.state_estimated = self.state_estimated + K @ (measured_state - self.state_estimated)
            self.covariance = (np.eye(self.covariance.shape[0]) - K) @ self.covariance


@dataclasses.dataclass
class Measurement:
    vector: np.ndarray
    agent_from: Agent
    agent_to: Agent
    covariance: np.ndarray


def get_rotation_matrix(psi: float):
    R = np.array([
        [np.cos(psi), -np.sin(psi), 0, ],
        [np.sin(psi), np.cos(psi), 0, ],
        [0, 0, 1, ],
    ])
    return R


def R_to_local(psi):
    return get_rotation_matrix(-psi)


def R_to_global(psi):
    return get_rotation_matrix(psi)


def calculate_measurement(agent_from: Agent, agent_to: Agent) -> np.ndarray:
    measurement_global = np.array(
        [
            agent_to.state[INDEX_X] - agent_from.state[INDEX_X],
            agent_to.state[INDEX_Y] - agent_from.state[INDEX_Y],
            agent_to.state[INDEX_PSI] - agent_from.state[INDEX_PSI],
        ]
    )
    measurement_local = R_to_local(agent_from.state[INDEX_PSI]) @ measurement_global
    return measurement_local


def get_weird_jacobian():
    ...


def get_measured_state(agent: Agent, measurement: Measurement) -> tuple[np.ndarray, np.ndarray]:
    agent_from = measurement.agent_from
    agent_to = measurement.agent_to

    if agent == agent_from:
        new_state = agent_to.state_estimated - R_to_global(float(agent_from.state_estimated[INDEX_PSI])) @ measurement.vector
        new_covariance = agent_to.covariance + R_to_global(
            agent_from.state_estimated[INDEX_PSI]) @ measurement.covariance @ R_to_global(agent_from.state_estimated[INDEX_PSI]).T

    elif agent == agent_to:
        new_state = agent_from.state_estimated + R_to_global(float(agent_from.state_estimated[INDEX_PSI])) @ measurement.vector

        new_covariance = agent_from.covariance + R_to_global(
            agent_from.state_estimated[INDEX_PSI]) @ measurement.covariance @ R_to_global(agent_from.state_estimated[INDEX_PSI]).T
    else:
        raise ValueError("Invalid measurement")

    return new_state, new_covariance


if __name__ == '__main__':
    agent_a = Agent(
        id="A",
        state=np.array([1, 2, 0]),
        state_estimated=np.array([0, 0, 0]),
        covariance=100 * np.eye(3)
    )

    agent_b = Agent(
        id="B",
        state=np.array([2, 5, 1]),
        state_estimated=np.array([0, 0, 0]),
        covariance=1 * np.eye(3)
    )

    measurement_a_2_b = Measurement(
        agent_from=agent_a,
        agent_to=agent_b,
        vector=calculate_measurement(agent_from=agent_a, agent_to=agent_b),
        covariance=0 * np.eye(3)
    )

    for i in range(50):
        state_meas_a, cov_meas_a = get_measured_state(agent=agent_a, measurement=measurement_a_2_b)
        state_meas_b, cov_meas_b = get_measured_state(agent=agent_b, measurement=measurement_a_2_b)

        agent_a.update_from_measured_state(state_meas_a, cov_meas_a)
        agent_b.update_from_measured_state(state_meas_b, cov_meas_b)

        print(f"--------------- Step {i + 1} ---------------")
        print(
            f"Agent A: {agent_a.state_estimated}. True: {agent_a.state}. Covariance: {np.linalg.norm(agent_a.covariance, "fro"):.1f}")
        print(
            f"Agent B: {agent_b.state_estimated}. True: {agent_b.state} Covariance: {np.linalg.norm(agent_b.covariance, "fro"):.1f}")
