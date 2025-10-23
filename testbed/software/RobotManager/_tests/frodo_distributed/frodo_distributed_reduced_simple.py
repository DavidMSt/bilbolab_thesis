from __future__ import annotations

import dataclasses

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Measurement:
    vector: np.ndarray
    agent_from: Agent
    agent_to: Agent
    covariance: np.ndarray


# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Agent:
    id: str
    state: np.ndarray
    state_estimated: np.ndarray
    covariance: np.ndarray

    def update_from_measured_state(self, measured_state: np.ndarray, measured_covariance: np.ndarray):
        K = self.covariance @ np.linalg.inv(self.covariance + measured_covariance)
        self.state_estimated = self.state_estimated + K @ (measured_state - self.state_estimated)
        self.covariance = (np.eye(2) - K) @ self.covariance


def get_measured_state(agent: Agent, measurement: Measurement) -> tuple[np.ndarray, np.ndarray]:
    if measurement.agent_from.id == agent.id:
        state = measurement.agent_to.state_estimated - measurement.vector
        covariance = measurement.agent_to.covariance + measurement.covariance

    elif measurement.agent_to.id == agent.id:
        state =  measurement.agent_from.state_estimated + measurement.vector
        covariance = measurement.agent_from.covariance + measurement.covariance
    else:
        raise ValueError("Invalid measurement")
    return state, covariance


def calculate_relative_measurement(agent_from: Agent, agent_to: Agent):
    return agent_to.state - agent_from.state

def example_1():
    agent_a = Agent(
        id="a",
        state=np.array([1, 2]),
        state_estimated=np.array([0, 0]),
        covariance=np.array([[10, 0], [0, 10]]),
    )

    agent_b = Agent(
        id="b",
        state=np.array([3,5]),
        state_estimated=np.array([30, 40]),
        covariance=np.array([[100, 0], [0, 100]]),
    )

    measurement_a_2_b = Measurement(
        agent_from=agent_a,
        agent_to=agent_b,
        vector=calculate_relative_measurement(agent_from=agent_a, agent_to=agent_b),
        covariance=np.array([[0.01, 0], [0, 0.01]]),
    )

    for i in range(500):
        state_meas_a, cov_meas_a = get_measured_state(agent=agent_a, measurement=measurement_a_2_b)
        state_meas_b, cov_meas_b = get_measured_state(agent=agent_b, measurement=measurement_a_2_b)

        agent_a.update_from_measured_state(state_meas_a, cov_meas_a)
        agent_b.update_from_measured_state(state_meas_b, cov_meas_b)
        print(f"--------------- Step {i+1} ---------------")
        print(f"Agent A: {agent_a.state_estimated}. True: {agent_a.state}. Covariance: {agent_a.covariance}")
        print(f"Agent B: {agent_b.state_estimated}. True: {agent_b.state} Covariance: {agent_b.covariance}")


if __name__ == '__main__':
    example_1()
