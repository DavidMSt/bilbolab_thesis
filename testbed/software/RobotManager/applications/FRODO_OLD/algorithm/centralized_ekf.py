import dataclasses
import math
import numpy as np
import qmt

from core.utils.logging_utils import Logger

logger = Logger('EKF')
logger.setLevel("INFO")


# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class VisionAgent:
    id: str
    index: int
    state: np.ndarray
    state_covariance: np.ndarray
    input: np.ndarray
    input_covariance: np.ndarray
    measurements: list['VisionAgentMeasurement']


# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class VisionAgentMeasurement:
    source: str
    source_index: int
    target: str
    target_index: int
    measurement: np.ndarray
    measurement_covariance: np.ndarray


# ----------------------------------------------------------------------------------------------------------------------
class CentralizedLocationAlgorithm:
    agents: dict[str, VisionAgent]

    Ts: float
    state: np.ndarray
    state_covariance: np.ndarray

    step: int = 0

    def __init__(self, Ts):
        self.Ts = Ts

    def init(self, agents: dict[str, VisionAgent]):
        self.agents = agents

        # Build the state:
        self.state = np.zeros(len(agents) * 3)
        for i, agent in enumerate(agents.values()):
            self.state[i * 3:(i + 1) * 3] = agent.state

        logger.info(f"State: {self.state}")

        # Build the state covariance
        self.state_covariance = np.zeros((len(agents) * 3, len(agents) * 3))
        for i, agent in enumerate(agents.values()):
            self.state_covariance[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = agent.state_covariance

        logger.info(f"State covariance: {self.state_covariance}")

    # ------------------------------------------------------------------------------------------------------------------
    def update(self):

        # STEP 1: PREDICTION
        x_hat_pre, P_hat_pre = self.prediction()

        # STEP 2: EXTRACT MEASUREMENTS
        measurements = self.getMeasurements()
        measurement_list = [f"{measurement.source}->{measurement.target}" for measurement in measurements]

        if len(measurements) > 0:
            # STEP 3: CALCULATE SPARSE MEASUREMENT JACOBIAN
            H = self.measurementJacobian_sparse(measurements)

            # STEP 4: CALCULATE THE KALMAN GAIN
            W = self.buildMeasurementCovariance_sparse(measurements)
            K = P_hat_pre @ H.T @ np.linalg.inv(H @ P_hat_pre @ H.T + W)

            # STEP 5: BUILD THE MEASUREMENT VECTOR
            y = self.buildMeasurementVector_sparse(measurements)

            # STEP 6: BUILD THE PREDICTED MEASUREMENT VECTOR
            y_est = self.measurementPrediction_sparse(measurements)

            # STEP 7: UPDATE
            diff = y - y_est

            # Wrap all angle values
            for i in range(len(diff)):
                if i % 3 == 2:
                    diff[i] = qmt.wrapToPi(diff[i])
                else:
                    continue

            correction_term = K @ diff
            new_state = x_hat_pre + K @ diff
            new_covariance = (np.eye(len(self.agents) * 3) - K @ H) @ P_hat_pre @ (
                    np.eye(len(self.agents) * 3) - K @ H).T + K @ W @ K.T
        else:
            new_state = x_hat_pre
            new_covariance = P_hat_pre

        # Wrap all angles
        for i in range(len(new_state)):
            if i % 3 == 2:
                new_state[i] = qmt.wrapToPi(new_state[i])
            else:
                continue

        self.state = new_state
        self.state_covariance = new_covariance

        # Write the state back to the agents
        for i in range(len(self.agents)):
            agent = self.getAgentByIndex(i)
            if agent is None:
                raise ValueError(f"Agent with index {i} does not exist.")
            agent.state = self.state[i * 3:(i + 1) * 3]
            agent.state_covariance = self.state_covariance[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3]

        self.step += 1

        if (self.step % 10) == 0 or self.step == 1:
            print("--------------------------------")
            print(f"Step: {self.step}")
            for agent in self.agents.values():
                print(f"{agent.id}: \t x: {agent.state[0]:.1f} \t y: {agent.state[1]:.1f} \t psi: {agent.state[2]:.1f} \t Cov: {np.linalg.norm(agent.state_covariance, 'fro'):.1f}")

    # ------------------------------------------------------------------------------------------------------------------
    def predictionAgent(self, state: np.ndarray, input: np.ndarray):
        """
        Prediction step of one agent
        Args:
            state: State of the agent
            input: Input to the agent

        Returns:

        """
        state_hat = np.array([
            state[0] + self.Ts * input[0] * math.cos(state[2]),
            state[1] + self.Ts * input[0] * math.sin(state[2]),
            state[2] + self.Ts * input[1]
        ])
        return state_hat

    # ------------------------------------------------------------------------------------------------------------------
    def enumerateAgentArray(self):
        """
        Enumerate the agents in the array
        Returns:

        """
        for i, agent in enumerate(self.agents.values()):
            agent.index = i

    # ------------------------------------------------------------------------------------------------------------------
    def getMeasurements(self):
        measurements = []

        for i in range(len(self.agents)):
            agent = self.getAgentByIndex(i)
            if agent is None:
                raise ValueError(f"Agent with index {i} does not exist.")

            if len(agent.measurements) > 0:
                measurements.extend(agent.measurements)

        return measurements

    # ------------------------------------------------------------------------------------------------------------------
    def prediction(self):
        """
        Calculate the prediction of the full system
        Returns:

        """

        # Predict the states
        x_hat = np.zeros(len(self.agents) * 3)
        for i in range(len(self.agents)):
            agent = self.getAgentByIndex(i)
            if agent is None:
                raise ValueError(f"Agent with index {i} does not exist.")
            x_hat[i * 3:(i + 1) * 3] = self.predictionAgent(agent.state, agent.input)

        # Predict the covariance

        # Calculate the dynamics jacobian
        F = self.dynamicsJacobian()

        # dynamics_noise = np.zeros_like(self.state_covariance)  # TODO
        dynamics_noise = np.eye(3 * len(self.agents)) * 1e-10
        P_hat = F @ self.state_covariance @ F.T + dynamics_noise

        return x_hat, P_hat

    # ------------------------------------------------------------------------------------------------------------------
    def jacobianAgent(self, state: np.ndarray, input: np.ndarray):
        """
        Calculate the Jacobian matrix of the agent's motion model
        Args:
            state:
            input:

        Returns:

        """
        F = np.array([
            [1, 0, -self.Ts * input[0] * math.sin(state[2])],
            [0, 1, self.Ts * input[0] * math.cos(state[2])],
            [0, 0, 1]
        ])
        return F

    # ------------------------------------------------------------------------------------------------------------------
    def dynamicsJacobian(self):
        """
        Calculate the Jacobian matrix of the full system
        Returns:

        """
        J = np.zeros((len(self.agents) * 3, len(self.agents) * 3))

        for i in range(len(self.agents)):
            agent = self.getAgentByIndex(i)
            if agent is None:
                raise ValueError(f"Agent with index {i} does not exist.")
            J[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self.jacobianAgent(agent.state, agent.input)

        return J

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def measurementPredictionAgent(agent_source_state, agent_target_state):
        """
        Prediction for one agent for the measurement model
        Args:
            agent_source_state:
            agent_target_state:

        Returns:

        """
        h_source_target = np.array([
            [math.cos(agent_source_state[2]) * (agent_target_state[0] - agent_source_state[0]) + math.sin(
                agent_source_state[2]) * (agent_target_state[1] - agent_source_state[1])],
            [-math.sin(agent_source_state[2]) * (agent_target_state[0] - agent_source_state[0]) + math.cos(
                agent_source_state[2]) * (agent_target_state[1] - agent_source_state[1])],
            [qmt.wrapToPi(agent_target_state[2] - agent_source_state[2])]
        ])  # TODO: Do i need some wrapping here?
        return h_source_target

    # # ------------------------------------------------------------------------------------------------------------------
    # def calculatePredictionCovariance(self, state_covariance, dynamics_jacobian, dynamics_noise_covariance):
    #     return dynamics_jacobian @ state_covariance @ dynamics_jacobian.T + dynamics_noise_covariance

    # ------------------------------------------------------------------------------------------------------------------
    def measurementJacobian_sparse(self, measurements: list[VisionAgentMeasurement]) -> np.ndarray:

        H = np.zeros((3 * len(measurements), 3 * len(self.agents)))

        for i, measurement in enumerate(measurements):
            H_meas = np.zeros((3, len(self.agents) * 3))
            index_source = measurement.source_index
            index_target = measurement.target_index
            H_source = self.measurementJacobianAgents(
                agent_source=self.getAgentByIndex(measurement.source_index),
                agent_target=self.getAgentByIndex(measurement.target_index),
                reference_agent=1
            )
            H_target = self.measurementJacobianAgents(
                agent_source=self.getAgentByIndex(measurement.source_index),
                agent_target=self.getAgentByIndex(measurement.target_index),
                reference_agent=2
            )

            H_meas[:, 3 * index_source:3 * (index_source + 1)] = H_source
            H_meas[:, 3 * index_target:3 * (index_target + 1)] = H_target

            H[3*i:3*(i+1), :] = H_meas

        return H

    # ------------------------------------------------------------------------------------------------------------------
    def buildMeasurementCovariance_sparse(self, measurements: list[VisionAgentMeasurement]) -> np.ndarray:
        W = np.zeros((3 * len(measurements), 3 * len(measurements)))

        for i, measurement in enumerate(measurements):
            W_meas = np.eye(3) * measurement.measurement_covariance
            W[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = W_meas

        return W

    # ------------------------------------------------------------------------------------------------------------------
    def buildMeasurementVector_sparse(self, measurements: list[VisionAgentMeasurement]) -> np.ndarray:
        y = np.zeros(3 * len(measurements))

        for i, measurement in enumerate(measurements):
            y[i * 3:(i + 1) * 3] = measurement.measurement

        return y

    # ------------------------------------------------------------------------------------------------------------------
    def measurementPrediction_sparse(self, measurements: list[VisionAgentMeasurement]) -> np.ndarray:
        y_est = np.zeros(3 * len(measurements))

        for i, measurement in enumerate(measurements):
            agent_source = self.getAgentByIndex(measurement.source_index)
            agent_target = self.getAgentByIndex(measurement.target_index)

            predicted_measurement = self.measurementPredictionAgent(agent_source_state=agent_source.state,
                                                                    agent_target_state=agent_target.state)

            y_est[i * 3:(i + 1) * 3] = predicted_measurement.flatten()

        return y_est
    # ------------------------------------------------------------------------------------------------------------------
    def measurementJacobianAgents(self, agent_source, agent_target, reference_agent):
        assert (reference_agent in [1, 2])

        if reference_agent == 1:
            psi_1 = agent_source.state[2]
            x_1 = agent_source.state[0]
            y_1 = agent_source.state[1]
            x_2 = agent_target.state[0]
            y_2 = agent_target.state[1]
            H = np.array([
                [-np.cos(psi_1), -np.sin(psi_1), -np.sin(psi_1) * (x_2 - x_1) + np.cos(psi_1) * (y_2 - y_1)],
                [np.sin(psi_1), -np.cos(psi_1), -np.cos(psi_1) * (x_2 - x_1) - np.sin(psi_1) * (y_2 - y_1)],
                [0, 0, -1]
            ])

            pass

        elif reference_agent == 2:
            psi_1 = agent_source.state[2]
            H = np.array([
                [np.cos(psi_1), np.sin(psi_1), 0, ],
                [-np.sin(psi_1), np.cos(psi_1), 0, ],
                [0, 0, 1]
            ])
        else:
            return None
        return H

    # ------------------------------------------------------------------------------------------------------------------
    def getAgentByIndex(self, index: int) -> (VisionAgent, None):
        for agent in self.agents.values():
            if agent.index == index:
                return agent
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def getAgentIndex(self, id: str) -> (int, None):
        for i, agent in enumerate(self.agents.values()):
            if agent.id == id:
                return i
