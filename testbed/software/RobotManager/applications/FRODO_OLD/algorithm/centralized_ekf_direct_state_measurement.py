import dataclasses
import math
import numpy as np
import qmt

from core.utils.logging_utils import Logger

logger = Logger('EKF')
logger.setLevel("INFO")


def R(psi):
    return np.array([
        [math.cos(psi), math.sin(psi), 0],
        [-math.sin(psi), math.cos(psi), 0],
        [0, 0, 1]
    ])

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
        x_hat_pre, P_hat_pre, F = self.prediction()

        # STEP 2: EXTRACT MEASUREMENTS
        measurements = self.getMeasurements()
        measurement_list = [f"{measurement['measurement'].source}->{measurement['measurement'].target}({measurement['type']})" for measurement in measurements]

        if len(measurements) > 0:
            # STEP 3: CALCULATE SPARSE MEASUREMENT JACOBIAN
            H = self.measurementJacobian(measurements)


            # STEP 4: CALCULATE THE KALMAN GAIN
            W = self.buildMeasurementCovariance(measurements)
            K = P_hat_pre @ H.T @ np.linalg.inv(H @ P_hat_pre @ H.T + W)

            V_unobs = self.debug_observability(F, H, steps=10, threshold=1e-5, state_names=['x1', 'y1', 'psi1',
                                                                                  'x2', 'y2', 'psi2',
                                                                                  'x3', 'y3', 'psi3',
                                                                                  'x4', 'y4', 'psi4',])

            # STEP 5: BUILD THE MEASUREMENT VECTOR
            y = self.buildMeasurementVector(measurements)

            # STEP 6: BUILD THE PREDICTED MEASUREMENT VECTOR
            y_est = self.measurementPrediction(measurements)

            pass

            # STEP 7: UPDATE
            diff = y - y_est

            # Wrap all angle values
            for i in range(len(diff)):
                if i % 3 == 2:
                    diff[i] = qmt.wrapToPi(diff[i])
                else:
                    continue

            new_state, new_covariance = self.constrained_ekf_update(x_hat_pre, P_hat_pre, y, y_est, H, W, V_unobs)
            # correction_term = K @ diff
            # new_state = x_hat_pre + K @ diff
            # new_covariance = (np.eye(len(self.agents) * 3) - K @ H) @ P_hat_pre @ (
            #         np.eye(len(self.agents) * 3) - K @ H).T + K @ W @ K.T



            pass
        else:
            new_state = x_hat_pre
            new_covariance = P_hat_pre

        # # Wrap all angles
        # for i in range(len(new_state)):
        #     if i % 3 == 2:
        #         new_state[i] = qmt.wrapToPi(new_state[i])
        #     else:
        #         continue
        #
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

    @staticmethod
    def constrained_ekf_update(x_hat_pre, P_hat_pre, y, y_est, H, W, V_unobs):
        """
        Perform an EKF update that updates only the observable subspace,
        preserving high covariance in the unobservable directions.

        Args:
            x_hat_pre (np.ndarray): The predicted state vector.
            P_hat_pre (np.ndarray): The predicted covariance matrix.
            y (np.ndarray): The measurement vector.
            y_est (np.ndarray): The predicted measurement vector.
            H (np.ndarray): The measurement Jacobian.
            W (np.ndarray): The measurement noise covariance.
            V_unobs (np.ndarray): A matrix whose columns form a basis for the
                                  unobservable subspace (from the SVD of the observability matrix).

        Returns:
            x_new (np.ndarray): The updated state vector.
            P_new (np.ndarray): The updated covariance matrix.
        """
        # Standard Kalman gain
        K = P_hat_pre @ H.T @ np.linalg.inv(H @ P_hat_pre @ H.T + W)

        # Compute the innovation
        innovation = y - y_est

        # Full EKF correction
        delta_x = K @ innovation

        # Compute the projection matrix onto the unobservable subspace
        P_unobs_proj = V_unobs.T @ V_unobs

        # Remove the component of the update in the unobservable directions:
        # This leaves only the observable component of the correction.
        delta_x_obs = delta_x - P_unobs_proj @ delta_x

        # Update the state using only the observable correction
        x_new = x_hat_pre + delta_x_obs

        # Standard covariance update (as if we updated everything)
        I = np.eye(P_hat_pre.shape[0])
        P_new_standard = (I - K @ H) @ P_hat_pre @ (I - K @ H).T + K @ W @ K.T

        # Now, we want to ensure that the covariance along the unobservable directions is not reduced.
        # One simple way is to "restore" the prior uncertainty in those directions.
        # We extract the unobservable portion from the predicted covariance:
        P_unobs_prior = P_hat_pre @ P_unobs_proj @ P_hat_pre.T

        # And we extract the observable portion from the updated covariance:
        P_obs = P_new_standard - P_unobs_proj @ P_new_standard @ P_unobs_proj

        # Finally, combine them so that the unobservable covariance remains as before:
        P_new = P_obs + P_unobs_prior

        return x_new, P_new

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

            for measurement in agent.measurements:

                agent_from = self.getAgentByIndex(measurement.source_index)
                agent_to = self.getAgentByIndex(measurement.target_index)

                # ---
                estimated_state_target = agent_from.state + R(-agent_from.state[2]) @ measurement.measurement
                estimated_state_target_covariance = agent_from.state_covariance + measurement.measurement_covariance  # TODO: Calculate the real covariance

                H_target = np.zeros((3, 3 * len(self.agents)))
                H_target[0:3, 3*measurement.target_index:3*(measurement.target_index+1)] = np.eye(3)

                measurements.append({
                    'measurement': measurement,
                    'type': 'target',
                    'estimated_state': estimated_state_target,
                    'covariance': estimated_state_target_covariance,
                    'H': H_target,
                })

                # ----
                estimated_state_source = agent_to.state - R(measurement.measurement[2]-agent_to.state[2]) @ measurement.measurement
                estimated_state_source_covariance = agent_to.state_covariance + measurement.measurement_covariance # TODO: Calculate the real covariance

                H_source = np.zeros((3, 3 * len(self.agents)))
                H_source[0:3, 3*measurement.source_index:3*(measurement.source_index+1)] = np.eye(3)
                measurements.append({
                    'measurement': measurement,
                    'type': 'source',
                    'estimated_state': estimated_state_source,
                    'covariance': estimated_state_source_covariance,
                    'H': H_source,
                })


        return measurements

    # ------------------------------------------------------------------------------------------------------------------
    def prediction(self):
        """
        Calculate the prediction of the full system
        Returns:

        """

        # Predict the states
        x_hat = np.zeros(len(self.agents) * 3) * 1e-5
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

        return x_hat, P_hat, F

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
    def measurementJacobian(self, measurements: list[dict]):

        H = np.zeros((3 * len(measurements), 3 * len(self.agents)))

        for i, measurement in enumerate(measurements):
            H[i*3:(i+1)*3, :] = measurement['H']

        return H


    # ------------------------------------------------------------------------------------------------------------------
    def measurementPrediction(self, measurements: list[dict]):
        y_est = np.zeros(3 * len(measurements))

        for i, measurement in enumerate(measurements):
            agent_source = self.getAgentByIndex(measurement['measurement'].source_index)
            agent_target = self.getAgentByIndex(measurement['measurement'].target_index)

            predicted_measurement = agent_source.state if measurement['type'] == 'source' else agent_target.state

            y_est[i * 3:(i + 1) * 3] = predicted_measurement.flatten()

        return y_est


    # ------------------------------------------------------------------------------------------------------------------
    def buildMeasurementVector(self, measurements: list[dict]):
        y = np.zeros(3 * len(measurements))

        for i, measurement in enumerate(measurements):
            y[i * 3:(i + 1) * 3] = measurement['estimated_state']
        return y

    def buildMeasurementCovariance(self, measurements: list[dict]):

        W = np.zeros((3 * len(measurements), 3 * len(measurements)))

        for i, measurement in enumerate(measurements):
            W_meas = np.eye(3) * measurement['covariance']
            W[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = W_meas

        return W


    # ------------------------------------------------------------------------------------------------------------------
    def getAgentByIndex(self, index: int) -> (VisionAgent):
        for agent in self.agents.values():
            if agent.index == index:
                return agent
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def getAgentIndex(self, id: str) -> (int, None):
        for i, agent in enumerate(self.agents.values()):
            if agent.id == id:
                return i

    @staticmethod
    def debug_observability(F, H, steps=10, threshold=1e-5, state_names=None):
        """
        Debugs the observability of the linearized system given F and H.

        Args:
            F (np.ndarray): The dynamics (state transition) Jacobian (n x n).
            H (np.ndarray): The measurement Jacobian (m x n).
            steps (int): Number of time steps (power iterations) to build the observability matrix.
            threshold (float): Threshold below which a singular value is considered zero.

        This function computes the observability matrix:

            O = [ H; H*F; H*F^2; ...; H*F^(steps-1) ]

        and then uses SVD to check its singular values. If the rank is less than n,
        it prints the basis of the unobservable subspace.
        """

        debug = False

        n = F.shape[0]
        O = H.copy()
        F_power = np.eye(n)

        for i in range(1, steps):
            F_power = F_power @ F  # Compute F^i
            O = np.vstack((O, H @ F_power))

        # Compute the SVD of the observability matrix
        U, S, Vh = np.linalg.svd(O)
        rank = np.sum(S > threshold)

        if debug:
            print("Observability matrix singular values:")
            print(S)
            print(f"Rank: {rank} out of {n}")

        unobservable_basis = Vh[rank:]

        if debug:
            if rank < n:
                if debug:
                    print(f"Warning: System is unobservable in {n - rank} direction(s).")
                    print("Basis for the unobservable subspace (each column is a basis vector):")
                    print(Vh[rank:].T)

                if state_names is not None:
                    for idx, vec in enumerate(unobservable_basis):
                        print(f"\nUnobservable subspace basis vector {idx + 1}:")
                        for name, weight in zip(state_names, vec):
                            print(f"  {name}: {weight:.3f}")
            else:
                print("System is fully observable.")

        return unobservable_basis

    @staticmethod
    def analyze_unobservable_states(Vh, rank, state_names):
        """
        Analyzes the unobservable states based on the SVD result of the observability matrix.

        Args:
            Vh (np.ndarray): The matrix of right singular vectors from the SVD (each row is a singular vector).
            rank (int): Effective rank of the observability matrix (number of singular values above the threshold).
            state_names (list[str]): Names corresponding to each state element.

        This function prints the contribution of each state variable in each unobservable subspace basis vector.
        """
        # The unobservable basis vectors are the rows of Vh starting from 'rank' to the end.
        unobservable_basis = Vh[rank:]

        for idx, vec in enumerate(unobservable_basis):
            print(f"\nUnobservable subspace basis vector {idx + 1}:")
            for name, weight in zip(state_names, vec):
                print(f"  {name}: {weight:.3f}")
