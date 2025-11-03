import math
import numpy as np
import dataclasses

# ------------------------------------------------------------------------------
# Dataclass Definitions
# ------------------------------------------------------------------------------
@dataclasses.dataclass
class VisionAgent:
    id: str
    index: int
    state: np.ndarray               # [x, y, psi]
    state_covariance: np.ndarray    # 3x3 covariance matrix
    input: np.ndarray               # [v, omega]
    input_covariance: np.ndarray    # 2x2 covariance matrix (not used in IEKF here)
    measurements: list = None       # List of VisionAgentMeasurement

    def __post_init__(self):
        if self.measurements is None:
            self.measurements = []


@dataclasses.dataclass
class VisionAgentMeasurement:
    source: str
    source_index: int
    target: str
    target_index: int
    measurement: np.ndarray         # relative transformation [x, y, psi] (in the source frame)
    measurement_covariance: np.ndarray  # 3x3 covariance matrix


# ------------------------------------------------------------------------------
# SE(2) Helper Functions
# ------------------------------------------------------------------------------
def compose_SE2(a, b):
    """
    Compose two SE(2) elements.
    Both a and b are [x, y, psi].
    """
    x1, y1, psi1 = a
    x2, y2, psi2 = b
    x = x1 + math.cos(psi1) * x2 - math.sin(psi1) * y2
    y = y1 + math.sin(psi1) * x2 + math.cos(psi1) * y2
    psi = psi1 + psi2  # You may want to wrap psi later if needed.
    return np.array([x, y, psi])


def inv_SE2(a):
    """
    Inverse of an SE(2) element a = [x, y, psi].
    """
    x, y, psi = a
    c = math.cos(psi)
    s = math.sin(psi)
    x_inv = -c * x - s * y
    y_inv =  s * x - c * y
    return np.array([x_inv, y_inv, -psi])


def exp_se2(xi):
    """
    Exponential map from se(2) to SE(2).
    xi is a 3-vector [v_x, v_y, omega].
    """
    vx, vy, w = xi
    if abs(w) < 1e-6:
        # For very small rotation, use a first-order approximation.
        return np.array([vx, vy, w])
    else:
        A = math.sin(w) / w
        B = (1 - math.cos(w)) / w
        x = A * vx - B * vy
        y = B * vx + A * vy
        return np.array([x, y, w])


def log_se2(a):
    """
    Logarithm map from SE(2) to se(2).
    This simple mapping assumes a direct correspondence; for more robust
    implementations consider handling larger rotations carefully.
    """
    x, y, psi = a
    return np.array([x, y, psi])


def Ad_SE2(a):
    """
    Compute the adjoint of an SE(2) element a = [x, y, psi].
    Returns a 3x3 matrix.
    """
    x, y, psi = a
    c = math.cos(psi)
    s = math.sin(psi)
    return np.array([
        [c, -s, y],
        [s,  c, -x],
        [0,  0,  1]
    ])


# ------------------------------------------------------------------------------
# Invariant EKF Implementation
# ------------------------------------------------------------------------------
class CentralizedLocationAlgorithm:
    def __init__(self, Ts):
        self.Ts = Ts
        self.step = 0

    def init(self, agents: dict):
        """
        Initialize with a dictionary of agents.
        agents: dict where key is agent id and value is a VisionAgent instance.
        """
        self.agents = agents
        self.num_agents = len(agents)
        # Build the state vector (each agent: [x, y, psi])
        self.state = np.zeros(self.num_agents * 3)
        for i, agent in enumerate(agents.values()):
            self.state[i * 3:(i + 1) * 3] = agent.state

        # Build the block-diagonal covariance matrix
        self.state_covariance = np.zeros((self.num_agents * 3, self.num_agents * 3))
        for i, agent in enumerate(agents.values()):
            self.state_covariance[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = agent.state_covariance

    def prediction(self):
        """
        IEKF prediction step.
        For each agent, the update is:
            x⁺ = x ⊕ exp(δ)
        where δ = [Ts * v, 0, Ts * ω].
        The covariance is propagated using the adjoint of the inverse increment.
        """
        x_hat = np.zeros_like(self.state)
        F_full = np.zeros((self.num_agents * 3, self.num_agents * 3))
        Q = np.eye(self.num_agents * 3) * 1e-10  # Process noise (tunable)

        for i, agent in enumerate(self.agents.values()):
            idx = slice(i * 3, (i + 1) * 3)
            # Control input: [v, ω]
            u = agent.input
            # Compute the increment in the Lie algebra: [Ts*v, 0, Ts*ω]
            delta = np.array([self.Ts * u[0], 0, self.Ts * u[1]])
            increment = exp_se2(delta)
            # Propagate the state by group composition
            x_hat[idx] = compose_SE2(agent.state, increment)
            # Covariance propagation: F = Ad(exp(-δ))
            T_delta_inv = exp_se2(-delta)
            F_i = Ad_SE2(T_delta_inv)
            F_full[idx, idx] = F_i

        P_hat = F_full @ self.state_covariance @ F_full.T + Q
        return x_hat, P_hat

    def getMeasurements(self):
        """
        Extract relative measurements.
        Each measurement is assumed to have:
          - source_index: index of the measuring agent
          - target_index: index of the observed agent
          - measurement: relative transformation [x, y, psi] from source to target (in the source frame)
          - measurement_covariance: 3x3 covariance matrix
        """
        measurements = []
        for agent in self.agents.values():
            for meas in agent.measurements:
                measurements.append(meas)
        return measurements

    def update(self):
        """
        IEKF measurement update.
        For each relative measurement from agent i to j:
          1. Compute the predicted relative transformation:
               z_pred = inv(x_i) ⊕ x_j
          2. Compute the invariant innovation:
               η = log( inv(z) ⊕ z_pred )
          3. Form the Jacobians:
               H_i = -Ad(inv(z_pred))
               H_j =  Ad(inv(z_pred))
             and build the full measurement Jacobian.
          4. Perform the standard EKF update on the error state and correct via group composition.
        """
        x_hat, P_hat = self.prediction()
        measurements = self.getMeasurements()
        if len(measurements) == 0:
            self.state = x_hat
            self.state_covariance = P_hat
            return

        H_list = []
        innovation_list = []
        R_list = []

        for meas in measurements:
            # For a measurement from agent i (source) to agent j (target)
            i = meas.source_index
            j = meas.target_index
            idx_i = slice(i * 3, (i + 1) * 3)
            idx_j = slice(j * 3, (j + 1) * 3)
            # Get predicted states of agents i and j
            x_i = x_hat[idx_i]
            x_j = x_hat[idx_j]
            # Predicted relative measurement:
            z_pred = compose_SE2(inv_SE2(x_i), x_j)
            # Innovation computed on the Lie algebra:
            # η = log( inv(z) ⊕ z_pred )
            innovation = log_se2(compose_SE2(inv_SE2(meas.measurement), z_pred))
            innovation_list.append(innovation)

            # Jacobians for the relative measurement:
            H_i = -Ad_SE2(inv_SE2(z_pred))
            H_j =  Ad_SE2(inv_SE2(z_pred))
            H = np.zeros((3, self.num_agents * 3))
            H[:, idx_i] = H_i
            H[:, idx_j] = H_j
            H_list.append(H)

            R_list.append(meas.measurement_covariance)

        # Stack the measurements (each measurement yields a 3D innovation)
        H_full = np.vstack(H_list)
        innovation_full = np.hstack(innovation_list)
        # Build the block-diagonal measurement covariance matrix
        R_full = np.block([
            [R_list[i] if i == j else np.zeros((3, 3)) for j in range(len(R_list))]
            for i in range(len(R_list))
        ])

        # Standard EKF gain calculation
        S = H_full @ P_hat @ H_full.T + R_full
        K = P_hat @ H_full.T @ np.linalg.inv(S)
        delta_x = K @ innovation_full

        # Correct each agent's state via the group update: x_new = x_hat ⊕ exp(δ)
        x_new = x_hat.copy()
        for i in range(self.num_agents):
            idx = slice(i * 3, (i + 1) * 3)
            delta_i = delta_x[idx]
            x_new[idx] = compose_SE2(x_hat[idx], exp_se2(delta_i))
        P_new = (np.eye(self.num_agents * 3) - K @ H_full) @ P_hat

        self.state = x_new
        self.state_covariance = P_new

        # Write updated states and covariances back to the agents
        for i, agent in enumerate(self.agents.values()):
            idx = slice(i * 3, (i + 1) * 3)
            agent.state = self.state[idx]
            agent.state_covariance = self.state_covariance[idx, idx]

        self.step += 1

        if (self.step % 10) == 0 or self.step == 1:
            print("--------------------------------")
            print(f"Step: {self.step}")
            for agent in self.agents.values():
                print(
                    f"{agent.group_id}: \t x: {agent.state[0]:.1f} \t y: {agent.state[1]:.1f} \t psi: {agent.state[2]:.1f} \t Cov: {np.linalg.norm(agent.state_covariance, 'fro'):.1f}")

    # ------------------------------------------------------------------------------------------------------------------
    def getAgentByIndex(self, index: int) -> (VisionAgent):
        for agent in self.agents.values():
            if agent.index == index:
                return agent
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def getAgentIndex(self, id: str) -> (int, None):
        for i, agent in enumerate(self.agents.values()):
            if agent.group_id == id:
                return i
# # ------------------------------------------------------------------------------
# # Example Usage (Can be removed or adapted)
# # ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Create two dummy agents for demonstration
    agent1 = VisionAgent(
        id="agent1",
        index=0,
        state=np.array([0.0, 0.0, 0.0]),
        state_covariance=np.eye(3) * 0.1,
        input=np.array([1.0, 0.1]),
        input_covariance=np.eye(2) * 0.01,
        measurements=[]
    )

    agent2 = VisionAgent(
        id="agent2",
        index=1,
        state=np.array([1.0, 0.0, 0.0]),
        state_covariance=np.eye(3) * 0.1,
        input=np.array([1.0, 0.1]),
        input_covariance=np.eye(2) * 0.01,
        measurements=[]
    )

    # Add a dummy relative measurement from agent1 to agent2
    relative_measurement = VisionAgentMeasurement(
        source="agent1",
        source_index=0,
        target="agent2",
        target_index=1,
        measurement=np.array([1.0, 0.0, 0.0]),  # relative transformation (in agent1 frame)
        measurement_covariance=np.eye(3) * 0.05
    )
    agent1.measurements.append(relative_measurement)

    agents = {"agent1": agent1, "agent2": agent2}

    # Initialize and run the IEKF
    Ts = 0.1
    iekf = CentralizedLocationAlgorithm(Ts)
    iekf.init(agents)

    # Run several update iterations and print the results
    for i in range(100):
        iekf.update()
        # print(f"Iteration {i+1}")
        # for agent in agents.values():
        #     print(f"{agent.id}: state = {agent.state}, covariance Frobenius norm = {np.linalg.norm(agent.state_covariance, 'fro'):.4f}")
