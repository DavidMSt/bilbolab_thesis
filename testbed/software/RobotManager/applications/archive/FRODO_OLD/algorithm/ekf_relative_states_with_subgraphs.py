import dataclasses
import math
import numpy as np

from core.utils.logging_utils import Logger
from core.utils.graphs import connected_subgraphs, plot_graph_from_adjacency_matrix

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
    is_graph_root: bool

    graph_root_index: int
    graph_root: 'VisionAgent'

    input: np.ndarray
    input_covariance: np.ndarray
    measurements: list['VisionAgentMeasurement']

    @property
    def absoluteCovariance(self):
        if self.is_graph_root:
            return self.state_covariance
        else:
            return self.state_covariance + self.graph_root.state_covariance


# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class VisionAgentMeasurement:
    source: str
    source_index: int
    target: str
    target_index: int
    measurement: np.ndarray
    measurement_covariance: np.ndarray


@dataclasses.dataclass
class EKF_Graph:
    root: int
    members: list[int]


# ----------------------------------------------------------------------------------------------------------------------
class CentralizedLocationAlgorithm:
    agents: dict[str, VisionAgent]

    graphs: list[EKF_Graph]

    Ts: float
    state: np.ndarray
    state_covariance: np.ndarray
    step: int = 0

    def __init__(self, Ts):
        self.Ts = Ts

        self.graphs = []

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
        measurement_list = [
            f"{measurement['measurement'].source}->{measurement['measurement'].target}({measurement['type']})" for
            measurement in measurements]

        if len(measurements) == 0:
            raise NotImplementedError
            # new_state = x_hat_pre
            # new_covariance = P_hat_pre

        # STEP 3: Build Adjacency Matrix
        adjacency_matrix = self.getAdjacencyMatrix(measurements)

        graphs = connected_subgraphs(adjacency_matrix)

        graphs = self.buildMeasurementGraphs(graphs, adjacency_matrix)

        return

    # ------------------------------------------------------------------------------------------------------------------
    def getAdjacencyMatrix(self, measurements: list[dict]):

        adjacency_matrix = np.zeros((len(self.agents), len(self.agents)))
        for i, measurement in enumerate(measurements):
            index_source = measurement['measurement'].source_index
            index_target = measurement['measurement'].target_index
            adjacency_matrix[index_target, index_source] = 1

        return adjacency_matrix

    # ------------------------------------------------------------------------------------------------------------------
    def buildMeasurementGraphs(self, new_subgraphs, adjacency_matrix: np.ndarray = None):
        """
        Build/update the EKF measurement graphs from the connected subgraphs.

        new_subgraphs: list of lists of agent indices (each list is one connected subgraph)
        """
        # If no graphs exist, initialize them.
        if not self.graphs:
            logger.info("No existing graphs found. Initializing graphs from connected subgraphs.")
            self.graphs = []
            for subgraph in new_subgraphs:
                # Select the root: the agent with the lowest Frobenius norm of its state covariance.
                root_index = min(subgraph,
                                 key=lambda idx: np.linalg.norm(self.getAgentByIndex(idx).state_covariance, 'fro'))
                ekf_graph = EKF_Graph(root=root_index, members=subgraph)
                self.graphs.append(ekf_graph)
                # Update each agent in the subgraph.
                for idx in subgraph:
                    agent = self.getAgentByIndex(idx)
                    if idx == root_index:
                        agent.is_graph_root = True
                        agent.graph_root_index = root_index
                        agent.graph_root = agent
                    else:
                        agent.is_graph_root = False
                        agent.graph_root_index = root_index
                        agent.graph_root = self.getAgentByIndex(root_index)
                logger.info(f"Initialized new graph: Root Agent {root_index}, Members {subgraph}")
            plot_graph_from_adjacency_matrix(adjacency_matrix)
            return

        # Graphs already exist; now we need to check for changes.
        # Build new EKF_Graph objects from the current connected subgraphs.
        new_graphs = []
        for subgraph in new_subgraphs:
            new_root = min(subgraph, key=lambda idx: np.linalg.norm(self.getAgentByIndex(idx).state_covariance, 'fro'))
            new_graphs.append(EKF_Graph(root=new_root, members=subgraph))

        graphs_changed = False

        updated_graphs = []
        # Compare each new graph with the current ones.
        for new_graph in new_graphs:
            new_set = set(new_graph.members)
            matched_graph = None
            for existing_graph in self.graphs:
                existing_set = set(existing_graph.members)
                if new_set == existing_set:
                    # Graph unchanged.
                    matched_graph = existing_graph
                    break
                elif existing_set.issubset(new_set):
                    # Graph expanded: additional agents joined.
                    added = new_set - existing_set
                    logger.info(
                        f"Graph expanded: Previous members {list(existing_set)} expanded with added agents {list(added)}")
                    matched_graph = existing_graph
                    graphs_changed = True
                    break
                elif new_set.issubset(existing_set):
                    # Graph contracted: some agents left.
                    removed = existing_set - new_set
                    logger.info(
                        f"Graph contracted: Previous members {list(existing_set)} contracted with removed agents {list(removed)}")
                    matched_graph = existing_graph
                    graphs_changed = True
                    break

            if matched_graph:
                # If the graph root has changed, print a debug message.
                if matched_graph.root != new_graph.root:
                    logger.info(
                        f"Graph root changed: from Agent {matched_graph.root} to Agent {new_graph.root}. Agents switching root require covariance recalculation.")
                # Update the existing graph with the new member set and new root.
                matched_graph.root = new_graph.root
                matched_graph.members = new_graph.members
                updated_graphs.append(matched_graph)
            else:
                # This is a completely new graph.
                logger.info(f"New graph formed: Root Agent {new_graph.root} with members {new_graph.members}")
                graphs_changed = True
                updated_graphs.append(new_graph)

            # Update all agents in the new graph with the correct graph root information.
            for idx in new_graph.members:
                agent = self.getAgentByIndex(idx)
                if idx == new_graph.root:
                    agent.is_graph_root = True
                    agent.graph_root_index = new_graph.root
                    agent.graph_root = agent
                else:
                    if agent.graph_root_index != new_graph.root:
                        logger.info(
                            f"Agent {agent.id} switching graph root from {agent.graph_root_index} to {new_graph.root}. Covariance transformation update required.")
                    agent.is_graph_root = False
                    agent.graph_root_index = new_graph.root
                    agent.graph_root = self.getAgentByIndex(new_graph.root)

        # Check for any graphs that have completely disappeared.
        for existing_graph in self.graphs:
            if set(existing_graph.members) not in [set(g.members) for g in updated_graphs]:
                logger.info(f"Graph with members {existing_graph.members} no longer exists and is removed.")

        self.graphs = updated_graphs

        if graphs_changed:
            plot_graph_from_adjacency_matrix(adjacency_matrix)

        return self.graphs

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
                H_target[0:3, 3 * measurement.target_index:3 * (measurement.target_index + 1)] = np.eye(3)

                measurements.append({
                    'measurement': measurement,
                    'type': 'target',
                    'estimated_state': estimated_state_target,
                    'covariance': estimated_state_target_covariance,
                    'H': H_target,
                })

                # ----
                estimated_state_source = agent_to.state - R(
                    measurement.measurement[2] - agent_to.state[2]) @ measurement.measurement
                estimated_state_source_covariance = agent_to.state_covariance + measurement.measurement_covariance  # TODO: Calculate the real covariance

                H_source = np.zeros((3, 3 * len(self.agents)))
                H_source[0:3, 3 * measurement.source_index:3 * (measurement.source_index + 1)] = np.eye(3)
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
