import dataclasses
import threading

import numpy as np

from applications.FRODO.algorithm.algorithm_centralized import CentralizedAgent
from applications.FRODO.algorithm.algorithm_distributed import DistributedAgent
from applications.FRODO.navigation.multi_agent_navigator import MultiAgentNavigator
from applications.FRODO.navigation.navigator import NavigatedObject
from applications.FRODO.navigation.utilities import FRODO_Real_NavigatedObject, FRODO_Sim_NavigatedObject
from applications.FRODO.testbed_manager import TestbedObject_FRODO, TestbedObject_STATIC, FRODO_TestbedManager, \
    TestbedObject
from applications.FRODO.simulation.frodo_simulation import FRODO_VisionAgent, FRODO_Static, FRODO_Simulation, \
    FRODO_VisionAgent_Config
from applications.FRODO.tracker.frodo_tracker import FRODO_Tracker
from core.utils.events import Event, event_definition, EventFlag
from core.utils.logging_utils import Logger
from core.utils.states import State
from robots.frodo.frodo import FRODO
from robots.frodo.frodo_manager import FRODO_Manager

"""
The agent manager manages real and simulated agents and statics as well as their interactions and prepares the measurements
for the algorithm. It also calculates the simulated measurements between real and virtual agents
"""


@dataclasses.dataclass
class AgentMeasurement:
    agent_from: str
    agent_to: str
    measurement: np.ndarray
    covariance: np.ndarray


@dataclasses.dataclass
class AgentState(State):
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    v: float = 0.0
    psi_dot: float = 0.0


@dataclasses.dataclass
class AgentContainer:
    id: str
    state: AgentState = dataclasses.field(init=False)
    measurements: list[AgentMeasurement] = dataclasses.field(default_factory=list)
    navigator_object: NavigatedObject | None = None


@dataclasses.dataclass(kw_only=True)
class RealAgentContainer(AgentContainer):
    testbed_object: TestbedObject_FRODO
    navigator_object: FRODO_Real_NavigatedObject | None = None

    @property
    def state(self) -> AgentState:
        return AgentState(
            x=self.testbed_object.dynamic_state.x,
            y=self.testbed_object.dynamic_state.y,
            psi=self.testbed_object.dynamic_state.psi,
            v=self.testbed_object.dynamic_state.v,
            psi_dot=self.testbed_object.dynamic_state.psi_dot,
        )


@dataclasses.dataclass(kw_only=True)
class SimulatedAgentContainer(AgentContainer):
    agent: FRODO_VisionAgent
    navigator_object: FRODO_Sim_NavigatedObject | None = None

    @property
    def state(self) -> AgentState:
        return AgentState(
            x=self.agent.state.x,
            y=self.agent.state.y,
            psi=self.agent.state.psi,
            v=self.agent.state.v,
            psi_dot=self.agent.state.psi_dot,
        )


@dataclasses.dataclass
class StaticContainer:
    id: str
    state: AgentState = dataclasses.field(init=False)


@dataclasses.dataclass(kw_only=True)
class RealStaticContainer(StaticContainer):
    testbed_object: TestbedObject_STATIC

    @property
    def state(self) -> AgentState:
        return AgentState(
            x=self.testbed_object.state.x,
            y=self.testbed_object.state.y,
            psi=self.testbed_object.state.psi,
            v=0.0,
            psi_dot=0.0,
        )


@dataclasses.dataclass(kw_only=True)
class SimulatedStaticContainer(StaticContainer):
    static: FRODO_Static

    @property
    def state(self) -> AgentState:
        return AgentState(
            x=self.static.state.x,
            y=self.static.state.y,
            psi=self.static.state.psi,
            v=0.0,
            psi_dot=0.0,
        )


@event_definition
class FRODO_AgentManager_Events:
    initialized: Event
    update: Event
    error: Event  # Error coming from one of the real agents or the testbed manager
    new_agent: Event = Event(copy_data_on_set=False, flags=[EventFlag('id', str), EventFlag('type', str)])
    new_static: Event = Event(copy_data_on_set=False, flags=[EventFlag('id', str), EventFlag('type', str)])
    removed_static: Event = Event(copy_data_on_set=False, flags=[EventFlag('id', str), EventFlag('type', str)])
    removed_agent: Event = Event(copy_data_on_set=False, flags=[EventFlag('id', str), EventFlag('type', str)])


# ======================================================================================================================
class FRODO_AgentManager:
    agents: dict[str, AgentContainer]
    statics: dict[str, StaticContainer]
    events: FRODO_AgentManager_Events

    simulation: FRODO_Simulation
    testbed_manager: FRODO_TestbedManager
    navigator: MultiAgentNavigator

    _agent_lock: threading.Lock

    # === INIT =========================================================================================================
    def __init__(self,
                 simulation: FRODO_Simulation,
                 testbed_manager: FRODO_TestbedManager,
                 ):

        self.simulation = simulation
        self.testbed_manager = testbed_manager

        self.testbed_manager.events.new_object.on(self._on_new_testbed_object)
        self.testbed_manager.events.object_removed.on(self._on_testbed_object_removed)

        self.navigator = MultiAgentNavigator()

        self.agents = {}
        self.statics = {}

        self.simulation.events.new_agent.on(self._on_new_simulation_agent)
        self.simulation.events.removed_agent.on(self._on_agent_removed_simulation)
        self.simulation.events.new_static.on(self._on_new_simulated_static)
        self.simulation.events.removed_static.on(self._on_removed_simulated_static)

        self.events = FRODO_AgentManager_Events()
        self.logger = Logger('AgentManager', 'DEBUG')
        self._agent_lock = threading.Lock()

    # === PROPERTIES ===================================================================================================
    @property
    def real_agents(self) -> dict[str, RealAgentContainer]:
        return {k: v for k, v in self.agents.items() if isinstance(v, RealAgentContainer)}

    @property
    def simulated_agents(self) -> dict[str, SimulatedAgentContainer]:
        return {k: v for k, v in self.agents.items() if isinstance(v, SimulatedAgentContainer)}

    @property
    def real_statics(self) -> dict[str, RealStaticContainer]:
        return {k: v for k, v in self.statics.items() if isinstance(v, RealStaticContainer)}

    @property
    def simulated_statics(self) -> dict[str, SimulatedStaticContainer]:
        return {k: v for k, v in self.statics.items() if isinstance(v, SimulatedStaticContainer)}

    # === METHODS ======================================================================================================
    def start(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def reset(self):
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    def clear(self):
        self.agents.clear()
        self.statics.clear()

    # ------------------------------------------------------------------------------------------------------------------
    def new_virtual_agent(self,
                          agent_id: str,
                          config: FRODO_VisionAgent_Config | None = None, ):
        """
        Adds a virtual agent to the simulation
        Returns:

        """

        self.simulation.new_agent(agent_id, config)

    # ------------------------------------------------------------------------------------------------------------------
    def remove_simulated_agent(self, agent_id: str):
        self.simulation.remove_agent(agent_id)

    # ------------------------------------------------------------------------------------------------------------------
    def update(self):

        # 1. Generate all measurements
        self._generate_measurements()

        self.events.update.set()

    # ------------------------------------------------------------------------------------------------------------------
    def get_navigation_objects(self) -> list[NavigatedObject]:
        navigated_objects = []
        for agent in self.agents.values():
            navigated_objects.append(agent.navigator_object)
        return navigated_objects

    # === PRIVATE METHODS ==============================================================================================
    def _generate_measurements(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def _generate_virtual_measurement(self, agent_from: AgentContainer, agent_to: AgentContainer):
        state_agent_from = agent_from.state
        state_agent_to = agent_to.state

    # ------------------------------------------------------------------------------------------------------------------
    def _on_new_simulation_agent(self, agent: FRODO_VisionAgent, *args, **kwargs):
        self._add_virtual_agent(agent)

    # ------------------------------------------------------------------------------------------------------------------
    def _add_virtual_agent(self, agent: FRODO_VisionAgent):
        if agent.agent_id in self.agents:
            self.logger.warning(f"Agent {agent.agent_id} already exists in agent manager")
            return

        agent_container = SimulatedAgentContainer(
            id=agent.agent_id,
            agent=agent,
            navigator_object=FRODO_Sim_NavigatedObject(
                agent=agent,
            )
        )
        self.agents[agent.agent_id] = agent_container
        self.navigator.add_agent(agent_container.navigator_object)

        self.logger.info(f"Added virtual agent {agent.agent_id} to agent manager")
        self.events.new_agent.set(agent_container, flags={'type': 'simulation', 'id': agent.agent_id})

    # ------------------------------------------------------------------------------------------------------------------
    def _on_agent_removed_simulation(self, agent: str, *args, **kwargs):
        if agent in self.agents:
            self._remove_virtual_agent(self.agents[agent])

    # ------------------------------------------------------------------------------------------------------------------
    def _remove_virtual_agent(self, agent: AgentContainer):
        if agent.id in self.agents:
            self.navigator.remove_agent(agent.navigator_object)
            del self.agents[agent.id]
            self.logger.info(f"Removed virtual agent {agent.id} from agent manager")
            self.events.removed_agent.set(agent, flags={'type': 'simulation', 'id': agent.id})

    # ------------------------------------------------------------------------------------------------------------------
    def _add_real_agent(self, robot: TestbedObject_FRODO):
        with self._agent_lock:
            if robot.id in self.agents:
                self.logger.warning(f"Agent {robot.id} already exists in agent manager")
                return
            agent_container = RealAgentContainer(
                id=robot.id,
                testbed_object=robot,
                navigator_object=FRODO_Real_NavigatedObject(
                    robot=robot.robot,
                )
            )
            self.agents[robot.id] = agent_container
            self.navigator.add_agent(agent_container.navigator_object)
            self.logger.info(f"Added real agent {robot.id} to agent manager")
            self.events.new_agent.set(agent_container, flags={'type': 'robot', 'id': robot.id})

    # ------------------------------------------------------------------------------------------------------------------
    def _remove_real_agent(self, robot: AgentContainer):
        if robot.id in self.agents:
            self.navigator.remove_agent(robot.navigator_object)
            del self.agents[robot.id]
            self.logger.info(f"Removed real agent {robot.id} from agent manager")
            self.events.removed_agent.set(robot, flags={'type': 'robot', 'id': robot.id})

    # ------------------------------------------------------------------------------------------------------------------
    def _on_new_simulated_static(self, static: FRODO_Static, *args, **kwargs):
        if static.agent_id in self.statics:
            self.logger.warning(f"Static {static.agent_id} already exists in agent manager")
            return
        static_container = SimulatedStaticContainer(
            id=static.agent_id,
            static=static,
        )
        self.statics[static.agent_id] = static_container
        self.logger.info(f"Added simulated static {static.agent_id} to agent manager")
        self.events.new_static.set(static_container, flags={'type': 'simulation', 'id': static.agent_id})

    # ------------------------------------------------------------------------------------------------------------------
    def _on_removed_simulated_static(self, static: str, *args, **kwargs):
        if static in self.statics:
            del self.statics[static]
            self.logger.info(f"Removed simulated static {static} from agent manager")
            self.events.removed_static.set(static, flags={'type': 'simulation', 'id': static})

    # ------------------------------------------------------------------------------------------------------------------
    def _on_new_testbed_object(self, object: TestbedObject, *args, **kwargs):
        if isinstance(object, TestbedObject_FRODO):
            self._add_real_agent(object)
        elif isinstance(object, TestbedObject_STATIC):
            self._add_real_static(object)

    # ------------------------------------------------------------------------------------------------------------------
    def _on_testbed_object_removed(self, object: TestbedObject, *args, **kwargs):
        if isinstance(object, TestbedObject_FRODO):
            if object.id in self.agents:
                self._remove_real_agent(self.agents[object.id])
        elif isinstance(object, TestbedObject_STATIC):
            if object.id in self.statics:
                self._remove_real_static(self.statics[object.id])

    # ------------------------------------------------------------------------------------------------------------------
    def _add_real_static(self, static: TestbedObject_STATIC):
        if static.id in self.statics:
            self.logger.warning(f"Static {static.id} already exists in agent manager")
            return
        static_container = RealStaticContainer(
            id=static.id,
            testbed_object=static,
        )
        self.statics[static.id] = static_container
        self.logger.info(f"Added real static {static.id} to agent manager")
        self.events.new_static.set(static_container, flags={'type': 'real', 'id': static.id})

    # ------------------------------------------------------------------------------------------------------------------
    def _remove_real_static(self, static: StaticContainer):
        if static.id in self.statics:
            del self.statics[static.id]
            self.logger.info(f"Removed real static {static.id} from agent manager")
            self.events.removed_static.set(static, flags={'type': 'real', 'id': static.id})
