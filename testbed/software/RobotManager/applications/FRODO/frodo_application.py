import dataclasses
import enum
import threading
import time

import numpy as np

from applications.FRODO.algorithm.algorithm import AlgorithmAgentState, AlgorithmAgentMeasurement, AlgorithmAgentInput
from applications.FRODO.algorithm.algorithm_centralized import CentralizedAgent, CentralizedAlgorithm
from applications.FRODO.algorithm.algorithm_distributed import DistributedAlgorithm, DistributedAgent
from applications.FRODO.data_aggregator import FRODO_DataAggregator, TestbedObject_FRODO, TestbedObject_STATIC
from applications.FRODO.gui.frodo_gui import FRODO_GUI
from applications.FRODO.tracker.frodo_tracker import FRODO_Tracker
from core.utils.callbacks import CallbackContainer, callback_definition
from core.utils.events import event_definition, Event
from core.utils.exit import register_exit_callback
from core.utils.logging_utils import Logger
from core.utils.network.network import getHostIP
from core.utils.sound.sound import SoundSystem, speak
from core.utils.time import IntervalTimer
from extensions.cli.cli import CLI, CommandSet, Command, CommandArgument
from extensions.joystick.joystick_manager import JoystickManager
from robots.frodo.frodo import FRODO
from robots.frodo.frodo_manager import FRODO_Manager

UPDATE_TIME = 0.02

INITIAL_GUESS_AGENTS = np.asarray([0.01, 0.012, 0.002])
INITIAL_GUESS_AGENTS_COVARIANCE = 1e5 * np.diag([1, 1, 1])

STATIC_AGENTS_COVARIANCE = 1e-8 * np.diag([1, 1, 1])


class AlgorithmState(enum.StrEnum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class AgentError:
    x: float
    y: float
    psi: float

    @property
    def position(self):
        return np.asarray([self.x, self.y])

    @property
    def distance(self):
        return np.linalg.norm(self.position)


# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class AgentContainer:
    centralized_algorithm_agent: CentralizedAgent
    distributed_algorithm_agent: DistributedAgent
    testbed_object: TestbedObject_FRODO
    robot: FRODO
    error: AgentError | None = None


@dataclasses.dataclass
class SimulatedAgentContainer:
    ...


@dataclasses.dataclass
class StaticContainer:
    centralized_algorithm_agent: CentralizedAgent
    distributed_algorithm_agent: DistributedAgent


@event_definition
class FRODO_Application_Events:
    algorithm_started: Event
    algorithm_stopped: Event
    update: Event


@callback_definition
class FRODO_Application_Callbacks:
    algorithm_started: CallbackContainer
    algorithm_stopped: CallbackContainer
    update: CallbackContainer


# ======================================================================================================================
class FRODO_Application:
    agents: dict
    manager: FRODO_Manager
    joystick_manager: JoystickManager

    aggregator: FRODO_DataAggregator

    algorithm_centralized: CentralizedAlgorithm
    algorithm_distributed: DistributedAlgorithm

    gui: FRODO_GUI
    tracker: FRODO_Tracker
    soundsystem: SoundSystem

    agents: dict[str, AgentContainer]
    statics: dict[str, StaticContainer]

    algorithm_state = AlgorithmState.STOPPED

    # === INIT =========================================================================================================
    def __init__(self):
        host = getHostIP()

        # Logger
        self.logger = Logger('FRODO Application', 'DEBUG')

        # Events
        self.events = FRODO_Application_Events()
        self.callbacks = FRODO_Application_Callbacks()

        # Manager
        self.manager = FRODO_Manager(host=host)
        self.manager.callbacks.new_robot.register(self._newRobot_callback)
        self.manager.callbacks.robot_disconnected.register(self._robotDisconnected_callback)

        # Joystick Manager
        self.joystick_manager = JoystickManager()

        # Sound
        self.soundsystem = SoundSystem(primary_engine='etts', volume=1)
        self.soundsystem.start()

        # Tracker
        self.tracker = FRODO_Tracker()

        # Data Aggregator
        self.aggregator = FRODO_DataAggregator(manager=self.manager, tracker=self.tracker)

        # Algorithm
        self.algorithm_centralized = CentralizedAlgorithm(Ts=UPDATE_TIME)
        self.algorithm_distributed = DistributedAlgorithm(Ts=UPDATE_TIME)

        # Objects
        self.agents = {}
        self.statics = {}

        # CLI
        self.command_set = FRODO_Application_CLI(self)
        self.cli = CLI(id='frodo_app_cli', root=self.command_set)

        # GUI
        self.gui = FRODO_GUI(host, application=self, tracker=self.tracker, cli=self.cli, manager=self.manager,
                             aggregator=self.aggregator, )

        # Timer
        self.timer = IntervalTimer(interval=UPDATE_TIME, raise_race_condition_error=False)

        # Thread
        self._exit = False
        self._thread = None

        register_exit_callback(self.close)

    # === METHODS ======================================================================================================
    def init(self):
        self.tracker.init()
        self.gui.init()
        self.manager.init()
        self.joystick_manager.init()
        self.cli.root.addChild(self.manager.cli)

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.tracker.start()
        self.gui.start()
        self.manager.start()
        self.joystick_manager.start()
        speak("Start Frodo Application")

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        speak("Closing Frodo Application")
        time.sleep(1)
        self.tracker.close()
        self._exit = True

    # ------------------------------------------------------------------------------------------------------------------
    def task(self):
        while not self._exit:
            self.update()
            self.timer.sleep_until_next()

    # ------------------------------------------------------------------------------------------------------------------
    def update(self):
        """
        This is the main update loop for acquiring data, processing the data, updating the algorithm and logging
        """

        # 1. Update the data aggregator
        self.aggregator.update()

        # 2. Update the Algorithms
        if self.algorithm_state == AlgorithmState.RUNNING:
            # 2.1 Prediction Centralized
            self._prediction_centralized()

            # 2.2 Prediction Distributed
            self._prediction_distributed()

            # 2.3 Update Centralized
            self._update_centralized()

            # 2.4 Update Distributed
            self._update_distributed()

            # 4. Calculate the estimation error
            self._calculate_estimation_errors()

        # Emit the events
        self.events.update.set()
        self.callbacks.update.call()

    # ------------------------------------------------------------------------------------------------------------------
    def init_application(self):

        if self._thread is not None:
            self.logger.warning("Application is already running")
            return

        self.agents = {}
        self.statics = {}

        # Clear the aggregator
        self.aggregator.clear()

        for robot in self.manager.robots.values():
            testbed_object = self.aggregator.addRobot(robot)

            centralized_algorithm_agent = CentralizedAgent(id=robot.id,
                                                           Ts=UPDATE_TIME,
                                                           state=AlgorithmAgentState.from_array(INITIAL_GUESS_AGENTS),
                                                           covariance=INITIAL_GUESS_AGENTS_COVARIANCE,
                                                           is_anchor=False)

            distributed_algorithm_agent = DistributedAgent(id=robot.id,
                                                           Ts=UPDATE_TIME,
                                                           state=AlgorithmAgentState.from_array(INITIAL_GUESS_AGENTS),
                                                           covariance=INITIAL_GUESS_AGENTS_COVARIANCE,
                                                           is_anchor=False)

            agent = AgentContainer(
                robot=robot,
                testbed_object=testbed_object,
                centralized_algorithm_agent=centralized_algorithm_agent,
                distributed_algorithm_agent=distributed_algorithm_agent,
            )

            self.agents[robot.id] = agent
            self.logger.important(f"Added agent {robot.id} to the application")

        # TODO: How to get the available statics?

        self.aggregator.initialize()

        self._exit = False
        self.logger.important("Application initialized")
        self._thread = threading.Thread(target=self.task, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def stop_application(self):
        self._exit = True

        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    # ------------------------------------------------------------------------------------------------------------------
    def start_algorithm(self):
        if self.algorithm_state == AlgorithmState.RUNNING:
            self.logger.warning("Algorithm is already running")
            return

        if self._thread is None:
            self.logger.warning("Application is not initialized. Please initialize the application first")
            return

        centralized_algorithm_agents = []
        distributed_algorithm_agents = []

        for agent_id, agent_container in self.agents.items():
            agent_container.centralized_algorithm_agent.state = AlgorithmAgentState.from_array(INITIAL_GUESS_AGENTS)
            agent_container.centralized_algorithm_agent.state_covariance = INITIAL_GUESS_AGENTS_COVARIANCE
            centralized_algorithm_agents.append(agent_container.centralized_algorithm_agent)

            agent_container.distributed_algorithm_agent.state = AlgorithmAgentState.from_array(INITIAL_GUESS_AGENTS)
            agent_container.distributed_algorithm_agent.state_covariance = INITIAL_GUESS_AGENTS_COVARIANCE
            distributed_algorithm_agents.append(agent_container.distributed_algorithm_agent)

        for static_id, static_container in self.statics.items():
            centralized_algorithm_agents.append(static_container.centralized_algorithm_agent)
            distributed_algorithm_agents.append(static_container.distributed_algorithm_agent)

        self.algorithm_centralized.initialize(centralized_algorithm_agents)
        self.algorithm_distributed.initialize(distributed_algorithm_agents)

        self.algorithm_state = AlgorithmState.RUNNING
        self.logger.important("Start FRODO Algorithm")
        self.soundsystem.speak('Start FRODO Algorithm')
        self.events.algorithm_started.set()

    # ------------------------------------------------------------------------------------------------------------------
    def stop_algorithm(self):
        self.algorithm_state = AlgorithmState.STOPPED
        self.logger.info("Stop FRODO Algorithm")
        self.soundsystem.speak('Stop FRODO Algorithm')
        self.events.algorithm_stopped.set()

    # ------------------------------------------------------------------------------------------------------------------
    def reset_algorithm(self):
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------

    # === PRIVATE METHODS ==============================================================================================
    def _newRobot_callback(self, robot: FRODO, *args, **kwargs):
        self.soundsystem.speak(f"New robot {robot.id} connected")
        # self.aggregator.addRobot(robot)

    # ------------------------------------------------------------------------------------------------------------------
    def _robotDisconnected_callback(self, robot: FRODO, *args, **kwargs):
        self.soundsystem.speak(f"Robot {robot.id} disconnected")
        # self.aggregator.removeRobot(robot)

    # === ALGORITHM METHODS ============================================================================================
    def _prediction_centralized(self):
        if self.algorithm_state != AlgorithmState.RUNNING:
            return

        for agent_id, agent_container in self.agents.items():
            # 3. Set the inputs
            agent_container.centralized_algorithm_agent.input = AlgorithmAgentInput.from_array(np.asarray([
                agent_container.testbed_object.dynamic_state.v,
                agent_container.testbed_object.dynamic_state.psi_dot
            ])
            )

        self.algorithm_centralized.prediction()

    # ------------------------------------------------------------------------------------------------------------------
    def _prediction_distributed(self):
        if self.algorithm_state != AlgorithmState.RUNNING:
            return

    # ------------------------------------------------------------------------------------------------------------------
    def _prediction(self):
        if self.algorithm_state != AlgorithmState.RUNNING:
            return

        for agent_id, agent_container in self.agents.items():
            agent_container.distributed_algorithm_agent.input = AlgorithmAgentInput.from_array(np.asarray([
                agent_container.testbed_object.dynamic_state.v,
                agent_container.testbed_object.dynamic_state.psi_dot
            ])
            )

            agent_container.centralized_algorithm_agent.input = AlgorithmAgentInput.from_array(np.asarray([
                agent_container.testbed_object.dynamic_state.v,
                agent_container.testbed_object.dynamic_state.psi_dot
            ]))


        self.algorithm_distributed.prediction()
        self.algorithm_centralized.prediction()


    # ------------------------------------------------------------------------------------------------------------------
    def _correction(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def _update_centralized(self):
        if self.algorithm_state != AlgorithmState.RUNNING:
            return

        # Update the real agents
        for agent_id, agent_container in self.agents.items():

            # 1. Clear measurements
            agent_container.centralized_algorithm_agent.measurements.clear()

            # 2. Add measurements from the real agent
            for measurement in agent_container.testbed_object.measurements:
                to_id = measurement.object_to.id

                algorithm_measurement = AlgorithmAgentMeasurement(
                    source=self.algorithm_centralized.agents[agent_id],
                    source_index=self.algorithm_centralized.agents[agent_id].index,
                    target=self.algorithm_centralized.agents[to_id],
                    target_index=self.algorithm_centralized.agents[to_id].index,
                    measurement=np.asarray([measurement.relative.x, measurement.relative.y, measurement.relative.psi]),
                    measurement_covariance=measurement.covariance
                )

                agent_container.centralized_algorithm_agent.measurements.append(algorithm_measurement)

        # Update the statics
        for static_id, static_container in self.statics.items():

            # 1. Clear measurements (should be empty anyway)
            static_container.algorithm_agent.measurements.clear()

            # 2. Check if the static has been moved and trigger an error here. For now, we assume that statics stay in the same position
            static_position_new = np.asarray(
                [static_container.testbed_object.state.x, static_container.testbed_object.state.y])
            if np.linalg.norm(static_position_new - static_container.algorithm_agent.state.as_array()) > 0.02:
                self.logger.error(
                    f"Static {static_id} moved. New position: {static_position_new}, original position: {static_container.algorithm_agent.state.as_array()}")

        # Update the centralized algorithm
        self.algorithm_centralized.update()

    # ------------------------------------------------------------------------------------------------------------------
    def _update_distributed(self):
        if self.algorithm_state != AlgorithmState.RUNNING:
            return

    # ------------------------------------------------------------------------------------------------------------------
    def _calculate_estimation_errors(self):

        for agent_id, agent_container in self.agents.items():
            x_true = agent_container.testbed_object.state.x
            y_true = agent_container.testbed_object.state.y
            psi_true = agent_container.testbed_object.state.psi

            x_estimated = agent_container.centralized_algorithm_agent.state.x
            y_estimated = agent_container.centralized_algorithm_agent.state.y
            psi_estimated = agent_container.centralized_algorithm_agent.state.psi

            agent_container.error = AgentError(
                x=x_true - x_estimated,
                y=y_true - y_estimated,
                psi=psi_true - psi_estimated,
            )


# === FRODO APPLICATION CLI COMMAND SET ================================================================================
class FRODO_Application_CLI(CommandSet):
    name = 'frodo_app_cli'

    def __init__(self, app: FRODO_Application):
        super().__init__(self.name)
        self.app = app

        joystick_command_set = CommandSet('joystick')
        assign_joystick_command = Command(name='assign',
                                          function=self._assign_joystick,
                                          description='Assign a joystick to a robot',
                                          allow_positionals=True,
                                          arguments=[
                                              CommandArgument(name='joystick', type=int,
                                                              short_name='j',
                                                              description='ID of the joystick'),
                                              CommandArgument(name='robot',
                                                              short_name='r',
                                                              type=str,
                                                              description='ID of the robot')
                                          ]
                                          )

        joystick_command_set.addCommand(assign_joystick_command)

        remove_joystick_command = Command(name='remove',
                                          function=self._remove_joystick,
                                          description='Remove a joystick from an agent',
                                          allow_positionals=False,
                                          arguments=[
                                              CommandArgument(name='robot',
                                                              short_name='r',
                                                              type=str,
                                                              optional=True,
                                                              default=None,
                                                              description='ID of the robot to remove the joystick from'),
                                              CommandArgument(name='joystick',
                                                              short_name='j',
                                                              type=int,
                                                              optional=True,
                                                              default=None,
                                                              description='ID of the joystick to remove'
                                                              )
                                          ]
                                          )

        joystick_command_set.addCommand(remove_joystick_command)

        start_command = Command(name='init',
                                function=self.app.init_application, )

        self.addCommand(start_command)

        self.addChild(joystick_command_set)

    # ------------------------------------------------------------------------------------------------------------------
    def _assign_joystick(self, joystick: int, robot: str):

        # 1. get the joystick from the joystick manager
        joystick = self.app.joystick_manager.getJoystickById(joystick)

        if joystick is None:
            self.app.logger.warning(f'Joystick with ID {joystick} does not exist')
            return

        # 1. Check if this joystick is already assigned to a robot
        for r in self.app.manager.robots.values():
            if joystick == r.interfaces.joystick:
                self.app.logger.warning(f'Joystick {joystick.id} is already assigned to robot {r.id}')
                return

        # 2. Assign the joystick to the robot
        robot = self.app.manager.getRobotById(robot)
        robot.interfaces.assignJoystick(joystick)

    # ------------------------------------------------------------------------------------------------------------------
    def _remove_joystick(self, robot: str = None, joystick: int = None):

        if robot is None and joystick is None:
            self.app.logger.warning('Either robot or joystick must be specified')
            return

        if robot is not None and joystick is not None:
            self.app.logger.warning('Either robot or joystick must be specified, not both')
            return

        if robot is not None:
            robot = self.app.manager.getRobotById(robot)
            if robot is None:
                self.app.logger.warning(f'Robot with ID {robot} does not exist')
                return
            robot.interfaces.removeJoystick()
        else:
            joystick = self.app.joystick_manager.getJoystickById(joystick)
            if joystick is None:
                self.app.logger.warning(f'Joystick with ID {joystick} does not exist')
                return

            owner_found = False
            for robot in self.app.manager.robots.values():
                if joystick == robot.interfaces.joystick:
                    robot.interfaces.removeJoystick()
                    owner_found = True
                    break

            if not owner_found:
                self.app.logger.warning(f'Joystick {joystick.id} is not assigned to any robot')
                return

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    app = FRODO_Application()
    app.init()
    app.start()

    while True:
        time.sleep(10)
