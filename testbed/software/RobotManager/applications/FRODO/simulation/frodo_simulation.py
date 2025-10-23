from __future__ import annotations

import dataclasses
import enum
import math
import time

import numpy as np
import qmt

import extensions.simulation.src.core as core
from applications.FRODO.navigation.navigator import Navigator, NavigatorExecutionMode, NavigatedObjectState, \
    CoordinatedMoveTo, MoveTo, NavigatorSpeedControlMode, TurnTo, TurnToPoint
from applications.FRODO.simulation.frodo_simulation_utils import is_in_fov, is_view_obstructed
from core.utils.exit import register_exit_callback
from core.utils.logging_utils import Logger
from core.utils.states import State
from extensions.cli.cli import CommandSet, Command, CommandArgument
from extensions.joystick.joystick_manager import Joystick
from extensions.simulation.src.core.environment import BASE_ENVIRONMENT_ACTIONS, Object
from extensions.simulation.src.objects.base_environment import BaseEnvironment
from extensions.simulation.src.objects.frodo.frodo import FRODO_DynamicAgent, FRODO_Input
from robots.frodo.frodo_definitions import FRODO_ControlMode

# Global registries
SIMULATED_AGENTS: dict[str, "FRODO_VisionAgent"] = {}
REAL_AGENTS: dict[str, "FRODO_VisionAgent_Real"] = {}
SIMULATED_STATICS: dict[str, "FRODO_SimulationObject"] = {}
REAL_STATICS: dict[str, "FRODO_SimulationObject"] = {}


# ======================================================================================================================
class FRODO_ENVIRONMENT_ACTIONS(enum.StrEnum):
    PREDICTION = 'frodo_prediction'
    MEASUREMENT = 'frodo_measurement'
    COMMUNICATION = 'frodo_communication'
    ESTIMATION = 'frodo_estimation'
    CORRECTION = 'frodo_correction'


# ======================================================================================================================
class FrodoEnvironment(BaseEnvironment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = Logger('FRODO ENV')
        self.logger.setLevel('INFO')

        # Put actions between communication and logic
        core.scheduling.Action(action_id=FRODO_ENVIRONMENT_ACTIONS.PREDICTION,
                               object=self,
                               function=self.action_prediction,
                               priority=21,
                               parent=self.scheduling.actions['objects'])

        # Dyanmics has priority 50

        # Put actions between communication and logic
        core.scheduling.Action(action_id=FRODO_ENVIRONMENT_ACTIONS.MEASUREMENT,
                               object=self,
                               function=self.action_measurement,
                               priority=81,
                               parent=self.scheduling.actions['objects'])

        core.scheduling.Action(action_id=FRODO_ENVIRONMENT_ACTIONS.CORRECTION,
                               object=self,
                               function=self.action_frodo_communication,
                               priority=86,
                               parent=self.scheduling.actions['objects'])

        # OUTPUT HAS 100

        # core.scheduling.Action(action_id=FRODO_ENVIRONMENT_ACTIONS.ESTIMATION,
        #                        object=self,
        #                        function=self.action_estimation,
        #                        priority=86,
        #                        parent=self.scheduling.actions['objects'])

    # ------------------------------------------------------------------------------------------------------------------
    def start(self, *args, **kwargs):
        self.logger.info("Starting FRODO Simulation Environment")
        super().start(*args, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    def action_prediction(self):
        self.logger.debug(f"{self.scheduling.tick}: Action Frodo Prediction")

    # ------------------------------------------------------------------------------------------------------------------
    def action_measurement(self):
        self.logger.debug(f"{self.scheduling.tick}: Action Frodo Measurement")

    # ------------------------------------------------------------------------------------------------------------------
    def action_frodo_communication(self):
        self.logger.debug(f"{self.scheduling.tick}: Action Frodo Communication")

    # ------------------------------------------------------------------------------------------------------------------
    def action_estimation(self):
        self.logger.debug(f"{self.scheduling.tick}: Action Frodo Estimation")

    # ------------------------------------------------------------------------------------------------------------------
    def action_dynamics(self, *args, **kwargs):
        self.logger.debug(f"{self.scheduling.tick}: Action Frodo Dynamics")
        super().action_dynamics(*args, **kwargs)


# ======================================================================================================================
class FRODO_SimulationObject(Object):
    """
    Base mixin for any object that can live in the simulation/world and be measured/occlude vision.
    Expected attributes (by convention):
      - state with x, y, psi (for statics / many dynamics)
      - or .position -> np.ndarray shape (2,)
      - optional .size (diameter) used for occlusion modeling
    """
    agent_id: str
    ...


# === FRODO SIMULATED AGENT ============================================================================================
@dataclasses.dataclass
class SimulatedAgentMeasurement:
    object_from: FRODO_SimulationObject
    object_to: FRODO_SimulationObject
    position: np.ndarray  # relative position in agent (body) frame (x,y)
    psi: float  # relative heading of target w.r.t. agent heading
    covariance: np.ndarray  # 3x3 covariance for [dx, dy, dpsi]

    def as_vector(self):
        return np.asarray([self.position[0], self.position[1], self.psi, ])


class FRODO_VisionAgent(FRODO_DynamicAgent, FRODO_SimulationObject):
    measurements: list[SimulatedAgentMeasurement]

    fov: float  # Field-of-view in radians
    vision_radius: float  # Maximum view range
    size: float  # Diameter of the circle encompassing the object. Is used to detect vision blockage

    control_mode: FRODO_ControlMode = FRODO_ControlMode.NAVIGATION

    navigator: Navigator | None

    cli: FRODO_VisionAgent_CommandSet

    # === INIT =========================================================================================================
    def __init__(self, agent_id, Ts, fov_deg: float = 100, vision_radius: float = 1.5, size: float = 0.2, *args,
                 **kwargs):
        super().__init__(agent_id, Ts=Ts, *args, **kwargs)


        self.logger = Logger(self.agent_id)
        self.logger.setLevel('INFO')

        self.fov = math.radians(fov_deg)
        self.vision_radius = vision_radius
        self.size = size

        self.navigator = Navigator(mode=NavigatorExecutionMode.EXTERNAL,
                                   speed_control_mode=NavigatorSpeedControlMode.SPEED_CONTROL,
                                   speed_command_function=self._navigator_set_speed,
                                   state_fetch_function=self._navigator_get_state)

        core.scheduling.Action(action_id=FRODO_ENVIRONMENT_ACTIONS.MEASUREMENT,
                               object=self,
                               function=self.action_measurement,
                               priority=1)

        core.scheduling.Action(action_id=FRODO_ENVIRONMENT_ACTIONS.COMMUNICATION,
                               object=self,
                               function=self.action_frodo_communication,
                               priority=2)

        core.scheduling.Action(action_id=FRODO_ENVIRONMENT_ACTIONS.ESTIMATION,
                               object=self,
                               function=self.action_estimation,
                               priority=3)

        self.scheduling.actions['output'].addAction(self.action_custom_output)

        self.scheduling.actions[BASE_ENVIRONMENT_ACTIONS.LOGIC].addAction(self._control)

        self.input = [0, 0]
        self.measurements = []

        self.cli = FRODO_VisionAgent_CommandSet(self)

    # === METHODS ======================================================================================================
    def action_prediction(self):
        self.logger.debug(f"{self.agent_id}: Action Frodo Prediction")

    def action_measurement(self):
        self._generate_measurements()

    def action_frodo_communication(self):
        self.logger.debug(f"{self.agent_id}: Action Frodo Communication")

    def action_estimation(self):
        self.logger.debug(f"{self.agent_id}: Action Frodo Estimation")

    def action_custom_output(self):
        self.logger.debug(f"{self.agent_id}: {self.state}")

    def set_mode(self, mode: FRODO_ControlMode):
        self.control_mode = mode

    # ------------------------------------------------------------------------------------------------------------------
    def move_to(self, x, y, force=False):

        # Start Navigation of not already started
        self.navigator.startNavigation()

        element = MoveTo(
            x=x,
            y=y
        )
        self.navigator.addElement(element, force_element=force)

    # ------------------------------------------------------------------------------------------------------------------
    def turn_to(self, psi):
        self.navigator.startNavigation()

        element = TurnTo(
            psi=psi
        )
        self.navigator.addElement(element)

    # ------------------------------------------------------------------------------------------------------------------
    def turn_to_point(self, x, y):
        self.navigator.startNavigation()
        element = TurnToPoint(
            x=x,
            y=y
        )
        self.navigator.addElement(element)

    # ------------------------------------------------------------------------------------------------------------------
    def abort_navigation_element(self):
        self.navigator.abort_element()
    # ------------------------------------------------------------------------------------------------------------------
    def set_state(self, x: float = None, y: float = None, psi: float = None):
        if x is not None:
            self.state.x = x
        if y is not None:
            self.state.y = y
        if psi is not None:
            self.state.psi = psi

    # === PRIVATE METHODS ==============================================================================================
    def _generate_covariance(self, rel_position_vec: np.ndarray, rel_psi: float) -> np.ndarray:
        """
        Placeholder covariance generator. Currently returns a small diagonal covariance.
        Override or modify later to reflect sensor characteristics.
        """
        var_pos = 1e-5
        var_ang = 1e-5
        return np.diag([var_pos, var_pos, var_ang])

    def _generate_measurements(self):
        """
        Simulated vision agent:
        - Computes measurements to ALL objects from global registries:
            * Agents: SIMULATED_AGENTS + REAL_AGENTS (excluding self)
            * Statics: SIMULATED_STATICS + REAL_STATICS
        - Uses FOV, range, and occlusion checks
        - Measurements are expressed in the agent's local frame
        """
        self.measurements = []
        # Targets from globals
        agent_targets = [a for a in list(SIMULATED_AGENTS.values()) + list(REAL_AGENTS.values())
                         if getattr(a, 'agent_id', None) != self.agent_id]
        static_targets = list(SIMULATED_STATICS.values()) + list(REAL_STATICS.values())

        # Occluders: everything except self (we'll exclude the current target per check)
        occluders: list[FRODO_SimulationObject] = []
        occluders.extend(agent_targets)
        occluders.extend(static_targets)

        def obj_position(o: FRODO_SimulationObject) -> np.ndarray:
            if hasattr(o, 'position'):
                return np.asarray(o.position).reshape(2)
            elif hasattr(o, 'state'):
                return np.array([o.state.x, o.state.y])
            else:
                raise AttributeError("Object has no position/state")

        def obj_psi(o: FRODO_SimulationObject) -> float:
            if hasattr(o, 'state'):
                return float(o.state.psi)
            else:
                # For objects without orientation, zero is fine
                return 0.0

        def obj_size(o: FRODO_SimulationObject) -> float:
            # Diameter; fall back to a small default if not present
            return float(getattr(o, 'size', 0.2))

        own_pos = np.array([self.state.x, self.state.y])
        own_psi = float(self.state.psi)

        # Local-frame transform: world -> agent body
        R_world_to_body = np.array([
            [math.cos(own_psi), math.sin(own_psi)],
            [-math.sin(own_psi), math.cos(own_psi)]
        ])

        for target in [*agent_targets, *static_targets]:
            t_pos = obj_position(target)

            # FOV + range check
            if not is_in_fov(
                    pos=[own_pos[0], own_pos[1]],
                    psi=own_psi,
                    fov=self.fov,
                    radius=self.vision_radius,
                    other_agent_pos=[t_pos[0], t_pos[1]],
            ):
                continue

            # Occlusion check (ignore the target itself)
            other_occluders = [o for o in occluders if o is not target]
            obstacles = []
            for o in other_occluders:
                c = obj_position(o)
                r = obj_size(o) * 0.5  # size is diameter -> radius
                obstacles.append((c, r))

            if is_view_obstructed(own_pos, t_pos, obstacles):
                continue

            # Relative vector in world frame -> transform to body frame
            rel_vec_world = t_pos - own_pos
            rel_vec_body = R_world_to_body @ rel_vec_world

            # Relative orientation (target vs self)
            rel_psi = qmt.wrapToPi(obj_psi(target) - own_psi)

            cov = self._generate_covariance(rel_vec_body, rel_psi)

            measurement = SimulatedAgentMeasurement(
                object_from=self,
                object_to=target,
                position=rel_vec_body,
                psi=rel_psi,
                covariance=cov
            )
            self.measurements.append(measurement)

    # ------------------------------------------------------------------------------------------------------------------
    def _control(self):
        if self.control_mode == FRODO_ControlMode.NAVIGATION:
            self.navigator.update()

    # ------------------------------------------------------------------------------------------------------------------
    def _navigator_set_speed(self, v, psi_dot):
        if self.control_mode == FRODO_ControlMode.NAVIGATION:
            self.input.v = v
            self.input.psi_dot = psi_dot

    # ------------------------------------------------------------------------------------------------------------------
    def _navigator_get_state(self) -> NavigatedObjectState:
        return NavigatedObjectState(x=self.state.x, y=self.state.y, psi=self.state.psi, v=self.state.v,
                                    psi_dot=self.state.psi_dot)


# ======================================================================================================================
class FRODO_VisionAgent_Interactive(FRODO_VisionAgent):
    joystick: Joystick | None

    last_input: FRODO_Input

    def __init__(self, agent_id, Ts, fov_deg: float = 100, vision_radius: float = 1.5, *args, **kwargs):
        super().__init__(agent_id, Ts, fov_deg, vision_radius, *args, **kwargs)
        self.joystick = None

        self.scheduling.actions[BASE_ENVIRONMENT_ACTIONS.INPUT].addAction(self._input_function)
        self.scheduling.actions[BASE_ENVIRONMENT_ACTIONS.OUTPUT].addAction(self._output_function)
        self.last_input = FRODO_Input(0, 0)

    # ------------------------------------------------------------------------------------------------------------------
    def add_joystick(self, joystick):
        self.logger.debug(f"{self.agent_id}: Adding Joystick {joystick}")
        self.joystick = joystick
        self.set_mode(FRODO_ControlMode.EXTERNAL)

    # ------------------------------------------------------------------------------------------------------------------
    def remove_joystick(self):
        self.joystick = None
        self.input = [0, 0]
        self.set_mode(FRODO_ControlMode.NAVIGATION)
        self.logger.debug(f"{self.agent_id}: Removed Joystick {self.joystick}")

    # ------------------------------------------------------------------------------------------------------------------
    def _input_function(self):
        if self.joystick is None:
            return

        axis_forward = self.joystick.getAxis('LEFT_VERTICAL')

        if abs(axis_forward) < 0.05:
            axis_forward = 0

        axis_turn = self.joystick.getAxis('RIGHT_HORIZONTAL')

        if abs(axis_turn) < 0.05:
            axis_turn = 0

        self.input.v = -3 * axis_forward * 0.2
        self.input.psi_dot = -3 * axis_turn

    def _output_function(self):
        self.last_input = FRODO_Input(self.input.v, self.input.psi_dot)


# ======================================================================================================================
class FRODO_VisionAgent_Real(FRODO_VisionAgent):

    def __init__(self, agent_id, fov_deg: float = 100, vision_radius: float = 1.5, *args, **kwargs):
        super().__init__(agent_id, fov_deg, vision_radius, *args, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    def _generate_measurements(self):
        """
        Real vision agent:
        - ONLY computes measurements to SIMULATED objects (agents + statics) from globals.
        - Measurements to REAL objects are assumed to come from the experiment and are not generated here.
        """
        self.measurements = []

        # Targets from globals (simulated only)
        sim_agent_targets = [a for a in SIMULATED_AGENTS.values()
                             if getattr(a, 'agent_id', None) != self.agent_id]
        sim_static_targets = list(SIMULATED_STATICS.values())

        # Occluders: include all bodies except self (both sim + real, agents + statics)
        occluders: list[FRODO_SimulationObject] = []
        occluders.extend([a for a in SIMULATED_AGENTS.values() if a.agent_id != self.agent_id])
        occluders.extend([a for a in REAL_AGENTS.values() if a.agent_id != self.agent_id])
        occluders.extend(SIMULATED_STATICS.values())
        occluders.extend(REAL_STATICS.values())

        def obj_position(o: FRODO_SimulationObject) -> np.ndarray:
            if hasattr(o, 'position'):
                return np.asarray(o.position).reshape(2)
            elif hasattr(o, 'state'):
                return np.array([o.state.x, o.state.y])
            else:
                raise AttributeError("Object has no position/state")

        def obj_psi(o: FRODO_SimulationObject) -> float:
            if hasattr(o, 'state'):
                return float(o.state.psi)
            else:
                return 0.0

        def obj_size(o: FRODO_SimulationObject) -> float:
            return float(getattr(o, 'size', 0.2))

        own_pos = np.array([self.state.x, self.state.y])
        own_psi = float(self.state.psi)

        R_world_to_body = np.array([
            [math.cos(own_psi), math.sin(own_psi)],
            [-math.sin(own_psi), math.cos(own_psi)]
        ])

        for target in [*sim_agent_targets, *sim_static_targets]:
            t_pos = obj_position(target)

            if not is_in_fov(
                    pos=[own_pos[0], own_pos[1]],
                    psi=own_psi,
                    fov=self.fov,
                    radius=self.vision_radius,
                    other_agent_pos=[t_pos[0], t_pos[1]],
            ):
                continue

            # Occlusion using all bodies except the target
            other_occluders = [o for o in occluders if o is not target]
            obstacles = []
            for o in other_occluders:
                c = obj_position(o)
                r = obj_size(o) * 0.5
                obstacles.append((c, r))

            if is_view_obstructed(own_pos, t_pos, obstacles):
                continue

            rel_vec_world = t_pos - own_pos
            rel_vec_body = R_world_to_body @ rel_vec_world
            rel_psi = qmt.wrapToPi(obj_psi(target) - own_psi)

            cov = self._generate_covariance(rel_vec_body, rel_psi)

            measurement = SimulatedAgentMeasurement(
                object_from=self,
                object_to=target,
                position=rel_vec_body,
                psi=rel_psi,
                covariance=cov
            )
            self.measurements.append(measurement)


# ======================================================================================================================
@dataclasses.dataclass
class FRODO_Static_State(State):
    x: float
    y: float
    psi: float


class FRODO_Static(FRODO_SimulationObject):
    state: FRODO_Static_State
    size: float = 0.2  # optional diameter for occlusion

    def __init__(self, static_id, x: float = None, y: float = None, psi: float = None, size: float = 0.2, *args,
                 **kwargs):
        super().__init__(static_id, *args, **kwargs)
        self.agent_id = static_id
        self.size = size
        self.state = FRODO_Static_State(0, 0, 0)
        self.setState(x, y, psi)
        print(f"Created static {static_id} at ({self.state.x}, {self.state.y})")

    def setState(self, x: float = None, y: float = None, psi: float = None):
        if x is not None:
            self.state.x = x
        if y is not None:
            self.state.y = y
        if psi is not None:
            self.state.psi = psi


# ======================================================================================================================


# === FRODO SIMULATION =================================================================================================
class FRODO_Simulation:
    environment: FrodoEnvironment

    cli: FRODO_Simulation_CommandSet | None = None

    # === INIT =========================================================================================================
    def __init__(self, Ts=0.05):
        self.Ts = Ts
        self.logger = Logger('FRODO Simulation', 'DEBUG')
        self.environment = FrodoEnvironment(Ts=Ts, run_mode='rt')

        self.cli = FRODO_Simulation_CommandSet(self)

        register_exit_callback(self.stop)

    # === METHODS ======================================================================================================
    def init(self):
        self.environment.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.logger.info("Starting FRODO Simulation")
        self.environment.start(thread=True, )

    # ------------------------------------------------------------------------------------------------------------------
    def stop(self):
        self.logger.info("Stopping FRODO Simulation")
        self.environment.stop()

    # ------------------------------------ AGENTS ----------------------------------------------------------------------
    def addSimulatedAgent(self, agent_id, fov_deg=100, vision_radius=1.5, interactive: bool = False, *args,
                          **kwargs) -> FRODO_VisionAgent:

        if interactive:
            agent = FRODO_VisionAgent_Interactive(agent_id, self.Ts, fov_deg, vision_radius, *args, **kwargs)
        else:
            agent = FRODO_VisionAgent(agent_id, self.Ts, fov_deg, vision_radius, *args, **kwargs)

        global SIMULATED_AGENTS
        SIMULATED_AGENTS[agent_id] = agent
        self.environment.addAgent(agent)
        self.logger.info(f"Simulated agent {agent_id} added")

        self.cli.addChild(agent.cli)

        return agent

    # ------------------------------------------------------------------------------------------------------------------
    def removeSimulatedAgent(self, agent: FRODO_VisionAgent | str):
        if isinstance(agent, FRODO_VisionAgent):
            agent = agent.agent_id

        if agent in SIMULATED_AGENTS:
            self.environment.removeObject(SIMULATED_AGENTS[agent])
            self.cli.removeChild(SIMULATED_AGENTS[agent].cli)
            del SIMULATED_AGENTS[agent]
            self.logger.info(f"Simulated agent {agent} removed")
        else:
            self.logger.warning(f"Simulated agent {agent} not found. Cannot remove it from the simulation")

    # ------------------------------------------------------------------------------------------------------------------
    def addRealAgent(self, agent_id, fov_deg=100, vision_radius=1.5, *args, **kwargs):
        agent = FRODO_VisionAgent_Real(agent_id, fov_deg, vision_radius, *args, **kwargs)
        REAL_AGENTS[agent_id] = agent
        self.environment.addAgent(agent)
        self.logger.info(f"Real agent {agent_id} added")

    # ------------------------------------------------------------------------------------------------------------------
    def removeRealAgent(self, agent: FRODO_VisionAgent_Real | str):
        if isinstance(agent, FRODO_VisionAgent_Real):
            agent = agent.agent_id

        if agent in REAL_AGENTS:
            self.environment.removeObject(REAL_AGENTS[agent])
            del REAL_AGENTS[agent]
            self.logger.info(f"Real agent {agent} removed")
        else:
            self.logger.warning(f"Real agent {agent} not found. Cannot remove it from the simulation")

    # ------------------------------------ STATICS ---------------------------------------------------------------------
    def addSimulatedStatic(self, static_id: str, *args, **kwargs) -> FRODO_Static | None:

        if static_id in SIMULATED_STATICS:
            self.logger.warning(f"Simulated static {static_id} already exists. Cannot add it again")
            return None

        static_obj = FRODO_Static(static_id, *args, **kwargs)
        SIMULATED_STATICS[static_id] = static_obj
        # statics are environment "objects" (not agents)
        self.environment.addObject(static_obj)
        self.logger.info(f"Simulated static {static_id} added")
        return static_obj

    # ------------------------------------------------------------------------------------------------------------------
    def removeSimulatedStatic(self, static: str | FRODO_SimulationObject):
        # allow passing id or object
        if not isinstance(static, str):
            # try to find id by object identity
            found_id = None
            for sid, sobj in SIMULATED_STATICS.items():
                if sobj is static:
                    found_id = sid
                    break
            static_id = found_id
        else:
            static_id = static

        if static_id in SIMULATED_STATICS:
            self.environment.removeObject(SIMULATED_STATICS[static_id])
            del SIMULATED_STATICS[static_id]
            self.logger.info(f"Simulated static {static_id} removed")
        else:
            self.logger.warning(f"Simulated static {static_id} not found. Cannot remove it from the simulation")

    # ------------------------------------------------------------------------------------------------------------------
    def addRealStatic(self, static_id: str, static_obj: FRODO_SimulationObject):
        REAL_STATICS[static_id] = static_obj
        self.environment.addObject(static_obj)
        self.logger.info(f"Real static {static_id} added")

    # ------------------------------------------------------------------------------------------------------------------
    def removeRealStatic(self, static: str | FRODO_SimulationObject):
        if not isinstance(static, str):
            found_id = None
            for sid, sobj in REAL_STATICS.items():
                if sobj is static:
                    found_id = sid
                    break
            static_id = found_id
        else:
            static_id = static

        if static_id in REAL_STATICS:
            self.environment.removeObject(REAL_STATICS[static_id])
            del REAL_STATICS[static_id]
            self.logger.info(f"Real static {static_id} removed")
        else:
            self.logger.warning(f"Real static {static_id} not found. Cannot remove it from the simulation")

    # === PRIVATE METHODS ==============================================================================================


class FRODO_VisionAgent_CommandSet(CommandSet):

    def __init__(self, agent: FRODO_VisionAgent):
        super().__init__(name=agent.agent_id)
        self.agent = agent

        command_move_to = Command(
            name='move_to',
            description='Move the agent to a given position',
            arguments=[
                CommandArgument(name='x', type=float, description='x position'),
                CommandArgument(name='y', type=float, description='y position'),
                CommandArgument(name='force', short_name='f', type=bool, is_flag=True, optional=True, default=False),
            ],
            function=self.agent.move_to,
            allow_positionals=True
        )

        command_turn_to = Command(
            name='turn_to',
            description='Turn the agent to a given orientation',
            arguments=[
                CommandArgument(name='psi', type=float, description='orientation in radians'),
            ],
            function=self.agent.turn_to,
            allow_positionals=True
        )

        command_turn_to_point = Command(
            name='turn_to_point',
            description='Turn the agent to a given orientation (pointing towards the given position)',
            arguments=[
                CommandArgument(name='x', type=float, description='x position'),
                CommandArgument(name='y', type=float, description='y position'),
            ],
            function=self.agent.turn_to_point,
            allow_positionals=True
        )

        command_skip_element = Command(
            name='skip',
            description='',
            arguments=[],
            function=self.agent.abort_navigation_element,
            allow_positionals=False
        )

        command_set_state = Command(
            name='set_state',
            description='Set the agent state directly',
            arguments=[
                CommandArgument(name='x', type=float, description='x position', optional=True, default=None),
                CommandArgument(name='y', type=float, description='y position', optional=True, default=None),
                CommandArgument(name='psi', type=float, description='orientation in radians', optional=True,
                                default=None),
            ],
            function=self.agent.set_state,
            allow_positionals=True
        )

        self.addCommand(command_move_to)
        self.addCommand(command_turn_to)
        self.addCommand(command_turn_to_point)
        self.addCommand(command_set_state)
        self.addCommand(command_skip_element)


class FRODO_Simulation_CommandSet(CommandSet):
    def __init__(self, sim: FRODO_Simulation):
        super().__init__(name='simulation')
        self.sim = sim
        command_list = Command(
            name='list',
            description='List all agents and statics',
            arguments=[],
            function=lambda: self.sim.logger.info(
                f"Agents: {list(SIMULATED_AGENTS.keys())}\nStatic: {list(SIMULATED_STATICS.keys())}")
        )

        self.addCommand(command_list)

    if __name__ == '__main__':
        sim = FRODO_Simulation()
        sim.init()

        # Example: add one simulated agent
        sim.addSimulatedAgent(agent_id='frodo1', fov_deg=100, vision_radius=1.5)

        sim.start()

        while True:
            time.sleep(10)
