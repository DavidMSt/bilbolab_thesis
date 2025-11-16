import math
import time
import copy

import numpy as np
from dataclasses import dataclass, field
from typing import List, Any
import logging

from bilbolab.applications.FRODO.simulation.frodo_simulation import FRODO_Simulation, FRODO_ENVIRONMENT_ACTIONS
from bilbolab.applications.FRODO.frodo_application_sim import FRODO_VisionAgent
from bilbolab.applications.FRODO.simulation.frodo_simulation import FrodoEnvironment

import bilbolab.extensions.simulation.src.core as core
from bilbolab.core.utils.logging_utils import Logger

@dataclass
class PhaseState():
    index: int = 0
    ticks_left: int | None = None

    def reset(self):
        self.index = 0
        self.ticks_left = 0

@dataclass(frozen=True, slots=True)
class ExecutionPhase:
    """Represents executable pre-planned motion phase. 

    Raises:
        ValueError: If the inputs and durations do not match.
        ValueError: If the states do not match the inputs.
    """
    # TODO: Inputs als vorsteuerung -> Zeithorizont verändert sich? (für execution)
    inputs: tuple[np.ndarray, ...] = field(default_factory=tuple)     # shape (2,)
    states: tuple[core.spaces.State, ...] | None = field(default=None)  # State objects at segment boundaries
    durations: tuple[float, ...] = field(default_factory=tuple)         # steps per input
    delta_t: float = 0.1 # time increment used during the planned phase (phase time % simulation time != 0 for compatibility reasons)
    phase_state: PhaseState = field(default_factory=PhaseState)

    def __post_init__(self):
        if len(self.inputs) != len(self.durations):
            raise ValueError("len(inputs) must equal len(durations).")
        if self.states is not None and len(self.states) != len(self.inputs) + 1:
            raise ValueError(f"len(states) must be len(inputs)+1 (or 0 if unknown). States has length: {len(self.states)}, Inputs has length: {len(self.inputs)}.")

class PhaseRunner:
    _phases: dict[str, ExecutionPhase] # individual phases that can be executed
    _sim_dt: float # simulation time step
    _active: str # name of the currently active phase
    _current_phase_state: dict[str, int | float]

    """Holds multiple ExecutionPhase objects; only one is active at a time and executed each step."""
    def __init__(self, simulation_dt: float, logger: Logger | None = None) -> None:
        self._sim_dt = float(simulation_dt)

        # Use logger from agent or create a new one if none provided
        self._logger = logger or logging.getLogger(__name__ + ".PhaseRunner")

        # create base idle phase
        idle_phase = ExecutionPhase(inputs=(np.zeros(2),), durations=(1,), delta_t=self._sim_dt)

        # Register the idle phase
        self._phases: dict[str, ExecutionPhase] = {}
        self.add_phase("idle", idle_phase)

        self._active = 'idle'
        self._pending_end: bool = False

    # ---------- Phase management ----------

    def add_phase(self, name: str, phase: ExecutionPhase) -> None:
        # Check 
        if name in self._phases:
            raise ValueError(f"Phase '{name}' already exists with a different object.")
        ratio = phase.delta_t / self._sim_dt
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Incompatible phase delta_t: {phase.delta_t}, simulation dt: {self._sim_dt}. "
                f"Expected an integer multiple; got ratio={ratio}"
            )
        self._phases[name] = phase
        self._logger.debug(f"Added phase '{name}': {phase}")

    def change_phase(self, name: str, *, reset: bool = True) -> None:
        if name not in self._phases:
            raise KeyError(f"Unknown phase '{name}', can't be started. Add it to the runner first.")
        if reset:
            p = self._phases[name]
            p.phase_state.index = 0
            p.phase_state.ticks_left = None # will be set on first step when active 

        # Clear any pending end carried over from a previous phase
        self._pending_end = False

        self.active = name
        if name != 'idle':
            self._logger.info(f"Starting phase '{name}'")

    @property
    def active(self) -> str:
        return self._active

    @active.setter
    def active(self, name: str) -> None:
        if name not in self._phases:
            raise KeyError(f"Unknown phase '{name}', can't be set as active. Add it to the runner first.")
        self._active = name

        if name != 'idle':
            self._logger.info(f"Active phase set to '{name}'")

    def get_phase(self, name: str) -> ExecutionPhase:
        return self._phases[name]

    # ---------- Stepping ----------
    def step(self) -> np.ndarray:
        """Advance the active phase by one simulation tick. Returns control or None when finished."""

        # If a phase ended on the previous tick, finalize the switch now
        if getattr(self, "_pending_end", False):
            self._pending_end = False
            self.phase_ended()

        phase = self._phases[self.active]
        ticks_left = phase.phase_state.ticks_left
        index = phase.phase_state.index

        if ticks_left is None or ticks_left == 0:
            r = phase.delta_t / self._sim_dt
            r_int = round(r)  # assert isclose(r, r_int)
            ticks_left = max(1, math.ceil(phase.durations[index] * r_int))

        u = phase.inputs[index]

        ticks_left -= 1
        if ticks_left == 0:
            index += 1
            if index >= len(phase.inputs):
                if self.active != 'idle':
                    # Defer switching to idle until the next call, so that external logging
                    # sees the phase that produced this tick's control `u`.
                    self._pending_end = True
                else:
                    # Idle should be infinite: wrap its index and keep producing zeros
                    phase.phase_state.index = 0
                phase.phase_state.ticks_left = 0
                return u
            phase.phase_state.index = index

        phase.phase_state.ticks_left = ticks_left
        return u

    def phase_ended(self):
        active_phase = self.active
        if active_phase != "idle": # no need to tell every time the idle phase ends
            self._logger.info(f"Phase '{active_phase}' ended, removing it now.")
            del self._phases[active_phase]
        
        self.change_phase("idle", reset=True)

# class InterfaceLogger(logging.LoggerAdapter):
#     def __init__(self, logger: Any, interface_name: str) -> None:
#             # Ensure the underlying logger is a stdlib logging.Logger
#             if not isinstance(logger, logging.Logger):
#                 logger = logging.getLogger(getattr(logger, "name", __name__))
#             # Store static context (printed by formatter via %(iface)s)
#             super().__init__(logger, {"iface": interface_name})
        
    # def process(self, msg: Any, kwargs):
    #         # Merge static context with any per-call extra; per-call wins on conflicts
    #     extra = kwargs.get("extra")

    #     merged = {**self.extra, **extra}

    #     kwargs["extra"] = merged
    #     return msg, kwargs

class FRODOGeneralAgent(FRODO_VisionAgent):
# class FRODO_GeneralAgent(FRODO_Agent_Virtual_Vision):

    length: float
    width: float
    height: float 
    runner: PhaseRunner

    def __init__(self, start_config: List[float], fov_deg = 360, view_range = 1.5, runner: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # physical representations, TODO: Use Frodo physical class already existing? But seems to do collision on its own so might not be suited perfectly
        self.length = 0.157
        self.width = 0.115
        self.height = 0.052

        # intialize configuration        
        self.setPosition(x = start_config[0], y = start_config[1])
        self.setOrientation(start_config[2])

        # For web interface
        # self.measurements = {} 
        # self._plotting_group = None
        # self.fov = math.radians(fov_deg)
        # self.view_range = view_range

        # logging
        self.logger = Logger(self.agent_id)
        self.logger.setLevel('INFO')

        # runner
        if runner: 
            self.create_runner()

    def _input_function(self):
        # FIRST: Let your runner decide the input
        if self.runner is not None:
            u = self.runner.step()
            self.input.v = float(u[0])
            self.input.psi_dot = float(u[1])
            self.logger.debug(f"{self.scheduling.tick}: Runner input applied ({self.input})")

        # THEN: fall back to BilboLab parent behavior
        super()._input_function()

    def create_runner(self):
        self.runner = PhaseRunner(simulation_dt=self.Ts, logger=self.logger)

    # def _execute_next_input_action(self):
    #     # idle or no active phase? send zero input
    #     if self.runner is None:
    #         self.setInput(0.0, 0.0)
    #         self.logger.debug(f"No runnner configured for the agent, setting the input to zero")
    #         return

    #     # get input from the runner
    #     u = self.runner.step()

    #     self.setInput(float(u[0]), float(u[1]))
    #     self.logger.debug(f"{self.scheduling.tick}: ({self.agent_id}) Action Frodo Input (Phase: {self.runner.active}, agent input: {self.input})")
    #     # self.logger.debug(f'current configuration: {self.state}')

    def add_input_phase(self, name:str, inputs: tuple[np.ndarray, ...], 
                        durations: tuple[int, ...] | None = None, delta_t: float = 0.1, 
                        states: tuple[core.spaces.State, ...] | None = None, compute_states: bool = False, origin_state = None):
        
        if durations is None:
            durations = tuple([1] * len(inputs))

        if compute_states:
            if states is not None:
                self.logger.error("Tried to add phase, compute states is true but states already exists, won't override")

            else:
                # take current as origin state for the integration
                if origin_state is None:
                    origin_state = self.getConfiguration()

                if not isinstance(origin_state, core.spaces.State):
                    raise TypeError(f"origin_state must be a core.spaces.State, got {type(origin_state)}")
                
                states = self.compute_states(inputs, durations, origin_state, delta_t)

        # TODO: Make this more pretty
        new_phase = ExecutionPhase(inputs, tuple(states) if states is not None else None, tuple(durations), delta_t)
        self.runner.add_phase(name, new_phase)

    def compute_states(self,
                    inputs: tuple[np.ndarray, ...],
                    durations: tuple[int, ...],
                    initial_state: core.spaces.State,
                    delta_t: float) -> tuple[core.spaces.State, ...]:

        ratio = delta_t / self.Ts
        r_int = round(ratio)
        if not math.isclose(ratio, r_int, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"delta_t ({delta_t}) must be an integer multiple of Ts ({self.Ts}).")

        x = copy.deepcopy(initial_state)
        states = [copy.deepcopy(x)]  # s0

        for duration, u_arr in zip(durations, inputs):
            # total sim ticks for this input segment
            sim_ticks = duration * r_int

            # map the numpy input to a Space State once
            u = self.input_space.getState()
            u['v'] = float(u_arr[0])
            u['psi_dot'] = float(u_arr[1])

            for _ in range(sim_ticks):
                x = self.dynamics._dynamics(state=x, input=u)

            states.append(copy.deepcopy(x))  # boundary at segment end

        # len(states) == len(inputs) + 1
        return tuple(states)

    def change_phase(self, phase: str, reset: bool = True) -> None:
        self.runner.change_phase(phase, reset=reset)

    def getPhaseInputs(self, phase: str) -> np.ndarray:
        try:
            ph = self.runner.get_phase(phase)
            return np.array(ph.inputs, dtype=float)
        except Exception:
            self.logger.error(f'Selected phase: "{phase}" does not have any input to return!')
            return np.empty((0, 2), dtype=float)
        
# def main():
#     sim = FRODO_Simulation(Ts = 0.1, use_web_interface = False,)
#     sim.init()
#     sim.start()

#     # ---------- Option A: using simulations add virtual agent ----------
#     test_agent_a = sim.add_agent("test_agent_a", agent_class= FRODO_SimulatedVisionAgent, start_config = [0.0, 0.0, 0.0], dt = sim.env.Ts)
#     test_agent_a = sim.addVirtualAgent("test_agent_a", agent_class= FRODO_GeneralAgent, start_config = [0.0, 0.0, 0.0], dt = sim.env.Ts)
#     # giving constant input
#     test_agent_a.setInput(v= 0.5, psi_dot = 0)

#     # create test_input phase
#     inputs = [np.array([1.0, 0.0]) for _ in range(10)]
#     durations = [1] * len(inputs)

#     # ---------- Option B: Adding the agent manually ----------
#     agent_id = 'test_agent'
#     test_agent_b = FRODO_GeneralAgent(start_config = [0.0, 0.0, 0.0], fov_deg=360, view_range=1.5, agent_id=agent_id, Ts=sim.env.Ts) 
    
#     sim.agents[agent_id] = test_agent_b

#     sim.env.addObject(test_agent_b)
#     test_agent_b.logger.setLevel('DEBUG')

#     # pick different delta t -> one step for phase now equals 4 steps in the simulation
#     test_agent_b.add_input_phase('test_phase', inputs = inputs, durations= durations, delta_t=0.4)
#     test_agent_b.change_phase('test_phase', reset= True)

#     while True:
#         time.sleep(1)

def main():
    # # --- 1) Create a standard BilboLab FRODO simulation ---
    # sim = FRODO_Simulation(
    #     Ts=0.1,
    # )
    # sim.init()

    # # --- 2) Add ONE vanilla BilboLab agent ---
    # # FRODO_Simulation.add_agent() expects: (id, agent_class, **kwargs)
    # start_config = [0.0, 0.0, 0.0]

    # agent = FRODO_VisionAgent(Ts = 0.1, config=)

    # agent = sim.add_agent(
    #     id="frodo1",
    #     agent_class=FRODO_VisionAgent,
    #     start_config=start_config,
    #     dt=sim.env.Ts
    # )

    # # (Optional) Apply a constant input for testing
    # agent.setInput(v=0.3, psi_dot=0.0)

    # # --- 3) Start simulation ---
    # sim.start()

    # # --- 4) Keep program alive ---
    # while True:
    #     time.sleep(1)
    sim = FRODO_Simulation()
    sim.init()


    # Example: add one simulated agent
    sim.new_agent(agent_id='frodo1', fov_deg=100, vision_radius=1.5)

    sim.start()

    while True:
        time.sleep(10)

if __name__ == '__main__':
    from bilbolab.applications.FRODO.simulation.frodo_app_standalone import FRODO_App_Standalone
    app = FRODO_App_Standalone()
    app.init()
    app.start()

    while True:
        time.sleep(10)