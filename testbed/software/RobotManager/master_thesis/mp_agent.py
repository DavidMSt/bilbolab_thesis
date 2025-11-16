import abc
from dataclasses import dataclass, field, replace
from time import sleep
import math
import threading
import time
import numpy as np
from typing import Type, cast, List, Callable
import logging

# bilbolab
from bilbolab.applications.FRODO.simulation.frodo_simulation_utils import frodo_virtual_agent_colors, is_in_fov
from bilbolab.applications.FRODO.simulation.frodo_simulation import FRODO_ENVIRONMENT_ACTIONS
from bilbolab.applications.FRODO.utilities.web_gui.FRODO_Web_Interface import FRODO_Web_Interface, Group 
import bilbolab.extensions.simulation.src.core as core
from bilbolab.core.utils.logging_utils import Logger
from bilbolab.applications.FRODO.simulation.frodo_simulation import FrodoEnvironment

from general.general_agents import FRODOGeneralAgent
from general.general_simulation import FRODO_general_Simulation, FrodoGeneralEnvironment
from motion_planning.helper.ompl_planner import OMPLPlannerFRODOKino, OMPLPlannerBase
from general.general_agents import PhaseRunner, ExecutionPhase

# TODO: Apply offset bidirectional from ompl to simulation and from simulation back (initialization of start config)





# TODO: Make this frozen
@dataclass
class MotionPlanningConfig:
    # timestep duration
    dt: float

    # physical dimensions
    L: float
    W: float
    H: float

    # start/ goal configurations
    start: np.ndarray
    goal: np.ndarray | None

    # samplingn boundaries
    # env_min: np.ndarray
    # env_max: np.ndarray
    env_limits: List[List[float]]
    obstacles: list

    type: str = "frodo"

    # hyperparameters
    # planner type selected
    planner: str = 'rrt'
    # timelimit for each motion planning problem
    timelimit: float = 60.0
    # bias for goal sampling
    goal_bias: float = 0.1
    # size of the goal region
    goal_eps: float = 0.1
    # weight of so2 relative to r2 in distance metric
    so_r2_weight: float = 1.0

    # control bounds
    theta_dot_bounds: tuple = (-np.pi/3, np.pi/3)
    v_bounds: tuple = (-1.0, 1.0)

class MPAgentModule():
    dt: float  | None # delta_t timestep used during sampling
    env: FrodoGeneralEnvironment
    id: str # mainly used for optional plotting of start/ goal configurations in the GUi

    # physical dimensions for all FRODOs
    length: float = 0.157
    width: float = 0.115
    height: float = 0.052  

    def __init__(self, env: FrodoGeneralEnvironment, runner: PhaseRunner, id: str, logger: Logger, plotting_group: Group | None = None) -> None:
        self.env = env
        self.runner = runner
        self.id = id
        self.plotting_group = plotting_group
        self.logger = logger


        # runner: "PhaseRunner | None" = None
        # algorithm: str = 'kinodynamic_rrt'
        # motion_planner: OMPLPlannerBase| None = None

    def plot_goal_config(self, goal_config: np.ndarray):
        """Plots a given goal config inside the GUI

        Args:
            goal_config (np.ndarray): _description_
        """
        # self.goal_config = goal_config
        if self.plotting_group: #TODO: Move this again to the simulation, since it will decide which agent to assign which goal? 
            self.plotting_group.add_point(
                id=f"goal_{self.id}",
                x=goal_config[0],
                y=goal_config[1],
                color=[0, 1, 0],  
                size=0.7         
            )

    def plan_motion(self, phase_key: str, start_config, goal_config, motion_planner = OMPLPlannerFRODOKino):
        if self.runner is None:
            self.logger.warning("Runner not initialized, Solution is not added as executable phase")
        self.motion_planner = motion_planner(self.export_config(start_config, goal_config))# TODO: initialize the planner once, but still be able to dynamically handle obstacles in the environment to enable obstacle creation after agent creation
        solved, path_length = self.motion_planner.solve_problem()

        if solved:
            solution_dict = self.motion_planner.export_solution_dict()
            phase = ExecutionPhase(
                inputs=solution_dict["actions"],
                states=solution_dict["states"],
                durations=solution_dict["durations"],
                delta_t=float(solution_dict["delta_t"]),
            )

            
            # (PhaseRunner.add_phase already checks duplicates)
            
            self.runner.add_phase(phase_key, phase)


            self.logger.info(
                f"Found solution with {len(phase.states)} states, "
                f"total length {path_length}, end config: {phase.states[-1]}"
            )
        else:
            self.logger.warning(f"No solution found! timeout after {self.export_config().timelimit} s")
    
    def export_config(self, start_config, goal_config) -> MotionPlanningConfig:
        # if self.env is None:
        #     error_message = "The Motion Planning Interface needs the environment instance used in the simulation to plan a path! Provide during initializaion or use bin_environment method."

        #     if self.logger is not None:
        #         self.logger.error(error_message)
        #     else:
        #         raise AttributeError(error_message)

        return MotionPlanningConfig(
            dt=self.env.Ts,
            L=self.length,
            W=self.width,
            H=self.height,
            start=start_config,
            goal=goal_config,
            env_limits = self.env.limits,
            obstacles=self.env.obstacles
        )

class FRODO_MotionPlanning_Agent(FRODOGeneralAgent):
    mp_interface: MPAgentModule
 
    def __init__(self, env, start_config: List[float], fov_deg = 360, view_range = 1.5, *args, **kwargs) -> None: # TODO: Change start config here and in the general class to tuple
        super().__init__(start_config, fov_deg, view_range, runner = True, *args, **kwargs)

        # motion_planning interface for OMPL configuration
        self.mpi = MPAgentModule(
            env=env,  # will be bound later
            id=self.agent_id,
            runner= self.runner,
            plotting_group = self._plotting_group,
            logger = self.logger
        )

    # def plan_motion(self, phase_key):
    #     self.mp_interface.plan_motion(phase_key)

    #     if self.mp_interface.runner is None:
    #         if self.mp_interface.dt is None: 
    #             raise RuntimeError("Environment dt is None, first bind an environment to the MPI before trying to plan motion")



    # def set_goal_config(self, goal_config: np.ndarray):
    #     self.mp_interface.plot_goal_config(goal_config )

    # def _execute_next_input_action(self):
    #     # idle or no active phase? send zero input
    #     if self.mp_interface.runner is None or self._execution_phase == 'idle' or self.mp_interface.active_phase() is None:
    #         self.setInput(0.0, 0.0)
    #         return

    #     u = self.mp_interface.step_active()
    #     if u is None:
    #         # phase finished this tick
    #         self.setInput(0.0, 0.0)
    #         self._execution_phase = 'idle'
    #         self.logger.debug("Phase completed; switching to idle.")
    #     else:
    #         self.setInput(float(u[0]), float(u[1]))

    # def setInput(self, v: float=0.0, psi_dot:float =0):
    #     self.input = [v, psi_dot]
    
    # def set_phase(self, phase: str):
    #     if self.mp_interface.has_phase(phase):
    #         self.mp_interface.start_phase(phase, reset=True)
    #         self._execution_phase = phase
    #         self.logger.info(f"Setting execution phase to '{phase}'")
    #     else:
    #         self.logger.warning(f"Phase '{phase}' not found")

    # def getPhaseInputs(self, phase: str) -> np.ndarray:
    #     try:
    #         ph = self.mp_interface.runner.get_phase(phase)  # type: ignore[union-attr]
    #         return np.array(ph.inputs, dtype=float)
    #     except Exception:
    #         return np.empty((0, 2), dtype=float)

    
    # def getPhaseStates(self, phase: str) -> np.ndarray:
    #     states = self.mp_interface.get_phase_states(phase)
    #     if states is None:
    #         return np.empty((0, 3), dtype=float)
    #     if states.size == 0:
    #         return states
    #     states[:, 2] += np.pi
    #     return states

if __name__ == '__main__':
    sim = FRODO_general_Simulation(Ts = 0.1, use_web_interface=True, env = FrodoGeneralEnvironment)
    sim.init()
    sim.start()
    start_config = [0.0,0.0,0.0]
    goal_config = [1.0,1.0,np.pi]
    agent = FRODO_MotionPlanning_Agent(env = sim.environment, agent_id = 'frodo1_v', start_config= start_config, dt = 0.1, Ts = 0.1) # TODO: Should not be needed to do dt and Ts here
    sim.addExistingVirtualAgent(agent)
    agent.mpi.plan_motion(phase_key='test_phase', start_config=start_config, goal_config=goal_config)
    
    
    agent.runner.change_phase('test_phase')

    sleep(3)
    # print(agent.getConfiguration())
    print(agent._configuration)
