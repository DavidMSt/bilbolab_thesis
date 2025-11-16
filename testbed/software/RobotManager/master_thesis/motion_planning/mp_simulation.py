from dataclasses import dataclass, field, replace
import time
import numpy as np
from typing import Type, cast, overload
from logging import Logger

# bilbolab
from bilbolab.applications.FRODO.simulation.frodo_simulation import FRODO_Simulation, FRODO_ENVIRONMENT_ACTIONS, FrodoEnvironment
from bilbolab.applications.FRODO.utilities.web_gui.FRODO_Web_Interface import FRODO_Web_Interface, Group
from bilbolab.extensions.simulation.src.objects.frodo import FRODO_DynamicAgent
import bilbolab.extensions.simulation.src.core as core


from motion_planning.mp_agent import FRODO_MotionPlanning_Agent
from general.general_simulation import FRODO_general_Simulation, FrodoGeneralEnvironment
from motion_planning.mp_agent import FRODO_MotionPlanning_Agent, MPAgentModule
from general.general_agents import FRODOGeneralAgent

class MPSimulationModule():
    agents: dict[str, FRODO_DynamicAgent]
    logger: Logger

    def __init__(self, agents: dict[str, FRODO_DynamicAgent], logger:Logger) -> None: # TODO: Make dataclass? 
        self.agents = agents
        self.logger = logger

    def agent_motion_planning(self, agent: str | FRODO_DynamicAgent, 
                              solution_phase_name: str, *, goal_config: tuple[float, ...], 
                              start_config: tuple[float]| None = None):
        if isinstance(agent, str):
            agent = self.agents[agent]
        
        mpi = getattr(agent, 'mpi', None)
        if mpi is not None and MPAgentModule:
            mpi.plan_motion(phase_key=solution_phase_name, start_config=start_config, goal_config=goal_config)
        
        else:
            self.logger.error(f'Tried to plan motion for agent: {agent.agent_id}, no Motion planning interface present')

    def multiple_agents_motion_planning(self, goal_configs: list, start_configs: list |None = None):
        raise NotImplementedError
        agents = self.agents

        for agent in agents.values():
            if start_configs == None:
                start_config = agent.configuration

class FRODO_MP_Simulation(FRODO_general_Simulation): # TODO: Move all the logic into an interface that uses the base class Or better create a master thesis class which can be used for all the master thesis tasks and then has the individual interfaces. 
    def __init__(self, Ts=0.05, use_web_interface: bool = False, env = FrodoGeneralEnvironment, limits: tuple[tuple[int, int], ...] = ((-3,3),(-3,3)), *args, **kwargs):
        super().__init__(Ts=Ts, use_web_interface= use_web_interface, env= env, limits=limits,*args, **kwargs)
        self.mpi = MPSimulationModule(agents = self.agents, logger= self.logger)


def motion_planning_task():
    app = FRODO_MP_Simulation(use_web_interface=True)
    app.init()

    ag1_start = [2, 0, np.pi]
    ag1_goal = [-1.0, 0.0, np.pi]
    ag2_start = [2, 0, np.pi]
    ag2_goal = [3.0,0, 0]

    ag1 = app.add_agent(id="frodo1_v", agent_class=FRODO_MotionPlanning_Agent, start_config=ag1_start)
    # ag1.set_goal_config(np.array([3.0,0, 0]))
    app.mpi.agent_motion_planning(ag1, solution_phase_name="goal", start_config= ag1_start, goal_config=ag1_goal)

    ag2 = app.add_agent(agent_class=FRODO_MotionPlanning_Agent, id="frodo2_v", start_config=ag2_start)
    # ag2 = app.addVirtualMotionPlanningAgent(agent_id="frodo2_m", start_config=np.array([0.0, 0.0, 0.0]), color= color_ag1)
    app.mpi.agent_motion_planning(ag2, solution_phase_name="goal", start_config= ag2_start, goal_config=ag2_goal)

    app.start() # TODO: move back to directly after init

    time.sleep(1)

    app.set_phase_all_agents(phase="goal")  # Set all agents to the goal phase to execute the planned paths
    # app.set_phase_agent(ag1, phase="goal")

    time.sleep(5)

    # app.set_phase_agent(ag2, phase="goal")

    while True:
        time.sleep(1)
    

if __name__ == '__main__':
    motion_planning_task()