# third party
import numpy as np
import time
import math
from typing import overload, override, Type

# bilbolab
from applications.FRODO.simulation.frodo_simulation import FRODO_Simulation, FRODO_ENVIRONMENT_ACTIONS, FrodoEnvironment, FRODO_Static, FRODO_Simulation_Events
from extensions.simulation.src.objects.frodo.frodo import FRODO_DynamicAgent
from applications.FRODO.simulation.frodo_simulation_utils import frodo_virtual_agent_colors
import extensions.simulation.src.core as core
import extensions.simulation.src.core.environment as core_env
from applications.FRODO.definitions import get_simulated_agent_definition_by_id
from core.utils.dataclass_utils import update_dataclass_from_dict
from extensions.cli.cli import CommandSet, Command, CommandArgument

# master thesis
from master_thesis.general.general_agents import FRODOGeneralAgent, FRODO_General_Config, FRODO_GeneralAgent_CommandSet
from master_thesis.general.general_obstacles import ObstacleObject

# Global registries
SIMULATED_AGENTS: dict[str, FRODOGeneralAgent] = {}
SIMULATED_STATICS: dict[str, FRODO_Static] = {}

# ======================================================================================================================
USE_AGENT_DEFINITIONS = True

class FRODO_General_CommandSet(CommandSet):
    def __init__(self, sim: "FRODO_general_Simulation"):
        super().__init__(name='simulation')
        self.sim = sim

        # ------------------------------------------------------------------
        # LIST
        # ------------------------------------------------------------------
        self.addCommand(Command(
            name='list',
            description='List all agents and statics',
            arguments=[],
            function=lambda: self.sim.logger.info(
                f"Agents: {list(SIMULATED_AGENTS.keys())}\nStatics: {list(SIMULATED_STATICS.keys())}"
            )
        ))

        # ------------------------------------------------------------------
        # ADD AGENT  (GeneralAgent only)
        # ------------------------------------------------------------------
        self.addCommand(Command(
            name='add_agent',
            description='Add a general agent',
            arguments=[
                CommandArgument('agent_id', type=str, description='Agent ID'),
                CommandArgument('x', type=float, description='start x', optional=True, default=0.0),
                CommandArgument('y', type=float, description='start y', optional=True, default=0.0),
                CommandArgument('psi', type=float, description='start orientation', optional=True, default=0.0),
                CommandArgument('color', type=list, description='RGB color', optional=True, default=None),
            ],
            function=self._add_general_agent
        ))

        # ------------------------------------------------------------------
        # REMOVE AGENT
        # ------------------------------------------------------------------
        self.addCommand(Command(
            name='remove_agent',
            description='Remove a general agent',
            allow_positionals=True,
            arguments=[
                CommandArgument('agent', type=str, description='Agent ID'),
            ],
            function=self.sim.remove_agent
        ))

        # ------------------------------------------------------------------
        # ADD STATIC
        # ------------------------------------------------------------------
        self.addCommand(Command(
            name='add_static',
            description='Add a static object',
            arguments=[
                CommandArgument('static_id', type=str),
                CommandArgument('x', type=float, optional=True, default=None),
                CommandArgument('y', type=float, optional=True, default=None),
                CommandArgument('psi', type=float, optional=True, default=None),
                CommandArgument('size', type=float, optional=True, default=0.2),
            ],
            function=self.sim.new_static
        ))

    # === private ---------------------------------------------------------
    def _add_general_agent(self, agent_id, x=0.0, y=0.0, psi=0.0, color=None):
        config = FRODO_General_Config(color=color)
        return self.sim.new_agent(
            agent_id=agent_id,
            config=config,
            start_config=[x, y, psi],
        )

class FrodoGeneralEnvironment(FrodoEnvironment):
    def __init__(self, Ts, run_mode, *args, **kwargs):
        self.space = core.spaces.Space2D()
        self._obstacles = []            

        super().__init__(Ts=Ts, run_mode=run_mode, *args, **kwargs)
        # # Call core environment init directly (skip FrodoEnvironment’s extra input registration)
        # core_env.Environment.__init__(self, Ts=Ts, run_mode=run_mode)

        # # adding input as extra phase here
        # core.scheduling.Action(
        #     action_id=FRODO_ENVIRONMENT_ACTIONS.INPUT,
        #     object=self,
        #     function=self.action_input,
        #     priority=34,  # one tick after the last phase of FrodoEnvironment
        #     parent=self.scheduling.actions['objects']
        # )

        core.scheduling.Action(action_id=FRODO_ENVIRONMENT_ACTIONS.COLLISION,
                        object=self,
                        function=self.collision_checking,
                        priority=65,
                        parent=self.scheduling.actions['objects'])

    def set_limits(self, limits: tuple[tuple[int, int], ...] = ((-3, 3), (-3, 3)), wrapping = [False, False]):
        pos_dim = self.space.dimensions[0] # Get the first dimension of the space (E(2) vector)
        pos_dim.kwargs['wrapping'] = wrapping
        pos_dim.limits = limits

    def action_input(self):
        # print(f"=== ENV INPUT PHASE @ tick {self.scheduling.tick}") # TODO: enabling this shows that this phase is called twice? bug? 
        self.logger.debug(f"{self.scheduling.tick}: Action Frodo Input")

    def collision_checking(self):

        for object_key in self.objects:
            print(object_key)
            print(self.objects[object_key].configuration)

    @property
    def limits(self) ->list[list[float]]:
        return self.space.dimensions[0].limits
    
    @property
    def obstacles(self):
        return self._obstacles

class FRODO_general_Simulation(FRODO_Simulation):

    environment: FrodoEnvironment

    cli: FRODO_General_CommandSet | None = None

    agents: dict[str, FRODOGeneralAgent]
    statics: dict[str, FRODO_Static]

    events: FRODO_Simulation_Events

    def __init__(self, Ts=0.1, use_web_interface: bool = False, limits: tuple[tuple[int, int], ...] = ((-3, 3), (-3, 3)), env = FrodoGeneralEnvironment):
        
        super().__init__(Ts)

        # override standard bilbo environment with my custom version
        self.environment = env(Ts=Ts, run_mode='rt')
        self.agents = SIMULATED_AGENTS
        self.statics = SIMULATED_STATICS
        self.cli = FRODO_General_CommandSet(self)

        # check if limits are valid
        for i, limit in enumerate(limits):
            if limit[0] > limit[1]:
                self.logger.error(f"Invalid environment limits for dimension: {i}: {limit[0]} > {limit[1]}")

        self.environment.set_limits(limits = limits)

        self.obstacles = {}

    def check_collisions(self):
        ...
        
    def addVirtualObstacle(self, obstacle_id: str, obstacle_class: type[ObstacleObject] = ObstacleObject, **obstacle_kwargs) -> ObstacleObject:
        # log
        self.logger.info(f"Add Virtual Obstacle: {obstacle_id}")

        # instantiate obstacle
        obstacle = obstacle_class(object_id=obstacle_id, **obstacle_kwargs)
        
        # register into environment
        self.environment.addObject(obstacle)

        # store locally
        self.obstacles[obstacle_id] = obstacle

        return obstacle
    
    @override
    def add_agent(self,
                agent: FRODOGeneralAgent) -> FRODOGeneralAgent:

        global SIMULATED_AGENTS
        SIMULATED_AGENTS[agent.agent_id] = agent

        # Enforce Ts on agent
        agent.scheduling.Ts = self.Ts
        agent.dynamics.Ts = self.Ts

        self.environment.addAgent(agent)
        self.logger.info(f"Simulated agent {agent.agent_id} added")
        self.cli.addChild(agent.cli)

        self.events.new_agent.set(agent)

        return agent

    @override
    def new_agent(self,
                  agent_id: str,
                  config: FRODO_General_Config | None = None,
                  agent_class: type[FRODOGeneralAgent] = FRODOGeneralAgent,
                  *args,
                  **kwargs) -> FRODOGeneralAgent | None:

        if agent_id in SIMULATED_AGENTS:
            self.logger.warning(f"Simulated agent {agent_id} already exists. Cannot add it again")
            return None

        if USE_AGENT_DEFINITIONS:
            agent_definition = get_simulated_agent_definition_by_id(agent_id)
            if agent_definition is None:
                self.logger.warning(
                    f"Agent definition for {agent_id} not found. Cannot add it. "
                    f"Either disable the use of predefined agent definitions by setting USE_AGENT_DEFINITIONS to False "
                    f"or define the agent definition in the definitions.py file.")
                return None

            config = FRODO_General_Config()

        if config is None:
            config = FRODO_General_Config()

        update_dataclass_from_dict(config, kwargs)

        agent = agent_class(
            agent_id=agent_id,
            Ts=self.Ts,
            config=config,
            **kwargs
            )
        
        self.add_agent(agent)
        return agent

    # # TODO: check if this is even necessary or if i should use the add_agent of bilbolab from now on
    # def addVirtualAgent(self, id: str,
    #                     agent_class: type[FRODOGeneralAgent] = FRODOGeneralAgent,
    #                     **agent_kwargs):

    #     if not issubclass(agent_class, FRODOGeneralAgent):
    #         raise TypeError(f"Only FRODOGeneralAgent allowed, got {agent_class.__name__}")

    #     # Instantiate your custom agent
    #     agent = agent_class(agent_id=id, Ts=self.environment.Ts, **agent_kwargs)

    #     # register with original BilboLab
    #     super().add_agent(agent)

    #     # additional registry (optional)
    #     self.agents[id] = agent

    #     return agent

    # def addExistingVirtualAgent(self, agent: FRODOGeneralAgent, logger_level: str = 'INFO'):
    #     # same logger message as from the addVirtual methods from parent class 
    #     self.logger.info(f"Add Virtual Agent: {agent.agent_id}")
    #     # add the agent to the simulation
    #     self.agents[agent.agent_id] = agent

    #     # add agent to environment
    #     self.environment.addObject(agent)
    #     # configure the logger
    #     agent.logger.setLevel(logger_level)

    #     # extract id string from the agent
    #     id = agent.agent_id

    #     return self.agents[id]
    
    def set_phase_all_agents(self, phase :str):
        for agent in self.agents.values():
            agent.change_phase(phase)

def main():
    # === Simulation setup ===
    env_size_half = 10
    sim = FRODO_general_Simulation(
        Ts=0.1,
        use_web_interface=True,
        limits=((-env_size_half, env_size_half), (-env_size_half, env_size_half)),
    )
    sim.init()

    # === Initial agent poses ===
    start_a = [0.0, 0.0, 0.0]
    start_b = [1.0, 0.5, 0.0]

    # === Colors (GUI) ===
    color_ag1 = (0.7, 0, 0)
    color_ag2 = (0, 0, 0.7)

    # === Add agents using new_general_agent ===
    config = FRODO_General_Config(color=color_ag1)
    agent_a = sim.new_agent(
        agent_id="vfrodo1",
        agent_class=FRODOGeneralAgent,
        start_config=start_a,
        config=config,
    )

    config = FRODO_General_Config(color=color_ag2)
    agent_b = sim.new_agent(
        agent_id="vfrodo2",
        agent_class=FRODOGeneralAgent,
        start_config=start_b,
        config=config,
    )

    # === Add virtual obstacle (optional) ===
    # sim.addVirtualObstacle(
    #     obstacle_id="wall1",
    #     position={"x": 3.0, "y": -1.0},
    #     orientation=90,
    #     length=2.0,
    #     width=0.5,
    #     height=1.0,
    # )

    # === Example: input phases for scripted motion (optional) ===
    # inputs = tuple([np.array([1.0, 0.0]) for _ in range(100)])
    # durations = tuple([1] * len(inputs))
    # agent_a.add_input_phase("forward", inputs=inputs, durations=durations, delta_t=0.4)
    # sim.set_phase_all_agents("forward")

    # === Start simulation ===
    sim.start()

    # === Infinite keep-alive ===
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
    

