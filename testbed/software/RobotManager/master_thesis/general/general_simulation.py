import numpy as np
import time
import math
from typing import overload

# bilbolab
from bilbolab.applications.FRODO.simulation.frodo_simulation import FRODO_Simulation, FRODO_ENVIRONMENT_ACTIONS, FrodoEnvironment
from bilbolab.extensions.simulation.src.objects.frodo.frodo import FRODO_DynamicAgent
from bilbolab.applications.FRODO.simulation.frodo_simulation_utils import frodo_virtual_agent_colors
import bilbolab.extensions.simulation.src.core as core
import bilbolab.extensions.simulation.src.core.environment as core_env

# master thesis
from general.general_agents import FRODOGeneralAgent
from general.general_obstacles import ObstacleObject



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

        core.scheduling.Action(action_id=FRODO_ENVIRONMENT_ACTIONS.COLLISION_CHECKING,
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
        print('collisions are being checked!')
        for object_key in self.objects:
            print(object_key)
            print(self.objects[object_key].configuration)

    @property
    def limits(self) ->list[list[float]]:
        return self.space.dimensions[0].limits
    
    @property
    def obstacles(self):
        print('obstalces not imlpemented')
        return self._obstacles

class FRODO_general_Simulation(FRODO_Simulation):

    environment: FrodoEnvironment
    agents: dict[str, FRODO_DynamicAgent]
    obstacles: dict 

    def __init__(self, Ts=0.1, use_web_interface: bool = False, limits: tuple[tuple[int, int], ...] = ((-3, 3), (-3, 3)), env = FrodoGeneralEnvironment):
        
        super().__init__(Ts)

        # override standard bilbo environment with my custom version
        self.environment = env(Ts=Ts, run_mode='rt')

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

    # TODO: check if this is even necessary or if i should use the add_agent of bilbolab from now on
    def addVirtualAgent(self, id: str,
                        agent_class: type[FRODOGeneralAgent] = FRODOGeneralAgent,
                        **agent_kwargs):

        if not issubclass(agent_class, FRODOGeneralAgent):
            raise TypeError(f"Only FRODOGeneralAgent allowed, got {agent_class.__name__}")

        # Instantiate your custom agent
        agent = agent_class(agent_id=id, Ts=self.environment.Ts, **agent_kwargs)

        # register with original BilboLab
        super().add_agent(agent)

        # additional registry (optional)
        self.agents[id] = agent

        return agent

    def addExistingVirtualAgent(self, agent: FRODOGeneralAgent, logger_level: str = 'INFO'):
        # same logger message as from the addVirtual methods from parent class 
        self.logger.info(f"Add Virtual Agent: {agent.agent_id}")
        # add the agent to the simulation
        self.agents[agent.agent_id] = agent

        # add agent to environment
        self.environment.addObject(agent)
        # configure the logger
        agent.logger.setLevel(logger_level)

        # extract id string from the agent
        id = agent.agent_id

        # if self.web_interface is not None:
            
        #     # Add the agent to the plotter
        #     group = self.virtual_agents_plotting_group.add_group(id)

            
        #     group.add_vision_agent(
        #         id=id,
        #         position=[0, 0],
        #         psi=0,
        #         vision_radius=self.agents[id].view_range,
        #         vision_fov=self.agents[id].fov,
        #         color=frodo_virtual_agent_colors[id] if id in frodo_virtual_agent_colors else [0.5, 0.5, 0.5]
        #     )
        return self.agents[id]
    
    def set_phase_all_agents(self, phase :str):
        for agent in self.agents.values():
            agent.change_phase(phase)

def main():
    # create simulation (no web gui)
    env_size_half = 10
    sim = FRODO_general_Simulation(Ts=0.1, use_web_interface=True, limits=((-env_size_half, env_size_half), (-env_size_half, env_size_half)))
    sim.init()

    # minimal agent start poses: [x, y, yaw]
    start_a = [0.0, 0.0, 0.0]
    start_b = [1.0, 0.5, 0.0]

    # set colors for the web gui
    color_ag1 = [0.7, 0, 0]
    color_ag2 = [0, 0, 0.7]

     # ---------- Option A: using simulations add virtual agent ----------
    test_agent_a = sim.addVirtualAgent(id = "frodo1_v", 
                                        agent_class= FRODOGeneralAgent, 
                                        start_config = start_a,
                                        dt = sim.environment.Ts,
                                        vision_radius=1.5,
                                        vision_fov=math.radians(120),
                                        color=color_ag1
                                        )


    # # ---------- Option B: Adding the agent manually - TODO: Not sure if this should be done at all ----------
    # agent_id = 'frodo2_v'
    # test_agent_b = FRODOGeneralAgent(start_config = start_b, fov_deg=360, view_range=1.5, agent_id = agent_id, Ts=sim.environment.Ts, color = color_ag2) 
    
    # sim.addExistingVirtualAgent(test_agent_b)

    # sim.addVirtualObstacle(
    #     obstacle_id="wall1",
    #     position={"x": 3.0, "y": -1.0},
    #     orientation = 90,
    #     length= 2.0,
    #     width = 0.5,
    #     height = 1.0,
    # )

    # # create test_input phase
    # inputs_a = tuple([np.array([1.0, 0.0]) for _ in range(1000)])
    # inputs_b = tuple([np.array([-1.0, 0.0]) for _ in range(1000)])
    # durations = tuple([1] * len(inputs_a))



    # # pick different delta t -> one step for phase now equals 4 steps in the simulation
    # test_agent_a.add_input_phase('test_phase', inputs = inputs_a, durations= durations, delta_t=0.4)
    # test_agent_b.add_input_phase('test_phase', inputs = inputs_a, durations= durations, delta_t=0.4)
    # # test_agent_b.change_phase('test_phase', reset= True)
    
    # sim.start()
    
    # sim.set_phase_all_agents('test_phase')

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
