from extensions.simulation.src.core.environment import Object
import extensions.simulation.src.core.spaces as spaces
from core.utils.states import State
from dataclasses import dataclass

@dataclass(frozen= True, slots= True)
class Obstacle_Config:
    length: float = 2.0
    height: float = 1.0
    width: float = 0.5

# @dataclass(frozen=True, slots= True)
# class Obstacle_State(State):
#     x: float
#     y: float
#     psi: float

class GeneralObstacle(Object):
    object_type = "obstacle"
    static = True

    def __init__(self, obstacle_id: str, x:float, y:float, config: Obstacle_Config | None = None, *args):
        self.space = spaces.Space2D()         # <-- creates a 2D space

        if config is None:
            config = Obstacle_Config()


        # define geometry / footprint
        self.space.dimensions[0].limits = [
            [-config.length/2, config.length/2],
            [-config.width/2,  config.width/2]
            ]


        super().__init__(object_id=obstacle_id, space=self.space)
        self.obstacle_id = obstacle_id
        self.setPosition(x,y)
