from extensions.simulation.src.core.environment import Object
import extensions.simulation.src.core.spaces as spaces

class ObstacleObject(Object):
    object_type = "obstacle"
    static = True

    def __init__(self, object_id, position, orientation, length = 2.0, height = 1.0 , width = 0.5, space=None):
        self.length = length
        self.width = width
        self.height = height
        self.space = spaces.Space2D()

        super().__init__(object_id=object_id, space=self.space)

        for k, v in position.items():
            self.setPosition(**{k: v})

        # self.setOrientation(orientation)
        self.setConfiguration(dimension='psi', value = orientation)

    # def setOrientation(self, psi) -> None:
    #     self.setConfiguration(dimension='psi', value=psi)

    # def getOrientation(self):
    #     return self.configuration['psi'].value