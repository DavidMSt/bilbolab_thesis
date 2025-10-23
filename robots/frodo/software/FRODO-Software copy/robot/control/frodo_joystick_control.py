import threading
import time

import numpy as np


from utils.exit import ExitHandler
from utils.joystick.joystick import JoystickManager, JoystickManager_Callbacks, Joystick
from utils.logging_utils import Logger

# from robot.setup import readSettings

logger = Logger("JoystickControl")
logger.setLevel('INFO')


# ======================================================================================================================
class StandaloneJoystickControl:
    joystick_manager: JoystickManager
    joystick: (None, Joystick)

    _exit: bool = False

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.joystick_manager = JoystickManager()

        self.joystick_manager.callbacks.new_joystick.register(self._newJoystick_callback)
        self.joystick_manager.callbacks.joystick_disconnected.register(self._joystickDisconnected_callback)

        self.joystick = None
        self.exit = ExitHandler()
        self.exit.register(self.close)

    # ------------------------------------------------------------------------------------------------------------------
    def init(self):
        self.joystick_manager.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.joystick_manager.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self._exit = True

    # ------------------------------------------------------------------------------------------------------------------
    def getInputs(self):
        ''' Read controller inputs, translate to motor inputs and return those '''
        if self.joystick is None:
            return 0.0, 0.0

        # Read the controller inputs
        axis_forward = -self.joystick.axis[1]  # Forward/Backward
        axis_turn = self.joystick.axis[3]  # Turning

        def map_input(x, factor=10.0):
            """
            Maps input x in [0,1] to another value in [0,1] using an exponential curve.
            """
            sign = np.sign(x)
            x = abs(x)
            return sign*(np.exp(x * np.log(factor)) - 1) / (factor - 1)

        axis_turn = map_input(axis_turn)
        
        sum_axis = abs(axis_forward) + abs(axis_turn)

        if sum_axis > 1:
            axis_forward = axis_forward/sum_axis
            axis_turn = axis_turn/sum_axis
            # Compute initial wheel speeds
        speed_left = axis_forward + axis_turn
        speed_right = axis_forward - axis_turn

        return speed_left, speed_right 
    
    # ------------------------------------------------------------------------------------------------------------------
    def _newJoystick_callback(self, joystick, *args, **kwargs):
        if self.joystick is None:
            self.joystick = joystick

        logger.info("Joystick connected and assigned")

    # ------------------------------------------------------------------------------------------------------------------
    def _joystickDisconnected_callback(self, joystick, *args, **kwargs):
        if joystick == self.joystick:
            self.joystick = None

        logger.info("Joystick disconnected")
