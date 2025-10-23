import threading
import time

from robot.sensing.camera.pycamera import PyCameraStreamer
from robot.utilities.video_streamer.video_streamer import VideoStreamer
from utils.exit import ExitHandler
from utils.joystick.joystick import JoystickManager, JoystickManager_Callbacks, Joystick
from utils.logging_utils import Logger
from robot.frodo import FRODO

# from robot.setup import readSettings

logger = Logger("JoystickControl")
logger.setLevel('INFO')


# ======================================================================================================================
class StandaloneJoystickControl:
    joystick_manager: JoystickManager
    joystick: (None, Joystick)
    frodo: FRODO

    _exit: bool = False
    _thread: threading.Thread

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, frodo: FRODO):
        self.frodo = frodo
        self.joystick_manager = JoystickManager()

        self.joystick_manager.callbacks.new_joystick.register(self._newJoystick_callback)
        self.joystick_manager.callbacks.joystick_disconnected.register(self._joystickDisconnected_callback)

        self.joystick = None
        self.exit = ExitHandler()
        self.exit.register(self.close)
        self._thread = threading.Thread(target=self._task, daemon=True)

    # ------------------------------------------------------------------------------------------------------------------
    def init(self):
        self.joystick_manager.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.joystick_manager.start()
        self._thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self._exit = True
        if self._thread.is_alive():
            self._thread.join()

    # ------------------------------------------------------------------------------------------------------------------
    def _task(self):
        while not self._exit:
            self._updateInputs()
            time.sleep(0.01)

    # ------------------------------------------------------------------------------------------------------------------
    def _updateInputs(self):
        if self.joystick is None:
            return

        # Read the controller inputs
        axis_forward = -self.joystick.axis[1]  # Forward/Backward
        axis_turn = self.joystick.axis[3]  # Turning

        # Compute initial wheel speeds
        speed_left = axis_forward + axis_turn
        speed_right = axis_forward - axis_turn

        # Find the maximum absolute value
        max_speed = max(abs(speed_left), abs(speed_right))

        # Scale speeds if they exceed the allowed range
        if max_speed > 1:
            speed_left /= max_speed
            speed_right /= max_speed

        # Set the speeds
        self.frodo.control.setSpeed(speed_left, speed_right)

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

    # ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    frodo = FRODO()
    joystick_control = StandaloneJoystickControl(frodo)
    frodo.init()
    joystick_control.init()
    frodo.start()
    joystick_control.start()

    streamer = VideoStreamer()
    streamer.image_fetcher = frodo.sensors.aruco_detector.getOverlayFrame
    streamer.start()

    while True:
        time.sleep(10)
