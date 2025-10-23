import time

from robot.control.frodo_control import FRODO_Control_Mode
from robot.control.frodo_joystick_control import StandaloneJoystickControl
from robot.frodo import FRODO
from robot.sensing.camera.pycamera import PyCameraStreamer
from robot.utilities.video_streamer.video_streamer import VideoStreamer

if __name__ == '__main__':
    frodo = FRODO()
    frodo.init()
    frodo.start()

    frodo.control.setMode(FRODO_Control_Mode.EXTERNAL)

    while True:
        time.sleep(1)
