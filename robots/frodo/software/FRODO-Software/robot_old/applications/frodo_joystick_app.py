import time

from robot_old.control.frodo_control import FRODO_Control_Mode
from robot_old.control.frodo_joystick_control import StandaloneJoystickControl
from robot_old.frodo import FRODO
from robot_old.sensing.camera.pycamera import PyCameraStreamer
from robot_old.utilities.video_streamer.video_streamer import VideoStreamer

if __name__ == '__main__':
    frodo = FRODO()
    frodo.init()
    frodo.start()

    frodo.control.setMode(FRODO_Control_Mode.EXTERNAL)

    while True:
        time.sleep(1)
