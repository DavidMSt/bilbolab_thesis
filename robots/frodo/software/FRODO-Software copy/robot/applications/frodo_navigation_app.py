import time

from robot.control.frodo_control import FRODO_Control_Mode
from robot.control.frodo_joystick_control import StandaloneJoystickControl
from robot.frodo import FRODO

if __name__ == '__main__':
    frodo = FRODO()
    frodo.init()
    frodo.start()
    frodo.control.setMode(FRODO_Control_Mode.NAVIGATION)

    while True:
        print("Start Mov 1")
        frodo.control.navigation.addMovement(dphi=1.57, radius=300, vtime=5)
        print("Start Mov 2")
        frodo.control.navigation.addMovement(dphi=-1.57, radius=0)
        print("Start Mov 3")
        frodo.control.navigation.addMovement(dphi=0, radius=0)
        time.sleep(15)