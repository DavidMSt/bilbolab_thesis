import time

from archive.robot_old.control.frodo_control import FRODO_Control_Mode
from archive.robot_old.frodo import FRODO

if __name__ == '__main__':
    frodo = FRODO()
    frodo.init()
    frodo.start()

    frodo.control.setMode(FRODO_Control_Mode.EXTERNAL)

    while True:
        time.sleep(1)
