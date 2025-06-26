import time

from robot.bilbo import BILBO
from core.utils.logging_utils import setLoggerLevel, Logger
from robot.control.bilbo_control_data import BILBO_Control_Mode

setLoggerLevel('wifi', 'ERROR')

logger = Logger('main')
logger.setLevel('DEBUG')


def main():
    bilbo = BILBO(reset_stm32=False)
    bilbo.init()
    bilbo.start()

    bilbo.control.setMode(BILBO_Control_Mode.BALANCING)
    time.sleep(2)

    trajectory = bilbo.experiment_handler.generateTestTrajectory(id=1, time=5, frequency=3, gain=0.2)
    bilbo.experiment_handler.runTrajectory(trajectory)


    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
