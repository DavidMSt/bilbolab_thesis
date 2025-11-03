import time

from core.utils.logging_utils import Logger
from core.utils.time import TimeoutTimer

if __name__ == '__main__':
    logger = Logger("TIMEOUT")
    timer = TimeoutTimer(2, lambda: logger.info("timeout"))
    time.sleep(5)
    logger.info("start timer")
    timer.start()

    time.sleep(100)