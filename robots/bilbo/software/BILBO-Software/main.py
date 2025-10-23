import time
from robot.bilbo import BILBO
from core.utils.logging_utils import setLoggerLevel, Logger
from robot.logging.bilbo_sample import BILBO_Sample


def main():
    bilbo = BILBO(reset_stm32=False)
    bilbo.init()
    bilbo.start()

    # def sample_callback(sample: BILBO_Sample):
    #     print(f"Theta: {np.degrees(sample.lowlevel.estimation.state.theta):.1f}")
    #
    # bilbo.core.events.sample.on(callback=sample_callback)


    while True:
        time.sleep(1)



if __name__ == '__main__':
    main()
