import time

from RPi import GPIO

pin_nrst = 17


def reset(reset_time):
    GPIO.setwarnings(False)
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin_nrst, GPIO.OUT)

    time.sleep(0.25)
    GPIO.output(pin_nrst, 1)
    time.sleep(reset_time)
    GPIO.output(pin_nrst, 0)
    time.sleep(0.01)

    GPIO.cleanup()
