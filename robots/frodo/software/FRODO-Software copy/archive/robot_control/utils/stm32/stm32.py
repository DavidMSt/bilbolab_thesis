import time
import RPi.GPIO as GPIO

pin_nrst = 17


def stm32_reset(reset_time):
    GPIO.setwarnings(False)
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin_nrst, GPIO.OUT)

    GPIO.output(pin_nrst, 1)
    time.sleep(reset_time)
    GPIO.output(pin_nrst, 0)
    GPIO.cleanup()
