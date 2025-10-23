import RPi.GPIO as GPIO
import serial
# import rc_hardware
import time

pin_nrst = 27
pin_boot0 = 22
pin_led = 24

def init_boot():
    GPIO.setwarnings(False)
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin_nrst, GPIO.OUT)
    GPIO.setup(pin_boot0, GPIO.OUT)


def reset(reset_time):
    time.sleep(reset_time)
    GPIO.output(pin_nrst, 1)
    time.sleep(reset_time)
    GPIO.output(pin_nrst, 0)
    time.sleep(reset_time)


def set_boot0(state: bool):
    if state is True:
        GPIO.output(pin_boot0, 1)
    else:
        GPIO.output(pin_boot0, 0)


if __name__ == "__main__":
    print("Exit STM32 bootloader...")
    init_boot()
    set_boot0(False)
    reset(0.5)
    GPIO.cleanup()
    print("Success")
