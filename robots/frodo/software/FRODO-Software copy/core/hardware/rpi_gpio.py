from RPi import GPIO
from utils.exit import ExitHandler
import atexit

INIT = False


def init():
    GPIO.setmode(GPIO.BCM)


def setup(pin, direction, pull_up_down):
    GPIO.setup(pin, direction, pull_up_down)


def write(pin, value):
    GPIO.output(pin, value)


def toggle(pin):
    GPIO.output(pin, not GPIO.input(pin))


def read(pin):
    return GPIO.input(pin)


def add_event_detect(pin, flanks: int, callback, bouncetime):
    GPIO.add_event_detect(pin, flanks, callback, bouncetime)


def close(*args, **kwargs):
    GPIO.cleanup()


init()
exitHandler = ExitHandler()
exitHandler.register(close)
atexit.register(close)