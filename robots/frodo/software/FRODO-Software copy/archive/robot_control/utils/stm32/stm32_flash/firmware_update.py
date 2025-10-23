import RPi.GPIO as GPIO
import serial
import time
import board
import busio

pin_nrst = 17
uart5 = "/dev/ttyAMA1"

i2c = board.I2C()


def init():
    # Initialize the GPIOs
    GPIO.setwarnings(False)
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin_nrst, GPIO.OUT)

    i2c.writeto(address=0x20, buffer=b'\x00\x00')
    i2c.writeto(address=0x20, buffer=b'\x03\x00')
    i2c.writeto(address=0x20, buffer=b'\x04\x00')
    i2c.writeto(address=0x20, buffer=b'\x07\xFE')


def set_boot0(state):
    if state:
        i2c.writeto(address=0x20, buffer=b'\x08\x01')
    else:
        i2c.writeto(address=0x20, buffer=bytes([0x7D, 0x12]))
        i2c.writeto(address=0x20, buffer=bytes([0x7D, 0x34]))


def reset(reset_time):
    time.sleep(0.25)
    GPIO.output(pin_nrst, 1)
    time.sleep(reset_time)
    GPIO.output(pin_nrst, 0)
    time.sleep(0.01)


def check_bootloader():
    check = False
    uart = serial.Serial(uart5, baudrate=57600, timeout=1)

    for i in range(0, 2):
        uart.write(b'\x7F')
        answer = uart.read(1)
        if answer == b'\x1F' or answer == b'\x79':
            check = True
            break
    uart.close()
    return check


def exit_boot():
    set_boot0(False)
    reset(0.5)
    GPIO.cleanup()
    print("Exit Bootloader")


if __name__ == "__main__":
    print("Entering STM32 UART bootloader...")
    success = False
    init()

    for i in range(0, 2):
        set_boot0(True)
        reset(0.1)
        success = check_bootloader()
        if success:
            time.sleep(2)
            break
        else:
            exit_boot()

    if success:
        print("Success")
        exit_boot()
    else:
        print("[Error] Cannot connect to Bootloader! Please check connections and try again")