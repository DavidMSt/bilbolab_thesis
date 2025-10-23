import enum

import board
from utils import bytes_utils as bt

RegInputDisable = 0x00
RegPullUp = 0x03
RegPullDown = 0x04
RegDir = 0x07
RegData = 0x08
RegReset = 0x7D
SX1508_I2C_ADDRESS = 0x20

data = 0


class SX1508_GPIO_MODE(enum.Enum):
    INPUT = 0
    OUTPUT = 1


class SX1508:
    address: int
    _i2c: board.I2C


    def __init__(self, address: int = SX1508_I2C_ADDRESS, reset=False):
        self.address = address
        self._i2c = board.I2C()
        if reset:
            self.reset()

    def configureGPIO(self, gpio: int, mode, pullup: bool = False, pulldown: bool = False):
        # set the Mode
        self._setGPIOMode(gpio, mode)

        # Set the pullup and pulldown registers
        self._setPullup(gpio, pullup)
        self._setPulldown(gpio, pulldown)

    def deinitGPIO(self, gpio):
        self.configureGPIO(gpio, mode=SX1508_GPIO_MODE.INPUT)

    def writeGPIO(self, gpio, state):
        global data
        data = bt.changeBit(data, gpio, state)
        self._writeReg(RegData, data)

    def toggleGPIO(self, gpio):
        global data
        data = bt.toggleBit(data, gpio)
        self._writeReg(RegData, data)

    def reset(self):
        self._writeReg(RegReset, 0x12)
        self._writeReg(RegReset, 0x34)

    def _setGPIOMode(self, gpio, mode: SX1508_GPIO_MODE):
        """
        Set the register RegInputDisable. A '0' indicates an INPUT, a '1' indicates an OUTPUT
        """
        # print("Set GPIO Mode")
        if isinstance(mode, SX1508_GPIO_MODE):
            mode = mode.value

        # Read the register RegInputDisable
        # data = self._readReg(RegInputDisable)
        # # Manipulate the content
        # data = bt.changeBit(data, gpio, mode)
        # self._writeReg(RegInputDisable, data)

        direction_data = self._readReg(RegDir)
        direction_data = bt.changeBit(direction_data, gpio, not mode)  # here 0 is an output and 1 is an input
        self._writeReg(RegDir, direction_data)

    def _setPullup(self, gpio, mode: bool):
        """
        Set the pullup for the GPIO pin. '0' disables the pullup, '1' enables the pullup
        """
        pullup_data = self._readReg(RegPullUp)
        pullup_data = bt.changeBit(pullup_data, gpio, mode)
        self._writeReg(RegPullUp, pullup_data)

    def _setPulldown(self, gpio, mode: bool):
        """
        Set the pullup for the GPIO pin. '0' disables the pullup, '1' enables the pullup
        """
        pulldown_data = self._readReg(RegPullDown)
        pulldown_data = bt.changeBit(pulldown_data, gpio, mode)
        self._writeReg(RegPullDown, pulldown_data)

    def _writeReg(self, reg, data):
        """

        """
        if not isinstance(reg, bytes):
            reg = bt.bytes_(reg)

        if not isinstance(data, bytes):
            data = bt.bytes_(data)

        x = reg + data
        self._i2c.writeto(address=self.address, buffer=reg + data)

    def _readReg(self, reg):
        """

        """
        if not isinstance(reg, bytes):
            reg = bt.bytes_(reg)
        data = [0]
        self._i2c.writeto_then_readfrom(address=self.address, buffer_out=reg, buffer_in=data)

        return data[0]
