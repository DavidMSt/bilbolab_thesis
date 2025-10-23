import enum

import utils.board as board
from utils import bytes as bt

RegInputDisable = 0x00
RegPullUp = 0x03
RegPullDown = 0x04
RegDir = 0x07
RegData = 0x08
RegReset = 0x7D

data = 0

SX1508_I2C_ADDRESS = 0x20


class SX1508_GPIO_MODE(enum.Enum):
    INPUT = 0
    OUTPUT = 1


class SX1508:
    address: int
    _i2c: board.I2C

    data: int

    def __init__(self, address: int = SX1508_I2C_ADDRESS, reset=False):
        self.address = address
        self._i2c = board.I2C()
        if reset:
            self.reset()
        self.data = 0

    def configureGPIO(self, gpio: int, mode, pullup: bool = False, pulldown: bool = False):
        # set the Mode
        self._setGPIOMode(gpio, mode)

        # Set the pullup and pulldown registers
        self._setPullup(gpio, pullup)
        self._setPulldown(gpio, pulldown)

    def deinitGPIO(self, gpio):
        self.configureGPIO(gpio, mode=SX1508_GPIO_MODE.INPUT)

    def writeGPIO(self, gpio, state):
        self.data = bt.changeBit(self.data, gpio, state)
        self._writeReg(RegData, self.data)

    def toggleGPIO(self, gpio):
        self.data = bt.toggleBit(self.data, gpio)
        self._writeReg(RegData, self.data)

    def reset(self):
        self._writeReg(RegReset, 0x12)
        self._writeReg(RegReset, 0x34)

    def _setGPIOMode(self, gpio, mode: SX1508_GPIO_MODE):
        """
        Set the register RegInputDisable. A '0' indicates an INPUT, a '1' indicates an OUTPUT
        """
        if isinstance(mode, SX1508_GPIO_MODE):
            mode = mode.value

        # Read the register RegInputDisable
        # data = self._readReg(RegInputDisable)
        # # Manipulate the content
        # data = bt.changeBit(data, gpio, mode)
        # self._writeReg(RegInputDisable, data)

        data = self._readReg(RegDir)
        data = bt.changeBit(data, gpio, not mode)  # here 0 is an output and 1 is an input
        self._writeReg(RegDir, data)

    def _setPullup(self, gpio, mode: bool):
        """
        Set the pullup for the GPIO pin. '0' disables the pullup, '1' enables the pullup
        """
        data = self._readReg(RegPullUp)
        data = bt.changeBit(data, gpio, mode)
        self._writeReg(RegPullUp, data)

    def _setPulldown(self, gpio, mode: bool):
        """
        Set the pullup for the GPIO pin. '0' disables the pullup, '1' enables the pullup
        """
        data = self._readReg(RegPullDown)
        data = bt.changeBit(data, gpio, mode)
        self._writeReg(RegPullDown, data)

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
