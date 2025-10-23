from core.communication.i2c.i2c import I2C_Interface
from core.communication.spi.spi import SPI_Interface
from core.interfaces.wifi_interface import WIFI_Interface
from core.interfaces.data_link import Command
from core.hardware.sx1508 import SX1508, SX1508_GPIO_MODE
import board.settings as board_parameters
from core.communication.serial.serial_interface import Serial_Interface
from board.board_utils import reset_uart
from board.io_extension.io_extension import RobotControl_IO_Extension
from utils.debug import *

class RobotControl_Board:
    wifi_interface: WIFI_Interface
    spi_interface: SPI_Interface
    serial_interface: Serial_Interface
    i2c_interface: I2C_Interface

    io_extension: RobotControl_IO_Extension

    # === INIT =========================================================================================================
    def __init__(self, device_class: str = 'board', device_type: str = 'RobotControl', device_revision: str = 'v3',
                 device_id: str = 0, device_name: str = 'c4'):
        self.wifi_interface = WIFI_Interface('wifi', device_class=device_class, device_type=device_type,
                                             device_revision=device_revision, device_id=device_id,
                                             device_name=device_name)

        self.spi_interface = SPI_Interface(notification_pin=board_parameters.PIN_INT_0, baudrate=10000000)

        self.serial_interface = Serial_Interface(port=board_parameters.RC_PARAMS_BOARD_STM32_UART,
                                                 baudrate=board_parameters.RC_PARAMS_BOARD_STM32_UART_BAUD)

        self.i2c_interface = I2C_Interface()

        self.io_extension = RobotControl_IO_Extension(interface=self.i2c_interface)

        # TODO: This should be defined somewhere else
        self.wifi_interface.addCommands(Command(identifier='print',
                                                callback=debug_print,
                                                arguments=['text'],
                                                description='Prints any given text'))

        self.wifi_interface.addCommand(identifier='rgbled', callback=self.io_extension.rgb_led_intern[0].setColor, arguments=['red', 'green', 'blue'], description='')

        # This too
        self.portExpander = SX1508()
        self.portExpander.configureGPIO(gpio=board_parameters.SX1508_PIN_LED, mode=SX1508_GPIO_MODE.OUTPUT,
                                        pullup=False, pulldown=True)

        self.setStatusLed(0)

        self.wifi_interface.registerCallback('connected', self.setStatusLed, parameters={'state': True},
                                             discard_inputs=True)
        self.wifi_interface.registerCallback('disconnected', self.setStatusLed, parameters={'state': False},
                                             discard_inputs=True)

    # === METHODS ======================================================================================================
    def init(self):
        reset_uart()

    def start(self):
        self.wifi_interface.start()

    # ------------------------------------------------------------------------------------------------------------------

    def setStatusLed(self, state, *args, **kwargs):
        self.portExpander.writeGPIO(gpio=board_parameters.SX1508_PIN_LED, state=state)
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def setRGBLEDIntern(self, position, color):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def setRGBLEDExtern(self, position, color):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def handle_exit(self, *args, **kwargs):
        self.setStatusLed(0)
