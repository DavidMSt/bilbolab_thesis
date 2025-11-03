import dataclasses
import time

from hardware.control_board import RobotControl_Board
from core.communication.wifi.tcp.protocols.tcp_json_protocol import TCP_JSON_Message
from archive.robot_old.communication.frodo_communication import FRODO_Communication
from archive.robot_old.control.frodo_control import FRODO_Control
from archive.robot_old.sensing.frodo_sensors import FRODO_Sensors, FRODO_SensorsData
from archive.robot_old.utilities.id import readID
from core.utils.exit import ExitHandler
from core.utils.json_utils import prepareForSerialization
from core.utils.logging_utils import Logger, setLoggerLevel

# === GLOBAL VARIABLES =================================================================================================
logger = Logger('FRODO')
logger.setLevel('INFO')
setLoggerLevel('wifi', 'ERROR')


@dataclasses.dataclass
class FRODO_GeneralSample:
    id: str = ''
    time: float = 0.0


@dataclasses.dataclass
class FRODO_Sample:
    general: FRODO_GeneralSample
    sensors: FRODO_SensorsData


# ======================================================================================================================
class FRODO:
    board: RobotControl_Board
    sensors: FRODO_Sensors
    communication: FRODO_Communication
    control: FRODO_Control

    exit: ExitHandler

    def __init__(self):
        self.id = readID()

        self.board = RobotControl_Board(device_class='robot', device_type='frodo', device_revision='v1',
                                        device_id=self.id, device_name=self.id)

        self.communication = FRODO_Communication(board=self.board)

        # self.sensors = FRODO_Sensors(communication=self.communication)

        self.control = FRODO_Control(self.communication)
        # Add commands

        # Set Speed
        self.communication.wifi.addCommand(identifier='setSpeed',
                                           callback=self.control.setSpeed,
                                           arguments=['speed_left', 'speed_right'],
                                           description='Set Speed')

        # Get Aruco Measurement
        self.communication.wifi.addCommand(identifier='getData',
                                           callback=self.getData,
                                           arguments=[],
                                           description='Get Data',
                                           )

        # Add Movement Command to Navigation
        self.communication.wifi.addCommand(identifier='addNavigationMovement',
                                           callback=self.control.navigation.addMovement,
                                           arguments=['dphi', 'radius', 'vtime'],
                                           description='Add Movement to Navigator Queue')

        # Start Navigation Movement
        self.communication.wifi.addCommand(identifier='startNavigationMovement',
                                           callback=self.control.navigation.startMovement,
                                           arguments=[],
                                           description='Start moving controlled by navigation movement queue.')

        # Stop Navigation Movement
        self.communication.wifi.addCommand(identifier='stopNavigationMovement',
                                           callback=self.control.navigation.stopMovement,
                                           arguments=[],
                                           description='Stop moving controlled by navigation movement queue and drop current movement.')

        # Pause Navigation Movement
        self.communication.wifi.addCommand(identifier='pauseNavigationMovement',
                                           callback=self.control.navigation.pauseMovement,
                                           arguments=[],
                                           description='Pause moving controlled by navigation and keep current movement.')

        # Continue Navigation Movement
        self.communication.wifi.addCommand(identifier='continueNavigationMovement',
                                           callback=self.control.navigation.continueMovement,
                                           arguments=[],
                                           description='Continue previously paused movement.')

        # Clear Navigation Movement Queue
        self.communication.wifi.addCommand(identifier='clearNavigationMovementQueue',
                                           callback=self.control.navigation.clearMovementQueue,
                                           arguments=[],
                                           description='Clear navigation movement queue.')

        # Switch FRODO Control Mode
        self.communication.wifi.addCommand(identifier='setControlMode',
                                           callback=self.control.setMode,
                                           arguments=['mode'],
                                           description='Switch Control Mode')

        # Test Command
        self.communication.wifi.addCommand(identifier='test',
                                           callback=self.test,
                                           arguments=['input'],
                                           description='Test the communication')

        # Set LEDs

        self.exit = ExitHandler(self.close)

    # ------------------------------------------------------------------------------------------------------------------
    def init(self):
        self.board.init()
        # self.sensors.init()
        self.communication.init()
        self.control.init()
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.board.start()
        # self.sensors.start()
        self.communication.start()
        self.control.start()
        logger.info(f"Start Frodo: {self.id}")

        if self.id == 'frodo1':
            self.board.setRGBLEDExtern(color=(int(15), int(20), int(0)))
        elif self.id == 'frodo2':
            self.board.setRGBLEDExtern(color=(int(64 / 10), int(224 / 10), int(208 / 8)))
        elif self.id == 'frodo3':
            self.board.setRGBLEDExtern(color=(int(255 / 10), int(0 / 10), int(0 / 8)))
        self.board.beep()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self.board.beep(frequency='low', repeats=2)
        self.board.setRGBLEDExtern(color=(2, 2, 2))
        time.sleep(1)

    # ------------------------------------------------------------------------------------------------------------------
    def test(self, input):
        return input

    # ------------------------------------------------------------------------------------------------------------------
    def getData(self):
        data_general = FRODO_GeneralSample()
        data_general.id = self.id
        data_general.time = self.communication.wifi.getTime()

        data_sensors = self.sensors.getData()

        data = FRODO_Sample(general=data_general, sensors=data_sensors)
        data_dict = prepareForSerialization(dataclasses.asdict(data))
        return data_dict

    # === PRIVATE METHODS ==============================================================================================



if __name__ == '__main__':
    frodo = FRODO()
    frodo.start()


    while True:
        time.sleep(1)