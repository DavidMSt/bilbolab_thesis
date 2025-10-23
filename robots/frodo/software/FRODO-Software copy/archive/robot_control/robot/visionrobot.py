import ctypes
import threading
import time

from board.board import RobotControl_Board
from bilbo_old.communication.twipr_communication import TWIPR_Communication
from utils.ctypes_utils import struct_to_dict

''' Define Structs START '''


class motor_input_struct(ctypes.Structure):
    _fields_ = [("input_left", ctypes.c_float), ("input_right", ctypes.c_float)]


class motor_speed_struct(ctypes.Structure):
    _fields_ = [("input_left", ctypes.c_float), ("input_right", ctypes.c_float)]


class CommunicationData(ctypes.Structure):
    _fields_ = [
        ("tick", ctypes.c_uint32),
        ("debug", ctypes.c_uint8),
        ("state", ctypes.c_uint8),
        ("battery_voltage", ctypes.c_float),

        ("goal_speed_l", ctypes.c_float),
        ("goal_speed_r", ctypes.c_float),
        ("rpm_l", ctypes.c_float),
        ("rpm_r", ctypes.c_float),
        ("velocity_l", ctypes.c_float),
        ("velocity_r", ctypes.c_float),
        ("velocity_forward", ctypes.c_float),
        ("velocity_turn", ctypes.c_float),

        ("imu_gyr_x", ctypes.c_float),
        ("imu_gyr_y", ctypes.c_float),
        ("imu_gyr_z", ctypes.c_float),
        ("imu_acc_x", ctypes.c_float),
        (
            "imu_acc_y", ctypes.c_float),
        ("imu_acc_z", ctypes.c_float),
    ]


''' Define Structs END '''


class VisionRobot:
    board: RobotControl_Board
    communication: TWIPR_Communication
    _thread: threading.Thread
    data: CommunicationData

    def __init__(self):
        self.board = RobotControl_Board(device_class='robot_old',
                                        device_type='visionrobot',
                                        device_revision='v2',
                                        device_id='visionrobot1',
                                        device_name='Vision Robot 1')

        self.communication = TWIPR_Communication(board=self.board)
        self.communication.serial.interface.registerCallback('rx_stream', self._rx_callback)

        self.communication.wifi.addCommand(identifier='setSpeed',
                                           callback=self.setSpeed,
                                           arguments=['speed'],
                                           description='Set the speed of the motors')

        self._thread = threading.Thread(target=self._threadFunction)

        self.data = CommunicationData()

        self.board.init()
        self.communication.init()

    # === METHODS ======================================================================================================

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        #self.aruco_detector.start()
        self.board.start()
        self.communication.start()
        #self._thread.start()
        print("START VISION ROBOT ...")

    # ------------------------------------------------------------------------------------------------------------------

    '''Debug Function to check UART functionality
        Turns on and off LED Below USB C Port
    '''

    def debug(self, state):
        self.communication.serial.executeFunction(module=0x01,
                                                  address=0x01,
                                                  data=state,
                                                  input_type=ctypes.c_uint8,
                                                  output_type=None)

    def registerExternalWifiCallback(self, identifier, callback, arguments, description):
        self.communication.wifi.addCommand(identifier=identifier, callback=callback, arguments=arguments,
                                           description=description)

    def setSpeed(self, speed):
        assert (isinstance(speed, list))
        print(f"Set Speed to {speed}")
        input_struct = motor_input_struct(input_left=speed[0], input_right=speed[1])
        self.communication.serial.executeFunction(module=0x01,
                                                  address=0x02,
                                                  data=input_struct,
                                                  input_type=motor_input_struct)

    def send_data(self, descriptor, data):
        sdata = {}
        sdata[descriptor] = data
        self.communication.wifi.sendStream(sdata)

    # === PRIVATE METHODS ==============================================================================================
    ''' Send Stream to Hardware Manager '''

    def _threadFunction(self):
        while True:
            data = {'test': 1}
            self.communication.wifi.sendStream(data)
            time.sleep(0.1)

    ''' get Data from MC Firmware '''

    def _rx_callback(self, msg_data, *args, **kwargs):
        data = CommunicationData.from_buffer_copy(msg_data)
        self.data = struct_to_dict(data)
        print(f"{data}")
