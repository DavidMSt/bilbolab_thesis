# def test_comm():
#     twipr = TWIPR()
#     twipr.init()
#     twipr.start()
#
#     def rx_samples(samples, *args, **kwargs):
#         sample = samples[0]
#         print(f"{np.rad2deg(sample['estimation']['state']['theta'])}")
#
#     twipr.communication.registerCallback('rx_samples', rx_samples)
#
#     state = 1
#     while True:
#         twipr.communication.serial.debug(state)
#         state = not state
#         time.sleep(0.25)
import numpy as np
import time

from extensions.joystick import rpi_joystick
from utils.stm32.stm32_flash.reset import reset as stm32_reset
import robot.visionrobot as vsrob
import navigation.agent as nav
import aruco_detection.aruco_detector as arc


def test_comm():
    # stm32_reset(0.25)
    robot = vsrob.VisionRobot()
    robot.init()
    robot.start()

    state = 1

    while True:
        robot.debug(state)
        robot.setSpeed([0.5, 0.25])
        state = not state
        time.sleep(0.25)


def test_board():
    robot = vsrob.VisionRobot()
    robot.init()
    robot.start()

    state = 1
    for i in range(0, 10):
        robot.board.setStatusLed(state)
        state = not state
        time.sleep(0.25)

    time.sleep(1)


def test_speed_control():
    stm32_reset(0.25)
    robot = vsrob.VisionRobot()
    robot.init()
    robot.start()
    time.sleep(2)

    def led_on():
        print("LED ON")
        robot.debug(1)

    def led_off():
        print("LED OFF")
        robot.debug(0)

    #joystick = rpi_joystick.RpiJoystick()
    #joystick.set_callback(event=rpi_joystick.A, callback=led_on)
    #joystick.set_callback(event=rpi_joystick.B, callback=led_off)

    while True:
        #val1 = joystick.axes[1]
        #val2 = joystick.axes[2]
        #speed_left = 0.5 * val1 + 0.5 * val2
        #speed_right = 0.5 * val1 - 0.5 * val2
        #robot_old.setSpeed([speed_left, speed_right])
        robot.debug(1)
        time.sleep(1)
        robot.debug(0)
        time.sleep(1)


def test_server_comm():
    stm32_reset(0.25)
    robot = vsrob.VisionRobot()
    robot.init()
    robot.start()

    while True:
        time.sleep(1)


def test_receive():
    robot = vsrob.VisionRobot()
    robot.start()
    robot.setSpeed([0.0, 0.0])

    while True:
        time.sleep(1)

def test_agent():
    stm32_reset(0.25)
    aruco_detector = arc.ArucoDetector(streaming=True)
    aruco_detector.start()

    agent = nav.Agent()
    agent.start()
    while True:
        data = [[],[],[]]
        data[0], data[1], data[2] = aruco_detector.measurement()
        data[0] = np.ndarray.tolist(data[0])
        data[1] = np.ndarray.tolist(data[1])
        data[2] = np.ndarray.tolist(data[2])
        agent.robot.send_data('msr', data)
        time.sleep(5)
    #while True:
    #    mov_in = input("time dphi radius:\n")
    #    mov_in = mov_in.split(sep=" ")
    #    if len(mov_in) == 2:
    #        dphi = float(mov_in[0])
    #        radius = float(mov_in[1])
    #        agent.add_movement(dphi=dphi, radius=radius)
#
    #    else:
    #        dphi = float(mov_in[0])
    #        radius = float(mov_in[1])
    #        vtime = float(mov_in[2])
    #        agent.add_movement(dphi=dphi, radius=radius, vtime=vtime)


if __name__ == '__main__':
    # test_board()
    # print("HALLO")
    #test_speed_control()
    #test_server_comm()
    #test_receive()
    test_agent()
