import math
import numpy as np
import threading
import time
import typing

import bilbo_old.visionrobot as vsrob

VELOCITY_AT_1 = 206
DIST_BETWEEN_MOTORS = 108  # [mm]
TIME_SCALOR = 4 / 3
RADIUS_SCALOR = 4 / 5


class Navigation:
    movements: list  # movement structure: [dphi, radius, time]
    '''List of Movement Commands'''

    robot_set_speed: typing.Callable[..., None]
    '''Function Pointer to setSpeed Function of Visionrobot'''

    update_thread: threading.Thread
    '''Thread for repetetive Movement List reading'''
    queue_lock: threading.Lock
    '''Queue to synchronize Read/Write on Movement List'''

    def __init__(self, set_speed_func):
        '''Initialize Navigation'''

        self.movements = []
        self.robot_set_speed = set_speed_func
        self.queue_lock = threading.Lock()
        self.update_thread = threading.Thread(target=self._movement_task)

    def start(self):
        '''Start Navigation'''
        self.update_thread.start()

    def add_movements(self, list):
        '''Add List of Movements to Movement List
            - movement structure: [dphi{rad}, radius{mm}, time{s}] (time parameter is optional)
            - list structure: [movement1, movement2, ...]'''
        for element in list:
            if len(element) == 2:
                self.add_movement(dphi=element[0], radius=element[1])
            elif len(element) == 3:
                self.add_movement(dphi=element[0], radius=element[1], vtime=element[2])
            else:
                continue

    def add_movement(self, dphi=0, radius=-1, vtime=-1):
        """Add single Movement to Movement List"""
        self.queue_lock.acquire()
        self.movements.append([dphi, radius, vtime])
        self.queue_lock.release()

    def _movement_task(self):
        """Task that repetitively checks movement list
            - if list not empty: consecutively execute movements
            - if list empty: sleep"""
        while True:
            try:
                if len(self.movements) > 0:
                    self.queue_lock.acquire()
                    movement = self.movements.pop(0)
                    self.queue_lock.release()
                    if len(movement) == 2:
                        self._calculate_movement(dphi=movement[0], radius=movement[1])
                    else:
                        self._calculate_movement(dphi=movement[0], radius=movement[1], vtime=movement[2])

            except Exception as e:
                print("Error in Movement Task!")
                print(e)
            time.sleep(1)

    def _speed_to_motor_input(self, velocity):
        '''Calculate relative velocity for motor input;
            motor input must be in range [-1.0, 1.0]'''
        return velocity / VELOCITY_AT_1

    def _vr_vl_from_radius(self, radius):
        '''Caculate left and right wheel velocity necessary for radius'''
        ratio_vr_vl = (RADIUS_SCALOR * 2 * radius + DIST_BETWEEN_MOTORS) / (
                    RADIUS_SCALOR * 2 * radius - DIST_BETWEEN_MOTORS)
        v_0 = VELOCITY_AT_1
        v_1 = v_0 / ratio_vr_vl

        return v_1, v_0

    def _calculate_movement(self, dphi=0, radius=-1, vtime=0):
        '''Calculate necessary parameters and perform movement
            - units: 
                - time in [s]
                - dphi in [rad]
                - radius in [mm] 
            - time behavior: 
                - if vtime > estimated_time: perform movement, sleep for rest of vtime
                - if vtime < estimated_time: perform movement, cancel after vtime
            - special commands:
                - stop: dphi=0, radius=0, time=any
                - straight: dphi=0, radius=any, time=any'''

        v_l = 0
        v_r = 0
        estimated_time = 0

        if dphi == 0:
            if radius == 0:
                # stop
                pass
            elif radius > 0:
                v_l = VELOCITY_AT_1
                v_r = VELOCITY_AT_1
                estimated_time = radius / VELOCITY_AT_1
            else:
                raise Exception("Invalid Combination of dphi and radius: dphi = 0 and radius < 0")

        else:
            if dphi > 0:
                v_l, v_r = self._vr_vl_from_radius(radius)
            else:
                v_r, v_l = self._vr_vl_from_radius(radius)

            estimated_time = TIME_SCALOR * (dphi * DIST_BETWEEN_MOTORS) / (v_r - v_l)

        s_l = self._speed_to_motor_input(v_l)
        s_r = self._speed_to_motor_input(v_r)
        self.robot_set_speed([s_l, s_r])

        if vtime > estimated_time:
            time.sleep(estimated_time)
            self.robot_set_speed([0.0, 0.0])
            time.sleep(vtime - estimated_time)
        else:
            time.sleep(vtime)
            if not len(self.movements) > 0:
                self.robot_set_speed([0.0, 0.0])
