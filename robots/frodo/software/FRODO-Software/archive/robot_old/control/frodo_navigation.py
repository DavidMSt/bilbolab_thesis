import threading
import time

from utils.logging_utils import Logger

logger = Logger('NAVIGATION')
logger.setLevel('INFO')

VELOCITY_AT_1 = 206
DIST_BETWEEN_MOTORS = 140  # [mm]
TIME_SCALER = 185/180
TIME_CONSTANT = 0.1
RADIUS_SCALER = 13/10
TASK_SLEEP_TIME = 0.01


# ======================================================================================================================
class FRODO_Navigator:
    speed_left : float 
    '''input for left motor\n
    -1.0 <= speed_left <= 1.0'''
    speed_right : float 
    '''input for right motor\n
    -1.0 <= speed_right <= 1.0'''

    movements : list  # movement structure: [dphi, radius, time]
    '''List of Movement Commands'''

    pause_event : threading.Event
    '''Event to be able to stop sleeps early to pause movement'''

    queue_lock : threading.Lock
    '''Lock to synchronize Read/Write on Movement List'''

    _speed_lock : threading.Lock
    '''Lock to synchronize Read/Write on motor speed requirement'''

    _update_task : threading.Thread
    _stopped : bool

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.movements = []

        self.pause_event = threading.Event()

        self.queue_lock = threading.Lock()
        self._speed_lock = threading.Lock()

        self.speed_left = 0.0
        self.speed_right = 0.0

        self._stopped = True

        self._update_task = threading.Thread(target=self._movementTask, daemon=True)

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        '''Start thread task but don't start moving'''
        self._update_task.start()

    # ------------------------------------------------------------------------------------------------------------------
    def startMovement(self):
        self._stopped = False    

    # ------------------------------------------------------------------------------------------------------------------
    def continueMovement(self):
        self.pause_event.clear()

    # ------------------------------------------------------------------------------------------------------------------
    def pauseMovement(self):
        self.pause_event.set()
        with self._speed_lock:
            self.speed_left = 0.0
            self.speed_right = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    def stopMovement(self):
        self._stopped = True
        with self._speed_lock:
            self.speed_left = 0.0
            self.speed_right = 0.0

    #  ------------------------------------------------------------------------------------------------------------------    
    def clearMovementQueue(self):
        with self.queue_lock:
            self.queue = []

    # ------------------------------------------------------------------------------------------------------------------
    def addMovements(self, movements):
        '''Add List of Movements to Movement List
            - movement structure: [dphi{rad}, radius{mm}, time{s}] (time parameter is optional)
            - list structure: [movement1, movement2, ...]'''
        for element in movements:
            if len(element) == 2:
                self.addMovement(dphi=element[0], radius=element[1])
            elif len(element) == 3:
                self.addMovement(dphi=element[0], radius=element[1], vtime=element[2])
            else:
                continue

    # ------------------------------------------------------------------------------------------------------------------
    def addMovement(self, dphi=0, radius=0, vtime=0):
        '''Add single Movement to Movement List'''
        with self.queue_lock:
            self.movements.append([dphi, radius, vtime])
        

    # ------------------------------------------------------------------------------------------------------------------
    def getInputs(self):
        '''return current required speeds'''
        with self._speed_lock:
            s_l = self.speed_left
            s_r = self.speed_right
        
        return s_l, s_r

    # ------------------------------------------------------------------------------------------------------------------
    def _movementTask(self):
        '''Task that repetitively checks movement list
            - if list not empty: consecutively execute movements
            - if list empty: sleep'''
        while True:
            try:
                if not self._stopped and len(self.movements) > 0:
                    with self.queue_lock:
                        movement = self.movements.pop(0)
                    if len(movement) == 2:
                        self._calculateMovement(dphi=movement[0], radius=movement[1])
                    elif len(movement) == 3:
                        self._calculateMovement(dphi=movement[0], radius=movement[1], vtime=movement[2])
                else:
                    time.sleep(TASK_SLEEP_TIME)
            except Exception as e:
                print("Error in Movement Task!")
                print(e)

    def _calculateMovement(self, dphi=0, radius=0, vtime=0):
        '''Calculate necessary parameters and write speeds to self.speed_left/right
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
                v_l, v_r = self._vrVlFromRadius(radius)
            else:
                v_r, v_l = self._vrVlFromRadius(radius)

            if radius == 0:
                estimated_time = (15/12) * (dphi * DIST_BETWEEN_MOTORS) / (v_r - v_l) + TIME_CONSTANT
            else:
                estimated_time = (TIME_SCALER * (2*abs(dphi)*abs(radius))) / ((v_r + v_l)) + TIME_CONSTANT
            
        s_l = self._speedToMotorInput(v_l)
        s_r = self._speedToMotorInput(v_r)
        with self._speed_lock:
            self.speed_left = s_l
            self.speed_right = s_r
    
        if vtime > estimated_time:
            if estimated_time > 0:
                self._sleep(estimated_time, s_l, s_r)
                
                with self._speed_lock:
                    self.speed_left = 0.0
                    self.speed_right = 0.0
            self._sleep(vtime-estimated_time, 0.0, 0.0)
        else:
            if vtime == -1:
                self._sleep(estimated_time, s_l, s_r)
            else:
                self._sleep(vtime, s_l, s_r)
            if len(self.movements) == 0:
                with self._speed_lock:
                    self.speed_left = 0.0
                    self.speed_right = 0.0

    def _sleep(self, sleep_time, speed_left, speed_right):
        '''try to sleep for $sleep_time, if interrupted: set speeds \n
            and try to get back to sleep until sleep_time is over!'''
        rest_rest_time = self._rest(sleep_time)
        while rest_rest_time > 0:
            with self._speed_lock:
                self.speed_left = speed_left
                self.speed_right = speed_right
            rest_rest_time = self._rest(rest_rest_time)

    def _rest(self, rest_time):
        t_start = time.time()
        self.pause_event.wait(rest_time)
        if self.pause_event.is_set():
            t_stop = time.time()
            nap_time = rest_time - (t_stop - t_start)
            while self.pause_event.is_set():
                time.sleep(TASK_SLEEP_TIME)
            alpha = 0.25
            return nap_time * (1-alpha + alpha * TIME_SCALER)
        return 0

    # === PRIVATE METHODS ==============================================================================================

    @staticmethod
    def _speedToMotorInput(velocity):
        """Calculate relative velocity for motor input;
            motor input must be in range [-1.0, 1.0]"""
        return velocity / VELOCITY_AT_1

    @staticmethod
    def _vrVlFromRadius(radius):
        """Calculate left and right wheel velocity necessary for radius"""
        ratio_vr_vl = (radius + RADIUS_SCALER *  DIST_BETWEEN_MOTORS/2) / (
                    radius - RADIUS_SCALER * DIST_BETWEEN_MOTORS/2)
        v_0 = VELOCITY_AT_1
        v_1 = v_0 / ratio_vr_vl

        return v_1, v_0
