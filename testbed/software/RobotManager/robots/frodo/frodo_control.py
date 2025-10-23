from core.communication.device_server import Device
from core.utils.logging_utils import Logger
from robots.frodo.frodo_definitions import FRODO_ControlMode, FRODO_Information


# === FRODO CONTROL ====================================================================================================

# === FRODO CONTROL ====================================================================================================

class FRODO_Control:
    mode: FRODO_ControlMode | None = None

    # === INIT =========================================================================================================
    def __init__(self, device: Device, information: FRODO_Information):
        self.device = device
        self.information = information
        self.logger = Logger(f"{self.information.id} Control")

    # === METHODS ======================================================================================================
    def setSpeed(self, speed_left, speed_right):
        self.device.executeFunction(function_name='setSpeed',
                                    arguments={
                                        'speed_left': speed_left,
                                        'speed_right': speed_right
                                    })

    def setSpeedNormalized(self, speed_left_normalized: float, speed_right_normalized: float):
        """Pass-through to robot-side setSpeedNormalized (expects values in [-1..1])."""
        self.device.executeFunction(function_name='setSpeedNormalized',
                                    arguments={
                                        'speed_left_normalized': speed_left_normalized,
                                        'speed_right_normalized': speed_right_normalized
                                    })

    # ------------------------------------------------------------------------------------------------------------------
    def setMode(self, mode: FRODO_ControlMode):
        """Switch control mode on the robot (EXTERNAL/NAVIGATION)."""
        success = self.device.executeFunction(function_name='setMode',
                                              arguments={'mode': mode.value},
                                              request_response=True)
        if success:
            self.mode = mode
        return success

    # --- Navigator control --------------------------------------------------------------------------------------------
    def startNavigation(self):
        """Start the navigator on the robot (process queue)."""
        return self.device.executeFunction(function_name='startNavigation',
                                           arguments={},
                                           request_response=True)

    def stopNavigation(self):
        """Stop the navigator and command zero speed."""
        return self.device.executeFunction(function_name='stopNavigation',
                                           arguments={},
                                           request_response=True)

    def pauseNavigation(self):
        """Pause the navigator (robot will hold with zero speed)."""
        return self.device.executeFunction(function_name='pauseNavigation',
                                           arguments={},
                                           request_response=True)

    def resumeNavigation(self):
        """Resume the navigator if it was paused."""
        return self.device.executeFunction(function_name='resumeNavigation',
                                           arguments={},
                                           request_response=True)

    def clearNavigation(self):
        """Stop navigation and clear the queued elements."""
        return self.device.executeFunction(function_name='clearNavigation',
                                           arguments={},
                                           request_response=True)

    # --- Convenience primitives ----------------------------------------------------------------------------------------
    def moveTo(self, x: float, y: float):
        """Enqueue a MoveTo(x, y) on the robot; robot starts nav if not running."""
        return self.device.executeFunction(function_name='moveTo',
                                           arguments={'x': x, 'y': y},
                                           request_response=True)

    # --- Queue-only versions (no auto-start) --------------------------------------------------------------------------
    def addMoveTo(self, x: float, y: float):
        return self.device.executeFunction(function_name='addMoveTo',
                                           arguments={'x': x, 'y': y},
                                           request_response=True)

    def addMoveToRelative(self, dx: float, dy: float):
        return self.device.executeFunction(function_name='addMoveToRelative',
                                           arguments={'dx': dx, 'dy': dy},
                                           request_response=True)

    def addRelativeStraightMove(self, distance: float):
        return self.device.executeFunction(function_name='addRelativeStraightMove',
                                           arguments={'distance': distance},
                                           request_response=True)

    def addTurnTo(self, psi: float):
        return self.device.executeFunction(function_name='addTurnTo',
                                           arguments={'psi': psi},
                                           request_response=True)

    def addRelativeTurn(self, dpsi: float):
        return self.device.executeFunction(function_name='addRelativeTurn',
                                           arguments={'dpsi': dpsi},
                                           request_response=True)

    def addTurnToPoint(self, x: float, y: float):
        return self.device.executeFunction(function_name='addTurnToPoint',
                                           arguments={'x': x, 'y': y},
                                           request_response=True)

    def addTimeWait(self, duration: float, reference: str = "PRIMITIVE"):
        return self.device.executeFunction(function_name='addTimeWait',
                                           arguments={'duration': duration, 'reference': reference},
                                           request_response=True)

    def addAbsoluteTimeWait(self, unix_time: float):
        return self.device.executeFunction(function_name='addAbsoluteTimeWait',
                                           arguments={'unix_time': unix_time},
                                           request_response=True)

    def addEventWait(self, event: str):
        return self.device.executeFunction(function_name='addEventWait',
                                           arguments={'event': event},
                                           request_response=True)

    def addCoordinatedMoveTo(self, x: float, y: float, psi_end: float | None = None):
        args = {'x': x, 'y': y}
        if psi_end is not None:
            args['psi_end'] = psi_end
        return self.device.executeFunction(function_name='addCoordinatedMoveTo',
                                           arguments=args,
                                           request_response=True)
