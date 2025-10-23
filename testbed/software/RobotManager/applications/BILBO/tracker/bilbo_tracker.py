import dataclasses

import numpy as np

from core.utils.events import event_definition, Event
from core.utils.logging_utils import Logger
from extensions.optitrack.optitrack import OptiTrack, RigidBodySample
from core.utils.callbacks import callback_definition, CallbackContainer


# ======================================================================================================================
@dataclasses.dataclass
class TrackedOrigin_Definition:
    points: list[int]
    origin: int
    x_axis_end: int
    y_axis_end: int
    vertical_offset: float


@dataclasses.dataclass
class TrackedOrigin_State:
    x: float
    y: float
    z: float
    orientation: np.ndarray


class TrackedOrigin:
    id: str
    definition: TrackedOrigin_Definition
    state: TrackedOrigin_State
    tracking_valid: bool = False

    # === INIT =========================================================================================================
    def __init__(self, id: str, definition: TrackedOrigin_Definition):
        self.id = id
        self.definition = definition
        self.state = TrackedOrigin_State(x=0, y=0, z=0, orientation=np.asarray([1, 0, 0, 0]))

    # === METHODS ======================================================================================================
    def update(self, data: RigidBodySample):
        # Check if tracking is valid
        if not data.valid:
            self.tracking_valid = False
            return


# ======================================================================================================================
@dataclasses.dataclass
class TrackedBILBO_Definition:
    points: list[int]
    point_x_axis_start: int
    point_x_axis_end: int
    point_y_axis_start: int
    point_y_axis_end: int

    vertical_offset: float
    wheel_diameter: float


@dataclasses.dataclass
class TrackedBILBO_State:
    x: float
    y: float
    z: float
    theta: float
    psi: float


@callback_definition
class TrackedBILBO_Callbacks:
    sample: CallbackContainer


class TrackedBILBO:
    id: str
    definition: TrackedBILBO_Definition
    state: TrackedBILBO_State
    tracking_valid: bool = False

    origin: TrackedOrigin | None = None

    callbacks: TrackedBILBO_Callbacks

    # === INIT =========================================================================================================
    def __init__(self, id: str, definition: TrackedBILBO_Definition, origin: TrackedOrigin = None):
        self.id = id
        self.definition = definition
        self.state = TrackedBILBO_State(x=0, y=0, z=self.definition.wheel_diameter / 2, theta=0, psi=0)
        self.origin = origin

        self.callbacks = TrackedBILBO_Callbacks()

    # ------------------------------------------------------------------------------------------------------------------
    def update(self, data: RigidBodySample):
        # Check if tracking is valid
        if not data.valid:
            self.tracking_valid = False
        else:
            ...

        self.callbacks.sample.call(self.state, self.tracking_valid)


# =====================================================================================================================

@callback_definition
class BILBO_Tracker_Callbacks:
    new_sample: CallbackContainer
    description_received: CallbackContainer


@event_definition
class BILBO_Tracker_Events:
    new_sample: Event
    description_received: Event


class BILBO_Tracker:
    optitrack: OptiTrack

    robots: dict[str, TrackedBILBO]
    origin: TrackedOrigin | None = None

    # === INIT =========================================================================================================
    def __init__(self, robots: dict[str, TrackedBILBO] = None, origin: TrackedOrigin = None):
        self.logger = Logger('BILBO Tracker', 'DEBUG')

        self.robots = robots
        self.origin = origin

        self.optitrack = OptiTrack(server_address='bree.local')
        self.optitrack.events.sample.on(self._onSample)
        self.optitrack.callbacks.description_received.register(self._onDescriptionReceived)

    # === METHODS ======================================================================================================
    def init(self):
        self.optitrack.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        success = self.optitrack.start()

        if not success:
            self.logger.error("Could not start OptiTrack. Tracking disabled")
            return False
        self.logger.info("Starting Tracker")

        return True

    # === PRIVATE METHODS ==============================================================================================
    def _onSample(self, sample: dict[str, RigidBodySample]):
        self.logger.info(f"Received sample from OptiTrack: {sample}")

        for robot in self.robots.values():
            robot.tracking_valid = False

        for name, data in sample.items():
            if name in self.robots:
                self.robots[name].update(data)
                self.robots[name].tracking_valid = data.valid

        if self.origin is not None:
            if self.origin.id in sample:
                self.origin.update(sample[self.origin.id])
                self.origin.tracking_valid = sample[self.origin.id].valid
            else:
                self.origin.tracking_valid = False

    # ------------------------------------------------------------------------------------------------------------------
    def _onDescriptionReceived(self, rigid_bodies: dict):
        self.logger.info(f"Received description from OptiTrack: {rigid_bodies}")


if __name__ == '__main__':
    tracker = BILBO_Tracker()
    tracker.init()
    tracker.start()
