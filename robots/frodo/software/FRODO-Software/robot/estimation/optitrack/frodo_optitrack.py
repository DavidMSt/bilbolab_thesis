import dataclasses
import enum

from core.utils.callbacks import CallbackContainer, callback_definition
from core.utils.events import event_definition, Event
from core.utils.exit import register_exit_callback
from core.utils.logging_utils import Logger
from robot.common import FRODO_Common
from robot.estimation.optitrack.lib.optitrack import OptiTrack, RigidBodySample
from robot.estimation.optitrack.optitrack_objects import TrackedOrigin, TrackedFRODO, TrackedFRODO_Definition

# ======================================================================================================================
@callback_definition
class FRODO_OptitrackListener_Callbacks:
    sample: CallbackContainer


@event_definition
class FRODO_OptitrackListener_Events:
    sample: Event


@dataclasses.dataclass
class FRODO_OptitrackListenerSettings:
    host: str = 'bree.local'


class FRODO_OptitrackListenerStatus(enum.StrEnum):
    OK = 'OK'
    ERROR = 'ERROR'

class FRODO_OptitrackListener:
    optitrack: OptiTrack
    origin: TrackedOrigin | None = None
    tracked_frodo: TrackedFRODO | None = None
    status: FRODO_OptitrackListenerStatus = FRODO_OptitrackListenerStatus.ERROR

    # === INIT =========================================================================================================
    def __init__(self, common: FRODO_Common, settings: FRODO_OptitrackListenerSettings = None) -> None:
        self.logger = Logger("OPTITRACK LISTENER", "DEBUG")
        self.callbacks = FRODO_OptitrackListener_Callbacks()
        self.events = FRODO_OptitrackListener_Events()
        self.common = common

        if settings is None:
            settings = FRODO_OptitrackListenerSettings()

        self.optitrack = OptiTrack(settings.host)

        self.optitrack.callbacks.description_received.register(self._description_received_callback)
        self.optitrack.events.sample.on(self._sample_callback, max_rate=20)

        register_exit_callback(self.close)

    # === METHODS ======================================================================================================
    def init(self):
        self.optitrack.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        success = self.optitrack.start()
        if success:
            self.status = FRODO_OptitrackListenerStatus.OK
        else:
            self.status = FRODO_OptitrackListenerStatus.ERROR
            self.logger.warning("Cannot start Optitrack listener")

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self.optitrack.close()

    # === PRIVATE METHODS ==============================================================================================
    def _description_received_callback(self, rigid_bodies: dict):
        from robot.definitions import OPTITRACK_ORIGINS, ORIGIN_FRODO
        self.logger.info(f"Description received: {rigid_bodies}")

        if self.common.id in rigid_bodies:
            self.logger.info(f"Frodo {self.common.id} found")
            settings = self.common.getDefinitions()
            self.tracked_frodo = TrackedFRODO(id=self.common.id,
                                              definition=TrackedFRODO_Definition(
                                                  points=settings.optitrack.points,
                                                  point_y_axis_start=settings.optitrack.y_start,
                                                  point_y_axis_end=settings.optitrack.y_end,
                                                  point_x_axis_start=settings.optitrack.x_start,

                                              ))
        else:
            self.logger.warning(f"Own FRODO ID \"{self.common.id}\" not found in Optitrack stream")

        for id, body in rigid_bodies.items():
            if id in OPTITRACK_ORIGINS:
                self.origin = ORIGIN_FRODO
                self.logger.info(f"Optitrack origin {id} found")
                if self.tracked_frodo is not None:
                    self.logger.info(f"Setting origin for tracked frodo {self.tracked_frodo.id}")
                    self.tracked_frodo.setOrigin(self.origin)
                break

    # ------------------------------------------------------------------------------------------------------------------
    def _sample_callback(self, sample: dict[str, RigidBodySample]):
        if self.origin is not None:
            if self.origin.id in sample:
                origin_sample: RigidBodySample = sample[self.origin.id]
                if origin_sample.valid:
                    self.origin.update(origin_sample)

        if self.common.id in sample:
            robot_sample: RigidBodySample = sample[self.common.id]
            if robot_sample.valid and self.tracked_frodo is not None:
                self.tracked_frodo.update(robot_sample)
                self.callbacks.sample.call(self.tracked_frodo.state)
                self.events.sample.set(self.tracked_frodo.state)



