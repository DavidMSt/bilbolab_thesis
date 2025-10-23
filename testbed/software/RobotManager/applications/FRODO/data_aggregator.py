from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import qmt

from applications.FRODO.tracker.definitions import TrackedStatic, TrackedFRODO
from applications.FRODO.tracker.frodo_tracker import FRODO_Tracker
from core.utils.callbacks import CallbackContainer, callback_definition
from core.utils.events import event_definition, Event
from core.utils.logging_utils import Logger
from robots.frodo.frodo import FRODO
from robots.frodo.frodo_definitions import FRODO_DynamicState, FRODO_ArucoMeasurement, getObjectFromArucoId
from robots.frodo.frodo_manager import FRODO_Manager
from robots.frodo.frodo_utilities import vector2LocalFrame


# ======================================================================================================================
@dataclasses.dataclass
class TestbedObjectState:
    x: float
    y: float
    psi: float


class TestbedObject:
    state: TestbedObjectState | None = None
    id: str

    def __init__(self, id: str):
        self.id = id
        self.logger = Logger(f"{self.id}_testbed_object", "DEBUG")


# ======================================================================================================================
class TestbedObject_FRODO(TestbedObject):
    dynamic_state: FRODO_DynamicState
    robot: FRODO
    tracked_object: TrackedFRODO
    measurements: list[VisionMeasurement]

    # === INIT =========================================================================================================
    def __init__(self, robot: FRODO, tracked_object: TrackedFRODO):
        super().__init__(robot.id)
        self.robot = robot
        self.tracked_object = tracked_object
        self.measurements = []

    # === PROPERTIES ===================================================================================================
    @property
    def dynamic_state(self) -> FRODO_DynamicState:
        return self.robot.core.data.estimation.state

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def state(self) -> TestbedObjectState:
        state = TestbedObjectState(x=self.dynamic_state.x, y=self.dynamic_state.y, psi=self.dynamic_state.psi)
        return state
    # === METHODS ======================================================================================================

    # === PRIVATE METHODS ==============================================================================================


# ======================================================================================================================
class TestbedObject_STATIC(TestbedObject):
    tracked_static: TrackedStatic

    # === INIT =========================================================================================================
    def __init__(self, static: TrackedStatic):
        super().__init__(static.id)
        self.tracked_static = static

    # === PROPERTIES ===================================================================================================
    @property
    def state(self) -> TestbedObjectState:
        state = TestbedObjectState(
            x=self.tracked_static.state.x,
            y=self.tracked_static.state.y,
            psi=self.tracked_static.state.psi
        )
        return state


# ======================================================================================================================
@dataclasses.dataclass
class VisionMeasurement:
    object_from: TestbedObject
    object_to: TestbedObject
    relative: TestbedObjectState
    raw_measurement: FRODO_ArucoMeasurement | None
    covariance: Optional[np.ndarray]


@event_definition
class FRODO_DataAggregator_Events:
    initialized: Event
    update: Event


@callback_definition
class FRODO_DataAggregator_Callbacks:
    initialized: CallbackContainer
    update: CallbackContainer


# === FRODO ALGORITHM ADAPTER ==========================================================================================
class FRODO_DataAggregator:
    manager: FRODO_Manager
    tracker: FRODO_Tracker

    events: FRODO_DataAggregator_Events
    callbacks: FRODO_DataAggregator_Callbacks

    objects: dict[str, TestbedObject]

    _initialized = False

    # === INIT =========================================================================================================
    def __init__(self, manager: FRODO_Manager, tracker: FRODO_Tracker):
        self.manager = manager
        self.tracker = tracker
        self.logger = Logger("DATA_AGGREGATOR", "DEBUG")
        self.objects = {}

        self.callbacks = FRODO_DataAggregator_Callbacks()
        self.events = FRODO_DataAggregator_Events()

    # === PROPERTIES ===================================================================================================
    @property
    def robots(self) -> dict[str, TestbedObject_FRODO]:
        return {k: v for k, v in self.objects.items() if isinstance(v, TestbedObject_FRODO)}

    @property
    def statics(self) -> dict[str, TestbedObject_STATIC]:
        return {k: v for k, v in self.objects.items() if isinstance(v, TestbedObject_STATIC)}

    # === METHODS ======================================================================================================
    def initialize(self):
        self.events.initialized.set()
        self.callbacks.initialized.call()
        self._initialized = True

    # ------------------------------------------------------------------------------------------------------------------
    def update(self):
        if not self._initialized:
            self.logger.error(f"Data aggregator not initialized before first update called")

        # Update the FRODO robots
        for frodo in self.robots.values():
            frodo.measurements = []
            self.update_robot_measurements(frodo)
            # self.update_robot_state(frodo)

        self.events.update.set()
        self.callbacks.update.call()

    # ------------------------------------------------------------------------------------------------------------------
    def update_robot_measurements(self, frodo: TestbedObject_FRODO):
        if frodo.robot.core.data is not None:
            for measurement in frodo.robot.core.data.measurements.aruco_measurements:
                self._process_aruco_measurement(frodo.robot, measurement)

    # # ------------------------------------------------------------------------------------------------------------------
    # def update_robot_state(self, frodo: TestbedObject_FRODO):
    #     tracked_object = frodo.tracked_object
    #     frodo.dynamic_state.x = tracked_object.state.x
    #     frodo.dynamic_state.y = tracked_object.state.y
    #     frodo.dynamic_state.psi = tracked_object.state.psi
    #     frodo.dynamic_state.v = tracked_object.state.v
    #     frodo.dynamic_state.psi_dot = tracked_object.state.psi_dot

    # ------------------------------------------------------------------------------------------------------------------
    def addRobot(self, robot: FRODO) -> TestbedObject_FRODO | None:
        if robot.id in self.objects:
            self.logger.error(f"Robot {robot.id} already exists")
            return None

        # Get the tracked object from the Tracker
        tracked_object = self.tracker.robots.get(robot.id)
        if tracked_object is None:
            self.logger.error(f"Robot {robot.id} does not exist as a tracked object")
            return None

        testbed_object = TestbedObject_FRODO(robot, tracked_object)

        self.objects[robot.id] = testbed_object

        return testbed_object

    # ------------------------------------------------------------------------------------------------------------------
    def addStatic(self, static: TrackedStatic) -> TestbedObject_STATIC | None:

        if static.id in self.objects:
            self.logger.error(f"Static object {static.id} already exists")
            return None

        testbed_object = TestbedObject_STATIC(static)
        self.objects[static.id] = testbed_object

        return testbed_object

    # ------------------------------------------------------------------------------------------------------------------
    def removeRobot(self, robot: FRODO):
        testbed_object = self._get_testbed_object_from_robot(robot)

        if testbed_object is not None:
            self.objects.pop(testbed_object.id)

    # ------------------------------------------------------------------------------------------------------------------
    def removeStatic(self, static: TrackedStatic):
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    def clear(self):
        self.objects.clear()

    # === PRIVATE METHODS ==============================================================================================
    def _get_object_from_marker_id(self, marker_id) -> TestbedObject | None:
        object_type, object_id, rotation = getObjectFromArucoId(marker_id)

        if object_type == 'frodo':
            if object_id in self.objects:
                return self.objects[object_id]
        elif object_type == 'static':
            if object_id in self.statics:
                return self.statics[object_id]
        else:
            self.logger.warning(f"Object with marker_id {marker_id} not found")

        return None

    # ------------------------------------------------------------------------------------------------------------------
    def _get_testbed_object_from_robot(self, robot: FRODO) -> TestbedObject_FRODO | None:
        for testbed_robot in self.robots.values():
            if testbed_robot.robot == robot:
                return testbed_robot
        return None

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _get_rotation_from_marker_id(marker_id) -> float:
        object_type, object_id, rotation = getObjectFromArucoId(marker_id)
        return rotation

    # ------------------------------------------------------------------------------------------------------------------
    def _process_aruco_measurement(self, robot: FRODO, measurement: FRODO_ArucoMeasurement) -> VisionMeasurement | None:
        # 0. Get the corresponding testbed object
        testbed_object = self._get_testbed_object_from_robot(robot)
        measured_object = self._get_object_from_marker_id(measurement.measured_aruco_id)

        if testbed_object is None:
            self.logger.warning(f"Robot {robot.id} does not exist")
            return None

        if measured_object is None:
            self.logger.warning(f"Object {measurement.measured_aruco_id} does not exist")
            return None

        # 1. Convert the measurement into a VisionMeasurement
        vision_measurement = self._convert_aruco_measurement(robot, measurement)

        # 2. Calculate the perfect measurement
        perfect_vision_measurement = self._calculate_perfect_vision_measurement(
            object_from=vision_measurement.object_from,
            object_to=vision_measurement.object_to
        )

        # 3. Fuse the measurement
        fused_measurement = self._fuse_aruco_measurement(
            measurement=vision_measurement,
            measurement_perfect=perfect_vision_measurement,
            fuse_factor=0.5
        )

        if any([object_to == measured_object for object_to in testbed_object.measurements]):
            self.logger.warning(
                f"Object {measured_object.id} already measured by robot {robot.id}. Measurements have not been cleaned up.")
            return None

        testbed_object.measurements.append(fused_measurement)

        return fused_measurement

    # ------------------------------------------------------------------------------------------------------------------
    def _convert_aruco_measurement(self, robot: FRODO, measurement: FRODO_ArucoMeasurement) -> VisionMeasurement | None:

        # 1. Get the participating objects
        emitting_robot = self._get_testbed_object_from_robot(robot)
        if emitting_robot is None:
            self.logger.error(f"Robot {robot.id} does not exist")
            return None

        measured_object = self._get_object_from_marker_id(measurement.measured_aruco_id)
        if measured_object is None:
            self.logger.error(f"Object {measurement.measured_aruco_id} does not exist")
            return None

        # 2. Extract the data from the measurement
        position = measurement.position
        psi = measurement.psi
        uncertainty_position = measurement.uncertainty_position
        uncertainty_psi = measurement.uncertainty_psi

        # 3. Rotate psi, based on the marker_id
        rotation = self._get_rotation_from_marker_id(measurement.measured_aruco_id)
        psi = psi + rotation

        # 4. Build the measurement
        vision_measurement = VisionMeasurement(
            object_from=emitting_robot,
            object_to=measured_object,
            relative=TestbedObjectState(x=float(position[0]), y=float(position[1]), psi=psi),
            covariance=np.block([
                [uncertainty_position, np.zeros((2, 1))],
                [np.zeros((1, 2)), np.array([[uncertainty_psi]])]
            ]),
            raw_measurement=measurement
        )

        psi_deg = np.rad2deg(psi)
        psi_wrap = np.rad2deg(qmt.wrapToPi(psi))

        return vision_measurement

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _calculate_perfect_vision_measurement(object_from: TestbedObject,
                                              object_to: TestbedObject) -> VisionMeasurement:

        from_x = object_from.state.x
        from_y = object_from.state.y
        from_psi = object_from.state.psi
        to_x = object_to.state.x
        to_y = object_to.state.y
        to_psi = object_to.state.psi

        relative_position = np.asarray([to_x - from_x, to_y - from_y])
        relative_psi = to_psi - from_psi

        # Transform the relative position into the frame of the from object
        relative_position_from_object = vector2LocalFrame(relative_position, from_psi)

        measurement = VisionMeasurement(
            object_from=object_from,
            object_to=object_to,
            relative=TestbedObjectState(x=relative_position_from_object[0], y=relative_position_from_object[1],
                                        psi=relative_psi),
            covariance=np.diag([0.001, 0.001, 0.001]),
            raw_measurement=None
        )

        perfect_psi = np.rad2deg(qmt.wrapToPi(relative_psi))

        return measurement

    # ------------------------------------------------------------------------------------------------------------------
    def _fuse_aruco_measurement(
            self,
            measurement: VisionMeasurement,
            measurement_perfect: VisionMeasurement,
            fuse_factor: float = 0.75,
            fuse_factor_psi=1.0
    ) -> VisionMeasurement:
        """
        Fuses two vision measurements based on a specified fuse factor. This method combines the real-world
        measurement with a theoretically perfect measurement to produce a modified measurement that is
        adjusted based on the fusion ratio. Useful in scenarios involving inaccurate real-world measurements,
        to enhance the reliability of downstream processing.

        Args:
            measurement (VisionMeasurement): The real-world measurement to be adjusted.
            measurement_perfect (VisionMeasurement): The perfect reference measurement used for adjustment.
            fuse_factor (float): A fusion ratio determining the weighting applied to the perfect
                vs real-world measurement. Values should range between 0 and 1. Defaults to 0.75.

        Returns:
            VisionMeasurement: The newly fused measurement object that incorporates adjustments
            based on the fusion ratio.
        """

        def _wrap_angle(angle: float) -> float:
            """Wrap angle to (-pi, pi]."""
            return (angle + np.pi) % (2.0 * np.pi) - np.pi

        # Validate & clamp factor
        alpha = float(np.clip(fuse_factor, 0.0, 1.0))
        alpha_psi = float(np.clip(fuse_factor_psi, 0.0, 1.0))

        # Sanity check the participants
        if (measurement.object_from.id != measurement_perfect.object_from.id or
                measurement.object_to.id != measurement_perfect.object_to.id):
            self.logger.warning(
                "Fuse called with mismatching participants: "
                f"{measurement.object_from.id}->{measurement.object_to.id} vs "
                f"{measurement_perfect.object_from.id}->{measurement_perfect.object_to.id}. "
                "Returning the original measurement."
            )
            return measurement

        # Pack state vectors (x, y, psi)
        z_meas = np.array([measurement.relative.x, measurement.relative.y, measurement.relative.psi], dtype=float)
        z_perf = np.array(
            [measurement_perfect.relative.x, measurement_perfect.relative.y, measurement_perfect.relative.psi],
            dtype=float)

        # Interpolate position linearly
        pos_fused = z_meas[:2] + alpha * (z_perf[:2] - z_meas[:2])

        # Interpolate angle on the circle (move a fraction alpha toward perfect angle)
        dpsi = _wrap_angle(float(z_perf[2] - z_meas[2]))
        psi_fused = _wrap_angle(float(z_meas[2] + alpha_psi * dpsi))

        # Fuse covariance:
        # If we assume independence between the two sources, then the variance of a convex combination
        # a*X + b*Y is a^2*Var(X) + b^2*Var(Y). We apply that with a=(1-alpha), b=alpha.
        C_meas = measurement.covariance
        C_perf = measurement_perfect.covariance

        a = (1.0 - alpha)
        b = alpha

        fused_cov = None
        if C_meas is not None and C_perf is not None:
            try:
                fused_cov = a * a * C_meas + b * b * C_perf
            except Exception as e:
                self.logger.warning(f"Covariance fusion failed ({e}), falling back to convex mix.")
                fused_cov = (1.0 - alpha) * (C_meas if C_meas is not None else np.zeros((3, 3))) + \
                            alpha * (C_perf if C_perf is not None else np.zeros((3, 3)))
        elif C_meas is not None:
            fused_cov = a * a * C_meas
        elif C_perf is not None:
            fused_cov = b * b * C_perf
        else:
            # Default small covariance if neither provided
            fused_cov = np.diag([1e-2, 1e-2, 1e-2])

        # Build fused VisionMeasurement (keep the original raw_measurement for traceability)
        fused = VisionMeasurement(
            object_from=measurement.object_from,
            object_to=measurement.object_to,
            relative=TestbedObjectState(x=float(pos_fused[0]), y=float(pos_fused[1]), psi=float(psi_fused)),
            covariance=fused_cov,
            raw_measurement=measurement.raw_measurement
        )

        # Optional debug logs
        # self.logger.debug(
        #     f"Fused measurement with alpha={alpha:.3f}: "
        #     f"meas=[{z_meas[0]:.3f},{z_meas[1]:.3f},{z_meas[2]:.3f}], "
        #     f"perf=[{z_perf[0]:.3f},{z_perf[1]:.3f},{z_perf[2]:.3f}] -> "
        #     f"fused=[{pos_fused[0]:.3f},{pos_fused[1]:.3f},{psi_fused:.3f}]"
        # )

        return fused
