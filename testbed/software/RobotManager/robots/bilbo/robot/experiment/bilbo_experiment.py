from __future__ import annotations

import abc
import dataclasses
import enum
import math
import pickle
import threading
from dataclasses import asdict
import numpy as np
from dacite import Config

from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.dataclass_utils import from_dict, from_dict_auto
from core.utils.logging_utils import Logger
from robots.bilbo.robot.bilbo_control import BILBO_Control

# === CUSTOM PACKAGES ==================================================================================================
from robots.bilbo.robot.bilbo_core import BILBO_Core
from core.utils.data import generate_time_vector, generate_random_input, resample
from robots.bilbo.robot.bilbo_definitions import BILBO_Control_Mode, BILBO_CONTROL_DT, MAX_STEPS_TRAJECTORY
from core.utils.events import event_definition, Event, EventFlag, pred_flag_equals, waitForEvents
from core.utils.plotting import UpdatablePlot
from core.utils.sound.sound import speak, playSound
from core.utils.ilc.ILC_DAMN_bib import noilc_self_para_v2, noilc_design, lift_vec2mat, \
    plot_bilbo_ilc_progression, generate_q_filter
from core.utils.archives.events import pred_flag_key_equals
from core.utils.colors import get_shaded_color
from core.utils.ilc.ILC_DAMN_bib import reference as ilc_reference
from robots.bilbo.robot.experiment.definitions import BILBO_InputTrajectory, BILBO_InputTrajectoryStep, \
    BILBO_TrajectoryExperimentData, BILBO_TrajectoryExperiment
from robots.bilbo.robot.experiment.helpers import generateRandomTestTrajectory, plotTrajectoryExperimentData


# ======================================================================================================================
class BILBO_Experiment_Status(enum.StrEnum):
    NONE = "none"
    RUNNING_TRAJECTORY = "running_trajectory"
    CALCULATING = "calculating"
    WAITING_FOR_USER = "waiting_for_user"
    FINISHED = "finished"
    ABORTED = "aborted"


@event_definition
class BILBO_Experiment_Events:
    started: Event
    finished: Event
    aborted: Event
    status_changed: Event = Event(flags=EventFlag('status', BILBO_Experiment_Status))


@callback_definition
class BILBO_Experiment_Callbacks:
    stopped: CallbackContainer


class BILBO_Experiment(abc.ABC):
    type: str
    events: BILBO_Experiment_Events

    status: BILBO_Experiment_Status

    _thread: threading.Thread | None = None
    _stopEvent: Event
    _exit: bool = False

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, core: BILBO_Core, experiment_handler: BILBO_ExperimentHandler, control: BILBO_Control):
        self.core = core
        self.experiment_handler = experiment_handler
        self.control = control
        self.events = BILBO_Experiment_Events()
        self.callbacks = BILBO_Experiment_Callbacks()

        self.status = BILBO_Experiment_Status.NONE
        self.logger = Logger("Experiment")
        self._thread = threading.Thread(target=self.task, daemon=True)
        self._stopEvent = Event()

    # === PROPERTIES ===================================================================================================
    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value: BILBO_Experiment_Status):
        self._status = value
        self.events.status_changed.set(data=value, flags={'status': value})

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.events.started.set()
        self._thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def stop(self, aborted: bool = False):
        self._exit = True
        self._stopEvent.set()

        if aborted:
            self.events.aborted.set()
        else:
            self.events.finished.set()

        self.callbacks.stopped.call()
        self.logger.info("Experiment stopped")

    # ------------------------------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def task(self):
        ...


# ======================================================================================================================
class BILBO_ExperimentHandler_Status(enum.StrEnum):
    IDLE = "idle"
    EXPERIMENT_RUNNING = "experiment_running"


# ======================================================================================================================
@event_definition
class BILBO_Experiments_Events:
    status_changed: Event = Event(flags=EventFlag('status', BILBO_ExperimentHandler_Status))

    ll_trajectory_finished: Event = Event(flags=EventFlag('trajectory_id', int))
    ll_trajectory_aborted: Event = Event(flags=EventFlag('trajectory_id', int))
    ll_trajectory_started: Event = Event(flags=EventFlag('trajectory_id', int))

    trajectory_finished: Event = Event(flags=EventFlag('trajectory_id', (int, str)),
                                       data_type=BILBO_TrajectoryExperiment)

    trajectory_loaded: Event = Event()

    waiting_for_user: Event = Event()

    experiment_started: Event = Event()
    experiment_finished: Event = Event()
    experiment_aborted: Event = Event()

    experiment_status_changed: Event


# ======================================================================================================================
class BILBO_ExperimentHandler:
    experiment: BILBO_Experiment | None = None
    control: BILBO_Control

    status: BILBO_ExperimentHandler_Status = BILBO_ExperimentHandler_Status.IDLE
    current_trajectory: BILBO_InputTrajectory | None = None

    _loadedTrajectory: BILBO_InputTrajectory | None = None

    def __init__(self, core: BILBO_Core, control: BILBO_Control):
        self.core = core
        self.control = control
        self.id = core.id
        self.logger = self.core.logger
        self.device = self.core.device

        self.events = BILBO_Experiments_Events()
        self.device.events.event.on(self._trajectory_event_callback,
                                    predicate=pred_flag_key_equals('event', 'trajectory')
                                    )

    # ------------------------------------------------------------------------------------------------------------------
    def runExperiment(self, experiment_type: type, *args, **kwargs) -> bool:
        if self.experiment is not None:
            self.logger.error(f"Experiment of type {type(self.experiment)} already running")
            return False

        self.experiment = experiment_type(self.core, self, self.control, *args, **kwargs)
        self.logger.important(f"Start Experiment of type {type(self.experiment).__name__}...")

        self.experiment.start()
        self.experiment.callbacks.stopped.register(self._currentExperimentStopped)
        self.experiment.events.status_changed.on(self.events.experiment_status_changed.set)

        self.events.experiment_started.set()
        self.status = BILBO_ExperimentHandler_Status.EXPERIMENT_RUNNING
        self.events.status_changed.set(data=self.status, flags={'status': self.status})

        return True

    # ------------------------------------------------------------------------------------------------------------------
    def stopExperiment(self):
        self.logger.info("Aborting Experiment ...")
        if self.experiment is None:
            self.logger.warning("No experiment running")
            return

        self.experiment.stop(aborted=True)

    # ------------------------------------------------------------------------------------------------------------------
    def _currentExperimentStopped(self):
        self.experiment = None
        self.logger.info("Experiment stopped")
        self.events.experiment_finished.set()
        self.status = BILBO_ExperimentHandler_Status.IDLE

        self.events.status_changed.set(data=self.status, flags={'status': self.status})
        self.events.experiment_status_changed.set(data=None)

    # ------------------------------------------------------------------------------------------------------------------
    def runTrajectory(self, trajectory: BILBO_InputTrajectory) -> BILBO_TrajectoryExperimentData | None:
        assert len(trajectory.inputs) <= MAX_STEPS_TRAJECTORY
        assert trajectory.length == len(trajectory.inputs)
        assert trajectory.time_vector.shape[0] == trajectory.length

        self.logger.info(f"Trying to run trajectory \"{trajectory.name}\" on device ...")

        self._loadedTrajectory = trajectory
        # Kick off on the device
        self.device.executeFunction(
            function_name='runTrajectory',
            arguments={'trajectory_data': asdict(trajectory)},
        )

        # Wait for either "finished" or "aborted" for this trajectory id
        res = waitForEvents(
            events=[self.events.ll_trajectory_finished, self.events.ll_trajectory_aborted],
            predicates=[pred_flag_key_equals('trajectory_id', trajectory.id), None],
            wait_for_all=False,
            timeout=float(trajectory.time_vector[-1] + 5.0),
            stale_event_time=0.5,  # catch just-before waits
        )

        if res.timeout or not res.ok:
            self.logger.error(f"Trajectory \"{trajectory.name}\" failed due to timeout")
            return None

        if res.first and res.first.event is self.events.ll_trajectory_aborted:
            self.logger.error(f"Trajectory \"{trajectory.name}\" aborted")
            return None

        # Finished: use the exact snapshot payload from the matched event
        finished_match = res.first  # first is the finished event here
        data = (finished_match.data if finished_match else self.events.ll_trajectory_finished.getData()) or {}

        output_data_dict: dict | None = data.get('data', None)

        if output_data_dict is None:
            self.logger.error(f"Trajectory \"{trajectory.name}\" failed due to missing data")
            return None

        experiment_data = self.getTrajectoryExperimentDataFromDict(output_data_dict)

        # plotTrajectoryExperimentData(experiment_data, show_figure=True)

        self.events.trajectory_finished.set(data=experiment_data, flags={'trajectory_id': trajectory.id})

        self.logger.important(f"Trajectory \"{trajectory.name}\" finished.")
        return experiment_data.data

    # ------------------------------------------------------------------------------------------------------------------
    def runRandomTestTrajectory(self, time_s, frequency=2, gain=0.25):

        trajectory = generateRandomTestTrajectory(1, time_s, frequency, gain)
        self.logger.info(
            f"Generated random trajectory: {trajectory.id} (Length: {trajectory.time_vector[-1]} s). "
            f"Waiting for resume event...")

        self._loadedTrajectory = trajectory
        self.events.trajectory_loaded.set(data=trajectory)
        self.events.waiting_for_user.set(data=trajectory)

        self.core.interface_events.resume.wait(timeout=None)
        return self.runTrajectory(trajectory=trajectory)

    # ------------------------------------------------------------------------------------------------------------------
    def startTrajectory(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def sendTrajectory(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def stopTrajectory(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def _trajectory_event_callback(self, message, *args, **kwargs):
        if 'event' not in message.data:
            self.logger.error(f"Robot {self.id}: Received trajectory event without event field")

        match message.data['event']:
            case 'finished':
                self.logger.info(f"Trajectory {message.data['trajectory_id']} finished.")
                # speak(f"{self.id}: Trajectory {message.data['trajectory_id']} finished")
                self.current_trajectory = None
                self._loadedTrajectory = None
                # self.status = BILBO_ExperimentHandler_Status.IDLE
                self.events.ll_trajectory_finished.set(data=message.data,
                                                       flags={'trajectory_id': message.data['trajectory_id']})
                self.events.status_changed.set(data=self.status, flags={'status': self.status})

            case 'started':
                self.logger.info(f"Trajectory {message.data['trajectory_id']} started.")
                # speak(f"{self.id}: Trajectory {message.data['trajectory_id']} started")
                self.current_trajectory = self._loadedTrajectory
                # self.status = BILBO_ExperimentHandler_Status.RUNNING
                self.events.ll_trajectory_started.set(data=message.data,
                                                      flags={'trajectory_id': message.data['trajectory_id']})
                self.events.status_changed.set(data=self.status, flags={'status': self.status})

            case 'aborted':
                self.logger.info(f"Trajectory {message.data['trajectory_id']} aborted.")
                speak(f"{self.id}: Trajectory {message.data['trajectory_id']} aborted")
                self.current_trajectory = None
                self._loadedTrajectory = None
                # self.status = BILBO_ExperimentHandler_Status.IDLE
                self.events.ll_trajectory_aborted.set(data=message.data,
                                                      flags={'trajectory_id': message.data['trajectory_id']})
                self.events.status_changed.set(data=self.status, flags={'status': self.status})

    # ------------------------------------------------------------------------------------------------------------------
    def getCurrentTrajectory(self) -> BILBO_InputTrajectory | None:
        return self.current_trajectory

    # ------------------------------------------------------------------------------------------------------------------
    def getLoadedTrajectory(self) -> BILBO_InputTrajectory | None:
        return self._loadedTrajectory

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def getTrajectoryExperimentDataFromDict(data: dict) -> BILBO_TrajectoryExperiment:
        config = Config(
            cast=[int, float],  # allow casting numbers where JSON gives str/float
            strict=False,  # ignore unknown fields if the device returns extra data
        )
        return from_dict_auto(BILBO_TrajectoryExperiment, data)
