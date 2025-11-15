import ctypes
import dataclasses
import enum
import time
from typing import Any, cast
from numpy.core.defchararray import isnumeric

# ======================================================================================================================
from robot.communication.bilbo_communication import BILBO_Communication
from robot.communication.serial.bilbo_serial_messages import BILBO_Sequencer_Event_Message
from robot.control.bilbo_control import BILBO_Control
from robot.control.bilbo_control_data import BILBO_Control_Mode
from robot.experiment.definitions import BILBO_InputTrajectory, BILBO_InputTrajectoryStep
from robot.lowlevel.stm32_general import MAX_STEPS_TRAJECTORY, LOOP_TIME_CONTROL
from robot.lowlevel.stm32_sequencer import bilbo_sequence_input_t, bilbo_sequence_description_t, BILBO_Sequence_LL
import robot.lowlevel.stm32_addresses as addresses
from robot.utilities.bilbo_utilities import BILBO_Utilities
from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.events import event_definition, Event, pred_flag_equals, EventFlag
from core.utils.logging_utils import Logger
from core.utils.dataclass_utils import from_dict
from core.utils.data import generate_random_input, generate_time_vector
from core.communication.wifi.data_link import CommandArgument


# ======================================================================================================================


class BILBO_LL_Sequencer_Event_Type(enum.IntEnum):
    STARTED = 1
    FINISHED = 2
    ABORTED = 3
    RECEIVED = 4


# ======================================================================================================================
logger = Logger('EXPERIMENT')
logger.setLevel('DEBUG')

# TODO: ADD OUTPUTS TO THE TRAJECTORY

# === EXPERIMENT =======================================================================================================
class BILBO_Experiment_Type(enum.IntEnum):
    FREE = 1
    TRAJECTORY = 2
    MULTI_TRAJECTORY = 3


@dataclasses.dataclass
class BILBO_Experiment:
    name: str
    type: BILBO_Experiment_Type


class BILBO_ExperimentMode(enum.StrEnum):
    RUNNING = 'RUNNING'
    ERROR = 'ERROR'
    IDLE = 'IDLE'

# === CALLBACKS ========================================================================================================
@callback_definition
class BILBO_ExperimentCallbacks:
    trajectory_started: CallbackContainer
    trajectory_finished: CallbackContainer
    trajectory_aborted: CallbackContainer
    trajectory_loaded: CallbackContainer


@event_definition
class BILBO_ExperimentEvents:
    trajectory_started: Event = Event(flags=EventFlag('trajectory_id', (str, int)))
    trajectory_finished: Event = Event(flags=EventFlag('trajectory_id', (str, int)))
    trajectory_aborted: Event = Event(flags=EventFlag('trajectory_id', (str, int)))
    trajectory_loaded: Event = Event(flags=EventFlag('trajectory_id', (str, int)))


# === BILBO_ExperimentHandler ==========================================================================================
class BILBO_ExperimentHandler:
    communication: BILBO_Communication
    callbacks: BILBO_ExperimentCallbacks
    events: BILBO_ExperimentEvents
    current_trajectory: BILBO_InputTrajectory = None

    running: bool = False

    mode: BILBO_ExperimentMode = BILBO_ExperimentMode.IDLE

    def __init__(self, communication: BILBO_Communication, utils: BILBO_Utilities, control: BILBO_Control):
        self.communication = communication
        self.callbacks = BILBO_ExperimentCallbacks()
        self.events = BILBO_ExperimentEvents()
        self.utils = utils
        self.logging = None
        self.control = control

        self.communication.serial.callbacks.event.register(self._sequencer_event_callback,
                                                           parameters={'messages': [BILBO_Sequencer_Event_Message]})

        self.communication.wifi.newCommand(
            identifier='runTrajectory',
            arguments=[
                CommandArgument(name='trajectory_id',
                                type=int,
                                optional=False,
                                description='ID of the trajectory to run'),
                CommandArgument(name='input',
                                type=list,
                                optional=False,
                                description='Input Left/Right to the trajectory'),
                CommandArgument(name='signals',
                                type=list,
                                optional=True,
                                description='Signals to return from the trajectory'),
            ],
            function=self._run_trajectory_external,
            description='Run a trajectory',
            execute_in_thread=True
        )

    # ------------------------------------------------------------------------------------------------------------------
    def init(self, logging):
        self.logging = logging

    # ------------------------------------------------------------------------------------------------------------------
    def loadTrajectoryFromFile(self) -> BILBO_Trajectory:
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def saveTrajectoryToFile(self, trajectory: BILBO_Trajectory, filename: str):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def loadTrajectoryToLL(self, trajectory: BILBO_InputTrajectory) -> bool:

        logger.info(f"Loading trajectory {trajectory.id} ... ")

        # First, check the trajectory length
        if trajectory.length != len(trajectory.inputs):
            logger.warning(f"Trajectory length does not match number of inputs. "
                           f"Trajectory length: {trajectory.length}, Number of inputs: {len(trajectory.inputs)}")
            return False

        # Load the trajectory data to the stm32
        success = self._setTrajectoryDescription_LL(trajectory)

        if not success:
            logger.warning("Failed to set trajectory")
            return False


        # Send the trajectory inputs via SPI
        trajectory_bytes = self._trajectoryInputToBytes(trajectory.inputs)
        self.communication.spi.sendTrajectoryData(trajectory.length, trajectory_bytes)

        success = self.events.trajectory_loaded.wait(
            timeout=2,
            stale_event_time=0.2,
            predicate=pred_flag_equals('trajectory_id', trajectory.id),
        )

        if not success:
            logger.warning("Failed to load trajectory")
            return False

        logger.info(f"Trajectory {trajectory.id} loaded!")
        return True

    # ------------------------------------------------------------------------------------------------------------------
    def startTrajectoryOnLL(self, trajectory_id) -> bool:

        # First, check if a trajectory is loaded
        trajectory_data = self._readTrajectoryDescription_LL()

        if trajectory_data is None:
            logger.warning("No trajectory loaded")
            return False

        if trajectory_data.sequence_id != trajectory_id:
            logger.warning(f"Wrong trajectory id. Expected {trajectory_id}, loaded: {trajectory_data.sequence_id}")

        if not trajectory_data.loaded:
            logger.warning("Trajectory not loaded")

        logger.debug("Start trajectory")

        success = self._startTrajectory_LL(trajectory_id)

        if not success:
            logger.warning("Failed to start trajectory")
            return False

        return True

    # ------------------------------------------------------------------------------------------------------------------
    def stopTrajectory(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def update(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def generateTestTrajectory(self, id, time, frequency=2, gain=0.25) -> BILBO_InputTrajectory | None:
        # generate a simple trajectory

        t_vector = generate_time_vector(start=0, end=time, dt=LOOP_TIME_CONTROL)

        if len(t_vector) > MAX_STEPS_TRAJECTORY:
            logger.warning("Trajectory too long")
            return None

        input = generate_random_input(t_vector=t_vector, f_cutoff=frequency, sigma_I=gain)

        trajectory_inputs = self._generateTrajectoryInputs(input)

        trajectory = BILBO_InputTrajectory(
            id=id,
            name='test',
            length=len(trajectory_inputs),
            inputs=trajectory_inputs,
            control_mode=BILBO_Control_Mode.BALANCING,
        )

        return trajectory

    # ------------------------------------------------------------------------------------------------------------------
    def runTrajectory(self, trajectory: BILBO_InputTrajectory, signals: list | str = None) -> dict | None:

        if signals is not None and not isinstance(signals, list):
            signals = [signals]

        logger.info(f"Running trajectory {trajectory.id} ...")

        # Set the trajectory data in the LL Module
        success = self.loadTrajectoryToLL(trajectory)

        if not success:
            logger.warning(f"Failed to load trajectory {trajectory.id}")
            return None

        # Start the trajectory
        success = self.startTrajectoryOnLL(trajectory.id)

        if not success:
            logger.warning(f"Failed to run trajectory {trajectory.id}")
            return None

        success = self.events.trajectory_started.wait(
            timeout=2,
            stale_event_time=0.2,
            predicate=pred_flag_equals('trajectory_id', trajectory.id),
        )

        if not success:
            logger.warning(f"Failed to start trajectory {trajectory.id}")
            return None

        start_tick = self.events.trajectory_started.getData()['tick']

        success = self.events.trajectory_finished.wait(
            timeout=trajectory.length * 0.01 + 2,
            stale_event_time=0.2,
            predicate=pred_flag_equals('trajectory_id', trajectory.id),
        )

        if not success:
            logger.warning(f"Failed to finish trajectory {trajectory.id}")
            return None

        end_tick = self.events.trajectory_finished.getData()['tick']

        if start_tick is None or end_tick is None:
            return None

        # Wait for the logger to reach the number of samples
        while self.logging.sample_index < (end_tick + 100):
            time.sleep(0.1)

        output_signals = {}
        if signals is not None:
            output_signals = self.logging.get_data(
                signals=signals,
                index_start=start_tick,
                index_end=end_tick
            )

        output_data = {
            'start_tick': start_tick,
            'end_tick': end_tick,
            'trajectory': trajectory,
            'output': output_signals
        }

        # Send the event via Wi-Fi
        try:
            self.communication.wifi.sendEvent(event='trajectory',
                                              data={
                                                  'event': 'finished',
                                                  'trajectory_id': trajectory.id,
                                                  'input': [[float(inp.left), float(inp.right)] for inp in
                                                            trajectory.inputs.values()],
                                                  'output': output_signals,
                                              })
        except Exception as e:
            logger.error(f"Failed to send trajectory data back to the server: {e}")

        return output_data

    # ------------------------------------------------------------------------------------------------------------------
    def getSample(self):
        sample = {}
        return sample

    # === PRIVATE METHODS ==============================================================================================
    def _setTrajectoryDescription_LL(self, trajectory: BILBO_InputTrajectory) -> bool:
        # Transform the trajectory into the corresponding ctypes structure

        sequence_description = bilbo_sequence_description_t(
            sequence_id=trajectory.id,
            length=trajectory.length,
            require_control_mode=False,
            wait_time_beginning=1,
            wait_time_end=1,
            control_mode=trajectory.control_mode.value,
            control_mode_end=trajectory.control_mode.value,
            loaded=False
        )

        # Send the trajectory to the STM32
        success = self.communication.serial.executeFunction(
            module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
            address=addresses.TWIPR_SequencerAddresses.LOAD,
            data=sequence_description,
            input_type=bilbo_sequence_description_t,  # type: ignore
            output_type=ctypes.c_bool,
            timeout=0.1
        )

        if not success:
            logger.warning(f'Failed to set trajectory description {trajectory.id}')
            return False
        else:
            ...
            # logger.debug(f'Trajectory description {trajectory.id} transferred')

        return True

    # ------------------------------------------------------------------------------------------------------------------
    def _startTrajectory_LL(self, trajectory_id) -> bool:
        self.running = True
        self.control.enable_external_input = False

        success = self.communication.serial.executeFunction(
            module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
            address=addresses.TWIPR_SequencerAddresses.START,
            data=trajectory_id,
            input_type=ctypes.c_uint16,
            output_type=ctypes.c_bool,
            timeout=0.1
        )

        return success

    # ------------------------------------------------------------------------------------------------------------------
    def _stopTrajectory_LL(self) -> bool:
        success = self.communication.serial.executeFunction(
            module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
            address=addresses.TWIPR_SequencerAddresses.STOP,
            data=None,
            input_type=None,
            output_type=None,
            timeout=0.1
        )

        return success

    # ------------------------------------------------------------------------------------------------------------------
    def _readTrajectoryDescription_LL(self) -> BILBO_Sequence_LL | None:

        logger.debug("Get trajectory data")
        trajectory_data_struct = self.communication.serial.executeFunction(
            module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
            address=addresses.TWIPR_SequencerAddresses.READ,
            data=None,
            input_type=None,
            output_type=bilbo_sequence_description_t,
            timeout=0.1
        )

        if trajectory_data_struct is None:
            logger.warning("Failed to get trajectory data")
            return None

        trajectory = from_dict(data=trajectory_data_struct, data_class=BILBO_Sequence_LL)

        return trajectory

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _trajectoryInputToBytes(trajectory_input: dict[int, BILBO_InputTrajectoryStep]) -> bytes:
        # Create a ctypes array type of the correct length

        ArrayType: Any = cast(Any, bilbo_sequence_input_t * len(trajectory_input))  # type: ignore
        c_array = ArrayType()  # Now this won't raise a warning

        # Populate the ctypes array with data from trajectory_input
        for i, inp in trajectory_input.items():
            c_array[i].step = inp.step
            c_array[i].u_1 = inp.left
            c_array[i].u_2 = inp.right

        # Get the byte representation of the array
        bytes_data = ctypes.string_at(ctypes.byref(c_array), ctypes.sizeof(c_array))
        return bytes_data

    # ------------------------------------------------------------------------------------------------------------------
    def _sendTrajectoryInputs_ll(self, trajectory: BILBO_InputTrajectory):
        trajectory_bytes = self._trajectoryInputToBytes(trajectory.inputs)
        self.communication.spi.sendTrajectoryData(trajectory.length, trajectory_bytes)

    # ------------------------------------------------------------------------------------------------------------------
    def _generateStep(self, gain, length):
        trajectory_input = {}

        for i in range(length):
            trajectory_input[i] = BILBO_InputTrajectoryStep(
                step=i,
                left=gain * 1.0,
                right=gain * 1.0
            )

        return trajectory_input

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _generateTrajectoryInputs(input_list: list):
        trajectory_inputs = {}

        for i, input in enumerate(input_list):
            if isinstance(input, list):
                input_left = float(input[0])
                input_right = float(input[1])
            else:
                input_left = float(input) / 2
                input_right = float(input) / 2

            trajectory_inputs[i] = BILBO_InputTrajectoryStep(
                step=i,
                left=input_left,
                right=input_right,
            )
        return trajectory_inputs

    # ------------------------------------------------------------------------------------------------------------------
    def _run_trajectory_external(self, trajectory_id, input, signals=None):
        print("Run Trajectory External")

        if signals is not None and not isinstance(signals, list):
            signals = [signals]

        trajectory = BILBO_InputTrajectory(
            id=trajectory_id,
            name='test',
            length=len(input),
            inputs=self._generateTrajectoryInputs(input),
            control_mode=BILBO_Control_Mode.BALANCING,
        )

        self.runTrajectory(trajectory, signals=signals)

    # ------------------------------------------------------------------------------------------------------------------
    def _sequencer_event_callback(self, message: BILBO_Sequencer_Event_Message, *args, **kwargs):
        event = BILBO_LL_Sequencer_Event_Type(message.data['event']).name
        trajectory_id = message.data['sequence_id']
        tick = message.data['tick']
        if event == 'STARTED':
            # self.utils.speak(f"Trajectory {trajectory_id} started")
            logger.info(f"Trajectory {trajectory_id} started (Tick: {tick})")
            self.callbacks.trajectory_started.call(trajectory_id=trajectory_id, tick=tick)

            self.events.trajectory_started.set(data={'tick': tick, 'trajectory_id': trajectory_id},
                                               flags={'trajectory_id': trajectory_id})

        elif event == 'FINISHED':
            # self.utils.speak(f"Trajectory {trajectory_id} finished")
            logger.info(f"Trajectory {trajectory_id} finished (Tick: {tick})")

            self.callbacks.trajectory_finished.call(trajectory_id=trajectory_id, tick=tick)
            self.events.trajectory_finished.set(data={'tick': tick, 'trajectory_id': trajectory_id},
                                                flags={'trajectory_id': trajectory_id})

            self.running = False
            self.control.enable_external_input = True

        elif event == 'RECEIVED':
            logger.debug(f"Trajectory {trajectory_id} loaded")
            self.callbacks.trajectory_loaded.call(trajectory_id=trajectory_id, tick=tick)
            self.events.trajectory_loaded.set(data={'tick': tick, 'trajectory_id': trajectory_id},
                                              flags={'trajectory_id': trajectory_id})

        elif event == 'ABORTED':
            # self.utils.speak(f"Trajectory {trajectory_id} aborted")
            logger.info(f"Trajectory {trajectory_id} aborted")

            self.callbacks.trajectory_aborted.call(trajectory_id=trajectory_id, tick=tick)
            self.events.trajectory_aborted.set(data={'tick': tick, 'trajectory_id': trajectory_id},
                                               flags={'trajectory_id': trajectory_id})

            self.running = False
            self.control.enable_external_input = True
# ======================================================================================================================
