# ======================================================================================================================
import dataclasses
import enum
import time

from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import numpy as np

from core.utils.json_utils import writeJSON, readJSON
from robot.bilbo_common import BILBO_Config
from robot.bilbo_definitions import BILBO_DynamicState
from robot.control.bilbo_control_data import BILBO_Control_Mode, BILBO_ControlConfig


# ======================================================================================================================
# LOW LEVEL
# ======================================================================================================================
class BILBO_LL_Sequencer_Event_Type(enum.IntEnum):
    STARTED = 1
    FINISHED = 2
    ABORTED = 3
    RECEIVED = 4


# ======================================================================================================================
# EXPERIMENT
# ======================================================================================================================


# ======================================================================================================================
# EXPERIMENT HANDLER
# ======================================================================================================================
class BILBO_ExperimentHandler_Mode(enum.StrEnum):
    RUNNING = 'RUNNING'
    ERROR = 'ERROR'
    IDLE = 'IDLE'


# ======================================================================================================================

INPUT_TRAJECTORY_FILE_EXTENSION = '.bitrj'


# ======================================================================================================================
@dataclasses.dataclass
class BILBO_InputTrajectoryStep:
    step: int
    left: float
    right: float


# ======================================================================================================================
@dataclasses.dataclass
class BILBO_InputTrajectory:
    name: str
    id: int
    length: int
    time_vector: np.ndarray
    control_mode: BILBO_Control_Mode
    inputs: list[BILBO_InputTrajectoryStep]


# ======================================================================================================================
@dataclasses.dataclass
class BILBO_StateTrajectory:
    time_vector: np.ndarray
    states: list[BILBO_DynamicState]


# ======================================================================================================================
@dataclasses.dataclass
class BILBO_TrajectoryExperimentData:
    input_trajectory: BILBO_InputTrajectory
    state_trajectory: BILBO_StateTrajectory


# ======================================================================================================================
@dataclasses.dataclass
class BILBO_TrajectoryExperimentMeta:
    robot_id: str
    robot_information: BILBO_Config
    control_config: BILBO_ControlConfig
    description: str
    software_revision: str
    timestamp: str


# ======================================================================================================================
@dataclasses.dataclass
class BILBO_TrajectoryExperiment:
    id: str
    meta: BILBO_TrajectoryExperimentMeta
    data: BILBO_TrajectoryExperimentData


# ======================================================================================================================
@dataclasses.dataclass
class BILBO_OutputTrajectory:
    time_vector: np.ndarray
    output_name: str
    output: list[float]


# ======================================================================================================================
# FILES
# ======================================================================================================================
@dataclasses.dataclass
class FrequencyComponent:
    frequency: float
    weight: float  # relative amplitude (normalized to 1)


@dataclasses.dataclass
class BILBO_InputAnalytics:
    steps: int
    Ts: float
    max_amplitude: float
    dominant_frequencies: list[FrequencyComponent]
    is_2d: bool


@dataclasses.dataclass
class BILBO_InputFileMeta:
    date: str
    version: str
    description: str
    experiment_id: str
    experiment_index: int


@dataclasses.dataclass
class BILBO_InputFileData:
    name: str
    meta: BILBO_InputFileMeta
    analytics: BILBO_InputAnalytics
    data: BILBO_InputTrajectory


# ======================================================================================================================
def writeInputFile(file_path, data: BILBO_InputFileData):
    data_dict = dataclasses.asdict(data)

    try:
        writeJSON(file_path, data_dict)
    except Exception as e:
        print(f"Error writing input file: {e}")


# ----------------------------------------------------------------------------------------------------------------------
def readInputFile(file_path) -> BILBO_InputFileData | None:
    try:
        data_dict = readJSON(file_path)
        return BILBO_InputFileData(**data_dict)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None


# ----------------------------------------------------------------------------------------------------------------------
def _analyzeTrajectory(input_trajectory: BILBO_InputTrajectory,
                       num_dominant: int = 5) -> BILBO_InputAnalytics:
    steps = input_trajectory.length
    time_vector = input_trajectory.time_vector
    Ts = float(time_vector[1] - time_vector[0])  # Sampling time

    # Extract signal vectors
    left_signal = np.array([input_trajectory.inputs[i].left for i in sorted(input_trajectory.inputs)])
    right_signal = np.array([input_trajectory.inputs[i].right for i in sorted(input_trajectory.inputs)])
    is_2d = not np.allclose(left_signal, right_signal)

    # Use average of both channels for analysis
    combined_signal = 0.5 * (left_signal + right_signal)

    # FFT analysis
    frequencies = rfftfreq(steps, Ts)
    fft_magnitude = np.abs(rfft(combined_signal))

    # Remove DC component
    fft_magnitude[0] = 0.0

    # Find all peaks above a threshold (e.g., 5% of max)
    peak_indices, _ = find_peaks(fft_magnitude, height=np.max(fft_magnitude) * 0.05)

    if len(peak_indices) == 0:
        dominant_components = []
    else:
        # Sort by amplitude
        sorted_indices = peak_indices[np.argsort(fft_magnitude[peak_indices])[::-1]]

        # Pick top N
        top_indices = sorted_indices[:num_dominant]
        dominant_frequencies = frequencies[top_indices]
        top_amps = fft_magnitude[top_indices]

        # Normalize weights
        total_amp = np.sum(top_amps)
        weights = top_amps / total_amp if total_amp > 0 else np.zeros_like(top_amps)

        dominant_components = [
            FrequencyComponent(frequency=freq, weight=float(weight))
            for freq, weight in zip(dominant_frequencies, weights)
        ]

    # Max signal amplitude (could be RMS, but sticking to peak for now)
    max_amplitude = np.max(np.abs(combined_signal))

    return BILBO_InputAnalytics(
        steps=steps,
        Ts=Ts,
        max_amplitude=max_amplitude,
        dominant_frequencies=dominant_components,
        is_2d=is_2d,
    )


# ----------------------------------------------------------------------------------------------------------------------
def generateInputTrajectoryFileData(input_trajectory: BILBO_InputTrajectory,
                                    name,
                                    description,
                                    experiment_id=None,
                                    experiment_index=None,
                                    version='1.0', ) -> BILBO_InputFileData:
    input_file_meta = BILBO_InputFileMeta(
        date=time.strftime("%Y-%m-%d-%H-%M-%S"),
        version=version,
        description=description,
        experiment_id=experiment_id,
        experiment_index=experiment_index,
    )

    analytics = _analyzeTrajectory(input_trajectory)

    data = BILBO_InputFileData(
        name=name,
        meta=input_file_meta,
        analytics=analytics,
        data=input_trajectory,
    )

    return data
