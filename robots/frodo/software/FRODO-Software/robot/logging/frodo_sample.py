import dataclasses


# === CUSTOM MODULES ===================================================================================================
from robot.estimation.frodo_estimation import FRODO_Estimation_Sample
from robot.sensing.frodo_sensors import FRODO_Measurements_Sample
from robot.control.frodo_control import FRODO_Control_Sample

# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class FRODO_Sample_General:
    id: str
    step: int
    time: float
    battery: float
    connection_strength: float
    internet_connection: bool

# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class FRODO_Sample:
    general: FRODO_Sample_General
    estimation: FRODO_Estimation_Sample
    measurements: FRODO_Measurements_Sample
    control: FRODO_Control_Sample
