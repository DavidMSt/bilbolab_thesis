import dataclasses
import enum

import numpy as np
import math
from dacite import Config

from core.utils.dataclass_utils import from_dict
from core.utils.files import fileExists, deleteFile, relativeToFullPath
from core.utils.json_utils import readJSON, writeJSON
from robot.settings import config_path


class BILBO_Shields(enum.StrEnum):
    BILBO_SHIELD_REV2 = 'bilbo_shield_rev2'


@dataclasses.dataclass
class ButtonSettings:
    type: str | None  # Can be 'internal', 'sx1508', 'sx1509' or None
    pin: int | None


@dataclasses.dataclass
class Buttons:
    primary: ButtonSettings
    secondary: ButtonSettings


@dataclasses.dataclass
class DisplaySettings:
    active: bool
    resolution: list | None


@dataclasses.dataclass
class SoundSettings:
    active: bool
    gain: float | None


@dataclasses.dataclass
class BILBO_Hardware_Electronics:
    board_revision: str
    shield: str | None
    display: DisplaySettings
    sound: SoundSettings
    battery_cells: int
    buttons: Buttons


@dataclasses.dataclass
class Model:
    type: str
    theta_offset: float = 0.0


@dataclasses.dataclass
class BILBO_Hardware:
    model: Model
    electronics: BILBO_Hardware_Electronics


# ----------------------------------------------------------------------------------------------------------------------
HARDWARE_BILBO1 = BILBO_Hardware(
    model=Model(type='normal', theta_offset=0.0),
    electronics=BILBO_Hardware_Electronics(
        board_revision='v4',
        shield=BILBO_Shields.BILBO_SHIELD_REV2,
        display=DisplaySettings(active=True, resolution=[128, 64]),
        sound=SoundSettings(active=True, gain=0.5),
        battery_cells=4,
        buttons=Buttons(
            primary=ButtonSettings(type='internal', pin=5),
            secondary=ButtonSettings(type='internal', pin=4),
        )
    )
)

HARDWARE_BILBO2 = BILBO_Hardware(
    model=Model(type='normal', theta_offset=0.0),
    electronics=BILBO_Hardware_Electronics(
        board_revision='v4',
        shield=BILBO_Shields.BILBO_SHIELD_REV2,
        display=DisplaySettings(active=True, resolution=[128, 64]),
        sound=SoundSettings(active=True, gain=0.5),
        battery_cells=4,
        buttons=Buttons(
            primary=ButtonSettings(type='internal', pin=5),
            secondary=ButtonSettings(type='internal', pin=4),
        )
    )
)

HARDWARE_MINI_BILBO = BILBO_Hardware(
    model=Model(type='small', theta_offset=0.0),
    electronics=BILBO_Hardware_Electronics(
        board_revision='v4',
        shield=None,
        display=DisplaySettings(active=True, resolution=[128, 64]),
        sound=SoundSettings(active=True, gain=0.5),
        battery_cells=3,
        buttons=Buttons(
            primary=ButtonSettings(type='internal', pin=5),
            secondary=ButtonSettings(type='internal', pin=4),
        )
    )
)

HARDWARE_BIG_BILBO = BILBO_Hardware(
    model=Model(type='big', theta_offset=np.deg2rad(4.0)),
    electronics=BILBO_Hardware_Electronics(
        board_revision='v4',
        shield=BILBO_Shields.BILBO_SHIELD_REV2,
        display=DisplaySettings(active=False, resolution=None),
        sound=SoundSettings(active=False, gain=None),
        battery_cells=4,
        buttons=Buttons(
            primary=ButtonSettings(type='internal', pin=5),
            secondary=ButtonSettings(type='internal', pin=4),
        )
    )
)


# ----------------------------------------------------------------------------------------------------------------------
def generateHardwareDefinition(bilbo_type: str) -> BILBO_Hardware | None:
    if bilbo_type == 'BILBO1':
        hardware_definition = HARDWARE_BILBO1
    elif bilbo_type == 'BILBO2':
        hardware_definition = HARDWARE_BILBO2
    elif bilbo_type == 'MINI':
        hardware_definition = HARDWARE_MINI_BILBO
    elif bilbo_type == 'BIG':
        hardware_definition = HARDWARE_BIG_BILBO
    else:
        raise ValueError("Bilbo type must be either 'BILBO1', 'BILBO2', 'MINI' or 'BIG'")

    return hardware_definition


# ----------------------------------------------------------------------------------------------------------------------
def writeHardwareDefinition(hardware_definition: BILBO_Hardware):
    writeJSON(relativeToFullPath(f"{config_path}hardware.json"), dataclasses.asdict(hardware_definition))


# ----------------------------------------------------------------------------------------------------------------------
def readHardwareDefinition() -> BILBO_Hardware | None:
    file = relativeToFullPath(f"{config_path}hardware.json")
    if not fileExists(file):
        return None

    data_dict = readJSON(file)

    dacite_config = Config(
        # Will convert "bilbo_shield_rev2" -> BILBO_Shields.BILBO_SHIELD_REV2
        # (Safe even if the field is currently typed as `str`)
        type_hooks={
            BILBO_Shields: BILBO_Shields,
        },
        strict=True,  # complain if extra keys are present
        check_types=True  # validate field types
    )

    return from_dict(BILBO_Hardware, data_dict, config=dacite_config)


if __name__ == '__main__':
    generateHardwareDefinition('BIG')
