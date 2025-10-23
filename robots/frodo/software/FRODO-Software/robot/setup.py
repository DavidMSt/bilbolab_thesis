import dataclasses
import json

import cv2
import numpy as np

# from core.utils.dataclass_utils import asdict_optimized
from core.utils.network import get_own_hostname
from robot.definitions import FRODO_Definition, FRODO_COLORS, CAMERA_SETTINGS, ARUCO_SETTINGS, OPTITRACK_SETTINGS, \
    FRODO_MODEL_GENERAL
from robot.sensing.camera.pycamera import PyCameraType
from robot.settings import settings_file_path


def setup(robot_id: str):
    settings = FRODO_Definition(
        id=robot_id,
        color=FRODO_COLORS[robot_id],
        camera=CAMERA_SETTINGS[robot_id],
        aruco=ARUCO_SETTINGS[robot_id],
        optitrack=OPTITRACK_SETTINGS[robot_id],
        physical_model=FRODO_MODEL_GENERAL,
    )
    settings_dict = dataclasses.asdict(settings)
    file = settings_file_path

    with open(file, 'w') as f:
        json.dump(settings_dict, f, indent=4)


def setup_interactive():
    robot_id = get_own_hostname()
    print(f"Setting up robot {robot_id}")
    if robot_id not in FRODO_COLORS:
        print(f"Robot {robot_id} not found.")
        return
    setup(robot_id)


if __name__ == '__main__':
    setup_interactive()
