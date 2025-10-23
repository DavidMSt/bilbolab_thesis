import dataclasses
import os
import threading
import time

import cv2
import numpy as np
import yaml

# ======================================================================================================================
from robot.sensing.camera.pycamera import PyCameraType, PyCamera, PyCameraStreamer
from utils.files import dirExists, makeDir, removeDir
from robot.sensing.aruco.calibration.calibration_utils.chess_calibration import chessboard_calibration
from paths import calibrations_path as global_calibration_dir


@dataclasses.dataclass
class CameraCalibrationData:
    camera_matrix: np.array
    dist_coeff: np.array
    resolution: tuple[int, int]


# ======================================================================================================================
class ArucoCalibration:
    camera_version: PyCameraType
    resolution: tuple

    camera: PyCamera

    _streamer: PyCameraStreamer
    _camera_lock: threading.Lock
    _exit: bool = False

    def __init__(self, camera_version: PyCameraType, resolution: tuple):
        self.camera_version = camera_version
        self.resolution = resolution

        self.calibration_name = self.getCalibrationName(camera_version, resolution)
        print(f"Calibration Name: {self.calibration_name}")

        self.calibration_folder = self._getCalibrationDirPath(self.calibration_name)
        self.image_folder = self._getCalibrationDirImagePath(self.calibration_folder)
        self.calibration_file = self._getCalibrationFilePath(self.calibration_name)
        self._makeCalibrationDir(self.calibration_name)

        self.camera = PyCamera(version=camera_version, resolution=resolution)

        self._streamer = PyCameraStreamer(pycamera=self.camera)

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.camera.start()
        time.sleep(0.25)
        self._streamer.start()

        self._collectCalibrationPictures()

        self.runChessBoardCalibration(images_path=self.image_folder,
                                      calibration_file=self.calibration_file,
                                      camera=self.camera_version,
                                      resolution=self.resolution, )

        self._exit = True

        print("Calibration Complete")
        os._exit(0)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def runChessBoardCalibration(images_path: str, calibration_file: str, camera: PyCameraType, resolution: tuple):
        chessboard_calibration(images_path=images_path,
                               calibration_file=calibration_file,
                               camera=camera.name,
                               resolution=resolution,
                               )

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def getCalibrationName(camera_version: PyCameraType, resolution: tuple):
        return f"{camera_version.name}_{resolution[0]}x{resolution[1]}"

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def readCalibrationFile(calibration_name: str):
        # First check if the file exists
        calibration_file = ArucoCalibration._getCalibrationFilePath(calibration_name)
        if not os.path.exists(calibration_file):
            return None

        with open(calibration_file) as f:
            loadeddict = yaml.safe_load(f)

        resolution = loadeddict.get("resolution")
        resolution = (resolution[0], resolution[1])

        mtx = loadeddict.get("camera_matrix")
        dist = loadeddict.get("dist_coeff")

        calibration_data = CameraCalibrationData(np.array(mtx), np.array(dist), resolution)

        return calibration_data

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _getCalibrationFilePath(calibration_name: str):
        calibration_dir = ArucoCalibration._getCalibrationDirPath(calibration_name)
        return f"{calibration_dir}/{calibration_name}.yaml"

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _getCalibrationDirPath(calibration_name: str):
        return f"{global_calibration_dir}/camera/{calibration_name}"

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _getCalibrationDirImagePath(calibration_path):
        return f"{calibration_path}/images"

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def _makeCalibrationDir(self, calibration_name: str):

        if not dirExists(f"{global_calibration_dir}/camera"):
            makeDir(f"{global_calibration_dir}/camera")

        dir_path = self._getCalibrationDirPath(calibration_name)

        if dirExists(dir_path):
            print(f"Remove Existing Calibration dir: {dir_path}")
            removeDir(dir_path)

        makeDir(dir_path)
        makeDir(self._getCalibrationDirImagePath(dir_path))
        print(f"Created Calibration dir: {dir_path}")

    def _collectCalibrationPictures(self):
        count = 0
        print("Start Data Generation, press Enter to Capture, write exit to continue with calibration build")
        while True:
            print(f"Press Enter to Capture Image {count + 1}")
            msg = input()
            if msg == 'exit':
                print("Exit Taking Pictures")
                break

            image = self.camera.takeFrame()
            image_file = f"{self.image_folder}/image_{count}.jpg"
            cv2.imwrite(image_file, image)
            print(f"Saved image {image_file}")
            count += 1
            time.sleep(0.1)


if __name__ == '__main__':
    calib = ArucoCalibration(camera_version=PyCameraType.GS,
                             resolution=(960, 540))


    calib.start()
    # ArucoCalibration.runChessBoardCalibration(images_path='/home/admin/robot/calibration/camera/V3_960x540/images',
    #                                           calibration_file='/home/admin/robot/calibration/camera/V3_960x540'
    #                                                            '/V3_960x540.yaml',
    #                                           camera=PyCameraType.V3,
    #                                           resolution=(960, 540))
