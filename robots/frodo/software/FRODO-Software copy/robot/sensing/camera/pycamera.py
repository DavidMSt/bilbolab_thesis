import enum
import threading
import time


def disableLibcameraLogs():
    import os
    os.environ["LIBCAMERA_LOG_LEVELS"] = "*:4"


disableLibcameraLogs()

import cv2
from libcamera import controls
from picamera2 import picamera2
from robot.utilities.video_streamer.video_streamer import VideoStreamer
from utils.logging_utils import Logger

# ======================================================================================================================
logger = Logger("PyCamera")
logger.setLevel('DEBUG')


class PyCameraType(enum.Enum):
    V1 = 1,
    V2 = 2,
    V3 = 3,
    GS = 4,


# ======================================================================================================================
class PyCamera:
    picam: picamera2.Picamera2
    resolution: tuple
    version: PyCameraType
    running: bool = False
    _camera_lock: threading.Lock

    def __init__(self, version: PyCameraType, resolution: tuple, auto_focus: bool = False, exposure_time: int = None,
                 gain: int = None, ):

        self.version = version
        self.resolution = resolution

        self.picam = picamera2.Picamera2()

        if self.version == PyCameraType.V2:
            self.picam_config = self.picam.create_video_configuration(raw={"size": (1640, 1232)},
                                                                      main={"format": "RGB888", "size": resolution},
                                                                      buffer_count=5)
        elif self.version == PyCameraType.V3:
            self.picam_config = self.picam.create_video_configuration(raw={"size": (2304, 1296)},
                                                                      main={"format": "RGB888", "size": resolution},
                                                                      buffer_count=5)
        elif self.version == PyCameraType.V1:
            self.picam_config = self.picam.create_video_configuration(raw={"size": (2592, 1944)},
                                                                      main={"format": "RGB888", "size": resolution},
                                                                      buffer_count=5)

        elif self.version == PyCameraType.GS:
            # self.picam_config = self.picam.create_video_configuration(raw={"size": (1456, 1088)},
            #                                                           main={"format": "YUV420", "size": resolution},
            #                                                           buffer_count=5)
            self.picam_config = self.picam.create_video_configuration(raw={"size": (1456, 1088)},
                                                                      main={"format": "RGB888", "size": resolution},
                                                                      buffer_count=5)

        self.picam.configure(self.picam_config)

        new_controls = {}
        if exposure_time is not None:
            new_controls["ExposureTime"] = exposure_time
        if gain is not None:
            new_controls["AnalogueGain"] = gain
        if auto_focus:
            new_controls["AfMode"] = controls.AfModeEnum.Continuous

        self.picam.set_controls(new_controls)

        self._camera_lock = threading.Lock()

    # === METHODS ======================================================================================================
    def init(self):
        ...

    def start(self):
        self.running = True
        self.picam.start()
        logger.info("PyCamera started!")
        self.getCameraSettings()

    def getCameraSettings(self):
        metadata = self.picam.capture_metadata()
        exposure_time = metadata.get("ExposureTime")
        analogue_gain = metadata.get("AnalogueGain")

        data = {
            'exposure_time': exposure_time,
            'analogue_gain': analogue_gain,
        }
        print(f"Exposure Time: {exposure_time} µs")
        print(f"Analogue Gain: {analogue_gain}")

        return data

    def takeFrame(self):
        with self._camera_lock:
            frame = self.picam.capture_array()
            if self.version == PyCameraType.GS:
                # YUV420 format: Y channel is first plane
                return frame
            return self.picam.capture_array()

    @staticmethod
    def getImageBuffer(frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer

    def getImageBufferBytes(self, frame):
        return self.getImageBuffer(frame).tobytes()


# ======================================================================================================================
class PyCameraStreamer(VideoStreamer):
    pycamera: PyCamera

    def __init__(self, pycamera: PyCamera = None, resolution: tuple = None):
        super().__init__()

        if pycamera is None:
            if resolution is None:
                resolution = (960, 540)
            self.camera = PyCamera(PyCameraType.V3, resolution, auto_focus=True)
        else:
            self.camera = pycamera

        self.image_fetcher = self.getCameraFrame

    def start(self):
        if not self.camera.running:
            self.camera.start()

        super().start()

    def getCameraFrame(self):
        frame = self.camera.takeFrame()
        return self.camera.getImageBufferBytes(frame)


if __name__ == '__main__':
    # camera = PyCamera(PyCameraType.GS, (1456, 1088), exposure_time=4000, gain=10)
    camera = PyCamera(PyCameraType.GS, (1456, 1088), exposure_time=4000, gain=10)
    # camera = PyCamera(PyCameraType.GS, (1456, 1088))
    streamer = PyCameraStreamer(pycamera=camera)
    streamer.start()
    while True:
        time.sleep(10)
