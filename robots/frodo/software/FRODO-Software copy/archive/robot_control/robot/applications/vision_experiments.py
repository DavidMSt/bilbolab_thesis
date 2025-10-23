import threading
import time

from bilbo_old.VisionRobot.VisionRobot import VisionRobot


class VisionAgentExperiment:
    #Jarne
    experiment_id: str
    movement_data: list  #

    _thread: threading.Thread

    def __init__(self):
        ...

    def init(self):
        ...
        self._thread = threading.Thread(target=self._threadFunction())

    def start(self):
        self._thread.start()

    def update(self):
        ...

    def _threadFunction(self):
        while True:
            self.update()
            time.sleep(0.1)
        ...


class VisionAgent:
    robot: VisionRobot  # Hardware
    arucoDetection: ...  # Camera Hardware and Software
    experiment: VisionAgentExperiment  # Execution of predefined movement experiment
    estimation: ...  # Estimation Algorithm for Localization
    
    def __init__(self):
        self.robot = VisionRobot()
        ...

    # Distinct Phases

    def phase_start(self):
        ...

    def phase_prediction(self):
        # Dustin
        ...

    def phase_cameraMeasurement(self):
        # Jarne
        ...

    def phase_communication(self):
        # Dustin
        ...

    def phase_estimation(self):
        # Dustin
        ...

    def phase_end(self):
        ...
