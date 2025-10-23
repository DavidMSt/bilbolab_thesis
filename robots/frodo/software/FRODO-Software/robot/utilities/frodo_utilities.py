from core.communication.wifi.data_link import CommandArgument
from robot.common import FRODO_Common
from robot.communication.frodo_communication import FRODO_Communication
from robot.frodo_core.frodo_core import FRODO_Core


class FRODO_Utilities:

    # === INIT =========================================================================================================
    def __init__(self, common: FRODO_Common, communication: FRODO_Communication, core: FRODO_Core):
        self.common = common
        self.communication = communication
        self.core = core

        self._addWifiCommands()

    # === METHODS ======================================================================================================
    def beep(self, frequency: str = 'medium', time_ms: int = 500, repeats: int = 1):
        self.core.lowlevel.board.beep(frequency, time_ms, repeats)

    # === PRIVATE METHODS ==============================================================================================
    def _addWifiCommands(self):
        self.communication.wifi.newCommand(identifier='beep',
                                           function=self.beep,
                                           description='',
                                           arguments=[
                                               CommandArgument(name='frequency',
                                                               type=str,
                                                               optional=True,
                                                               default='medium',
                                                               description='Frequency of the beep. Can be "low", "medium", "high" or a number in Hz'),
                                               CommandArgument(name='time_ms',
                                                               type=int,
                                                               optional=True,
                                                               default=500,
                                                               description='Duration of the beep in ms'),
                                               CommandArgument(name='repeats',
                                                               type=int,
                                                               optional=True,
                                                               default=1,
                                                               description='Number of repeats')
                                           ])
