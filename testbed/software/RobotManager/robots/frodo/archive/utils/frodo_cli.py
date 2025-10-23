# ======================================================================================================================
from extensions.cli.src.cli import Command, CommandSet, CommandArgument
from robots.frodo.archive.frodo import Frodo
from robots.frodo.archive.utils.frodo_utils import test_response_time
from core.utils.callbacks import Callback


# ======================================================================================================================

class FRODO_CommandSet(CommandSet):
    frodo: Frodo

    def __init__(self, frodo: Frodo):
        self.frodo = frodo
        beep_command = Command(name='beep',
                               callback=frodo.beep,
                               description='Beep the internal beeper',
                               arguments=[])

        stop_command = Command(name='stop',
                               callback=Callback(function=frodo.setSpeed,
                                                 parameters={'speed_left': 0,
                                                             'speed_right': 0}, discard_inputs=True),
                               arguments=[])

        turn_command = Command(name='turn',
                               callback=self.turn_command,
                               description='Turn the robot',
                               arguments=[
                                   CommandArgument('speed',
                                                   short_name='s',
                                                   type=float),
                                   CommandArgument('direction',
                                                   short_name='d',
                                                   type=int)

                               ])

        speed_command = Command(name='speed',
                                callback=frodo.setSpeed,
                                arguments=[
                                    CommandArgument(name='speed_left',
                                                    short_name='l',
                                                    type=float, ),
                                    CommandArgument(name='speed_right',
                                                    short_name='r',
                                                    type=float, ),
                                ])

        read_data_command = Command(name='read',
                                    callback=self.read_data,
                                    arguments=[])
        
        add_nav_movement = Command(name='addMov',
                                callback=self.frodo.addMovement,
                                arguments=[
                                    CommandArgument(name='dphi',
                                                    short_name='p',
                                                    type=float),
                                    CommandArgument(name='radius',
                                                    short_name='r',
                                                    type=int),
                                    CommandArgument(name='time',
                                                    short_name='t',
                                                    type=float)
                                ])
        
        start_nav_movement = Command(name='startNavMov',
                                        callback=self.frodo.startNavigationMovement)
        
        stop_nav_movement = Command(name='stopNavMov',
                                        callback=self.frodo.stopNavigationMovement)
        
        pause_nav_movement = Command(name='pauseNavMov',
                                        callback=self.frodo.pauseNavigationMovement)

        continue_nav_movement = Command(name='continueNavMov',
                                        callback=self.frodo.continueNavigationMovement)

        clear_nav_movement_q = Command(name='clearMovQ',
                                        callback=self.frodo.clearNavigationMovementQueue)
        
        set_control_mode = Command(name='setMod',
                                callback=self.frodo.setControlMode,
                                arguments=[
                                    CommandArgument(name='mode',
                                                    short_name='m',
                                                    type=int)],
                                allow_positionals=True
                                )

        test_communication = Command(name='testComm',
                                     callback=self.test_communication,
                                     description='Tests the communication with the robot',
                                     arguments=[
                                         CommandArgument(name='iterations',
                                                         short_name='i',
                                                         type=int,
                                                         optional=True,
                                                         default=10,
                                                         description='Number of iterations to test')
                                     ])

        utils_command_set = CommandSet(name='utilities', commands=[test_communication])

        super(FRODO_CommandSet, self).__init__(name=frodo.id,
                                               child_sets=[utils_command_set],
                                               commands=[beep_command,
                                                         stop_command,
                                                         speed_command,
                                                         turn_command,
                                                         read_data_command,
                                                         add_nav_movement,
                                                         start_nav_movement,
                                                         pause_nav_movement,
                                                         continue_nav_movement,
                                                         stop_nav_movement,
                                                         clear_nav_movement_q,
                                                         set_control_mode])

    # ------------------------------------------------------------------------------------------------------------------
    def turn_command(self, speed: float, direction: int):
        if direction == 1:
            self.frodo.setSpeed(speed_left=-speed, speed_right=speed)
        elif direction == -1:
            self.frodo.setSpeed(speed_left=speed, speed_right=-speed)

    # ------------------------------------------------------------------------------------------------------------------
    def read_data(self):
        data = self.frodo.getData()
        if data is not None:
            return f"{data}"

    # ------------------------------------------------------------------------------------------------------------------
    def test_communication(self, iterations=10):
        test_response_time(self.frodo, iterations=iterations, print_response_time=True)
