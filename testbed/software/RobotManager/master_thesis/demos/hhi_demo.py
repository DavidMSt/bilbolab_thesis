from __future__ import annotations

import dataclasses
import random
import time

import numpy as np

from core.utils.callbacks import Callback
from core.utils.exit import register_exit_callback
from core.utils.logging_utils import Logger, addLogRedirection, LOGGING_COLORS
from core.utils.sound.sound import SoundSystem
from extensions.babylon.src.babylon import BabylonVisualization
from extensions.babylon.src.lib.objects.bilbo.bilbo import BabylonBilbo
from extensions.babylon.src.lib.objects.box.box import WallFancy, Wall
from extensions.babylon.src.lib.objects.floor.floor import SimpleFloor
from extensions.babylon.src.lib.objects.frodo.frodo import BabylonFrodo
from extensions.cli.cli import CommandSet, CLI, Command, CommandArgument
from extensions.gui.src.gui import GUI, Category, Page
from extensions.gui.src.lib.objects.python.babylon_widget import BabylonWidget
from extensions.gui.src.lib.objects.python.buttons import Button
from extensions.simulation.src.core.environment import BASE_ENVIRONMENT_ACTIONS
from extensions.simulation.src.objects.base_environment import BaseEnvironment
from extensions.simulation.src.objects.bilbo import BILBO_DynamicAgent, BILBO_Control_Mode, DEFAULT_BILBO_MODEL, \
    BILBO_EIGENSTRUCTURE_ASSIGNMENT_DEFAULT_POLES, BILBO_EIGENSTRUCTURE_ASSIGNMENT_EIGEN_VECTORS
from extensions.joystick.joystick_manager import JoystickManager, Joystick


@dataclasses.dataclass
class RobotContainer:
    babylon: BabylonFrodo


# === BILBO INTERACTIVE EXAMPLE ========================================================================================
class HHI_demo:
    joystick_manager: JoystickManager
    babylon_visualization: BabylonVisualization
    robots: dict[str, RobotContainer]

    cli: CLI
    gui: GUI
    command_set: BILBO_Interactive_CommandSet
    soundsystem: SoundSystem

    # === INIT =========================================================================================================
    def __init__(self):
        self.logger = Logger('BILBO_InteractiveExample', 'DEBUG')

        self.robots = {}

        self.command_set = BILBO_Interactive_CommandSet(self)

        self.cli = CLI(id='example_david', root=self.command_set)

        self.gui = GUI(id='bilbo_interactive', host='localhost', run_js=True)
        self.gui.cli_terminal.setCLI(self.cli)

        self.babylon_visualization = BabylonVisualization(id='babylon', babylon_config={
            'title': 'Example David'})

        # Sound System for speaking and sounds
        self.soundsystem = SoundSystem(primary_engine='etts', volume=1)
        self.soundsystem.start()

        # Simulation Environment
        self.env = BaseEnvironment(Ts=0.01, run_mode='rt')

        self.env.scheduling.actions[BASE_ENVIRONMENT_ACTIONS.OUTPUT].addAction(self._simulationOutputStep)

        # Make a logging redirection
        addLogRedirection(self._logRedirection, minimum_level='DEBUG')

        register_exit_callback(self.close)

    # === METHODS ======================================================================================================
    def init(self):
        self._buildGUI()
        self._buildBabylon()
        self.babylon_visualization.init()
        self.env.init()
        self.env.initialize()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.gui.start()
        self.babylon_visualization.start()
        self.env.start()
        self.logger.info("Example David started")
        self.soundsystem.speak('Start Example David')

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self.soundsystem.speak('Example David stopped')
        self.joystick_manager.exit()
        self.logger.info("Example David stopped")
        time.sleep(2)

    # ------------------------------------------------------------------------------------------------------------------
    def addRobot(self, robot_id: str) -> RobotContainer | None:

        # Check if the robot already exists
        if robot_id in self.robots:
            self.logger.warning(f'Robot with ID {robot_id} already exists')
            return None

        robot_babylon = BabylonFrodo(object_id=robot_id, color=[1, 0, 0], fov=0, text='1')
        self.babylon_visualization.addObject(robot_babylon)

        self.robots[robot_id] = RobotContainer(babylon=robot_babylon)
        self.logger.info(f'Robot with ID {robot_id} added')

        return self.robots[robot_id]

    # ------------------------------------------------------------------------------------------------------------------
    def removeRobot(self, robot: str | RobotContainer):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def assignJoystick(self, joystick: int, robot: str | RobotContainer):

        ...

    # ------------------------------------------------------------------------------------------------------------------
    def removeJoystick(self, robot: str | RobotContainer):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def getRobotByID(self, robot_id: str) -> RobotContainer | None:
        if robot_id in self.robots:
            return self.robots[robot_id]
        else:
            return None

    # === PRIVATE METHODS ==============================================================================================
    def _newJoystick_callback(self, joystick: Joystick):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def _joystickDisconnected_callback(self, joystick: Joystick):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def reset(self):

        ...
        #     robot['robot'].removeJoystick()
        #     self.env.removeObject(robot['robot'])
        #     self.babylon_visualization.removeObject(robot['babylon'])
        #     self.logger.info(f'Robot {robot["robot"].agent_id} removed')
        #     self.robots.pop(robot['robot'].agent_id)

    # ------------------------------------------------------------------------------------------------------------------
    def _buildGUI(self):

        # Add a simple category
        cat1 = Category('cat1', max_pages=10)

        # Add a page
        page1 = Page('page1')
        cat1.addPage(page1)

        # Add it to the GUI
        self.gui.addCategory(cat1)

        # Add the Babylon Widget
        self.babylon_widget = BabylonWidget(widget_id='babylon_widget')
        page1.addWidget(self.babylon_widget, row=1, column=1, height=18, width=36)

        # Reset Button
        reset_button = Button(text="Reset", callback=self.reset)
        page1.addWidget(reset_button, height=2, width=4)

        bilbo1_button = Button(text="Add BILBO 1", callback=Callback(
            function=self.addRobot,
            inputs={
                'robot_id': 'bilbo1'
            },
            discard_inputs=True,
        ))
        page1.addWidget(bilbo1_button, height=2, width=4)

    # ------------------------------------------------------------------------------------------------------------------
    def _buildBabylon(self):

        floor = SimpleFloor('floor', size_y=50, size_x=50, texture='floor_bright.png')
        self.babylon_visualization.addObject(floor)

        wall1 = WallFancy('wall1', length=5, texture='wood4.png', include_end_caps=True)
        wall1.setPosition(y=2.5)
        self.babylon_visualization.addObject(wall1)

        wall2 = WallFancy('wall2', length=5, texture='wood4.png', include_end_caps=True)
        self.babylon_visualization.addObject(wall2)
        wall2.setPosition(y=-2.5)

        wall3 = WallFancy('wall3', length=5, texture='wood4.png')
        wall3.setPosition(x=2.5)
        wall3.setAngle(np.pi / 2)
        self.babylon_visualization.addObject(wall3)

        wall4 = WallFancy('wall4', length=5, texture='wood4.png')
        wall4.setPosition(x=-2.5)
        wall4.setAngle(np.pi / 2)
        self.babylon_visualization.addObject(wall4)

    # ------------------------------------------------------------------------------------------------------------------
    def _logRedirection(self, log_entry, log, logger, level):
        print_text = f"[{logger.name}] {log}"
        color = LOGGING_COLORS[level]
        color = [c / 255 for c in color]
        self.gui.print(print_text, color=color)

    # ------------------------------------------------------------------------------------------------------------------
    def _simulationOutputStep(self):
        ...
        for robot in list(self.robots.values()):
            robot.babylon.setState(
                x=random.uniform(-2, 2),
                y=random.uniform(-2, 2),
                psi=random.uniform(-np.pi, np.pi),
            )
        # # Update all BILBOs
        # for robot in self.robots.values():
        #     try:
        #         state = robot['robot'].state
        #         robot['babylon'].set_state(x=state.x,
        #                                    y=state.y,
        #                                    theta=state.theta,
        #                                    psi=state.psi)
        #     except Exception as e:
        #         self.logger.error(f'Error updating robot {robot["robot"].agent_id}: {e}')


# === BILBO INTERACTIVE CLI ============================================================================================
class BILBO_Interactive_CommandSet(CommandSet):

    def __init__(self, example: HHI_demo):
        super().__init__('example_david')
        self.example = example

        add_robot_command = Command(
            function=self.example.addRobot,
            name='add_robot',
            description='Add a new robot to the simulation',
            allow_positionals=True,
            arguments=[
                CommandArgument(name='robot_id', type=str, description='ID of the robot to add')
            ]
        )
        self.addCommand(add_robot_command)


def main():
    example = HHI_demo()

    example.init()
    example.start()

    while True:
        time.sleep(10)


if __name__ == '__main__':
    main()
