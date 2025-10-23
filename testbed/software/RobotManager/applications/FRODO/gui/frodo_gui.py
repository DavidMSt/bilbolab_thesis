from __future__ import annotations

import dataclasses
import math

import numpy as np

from applications.FRODO.data_aggregator import FRODO_DataAggregator, TestbedObject, TestbedObject_FRODO, \
    TestbedObject_STATIC

from applications.FRODO.tracker.definitions import TrackedFRODO, TrackedStatic
from applications.FRODO.tracker.frodo_tracker import FRODO_Tracker
from core.utils.callbacks import Callback
from core.utils.colors import random_color_from_palette
from core.utils.logging_utils import Logger, addLogRedirection, LOGGING_COLORS
from core.utils.time import Timer
from extensions.cli.cli import CLI
from extensions.gui.src.app import App
from extensions.gui.src.gui import GUI, Page, Category
from extensions.gui.src.lib.map.map import MapWidget
from extensions.gui.src.lib.map.map_objects import Agent, CoordinateSystem, VisionAgent, MapObjectGroup, Point, \
    Line
from extensions.gui.src.lib.objects.objects import Widget_Group
from extensions.gui.src.lib.objects.python.indicators import BatteryIndicatorWidget, ConnectionIndicator, \
    InternetIndicator, JoystickIndicator
from extensions.gui.src.lib.objects.python.video import VideoWidget
from extensions.gui.src.lib.plot.realtime.rt_plot import ServerMode, UpdateMode
from robots.frodo.frodo import FRODO
from robots.frodo.frodo_definitions import FRODO_DEFINITIONS, STATIC_DEFINITIONS, FRODO_Sample, TESTBED_SIZE, \
    TESTBED_TILE_SIZE, FRODO_VIDEO_PORT
from robots.frodo.frodo_manager import FRODO_Manager
from core.utils.lipo import lipo_soc
from robots.frodo.frodo_utilities import vector2GlobalFrame

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from applications.FRODO.frodo_application import FRODO_Application


# === TRACKER PAGE =====================================================================================================
class FRODO_Tracker_Page:
    page: Page
    gui: GUI

    agents: dict
    statics: dict

    # === INIT =========================================================================================================
    def __init__(self, gui: GUI, tracker: FRODO_Tracker):
        self.gui = gui
        self.tracker = tracker
        self.logger = Logger('FRODO Tracker Page', 'DEBUG')
        self.page = Page(id='tracker_page', name='Tracker')

        # Build the map
        self.map_widget = MapWidget(widget_id='map_widget',
                                    limits={"x": [-2, 2], "y": [-2, 2]},
                                    initial_display_center=[0, 0],
                                    tiles=False,
                                    show_grid=True,
                                    major_grid_size=0.5,
                                    minor_grid_size=0.1,
                                    )

        self.page.addWidget(self.map_widget, width=18, height=18)
        self.map = self.map_widget.map
        self.agents = {}
        self.statics = {}

        self.tracker.events.description_received.on(self._onTrackerDescriptionReceived, once=True)

    # === METHODS ======================================================================================================

    # === PRIVATE METHODS ==============================================================================================
    def _onTrackerDescriptionReceived(self):
        for frodo_id, frodo_tracked_agent in self.tracker.robots.items():

            if not frodo_id in FRODO_DEFINITIONS:
                self.logger.warning(f"FRODO ID {frodo_id} not found in FRODO_DEFINITIONS")
                continue
            frodo_definition = FRODO_DEFINITIONS[frodo_id]

            # Add an agent to the map
            map_agent = Agent(id=frodo_id,
                              color=frodo_definition.color,
                              size=0.07,
                              arrow_length=0.25,
                              arrow_width=0.05,
                              x=0,
                              y=0,
                              )

            self.agents[frodo_id] = {
                'map_object': map_agent,
                'tracked_object': frodo_tracked_agent,
            }
            frodo_tracked_agent.events.update.on(
                callback=Callback(function=self._onTrackedAgentUpdate, inputs={'frodo_id': frodo_id},
                                  discard_inputs=True),
                max_rate=10)

            self.map.addObject(map_agent)

        for static_id, static_marker in self.tracker.statics.items():
            if not static_id in STATIC_DEFINITIONS:
                self.logger.warning(f"Static ID {static_id} not found in STATIC_DEFINITIONS")
                continue
            static_definition = STATIC_DEFINITIONS[static_id]

            map_static = CoordinateSystem(id=static_id,
                                          name=static_id,
                                          show_name=True
                                          )

            self.statics[static_id] = {
                'map_object': map_static,
                'tracked_object': static_marker,
            }
            static_marker.events.update.on(
                callback=Callback(function=self._onTrackedStaticUpdate, inputs={'static_id': static_id},
                                  discard_inputs=True),
                max_rate=10)
            self.map.addObject(map_static)

    # ------------------------------------------------------------------------------------------------------------------
    def _onTrackedAgentUpdate(self, frodo_id):
        tracked_object: TrackedFRODO = self.agents[frodo_id]['tracked_object']
        x = tracked_object.state.x
        y = tracked_object.state.y
        psi = tracked_object.state.psi

        map_object: Agent = self.agents[frodo_id]['map_object']
        map_object.update(x=x, y=y, psi=psi)

    # ------------------------------------------------------------------------------------------------------------------
    def _onTrackedStaticUpdate(self, static_id):
        tracked_object: TrackedStatic = self.statics[static_id]['tracked_object']
        x = tracked_object.state.x
        y = tracked_object.state.y
        psi = tracked_object.state.psi

        map_object: CoordinateSystem = self.statics[static_id]['map_object']
        map_object.update(x=x, y=y, psi=psi)


# === OVERVIEW PAGE ====================================================================================================

@dataclasses.dataclass
class RobotOverviewWidgets:
    battery_widget: BatteryIndicatorWidget
    connection_strength_widget: ConnectionIndicator
    internet_indicator_widget: InternetIndicator
    joystick_indicator_widget: JoystickIndicator


class FRODO_Robots_Page:
    page: Page
    gui: GUI
    manager: FRODO_Manager

    robots: dict
    _num_robots: int = 0

    _timer_overview_updates: Timer

    # === INIT =========================================================================================================
    def __init__(self, gui: GUI, manager: FRODO_Manager):
        self.gui = gui
        self.manager = manager
        self.page = Page(id='robots_page', name='Robots')

        self.manager.events.new_robot.on(self._buildPage)
        self.manager.events.robot_disconnected.on(self._buildPage)
        self.robots = {}

        self._timer_overview_updates = Timer()

    # === METHODS ======================================================================================================
    def init(self):
        ...

    # === PRIVATE METHODS ==============================================================================================
    def _buildPage(self, *args, **kwargs):
        self.page.clear()
        self._num_robots = 0
        self.robots = {}
        for robot in self.manager.robots.values():
            self._addRobot(robot)
            self._num_robots += 1

    # ------------------------------------------------------------------------------------------------------------------
    def _onRobotUpdate(self, robot_id):
        data: FRODO_Sample = self.robots[robot_id]['robot'].core.data

        if self._timer_overview_updates > 2:
            self._timer_overview_updates.reset()
            # Update the overview widgets
            overview_widgets: RobotOverviewWidgets = self.robots[robot_id]['overview_widgets']
            lipo_percentage = lipo_soc(data.general.battery, cells=2)
            overview_widgets.battery_widget.setValue(percentage=lipo_percentage, voltage=data.general.battery)

            overview_widgets.connection_strength_widget.setValue(
                self._classify_connection_strength(data.general.connection_strength))
            overview_widgets.internet_indicator_widget.setValue(data.general.internet_connection)

            overview_widgets.joystick_indicator_widget.setValue(False)

    # ------------------------------------------------------------------------------------------------------------------
    def _addRobot(self, robot: FRODO):
        column = int(36 / 4 * self._num_robots) + 1

        frodo_definition = FRODO_DEFINITIONS[robot.id]

        robot_group = Widget_Group(group_id=f'robot_{robot.id}',
                                   title=robot.id,
                                   show_title=True,
                                   rows=18,
                                   columns=9,
                                   border_color=frodo_definition.color,
                                   border_width=2, )
        self.page.addWidget(robot_group, column=column, row=1, width=int(36 / 4), height=18)

        overview_group = Widget_Group(group_id=f'overview_{robot.id}',
                                      rows=1,
                                      columns=4,
                                      )
        robot_group.addWidget(overview_group, row=1, column=1, width=9, height=2)

        battery_widget = BatteryIndicatorWidget(widget_id=f'battery_{robot.id}',
                                                label_position='center',
                                                show='voltage',
                                                )
        overview_group.addWidget(battery_widget, row=1, column=1, width=1, height=1)

        connection_strength_widget = ConnectionIndicator(widget_id=f'connection_{robot.id}')
        overview_group.addWidget(connection_strength_widget, row=1, column=2, width=1, height=1)

        internet_indicator_widget = InternetIndicator(widget_id=f'internet_{robot.id}')
        overview_group.addWidget(internet_indicator_widget, row=1, column=3, width=1, height=1)

        joystick_indicator_widget = JoystickIndicator(widget_id=f'joystick_{robot.id}')
        overview_group.addWidget(joystick_indicator_widget, row=1, column=4, width=1, height=1)

        self.robots[robot.id] = {}
        self.robots[robot.id]['robot'] = robot
        self.robots[robot.id]['overview_widgets'] = RobotOverviewWidgets(battery_widget,
                                                                         connection_strength_widget,
                                                                         internet_indicator_widget,
                                                                         joystick_indicator_widget)

        robot.core.events.stream.on(
            callback=Callback(function=self._onRobotUpdate, inputs={'robot_id': robot.id}, discard_inputs=True), )

        control_status_group = Widget_Group(group_id=f'control_status_{robot.id}',
                                            rows=1,
                                            columns=1,
                                            )
        robot_group.addWidget(control_status_group, column=1, width=9, height=3)

        robot_tiny_map = MapWidget(widget_id=f"{robot.id}_tiny_map",
                                   limits={"x": [0, TESTBED_SIZE[0]], "y": [0, TESTBED_SIZE[1]]},
                                   initial_display_center=[TESTBED_SIZE[0] / 2, TESTBED_SIZE[0] / 2],
                                   initial_display_zoom=0.6,
                                   tiles=True,
                                   tile_size=TESTBED_TILE_SIZE,
                                   show_grid=False,
                                   major_grid_size=0.5,
                                   minor_grid_size=0.1,
                                   server_port=8333 + self._num_robots,
                                   )
        robot_group.addWidget(robot_tiny_map, row=11, column=1, width=9, height=8)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _classify_connection_strength(signal_strength: float) -> str:
        """Map a numeric signal strength (0..100) to low/medium/high."""
        if signal_strength > 85:
            return 'high'
        if signal_strength > 30:
            return 'medium'
        return 'low'


# === VISION PAGE ======================================================================================================
class FRODO_Vision_Page:
    page: Page
    gui: GUI
    manager: FRODO_Manager

    robots: dict
    num_robots: int = 0

    # === INIT =========================================================================================================
    def __init__(self, gui: GUI, manager: FRODO_Manager):
        self.gui = gui
        self.manager = manager
        self.page = Page(id='vision_page', name='Vision')
        self._buildPage()

        self.manager.callbacks.new_robot.register(self._addRobot)
        self.manager.callbacks.robot_disconnected.register(self._removeRobot)

        self.robots = {}

    # === METHODS ======================================================================================================
    ...

    # === PRIVATE METHODS ==============================================================================================
    def _buildPage(self, *args, **kwargs):
        # Add a big map
        self.map_widget = MapWidget(widget_id='vision_map_widget',
                                    limits={"x": [0, TESTBED_SIZE[0]], "y": [0, TESTBED_SIZE[1]]},
                                    initial_display_center=[TESTBED_SIZE[0] / 2, TESTBED_SIZE[1] / 2],
                                    tiles=True,
                                    tile_size=TESTBED_TILE_SIZE,
                                    show_grid=False,
                                    server_port=8101,
                                    # major_grid_size=0.5,
                                    # minor_grid_size=0.1,
                                    )
        self.page.addWidget(self.map_widget, width=18, height=18)

        for robot in self.manager.robots.values():
            self._addRobot(robot)

    # ------------------------------------------------------------------------------------------------------------------
    def _addRobot(self, robot: FRODO):
        self.robots[robot.id] = {}
        self.robots[robot.id]['robot'] = robot

        robot_group = MapObjectGroup(id=f'robot_{robot.id}_vision', )
        vision_elements_group = MapObjectGroup(id=f'robot_{robot.id}_vision_elements', )
        self.robots[robot.id]['group'] = robot_group
        self.robots[robot.id]['vision_elements_group'] = vision_elements_group

        self.robots[robot.id]['vision_map_widget'] = VisionAgent(id=robot.id,
                                                                 color=FRODO_DEFINITIONS[robot.id].color,
                                                                 size=0.07,
                                                                 arrow_length=0.3,
                                                                 arrow_width=0.05,
                                                                 vision_radius=1.5,
                                                                 vision_fov=math.radians(120),
                                                                 x=0,
                                                                 y=0,
                                                                 )

        robot_group.addObject(self.robots[robot.id]['vision_map_widget'])
        robot_group.addGroup(vision_elements_group)
        self.map_widget.map.addGroup(robot_group)

        # Add the video output
        robot_video_widget = VideoWidget(widget_id=f'{robot.id}_video_widget',
                                         path=f"http://{robot.id}.local:{FRODO_VIDEO_PORT}/video",
                                         title=f"{robot.id}",
                                         title_color=FRODO_DEFINITIONS[robot.id].color, )

        row, column = self._get_robot_video_spot(robot.id)
        self.page.addWidget(robot_video_widget, column=column, row=row, width=12, height=9)

        self.robots[robot.id]['video_widget'] = robot_video_widget

        robot.core.events.stream.on(
            callback=Callback(function=self._onRobotUpdate, inputs={'robot_id': robot.id}, discard_inputs=True), )

        self.num_robots += 1

    # ------------------------------------------------------------------------------------------------------------------
    def _removeRobot(self, robot: FRODO):

        robot_id = robot.id

        # Remove the video widget
        self.page.removeWidget(self.robots[robot_id]['video_widget'])

        # Remove the robot from the map
        robot_group: MapObjectGroup = self.robots[robot_id]['group']
        self.map_widget.map.removeGroup(robot_group)

    # ------------------------------------------------------------------------------------------------------------------
    def _onRobotUpdate(self, robot_id):
        robot: FRODO = self.robots[robot_id]['robot']
        data: FRODO_Sample = self.robots[robot_id]['robot'].core.data

        # Update the vision map

        # 1. Update the robot
        vision_map_widget: VisionAgent = self.robots[robot_id]['vision_map_widget']
        vision_map_widget.update(x=data.estimation.state.x, y=data.estimation.state.y, psi=data.estimation.state.psi)

        # 2. Update the vision elements

        vision_elements_group: MapObjectGroup = self.robots[robot_id]['vision_elements_group']

        # Make all measurements invisible
        for element in vision_elements_group.objects.values():
            element.visible(False)

        # Update the measurements that are visible
        for measurement in data.measurements.aruco_measurements:
            object_id = str(measurement.measured_aruco_id)
            position = measurement.position
            psi = measurement.psi

            position_global = vector2GlobalFrame(position, robot.core.data.estimation.state.psi)
            position_global = [position_global[0] + robot.core.data.estimation.state.x,
                               position_global[1] + robot.core.data.estimation.state.y, ]

            if vision_elements_group.objectInGroup(object_id):
                element = vision_elements_group.getObjectByPath(object_id)
                element.visible(True)
                element.update(x=position_global[0], y=position_global[1])

            else:
                vision_element = Point(id=object_id,
                                       color=[0.8, 0.8, 0.8],
                                       size=0.05,
                                       x=position_global[0],
                                       y=position_global[1],
                                       )
                vision_elements_group.addObject(vision_element)

    # ------------------------------------------------------------------------------------------------------------------
    def _get_robot_video_spot(self, robot_id):

        match robot_id:
            case 'frodo1':
                return (1, 24)
            case 'frodo2':
                return (1, 36)
            case 'frodo3':
                return (10, 24)
            case 'frodo4':
                return (10, 36)
            case _:
                raise ValueError(f"Unknown robot ID {robot_id}")


# === DATA PAGE ========================================================================================================
class FRODO_Data_Page:
    page: Page
    gui: GUI
    manager: FRODO_Manager
    aggregator: FRODO_DataAggregator

    agents: dict[str, AgentContainer]
    statics: dict[str, StaticContainer]

    # === INIT =========================================================================================================
    def __init__(self, gui: GUI, manager: FRODO_Manager, data_aggregator: FRODO_DataAggregator):
        self.gui = gui
        self.manager = manager
        self.aggregator = data_aggregator
        self.page = Page(id='data_page', name='Data')
        self._buildPage()

        self.agents = {}
        self.statics = {}
        self._add_listeners()

    # === CLASSES ======================================================================================================
    @dataclasses.dataclass
    class AgentContainer:
        object: TestbedObject_FRODO
        group: MapObjectGroup
        map_agent: Agent
        measurements: MapObjectGroup
        lines: MapObjectGroup

    @dataclasses.dataclass
    class StaticContainer:
        object: TestbedObject_STATIC
        map: CoordinateSystem

    # === PROPERTIES ===================================================================================================

    # === METHODS ======================================================================================================

    # === PRIVATE METHODS ==============================================================================================
    def _buildPage(self):

        self.map_widget = MapWidget(widget_id='data_map_widget',
                                    limits={"x": [0, TESTBED_SIZE[0]], "y": [0, TESTBED_SIZE[1]]},
                                    initial_display_center=[TESTBED_SIZE[0] / 2, TESTBED_SIZE[1] / 2],
                                    tiles=True,
                                    tile_size=TESTBED_TILE_SIZE,
                                    show_grid=False,
                                    server_port=8102,
                                    )
        self.page.addWidget(self.map_widget, width=18, height=18)

    # ------------------------------------------------------------------------------------------------------------------
    def _add_listeners(self):
        self.aggregator.events.initialized.on(self._on_aggregator_initialized)
        self.aggregator.events.update.on(self._on_aggregator_update)

    # ------------------------------------------------------------------------------------------------------------------
    def _on_aggregator_initialized(self):
        # Loop through the aggregators agents and statics and add them to the map
        for agent in self.aggregator.robots.values():
            self.addTestbedObject(agent)

    # ------------------------------------------------------------------------------------------------------------------
    def _on_aggregator_update(self):
        for agent in list(self.agents.values()):
            agent.map_agent.update(x=agent.object.state.x, y=agent.object.state.y, psi=agent.object.state.psi)

            current_measurements = []
            for measurement in agent.object.measurements:
                measured_object_id = measurement.object_to.id
                measurement_id = f"{agent.object.id} -> {measured_object_id}"
                measurement_line_id = f"{agent.object.id}_to_{measured_object_id}_line"

                if measurement_id in agent.measurements.objects:
                    agent.measurements.objects[measurement_id].visible(True)
                    agent.lines.objects[measurement_line_id].visible(True)
                else:
                    measurement_object = Agent(id=measurement_id,
                                               color=[0.8, 0.8, 0.8],
                                               size=0.05,
                                               opacity=0.5,
                                               )
                    agent.measurements.addObject(measurement_object)

                    measurement_line = Line(id=measurement_line_id,
                                            name=measurement_id,
                                            color=[0.8, 0.8, 0.8],
                                            start=agent.map_agent,
                                            end=measurement_object,
                                            )
                    agent.lines.addObject(measurement_line)

                # Set the measurement position
                measurement_position_global = vector2GlobalFrame(np.asarray([measurement.relative.x,
                                                                             measurement.relative.y]),
                                                                 agent.object.state.psi)
                measurement_position_global = [measurement_position_global[0] + agent.object.state.x,
                                               measurement_position_global[1] + agent.object.state.y, ]
                measurement_psi_global = measurement.relative.psi + agent.object.state.psi
                agent.measurements.objects[measurement_id].update(x=measurement_position_global[0],
                                                                  y=measurement_position_global[1],
                                                                  psi=measurement_psi_global, )

    # ------------------------------------------------------------------------------------------------------------------
    def addTestbedObject(self, object: TestbedObject):

        if isinstance(object, TestbedObject_FRODO):

            group = MapObjectGroup(id=f'object_{object.id}_group')
            agent = Agent(id=object.id,
                          color=FRODO_DEFINITIONS[object.id].color,
                          size=0.07,
                          arrow_length=0.3,
                          arrow_width=0.05,
                          opacity=0.5,
                          )

            group.addObject(agent)
            self.map_widget.map.addGroup(group)

            agent_container = self.AgentContainer(object=object,
                                                  group=group,
                                                  map_agent=agent,
                                                  measurements=MapObjectGroup(id=f'object_{object.id}_measurements'),
                                                  lines=MapObjectGroup(id=f'object_{object.id}_lines'),
                                                  )

            agent_container.group.addGroup(agent_container.measurements)
            agent_container.group.addGroup(agent_container.lines)

            self.agents[object.id] = agent_container

        elif isinstance(object, TestbedObject_STATIC):
            ...
        else:
            raise ValueError(f"Unknown object type {type(object)}")

    # ------------------------------------------------------------------------------------------------------------------
    def removeTestbedObject(self, robot_id):
        ...
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------


# === FRODO ALGORITHM PAGE =============================================================================================
class FRODO_Algorithm_Page:
    page: Page

    # === INIT =========================================================================================================
    def __init__(self, application: FRODO_Application):
        self.application = application
        self._build_page()
        self.add_listeners()

    # === METHODS ======================================================================================================
    def _build_page(self):
        self.page = Page(id='algorithm_page', name='Algorithm')

        self.map_widget = MapWidget(widget_id='algorithm_map_widget',
                                    limits={"x": [0, TESTBED_SIZE[0]], "y": [0, TESTBED_SIZE[1]]},
                                    initial_display_center=[TESTBED_SIZE[0] / 2, TESTBED_SIZE[1] / 2],
                                    tiles=True,
                                    tile_size=TESTBED_TILE_SIZE,
                                    show_grid=False,
                                    server_port=8103,
                                    )
        self.page.addWidget(self.map_widget, width=18, height=18)

        # Overview Group
        overview_group = Widget_Group(group_id='overview', rows=1, columns=4, show_title=True,
                                      title='Algorithm Overview')

        self.page.addWidget(overview_group, column=19, row=1, width=9, height=3)

        controls_group = Widget_Group(group_id='controls', rows=1, columns=1, show_title=True,
                                      title='Algorithm Control')
        self.page.addWidget(controls_group, column=19, row=4, width=9, height=15)

        # self.error_plot_widget = PlotWidget(widget_id='plot_widget_1', title='Plot 1',
        #                                     server_mode=ServerMode.EXTERNAL,
        #                                     update_mode=UpdateMode.CONTINUOUS)

        # dataseries_1 = JSPlotTimeSeries(timeseries_id='ds1',
        #                                 name='Data 1',
        #                                 unit='V',
        #                                 min=-10,
        #                                 max=10,
        #                                 color=random_color_from_palette('pastel'), )
        # dataseries_2 = JSPlotTimeSeries(timeseries_id='ds2',
        #                                 name='Data 2',
        #                                 unit='V',
        #                                 min=-10,
        #                                 max=10,
        #                                 color=random_color_from_palette('pastel'), )
        # dataseries_3 = JSPlotTimeSeries(timeseries_id='ds3',
        #                                 name='Data 3',
        #                                 unit='V',
        #                                 min=-10,
        #                                 max=10,
        #                                 color=random_color_from_palette('pastel'), )
        # dataseries_4 = JSPlotTimeSeries(timeseries_id='ds4',
        #                                 name='Data 4',
        #                                 unit='V',
        #                                 min=-10,
        #                                 max=10,
        #                                 color=random_color_from_palette('pastel'), )
        # self.error_plot_widget.plot.addTimeseries(dataseries_1)
        # self.error_plot_widget.plot.addTimeseries(dataseries_2)
        # self.error_plot_widget.plot.addTimeseries(dataseries_3)
        # self.error_plot_widget.plot.addTimeseries(dataseries_4)

        # self.page.addWidget(self.error_plot_widget, column=28, row=1, width=9, height=9)

    # ------------------------------------------------------------------------------------------------------------------
    def add_listeners(self):
        ...


# === GUI ==============================================================================================================
class FRODO_GUI:
    gui: GUI
    app: App

    categories: dict

    tracker: FRODO_Tracker

    # === INIT =========================================================================================================
    def __init__(self, host, application, tracker: FRODO_Tracker, manager: FRODO_Manager,
                 aggregator: FRODO_DataAggregator,
                 cli: CLI = None):
        self.logger = Logger('FRODO GUI', 'DEBUG')
        self.gui = GUI(
            id='frodo_gui',
            host=host,
            run_js=True
        )

        if cli is not None:
            self.gui.cli_terminal.setCLI(cli)
        self.categories = {}

        self.application = application
        self.tracker = tracker
        self.manager = manager
        self.aggregator = aggregator

        self._buildOverviewCategory()
        addLogRedirection(self._logRedirection, minimum_level='DEBUG')

    # === METHODS ======================================================================================================
    def init(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.gui.start()

    # === PRIVATE METHODS ==============================================================================================
    def _buildOverviewCategory(self):
        self.overview_category = Category(id='overview', name='FRODO App')
        self.gui.addCategory(self.overview_category)

        self.robots_page = FRODO_Robots_Page(self.gui, self.manager)
        self.overview_category.addPage(self.robots_page.page)

        self.tracker_page = FRODO_Tracker_Page(self.gui,
                                               tracker=self.tracker)

        self.overview_category.addPage(self.tracker_page.page)

        self.vision_page = FRODO_Vision_Page(self.gui, manager=self.manager)
        self.overview_category.addPage(self.vision_page.page)

        self.data_page = FRODO_Data_Page(self.gui, manager=self.manager, data_aggregator=self.aggregator)
        self.overview_category.addPage(self.data_page.page)

        self.algorithm_page = FRODO_Algorithm_Page(self.application)
        self.overview_category.addPage(self.algorithm_page.page)

    # ------------------------------------------------------------------------------------------------------------------
    def _logRedirection(self, log_entry, log, logger, level):
        print_text = f"[{logger.name}] {log}"
        color = LOGGING_COLORS[level]
        color = [c / 255 for c in color]
        self.gui.print(print_text, color=color)
    # ------------------------------------------------------------------------------------------------------------------
