from extensions.gui._archive.nodejs_gui.nodejs_gui import NodeJSGui
from extensions.joystick._archive.joystick_manager import Joystick


class IdeenExpoGUI(NodeJSGui):
    gui_joystick_array: []

    def __init__(self):
        super().__init__()

        self.gui_joystick_array = []

    def sendJoystickMessage(self):
        self.sendMessage(message_type='joysticksChanged', data={'joysticks': self.gui_joystick_array})

    def addJoystick(self, added_joystick: Joystick):
        for joystick in self.gui_joystick_array:
            if joystick.get('id') == added_joystick.id:
                return

        joystick = {"id": added_joystick.id, "name": added_joystick.name, "assignedBot": ""}
        self.gui_joystick_array.append(joystick)
        self.sendJoystickMessage()

    def removeJoystick(self, removed_joystick):
        for joystick in self.gui_joystick_array:
            if joystick.get('id') == removed_joystick.group_id:
                self.gui_joystick_array.remove(joystick)
                self.sendJoystickMessage()

    def addRobot(self, robot):
        self.sendEvent(event='robot_connected', device_id=robot.device.information.device_id)

    def removeRobot(self, robot):
        self.sendEvent(event='robot_disconnected', device_id=robot.device.information.device_id)

    @staticmethod
    def convertTwiprSample(sample):
        sample = sample.data
        new_sample = {
            'general': {
                'status': "normal",  # str
                'configuration': 'default',  # str
                'time': sample['general']['time'],  # int [ms]
                'tick': sample['general']['tick'],  # int [-]
                'sample_time': sample['general']['sample_time'],  # float [s]
                'name': sample['general']['id'],  # str
                'id': sample['general']['id']  # str
            },
            'control': {
                'status': sample['control']['status'],  # str
                'mode': sample['control']['mode'],  # str [X]
                'configuration': "default",  # str
                'u_ext': [0.0, 0.0],  # list[float] [-] [X]
                'u': {
                    'left': 0,  # float [Nm][0-1] [X]
                    'right': 0,  # float [Nm][0-1] [X]
                },
                'balancing_control': {
                    'status': 'error',  # str
                    'u_ext': [0.0, 0.0],  # list[float] [Nm][0-1]
                    'u': [0.0, 0.0]  # list[float] [Nm][0-1]
                },
                'speed_control': {
                    'status': 'off',  # str
                    'u_ext': {
                        'v': 0.0,  # float [m/s]
                        'psi_dot': 0.0,  # float [rad/s]
                    },
                    'u': [0.0, 0.0]  # list[float] [-]
                }
            },
            'estimation': {
                'status': 'normal',  # str
                'mode': 'kalman_ext',  # str
                'configuration': 'default',  # str
                'state': {
                    'x': sample['estimation']['state']['x'],  # float [m] [X]
                    'y': sample['estimation']['state']['y'],  # float [m] [X]
                    'v': sample['estimation']['state']['v'],  # float [m/s] [X]
                    'theta': sample['estimation']['state']['theta'],  # float [grad] [X]
                    'theta_dot': sample['estimation']['state']['theta_dot'],  # float [grad/s] [X]
                    'psi': sample['estimation']['state']['psi'],  # float [grad] [X]
                    'psi_dot': sample['estimation']['state']['psi_dot'],  # float [grad/s] [X]
                },
                'uncertainties':
                    {
                        'x': 0,  # float [-]
                        'y': 0,  # float [-]
                        'v': 0,  # float [-]
                        'theta': 0,  # float [-]
                        'theta_dot': 0,  # float [-]
                        'psi': 0,  # float [-]
                        'psi_dot': 0,  # float [-]
                    }
            },
            'sensors': {
                'imu': {
                    'status': 'normal',  # str
                    'gyr': [sample['sensors']['imu']['gyr']['x'], sample['sensors']['imu']['gyr']['y'],
                            sample['sensors']['imu']['gyr']['z']],  # list[float] [grad/s] [X]
                    'acc': [sample['sensors']['imu']['acc']['x'], sample['sensors']['imu']['acc']['y'],
                            sample['sensors']['imu']['acc']['z']],  # list[float] [m/s^2] [X]
                },
                'wheel': {
                    'status': 'normal',  # str
                    'speed': [sample['drive']['left'], sample['drive']['right']],  # list[float] [grad/s] [X]
                    'angle': [0, 0]  # list[float] [grad]
                },
                'distance': {
                    'front': 0,  # float [m] [X]
                    'back': 0,  # float [m] [X]
                },
                'temperature': {
                    'board': 0,  # float [C]
                    'bms': 0  # float [C]
                }
            },
            'board': {
                'status': 'normal',  # str
                'battery': sample['sensors']['power']['bat_voltage'],  # float [V] [X]
                'charging': False,  # bool
            },
            'drive': {
                'left': {
                    'status': 'normal',  # str
                    'u': 1,  # float [Nm] [X]
                    'torque': 1,  # float [Nm] [X]
                    'speed': 1,  # float [grad/s] [X]
                    'angle': 1  # float [grad]
                },
                'right': {
                    'status': 1,  # str
                    'u': 1,  # float [Nm] [X]
                    'torque': 1,  # float [Nm] [X]
                    'speed': 1,  # float [grad/s] [X]
                    'angle': 2  # float [grad]
                }
            },

        }

        if new_sample['estimation']['state']['x'] == 0:
            new_sample['estimation']['state']['x'] = 0.001
        if new_sample['estimation']['state']['y'] == 0:
            new_sample['estimation']['state']['y'] = 0.001
        return new_sample

    def _websocketClientConnected_callback(self, *args, **kwargs):
        super()._websocketClientConnected_callback(*args, **kwargs)
        self.sendJoystickMessage()
