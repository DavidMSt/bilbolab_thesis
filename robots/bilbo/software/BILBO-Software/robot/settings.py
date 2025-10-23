import os

# === PATHS ============================================================================================================
robot_path = os.path.expanduser('~/robot/')
settings_file_path = os.path.join(robot_path, 'settings.json')
config_path = os.path.expanduser('~/robot/config/')
experiments_path = os.path.expanduser('~/robot/experiments/')
logs_path = os.path.expanduser('~/robot/logs/')
control_config_path = os.path.expanduser('~/robot/control/')
calibrations_path = os.path.expanduser('~/robot/calibration/')

# === PORTS ============================================================================================================
STREAM_UDP_PORT = 5555
GUI_PORT = 5556


# === DEVICE INFORMATION ===============================================================================================
DEVICE_CLASS = 'BILBO'
DEVICE_TYPE = 'BILBO'


# === LOGIN INFORMATION ================================================================================================
USERNAME = 'admin'
PASSWORD = 'beutlin'