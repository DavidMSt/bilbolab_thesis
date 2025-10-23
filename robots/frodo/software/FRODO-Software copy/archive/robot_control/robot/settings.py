import os

from utils.json_utils import writeJSON, readJSON

settings_file_path = os.path.expanduser('~/robot_settings.json')

# Samples LL
SAMPLE_BUFFER_SIZE = 10

# Register Tables
REGISTER_TABLE_GENERAL = 0x01
REGISTER_TABLE_CONTROL = 0x02

TWIPR_GPIO_INTERRUPT_NEW_SAMPLES = 6  # 6
TWIPR_GPIO_NEW_TRAJECTORY = 5


def generate_settings_file():
    settings = {
        'id': 'twipr1',
        'name': 'Vision Robot 1',
        'balancing_gain': [0.035, 0.06, 0.01, 0.009,
                           0.035, 0.06, 0.01, -0.009],
    }

    writeJSON(settings_file_path, settings)


def readSettings():
    settings = readJSON(settings_file_path)
    return settings


if __name__ == '__main__':
    generate_settings_file()
