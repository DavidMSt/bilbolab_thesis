import time
import core.hardware.eeprom as eeprom
from control_board.board_config import getBoardConfig
from control_board.hardware.hardware import GPIO_Input, GPIO_Output, PullupPulldown
from utils.files import fileExists, deleteFile, relativeToFullPath
from paths import config_path
from control_board.shields.definitions import BILBO_SHIELD_REV_2_ID, SHIELD_ID_ADDRESS, BILBO_SHIELD_REV2_CONFIG_FILE
from utils.json_utils import writeJSON
from utils.exit import ExitHandler
from utils.button import Button

bilbo_shield_rev_2_config = {
    'id': BILBO_SHIELD_REV_2_ID,
    'pins': {
        'button1': {
            'type': 'internal',
            'pin': 4
        },
        'button2': {
            'type': 'internal',
            'pin': 5
        },
        'button1_led': {
            'type': 'sx1508',
            'pin': 1
        },
        'button2_led': {
            'type': 'sx1508',
            'pin': 7
        }

    }
}


class BILBO_Shield_Rev2:
    button1: Button
    button2: Button
    button1_led: GPIO_Output
    button2_led: GPIO_Output

    def __init__(self):

        self.button1 = Button(pin=bilbo_shield_rev_2_config['pins']['button1']['pin'])
        self.button2 = Button(pin=bilbo_shield_rev_2_config['pins']['button2']['pin'])

        self.button1_led = GPIO_Output(pin=bilbo_shield_rev_2_config['pins']['button1_led']['pin'],
                                       pin_type=bilbo_shield_rev_2_config['pins']['button1_led']['type'],
                                       value=0)

        self.button2_led = GPIO_Output(pin=bilbo_shield_rev_2_config['pins']['button2_led']['pin'],
                                       pin_type=bilbo_shield_rev_2_config['pins']['button2_led']['type'],
                                       value=0)

        self.exit = ExitHandler()
        self.exit.register(self.close)

    def close(self, *args, **kwargs):
        self.button1_led.write(0)
        self.button2_led.write(0)


# ======================================================================================================================
def write_shield_address():
    config = getBoardConfig()

    eeprom.write_bytes(eeprom_address=config['devices']['SHIELD_EEPROM_I2C_ADDRESS'],
                       byte_address=SHIELD_ID_ADDRESS,
                       data=bilbo_shield_rev_2_config['id'])
    time.sleep(0.01)
    print(eeprom.read_bytes(
        eeprom_address=config['devices']['SHIELD_EEPROM_I2C_ADDRESS'],
        byte_address=SHIELD_ID_ADDRESS,
        num_bytes=1
    ))


# ======================================================================================================================
def generate_shield_config():
    config_file = relativeToFullPath(f"{config_path}{BILBO_SHIELD_REV2_CONFIG_FILE}")
    if fileExists(config_file):
        deleteFile(config_file)

    writeJSON(config_file, bilbo_shield_rev_2_config)
    print("Generated shield config for BILBO Shield Rev 2.")


# ======================================================================================================================
if __name__ == '__main__':
    # write_shield_address()
    generate_shield_config()
