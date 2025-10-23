import utils.json_utils as json_utils
import json
import uuid
import os

uuid_file_path = os.path.expanduser('~/board_uuid.json')
config_file_path = os.path.expanduser('~/board_config.json')


# ======================================================================================================================
def getBoardConfig():
    # First check if the files exist

    config = {}
    with open(config_file_path) as config_file:
        config = {**config, **json.load(config_file)}
    with open(uuid_file_path) as uuid_file:
        config = {**config, **json.load(uuid_file)}

    return config


# ======================================================================================================================
def generate_uuid_file():
    uid = str(uuid.uuid4())
    data = {
        'uid': uid
    }
    json_utils.writeJSON(uuid_file_path, data)


#
#
# ======================================================================================================================
def generate_board_config():
    board_config = {
        'name': 'c4_02',
        'address': list(b'\x01\x01'),
    }
    json_utils.writeJSON(config_file_path, board_config)


if __name__ == '__main__':
    generate_uuid_file()
    generate_board_config()
