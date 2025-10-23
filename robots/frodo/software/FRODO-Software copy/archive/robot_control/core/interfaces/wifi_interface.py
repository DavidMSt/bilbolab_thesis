import logging

from core.communication.protocol import Protocol, Message
from core.communication.wifi.protocols.tcp_json_protocol import TCP_JSON_Protocol, TCP_JSON_Message
from core.communication.wifi.wifi_connection import WIFI_Connection
import utils as utils
from core.interfaces.data_link import DataLink, Command, generateDataDict, generateCommandDict
from utils.callbacks import Callback
from utils.board_config import *

class Stream:
    parameters: list
    connection: object
    interval: float


class WIFI_Interface:
    """
    This class manages the communication to external partners over WI-FI, USB or Serial
    """
    name: str
    id: str
    device_class: str
    device_type: str
    device_revision: str
    # parent: 'WIFI_Interface'
    # children: list['WIFI_Interface']

    uid: bytes

    data: dict[str, (dict, DataLink)]
    commands: dict[str, (dict, Command)]

    connection: WIFI_Connection

    streams: list[Stream]  # TODO: Could be removed. Not used

    connected: bool

    protocol: Protocol = TCP_JSON_Protocol

    # === INIT =========================================================================================================
    def __init__(self, interface_type: str = 'wifi', device_class: str = None, device_type: str = None, device_revision: str = None,
                 device_name: str = None, device_id: str = None):
        # Read the device config file
        board_config = getBoardConfig()

        self.device_class = device_class
        self.device_type = device_type
        self.device_revision = device_revision
        self.name = device_name
        self.id = device_id
        # self.parent = None
        # self.children = []
        self.connected = False

        self.callbacks = {
            'connected': [],
            'disconnected': [],
            'sync': [],
        }

        # --- WIFI ---
        # Add the WI-FI connection with the pre-configured address and name
        if interface_type == 'wifi':
            self.connection = WIFI_Connection(name=board_config['name'], address=board_config['address'])
        else:
            raise Exception("Not implemented yet")

        # Configure the WI-FI connection
        self.connection.registerCallback('connected', self._connected_callback)
        self.connection.registerCallback('disconnected', self._disconnected_callback)
        self.connection.registerCallback('rx', self._rx_callback)

        # --- PARAMETERS ---
        # Set up the parameters
        self.data = {}  # TODO: Where to get the parameters from? Import them from a certain python file for this board?

        # --- COMMANDS ---
        self.commands = {}

    # === METHODS ======================================================================================================
    def start(self):
        self.connection.start()

    # ------------------------------------------------------------------------------------------------------------------
    def sendEventMessage(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------

    def sendStreamMessage(self, data):
        msg = TCP_JSON_Message()
        msg.source = self.id
        msg.address = 0
        msg.type = 'stream'
        msg.data = data

        self._wifi_send(msg)

    # ------------------------------------------------------------------------------------------------------------------
    def addCommands(self, commands):
        if isinstance(commands, Command):
            self.commands[commands.identifier] = commands
        elif isinstance(commands, list):
            assert all(isinstance(command, Command) for command in commands)
            for command in commands:
                self.commands[command.identifier] = command
        elif isinstance(commands, dict):
            for command_identifier,command in commands.items():
                assert(isinstance(command, Command))
                self.commands[command_identifier] = command

    # ------------------------------------------------------------------------------------------------------------------
    def addCommand(self, identifier: str, callback: (callable, Callback), arguments: list[str], description: str):
        self.commands[identifier] = Command(identifier=identifier, callback=callback, arguments=arguments, description=description)

    # ------------------------------------------------------------------------------------------------------------------
    def registerCallback(self, callback_id, function: callable, parameters: dict = None, lambdas: dict = None,
                         **kwargs):
        callback = Callback(function, parameters, lambdas, **kwargs)

        if callback_id in self.callbacks:
            self.callbacks[callback_id].append(callback)
        else:
            raise Exception("Invalid Callback type")

    # === PRIVATE METHODS ==============================================================================================
    def _wifi_send(self, message):
        self.connection.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _connected_callback(self):
        self._sendDeviceIdentification()
        self.connected = True
        for callback in self.callbacks['connected']:
            callback(self)

    # ------------------------------------------------------------------------------------------------------------------
    def _disconnected_callback(self):
        self.connected = False
        for callback in self.callbacks['disconnected']:
            callback(self)

    # ------------------------------------------------------------------------------------------------------------------
    def _rx_callback(self, message, *args, **kwargs):

        # Handle the message
        self._handleRxMessage(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _handleRxMessage(self, message: Message):
        # Make sure the message is of the correct type. If this is not the case, the communication channels are not
        # configured correctly
        assert (isinstance(message, self.protocol.Message))
        # Check if the message has the correct command
        assert (message.type in self.protocol.allowed_types)
        # Handle the message based on the issued command
        if message.type == 'write':
            self._handler_writeMessage(message.data)
        elif message.type == 'read':
            self._handler_readMessage(message.data)
        elif message.type == 'command':
            self._handler_commandMessage(message.data)
        elif message.type == 'event':
            self._handlerEventMessage(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _handler_writeMessage(self, data: dict):
        # Go over the entries in the data dictionary
        for entry, value in data.items():

            # Check if the entry is also in the parameters dict
            if entry not in self.data:
                logging.warning(f"Received parameter {entry}, which is not a valid entry.")
                continue

            # Check if the entry is a parameter or a dict (group of parameters).
            # TODO: We only allow one level of grouping for now

            # The entry is a parameter
            if isinstance(self.data[entry], DataLink):
                self.data[entry].set(value)
                logging.debug(f"Set parameter {entry} to value {value}.")

            # The entry is a group of parameters
            elif isinstance(self.data[entry], dict) and isinstance(value, dict):
                for sub_entry, sub_value in value.items():
                    if not sub_entry in self.data[entry]:
                        logging.warning(f"Received parameter {entry}:{sub_entry}, which is not a valid entry.")
                        continue
                    if not (isinstance(self.data[entry][sub_entry], DataLink)):
                        logging.warning(f"Cannot set parameter {sub_entry}: Grouping level exceeds allowed level of: 1")
                    self.data[entry][sub_entry].set(sub_value)
                    logging.debug(f"Set parameter {entry}:{sub_entry} to value {sub_value}.")
            else:
                print("OH NOOO")

    # ------------------------------------------------------------------------------------------------------------------
    def _handler_readMessage(self, data):
        logging.warning("Read messages are not implemented yet")

    # ------------------------------------------------------------------------------------------------------------------
    def _handler_commandMessage(self, commands):
        # Go over the entries in the data dictionary
        for entry, value in commands.items():
            # Check if the entry is also in the parameters dict
            if entry not in self.commands:
                logging.warning(f"Received command {entry}, which is not a valid entry.")
                continue

            # Check if the entry is a parameter or a dict (group of parameters).
            # TODO: We only allow one level of grouping for now

            # The entry is a command
            if isinstance(self.commands[entry], Command):
                if not isinstance(value, dict) and len(self.commands[entry].arguments) > 0:
                    logging.warning(f"Command value is not a dict. Skipping command execution")
                    continue
                self.commands[entry].execute(value)
            #
            # # The entry is a group of commands
            # elif isinstance(self.parameters[entry], dict) and isinstance(value, dict):
            #     for sub_entry, sub_value in value.items():
            #         if not (isinstance(self.parameters[entry][sub_entry], Parameter)):
            #             logging.warning(f"Cannot set parameter {sub_entry}: Grouping level exceeds allowed level of: 1")
            #         self.parameters[entry][sub_entry].set(sub_value)

    # ------------------------------------------------------------------------------------------------------------------
    def _handlerEventMessage(self, message):
            if message.event == 'sync':
                for callback in self.callbacks['sync']:
                    callback(message.data)

    # ------------------------------------------------------------------------------------------------------------------
    def _sendDeviceIdentification(self):
        msg = TCP_JSON_Message()
        msg.source = self.id
        msg.address = ''
        msg.type = 'event'
        msg.event = 'device_identification'

        # if self.parent is not None:
        #     parent_id = self.parent.id
        # else:
        #     parent_id = None

        # if len(self.children) > 0:
        #     child_id = [child.id for child in self.children]
        # else:
        #     child_id = []

        msg.data = {
            'device_class': self.device_class,
            'device_type': self.device_type,
            'device_name': self.name,
            'device_id': self.id,
            'address': self.id,
            'revision': self.device_revision,
            'data': generateDataDict(self.data),
            'commands': generateCommandDict(self.commands)
        }

        self._wifi_send(msg)
