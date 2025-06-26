# from textual.app import App
import logging
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up four levels (adjust the number of ".." if your structure changes)
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# ======================================================================================================================
# Insert the top-level directory into sys.path so that Python can find top_level_module.py
if top_level_dir not in sys.path:
    sys.path.insert(0, top_level_dir)

from extensions.cli.src.cli import CLI, CommandSet
from extensions.cli.src.cli_gui_app import CLI_GUI_App
from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.exit import register_exit_callback
import core.utils.logging_utils as logging_utils
from core.utils.logging_utils import Logger
from core.utils.time import delayed_execution

# === OWN PACKAGES =====================================================================================================
from core.utils.websockets.websockets import WebsocketClient, WebsocketServer


# ======================================================================================================================

@callback_definition
class CLI_GUI_Server_Callbacks:
    command: CallbackContainer


class CLI_GUI_Server:
    server: WebsocketServer
    cli: CLI
    callbacks: CLI_GUI_Server_Callbacks
    clients: list

    def __init__(self, address='localhost', port=8090, buffer_logs: bool = True):

        self.server = WebsocketServer(host=address, port=port)
        self.callbacks = CLI_GUI_Server_Callbacks()
        self.buffer_logs = buffer_logs
        if self.buffer_logs:
            self.log_queue = []  # Initialize the log queue if buffering is enabled
        self.server.callbacks.message.register(self._gui_message_callback)
        self.server.callbacks.new_client.register(self._new_client_callback)

        self.cli = CLI()

        self.cli.callbacks.update.register(self.sendCLIDescription)

        logging_utils.enable_redirection(self._log_redirect, redirect_all=False)

        # Redirect CLI text output through sendLog (which now handles buffering)
        self.cli.text_output_function = self.sendLog
        self.clients = []

    # === METHODS ======================================================================================================
    def start(self):
        self.server.start()

    # ------------------------------------------------------------------------------------------------------------------
    def updateCLI(self, root_set: CommandSet = None):
        if root_set is None:
            root_set = self.cli.root_set

        self.cli.setRootSet(root_set)
        self.sendCLIDescription()

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def connected(self):
        return len(self.clients) > 0

    # ------------------------------------------------------------------------------------------------------------------
    def _send_log_data(self, data):
        """
        Sends a log message to the client(s). If buffering is enabled and no client is connected,
        the log data is queued. Otherwise, any queued logs are flushed before sending the new log.
        """
        if self.buffer_logs and not self.connected:
            self.log_queue.append(data)
        else:
            if self.buffer_logs and self.log_queue:
                # Flush any buffered log messages one by one
                for queued_data in self.log_queue:
                    self.server.send(queued_data)
                self.log_queue.clear()
            self.server.send(data)

    # ------------------------------------------------------------------------------------------------------------------
    def sendLog(self, message):
        data = {
            'type': 'log',
            'data': {
                'text': message,
                'color': '#A0A0A0'}
        }
        self._send_log_data(data)

    # ------------------------------------------------------------------------------------------------------------------
    def sendWarning(self, message):
        data = {
            'type': 'log',
            'data': {
                'text': message,
                'color': "#FFA500",
            },
        }
        self._send_log_data(data)

    # ------------------------------------------------------------------------------------------------------------------
    def sendError(self, message):
        data = {
            'type': 'log',
            'data': {
                'text': message,
                'color': "#B53737",
            },
        }
        self._send_log_data(data)

    # ------------------------------------------------------------------------------------------------------------------
    def sendCLIDescription(self):
        commands = self.cli.getCommandSetDescription()
        data = {
            'type': 'commands',
            'data': commands
        }
        self.server.send(data)

    # === PRIVATE METHODS ==============================================================================================
    def _gui_message_callback(self, client, message, *args, **kwargs):

        if message['type'] == 'command':
            if 'data' in message:
                try:
                    ret = self.cli.executeFromConnectorDict(message['data'])
                    if message['data']['name'] == 'help':
                        self.sendLog(ret)
                except Exception as e:
                    print(f"Something went wrong when executing command in cli from cli_gui_server: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    def _new_client_callback(self, client, *args, **kwargs):
        self.sendCLIDescription()
        self.clients.append(client)
        # As soon as a new client connects, flush any buffered logs.
        if self.buffer_logs and hasattr(self, 'log_queue') and self.log_queue:
            for queued_data in self.log_queue:
                self.server.send(queued_data)
            self.log_queue.clear()

    # ------------------------------------------------------------------------------------------------------------------
    def _log_redirect(self, log_entry, unformatted_entry, logger: Logger, level, *args, **kwargs):
        # Total visible width for the header (including the brackets and colon)
        if logger.name == 'tcp':
            pass
        header_width = 25

        # The visible header will be of the form: "[" + logger_name + "]:"
        # That means there are 3 extra characters (the opening bracket, closing bracket, and colon).
        max_name_length = header_width - 3

        # Get the raw logger name and optionally truncate if too long.
        raw_name = logger.name
        if len(raw_name) > max_name_length:
            raw_name = raw_name[:max_name_length]

        # Build the visible header (without ANSI codes) to compute the needed padding.
        visible_header = f"[{raw_name.upper()}]:"
        # Calculate the number of spaces needed after the colon.
        padding = " " * (header_width - len(visible_header))

        # # Only the logger name is colored, while the brackets and colon remain default.
        # if logger.color is None:
        #     format_string = f"[rgb(150,150,150)]"
        # else:
        #     format_string = f"[rgb({logger.color[0]},{logger.color[1]},{logger.color[2]})]"
        #
        # colored_header = f"{format_string}{visible_header}[/]"

        # Combine the colored header and the computed padding.
        final_header = f"{visible_header}{padding}"

        # Create the final string: header followed by the log message.
        string = f"{final_header}{unformatted_entry}"

        if level == logging.INFO:
            self.sendLog(string)
        elif level == logging.WARNING:
            self.sendWarning(string)
        elif level == logging.ERROR:
            self.sendError(string)


# ======================================================================================================================
class CLI_GUI_Client:
    app: CLI_GUI_App
    websocket: WebsocketClient

    connected: bool

    _exit: bool = False

    def __init__(self, address, port=8090):
        self.websocket = WebsocketClient(address, port, debug=False)
        self.websocket.callbacks.connected.register(self._websocket_connected_callback)
        self.websocket.callbacks.message.register(self._websocket_message_callback)
        self.websocket.callbacks.disconnected.register(self._websocket_disconnected_callback)

        self.connected = False
        self.app = CLI_GUI_App()

        self.app.callbacks.command.register(self._gui_command_callback)

        self._exit = False
        register_exit_callback(self.close)

    # ------------------------------------------------------------------------------------------------------------------
    def init(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        delayed_execution(self.websocket.connect, delay=1)
        self.app.run()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self._exit = True
        self.websocket.close()
        self.app.exit()

    # === PRIVATE METHODS ==============================================================================================
    def _websocket_connected_callback(self, *args, **kwargs):
        self.connected = True
        self.app.addLog("Websocket connected")

    def _websocket_disconnected_callback(self, *args, **kwargs):
        self.connected = False
        self.app.addLog("Websocket disconnected")
        self.app.call_from_thread(self.app.setCommands, None)

    def _websocket_message_callback(self, message, *args, **kwargs):
        if 'type' not in message:
            print("Faulty message")
            return

        if message['type'] == 'log':
            self._handle_log_message(message)

        if message['type'] == 'commands':
            self._handle_command_set_message(message)

        if message['type'] == 'robot_data':
            self._handle_robot_data_message(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _handle_log_message(self, message):
        log_message = message['data']['text']
        color = message['data'].get('color', None)
        self.app.addLog(log_message, color=color)

    # ------------------------------------------------------------------------------------------------------------------
    def _handle_command_set_message(self, message):
        self.app.call_from_thread(self.app.setCommands, message['data'])

    def _handle_robot_data_message(self, message):
        self.app.call_from_thread(self.app.setRobotData, message['data'])

    # ------------------------------------------------------------------------------------------------------------------
    def _gui_command_callback(self, command, *args, **kwargs):
        if self.websocket.connected:
            message = {
                'type': 'command',
                'data': command,
            }
            self.websocket.send(message)


# ======================================================================================================================
if __name__ == "__main__":

    gui = CLI_GUI_Client(address='localhost', port=8080)
    gui.init()
    gui.start()

    while True:
        time.sleep(1)
