import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.websockets import WebsocketServer as WebsocketClass
from core.utils.exit import register_exit_callback

frontend_dir = f"{os.path.dirname(__file__)}/frontend/"


@callback_definition
class NodeJSGui_Callbacks:
    websocket_client_connected: CallbackContainer
    rx_message: CallbackContainer


def install():
    os.chdir(frontend_dir)
    os.system("npm install")


class NodeJSGui:
    websocket_stream: WebsocketClass
    websocket_messages: WebsocketClass

    _frontend_thread: threading.Thread
    client_connected: bool

    callbacks: NodeJSGui_Callbacks

    def __init__(self):
        self.websocket_stream = WebsocketClass('localhost', 8765)
        self.websocket_messages = WebsocketClass('localhost', 8766)

        self.websocket_messages.callbacks.message.register(self._rxMessage_callback)
        self.websocket_messages.callbacks.new_client.register(self._websocketClientConnected_callback)

        self.frontend_process = None
        self.browser_process = None  # Track the browser process

        self.client_connected = False

        self.callbacks = NodeJSGui_Callbacks()

        register_exit_callback(self.close)

    # ------------------------------------------------------------------------------------------------------------------
    def init(self):
        if not (self.checkInstallation()):
            raise Exception("GUI is not installed! Run npm install in the frontend folder.")
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.websocket_stream.start()
        self.websocket_messages.start()
        self._run_frontend()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        if self.frontend_process is not None:
            self.frontend_process.terminate()
        if self.browser_process is not None:
            self.browser_process.kill()

    # ------------------------------------------------------------------------------------------------------------------
    def sendMessage(self, message_type, data):
        message = {"timestamp": time.time(), "type": message_type, "data": data}
        self.websocket_messages.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def sendEvent(self, event, device_id):
        message = {"timestamp": time.time(), "event": event, device_id: device_id}
        self.websocket_messages.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def print(self, message, *args, **kwargs):
        self.sendMessage(message_type="message", data=message)

    # ------------------------------------------------------------------------------------------------------------------

    def sendStream(self, data):
        self.websocket_stream.send(data)

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def checkInstallation(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the path to the target folder
        target_folder = os.path.join(script_dir, 'frontend', 'node_modules')

        # Check if the folder exists and is a directory
        return os.path.isdir(target_folder)

    # === PRIVATE METHODS ==============================================================================================
    def _rxMessage_callback(self, message, *args, **kwargs):
        message = json.loads(message)
        for callback in self.callbacks.rx_message:
            callback(message, *args, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    def _websocketClientConnected_callback(self, *args, **kwargs):
        self.client_connected = True
        for callback in self.callbacks.websocket_client_connected:
            callback(*args, **kwargs)

    def _run_frontend(self):
        os.chdir(frontend_dir)

        # Start the frontend server
        self.frontend_process = subprocess.Popen(
            "npm run dev",
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Determine the URL to open in the browser
        url = "http://localhost:5173"

        # Try to find the Chrome executable path dynamically
        chrome_path = None

        # Attempt to find Chrome across different platforms
        if sys.platform.startswith('win'):
            # Windows
            chrome_path = shutil.which("chrome") or shutil.which("google-chrome")
            if chrome_path is None:
                default_path = "C:/Program Files/Google/Chrome/Application/chrome.exe"
                if os.path.exists(default_path):
                    chrome_path = default_path
        elif sys.platform == 'darwin':
            # macOS
            chrome_path = shutil.which("chrome") or shutil.which("google-chrome")
            if chrome_path is None:
                default_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                if os.path.exists(default_path):
                    chrome_path = default_path
        elif sys.platform.startswith('linux'):
            # Linux/Ubuntu
            chrome_path = shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium-browser")

        # # If Chrome path is found, open the URL using Chrome in a new window
        # if chrome_path is not None:
        #     self.browser_process = subprocess.Popen([chrome_path, "--new-window", url])
        # else:
        #     # Fallback: open URL in the default browser
        #     webbrowser.open(url)
        webbrowser.open(url)
