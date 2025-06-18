from __future__ import annotations

import dataclasses
import time
import logging
from websocket_server import WebsocketServer as ws_server
import websocket
import threading
import json

from core.utils.events import event_definition, ConditionEvent
from core.utils.exit import register_exit_callback
from core.utils.callbacks import CallbackContainer, callback_definition
from core.utils.logging_utils import Logger


# ======================================================================================================================
@callback_definition
class WebsocketServerClient_Callbacks:
    disconnected: CallbackContainer
    message: CallbackContainer


class WebsocketServerClient:
    client: dict
    callbacks: WebsocketServerClient_Callbacks

    server: WebsocketServer

    # === INIT =========================================================================================================
    def __init__(self, client, server):
        self.client = client
        self.callbacks = WebsocketServerClient_Callbacks()
        self.server = server

    # === PROPERTIES ===================================================================================================
    @property
    def address(self):
        return self.client['address'][0]

    @property
    def port(self):
        return self.client['address'][1]

    # === METHODS ======================================================================================================
    def onMessage(self, message):
        self.callbacks.message.call(message)

    # ------------------------------------------------------------------------------------------------------------------
    def send(self, message):
        self.server.sendToClient(self.client, message)

    # ------------------------------------------------------------------------------------------------------------------
    def onDisconnect(self):
        self.callbacks.disconnected.call()
        # del self


# ======================================================================================================================
@callback_definition
class SyncWebsocketServer_Callbacks:
    new_client: CallbackContainer
    client_disconnected: CallbackContainer
    message: CallbackContainer


@event_definition
class SyncWebsocketServer_Events:
    new_client: ConditionEvent
    message: ConditionEvent
    client_disconnected: ConditionEvent


class WebsocketServer:
    callbacks: SyncWebsocketServer_Callbacks
    events: SyncWebsocketServer_Events

    clients: list[WebsocketServerClient]

    _server: ws_server | None

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._server = None
        self.clients = []  # Store the connected clients
        self.running = False
        self.thread = None

        self.logger = Logger('Websocket Server', 'DEBUG')

        self.events = SyncWebsocketServer_Events()
        self.callbacks = SyncWebsocketServer_Callbacks()

        # Exit handling
        register_exit_callback(self.stop)

    def start(self):
        """
        Start the WebSocket server in a separate thread (non-blocking).
        """
        if not self.running:
            self._server = ws_server(host=self.host, port=self.port)
            self.running = True
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()

    def _run_server(self):
        """
        Run the WebSocket server (blocking call). Should be run in a separate thread.
        """
        # Attach callbacks
        self._server.set_fn_new_client(self._on_new_client)
        self._server.set_fn_client_left(self._on_client_left)
        self._server.set_fn_message_received(self._on_message_received)

        try:
            self._server.run_forever()
        except Exception as e:
            print(f"Error in server loop: {e}")
        finally:
            self.running = False

    # ------------------------------------------------------------------------------------------------------------------
    def _on_new_client(self, client, server):
        websocket_client = WebsocketServerClient(client, self)
        self.clients.append(websocket_client)  # Add a client to the list
        self.logger.info(f"New client connected: {client['address']}")
        self.callbacks.new_client.call(websocket_client)
        self.events.new_client.set(websocket_client)

    # ------------------------------------------------------------------------------------------------------------------
    def _on_client_left(self, client, server):

        websocket_client = next((c for c in self.clients if c.client == client), None)

        if websocket_client:
            self.logger.info(f"Client disconnected: {client['address']}")
            self.clients.remove(websocket_client)
            self.callbacks.client_disconnected.call(websocket_client)
            self.events.client_disconnected.set(websocket_client)
            websocket_client.onDisconnect()

    # ------------------------------------------------------------------------------------------------------------------
    def _on_message_received(self, client, server, message):
        message = json.loads(message)

        websocket_client = next((c for c in self.clients if c.client == client), None)

        if websocket_client:
            self.logger.debug(f"Message received from {client['address']}: {message}")
            self.callbacks.message.call(websocket_client, message)
            self.events.message.set(websocket_client, message)
            websocket_client.onMessage(message)

    # ------------------------------------------------------------------------------------------------------------------
    def send(self, message):
        """
        Send a message to all connected clients.
        """
        if isinstance(message, dict):
            message = json.dumps(message)
        for client in self.clients:
            self._server.send_message(client.client, message)

    def sendToClient(self, client, message):
        """
        Send a message to a specific client.
        """
        if isinstance(message, dict):
            message = json.dumps(message)

        if isinstance(client, WebsocketServerClient):
            if client in self.clients:
                self._server.send_message(client.client, message)
        elif isinstance(client, dict):
            self._server.send_message(client, message)

    def stop(self, *args, **kwargs):
        """
        Stop the WebSocket server.
        """
        if self.running:
            self._server.server_close()
            self._server.disconnect_clients_gracefully()
            self._server.shutdown()
            if self.thread:
                self.thread.join()
            self.running = False

        self.logger.info("Server stopped")


# ======================================================================================================================
@callback_definition
class SyncWebsocketClient_Callbacks:
    message: CallbackContainer
    connected: CallbackContainer
    disconnected: CallbackContainer
    error: CallbackContainer


@event_definition
class SyncWebsocketClient_Events:
    message: ConditionEvent
    connected: ConditionEvent
    disconnected: ConditionEvent
    error: ConditionEvent


class WebsocketClient:
    callbacks: SyncWebsocketClient_Callbacks
    events: SyncWebsocketClient_Events

    _thread: threading.Thread
    _exit: bool = False
    _debug: bool

    # === INIT =========================================================================================================
    def __init__(self, address, port, debug=True, reconnect=True):

        self.address = address
        self.port = port

        self.uri = f"ws://{address}:{port}"
        self.ws = None
        self.connected = False

        self.reconnect = reconnect
        self._debug = debug

        self.callbacks = SyncWebsocketClient_Callbacks()
        self.events = SyncWebsocketClient_Events()

        self._thread = threading.Thread(target=self.task, daemon=True)
        self.ws_thread = None

        # Disable the internal websocket logger, since it messes with other modules
        self.logger = Logger('Websocket Client', 'DEBUG')
        logging.getLogger("websocket").setLevel(logging.CRITICAL)
        register_exit_callback(self.close)

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self.logger.info("Connection closed")
        self.ws.close()
        self._exit = True
        self._thread.join()

    # ------------------------------------------------------------------------------------------------------------------
    def task(self):
        while not self._exit:
            if not self.connected:
                self.logger.debug("Attempting to connect...")
                self._connect()
            time.sleep(1)

    # ------------------------------------------------------------------------------------------------------------------
    def connect(self):
        self._thread = threading.Thread(target=self.task, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def _connect(self):
        """
        Attempt to connect to the WebSocket server with retry logic.
        """

        try:
            # Create WebSocket app
            self.ws = websocket.WebSocketApp(self.uri,
                                             on_open=self.on_open,
                                             on_close=self.on_close,
                                             on_message=self.on_message,
                                             on_error=self.on_error)

            # Run in a separate thread
            self.ws_thread = threading.Thread(
                target=lambda: self.ws.run_forever(
                    ping_interval=5,
                    ping_timeout=2,
                    reconnect=False
                ),
                daemon=True
            )
            self.ws_thread.start()

            # Wait for connection success or failure
            timeout = 5  # Adjust timeout as needed
            start_time = time.time()

            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.5)

            if self.connected:
                return True
            else:
                self.ws.close()
                self.ws_thread.join()
                return False

        except Exception as e:
            if self._debug:
                self.logger.warning(f"Error in connection attempt: {e}")
                return False

    # ------------------------------------------------------------------------------------------------------------------
    def send(self, message):
        """
        Send a message to the server.
        """
        if self.connected:
            if isinstance(message, dict):
                message = json.dumps(message)
            self.ws.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def disconnect(self):
        """
        Close the WebSocket connection.
        """
        if self.connected:
            self.ws.close()
            self._thread.join()

    # ------------------------------------------------------------------------------------------------------------------
    def on_open(self, ws):
        self.connected = True
        self.logger.info("Connection successful.")
        self.callbacks.connected.call()
        self.events.connected.set()

    # ------------------------------------------------------------------------------------------------------------------
    def on_close(self, ws, close_status_code, close_msg):
        if self.connected:
            self.connected = False
            self.callbacks.disconnected.call()
            self.events.disconnected.set()
            self.logger.info("Connection closed by server")

    # ------------------------------------------------------------------------------------------------------------------
    def on_message(self, ws, message):
        self.logger.debug(f"Message received: {message}")
        message = json.loads(message)
        self.callbacks.message.call(message)
        self.events.message.set(message)


    # ------------------------------------------------------------------------------------------------------------------
    def on_error(self, ws, error):
        self.callbacks.error.call(error)
        self.events.error.set(error)


# ======================================================================================================================
@dataclasses.dataclass
class WebsocketMessage:
    address: str
    source: str

    id: str
    type: str
    data: dict

    request: bool = False


if __name__ == '__main__':
    host = 'localhost'
    port = 8080
    # Start the server
    server = WebsocketServer('localhost', 8080)
    server.start()

    # Start the client
    client = WebsocketClient(host, port)
    client.connect()

    time.sleep(5)

    server.stop()
    time.sleep(6)
    server.start()

    while True:
        client.send({'test': 'test'})
        time.sleep(1)
        server.send({'a': 2})
        time.sleep(1)
