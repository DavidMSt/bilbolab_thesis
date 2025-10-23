# import dataclasses
# import time
#
# # === OWN PACKAGES =====================================================================================================
# from core.communication.protocol import Message
# from core.communication.wifi.tcp.protocols.tcp_json_protocol import TCP_JSON_Message
# from core.communication.wifi.tcp.tcp_connection import TCP_Connection
# from core.utils.callbacks import callback_definition, CallbackContainer
# from core.utils.events import event_definition, ConditionEvent
# from core.utils.logging_utils import Logger
# from core.utils.time import TimeoutTimer
#
# # === GLOBAL VARIABLES =================================================================================================
# logger = Logger('device')
# logger.setLevel('INFO')
#
#
# # === CALLBACKS ========================================================================================================
# @callback_definition
# class DeviceCallbacks:
#     registered: CallbackContainer
#     disconnected: CallbackContainer
#     rx: CallbackContainer
#     stream: CallbackContainer
#     event: CallbackContainer
#     timeout: CallbackContainer
#
#
# # === EVENTS ===========================================================================================================
# @event_definition
# class DeviceEvents:
#     rx: ConditionEvent
#     stream: ConditionEvent
#     event: ConditionEvent = ConditionEvent(flags=[('event', str)])
#     timeout: ConditionEvent
#
#
# # ======================================================================================================================
# class Request:
#     event: ConditionEvent
#     id: int
#
#     def __init__(self):
#         self.event = ConditionEvent()
#
#
# # ======================================================================================================================
# @dataclasses.dataclass
# class DataValue:
#     identifier: str
#     description: str
#     datatype: (tuple, type)
#     limits: list
#     writable: bool
#     value: object
#
#
# # ======================================================================================================================
# @dataclasses.dataclass
# class DeviceInformation:
#     device_class: str = ''
#     device_type: str = ''
#     device_name: str = ''
#     device_id: str = ''
#     address: str = ''
#     revision: int = 0
#
#
# # ======================================================================================================================
# class Device:
#     information: DeviceInformation
#
#     data: dict
#     commands: dict
#
#     tcp_connection: TCP_Connection
#
#     callbacks: DeviceCallbacks
#     events: DeviceEvents
#
#     last_heartbeat: float | None
#     heartbeat_timer: TimeoutTimer
#
#     _readRequests = dict[int, Request]
#
#     # === INIT =========================================================================================================
#     def __init__(self, connection: TCP_Connection = None):
#         self.tcp_connection = connection
#
#         self.information = DeviceInformation()
#
#         self.data = {}
#         self.commands = {}
#         self.last_heartbeat = None
#         self.heartbeat_timer = TimeoutTimer(timeout_time=5, timeout_callback=self._heartBeatTimeout_callback)
#
#         self._readRequests = {}
#         self.callbacks = DeviceCallbacks()
#         self.events = DeviceEvents()
#
#     # === PROPERTIES ===================================================================================================
#     @property
#     def address(self):
#         if self.tcp_connection is not None:
#             return self.tcp_connection.address
#         else:
#             return None
#
#     # ------------------------------------------------------------------------------------------------------------------
#     @property
#     def tcp_connection(self):
#         return self._connection
#
#     @tcp_connection.setter
#     def tcp_connection(self, connection):
#         self._connection = connection
#         if self._connection is not None:
#             self.tcp_connection.callbacks.rx.register(self._rx_callback)
#             self.tcp_connection.callbacks.disconnected.register(self._disconnected_callback)
#
#     # === METHODS ======================================================================================================
#     def close(self):
#         self.tcp_connection.close()
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def write(self, parameter, value, request_response: bool = False, timeout: float = 0.1):
#         msg = TCP_JSON_Message()
#         msg.type = 'write'
#         msg.address = ''
#         msg.source = ''
#         msg.request_response = request_response
#
#         if isinstance(parameter, str):
#             params = parameter.split('/')
#
#             if len(params) == 1:
#                 msg.data = {
#                     parameter: value
#                 }
#             elif len(params) == 2:
#                 msg.data = {
#                     params[0]: {
#                         params[1]: value
#                     }
#                 }
#             else:
#                 raise Exception("Levels >1 are not allowed for parameters")
#
#         elif isinstance(parameter, dict):
#             msg.data = parameter
#
#         # --- FIX: Add request before sending ---
#         if request_response:
#             request = self._addRequest(message_id=msg.id)
#         self.send(message=msg)
#         if request_response:
#             if request.event.wait(timeout=timeout):
#                 self._readRequests.pop(request.id)
#                 data = request.event.get_data()
#                 success = data['success']
#                 return success
#             else:
#                 self._readRequests.pop(request.id)
#                 return Exception("Timeout")
#         else:
#             return True
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def read(self, parameter: str, return_type: type, timeout: float = 0.1):
#         msg = TCP_JSON_Message()
#         msg.address = ''
#         msg.source = ''
#         msg.type = 'read'
#         msg.request_response = True
#
#         msg.data = {
#             'parameter': parameter
#         }
#
#         # --- FIX: Add request before sending ---
#         request = self._addRequest(message_id=msg.id)
#         self.send(msg)
#
#         if request.event.wait(timeout=timeout):
#             self._readRequests.pop(request.id)
#             data = request.event.get_data()
#             # Check if the read was a success
#             if data['success']:
#                 return data['output']
#             else:
#                 raise NotImplementedError("TODO")
#         else:
#             self._readRequests.pop(request.id)
#             return Exception("Timeout")
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def function(self, function: str, data, return_type: type = None, request_response: bool = False,
#                  timeout: float = 1):
#         msg = TCP_JSON_Message()
#         msg.address = ''
#         msg.source = ''
#         msg.type = 'function'
#         msg.request_response = request_response
#
#         msg.data = {
#             'function': function,
#             'input': data
#         }
#         time1 = time.perf_counter()
#         # --- FIX: Register request before sending ---
#         if request_response:
#             request = self._addRequest(message_id=msg.id)
#
#         self.send(msg)
#
#         if request_response:
#             if request.event.wait(timeout=timeout, stale_event_time=0.1):
#                 response_time = (time.perf_counter() - time1) * 1000
#                 # logger.warning(f"Got response for function \"{function}\"! Response time: {response_time:.0f}")
#                 # Get the response data
#                 data = request.event.get_data()
#
#                 # Check if it was a success
#                 success = data['success']
#
#                 self._readRequests.pop(request.id)
#                 if return_type is None:
#                     return success
#                 else:
#                     if success:
#                         return data['output']
#                     else:
#                         return None
#
#             else:
#                 self._readRequests.pop(request.id)
#                 logger.error(f"Timeout for function request {request.id}")
#                 raise TimeoutError
#         else:
#             return True
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def sendEvent(self, event, data, request_response: bool = False, timeout: float = 1):
#         msg = TCP_JSON_Message()
#         msg.address = ''
#         msg.source = ''
#         msg.type = 'event'
#         msg.request_response = request_response
#         msg.event = event
#
#         msg.data = {
#             'data': data
#         }
#
#         request = None
#
#         if request_response:
#             request = self._addRequest(message_id=msg.id)
#
#         self.send(msg)
#
#         if request_response:
#             if request.event.wait(timeout=timeout):
#                 self._readRequests.pop(request.id)
#                 return True
#             else:
#                 self._readRequests.pop(request.id)
#                 return Exception("Timeout")
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def send(self, message: Message):
#         # Check if the message has the correct protocol
#         # Send the message via the connection
#         if self.tcp_connection.connected:
#             try:
#                 self.tcp_connection.send(message)
#             except OSError:
#                 logger.warning("Cannot send message")
#
#     # === PRIVATE METHODS ==============================================================================================
#     def _rx_callback(self, msg, *args, **kwargs):
#         # Check if this message has the correct protocol
#         # Handle the message based on the type of message
#         if msg.type == 'response':
#             self._handleResponseMessage(msg)
#         elif msg.type == 'event':
#             self._handleEventMessage(msg)
#         elif msg.type == 'stream':
#             self._handleStreamMessage(msg)
#         else:
#             logger.warning(f"Got an unsupported message type: {msg.type}")
#
#         self.callbacks.rx.call(msg, self)
#         # self.events.rx.set(resource=msg)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def _handleEventMessage(self, message: TCP_JSON_Message):
#
#         if message.event == 'device_identification':
#             self._handleIdentificationEvent(message.data)
#         elif message.event == 'heartbeat':
#             self.heartbeat_timer.reset()
#
#         for callback in self.callbacks.event:
#             callback(message, self)
#
#         self.events.event.set(resource=message, flags={'event': message.event})
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def _handleStreamMessage(self, message: TCP_JSON_Message):
#         for callback in self.callbacks.stream:
#             callback(message, self)
#
#         self.events.stream.set(resource=message)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def _handleResponseMessage(self, message: TCP_JSON_Message):
#         # Check if the response was in the requests
#         if message.request_id in self._readRequests:
#             read_request = self._readRequests[message.request_id]
#             read_request.event.set(resource=message.data)
#         else:
#             logger.debug(f"Got a response for an unknown request: {message.request_id}")
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def _handleIdentificationEvent(self, data):
#
#         self.information.device_class = data['device_class']
#         self.information.device_type = data['device_type']
#         self.information.device_name = data['device_name']
#         self.information.device_id = data['device_id']
#         self.information.address = data['address']
#         self.information.revision = data['revision']
#
#         # Set the data
#
#         # Set the commands
#         self.tcp_connection.registered = True
#         # self.heartbeat_timer.start()
#
#         for callback in self.callbacks.registered:
#             callback(self)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def _disconnected_callback(self, connection: TCP_Connection):
#         for callback in self.callbacks.disconnected:
#             callback(self)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def _heartBeatTimeout_callback(self):
#         for callback in self.callbacks.timeout:
#             callback(self)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     def _addRequest(self, message_id) -> Request:
#         read_request = Request()
#         read_request.id = message_id
#         self._readRequests[read_request.id] = read_request
#         return read_request
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # ------------------------------------------------------------------------------------------------------------------
#     # ------------------------------------------------------------------------------------------------------------------
#     # ------------------------------------------------------------------------------------------------------------------
