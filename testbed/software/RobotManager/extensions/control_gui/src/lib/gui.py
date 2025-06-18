from __future__ import annotations

import abc
import math
import random
import time
from abc import abstractmethod

from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.colors import rgb_to_hex
from core.utils.exit import register_exit_callback
from core.utils.files import relativeToFullPath
from core.utils.js.vite import run_vite_app
from core.utils.logging_utils import Logger, setLoggerLevel
from core.utils.network.network import pingAddress, getHostIP
from core.utils.websockets.websockets import WebsocketServer, WebsocketClient, WebsocketServerClient
from extensions.control_gui.src.lib.plot.jsplot import JSPlotTimeSeries
from extensions.control_gui.src.lib.plot.plot_widget import PlotWidget
from extensions.control_gui.src.lib.widgets.buttons import Button
from extensions.control_gui.src.lib.objects import GUI_Object_Group, GUI_Object
from extensions.control_gui.src.lib.utilities import check_for_spaces, split_path, strip_id
from extensions.control_gui.src.lib.map.map_widget import MapWidget

# ======================================================================================================================
'''
Message from Frontend looks like:

'type': 'event',
'event': event_name,
'id': id_of_the_object
'data': data_of_the_object
'''

'''
Messages from the Backend look like:

'type': 'update'
'id': id_of_the_object
'data': data_of_the_object


'type': 'init'
'configuration': ...


'type': 'add'
'id': id_of_the_object
'object_type': type_of_the_object
'configuration': data_of_the_object



'type': 'remove'
'id': id_of_the_object
'''


# === CATEGORY =========================================================================================================
class ControlGUI_Category_Headbar:
    ...

    def __init__(self):
        ...

    def getPayload(self):
        payload = {}
        return payload


# ----------------------------------------------------------------------------------------------------------------------
@callback_definition
class ControlGUI_Category_Callbacks:
    update: CallbackContainer
    add: CallbackContainer
    remove: CallbackContainer


# ----------------------------------------------------------------------------------------------------------------------
class ControlGUI_Category:
    id: str
    pages: dict[str, ControlGUI_Page]
    categories: dict[str, ControlGUI_Category]

    name: str
    icon: str
    headbar: ControlGUI_Category_Headbar

    configuration: dict

    parent: GUI | ControlGUI_Category | None

    # === INIT =========================================================================================================
    def __init__(self, id: str, name: str = None, **kwargs):

        if check_for_spaces(id):
            raise ValueError(f"Category id '{id}' contains spaces")
        if '/' in id:
            raise ValueError(f"Category id '{id}' contains slashes")
        if ":" in id:
            raise ValueError(f"Category id '{id}' contains colons")

        id = f"c_{id}"

        default_config = {
            'color': None,
            'max_pages': 10,
            'collapsed': False,
            'icon': '📁',
        }

        self.configuration = {**default_config, **kwargs}

        self.id = id

        self.name = name if name is not None else id

        self.pages = {}
        self.categories = {}

        self.parent = None

        self.callbacks = ControlGUI_Category_Callbacks()

        self.headbar = ControlGUI_Category_Headbar()

        self.logger = Logger(f"Category {self.id}", 'DEBUG')

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def uid(self):
        if isinstance(self.parent, GUI):
            return f"{self.parent.id}::{self.id}"
        elif isinstance(self.parent, ControlGUI_Category):
            return f"{self.parent.uid}/{self.id}"
        else:
            return self.id

    # ------------------------------------------------------------------------------------------------------------------
    def getGUI(self):
        if isinstance(self.parent, GUI):
            return self.parent
        elif isinstance(self.parent, ControlGUI_Category):
            return self.parent.getGUI()
        else:
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def getObjectByPath(self, path: str):
        """
        Given a relative (or even absolute) path such as
          "page1/button1/subgroup/obj2"
        or
          "myGui::category1/page1/button1",
        this will strip off any gui_id:: prefix, then
        split off the page‐ID and delegate the rest.
        """
        # 1) normalize slashes
        trimmed = path.strip("/")

        # 3) now trimmed should be “pageID[/rest]”
        first_segment, remainder = split_path(trimmed)
        if not first_segment:
            return None

        if first_segment in self.pages:
            page = self.pages[first_segment]
            if not remainder:
                return page
            return page.getObjectByPath(remainder)

        elif first_segment in self.categories:
            category = self.categories[first_segment]
            if not remainder:
                return category
            return category.getObjectByPath(remainder)
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def getObjectByUID(self, uid):
        gui = self.getGUI()

        if gui is not None:
            return gui.getElementByUID(uid)
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def addPage(self, page: ControlGUI_Page, position=None):

        # Fist check if the position is valid if given
        if position is not None and position > self.configuration['max_pages']:
            raise ValueError(f"Position {position} is out of range")

        if page.uid in self.pages:
            raise ValueError(f"Page with id {page.id} already exists")
        self.pages[page.id] = page
        page.category = self
        page.position = position

        message = {
            'type': 'add',
            'data': {
                'type': 'page',
                'parent': self.uid,
                'id': page.uid,
                'position': position,
                'config': page.getPayload(),
            }
        }

        self.sendMessage(message)

        return page

    # ------------------------------------------------------------------------------------------------------------------
    def removePage(self, page: ControlGUI_Page):
        if page.id not in self.pages:
            raise ValueError(f"Page with id {page.id} does not exist")
        del self.pages[page.id]

        message = {
            'type': 'remove',
            'data': {
                'type': 'page',
                'parent': self.uid,
                'id': page.uid,
            }
        }

        self.sendMessage(message)

    # ------------------------------------------------------------------------------------------------------------------
    def addCategory(self, category: ControlGUI_Category):
        if category.id in self.categories:
            raise ValueError(f"Category with id {category.id} already exists")
        category.parent = self
        self.categories[category.id] = category

        message = {
            'type': 'add',
            'data': {
                'type': 'category',
                'parent': self.uid,
                'id': category.uid,
                'config': category.getPayload(),
            }
        }

        self.sendMessage(message)

    # ------------------------------------------------------------------------------------------------------------------
    def removeCategory(self, category: ControlGUI_Category):
        if category.id not in self.categories:
            raise ValueError(f"Category with id {category.id} does not exist")
        del self.categories[category.id]

        message = {
            'type': 'remove',
            'data': {
                'type': 'category',
                'parent': self.uid,
                'id': category.uid,
            }
        }

        self.sendMessage(message)

    # ------------------------------------------------------------------------------------------------------------------
    def update(self):
        message = {
            'type': 'update',
            'id': self.uid,
            'data': self.getPayload()
        }
        self.sendMessage(message)

    # ------------------------------------------------------------------------------------------------------------------
    def getConfiguration(self) -> dict:
        configuration = {
            'id': self.uid,
            'type': 'category',
            'name': self.name,
            **self.configuration
        }
        return configuration

    # ------------------------------------------------------------------------------------------------------------------
    def getPayload(self) -> dict:
        payload = {
            'id': self.uid,
            'type': 'category',
            'config': self.getConfiguration(),
            'headbar': self.headbar.getPayload(),
            'pages': {k: v.getPayload() for k, v in self.pages.items()},
            'categories': {k: v.getPayload() for k, v in self.categories.items()},
        }
        return payload

    # ------------------------------------------------------------------------------------------------------------------
    def onMessage(self, message):
        self.logger.debug(f"Received message: {message}")
        object_path = message['id']

    # ------------------------------------------------------------------------------------------------------------------
    def sendMessage(self, message):
        gui = self.getGUI()
        if gui is not None:
            try:
                gui.broadcast(message)
            except Exception as e:
                self.logger.error(f"Error sending message: {e}")


# === PAGE =============================================================================================================
@callback_definition
class ControlGUI_Page_Callbacks:
    update: CallbackContainer
    add: CallbackContainer
    remove: CallbackContainer


# ----------------------------------------------------------------------------------------------------------------------
class ControlGUI_Page:
    """
    Represents a page in the Control GUI that holds GUI_Object instances
    in a fixed grid layout. Tracks occupied cells and supports manual
    or automatic placement of objects.
    """
    id: str
    objects: dict[str, dict]
    category: ControlGUI_Category | None
    config: dict

    name: str
    icon: str
    position: int | None = None

    def __init__(self, id: str,
                 icon: str = None,
                 name: str = None,
                 **kwargs):

        if check_for_spaces(id):
            raise ValueError(f"Page id '{id}' contains spaces")
        if '/' in id:
            raise ValueError(f"Category id '{id}' contains slashes")
        if ":" in id:
            raise ValueError(f"Category id '{id}' contains colons")

        default_config = {
            'color': None,
            'pageColor': [60, 60, 60, 1],
            'grid_size': (18, 50),  # (rows, columns)
        }

        self.config = {**default_config, **kwargs}

        self.id = f"p_{id}"
        self.icon = icon
        self.name = name if name is not None else id

        # Grid dimensions
        self._rows, self._cols = self.config['grid_size']
        # Occupancy grid: False = free, True = occupied
        self._occupied = [[False for _ in range(self._cols)] for _ in range(self._rows)]

        self.objects = {}
        self.category = None
        self.callbacks = ControlGUI_Page_Callbacks()
        self.logger = Logger(f"Page {self.id}", 'DEBUG')

    @property
    def uid(self):
        category_id = self.category.uid if self.category is not None else ''
        return f"{category_id}/{self.id}"

    # ------------------------------------------------------------------------------------------------------------------
    def getObjectByPath(self, path: str):
        """
        Given a path within this page (e.g. "button1" or
        "group1/subobj2"), find the direct child whose .id
        matches the first segment, and either return it or
        recurse into it if it’s a GUI_Object_Group.
        """
        # 1) normalize slashes
        trimmed = path.strip("/")

        # 3) split first id vs. remainder
        first_segment, remainder = split_path(trimmed)
        if not first_segment:
            return None

        # 4) search our objects (keyed by full uid, but match on obj.id)
        for full_uid, info in self.objects.items():
            obj = info["object"]
            if obj.id == first_segment:
                if not remainder:
                    return obj
                # must be a group to descend further
                if isinstance(obj, GUI_Object_Group):
                    return obj.getObjectByPath(remainder)
                return None

        return None

    # ------------------------------------------------------------------------------------------------------------------
    def getObjectByUID(self, uid):
        gui = self.getGUI()

        if gui is not None:
            return gui.getElementByUID(uid)
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def update(self):

        message = {
            'type': 'update',
            'id': self.uid,
            'data': self.getPayload()
        }

        self.sendMessage(message)

    # ------------------------------------------------------------------------------------------------------------------
    def addObject(self, obj: GUI_Object, row=None, column=None, width=2, height=2) -> GUI_Object:
        """
        Adds an object to the page at a given grid position.
        If the row or column is None, we automatically find the first available
        position for the object's size.
        """
        if obj.uid in self.objects:
            raise ValueError(f"Object with id {obj.uid} already exists on page {self.id}")

        # Determine placement
        if row is None or column is None:
            row, column = self._placeObject(row, column, width, height)
        else:
            self._checkSpace(row, column, width, height)

        # Mark cells occupied
        self._markSpace(row, column, width, height)

        # Store object placement
        self.objects[obj.uid] = {
            'object': obj,
            'row': row,
            'column': column,
            'width': width,
            'height': height,
        }
        obj.parent = self

        # self.logger.debug(
        #     f"Added object {obj.uid} to page {self.id} at ({row}, {column}) with size ({width}, {height})")

        message = {
            'type': 'add',
            'data': {
                'type': 'object',
                'parent': self.uid,
                'id': obj.uid,
                'config': {
                    'row': row,
                    'column': column,
                    'width': width,
                    'height': height,
                    **obj.getPayload(),
                }
            }
        }

        self.sendMessage(message)

        return obj

    # ------------------------------------------------------------------------------------------------------------------
    def removeObject(self, obj: GUI_Object):

        if obj.id not in self.objects:
            raise ValueError(f"Object with id {obj.uid} does not exist on page {self.id}")

        message = {
            'type': 'remove',
            'data': {
                'type': 'object',
                'parent': self.uid,
                'id': obj.uid,
            }
        }
        self.sendMessage(message)

        obj.parent = None
        del self.objects[obj.uid]

    # ------------------------------------------------------------------------------------------------------------------
    def getGUI(self):
        return self.category.getGUI()

    # ------------------------------------------------------------------------------------------------------------------
    def _checkSpace(self, row, column, width, height):
        # Validate bounds
        if row < 1 or column < 1 or row + height - 1 > self._rows or column + width - 1 > self._cols:
            raise ValueError("Object does not fit within grid bounds")
        # Check occupancy
        for r in range(row - 1, row - 1 + height):
            for c in range(column - 1, column - 1 + width):
                if self._occupied[r][c]:
                    raise ValueError("Grid cells already occupied")

    # ------------------------------------------------------------------------------------------------------------------
    def _markSpace(self, row, column, width, height):
        # Mark the grid cells as occupied
        for r in range(row - 1, row - 1 + height):
            for c in range(column - 1, column - 1 + width):
                self._occupied[r][c] = True

    # ------------------------------------------------------------------------------------------------------------------
    def _placeObject(self, row, column, width, height):
        """
        Finds the first available position for an object of given size.
        If one coordinate is fixed, searches along the other.
        """

        # Helper to test a candidate position
        def fits(r, c):
            if r < 1 or c < 1 or r + height - 1 > self._rows or c + width - 1 > self._cols:
                return False
            for rr in range(r - 1, r - 1 + height):
                for cc in range(c - 1, c - 1 + width):
                    if self._occupied[rr][cc]:
                        return False
            return True

        # Neither fixed: scan rows then cols
        if row is None and column is None:
            for r in range(1, self._rows - height + 2):
                for c in range(1, self._cols - width + 2):
                    if fits(r, c):
                        return r, c
        # Row fixed: scan columns
        elif row is not None and column is None:
            for c in range(1, self._cols - width + 2):
                if fits(row, c):
                    return row, c
        # Column fixed: scan rows
        elif column is not None and row is None:
            for r in range(1, self._rows - height + 2):
                if fits(r, column):
                    return r, column

        raise ValueError("No available space to place object")

    # ------------------------------------------------------------------------------------------------------------------
    def getConfiguration(self) -> dict:
        return {
            'id': self.uid,
            'name': self.name,
            'icon': self.icon,
            # 'color': rgb_to_hex(self.config.get('color', [1, 1, 1, 1])),
        }

    # ------------------------------------------------------------------------------------------------------------------
    def getPayload(self) -> dict:
        # Build payload for each object
        objs = {}
        for uid, info in self.objects.items():
            obj = info['object']
            payload = obj.getPayload()
            payload.update({
                'row': info['row'],
                'column': info['column'],
                'width': info['width'],
                'height': info['height'],
            })
            objs[uid] = payload

        return {
            'id': self.uid,
            'type': 'page',
            'position': self.position,
            'config': self.getConfiguration(),
            'objects': objs
        }

    # ------------------------------------------------------------------------------------------------------------------
    def onMessage(self, message):
        self.logger.debug(f"Received message: {message}")

    # ------------------------------------------------------------------------------------------------------------------
    def sendMessage(self, message):
        gui = self.getGUI()
        if gui is not None:
            try:
                gui.broadcast(message)
            except Exception as e:
                self.logger.error(f"Error sending message: {e}")


# === CHILD GUI ========================================================================================================

@callback_definition
class ControlGUI_ChildGUI_Callbacks:
    connect: CallbackContainer
    disconnect: CallbackContainer
    message: CallbackContainer


class ControlGUI_Child(abc.ABC):
    id: str
    address: str
    port: int

    def __init__(self, id, address, port, category_path: str = ''):
        self.id = id
        self.address = address
        self.port = port
        self.category_path = category_path
        self.callbacks = ControlGUI_ChildGUI_Callbacks()

    @abstractmethod
    def send(self, message):
        ...

    def _onConnect(self, *args, **kwargs):
        self.callbacks.connect.call()

    def _onDisconnect(self, *args, **kwargs):
        print("Disconnected from child GUI")
        self.callbacks.disconnect.call()

    def _onMessage(self, message):
        self.callbacks.message.call(message)


class ChildGUI_ServerClient(ControlGUI_Child):

    def __init__(self, id, address, port, client: WebsocketServerClient):
        super().__init__(id, address, port)
        self.client = client

        self.client.callbacks.message.register(self._onMessage)
        self.client.callbacks.disconnected.register(self._onDisconnect)

    def send(self, message):
        self.client.send(message)



class Child_Category(ControlGUI_Category):

    def __init__(self, ):

class ChildGUI_WebsocketClient(ControlGUI_Child):
    gui: GUI
    category: ControlGUI_Category | None
    path_in_gui: str

    def __init__(self, id, address, port, category_path, client: WebsocketClient, gui: GUI):
        super().__init__(id, address, port, category_path)
        self.client = client
        self.gui = gui

        self.client.callbacks.connected.register(self._onConnect)
        self.client.callbacks.disconnected.register(self._onDisconnect)
        self.client.callbacks.message.register(self._onMessage)

        self.client.connect()

        self.category = None
        self.path_in_gui = category_path

    # ------------------------------------------------------------------------------------------------------------------
    def send(self, message):
        self.client.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _onConnect(self, *args, **kwargs):
        super()._onConnect(*args, **kwargs)
        message = {
            'type': 'handshake',
            'data': {
                'client_type': 'parent_gui',
            }
        }

        self.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _onMessage(self, message):
        super()._onMessage(message)

        match message['type']:
            case 'init':
                self._handleInit(message)
            case _:
                self.gui.logger.warning(f"Unhandled message from child GUI {self.id}: {message}")

    # ------------------------------------------------------------------------------------------------------------------
    def _handleInit(self, message):
        self.gui.logger.debug(f"Handling init message from child GUI {self.id}")


# === PARENT ===========================================================================================================
@callback_definition
class ControlGUI_ParentGUI_Callbacks:
    disconnect: CallbackContainer
    message: CallbackContainer


class ControlGUI_Parent:

    def __init__(self, id, client: WebsocketServerClient):
        self.id = id
        self.client = client
        self.callbacks = ControlGUI_ParentGUI_Callbacks()

        self.client.callbacks.disconnected.register(self._onDisconnect)
        self.client.callbacks.message.register(self._onMessage)

    def send(self, message):
        self.client.send(message)

    def _onMessage(self, message):
        print(f"Received message from parent GUI: {message}")
        self.callbacks.message.call(message)

    def _onDisconnect(self):
        self.callbacks.disconnect.call()


# === GUI ==============================================================================================================
class GUI:
    id: str
    server: WebsocketServer
    client: WebsocketClient | None
    categories: dict[str, ControlGUI_Category]

    frontends: list
    child_guis: dict[str, ControlGUI_Child]
    parent_guis: dict[str, ControlGUI_Parent]

    # === INIT =========================================================================================================
    def __init__(self, id, host, ws_port=8099, run_js: bool = False, js_app_port=8400, options=None):

        self.id = id
        if options is None:
            options = {}

        default_options = {
            'color': [31 / 255, 32 / 255, 35 / 255, 1],
            'rows': 18,
            'columns': 50,
            'max_pages': 10,
            'logo_path': '',
            'name': None,
        }

        self.options = {**default_options, **options}

        if self.options['name'] is None:
            self.options['name'] = self.id

        self.categories = {}

        self.server = WebsocketServer(host=host, port=ws_port)
        self.server.callbacks.new_client.register(self._new_client_callback)
        self.server.callbacks.client_disconnected.register(self._client_disconnected_callback)
        self.server.callbacks.message.register(self._serverMessageCallback)
        self.server.logger.switchLoggingLevel('INFO', 'DEBUG')

        self.client = None

        self.logger = Logger(f'GUI: {self.id}', 'DEBUG')

        self.frontends = []
        self.child_guis = {}
        self.parent_guis = {}
        self.server.start()

        self.logger.info(f"Started GUI \"{self.id}\" on websocket {host}:{ws_port}")

        self.js_app_port = js_app_port
        self.js_process = None

        register_exit_callback(self.close)

        if run_js:
            self.runJSApp()

    # === PROPERTIES ===================================================================================================
    @property
    def uid(self):
        return f"{self.id}"

    # === METHODS ======================================================================================================
    def close(self):
        self.logger.info("Closing GUI")
        if self.js_process is not None:
            self.js_process.terminate()

    # ------------------------------------------------------------------------------------------------------------------
    def runJSApp(self):
        app_path = relativeToFullPath("../../")
        self.js_process = run_vite_app(app_path, host=self.server.host, port=self.js_app_port, env_vars={
            'WS_PORT': str(self.server.port),
            'WS_HOST': self.server.host,
        }, print_link=False, )

        self.logger.debug(f"Running JS app at http://{self.server.host}:{self.js_app_port}/")

    # ------------------------------------------------------------------------------------------------------------------
    def addCategory(self, category: ControlGUI_Category):
        if category.id in self.categories:
            raise ValueError(f"Category with id {category.id} already exists")
        category.parent = self
        self.categories[category.uid] = category

        return category
        # TODO: Send an add message

    # ------------------------------------------------------------------------------------------------------------------
    def removeCategory(self, category: ControlGUI_Category):
        if category.id not in self.categories:
            raise ValueError(f"Category with id {category.id} does not exist")
        del self.categories[category.uid]

        message = {
            'type': 'remove',
            'data': {
                'type': 'category',
                'parent': self.uid,
                'id': category.uid,
            }
        }

        self.broadcast(message)

    # ------------------------------------------------------------------------------------------------------------------
    def sendToFrontends(self, message):
        for frontend in self.frontends:
            self.server.sendToClient(frontend, message)

    # ------------------------------------------------------------------------------------------------------------------
    def sendToParents(self, message):
        for parent in self.parent_guis.values():
            parent.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def broadcast(self, message):
        self.sendToFrontends(message)
        self.sendToParents(message)

    # ------------------------------------------------------------------------------------------------------------------
    def connectToParent(self, parent_address, parent_port):
        """
        Connects to a parent GUI.
        """

        raise NotImplementedError("This is currently not supported.")

        # self.logger.debug(f"Connecting to parent GUI at {parent_address}:{parent_port}")
        # # 1. Check if we are already connected to this parent. TODO
        #
        # # 2. Check if the parent is reachable
        # result = pingAddress(parent_address, 5)
        # if not result:
        #     self.logger.warning(f"Parent at {parent_address} is not reachable")
        #     return
        # self.logger.debug(f"Parent at {parent_address} is reachable")
        #
        # # 3. Open up a websocket
        # self.client = WebsocketClient(parent_address, parent_port)
        # self.client.callbacks.connected.register(self._parent_client_connected)
        # self.client.callbacks.disconnected.register(self._parent_client_disconnected)
        # self.client.callbacks.message.register(self._parent_client_message)
        # self.client.logger.switchLoggingLevel('INFO', 'DEBUG')
        # self.client.connect()

    # ------------------------------------------------------------------------------------------------------------------
    # def _parent_client_connected(self, *args, **kwargs):
    #     # self.logger.info(f"Connected to parent GUI at {self.client.address}:{self.client.port}")
    #     self.logger.info(f"Connected to parent GUI at {self.client.address}:{self.client.port}")
    #     # Send a handshake to the parent
    #     message = {
    #         'type': 'handshake',
    #         'data': {
    #             'client_type': 'child_gui',
    #             'id': self.uid,
    #             'payload': self.getPayload(),
    #         }
    #     }
    #     self.client.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    # def _parent_client_disconnected(self, *args, **kwargs):
    #     self.logger.info(f"Parent GUI disconnected")
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # def _parent_client_message(self, client, message):
    #     ...

    # ------------------------------------------------------------------------------------------------------------------
    def connectToChild(self, id, child_address, child_port, category_path: str = ''):

        self.logger.debug(f"Connecting to child GUI at {child_address}:{child_port}")

        child = ChildGUI_WebsocketClient(id=id,
                                         address=child_address,
                                         port=child_port,
                                         client=WebsocketClient(child_address, child_port),
                                         category_path=category_path,
                                         gui=self)

        child.callbacks.connect.register(lambda *args, **kwargs:
                                         self.logger.info(f"Connected to child GUI at {child_address}:{child_port}"))

    # === PRIVATE METHODS ==============================================================================================
    def getElementByUID(self, uid: str):
        """
        Given a full UID, e.g.
           "myGui::category1/page1/button1"
        or even
           "/myGui::category1/page1/button1",
        this will strip slashes and the gui_id:: prefix,
        then split off the category id and delegate to
        that category’s getObjectByPath.
        """
        if not uid:
            return None

        # 1) drop any leading slash
        trimmed = uid.lstrip("/")

        # 3) now trimmed is "categoryID[/rest]"
        category_id, remainder = split_path(trimmed)
        if not category_id:
            return None

        # 4) categories are stored under key == "<gui_id>::<category_id>"
        category_key = f"{category_id}"
        category = self.categories.get(category_key)
        if category is None:
            return None

        if not remainder:
            return category

        return category.getObjectByPath(remainder)

    # ------------------------------------------------------------------------------------------------------------------
    def getPayload(self):
        payload = {
            'type': 'gui',
            'id': self.uid,
            'name': self.options['name'],
            'options': self.options,
            'categories': {k: v.getPayload() for k, v in self.categories.items()}
        }
        return payload

    # ------------------------------------------------------------------------------------------------------------------
    def _initializeFrontend(self, frontend):
        # Send Initialize Message
        message = {
            'type': 'init',
            'configuration': self.getPayload(),
        }
        self.server.sendToClient(frontend, message)

    # ------------------------------------------------------------------------------------------------------------------
    def _initializeParent(self, parent_client):

        parent = ControlGUI_Parent(id='', client=parent_client)
        self.parent_guis[parent_client] = parent
        message = {
            'type': 'init',
            'configuration': self.getPayload(),
        }
        parent.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _new_client_callback(self, client):
        self.logger.debug(f"New client connected: {client}")

    # ------------------------------------------------------------------------------------------------------------------
    def _client_disconnected_callback(self, client: WebsocketServerClient):
        self.logger.debug(f"Client disconnected: {client.address}:{client.port}")

        if client in self.frontends:
            self.frontends.remove(client)
            self.logger.info(f"Frontend disconnected: {client.address}:{client.port} ({len(self.frontends)})")
        elif client in self.child_guis:
            self.logger.warning(f"TODO: Child GUI disconnected: {client.address}:{client.port}")

    # ------------------------------------------------------------------------------------------------------------------
    def _serverMessageCallback(self, client, message, *args, **kwargs):
        self.logger.debug(f"Message received: {message}")

        match message['type']:
            case 'handshake':
                self._handleHandshakeMessage(client, message)
            case 'event':
                self._handleEventMessage(message)
            case _:
                self.logger.debug(f"Unknown message type: {message['type']}")

    # ------------------------------------------------------------------------------------------------------------------
    def _handleHandshakeMessage(self, client: WebsocketServerClient, message):
        self.logger.debug(f"Received handshake message from {client}: {message}")

        data = message.get('data')
        if data is None:
            self.logger.warning(f"Handshake message from {client} did not contain data")
            return

        match data['client_type']:
            case 'frontend':
                self.frontends.append(client)
                self._initializeFrontend(client)
                self.logger.info(f"New frontend connected: {client.address}:{client.port} ({len(self.frontends)})")
            case 'child_gui':
                self.logger.info(f"New child GUI connected: {client.address}:{client.port}")
                pass
            case 'parent_gui':
                self.logger.info(f"New parent GUI connected: {client.address}:{client.port}")
                self._initializeParent(client)
            case _:
                self.logger.warning(f"Unknown client type: {data['client_type']}")
                return

    # ------------------------------------------------------------------------------------------------------------------
    def _handleEventMessage(self, message):

        # Check if the message has an ID
        if not message.get('id'):
            self.logger.warning(f"Event message received without id: {message}")
            return

        # Check if the message belongs to us, or to a child
        # TODO

        # Try to find the element with the given ID
        element = self.getElementByUID(message['id'])
        if element is None:
            self.logger.warning(f"Event message received for nonexistent element {message['id']}: {message}")
            return
        element.onMessage(message['data'])
