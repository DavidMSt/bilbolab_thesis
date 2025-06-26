from __future__ import annotations

import abc
import copy
import math
import random
import time
import uuid
from abc import abstractmethod

from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.colors import rgb_to_hex
from core.utils.events import ConditionEvent
from core.utils.exit import register_exit_callback
from core.utils.files import relativeToFullPath
from core.utils.js.vite import run_vite_app
from core.utils.logging_utils import Logger, setLoggerLevel
from core.utils.network.network import pingAddress, getHostIP
from core.utils.websockets.websockets import WebsocketServer, WebsocketClient, WebsocketServerClient
from extensions.control_gui.src.lib.plot.jsplot import JSPlotTimeSeries
from extensions.control_gui.src.lib.plot.plot_widget import PlotWidget
from extensions.control_gui.src.lib.widgets.buttons import Button
from extensions.control_gui.src.lib.objects import GUI_Object_Group, GUI_Object, GUI_Object_Instance
from extensions.control_gui.src.lib.utilities import check_for_spaces, split_path, strip_id, addIdPrefix
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
class Category_Headbar:
    ...

    def __init__(self):
        ...

    def getPayload(self):
        payload = {}
        return payload


# ----------------------------------------------------------------------------------------------------------------------
@callback_definition
class Category_Callbacks:
    update: CallbackContainer
    add: CallbackContainer
    remove: CallbackContainer


# ----------------------------------------------------------------------------------------------------------------------
class Category:
    id: str
    pages: dict[str, Page]
    categories: dict[str, Category]

    name: str
    icon: str
    headbar: Category_Headbar

    configuration: dict

    parent: GUI | Category | None

    # === INIT =========================================================================================================
    def __init__(self, id: str, name: str = None, **kwargs):

        if check_for_spaces(id):
            raise ValueError(f"Category id '{id}' contains spaces")
        if '/' in id:
            raise ValueError(f"Category id '{id}' contains slashes")
        # if ":" in id:
        #     raise ValueError(f"Category id '{id}' contains colons")

        id = f"{id}"

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

        self.callbacks = Category_Callbacks()
        self.headbar = Category_Headbar()

        self.logger = Logger(f"Category {self.id}", 'DEBUG')

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def uid(self):
        if isinstance(self.parent, GUI):
            return f"{self.parent.id}/{self.id}"
        elif isinstance(self.parent, Category):
            return f"{self.parent.uid}/{self.id}"
        else:
            return self.id

    # ------------------------------------------------------------------------------------------------------------------
    def getGUI(self):
        if isinstance(self.parent, GUI):
            return self.parent
        elif isinstance(self.parent, Category):
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
            return gui.getObjectByUID(uid)
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def addPage(self, page: Page, position=None):

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
    def removePage(self, page: Page):
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
    def addCategory(self, category: Category):
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
    def removeCategory(self, category: Category):
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
class Page_Callbacks:
    update: CallbackContainer
    add: CallbackContainer
    remove: CallbackContainer


# ----------------------------------------------------------------------------------------------------------------------
class Page:
    """
    Represents a page in the Control GUI that holds GUI_Object instances
    in a fixed grid layout. Tracks occupied cells and supports manual
    or automatic placement of objects.
    """
    id: str
    objects: dict[str, dict]
    category: Category | None
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

        default_config = {
            'color': None,
            'pageColor': [60, 60, 60, 1],
            'grid_size': (18, 50),  # (rows, columns)
            'text_color': [1,1,1]
        }

        self.config = {**default_config, **kwargs}

        self.id = f"{id}"
        self.icon = icon
        self.name = name if name is not None else id

        # Grid dimensions
        self._rows, self._cols = self.config['grid_size']
        # Occupancy grid: False = free, True = occupied
        self._occupied = [[False for _ in range(self._cols)] for _ in range(self._rows)]

        self.objects = {}
        self.category = None
        self.callbacks = Page_Callbacks()
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
            obj = info["instance"]
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
            return gui.getObjectByUID(uid)
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
    def addObject(self, obj: GUI_Object, row=None, column=None, width=2, height=2) -> GUI_Object_Instance:
        """
        Adds an object to the page at a given grid position.
        If the row or column is None, we automatically find the first available
        position for the object's size.
        """

        instance = obj.newInstance()

        if instance.id in self.objects:
            obj.removeInstance(instance)
            raise ValueError(f"Object with id {obj.id} already exists on page {self.id}")

        # Determine placement
        if row is None or column is None:
            row, column = self._placeObject(row, column, width, height)
        else:
            self._checkSpace(row, column, width, height)

        # Mark cells occupied
        self._markSpace(row, column, width, height)

        # Store object placement
        self.objects[instance.id] = {
            'instance': instance,
            'row': row,
            'column': column,
            'width': width,
            'height': height,
        }
        instance.parent = self

        message = {
            'type': 'add',
            'data': {
                'type': 'object',
                'parent': self.uid,
                'id': instance.uid,
                'config': {
                    'row': row,
                    'column': column,
                    'width': width,
                    'height': height,
                    **instance.getPayload(),
                }
            }
        }

        self.sendMessage(message)

        return instance

    # ------------------------------------------------------------------------------------------------------------------
    def removeObject(self, obj: GUI_Object_Instance | GUI_Object):
        if obj.id not in self.objects:
            raise ValueError(f"Object with id {obj.id} does not exist on page {self.id}")

        if isinstance(obj, GUI_Object_Instance):
            instance = obj
            instance.obj.removeInstance(instance)  # Remove the instance from the parent object
        elif isinstance(obj, GUI_Object):
            instance = self.objects[obj.id]['instance']
            obj.removeInstance(instance)
        else:
            raise ValueError(f"Object {obj} is neither an instance nor a parent object")

        message = {
            'type': 'remove',
            'data': {
                'type': 'object',
                'parent': self.uid,
                'id': instance.uid,
            }
        }
        self.sendMessage(message)

        instance.parent = None
        del self.objects[instance.id]

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
            obj = info['instance']
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


class Child_Category(Category):
    child: Child

    def __init__(self, id, name, child, **kwargs):
        super(Child_Category, self).__init__(id, name, **kwargs)
        self.child = child

    def getPayload(self):
        # ask the child GUI for its full GUI payload (includes export + categories)
        gui_payload = self.child.requestPayloadForCategory(self.id)
        if gui_payload is None:
            self.child.gui.logger.warning(
                f"Could not get payload for GUI {self.child.id}"
            )
            return super().getPayload()

        # extract export subtree (should be a dict)
        export_payload = gui_payload.get('export', {})
        # extract all its other categories
        gui_categories = gui_payload.get('categories', {})

        # deepcopy so we don’t mutate the child’s own data
        export_copy = copy.deepcopy(export_payload)
        # merge in the child's categories under the export node
        export_copy.setdefault('categories', {}).update(copy.deepcopy(gui_categories))

        # # now prefix every id under the mount point
        # prefix = self.child.path_in_gui.rstrip('/') + '/'
        #
        # def _adjust(node: dict):
        #     # prefix this node’s id
        #     if 'id' in node and isinstance(node['id'], str):
        #         node['id'] = prefix + node['id'].lstrip('/')
        #     # keep config.id in sync
        #     cfg = node.get('config')
        #     if isinstance(cfg, dict) and 'id' in cfg:
        #         cfg['id'] = node['id']
        #     # recurse into pages / categories / objects
        #     for sub in node.get('pages', {}).values():
        #         _adjust(sub)
        #     for sub in node.get('categories', {}).values():
        #         _adjust(sub)
        #     for sub in node.get('objects', {}).values():
        #         _adjust(sub)
        #
        # _adjust(export_copy)
        return export_copy


@callback_definition
class Child_Callbacks:
    connect: CallbackContainer
    disconnect: CallbackContainer
    message: CallbackContainer


class Child:
    gui: GUI
    category: Child_Category | None

    id: str
    name: str | None
    path_in_gui: str
    request_event: ConditionEvent

    child_object_id: str | None

    def __init__(self,
                 address,
                 port,
                 parent_object_uid,
                 name: str = None,
                 client: WebsocketClient = None,
                 gui: GUI = None,
                 child_object_id=None,
                 ):

        self.address = address
        self.port = port
        self.name = name

        self.client = client
        self.gui = gui

        self.callbacks = Child_Callbacks()

        self.client.events.message.on(self._onMessage)
        self.client.callbacks.connected.register(self._onConnect)
        self.client.callbacks.disconnected.register(self._onDisconnect)
        # self.client.callbacks.message.register(self._onMessage)

        self.client.connect()

        self.category = None
        self.path_in_gui = parent_object_uid

        self.request_event = ConditionEvent(flags=[("request_id", str)])

    # ------------------------------------------------------------------------------------------------------------------
    def send(self, message):
        self.client.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _onConnect(self, *args, **kwargs):
        self.callbacks.connect.call()
        message = {
            'type': 'handshake',
            'data': {
                'client_type': 'parent_gui',
            }
        }

        self.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _onDisconnect(self, *args, **kwargs):
        self.gui.logger.warning(f"Child GUI {self.address}:{self.port} disconnected!")

        # Remove the category
        if self.category is not None:
            self.gui.logger.debug(f"Removing category {self.category.uid} from GUI {self.gui.uid}")
            parent = self.gui.getObjectByUID(self.path_in_gui)
            parent.removeCategory(self.category)

        self.callbacks.disconnect.call()

    # ------------------------------------------------------------------------------------------------------------------
    def requestPayloadForCategory(self, category_id):
        # 1) clear any previous answer

        # 2) send the request to the child GUI
        request_id = str(uuid.uuid4())
        message = {
            'type': 'request',
            'request_id': request_id,
            'data': {
                'type': 'category_payload',
                'id': category_id,
            }
        }
        self.send(message)

        # 3) wait for the answer
        success = self.request_event.wait(flags={'request_id': request_id}, timeout=5)
        if not success:
            self.gui.logger.warning(
                f"Timeout waiting for payload for category {category_id} from child GUI {self.id}"
            )
            return None

        payload = self.request_event.get_data()

        # 4) now recursively prefix every 'id' (and matching config['id']) in the payload
        prefix = self.path_in_gui.rstrip('/') + '/'

        addIdPrefix(payload, prefix, ['id', 'parent'])

        # def _adjust(node: dict):
        #     # prefix node['id']
        #     if 'id' in node and isinstance(node['id'], str):
        #         node['id'] = prefix + node['id'].lstrip('/')
        #
        #     # keep config.id in sync
        #     cfg = node.get('config')
        #     if isinstance(cfg, dict) and 'id' in cfg:
        #         cfg['id'] = node['id']
        #
        #     # dive into pages
        #     for page in node.get('pages', {}).values():
        #         _adjust(page)
        #
        #     # dive into sub‐categories
        #     for cat in node.get('categories', {}).values():
        #         _adjust(cat)
        #
        #     # dive into objects on pages
        #     for obj in node.get('objects', {}).values():
        #         _adjust(obj)
        #
        # _adjust(payload)
        return payload

    # ------------------------------------------------------------------------------------------------------------------
    def _onMessage(self, message):
        self.callbacks.message.call(message)
        match message['type']:
            case 'init':
                self._handleInit(message)
            case 'answer':
                self._handleAnswer(message)
            case 'update':
                self._handleUpdate(message)
            case 'widget_message':
                self._handleWidgetMessage(message)
            case 'add':
                self._handleAdd(message)
            case 'remove':
                self._handleRemove(message)
            case _:
                self.gui.logger.warning(f"Unhandled message from child GUI {self.id}: {message}")

    # ------------------------------------------------------------------------------------------------------------------
    def _handleInit(self, message):
        # Extract the data from the message
        data = message.get('data')

        if not data:
            self.gui.logger.warning(f"Init message from child GUI {self.address}:{self.port} has no data")
            return

        # Get the GUIs ID:
        self.id = data.get('id')
        if not self.id:
            self.gui.logger.warning(f"Init message from child GUI {self.address}:{self.port} has no id")
            return

        # If the name is currently unset, use the ID:
        if self.name is None:
            self.name = self.id

        # Now let's add a category for the child
        self.category = Child_Category(id=self.id,
                                       name=self.name,
                                       child=self,
                                       icon='🌐')

        # Get the intended parent category
        parent_object = self.gui.getObjectByUID(self.path_in_gui)
        if parent_object is None:
            self.gui.logger.warning(
                f"Could not find parent category for child GUI {self.id} with path {self.path_in_gui}")
            return

        # Check if the parent is either a category or the gui itself
        if not isinstance(parent_object, Category) and not isinstance(parent_object, GUI):
            self.gui.logger.warning(f"Parent object for child GUI {self.id} is not a category or the gui itself")
            return

        # Add the child to the parent
        parent_object.addCategory(self.category)
        self.callbacks.connect.call()

    # ------------------------------------------------------------------------------------------------------------------
    def _handleAnswer(self, message):
        self.request_event.set(resource=message['data'], flags={'request_id': message['request_id']})

    # ------------------------------------------------------------------------------------------------------------------
    def _handleUpdate(self, message):
        message['id'] = self.path_in_gui + '/' + message['id']
        message['data']['id'] = self.path_in_gui + '/' + message['data']['id']

        self.gui.broadcast(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _handleWidgetMessage(self, message):
        # Forward the message to the parent GUI
        message['id'] = self.path_in_gui + '/' + message['id']
        self.gui.broadcast(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _handleAdd(self, message):
        self.gui.logger.important(f"Handling add message from child GUI {self.id}: {message}")

        addIdPrefix(node=message,
                    prefix=self.path_in_gui.rstrip('/') + '/',
                    field_names=['id', 'parent'])

        self.gui.logger.debug(f"Broadcasting add message: {message}")
        self.gui.broadcast(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _handleRemove(self, message):
        addIdPrefix(node=message,
                    prefix=self.path_in_gui.rstrip('/') + '/',
                    field_names=['id', 'parent'])
        self.gui.broadcast(message)


# === PARENT ===========================================================================================================
@callback_definition
class Parent_Callbacks:
    disconnect: CallbackContainer
    message: CallbackContainer


class Parent:

    def __init__(self, id, gui, client: WebsocketServerClient):
        self.id = id
        self.gui = gui
        self.client = client
        self.callbacks = Parent_Callbacks()

        self.client.callbacks.disconnected.register(self._onDisconnect)
        self.client.callbacks.message.register(self._onMessage)

    def send(self, message):
        self.client.send(message)

    def _onMessage(self, message):

        match message.get('type'):
            case 'request':
                self._handleRequest(message)

        self.callbacks.message.call(message)

    def _onDisconnect(self):
        self.callbacks.disconnect.call()

    def _handleRequest(self, message):
        self.gui.logger.debug(f"Handling request message from parent GUI {self.id}: {message}")

        data = message.get('data')

        if data:
            match data.get('type'):
                case 'category_payload':
                    obj = self.gui.getObjectByUID(data.get('id'))
                    if obj is None:
                        self.gui.logger.warning(f"Could not find object with UID {data.get('id')} for request")
                        return
                    payload = obj.getPayload() if isinstance(obj, Category | GUI) else None

                    if payload is None:
                        self.gui.logger.warning(f"Object with UID {data.get('id')} is not a category or GUI")
                        return

                    response = {
                        'type': 'answer',
                        'request_id': message.get('request_id'),
                        'data': payload,
                    }
                    self.send(response)

                case _:
                    self.gui.logger.warning(f"Unhandled request type {data.get('type')} in message {message}")


# === GUI ==============================================================================================================
class GUI:
    id: str
    server: WebsocketServer
    client: WebsocketClient | None
    categories: dict[str, Category]

    frontends: list
    child_guis: dict[str, Child]
    parent_guis: dict[str, Parent]

    export_category: Category
    export_page: Page

    # === INIT =========================================================================================================
    def __init__(self, id, host, ws_port=8099, run_js: bool = False, js_app_port=8400, options=None):

        self.id = self._prepareID(id)
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

        self.export_category, self.export_page = self._prepareExportCategory()

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

        self.request_event = ConditionEvent(flags=[("request_id", str)])

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
        self.server.stop()
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
    def addCategory(self, category: Category):
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
        self.broadcast(message)
        return category

    # ------------------------------------------------------------------------------------------------------------------
    def removeCategory(self, category: Category | str):

        if isinstance(category, str):
            if category in self.categories:
                category = self.categories[category]
            else:
                category = self.getObjectByUID(category)
            if category is None or not isinstance(category, Category):
                self.logger.warning(f"Category with id {category} does not exist")
                return

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
    def sendRequest(self, client, data):
        request_id = str(uuid.uuid4())
        message = {
            'type': 'request',
            'id': request_id,
            'data': data
        }
        client.send(message)
        return request_id

    # ------------------------------------------------------------------------------------------------------------------
    def addChildGUI(self,
                    child_address,
                    child_port,
                    parent_object: Category | GUI | str = '',
                    name: str = None,
                    child_object_id=None):

        self.logger.debug(f"Connecting to child GUI at {child_address}:{child_port}")

        if isinstance(parent_object, GUI | Category):
            # If the parent is a GUI or Category, use its UID
            parent_object = parent_object.uid

        child = Child(name=name,
                      address=child_address,
                      port=child_port,
                      client=WebsocketClient(child_address, child_port),
                      parent_object_uid=parent_object,
                      gui=self,
                      child_object_id=child_object_id)

        self.child_guis[f"{child_address}:{child_port}"] = child
        child.callbacks.connect.register(lambda *args, **kwargs:
                                         self.logger.info(f"Connected to child GUI at {child_address}:{child_port}"))

    # === PRIVATE METHODS ==============================================================================================
    def getObjectByUID(self, uid: str):
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
        gui_id, remainder = split_path(trimmed)
        if not gui_id or gui_id != self.id:
            self.logger.warning(f"UID '{uid}' does not match this GUI's ID '{self.id}'")
            return None

        # If the remainder is empty, we are looking for the GUI itself
        if not remainder:
            return self
        # 4) Split off the category ID
        #    e.g. "category1/page1/button1" -> "category1",
        category_id, remainder = split_path(remainder)
        if not category_id:
            self.logger.warning(f"UID '{uid}' does not contain a valid category ID")
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
            'export': self.export_category.getPayload(),
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

        parent = Parent(id='', gui=self, client=parent_client)
        self.parent_guis[parent_client] = parent
        message = {
            'type': 'init',
            'data': {
                'id': self.id,
            },
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
            case 'request':
                # These are handled by the parent objects, so no need to do something here
                pass
            case _:
                self.logger.warning(f"Unknown message type: {message['type']}")

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
        """
        Handle an 'event' from a frontend. If the event's id belongs to
        a child GUI (i.e. starts with `<parent_mount>/<child.id>`), strip
        off the parent and child prefixes, re-prepend the child.id, and
        forward it; otherwise route it to a local element.
        """
        # 1) must have an id
        msg_id = message.get('id')
        if not msg_id:
            self.logger.warning(f"Event message received without id: {message}")
            return

        # Check if the message is meant for a child GUI
        child, child_obj_id = self._checkPathForChildGUI(msg_id)

        if child is not None:
            # child GUI matched → forward the event to the child

            # rebuild the event for the child
            child_event = {
                'type': 'event',
                'event': message.get('event'),
                'id': child_obj_id,
                'data': message.get('data'),
            }

            self.logger.debug(
                f"Forwarding event to child GUI {child.address}:{child.port}: {child_event}"
            )
            child.send(child_event)
            return

        # 3) no child matched → handle locally
        element = self.getObjectByUID(msg_id)
        if element is None:
            self.logger.warning(f"Event for unknown element {msg_id}: {message}")
            return
        if isinstance(element, GUI_Object_Instance):
            element.obj.onMessage(message.get('data'))
        else:
            element.onMessage(message.get('data'))

    # ------------------------------------------------------------------------------------------------------------------
    def _checkPathForChildGUI(self, path) -> (Child | None, str):
        for child in self.child_guis.values():
            parent_mount = child.path_in_gui.rstrip('/')  # e.g. ":myGui:/categoryX"
            full_mount = f"{parent_mount}/{child.id}"  # e.g. ":myGui:/categoryX/childGuiID"

            # Does the event target live at or under this full_mount?
            if path == full_mount or path.startswith(full_mount + '/'):
                # strip off the full_mount prefix
                suffix = path[len(full_mount):]  # e.g. "" or "/page1/button1"
                if suffix.startswith('/'):
                    suffix = suffix[1:]  # e.g. "page1/button1"

                # build the ID the child expects: always start with its own id
                child_obj_id = child.id + (f"/{suffix}" if suffix else "")
                return child, child_obj_id

        # No child GUI matched
        return None, ''

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _prepareID(gui_id: str) -> str:
        """
        Prepare an ID for use in the GUI by removing spaces and slashes.
        """
        if check_for_spaces(gui_id):
            raise ValueError(f"ID '{gui_id}' contains spaces")
        if '/' in gui_id:
            raise ValueError(f"ID '{gui_id}' contains slashes")
        if ':' in gui_id:
            raise ValueError(f"ID '{gui_id}' contains colons")

        gui_id = f":{gui_id}:"

        return gui_id

    def _prepareExportCategory(self):
        category = Category(id=self.id,
                            name=f"{self.id}_exp",
                            icon='🌐',
                            max_pages=1,
                            top_icon='🌐'
                            )

        export_page = Page(id=f"{self.id}_exp_page1",
                           name=f"{self.id}_exp_page1",
                           )

        category.addPage(export_page)

        return category, export_page
