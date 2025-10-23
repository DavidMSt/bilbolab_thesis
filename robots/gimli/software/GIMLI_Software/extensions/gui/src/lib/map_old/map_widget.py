import enum
from typing import Any

from core.utils.logging_utils import Logger
from extensions.gui.src.lib.objects.objects import Widget
from extensions.gui.src.lib.map.map import Map


class MapServerMode(enum.StrEnum):
    STANDALONE = 'standalone'
    EXTERNAL = 'external'


class MapWidget(Widget):
    type = 'map_new'
    map: Map

    # === INIT =========================================================================================================
    def __init__(self, widget_id: str, config=None, map_config=None, **kwargs):
        super().__init__(widget_id)

        default_map_config = {

        }

        default_config = {
            'host': 'localhost',
            'port': 8001,
        }

        self.config = {**default_config, **(config or {})}
        self.map_config = {**default_map_config, **(map_config or {})}
        self.logger = Logger(f"Map {self.id}", 'DEBUG')

        self.map = Map(host=self.config['host'], port=self.config['port'], options=self.map_config)

        # Update the map options with the kwargs provided during initialization
        self.map.options.update(kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    def getConfiguration(self) -> dict:
        config = {
            'type': self.type,
            'id': self.id,
            'config': self.config,
            'map_config': self.map_config,
        }

        return config

    # ------------------------------------------------------------------------------------------------------------------
    def handleEvent(self, message, sender=None) -> Any:
        self.logger.debug(f"Received message: {message}")

    # ------------------------------------------------------------------------------------------------------------------
    def updateConfig(self, *args, **kwargs):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    def init(self, *args, **kwargs):
        pass
