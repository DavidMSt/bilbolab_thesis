import dataclasses
import time
from typing import Iterable, List, Sequence, Tuple, Union

from core.utils.exit import register_exit_callback
from core.utils.websockets import WebsocketClient

WEBSOCKET_PORT = 7777
DEFAULT_HOST = "bilbolab-testbed.local"

# Local defaults (also carried in the config we can push to the Pi)
HEIGHT_OFFSET_TO_PIXEL_0 = 0.0   # physical height (same units as 'height') at pixel 0
DISTANCE_BETWEEN_PIXELS = 17    # distance (same units) between successive pixels
DEFAULT_HEIGHT_COLOR = (0, 0, 255)
DEFAULT_NUM_PIXELS = 30


@dataclasses.dataclass
class HeightIndicatorConfig:
    height_offset_to_pixel_0: float = HEIGHT_OFFSET_TO_PIXEL_0
    distance_between_pixels: float = DISTANCE_BETWEEN_PIXELS
    height_color: Tuple[int, int, int] = DEFAULT_HEIGHT_COLOR
    num_pixels: int = DEFAULT_NUM_PIXELS
    websocket_port: int = WEBSOCKET_PORT


def _as_color_triplet(color: Union[Sequence[int], Tuple[int, int, int]]) -> Tuple[int, int, int]:
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


class HeightIndicator:
    client: WebsocketClient
    connected: bool = False

    # === INIT =========================================================================================================
    def __init__(self, host: str = DEFAULT_HOST, port: int = WEBSOCKET_PORT):
        self.client = WebsocketClient(host, port)
        self.client.callbacks.connected.register(self._websocket_connected_callback)
        self.client.callbacks.disconnected.register(self._websocket_disconnected_callback)

        register_exit_callback(self.close, priority=100)

    # === METHODS ======================================================================================================
    def start(self):
        self.client.connect()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        try:
            self.clear()
            time.sleep(2)
        finally:
            self.client.close()

    # ------------------------------------------------------------------------------------------------------------------
    def setConfig(self, config: HeightIndicatorConfig):
        """
        Send runtime configuration to the Pi.
        You can change mapping parameters, num_pixels, and height_color.
        """
        config_dict = dataclasses.asdict(config)
        self.send({'type': 'set_config', 'data': config_dict})

    # ------------------------------------------------------------------------------------------------------------------
    def clear(self):
        """Turn everything off."""
        self.send({'type': 'clear'})

    # ------------------------------------------------------------------------------------------------------------------
    def setPixel(self, index: int, color: Union[Sequence[int], Tuple[int, int, int]]):
        """Set a single pixel to a color."""
        r, g, b = _as_color_triplet(color)
        self.send({'type': 'set_pixel', 'data': {'index': int(index), 'color': [r, g, b]}})

    # ------------------------------------------------------------------------------------------------------------------
    def fill(self, color: Union[Sequence[int], Tuple[int, int, int]]):
        """Fill the entire strip with a color."""
        r, g, b = _as_color_triplet(color)
        self.send({'type': 'set_all', 'data': {'color': [r, g, b]}})

    # ------------------------------------------------------------------------------------------------------------------
    def blink(self, color: Union[Sequence[int], Tuple[int, int, int]], num_blinks: int = 1, time_ms: int = 250):
        """Flash the whole strip a number of times."""
        r, g, b = _as_color_triplet(color)
        self.send({'type': 'blink', 'data': {'color': [r, g, b], 'times': int(num_blinks), 'time_ms': int(time_ms)}})

    # ------------------------------------------------------------------------------------------------------------------
    def blinkRed(self):
        self.blink((255, 0, 0), 3, 150)

    # ------------------------------------------------------------------------------------------------------------------
    def blinkGreen(self):
        self.blink((0, 255, 0), 3, 150)
    # ------------------------------------------------------------------------------------------------------------------
    def setHeight(self, height: float):
        """
        Ask the Pi to compute and render the nearest pixel for a given physical height
        using its current mapping config.
        """
        self.send({'type': 'set_height', 'data': {'height': float(height)}})

    # ------------------------------------------------------------------------------------------------------------------
    def setHeightPixels(self, height_pixels: Iterable[int]):
        """
        Light multiple pixel indices at once using the Pi's configured height color.
        """
        indices: List[int] = [int(i) for i in height_pixels]
        self.send({'type': 'set_height_pixels', 'data': {'indices': indices}})

    # === PRIVATE METHODS ==============================================================================================
    def send(self, data: dict):
        if self.connected:
            self.client.send(data)

    # ------------------------------------------------------------------------------------------------------------------
    def _websocket_connected_callback(self, *args, **kwargs):
        self.connected = True

    # ------------------------------------------------------------------------------------------------------------------
    def _websocket_disconnected_callback(self, *args, **kwargs):
        self.connected = False


if __name__ == '__main__':
    height_indicator = HeightIndicator()
    height_indicator.start()

    while not height_indicator.connected:
        time.sleep(1)
    # Tiny demo script (safe to remove)
    height_indicator.clear()
    time.sleep(0.5)
    height_indicator.setConfig(HeightIndicatorConfig())

    time.sleep(1)

    height_indicator.setHeight(200)

    time.sleep(3)

    while True:
        height_indicator.blinkRed()
        time.sleep(3)
        height_indicator.blinkGreen()
        time.sleep(3)