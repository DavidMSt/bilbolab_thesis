from typing import Any

from extensions.gui.src.lib.objects.objects import Widget


class DigitalNumberWidget(Widget):

    type: str = "digital_number"
    value: int | float

    def __init__(self, widget_id: str,
                 value: int | float = 0,
                 min_value: int | float = 0,
                 max_value: int | float = 100,
                 increment: int | float = 1,
                 warn_on_out_of_bounds: bool = True,
                 **kwargs):
        super().__init__(widget_id)

        # Default configuration

        default_config = {
            'title': None,
            'title_position': 'left',  # 'left' or 'top'
            'show_unused_digits': True,
            'color': [0.5, 0.5, 0.5],
            'text_color': [1.0, 1.0, 1.0],
            'value_color': [1.0, 1.0, 1.0],
            'color_ranges': [],
        }

        self.config = {**default_config, **kwargs}

        self.warn_on_out_of_bounds = warn_on_out_of_bounds

        self.min_value = min_value
        self.max_value = max_value
        self.increment = increment

        self.value = value

        if self.config['title'] is None:
            self.config['title'] = widget_id

    # ------------------------------------------------------------------------------------------------------
    @property
    def value(self) -> int | float:
        return self._value

    @value.setter
    def value(self, new_value: int | float):
        if not isinstance(new_value, (int, float)):
            raise ValueError("Value must be an integer or float.")

        if (new_value < self.min_value or new_value > self.max_value) and self.warn_on_out_of_bounds:
            self.logger.warning(f"Value {new_value} is out of bounds ({self.min_value}, {self.max_value})")
        self._value = new_value
        self._sendValueToFrontend(new_value)

    # ------------------------------------------------------------------------------------------------------
    def _sendValueToFrontend(self, value):
        self.sendUpdate(value)

    # ------------------------------------------------------------------------------------------------------
    def getConfiguration(self) -> dict:
        config = {
            'value': self.value,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'increment': self.increment,
            **self.config
        }
        return config

    def handleEvent(self, message, sender=None) -> Any:
        self.logger.error(f"DigitalNumberWidget does not support handleEvent: {message}")

    def init(self, *args, **kwargs):
        pass