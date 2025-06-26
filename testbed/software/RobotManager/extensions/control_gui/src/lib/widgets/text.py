import dataclasses
from typing import Any

from core.utils.dict import ObservableDict
from extensions.control_gui.src.lib.objects import GUI_Object


class TextWidget(GUI_Object):

    type = 'text'

    def __init__(self, widget_id: str, text: str = "", **kwargs):
        super().__init__(widget_id)

        default_config = {
            'color': 'transparent',
            'text_color': [1,1,1],
            'title': None,
            'font_size': 10,
            'font_family': 'inherit',
            'vertical_alignment': 'center',  # 'center', 'top, 'bottom'
            'horizontal_alignment': 'center',  # 'left', 'right', 'center'
            'font_weight': 'normal',
            'font_style': 'normal',
        }

        self.text = text

        self.config = {**default_config, **kwargs}


    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, new_text):
        self._text = new_text
        self.function(
            function_name="setText",
            args=new_text
        )

    def getConfiguration(self) -> dict:
        config = {
            'text': self.text,
            **self.config
        }
        return config

    def onMessage(self, message) -> Any:
        pass

    def init(self, *args, **kwargs):
        pass


# ======================================================================================================================
@dataclasses.dataclass
class StatusWidgetElement:
    label: str = ''
    color: list = dataclasses.field(default_factory=lambda: [0.5, 0.5, 0.5])
    status: str = ''
    label_color: list = dataclasses.field(default_factory=lambda:[1,1,1])
    status_color: list = dataclasses.field(default_factory=lambda:[1,1,1])


class StatusWidget(GUI_Object):


    type = 'status'
    elements: dict[str, StatusWidgetElement]

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, widget_id, elements, **kwargs):
        super().__init__(widget_id)


        default_config = {
            'color': 'transparent',
            'text_color': [1,1,1],
            'title': None,
            'font_size': 10,
        }

        self.config = {**default_config, **kwargs}

        if elements is None:
            elements = {}

        self._elements = ObservableDict(elements, on_change=self._on_elements_changed)

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, value):
        self._elements = ObservableDict(value, on_change=self._on_elements_changed)
        self._on_elements_changed()

    # ------------------------------------------------------------------------------------------------------------------
    def _on_elements_changed(self):
       self.update()
    # ------------------------------------------------------------------------------------------------------------------
    def getConfiguration(self) -> dict:
        config = {
            'elements': {k: dataclasses.asdict(v) for k, v in self.elements.items()},
            **self.config
        }
        return config

    # ------------------------------------------------------------------------------------------------------------------
    def onMessage(self, message) -> Any:
        self.logger.warning(f"StatusWidget does not support onMessage: {message}")

    # ------------------------------------------------------------------------------------------------------------------
    def init(self, *args, **kwargs):
        pass

