import dataclasses
from typing import Sequence, Iterable

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from core.utils.uuid_utils import generate_uuid


# === CUSTOM PLOT CLASS ================================================================================================
@dataclasses.dataclass
class PlotConfig:
    """Global plot / figure-level configuration."""
    size: tuple[float, float] = (6.4, 4.8)
    dpi: int = 100
    facecolor: str | Sequence[float] | None = 'white'
    tight_layout: bool = True

    # Global style
    use_latex: bool = False
    font_family: str = "sans-serif"
    font_size: float = 10.0

    # Save config
    save_dpi: int | None = None
    save_transparent: bool = False


@dataclasses.dataclass
class AxisConfig:
    """Axis-level configuration (titles, labels, ticks, grid, legend)."""
    facecolor: str | Sequence[float] | None = 'white'

    # Titles
    title: str | None = None
    title_font_size: float | None = None
    title_color: str | Sequence[float] | None = 'black'

    # Labels
    xlabel: str | None = None
    ylabel: str | None = None
    label_font_size: float | None = None
    label_color: str | Sequence[float] | None = 'black'

    # Ticks
    tick_font_size: float | None = None
    xtick_rotation: float = 0.0
    ytick_rotation: float = 0.0
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    xticklabels: list[str] | None = None
    yticklabels: list[str] | None = None

    # Limits
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None

    # Grid
    grid: bool = True
    grid_alpha: float = 0.8
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.5
    grid_color: str | Sequence[float] | None = dataclasses.field(default_factory=lambda: [0.2, 0.2, 0.2])

    # Legend
    legend: bool = True
    legend_loc: str = "upper right"
    legend_font_size: float | None = None
    legend_marker_scale: float = 1.0
    legend_line_width: float = 1.0
    legend_font_color: str | Sequence[float] | None = 'black'


@dataclasses.dataclass
class SeriesConfig:
    """Defaults for lines."""
    color: str | Sequence[float] | None = 'blue'
    linewidth: float = 1.5
    linestyle: str = "-"
    alpha: float = 1.0
    label: str | None = None
    visible: bool = True

    # Markers
    marker: str | None = None
    marker_size: float = 6.0
    marker_facecolor: str | Sequence[float] | None = None
    marker_edgecolor: str | Sequence[float] | None = None

    # Special
    stairs: bool = False  # if True -> use ax.step


@dataclasses.dataclass
class LineConfig:
    color: str | Sequence[float] | None = 'black'
    linewidth: float = 1.5
    linestyle: str = "-"
    alpha: float = 1.0


class Line:
    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    config: LineConfig

    def __init__(self,
                 start: tuple[float, float],
                 end: tuple[float, float],
                 id: str | None = None,
                 config: LineConfig | None = None,
                 **overrides):

        if id is None:
            id = generate_uuid()
        self.id = id
        self.start = start
        self.end = end
        if config is None:
            config = LineConfig()
        self.config = config
        if overrides:
            self.config = dataclasses.replace(self.config, **overrides)


# === WRAPPERS =========================================================================================================
class Series:
    id: str
    x_data: list[float]
    y_data: list[float]
    line: Line2D
    config: SeriesConfig

    # ------------------------------------------------------------------------------------------------------------------
    def update(self, **kwargs) -> None:
        if "config" in kwargs:
            self.config = kwargs.pop("config")
        if kwargs:
            self.line.set(**kwargs)


# === AXIS =============================================================================================================
class Axis:
    id: str
    ax: Axes
    config: AxisConfig
    series: dict[str, Series]

    # ------------------------------------------------------------------------------------------------------------------
    def add_series(self, series: Series) -> Series:
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def plot(self,
             x_data: Iterable[float | int],
             y_data: Iterable[float | int],
             line_config: SeriesConfig | None = None,
             **overrides):

        if line_config is None:
            line_config = SeriesConfig()

        if overrides:
            line_config = dataclasses.replace(line_config, **overrides)


# === PLOT =============================================================================================================
class Plot:
    config: PlotConfig
    figure: Figure
    axes: dict[str, Axis]

    def __init__(self,
                 rows: int = 1,
                 columns: int = 1,
                 config: PlotConfig | None = None,
                 **overrides):

        if config is None:
            config = PlotConfig()

        if overrides:
            config = dataclasses.replace(config, **overrides)

    # ------------------------------------------------------------------------------------------------------------------
    def add_subplot(self, row, column, axis: Axis):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def save(self, format: str, filename: str, **kwargs):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def save_as_pgfplot(self, filename: str, **kwargs):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def show_as_pdf(self):
        ...


# === SPECIAL PLOTS ====================================================================================================
class UpdatablePlot:
    ...


class RealTimePlot:
    ...


class PDF_Plot:
    ...

# === SERIALIZATION ====================================================================================================
