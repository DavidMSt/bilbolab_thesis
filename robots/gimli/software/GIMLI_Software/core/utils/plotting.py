import base64
import io
import multiprocessing as mp
import platform
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Callable, Any, Optional, Iterable

import matplotlib
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import queue  # For queue.Empty exceptions

# === CUSTOM PACKAGES ==================================================================================================
from core.utils.callbacks import callback_definition, CallbackContainer


# ======================================================================================================================
@callback_definition
class ThreadPlotCallbacks:
    close: CallbackContainer


class ThreadPlotProcess(mp.Process):
    """
    GUI process: receives PNG bytes and displays them in a Matplotlib window.
    """

    def __init__(self, command_queue, event_queue, *, title: str | None, keep_aspect: bool,
                 show_axes: bool, bgcolor: str | None):
        super().__init__()
        self.command_queue = command_queue
        self.event_queue = event_queue
        self.title = title
        self.keep_aspect = keep_aspect
        self.show_axes = show_axes
        self.bgcolor = bgcolor

    def run(self):
        # Choose a GUI backend (Qt/Tk/etc.) since we are in a separate process that owns the window.
        choose_gui_backend()

        import matplotlib.pyplot as plt  # noqa: E402
        plt.ion()

        fig, ax = plt.subplots()
        if self.bgcolor is not None:
            try:
                fig.patch.set_facecolor(self.bgcolor)
            except Exception:
                pass
        if self.title:
            fig.suptitle(self.title)

        if not self.show_axes:
            ax.axis("off")

        ax.set_aspect("equal" if self.keep_aspect else "auto")

        img_artist = None

        # Close-event → notify parent.
        def handle_close(_event):
            try:
                self.event_queue.put({'event': 'close'})
            except Exception:
                pass

        fig.canvas.mpl_connect('close_event', handle_close)
        fig.show()
        fig.canvas.draw()

        # Main event loop
        while True:
            # Drain all available commands.
            while not self.command_queue.empty():
                cmd = self.command_queue.get()
                name = cmd.get('command')

                if name == 'set_image':
                    raw = cmd.get('png', b'')
                    if not raw:
                        continue
                    try:
                        # Decode with PIL → numpy → show
                        img = Image.open(io.BytesIO(raw))
                        arr = np.array(img)

                        if img_artist is None:
                            img_artist = ax.imshow(arr, origin='upper', interpolation='nearest')
                        else:
                            img_artist.set_data(arr)

                        # Fit view to image size
                        h, w = arr.shape[0], arr.shape[1]
                        ax.set_xlim(0, w)
                        ax.set_ylim(h, 0)

                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                    except Exception:
                        # Keep process alive even if one update fails
                        pass

                elif name == 'close':
                    plt.close(fig)
                    return

            plt.pause(0.05)


class ThreadPlot:
    """
    Show figures produced in *other threads* (Agg backend) inside a real Matplotlib window
    that lives in its own process. Use .setFigure(fig) from any thread.
    """

    def __init__(self,
                 *,
                 title: str | None = "Thread Plot",
                 keep_aspect: bool = True,
                 show_axes: bool = False,
                 bgcolor: str | None = "white"):
        self.command_queue = mp.Queue()
        self.event_queue = mp.Queue()
        self.proc = ThreadPlotProcess(
            self.command_queue,
            self.event_queue,
            title=title,
            keep_aspect=keep_aspect,
            show_axes=show_axes,
            bgcolor=bgcolor,
        )
        self.proc.daemon = True
        self.proc.start()

        self.callbacks = ThreadPlotCallbacks()
        self._event_listener_thread = threading.Thread(target=self._event_listener, daemon=True)
        self._event_listener_thread.start()

    # --- public API ---------------------------------------------------------------------------------

    def setFigure(self,
                  fig,
                  *,
                  dpi: int = 120,
                  transparent: bool = False,
                  bbox_inches: Optional[str] = "tight",
                  pad_inches: float = 0.05) -> None:
        """
        Serialize an Agg figure to PNG bytes and display/update it in the window.
        Safe to call from any thread.
        """
        raw = fig_to_png_bytes(
            fig,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        self.setImageBytes(raw)

    def setImageBytes(self, raw_png: bytes) -> None:
        """
        Push already-encoded PNG bytes to the window process.
        """
        self.command_queue.put({'command': 'set_image', 'png': raw_png})

    def setImageArray(self, arr: "np.ndarray") -> None:
        """
        Convenience: accept a numpy array (H×W, H×W×3, or H×W×4), encode to PNG, and display.
        """
        img = array_to_pil(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.setImageBytes(buf.getvalue())

    def close(self) -> None:
        """
        Close the GUI window/process.
        """
        if self.proc is not None:
            try:
                self.command_queue.put({'command': 'close'})
            except Exception:
                pass
            self.proc.join()

    # --- internals ----------------------------------------------------------------------------------

    def _event_listener(self):
        while True:
            try:
                event = self.event_queue.get(timeout=0.1)
                if event.get('event') == 'close':
                    self.on_close()
                    break
            except queue.Empty:
                continue

    def on_close(self):
        self.callbacks.close.call()


# ======================================================================================================================

# ======================================================================================================================
class UpdatablePlotProcess(mp.Process):
    def __init__(self, command_queue, event_queue, x_label="X", y_label="Y", xlim=None, ylim=None, common_x=None):
        super().__init__()
        self.command_queue = command_queue
        self.event_queue = event_queue  # Used to notify main process of events (e.g. close)
        self.x_label = x_label
        self.y_label = y_label
        self.xlim = xlim
        self.ylim = ylim
        self.common_x = common_x

    def run(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        if self.common_x is not None:
            self.ax.set_xticks(self.common_x)

        self.lines = []
        self.fig.show()
        self.fig.canvas.draw()
        self.ax.grid()

        # Register a close-event handler in the GUI process.
        def handle_close(event):
            self.event_queue.put({'event': 'close'})

        self.fig.canvas.mpl_connect('close_event', handle_close)

        # Main loop: poll for commands and process GUI events.
        while True:
            while not self.command_queue.empty():
                cmd = self.command_queue.get()
                if cmd['command'] == 'append':
                    x = cmd['x']
                    y = cmd['y']
                    color = cmd['color']
                    label = cmd['label']

                    # make sure all elements in color are between 0 and 1
                    if color is not None and isinstance(color, (list, tuple, np.ndarray)):
                        color = [max(c, 0) for c in color]
                        color = [min(c, 1) for c in color]
                    line, = self.ax.plot(x, y, color=color, label=label)
                    self.lines.append(line)
                    handles, labels = self.ax.get_legend_handles_labels()
                    if handles:
                        self.ax.legend()
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                elif cmd['command'] == 'remove':
                    if cmd.get('last', False):
                        if self.lines:
                            line = self.lines.pop()
                            line.remove()
                    elif 'index' in cmd:
                        idx = cmd['index']
                        if 0 <= idx < len(self.lines):
                            line = self.lines.pop(idx)
                            line.remove()
                    handles, labels = self.ax.get_legend_handles_labels()
                    if handles:
                        self.ax.legend()
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                elif cmd['command'] == 'close':
                    plt.close(self.fig)
                    return
            plt.pause(0.1)


# ======================================================================================================================
@callback_definition
class UpdatablePlotCallbacks:
    close: CallbackContainer


# ======================================================================================================================
class UpdatablePlot:
    """
    A proxy class that communicates with the plotting process for static/updatable plots.
    """

    def __init__(self, x_label="X", y_label="Y", xlim=None, ylim=None, common_x=None):
        self.command_queue = mp.Queue()
        self.event_queue = mp.Queue()  # Event queue for detecting GUI events from the process.
        self.plot_process = UpdatablePlotProcess(self.command_queue, self.event_queue,
                                                 x_label, y_label, xlim, ylim, common_x)
        self.plot_process.start()
        self.callbacks = UpdatablePlotCallbacks()

        # Start an event listener thread in the main process.
        self._event_listener_thread = threading.Thread(target=self._event_listener, daemon=True)
        self._event_listener_thread.start()

    def _event_listener(self):
        while True:
            try:
                event = self.event_queue.get(timeout=0.1)
                if event.get('event') == 'close':
                    self.on_close()
                    break
            except queue.Empty:
                continue

    def on_close(self):
        self.callbacks.close.call()

    def appendPlot(self, x, y, color=None, label=None):
        self.command_queue.put({
            'command': 'append',
            'x': x,
            'y': y,
            'color': color,
            'label': label
        })

    def removePlot(self, index=None, last=False):
        if last:
            self.command_queue.put({'command': 'remove', 'last': True})
        elif index is not None:
            self.command_queue.put({'command': 'remove', 'index': index})
        else:
            print("Please provide an index or set last=True to remove a plot.")

    def close(self, *args, **kwargs):
        self.command_queue.put({'command': 'close'})
        self.plot_process.join()


# ======================================================================================================================
@callback_definition
class RealTimePlotCallbacks:
    close: CallbackContainer


# NEW: Separate process class for the RealTimePlot.
class RealTimePlotProcess(mp.Process):
    def __init__(self, data_queue, control_queue, event_queue,
                 window_length, signals_info, value_format, title):
        super().__init__()
        self.data_queue = data_queue
        self.control_queue = control_queue
        self.event_queue = event_queue
        self.window_length = window_length
        self.signals_info = signals_info
        self.value_format = value_format
        self.title = title
        self.num_signals = len(signals_info)

    def run(self):
        times = []
        values = [[] for _ in range(self.num_signals)]
        fig = plt.figure(figsize=(10, 6))
        main_ax = fig.add_subplot(111)
        if self.title:
            main_ax.set_title(self.title)
        axes = [main_ax]

        # For additional signals, create twin axes.
        if self.num_signals > 1:
            for i in range(1, self.num_signals):
                new_ax = main_ax.twinx()
                new_ax.spines["right"].set_position(("outward", 60 * (i - 1)))
                axes.append(new_ax)

        for i, ax in enumerate(axes):
            ax.set_ylim(self.signals_info[i]["ymin"], self.signals_info[i]["ymax"])
            ax.set_ylabel(self.signals_info[i]["name"])
        main_ax.set_xlabel("Time (s)")
        main_ax.grid()

        lines = []
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(self.num_signals):
            line, = axes[i].plot([], [],
                                 color=color_cycle[i % len(color_cycle)],
                                 label=self.signals_info[i]["name"])
            lines.append(line)
        main_ax.legend(loc="upper left")

        texts = []
        for i in range(self.num_signals):
            x_pos = 0.1 + i * 0.3
            txt = fig.text(x_pos, 0.02, "", fontfamily="monospace", fontsize=14)
            texts.append(txt)

        def update_plot(frame):
            # Process all new data.
            while not self.data_queue.empty():
                try:
                    t, vals = self.data_queue.get_nowait()
                    times.append(t)
                    for i in range(self.num_signals):
                        values[i].append(vals[i])
                    # Remove old data outside the rolling window.
                    while times and times[0] < t - self.window_length:
                        times.pop(0)
                        for v in values:
                            v.pop(0)
                except Exception:
                    break

            # Process any control commands.
            try:
                while True:
                    cmd = self.control_queue.get_nowait()
                    if cmd.get('command') == 'close':
                        plt.close(fig)
                        return
            except queue.Empty:
                pass

            if not times:
                return

            current_time = times[-1]
            for i, line in enumerate(lines):
                line.set_data(times, values[i])
            axes[0].set_xlim(current_time - self.window_length, current_time)

            fixed_width = 7  # Fixed width formatting for displayed values.
            for i, txt in enumerate(texts):
                if values[i]:
                    current_val = values[i][-1]
                    formatted_val = f"{current_val:{fixed_width}{self.value_format}}"
                    txt.set_text(f"{self.signals_info[i]['name']}: {formatted_val}")

            fig.canvas.draw_idle()

        def handle_close(event):
            self.event_queue.put({'event': 'close'})

        fig.canvas.mpl_connect('close_event', handle_close)

        ani = animation.FuncAnimation(fig, update_plot, interval=100, frames=20)
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)


# ======================================================================================================================
class RealTimePlot:
    """
    A real-time rolling plot for timeseries data.
    """

    def __init__(self, window_length, signals_info, value_format=".2f", title=None):
        self.window_length = window_length
        self.signals_info = signals_info
        self.value_format = value_format
        self.title = title
        self.num_signals = len(signals_info)
        self.start_time = 0

        self.data_queue = mp.Queue()
        self.control_queue = mp.Queue()
        self.event_queue = mp.Queue()
        self.proc = None

        self.callbacks = RealTimePlotCallbacks()
        self._event_listener_thread = threading.Thread(target=self._event_listener, daemon=True)
        self._event_listener_thread.start()

    def _event_listener(self):
        while True:
            try:
                event = self.event_queue.get(timeout=0.1)
                if event.get('event') == 'close':
                    self.on_close()
                    break
            except queue.Empty:
                continue

    def on_close(self):
        self.callbacks.close.call()

    def start(self):
        self.proc = RealTimePlotProcess(
            data_queue=self.data_queue,
            control_queue=self.control_queue,
            event_queue=self.event_queue,
            window_length=self.window_length,
            signals_info=self.signals_info,
            value_format=self.value_format,
            title=self.title
        )
        self.proc.daemon = True
        self.proc.start()
        self.start_time = time.time()

    def close(self):
        if self.proc is not None:
            self.control_queue.put({'command': 'close'})
            self.proc.join()

    def push_data(self, values):
        if not isinstance(values, list):
            values = [values]
        timestamp = time.time() - self.start_time
        self.data_queue.put((timestamp, values))


# ======================================================================================================================


def use_headless_backend(force: bool = True) -> None:
    """
    Force Matplotlib to use a headless backend (Agg). Call this BEFORE importing
    pyplot if you ever will. Safe for threads and child processes.

    In threads, never create GUI windows; use Agg only.
    """
    matplotlib.use("Agg", force=force)


def choose_gui_backend(preferred: Iterable[str] = ("macosx", "QtAgg", "TkAgg")) -> str:
    """
    Attempt to select an interactive GUI backend (for mp.Process that opens real windows).
    Call this at the very beginning of the .run() method *before* importing pyplot.

    Returns the backend string that was set (or remains unchanged if none matched).
    """
    for name in preferred:
        try:
            matplotlib.use(name, force=True)
            return name
        except Exception:
            continue
    # Fallback to Agg if nothing is available
    matplotlib.use("Agg", force=True)
    return "Agg"


# ==============================================================================
# Figure creation (Agg-only, thread/process safe)
# ==============================================================================

def new_figure_agg(
        figsize: tuple[float, float] = (5, 4),
        dpi: int = 120,
):
    """
    Create a headless Matplotlib Figure using Agg, without importing pyplot.
    Returns (fig, ax).

    Example:
        fig, ax = new_figure_agg((5,4), 120)
        ax.plot(x, y)
        uri = fig_to_data_uri(fig)
    """
    use_headless_backend()  # ensure Agg
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401 (ensures Agg canvas)

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    return fig, ax


@contextmanager
def figure_agg(
        figsize: tuple[float, float] = (5, 4),
        dpi: int = 120,
):
    """
    Context manager to create & auto-close an Agg figure.

    with figure_agg((6,4), 150) as (fig, ax):
        ax.plot(...)
        uri = fig_to_data_uri(fig)
    """
    fig, ax = new_figure_agg(figsize=figsize, dpi=dpi)
    try:
        yield fig, ax
    finally:
        safe_close(fig)


class AggPDFPreviewer:
    """
    Save an Agg-backed Matplotlib figure to a temporary PDF and open it
    with a platform-appropriate viewer. Reusable across your project.
    """

    def __init__(self, prefer_mac_preview: bool = True):
        self.prefer_mac_preview = prefer_mac_preview

    def save_to_temp_pdf(self, fig, save_fn, *, transparent=False,
                         bbox_inches="tight", pad_inches=0.1) -> str:
        """
        Save the given figure `fig` to a temporary .pdf using your provided
        `save_fn(fig, path, fmt='pdf', ...)` utility. Returns the file path.
        """
        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = pdf_file.name
        pdf_file.close()

        # Your project helper `save_fn` is expected to accept fmt="pdf"
        save_fn(fig, pdf_path, fmt="pdf",
                transparent=transparent,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches)
        return pdf_path

    def open_pdf(self, pdf_path: str) -> None:
        """
        Try to open the PDF with a sensible default viewer.
        - macOS: Preview (if available), fallback to `open`.
        - Linux: `xdg-open` if available.
        - Windows: `start` via shell.
        """
        system = platform.system()

        try:
            if system == "Darwin":
                if self.prefer_mac_preview and shutil.which("open"):
                    # Try Preview explicitly; fallback to generic open
                    try:
                        subprocess.Popen(["open", "-a", "Preview", pdf_path])
                        return
                    except Exception:
                        pass
                if shutil.which("open"):
                    subprocess.Popen(["open", pdf_path])
                    return

            elif system == "Linux":
                if shutil.which("xdg-open"):
                    subprocess.Popen(["xdg-open", pdf_path])
                    return

            elif system == "Windows":
                # Use 'start' through the shell
                os.startfile(pdf_path)  # type: ignore[attr-defined]
                return
        except Exception:
            # Swallow open errors silently; caller may handle if needed.
            pass


# ==============================================================================
# Converters: PIL/bytes/data-URI
# ==============================================================================

def pil_to_data_uri(img: Image.Image, fmt: str = "PNG", **save_kwargs) -> str:
    """
    Encode a PIL image into a data URI (default PNG).
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def bytes_to_data_uri(raw: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def array_to_pil(arr: "np.ndarray") -> Image.Image:
    """
    Convert HxW, HxWx3, HxWx4 numpy arrays to PIL.
    Accepts float [0..1] or uint8.
    """
    if np is None:
        raise RuntimeError("NumPy not available; cannot convert array to image.")
    a = arr
    if a.dtype.kind in ("f", "d"):
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255.0 + 0.5).astype("uint8")
    if a.ndim == 2:
        mode = "L"
    elif a.ndim == 3 and a.shape[2] == 3:
        mode = "RGB"
    elif a.ndim == 3 and a.shape[2] == 4:
        mode = "RGBA"
    else:
        raise ValueError("Unsupported array shape; expected HxW, HxWx3, or HxWx4.")
    return Image.fromarray(a, mode=mode)


# ==============================================================================
# Figure renderers (Agg)
# ==============================================================================

def fig_to_png_bytes(
        fig,
        *,
        dpi: int = 120,
        transparent: bool = True,
        bbox_inches: Optional[str] = "tight",
        pad_inches: float = 0.05,
) -> bytes:
    """
    Render a (Agg) Figure to PNG bytes without touching disk.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        transparent=transparent,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    return buf.getvalue()


def fig_to_data_uri(
        fig,
        *,
        dpi: int = 120,
        fmt: str = "png",
        transparent: bool = False,
        bbox_inches: Optional[str] = "tight",
        pad_inches: float = 0.05,
) -> str:
    """
    Render a (Agg) Figure to a data URI.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format=fmt,
        dpi=dpi,
        transparent=transparent,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    mime = f"image/{fmt.lower()}"
    return bytes_to_data_uri(buf.getvalue(), mime=mime)


def safe_close(fig) -> None:
    """
    Close a figure without requiring pyplot to be present.
    """
    try:
        import matplotlib.pyplot as plt  # optional
        plt.close(fig)
    except Exception:
        # Best-effort close if pyplot isn't available
        try:
            fig.clf()
        except Exception:
            pass


def clamp_color(color: Any) -> Any:
    """
    Clamp iterable colors to [0, 1]; pass-through for strings/None.
    """
    if color is None:
        return None
    if isinstance(color, (str, bytes)):
        return color
    try:
        return [max(0.0, min(1.0, float(c))) for c in color]
    except Exception:
        return color


def run_periodic(stop_flag: Callable[[], bool], interval_s: float, tick: Callable[[], None]) -> None:
    """
    Call tick() every interval while stop_flag() is False.
    Helpful for threaded animation loops.
    """
    import time
    dt = max(1e-6, float(interval_s))
    while not stop_flag():
        tick()
        time.sleep(dt)


# ==============================================================================
# Quick recipes
# ==============================================================================

def quick_line_plot_data_uri(
        x, y,
        *,
        figsize=(5, 4), dpi=120,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[Any] = None,
        label: Optional[str] = None,
        grid_alpha: float = 0.3,
) -> str:
    """
    One-shot helper: build an Agg figure, plot (x,y), return data URI.
    No pyplot; safe in threads/processes.
    """
    fig, ax = new_figure_agg(figsize=figsize, dpi=dpi)
    ax.plot(x, y, lw=2, color=color, label=label)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if label:
        ax.legend(loc="best")
    ax.grid(True, alpha=grid_alpha)
    uri = fig_to_data_uri(fig, dpi=dpi)
    safe_close(fig)
    return uri


# ======================================================================================================================
def save_figure(
        fig,
        filepath: str,
        fmt: str = "png",
        dpi: int = 150,
        transparent: bool = False,
        bbox_inches: str | None = "tight",
        pad_inches: float = 0.05,
        **kwargs
) -> str:
    """
    Save a Matplotlib figure to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure handle.
    filepath : str
        Target file path. The function will add the correct extension if missing.
    fmt : str, default "png"
        Output format: "png", "pdf", "svg", etc.
        - PNG → raster (bitmap)
        - PDF/SVG → vector graphics
    dpi : int, default 150
        Dots per inch (only relevant for raster formats like PNG).
    transparent : bool, default False
        If True, make the figure background transparent.
    bbox_inches : str or None, default "tight"
        Bounding box option for savefig.
    pad_inches : float, default 0.05
        Padding around the figure.
    **kwargs : dict
        Passed through to `fig.savefig`.

    Returns
    -------
    str : The absolute path of the saved file.
    """
    import os
    root, ext = os.path.splitext(filepath)
    ext = ext.lstrip(".").lower()
    # If no extension in filepath, add fmt
    if not ext:
        filepath = f"{filepath}.{fmt}"
    else:
        fmt = ext  # override fmt with extension

    fig.savefig(
        filepath,
        format=fmt,
        dpi=dpi,
        transparent=transparent,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        **kwargs
    )
    return os.path.abspath(filepath)


# ======================================================================================================================

# ======================================================================================================================

# ======================================================================================================================
# Test functions demonstrating the functionality.
def testUpdatablePlot():
    """
    Demonstrates the static/updatable plot by appending and removing curves.
    """
    up = UpdatablePlot(x_label="Time", y_label="Value", xlim=(0, 10), ylim=(-1, 1))
    up.callbacks.close.register(function=lambda: print("UpdatablePlot: Plot window closed callback triggered."))
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    up.appendPlot(x, y1, color='blue', label='sin')
    time.sleep(2)
    y2 = np.cos(x)
    up.appendPlot(x, y2, color='red', label='cos')
    time.sleep(2)
    up.removePlot(index=0)
    time.sleep(2)
    up.removePlot(last=True)
    time.sleep(2)
    up.close()


def testRealTimePlot():
    """
    Demonstrates the real-time rolling plot by pushing random data.
    """
    signals_info = [
        {"name": "Signal 1", "ymin": -1, "ymax": 1},
        {"name": "Signal 2", "ymin": 0, "ymax": 10}
    ]
    rt = RealTimePlot(window_length=10, signals_info=signals_info, title="Real-Time Plot")
    rt.callbacks.close.register(function=lambda: print("RealTimePlot: Plot window closed callback triggered."))
    rt.start()
    start_time = time.time()
    while time.time() - start_time < 15:
        data1 = np.sin(time.time())
        data2 = np.random.random() * 10
        rt.push_data([data1, data2])
        time.sleep(0.1)
    rt.close()


if __name__ == "__main__":
    print("Testing UpdatablePlot...")
    testUpdatablePlot()
    time.sleep(1)
    print("Testing RealTimePlot...")
    testRealTimePlot()
