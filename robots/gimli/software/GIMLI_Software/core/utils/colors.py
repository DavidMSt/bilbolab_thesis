import colorsys
import random
from typing import Union, Tuple, List, Literal, Sequence

import seaborn as sns
import matplotlib.colors as mcolors

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GREY = (128, 128, 128)
LIGHT_GREY = (192, 192, 192)

# Dark colors
DARK_RED = (139, 0, 0)
DARK_GREEN = (0, 100, 0)
DARK_BLUE = (0, 0, 139)
DARK_YELLOW = (204, 204, 0)
DARK_CYAN = (0, 139, 139)
DARK_MAGENTA = (139, 0, 139)
DARK_PURPLE = (128, 0, 128)
DARK_ORANGE = (255, 140, 0)
DARK_BROWN = (139, 69, 19)
DARK_GREY = (100, 100, 100)

# Light colors
LIGHT_RED = (255, 99, 71)
LIGHT_GREEN = (144, 238, 144)
LIGHT_BLUE = (173, 216, 230)
LIGHT_YELLOW = (255, 255, 224)
LIGHT_CYAN = (224, 255, 255)
LIGHT_MAGENTA = (255, 0, 255)
LIGHT_PURPLE = (218, 112, 214)
LIGHT_ORANGE = (255, 165, 0)
LIGHT_BROWN = (205, 133, 63)

# Medium colors
MEDIUM_RED = (220, 20, 60)
MEDIUM_GREEN = (60, 179, 113)
MEDIUM_BLUE = (0, 0, 205)
MEDIUM_YELLOW = (255, 255, 0)
MEDIUM_CYAN = (0, 255, 255)
MEDIUM_MAGENTA = (255, 0, 255)
MEDIUM_PURPLE = (147, 112, 219)
MEDIUM_ORANGE = (255, 165, 0)
MEDIUM_BROWN = (165, 42, 42)


def rgb_to_hex(rgb):
    """
    Convert a list of RGB(A) values (0–1 floats) to a hex HTML color.
    Examples:
      rgb_to_hex([0.5, 0.2, 0.8])       -> "#8033cc"
      rgb_to_hex([0.5, 0.2, 0.8, 0.3])  -> "#8033cc4d"
    """

    def clamp(x):
        return max(0, min(x, 1))

    if rgb is None:
        return None

    if len(rgb) == 3 or len(rgb) == 4:
        # clamp & scale
        comps = [int(clamp(c) * 255) for c in rgb]
        # format RGB
        hex_str = "#{:02x}{:02x}{:02x}".format(*comps[:3])
        # if alpha present, append as two hex digits
        if len(comps) == 4:
            hex_str += "{:02x}".format(comps[3])
        return hex_str

    # fallback on bad input
    return "#FFFFFF"


def random_color(len=3):
    if len == 3:
        return [random.random(), random.random(), random.random()]
    elif len == 4:
        return [random.random(), random.random(), random.random(), random.random()]
    else:
        return None


_PREDEFINED_PALETTES = {
    "muted": sns.color_palette("muted"),  # ~8 colors
    "pastel": sns.color_palette("pastel"),  # ~8 colors
    "dark": sns.color_palette("dark"),  # ~8 colors
    "bright": sns.color_palette("bright"),  # ~8 colors
    "colorblind": sns.color_palette("colorblind"),  # ~8 colors
    "deep": sns.color_palette("deep"),  # ~8 colors
    # ... you can add more, e.g.:
    # "cubehelix":    sns.color_palette("cubehelix", 8),
    # "viridis":      sns.color_palette("viridis",   8),
    # "inferno":      sns.color_palette("inferno",   8),
    # "cividis":      sns.color_palette("cividis",   8),
}


def get_palette(name, n_colors=8):
    """
    Return a list of `n_colors` float‐RGB tuples in [0,1].
    Examples:
      get_palette("muted", 5)   → 5 “muted” colors
      get_palette("bright", 10) → 10 “bright” colors
    If `name` is not found, raises KeyError.
    """
    if name not in _PREDEFINED_PALETTES:
        raise KeyError(f"Palette '{name}' is not defined. Available: {list(_PREDEFINED_PALETTES.keys())}")
    # Seaborn will automatically cycle/ interpolate if you ask for > base size.
    return sns.color_palette(name, n_colors)


def get_color_from_palette(name, n_colors, index):
    if name not in _PREDEFINED_PALETTES:
        raise KeyError(f"Palette '{name}' is not defined. Available: {list(_PREDEFINED_PALETTES.keys())}")
    return _PREDEFINED_PALETTES[name][index % n_colors]


def get_palette_hex(name, n_colors=8):
    """
    Same as get_palette(), but each color is converted to a hex string "#RRGGBB".
    """
    float_list = get_palette(name, n_colors)
    # rgb_to_hex expects a list [r,g,b] in floats 0..1
    return [rgb_to_hex(color) for color in float_list]


def random_color_from_palette(name):
    """
    Return one random float‐RGB tuple from the named palette (using its standard size).
    """
    base = _PREDEFINED_PALETTES.get(name)
    if base is None:
        raise KeyError(f"Palette '{name}' is not defined.")
    return random.choice(base)


def random_color_from_palette_hex(name):
    """
    Return one random color from the named palette, as "#RRGGBB".
    """
    c = random_color_from_palette(name)
    return rgb_to_hex(c)


def get_shaded_color(base_color: str | tuple[float, float, float] | list[float],
                     total_steps: int,
                     index: int) -> tuple:
    """
    Return an RGBA color with increasing alpha (transparency) based on index.

    Args:
        base_color (str | list | tuple): Base color (name, hex, or RGB [0-1]).
        total_steps (int): Total number of curves.
        index (int): Current index (0-based).

    Returns:
        tuple: RGBA color (r, g, b, a)
    """
    if isinstance(base_color, (list, tuple)):
        rgb = tuple(base_color)
    else:
        rgb = mcolors.to_rgb(base_color)

    if total_steps <= 1:
        alpha = 1.0
    else:
        alpha = 0.2 + 0.8 * index / (total_steps - 1)  # from 0.2 to 1.0

    return (*rgb, alpha)


from typing import List

Color = List[float]
Mode = Literal["interpolate", "add", "multiply", "screen", "overlay", "darken", "lighten"]


def getColorGradient(color1: Color, color2: Color, num_colors: int) -> List[Color]:
    """
    Linearly interpolate between color1 and color2 (0–1 RGB or RGBA lists/tuples),
    returning `num_colors` colors including both endpoints.

    - If num_colors <= 0 → []
    - If num_colors == 1 → [color1 (trimmed to common channel length)]
    - Supports RGB or RGBA; uses the common channel count (min length).
    """
    if num_colors <= 0:
        return []

    length = min(len(color1), len(color2))
    c1 = list(color1[:length])
    c2 = list(color2[:length])

    if num_colors == 1:
        return [c1]

    out = []
    for i in range(num_colors):
        t = i / (num_colors - 1)
        out.append(mix_colors(c1, c2, mode="interpolate", t=t))
    return out


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, value))


def mix_colors(color1: Color, color2: Color, mode: Mode = "interpolate", t: float = 0.5) -> Color:
    length = min(len(color1), len(color2))
    c1 = color1[:length]
    c2 = color2[:length]

    def interp(a, b):
        return (1 - t) * a + t * b

    mixed = []

    for i in range(length):
        a = c1[i]
        b = c2[i]

        if mode == "interpolate":
            val = interp(a, b)
        elif mode == "add":
            val = clamp(a + b)
        elif mode == "multiply":
            val = a * b
        elif mode == "screen":
            val = 1 - (1 - a) * (1 - b)
        elif mode == "overlay":
            val = 2 * a * b if a < 0.5 else 1 - 2 * (1 - a) * (1 - b)
        elif mode == "darken":
            val = min(a, b)
        elif mode == "lighten":
            val = max(a, b)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        mixed.append(clamp(val))

    return mixed











def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _interp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

def _interp_color_rgb(c1: Color, c2: Color, t: float) -> Color:
    length = min(len(c1), len(c2), 4)
    return [
        _clamp01(_interp(c1[i], c2[i], t))
        for i in range(min(length, 3))
    ] + ([ _clamp01(_interp(c1[3], c2[3], t)) ] if length >= 4 else [])


# 1) Multi-hue spiral with rising brightness (HSV)
def get_progression_colors_multi_hue(n: int,
                                     cycles: float = 2.5,
                                     start_h: float = 0.0,
                                     s: float = 0.8,
                                     v_min: float = 0.35,
                                     v_max: float = 0.9) -> List[Color]:
    """
    Generates n colors by cycling the hue 'cycles' times while ramping value (brightness).
    Great for 20–30 trials: adjacent colors jump around the wheel but overall get brighter.
    """
    if n <= 0:
        return []
    if n == 1:
        h = start_h % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v_max)
        return [[r, g, b]]
    out = []
    for i in range(n):
        t = i / (n - 1)
        h = (start_h + cycles * t) % 1.0
        v = _interp(v_min, v_max, t)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append([r, g, b])
    return out


# 2) Segmented interpolation across multiple anchors
def get_segmented_progression_colors(n: int,
                                     anchors: Sequence[Color],
                                     gamma: float = 1.0) -> List[Color]:
    """
    Interpolates across a list of anchor colors (RGB 0–1) to produce n ordered colors.
    'gamma' > 1 biases toward anchors; gamma < 1 densifies the middle.
    Example anchors: [[0,0,0.8],[0,0.6,0.3],[0.9,0.5,0.0],[0.8,0.0,0.0]]
    """
    if n <= 0:
        return []
    if n == 1 or len(anchors) == 1:
        return [list(anchors[0])]

    # Normalize anchors lengths to 3
    anchors = [list(a[:3]) for a in anchors]

    out = []
    m = len(anchors) - 1
    for i in range(n):
        t = i / (n - 1)
        # gamma shaping (monotonic)
        if gamma != 1.0:
            t = t ** gamma
        # Which segment?
        seg = min(int(t * m), m - 1)
        local_start = seg / m
        local_end = (seg + 1) / m
        u = 0.0 if local_end == local_start else (t - local_start) / (local_end - local_start)
        c = _interp_color_rgb(anchors[seg], anchors[seg + 1], u)
        out.append(c)
    return out


# 3) Cycle a qualitative palette while ramping lightness (HLS)
def get_palette_cycling_with_lightness(n: int,
                                       base_palette: Sequence[Color],
                                       l_min: float = 0.38,
                                       l_max: float = 0.82) -> List[Color]:
    """
    Repeats a provided palette (e.g., sns.color_palette('colorblind', 8)), but
    gradually increases lightness across indices to suggest progression.
    """
    if n <= 0:
        return []
    k = len(base_palette)
    if k == 0:
        return []

    out = []
    for i in range(n):
        t = 0 if n == 1 else i / (n - 1)
        l_target = _interp(l_min, l_max, t)
        r0, g0, b0 = base_palette[i % k][:3]
        h, l, s = colorsys.rgb_to_hls(r0, g0, b0)
        r, g, b = colorsys.hls_to_rgb(h, l_target, s)
        out.append([_clamp01(r), _clamp01(g), _clamp01(b)])
    return out