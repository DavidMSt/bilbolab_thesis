import random
import seaborn as sns


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


# ── NEW ADDITIONS BELOW ─────────────────────────────────────────────────────

# 1) Define some “named” palettes (using Seaborn’s built-in lookup).
#    Each entry is just a *default* small palette of, say, 8 colors.
#    You can of course override the size every time you call get_palette().

_PREDEFINED_PALETTES = {
    "muted":        sns.color_palette("muted"),        # ~8 colors
    "pastel":       sns.color_palette("pastel"),       # ~8 colors
    "dark":         sns.color_palette("dark"),         # ~8 colors
    "bright":       sns.color_palette("bright"),       # ~8 colors
    "colorblind":   sns.color_palette("colorblind"),   # ~8 colors
    "deep":         sns.color_palette("deep"),         # ~8 colors
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

# ── END OF NEW ADDITIONS ─────────────────────────────────────────────────────