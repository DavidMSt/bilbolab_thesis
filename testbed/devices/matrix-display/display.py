# gfx_matrix.py
import math
from typing import Iterable, Tuple, Optional, Union
# Example 4: Mix seven-seg and normal text
import time
from rgbmatrix import RGBMatrix, RGBMatrixOptions
from PIL import Image, ImageDraw, ImageFont

RGB = Tuple[int, int, int]
Point = Tuple[int, int]


class MatrixGFX:
    """
    Convenience drawing wrapper for rpi-rgb-led-matrix.
    - Uses a Pillow backbuffer (RGB) sized to matrix.width x matrix.height.
    - Double-buffers via CreateFrameCanvas()/SwapOnVSync() if available, else SetImage().
    - Provides crisp primitives, crisp text (anti-aliasing optional), and beveled seven-seg digits.

    Args:
        matrix: an instance of rgbmatrix.RGBMatrix.
        use_vsync: if True, uses a FrameCanvas and SwapOnVSync() for tear-free updates.
        default_font_path: optional TTF path (monospace recommended, e.g. DejaVuSansMono.ttf).
        default_font_size: default text font size in pixels.

    Notes:
        - Colors are (r, g, b) 0..255.
        - Coordinates are (x, y) with (0,0) top-left; typical panel is 64x32 (landscape).
    """
    def __init__(self,
                 matrix,
                 use_vsync: bool = True,
                 default_font_path: Optional[str] = None,
                 default_font_size: int = 8):
        self.matrix = matrix
        self.width = matrix.width
        self.height = matrix.height

        self.image = Image.new('RGB', (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)

        self.use_vsync = use_vsync
        self.frame = None
        if self.use_vsync and hasattr(matrix, "CreateFrameCanvas"):
            self.frame = matrix.CreateFrameCanvas()

        # Text font (attempt monospaced -> fallback to default)
        self.font = self._load_font(default_font_path, default_font_size)

    # ---------- Utility ----------

    def _load_font(self, path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
        try:
            if path is not None:
                return ImageFont.truetype(path, size)
            # Try common monospace if path not provided
            for candidate in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            ]:
                try:
                    return ImageFont.truetype(candidate, size)
                except Exception:
                    continue
            # Fallback
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    def set_font(self, font_path: str, size: int) -> None:
        """Swap the default text font."""
        self.font = self._load_font(font_path, size)

    # ---------- Buffer control ----------

    def clear(self, color: RGB = (0, 0, 0)) -> None:
        """Clear backbuffer to a solid color."""
        self.draw.rectangle([(0, 0), (self.width - 1, self.height - 1)], fill=color)

    def fill(self, color: RGB) -> None:
        """Alias for clear(color)."""
        self.clear(color)

    def set_pixel(self, x: int, y: int, color: RGB) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.image.putpixel((x, y), color)

    def blit(self) -> None:
        """
        Push backbuffer to the panel using a method that works across bindings.
        - Prefer FrameCanvas + SetImage() + SwapOnVSync().
        - Fall back to Matrix.SetImage() if no frame canvas available.
        """
        if self.frame is not None:
            if hasattr(self.frame, "SetImage"):
                self.frame.SetImage(self.image)
            else:
                self.matrix.SetImage(self.image)
            self.frame = self.matrix.SwapOnVSync(self.frame)  # must reassign
        else:
            self.matrix.SetImage(self.image)

    # ---------- Drawing primitives ----------

    def line(self, p1: Point, p2: Point, color: RGB) -> None:
        # Pillow draws crisp 1px lines if coordinates are integers (they are).
        self.draw.line([p1, p2], fill=color)

    def rectangle(self, xy: Tuple[int, int, int, int], outline: Optional[RGB] = None, fill: Optional[RGB] = None) -> None:
        """xy = (x, y, w, h)"""
        x, y, w, h = xy
        self.draw.rectangle([x, y, x + w - 1, y + h - 1], outline=outline, fill=fill)

    def rounded_rect(self, xy: Tuple[int, int, int, int], radius: int, outline: Optional[RGB] = None, fill: Optional[RGB] = None) -> None:
        x, y, w, h = xy
        # rounded_rectangle can AA edges slightly; draw integer coords to minimize it
        self.draw.rounded_rectangle([x, y, x + w - 1, y + h - 1], radius=radius, outline=outline, fill=fill)

    def circle(self, center: Point, r: int, outline: Optional[RGB] = None, fill: Optional[RGB] = None) -> None:
        cx, cy = center
        self.draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=outline, fill=fill)

    def triangle(self, p1: Point, p2: Point, p3: Point, outline: Optional[RGB] = None, fill: Optional[RGB] = None) -> None:
        self.draw.polygon([p1, p2, p3], outline=outline, fill=fill)

    def polygon(self, points: Iterable[Point], outline: Optional[RGB] = None, fill: Optional[RGB] = None) -> None:
        self.draw.polygon(list(points), outline=outline, fill=fill)

    # ---------- Text (crisp by default) ----------

    def _text_bbox(self, s: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        try:
            bbox = font.getbbox(s)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            w, h = font.getsize(s)
        return w, h

    def text(self,
             xy: Point,
             s: str,
             color: RGB = (255, 255, 255),
             font: Optional[ImageFont.ImageFont] = None,
             anchor: Optional[str] = None,
             crisp: bool = True,
             threshold: int = 1) -> Tuple[int, int]:
        """
        Draw text. If crisp=True, render to a mask and hard-threshold it so every lit pixel is full brightness
        (eliminates half-bright anti-aliased edges). threshold=1..255 controls how aggressive the mask is.
        Returns (w, h) of the rendered text.
        """
        f = font or self.font
        w, h = self._text_bbox(s, f)

        if not crisp:
            self.draw.text(xy, s, fill=color, font=f, anchor=anchor)
            return w, h

        # Render to a grayscale mask, then threshold to {0,255}, and paste with mask
        mask = Image.new("L", (w, h), 0)
        mdraw = ImageDraw.Draw(mask)
        # draw at (0,0) because mask is tightly sized
        mdraw.text((0, 0), s, fill=255, font=f)
        if threshold > 1:
            mask = mask.point(lambda p: 255 if p >= threshold else 0, mode="1").convert("L")
        # Paste solid color through a hard mask (no anti-alias)
        patch = Image.new("RGB", (w, h), color)
        self.image.paste(patch, xy, mask)
        return w, h

    # ---------- Big seven-segment numbers (beveled, crisp) ----------

    def big_digits(self,
                   xy: Point,
                   text: str,
                   color_on: RGB = (255, 0, 0),
                   color_off: Optional[RGB] = None,
                   height: int = 28,
                   thickness: int = 5,
                   spacing: int = 2,
                   colon_gap: int = 3,
                   bevel: int = 2) -> Tuple[int, int]:
        """
        Draw large, real seven-segment style digits (0-9, space, '-', ':') with beveled ends.
        - height: total digit height in pixels.
        - thickness: segment thickness.
        - color_off: if provided, draws unlit segments in this color (also crisp).
        - bevel: size of diagonal bevel on segment ends.
        Returns total (w, h).
        """
        x, y = xy
        total_w = 0
        max_h = 0
        for ch in text:
            if ch == ":":
                w, h = self._draw_colon((x, y), height, thickness, color_on)
                x += w + colon_gap
                total_w += w + colon_gap
                max_h = max(max_h, h)
            else:
                w, h = self._draw_seven_seg_digit((x, y), ch, height, thickness, bevel, color_on, color_off)
                x += w + spacing
                total_w += w + spacing
                max_h = max(max_h, h)
        total_w = max(0, total_w - spacing)
        return total_w, max_h

    def _digit_segments(self, ch: str) -> Tuple[bool, ...]:
        """
        Segment order: A, B, C, D, E, F, G (top, top-right, bottom-right, bottom, bottom-left, top-left, middle)
        """
        segs = {
            "0": (1, 1, 1, 1, 1, 1, 0),
            "1": (0, 1, 1, 0, 0, 0, 0),
            "2": (1, 1, 0, 1, 1, 0, 1),
            "3": (1, 1, 1, 1, 0, 0, 1),
            "4": (0, 1, 1, 0, 0, 1, 1),
            "5": (1, 0, 1, 1, 0, 1, 1),
            "6": (1, 0, 1, 1, 1, 1, 1),
            "7": (1, 1, 1, 0, 0, 0, 0),
            "8": (1, 1, 1, 1, 1, 1, 1),
            "9": (1, 1, 1, 1, 0, 1, 1),
            "-": (0, 0, 0, 0, 0, 0, 1),
            " ": (0, 0, 0, 0, 0, 0, 0),
        }
        return segs.get(ch, segs[" "])

    def _seg_metrics(self, height: int, thickness: int) -> Tuple[int, int, int, int]:
        """
        Compute digit layout and metrics.
        Returns (digit_w, seg_len_vert, seg_thick, mid_y).
        """
        t = max(1, int(thickness))
        # Classic seven-seg width ≈ 0.56 * height works well visually on 64x32
        digit_w = max(5, int(round(height * 0.56)))
        mid_y = height // 2
        seg_len_vert = mid_y - t  # vertical segment usable length (each half)
        return digit_w, seg_len_vert, t, mid_y

    def _hseg_polygon(self, x: int, y: int, w: int, t: int, bevel: int) -> list:
        """
        Horizontal segment polygon (beveled). (x,y) is top-left of the segment body box.
        """
        b = min(bevel, t, w // 2)
        # points clockwise
        return [
            (x + b, y),
            (x + w - b, y),
            (x + w, y + t // 2),
            (x + w - b, y + t),
            (x + b, y + t),
            (x, y + t // 2),
        ]

    def _vseg_polygon(self, x: int, y: int, t: int, h: int, bevel: int) -> list:
        """
        Vertical segment polygon (beveled). (x,y) is top-left of the segment body box.
        """
        b = min(bevel, t, h // 2)
        return [
            (x, y + b),
            (x + t // 2, y),
            (x + t, y + b),
            (x + t, y + h - b),
            (x + t // 2, y + h),
            (x, y + h - b),
        ]

    def _draw_seven_seg_digit(self,
                              xy: Point,
                              ch: str,
                              height: int,
                              thickness: int,
                              bevel: int,
                              color_on: RGB,
                              color_off: Optional[RGB]) -> Tuple[int, int]:
        """
        Draw a single seven-seg digit with beveled polygons, crisp (no AA).
        """
        x, y = xy
        digit_w, seg_len, t, mid_y = self._seg_metrics(height, thickness)
        mask = self._digit_segments(ch)

        # Precompute rectangles for segment bodies (before bevel triangulation)
        # layout padding inside digit box
        pad_x = t
        pad_y = t

        # Horizontal segments A (top), D (bottom), G (middle)
        # body widths subtract left/right t to leave room for verticals
        body_w = max(1, digit_w - 2 * pad_x)
        # A
        ax, ay = x + pad_x, y
        # D
        dx, dy = x + pad_x, y + height - t
        # G
        gx, gy = x + pad_x, y + mid_y - t // 2

        # Vertical segments F(top-left), B(top-right), E(bottom-left), C(bottom-right)
        # Each vertical segment spans seg_len with a gap around G
        # Top half
        fx, fy = x, y + pad_y
        bx, by = x + digit_w - t, y + pad_y
        # Bottom half (below middle + t)
        ex, ey = x, y + mid_y + t // 2
        cx, cy = x + digit_w - t, y + mid_y + t // 2

        def paste_poly(points: list, on: bool):
            col = color_on if on else color_off
            if col is None:
                if on:
                    # draw directly full color (fills are crisp on integer points)
                    self.draw.polygon(points, fill=color_on)
                # else skip entirely
                return
            # draw through a hard mask for both on/off to keep consistent edges
            # compute bounds
            minx = max(0, min(p[0] for p in points))
            miny = max(0, min(p[1] for p in points))
            maxx = min(self.width - 1, max(p[0] for p in points))
            maxy = min(self.height - 1, max(p[1] for p in points))
            if maxx < minx or maxy < miny:
                return
            w = maxx - minx + 1
            h = maxy - miny + 1
            local = [(px - minx, py - miny) for (px, py) in points]
            m = Image.new("1", (w, h), 0)
            md = ImageDraw.Draw(m)
            md.polygon(local, fill=1)
            patch = Image.new("RGB", (w, h), col)
            self.image.paste(patch, (minx, miny), m)

        # A
        if True:
            poly = self._hseg_polygon(ax, ay, body_w, t, bevel)
            paste_poly(poly, bool(mask[0]))
        # D
        if True:
            poly = self._hseg_polygon(dx, dy, body_w, t, bevel)
            paste_poly(poly, bool(mask[3]))
        # G
        if True:
            poly = self._hseg_polygon(gx, gy, body_w, t, bevel)
            paste_poly(poly, bool(mask[6]))
        # F (top-left)
        poly = self._vseg_polygon(fx, fy, t, seg_len, bevel)
        paste_poly(poly, bool(mask[5]))
        # B (top-right)
        poly = self._vseg_polygon(bx, by, t, seg_len, bevel)
        paste_poly(poly, bool(mask[1]))
        # E (bottom-left)
        poly = self._vseg_polygon(ex, ey, t, seg_len, bevel)
        paste_poly(poly, bool(mask[4]))
        # C (bottom-right)
        poly = self._vseg_polygon(cx, cy, t, seg_len, bevel)
        paste_poly(poly, bool(mask[2]))

        return digit_w, height

    def _draw_colon(self,
                    xy: Point,
                    height: int,
                    thickness: int,
                    color_on: RGB) -> Tuple[int, int]:
        """
        Colon as two crisp squares (rects), avoids AA from ellipse edges.
        """
        x, y = xy
        t = max(1, thickness)
        gap = max(2, height // 6)
        cy = y + height // 2
        # top square
        self.rectangle((x, cy - gap - t, t, t), fill=color_on)
        # bottom square
        self.rectangle((x, cy + gap, t, t), fill=color_on)
        w = t
        return w, height

    # ---------- Convenience helpers ----------

    def image_from_pillow(self, img: Image.Image, fit: bool = True) -> None:
        """
        Draw a PIL image onto the backbuffer. If fit=True, scale to fit using NEAREST (no AA).
        """
        if fit:
            img = img.copy()
            # Avoid LANCZOS to prevent half-brightness blend; use NEAREST for LED matrices.
            img.thumbnail((self.width, self.height), Image.NEAREST)
        self.image.paste(img.convert('RGB'), (0, 0))

    # ---------- Extra helpers for common tasks ----------

    def measure_text(self, s: str, font: Optional[ImageFont.ImageFont] = None) -> Tuple[int, int]:
        """Return (w,h) of text using current or provided font."""
        return self._text_bbox(s, font or self.font)

    def text_centered(self, cy: int, s: str, color: RGB, crisp: bool = True) -> None:
        """Horizontally center text on the display at vertical y=cy."""
        w, h = self.measure_text(s)
        x = (self.width - w) // 2
        self.text((x, cy), s, color=color, crisp=crisp)






#
# # Example 3: Scrolling crisp text
# import time
# from rgbmatrix import RGBMatrix, RGBMatrixOptions
#
# options = RGBMatrixOptions()
# options.hardware_mapping = 'adafruit-hat'
# options.rows = 32
# options.cols = 64
# matrix = RGBMatrix(options=options)
# gfx = MatrixGFX(matrix, default_font_size=20)
#
# msg = "  CRISP TEXT SCROLL  "
# w, h = gfx.measure_text(msg)
# x = gfx.width
# try:
#     while True:
#         gfx.clear((0, 0, 0))
#         gfx.text((x, (gfx.height - h) // 2), msg, color=(255, 0, 0), crisp=False)
#         gfx.blit()
#         x -= 1
#         if x < -w:
#             x = gfx.width
#         time.sleep(0.1)
# except KeyboardInterrupt:
#     pass



# Example 3: Scrolling crisp text
import time
from rgbmatrix import RGBMatrix, RGBMatrixOptions

options = RGBMatrixOptions()
options.hardware_mapping = 'adafruit-hat'
options.rows = 32
options.cols = 64
matrix = RGBMatrix(options=options)
gfx = MatrixGFX(matrix, default_font_size=12)

trial_num = 1
x = 3
y= 8
try:
    while True:
        gfx.clear((0, 0, 0))
        msg = "TRIAL " + str(trial_num)
        gfx.text((x, y), msg, color=(150, 150, 150), crisp=False)
        gfx.blit()
        trial_num += 1
        time.sleep(0.03)
except KeyboardInterrupt:
    pass