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



# # Example 3: Scrolling crisp text
# import time
# from rgbmatrix import RGBMatrix, RGBMatrixOptions
#
# options = RGBMatrixOptions()
# options.hardware_mapping = 'adafruit-hat'
# options.rows = 32
# options.cols = 64
# matrix = RGBMatrix(options=options)
# gfx = MatrixGFX(matrix, default_font_size=12)
#
# trial_num = 1
# x = 3
# y= 8
# try:
#     while True:
#         gfx.clear((0, 0, 0))
#         msg = "TRIAL " + str(trial_num)
#         gfx.text((x, y), msg, color=(150, 150, 150), crisp=False)
#         gfx.blit()
#         trial_num += 1
#         time.sleep(0.03)
# except KeyboardInterrupt:
#     pass

# --- Robot pictogram -------------------------------------------------

# def draw_robot_pictogram(
#     gfx: MatrixGFX,
#     base_x: int,
#     ground_y: int,
#     pitch: float = 0.0,
#     body_color=(40, 40, 40),
#     wheel_color=(10, 10, 10),
#     wheel_outline=(120, 120, 120),
#     accent_color=(0, 200, 0),
# ):
#     """
#     Side-view robot:
#       - one big wheel
#       - a body sitting on top of it
#     base_x   : left edge of the body
#     ground_y : y position of the ground line (bottom of the wheel)
#     pitch    : small nose-down / nose-up angle (-0.3 .. +0.3 typical)
#                positive = nose down (front lower)
#     """
#
#     # --- basic sizes tuned for 64x32 panel ---
#     wheel_r = 6                 # radius of wheel
#     body_w  = 10               # width of robot body
#     body_h  = 18                 # height of robot body
#
#     # Wheel centre: middle of body
#     wheel_cx = base_x + body_w // 2
#     wheel_cy = ground_y - wheel_r
#
#     # --- body with small pitch (parallelogram) ---
#     # "Neutral" body top position (no pitch)
#     body_top_y = wheel_cy - body_h - 1
#
#     # Pitch amount in pixels (clamped)
#     pitch = max(-6, min(6, pitch))  # safety
#     d = int(pitch * body_h)  # how much front is lower/higher
#
#     back_top_y = body_top_y - d
#     front_top_y = body_top_y + d
#     back_bot_y = back_top_y + body_h
#     front_bot_y = front_top_y + body_h
#
#     body_points = [
#         (base_x, back_top_y),
#         (base_x + body_w, front_top_y),
#         (base_x + body_w, front_bot_y),
#         (base_x, back_bot_y),
#     ]
#     gfx.polygon(body_points, outline=body_color, fill=body_color)
#
#     # --- wheel ---
#     gfx.circle((wheel_cx, wheel_cy), wheel_r,
#                outline=wheel_outline, fill=wheel_color)
#
#     # simple hub
#     gfx.circle((wheel_cx, wheel_cy), 2,
#                outline=(180, 180, 180), fill=(60, 60, 60))
#
#
#     # --- little "sensor head" at the front ---
#     head_w, head_h = 4, 3
#     head_x = base_x + body_w - head_w
#     head_y = front_top_y + 1
#     gfx.rectangle((head_x, head_y, head_w, head_h),
#                   outline=accent_color, fill=accent_color)
#
#     # tiny eye
#     gfx.rectangle((head_x + head_w - 2, head_y + 1, 1, 1),
#                   fill=(0, 0, 0))

def draw_robot_pictogram(
    gfx: MatrixGFX,
    base_x: int,
    ground_y: int,
    pitch: float = 0.0,        # degrees, + = nose down
    body_color=(200, 200, 200),
    wheel_color=(10, 10, 10),
    wheel_outline=(120, 120, 120),
    accent_color=(0, 200, 0),
):
    """
    Side-view robot with one wheel.
    The body starts at the wheel centre and extends upward.
    The entire body + head rotates around the wheel axis.
    """
    import math

    # --- sizes ---
    wheel_r = 6          # radius of wheel
    body_w  = 10         # width of body
    body_h  = 18         # height of body (above wheel centre)

    # Wheel centre: middle of body
    wheel_cx = base_x + body_w // 2
    wheel_cy = ground_y - wheel_r

    # Clamp & convert pitch to radians
    pitch = max(-15.0, min(15.0, pitch))   # little tilt only
    angle = math.radians(pitch)
    ca = math.cos(angle)
    sa = math.sin(angle)

    def rot(pt):
        """Rotate point around wheel centre by 'angle'."""
        x, y = pt
        dx = x - wheel_cx
        dy = y - wheel_cy
        return (
            int(round(wheel_cx + dx * ca - dy * sa)),
            int(round(wheel_cy + dx * sa + dy * ca)),
        )

    # --- neutral body rectangle (before rotation) ---
    # Body bottom is exactly at wheel centre; body extends upward.
    body_bottom_y = wheel_cy
    body_top_y = body_bottom_y - body_h
    left = base_x
    right = base_x + body_w

    body_points = [
        rot((left,  body_top_y)),     # back-top
        rot((right, body_top_y)),     # front-top
        rot((right, body_bottom_y)),  # front-bottom (at wheel centre)
        rot((left,  body_bottom_y)),  # back-bottom (at wheel centre)
    ]

    # --- body ---
    gfx.polygon(body_points, outline=body_color, fill=body_color)

    # --- wheel ---
    gfx.circle((wheel_cx, wheel_cy), wheel_r,
               outline=wheel_outline, fill=wheel_color)
    gfx.circle((wheel_cx, wheel_cy), 2,
               outline=(180, 180, 180), fill=(60, 60, 60))



    # # --- sensor head (also rotated) ---
    # head_w, head_h = 4, 4
    # head_left = right - head_w
    # head_top = body_top_y + 2
    # head_bottom = head_top + head_h
    #
    # head_pts = [
    #     rot((head_left,             head_top)),
    #     rot((head_left + head_w,    head_top)),
    #     rot((head_left + head_w,    head_bottom)),
    #     rot((head_left,             head_bottom)),
    # ]
    # gfx.polygon(head_pts, outline=accent_color, fill=accent_color)

    # # tiny eye at front of head
    # eye_x = head_left + head_w - 2
    # eye_y = head_top + 1
    # eye_pts = [
    #     rot((eye_x,       eye_y)),
    #     rot((eye_x + 2,   eye_y)),
    #     rot((eye_x + 2,   eye_y + 2)),
    #     rot((eye_x,       eye_y + 2)),
    # ]
    # gfx.polygon(eye_pts, fill=(0, 0, 0))



# --- IMES logo pictogram ---------------------------------------------

def draw_imes_logo(
    gfx: MatrixGFX,
    x: int,
    y: int,
    square_size: int = 16,
):
    """
    Simplified IMES logo for 64x32:
      - blue square top-left
      - orange square bottom-right
      - white overlap
      - 'imes' text in orange square
    (x, y) is the top-left of the blue square.
    """

    s = square_size
    overlap = 5  # size of white overlap

    blue  = (0, 80, 155)
    orange = (231, 123, 41)
    white = (255, 255, 255)

    # Blue square (top-left)
    gfx.rectangle((x, y, s, s), fill=blue)

    # Orange square (bottom-right, shifted down/right)
    ox = x + s - overlap
    oy = y + s - overlap
    gfx.rectangle((ox, oy, s, s), fill=orange)

    # White overlap square
    gfx.rectangle((ox, oy, overlap, overlap), fill=white)

    # "imes" text inside orange square near the bottom
    # text = "imes"
    # tw, th = gfx.measure_text(text)
    # # Place text flush right with a 1px margin
    # tx = ox + s - tw - 1
    # ty = oy + s - th - 1
    # gfx.text((tx, ty), text, color=white, crisp=True)


def test_pictograms():
    options = RGBMatrixOptions()
    options.hardware_mapping = 'adafruit-hat'
    options.rows = 32
    options.cols = 64
    matrix = RGBMatrix(options=options)

    gfx = MatrixGFX(matrix, default_font_size=7)

    try:
        while True:
            gfx.clear((0, 0, 0))

            # ground line
            ground_y = 30
            gfx.line((0, ground_y), (gfx.width - 1, ground_y), (60, 60, 60))

            # robot on the left
            draw_robot_pictogram(gfx, base_x=4, ground_y=ground_y, pitch=10)

            # IMES logo on the right
            draw_imes_logo(
                gfx,
                x=gfx.width - 30,   # a bit from the right edge
                y=2,
            )

            gfx.blit()
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

def animate_robot_and_logo():
    options = RGBMatrixOptions()
    options.hardware_mapping = 'adafruit-hat'
    options.rows = 32
    options.cols = 64
    matrix = RGBMatrix(options=options)

    gfx = MatrixGFX(matrix, default_font_size=7)

    WIDTH, HEIGHT = gfx.width, gfx.height
    ground_y = 30

    # --- constants that match the drawing helpers ---
    BODY_W = 10                  # body_w in draw_robot_pictogram
    SQUARE_SIZE = 16             # square_size in draw_imes_logo
    OVERLAP = 5
    LOGO_W = SQUARE_SIZE * 2 - OVERLAP
    LOGO_H = SQUARE_SIZE * 2 - OVERLAP   # not really needed, but for clarity
    LOGO_Y = 2

    # center x so that the total logo width is centered on 64 px
    LOGO_CENTER_X = (WIDTH - LOGO_W) // 2

    # animation parameters
    robot_speed = 1.0          # pixels per frame
    pitch_forward = 8.0        # robot nose-down angle when moving

    # --- state machine ---
    state = "drag_in"
    robot_x = -BODY_W          # start fully off-screen on the left
    logo_x = robot_x - (LOGO_W + 6)  # logo trails behind robot
    logo_attached = True       # robot is dragging the logo

    wait_until = 0.0

    try:
        while True:
            now = time.time()

            # --- STATE UPDATES ------------------------------------------------
            if state == "drag_in":
                robot_x += robot_speed

                # While dragging: logo stays at fixed offset behind robot
                if logo_attached:
                    logo_x = robot_x - (LOGO_W + 6)

                    # If logo would move past the center, clamp and detach
                    if logo_x >= LOGO_CENTER_X:
                        logo_x = LOGO_CENTER_X
                        logo_attached = False  # robot keeps going, logo stays
                # When robot leaves to the right, start pause with only logo visible
                if robot_x > WIDTH:
                    state = "wait_after_drag"
                    wait_until = now + 1.5  # seconds pause

            elif state == "wait_after_drag":
                # robot is off-screen, logo stays centered
                if now >= wait_until:
                    # reset robot on the left to come in and push the logo out
                    state = "push_out"
                    robot_x = -BODY_W

            elif state == "push_out":
                robot_x += robot_speed

                # Before contact, logo stays at its current place.
                # Detect when front of robot hits the left edge of the logo.
                front_x = robot_x + BODY_W
                if front_x >= logo_x:
                    # Robot is touching/pushing the logo: keep the logo right
                    # in front of the robot.
                    logo_x = front_x

                # When the logo has completely left the screen, pause and reset
                if logo_x > WIDTH:
                    state = "wait_after_push"
                    wait_until = now + 1.5

            elif state == "wait_after_push":
                if now >= wait_until:
                    # Start over: robot dragging logo in again
                    state = "drag_in"
                    robot_x = -BODY_W
                    logo_x = robot_x - (LOGO_W + 6)
                    logo_attached = True

            # --- DRAW FRAME ---------------------------------------------------
            gfx.clear((0, 0, 0))

            # ground line
            gfx.line((0, ground_y), (WIDTH - 1, ground_y), (60, 60, 60))

            # draw logo if any part of it is on the screen
            if -LOGO_W < logo_x < WIDTH:
                draw_imes_logo(gfx, int(logo_x), LOGO_Y, square_size=SQUARE_SIZE)

            # robot only appears in the active movement states
            if state in ("drag_in", "push_out"):
                draw_robot_pictogram(
                    gfx,
                    base_x=int(robot_x),
                    ground_y=ground_y,
                    pitch=pitch_forward,   # pitched slightly down while driving
                )

            gfx.blit()
            time.sleep(0.03)   # ~33 FPS

    except KeyboardInterrupt:
        pass

import math
import time
from PIL import ImageFont

from PIL import Image, ImageDraw, ImageFont


def draw_bilb_text(gfx: MatrixGFX, x: int, y: int, color=(255, 255, 255)):
    #draw_italic_text(gfx, int(x), int(y), "Bilb", color=(255, 255, 255))
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf", size=20
    )
    gfx.text((int(x), int(y)), "Bilb",font=font, color=color, crisp=False)


def animate_robot_logo_and_bilb():
    options = RGBMatrixOptions()
    options.hardware_mapping = 'adafruit-hat'
    options.rows = 32
    options.cols = 64
    matrix = RGBMatrix(options=options)

    gfx = MatrixGFX(matrix, default_font_size=18)

    WIDTH, HEIGHT = gfx.width, gfx.height
    ground_y = 30

    # --- sizes that must match your draw_* functions -------------------
    BODY_W = 10
    SQUARE_SIZE = 16
    OVERLAP = 5
    LOGO_W = SQUARE_SIZE * 2 - OVERLAP

    # logo vertical placement
    LOGO_Y = 2
    LOGO_CENTER_X = (WIDTH - LOGO_W) // 2

    # Bilb text metrics & placement
    BILB_TEXT = "Bilb"
    bilb_w, bilb_h = gfx.measure_text(BILB_TEXT)
    BILB_Y = 11  # small offset from top

    # distances between robot and the items it drags/pushes
    TRAIL_GAP = 6
    PUSH_GAP = 4          # front of robot touches object

    # speeds (px per frame)
    FORWARD_SPEED = 1.0   # left -> right
    BACKWARD_SPEED = 1.5  # right -> left
    LOGO_CENTER_SPEED = 0.5

    # positions
    robot_x = -BODY_W
    logo_x = robot_x - (LOGO_W + TRAIL_GAP)
    bilb_x = -bilb_w

    # state machine
    state = "drag_logo_in"
    wait_until = 0.0

    # where robot & logo stop together the first time
    STOP_X_LOGO = 45
    STOP_X_BILB =50

    start_time = time.time()

    def compute_pitch(forward: bool, moving: bool, t: float) -> float:
        """Dynamic pitch: bobbing while moving, flat when stopped."""
        if not moving:
            return 0.0
        base = 7.0 if forward else -9.0      # nose down/up
        osc = 2.0 * math.sin(t * 6.0)        # ~1 Hz bobbing
        return base + osc

    try:
        while True:
            now = time.time()
            t = now - start_time

            # --- STATE UPDATES ----------------------------------------
            if state == "drag_logo_in":
                # robot moving forward, dragging logo
                robot_x += FORWARD_SPEED
                logo_x = robot_x - (LOGO_W + TRAIL_GAP)

                if robot_x >= STOP_X_LOGO:
                    # stop both
                    robot_x = STOP_X_LOGO
                    logo_x = robot_x - (LOGO_W + TRAIL_GAP)
                    state = "stop_with_logo"
                    wait_until = now + 0.6

            elif state == "stop_with_logo":
                # robot & logo stand still where they are
                if now >= wait_until:
                    state = "robot_leaves_logo_center"
                # (robot_x, logo_x stay constant)

            elif state == "robot_leaves_logo_center":
                # robot drives off to the right, logo slowly moves to center
                robot_x += FORWARD_SPEED

                if logo_x < LOGO_CENTER_X:
                    logo_x += LOGO_CENTER_SPEED
                    if logo_x > LOGO_CENTER_X:
                        logo_x = LOGO_CENTER_X

                if robot_x > WIDTH:
                    state = "wait_logo_centered"
                    wait_until = now + 0.8

            elif state == "wait_logo_centered":
                # only logo centered, robot off-screen
                if now >= wait_until:
                    # reset robot to come back and push the logo
                    robot_x = -BODY_W
                    state = "push_logo_out"

            elif state == "push_logo_out":
                # robot moves forward, eventually contacting and pushing logo
                robot_x += FORWARD_SPEED

                front_x = robot_x + BODY_W
                if front_x >= logo_x - PUSH_GAP:
                    # push: keep logo right in front of robot
                    logo_x = front_x + PUSH_GAP

                if logo_x > WIDTH:
                    # logo has left screen, now robot drives backward
                    state = "robot_back_to_left"

            elif state == "robot_back_to_left":
                robot_x -= BACKWARD_SPEED

                if robot_x < -BODY_W:
                    # prepare for Bilb sequence
                    robot_x = -BODY_W
                    bilb_x = robot_x - (bilb_w + TRAIL_GAP)
                    state = "drag_bilb_in"

            elif state == "drag_bilb_in":
                # robot drags "Bilb" text in
                robot_x += FORWARD_SPEED
                bilb_x = robot_x - (bilb_w + TRAIL_GAP)

                if robot_x >= STOP_X_BILB:
                    robot_x = STOP_X_BILB
                    bilb_x = robot_x - (bilb_w + TRAIL_GAP)
                    state = "pause_with_bilb"
                    wait_until = now + 0.7

            elif state == "pause_with_bilb":
                if now >= wait_until:
                    state = "drag_bilb_out"

            elif state == "drag_bilb_out":
                # robot and Bilb leave screen together
                robot_x += FORWARD_SPEED
                bilb_x = robot_x - (bilb_w + TRAIL_GAP)
                if bilb_x > WIDTH:
                    # end of sequence: start over with logo again
                    state = "drag_logo_in"
                    robot_x = -BODY_W
                    logo_x = robot_x - (LOGO_W + TRAIL_GAP)

            # --- DRAW FRAME -------------------------------------------
            gfx.clear((0, 0, 0))

            # ground line
            gfx.line((0, ground_y), (WIDTH - 1, ground_y), (60, 60, 60))

            # Decide what to draw and what pitch to use
            if state in ("drag_logo_in", "robot_leaves_logo_center",
                         "drag_bilb_in","drag_bilb_out",
                         "push_logo_out"):
                # robot moving forward
                moving = state != "wait_logo_centered"
                pitch = compute_pitch(forward=True, moving=True, t=t)
                draw_robot_pictogram(
                    gfx,
                    base_x=int(robot_x),
                    ground_y=ground_y,
                    pitch=pitch,
                )
            elif state in ("robot_back_to_left"):
                # robot moving backward
                pitch = compute_pitch(forward=False, moving=True, t=t)
                draw_robot_pictogram(
                    gfx,
                    base_x=int(robot_x),
                    ground_y=ground_y,
                    pitch=pitch,
                )
            elif state in ("stop_with_logo", "pause_with_bilb"):
                # robot present but stationary
                draw_robot_pictogram(
                    gfx,
                    base_x=int(robot_x),
                    ground_y=ground_y,
                    pitch=0.0,
                )
            # states where robot is off-screen: nothing drawn

            # draw IMES logo during its phases
            if state in ("drag_logo_in", "stop_with_logo",
                         "robot_leaves_logo_center",
                         "wait_logo_centered",
                         "push_logo_out"):
                if -LOGO_W < logo_x < WIDTH:
                    draw_imes_logo(
                        gfx,
                        x=int(logo_x),
                        y=LOGO_Y,
                        square_size=SQUARE_SIZE,
                    )

            # draw Bilb text during its phases
            if state in ("drag_bilb_in", "pause_with_bilb", "drag_bilb_out"):
                if -bilb_w < bilb_x < WIDTH:
                    draw_bilb_text(gfx, int(bilb_x), BILB_Y)

            gfx.blit()
            time.sleep(0.03)  # ~33 FPS

    except KeyboardInterrupt:
        pass

# ---- easing helpers ----
def clamp01(t): return max(0.0, min(1.0, t))
def lerp(a, b, t): return a + (b - a) * t
def ease_in_out_quad(t):
    t = clamp01(t)
    return 2*t*t if t < 0.5 else 1 - pow(-2*t + 2, 2)/2

# ---- example robot animation ----
def animate_robot_logo_and_bilb_smooth():
    options = RGBMatrixOptions()
    options.hardware_mapping = 'adafruit-hat'
    options.rows = 32
    options.cols = 64
    matrix = RGBMatrix(options=options)
    gfx = MatrixGFX(matrix, default_font_size=7)

    WIDTH, HEIGHT = gfx.width, gfx.height
    ground_y = 30
    BODY_W = 10
    SQUARE_SIZE, OVERLAP = 16, 5
    LOGO_W = SQUARE_SIZE*2 - OVERLAP
    LOGO_Y = 2
    LOGO_CENTER_X = (WIDTH - LOGO_W)//2
    BILB_TEXT = "Bilb"
    bilb_w, _ = gfx.measure_text(BILB_TEXT)
    BILB_Y = 3
    TRAIL_GAP = 6
    PUSH_GAP = 0

    # --- reusable phase runner ---
    def run_phase(phase_name, x_start, x_end,
                  pitch_a, pitch_b,
                  logo_follow=False, logo_push=False,
                  bilb_follow=False, duration=2.0):
        """Move robot smoothly from x_start→x_end over 'duration' seconds."""
        nonlocal robot_x, logo_x, bilb_x
        t0 = time.time()
        while True:
            now = time.time()
            p = clamp01((now - t0)/duration)
            eased = ease_in_out_quad(p)
            robot_x = lerp(x_start, x_end, eased)
            pitch = lerp(pitch_a, pitch_b, eased)

            # --- dependent objects ---
            if logo_follow:
                logo_x = robot_x - (LOGO_W + TRAIL_GAP)
            if logo_push:
                front = robot_x + BODY_W
                logo_x = front + PUSH_GAP
            if bilb_follow:
                bilb_x = robot_x - (bilb_w + TRAIL_GAP)

            # --- draw frame ---
            gfx.clear((0,0,0))
            gfx.line((0,ground_y),(WIDTH-1,ground_y),(60,60,60))

            if -LOGO_W < logo_x < WIDTH:
                draw_imes_logo(gfx, int(logo_x), LOGO_Y, square_size=SQUARE_SIZE)
            if -bilb_w < bilb_x < WIDTH:
                gfx.text((int(bilb_x), BILB_Y), BILB_TEXT, color=(255,255,255), crisp=True)

            draw_robot_pictogram(gfx, int(robot_x), ground_y, pitch)
            gfx.blit()
            time.sleep(0.03)

            if p >= 1.0: break

    # ---- initial positions ----
    robot_x = -BODY_W
    logo_x  = robot_x - (LOGO_W + TRAIL_GAP)
    bilb_x  = -bilb_w

    try:
        while True:
            # 1️⃣ Drag logo in
            run_phase("drag_logo_in", -BODY_W, 20, 0, 10,
                      logo_follow=True, duration=2.2)

            # pause both
            time.sleep(0.6)

            # 2️⃣ Robot leaves, logo recenters
            run_phase("robot_leaves_logo_center", 20, WIDTH+BODY_W, 10, 5,
                      duration=2.0)
            # move logo gently to center after robot leaves
            t0 = time.time()
            while logo_x < LOGO_CENTER_X:
                logo_x += 0.5
                gfx.clear((0,0,0))
                gfx.line((0,ground_y),(WIDTH-1,ground_y),(60,60,60))
                draw_imes_logo(gfx,int(logo_x),LOGO_Y,square_size=SQUARE_SIZE)
                gfx.blit()
                time.sleep(0.03)
            time.sleep(0.8)

            # 3️⃣ Robot pushes logo off
            logo_x = LOGO_CENTER_X
            run_phase("push_logo_out", -BODY_W, WIDTH+BODY_W, 0, -10,
                      logo_push=True, duration=2.3)

            # 4️⃣ Drive backwards to left
            run_phase("robot_back_to_left", WIDTH+BODY_W, -BODY_W, -10, 5,
                      duration=2.0)

            # 5️⃣ Drag Bilb in
            bilb_x = robot_x - (bilb_w + TRAIL_GAP)
            run_phase("drag_bilb_in", -BODY_W, 18, 0, 12,
                      bilb_follow=True, duration=2.2)
            time.sleep(0.7)

            # 6️⃣ Drag Bilb out
            run_phase("drag_bilb_out", 18, WIDTH+BODY_W, 12, -5,
                      bilb_follow=True, duration=2.5)

            # loop again
            robot_x = -BODY_W
            logo_x  = robot_x - (LOGO_W + TRAIL_GAP)
            bilb_x  = -bilb_w
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    animate_robot_logo_and_bilb()