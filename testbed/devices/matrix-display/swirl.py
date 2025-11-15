import time
import math
import random
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# ----------------------------
# SUPER-FAST Rainbow Flow Swirl
# ----------------------------

# One knob for overall motion speed (try 3, 6, 9… go wild)
SPEED = 6.0   # higher = faster motion

def hsv_to_rgb(h, s, v):
    h = h % 1.0
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if   i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else:        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)

def hash2(ix, iy, seed):
    n = (ix * 374761393 + iy * 668265263 + seed * 83492791) & 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    n = (n ^ (n >> 16)) & 0xFFFFFFFF
    return (n / 0xFFFFFFFF)

def smoothstep(t):
    return t * t * (3 - 2 * t)

def value_noise(x, y, seed=0):
    xi = math.floor(x); yi = math.floor(y)
    xf = x - xi;        yf = y - yi
    v00 = hash2(xi,     yi,     seed)
    v10 = hash2(xi + 1, yi,     seed)
    v01 = hash2(xi,     yi + 1, seed)
    v11 = hash2(xi + 1, yi + 1, seed)
    u = smoothstep(xf); v = smoothstep(yf)
    a = v00 * (1 - u) + v10 * u
    b = v01 * (1 - u) + v11 * u
    return a * (1 - v) + b * v

if __name__ == "__main__":
    WIDTH, HEIGHT = 64, 32
    options = RGBMatrixOptions()
    options.hardware_mapping = "adafruit-hat"
    options.rows = HEIGHT
    options.cols = WIDTH
    options.chain_length = 1
    options.parallel = 1
    options.brightness = 80
    options.gpio_slowdown = 2   # adjust if you see tearing

    matrix = RGBMatrix(options=options)
    canvas = matrix.CreateFrameCanvas()

    # --- Style & speed (boosted) ---
    swirl_strength    = 1.2
    swirl_freq        = 0.035
    swirl_time_scale  = 0.16 * SPEED   # ↑ faster swirl wobble

    hue_scale         = 0.12
    hue_time_scroll   = 0.06 * SPEED   # ↑ faster hue scroll

    noise_scale       = 0.08
    noise_time        = 0.12 * SPEED   # ↑ faster fluid drift
    noise_amount      = 0.55
    noise_seed        = random.randint(0, 10_000)

    pulse_speed       = 0.15 * SPEED   # ↑ faster subtle pulsing
    pulse_amount      = 0.06

    gamma = 2.2

    cx = (WIDTH - 1) * 0.5
    cy = (HEIGHT - 1) * 0.5
    inv_w = 1.0 / max(1, WIDTH - 1)
    inv_h = 1.0 / max(1, HEIGHT - 1)

    print("FAST rainbow swirl running… Ctrl+C to quit.")
    t0 = time.time()

    try:
        while True:
            t = time.time() - t0

            pulse = 1.0 - pulse_amount + pulse_amount * (math.sin(t * 2 * math.pi * pulse_speed) * 0.5 + 0.5)
            swirl_wiggle = math.sin(t * 2 * math.pi * swirl_time_scale) * 0.8

            for y in range(HEIGHT):
                # precompute normalized y terms once per row (tiny speed win)
                ny = (y - cy) * inv_h * 2.0
                for x in range(WIDTH):
                    nx = (x - cx) * inv_w * 2.0

                    r = math.hypot(nx, ny) + 1e-6
                    base_angle = math.atan2(ny, nx)

                    swirl = base_angle + swirl_strength * (1.0 / (1.0 + 3.0 * r)) \
                            + math.sin((r / max(1e-6, swirl_freq)) + swirl_wiggle) * 0.25

                    n = value_noise(x * noise_scale + t * noise_time,
                                    y * noise_scale - t * 0.5 * noise_time,
                                    noise_seed)
                    n = (n - 0.5) * 2.0

                    hue = (swirl * hue_scale) + (n * noise_amount) + (t * hue_time_scroll)

                    sat = 0.9 - 0.25 * min(1.0, r)
                    val = pulse * (0.9 - 0.25 * (r * 0.8))

                    r8, g8, b8 = hsv_to_rgb(hue, max(0.0, min(1.0, sat)), max(0.0, min(1.0, val)))
                    r8 = int((r8 / 255.0) ** (1.0 / gamma) * 255 + 0.5)
                    g8 = int((g8 / 255.0) ** (1.0 / gamma) * 255 + 0.5)
                    b8 = int((b8 / 255.0) ** (1.0 / gamma) * 255 + 0.5)

                    canvas.SetPixel(x, y, r8, g8, b8)

            canvas = matrix.SwapOnVSync(canvas)

            # Shorter sleep = faster visual motion AND higher FPS
            # If your Pi struggles, bump to 0.008–0.012
            time.sleep(0.004)  # ~250 FPS target in loop pacing; panel limits apply

    except KeyboardInterrupt:
        matrix.Clear()
        print("Exiting cleanly.")