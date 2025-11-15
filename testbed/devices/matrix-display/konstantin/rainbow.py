import time
import math
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# Smooth animated rainbow across the 64x32 matrix
if __name__ == "__main__":
    options = RGBMatrixOptions()
    options.hardware_mapping = "adafruit-hat"  # or "regular" depending on your setup
    options.rows = 32
    options.cols = 64
    options.chain_length = 1
    options.parallel = 1
    options.brightness = 75  # reduce if too bright

    matrix = RGBMatrix(options=options)

    print("Press Ctrl+C to exit")

    t = 0.0
    try:
        while True:
            for y in range(32):
                for x in range(64):
                    # Create a rainbow gradient using sine waves for R, G, B
                    r = int((math.sin((x + t) * 0.15) * 127) + 128)
                    g = int((math.sin((x + t) * 0.15 + 2 * math.pi / 3) * 127) + 128)
                    b = int((math.sin((x + t) * 0.15 + 4 * math.pi / 3) * 127) + 128)
                    matrix.SetPixel(x, y, r, g, b)

            t += 2  # controls the rainbow scroll speed
            time.sleep(0.01)  # controls animation frame rate
    except KeyboardInterrupt:
        matrix.Clear()
        print("Exiting cleanly.")