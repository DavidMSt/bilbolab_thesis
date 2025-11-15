import time
from random import randrange
from rgbmatrix import RGBMatrix, RGBMatrixOptions


if __name__ == '__main__':

    options = RGBMatrixOptions()

    options.hardware_mapping = 'adafruit-hat'
    options.rows = 32
    options.cols = 64

    matrix = RGBMatrix(options=options)

    while True:
        matrix.Clear()
        for _ in range(5000):
            r = randrange(256)
            g = randrange(256)
            b = randrange(256)
            x = randrange(64)
            y = randrange(32)

            matrix.SetPixel(x, y, r, g, b)

            time.sleep(0.001)