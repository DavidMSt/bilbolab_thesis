#!/usr/bin/env python3
# test_ssd1306.py

from time import sleep
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw, ImageFont

def main():
    # Initialize I²C (bus=1, addr=0x3C)
    serial = i2c(port=1, address=0x3C)
    device = ssd1306(serial, width=128, height=32)

    # Clear
    device.clear()
    device.show()

    # Blank image buffer
    image = Image.new("1", (device.width, device.height))
    draw  = ImageDraw.Draw(image)

    # Border
    draw.rectangle([0, 0, device.width-1, device.height-1], outline=255, fill=0)

    # Choose a font (TTF if present, else default)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    text = "Hello!"

    # Measure text size robustly
    try:
        # Pillow ≥8.0
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width  = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback: render to mask
        mask = font.getmask(text)
        text_width, text_height = mask.size

    # Center
    x = (device.width  - text_width ) // 2
    y = (device.height - text_height) // 2

    # Draw and display
    draw.text((x, y), text, font=font, fill=255)
    device.display(image)

    sleep(5)

if __name__ == "__main__":
    main()
