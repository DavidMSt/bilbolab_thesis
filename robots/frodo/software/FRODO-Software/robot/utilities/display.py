import time
from time import sleep
import socket
import re
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw, ImageFont


def get_hostname_text():
    # Get the hostname of the Raspberry Pi
    hostname = socket.gethostname()

    # Insert a space between letters and trailing numbers (e.g., frodo3 -> frodo 3)
    formatted = re.sub(r"([a-zA-Z]+)(\d+)$", r"\1 \2", hostname)

    # Convert to uppercase
    return formatted.upper()


def main():
    serial = i2c(port=1, address=0x3C)
    device = ssd1306(serial, width=128, height=32)

    device.clear()
    device.show()

    # Blank image buffer
    image = Image.new("1", (device.width, device.height))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except IOError:
        font = ImageFont.load_default()

    text = get_hostname_text()

    # Get text bounding box (x0, y0, x1, y1)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center horizontally and vertically
    x = (device.width - text_width) // 2
    y = (device.height - text_height) // 2

    draw.text((x, y), text, font=font, fill=255)
    device.display(image)

    while True:
        time.sleep(100)

if __name__ == '__main__':
    main()