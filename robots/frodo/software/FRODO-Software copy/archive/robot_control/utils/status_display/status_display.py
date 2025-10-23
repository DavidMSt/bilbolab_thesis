import socket
import threading
import board
import time
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
from subprocess import check_output
import os
import re


class Display:
    _thread: threading.Thread
    width: int
    height: int
    address: int

    i2c: board.I2C
    display: adafruit_ssd1306.SSD1306_I2C

    border_width = 0
    skip = 2
    display_found: bool

    lines: list
    rotation: int

    def __init__(self, width: int = 128, height: int = 32, address=0x3C, rotation=0):
        self.width = width
        self.height = height
        self.address = address
        self.rotation = rotation

        self.i2c = board.I2C()  # uses board.SCL and board.SDA

        self.display_found = False

        self.lines = ['Line 1', 'Line 2', 'Line 3']

        # self.font = ImageFont.load_default()
        self.font = ImageFont.truetype("DejaVuSansMono.ttf", 10)

        self._thread = threading.Thread(target=self._threadFunction, daemon=False)

    def init(self):
        ...

    def start(self):
        self._thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def _threadFunction(self):

        while True:

            # Search the Display
            if not self.display_found:
                self.display = self._searchDisplay()

                if self.display is not None:
                    self.display_found = True
                    print("Display Found")
                else:
                    time.sleep(1)

            if self.display_found:
                self._updateDisplay()
                time.sleep(0.1)

    # ------------------------------------------------------------------------------------------------------------------
    def _searchDisplay(self):
        try:
            display = adafruit_ssd1306.SSD1306_I2C(self.width, self.height, self.i2c, addr=self.address)
            display.rotate(self.rotation)
        except Exception:
            display = None

        return display

    # ------------------------------------------------------------------------------------------------------------------
    def _clearDisplay(self):
        self.display.fill(0)
        self.display.show()

    # ------------------------------------------------------------------------------------------------------------------
    def _writeLine(self, line_number: int, text: str):
        self.lines[line_number] = text

    # ------------------------------------------------------------------------------------------------------------------
    def _updateDisplay(self):
        image = Image.new("1", (self.width, self.height))

        # Get drawing object to draw on image.
        draw = ImageDraw.Draw(image)

        # Draw a white background
        draw.rectangle((0, 0, self.width, self.height), outline=255, fill=255)

        # Draw a smaller inner rectangle
        draw.rectangle(
            (self.border_width, self.border_width, self.width - self.border_width - 1,
             self.height - self.border_width - 1),
            outline=0,
            fill=0,
        )

        # Draw Some Text
        text = "Hello World!"
        bbox = self.font.getbbox(text)
        (font_width, font_height) = bbox[2] - bbox[0], bbox[3] - bbox[1]

        draw.text(
            (5, self.skip),
            self.lines[0],
            font=self.font,
            fill=255,
        )

        draw.text(
            (5, font_height + 2 * self.skip),
            self.lines[1],
            font=self.font,
            fill=255,
        )

        draw.text(
            (5, 2 * font_height + 3 * self.skip),
            self.lines[2],
            font=self.font,
            fill=255,
        )

        # Display image
        try:
            self.display.image(image)
            self.display.show()
        except Exception:
            self.display_found = False
            print("Lost Display")


class ConnectionStatusDisplay(Display):
    _connectionThread: threading.Thread

    def __init__(self, width: int = 128, height: int = 32, address=0x3C, rotation=0):
        super().__init__(width, height, address, rotation)

        self._connectionThread = threading.Thread(target=self._connectionThreadFunction, daemon=False)

    def start(self):
        self.lines[0] = ''
        self.lines[1] = ''
        self.lines[2] = ''
        super().start()
        self._connectionThread.start()

    def _connectionThreadFunction(self):
        global state
        while True:
            username, hostname, ssid, local_ip = self._getNetworkInformation()

            self.font = ImageFont.truetype("DejaVuSansMono.ttf", 10)
            if hostname is not None:
                self.lines[0] = f"{username}@{hostname}"
            else:
                self.lines[0] = ''

            if ssid is not None:
                self.lines[1] = f"SSID: {ssid}"
            else:
                self.lines[1] = ''

            if local_ip is not None:
                self.lines[2] = f"IP: {local_ip}"
            else:
                self.lines[2] = ''

            # if hostname is None or ssid is None or local_ip is None:
            #     self._setNotConnectedScreen()

            time.sleep(0.5)

    def _setNotConnectedScreen(self):
        self.font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
        self.lines[0] = ' Not Connected'
        self.lines[1] = ''
        self.lines[2] = ''

    def _getNetworkInformation(self):

        try:
            # username = os.environ['USER']
            usernames = os.listdir('/home/')
            username = usernames[0]
        except Exception:
            username = None

        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = None

        try:
            ssid = check_output(['/sbin/iwgetid', '-r']).decode().rstrip()
            if ssid == '':
                ssid = None
        except Exception:
            ssid = None

        try:
            ip_string = check_output(['hostname', '-I']).decode()
            ips = re.findall('[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', ip_string)
            local_ip = re.findall('192\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', ip_string)

            if local_ip:
                local_ip = local_ip[0]
            else:
                local_ip = None

        except Exception:
            local_ip = None

        return username, hostname, ssid, local_ip


def run_status_display():
    display = ConnectionStatusDisplay()
    display.start()

    while True:
        time.sleep(10)


if __name__ == '__main__':
    run_status_display()
