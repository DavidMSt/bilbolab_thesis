"""
Video Streamer: A Flask-based streaming server.

This script sets up a video streaming server using Flask.
It captures images from a callable image source and serves them
as an MJPEG stream.
"""

# ================================
# Package Imports
# ================================
import threading
import time
import logging
from typing import Callable

# import click

from flask import Flask, render_template, Response

# ================================
# Internal Utility Imports
# ================================
from core.utils.files import relativeToFullPath
from core.utils.network import getInterfaceIP
from core.utils.logging_utils import Logger


# # ================================
# # Disable Click Output (Suppress CLI Messages)
# # ================================
# def secho(text, file=None, nl=None, err=None, color=None, **styles):
#     """Override click.secho to suppress output."""
#     pass
#
#
# def echo(text, file=None, nl=None, err=None, color=None, **styles):
#     """Override click.echo to suppress output."""
#     pass
#
#
# click.echo = echo
# click.secho = secho

# ================================
# Logger Setup
# ================================
logger = Logger("Video Streamer")
logger.setLevel('INFO')


# ================================
# VideoStreamer Class Definition
# ================================
class VideoStreamer:
    """
    VideoStreamer class sets up a Flask-based MJPEG stream server.
    """
    _thread: threading.Thread  # Thread for running the Flask server
    image_fetcher: Callable = None  # Callable function to fetch images

    def __init__(self, image_fetcher: Callable = None):
        """Initialize the VideoStreamer class."""
        self.image_fetcher = image_fetcher
        self._thread = threading.Thread(target=self.task, daemon=True)

    def start(self):
        """Start the video streamer thread."""
        self._thread.start()

    def task(self):
        """Main task to run the Flask server and stream video frames."""
        # Initialize Flask app
        app = Flask(__name__, template_folder=relativeToFullPath('.'))

        # Suppress Flask logging output
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        # Get the server IP address
        ip = getInterfaceIP('wlan0')

        # ================================
        # Flask Route Handlers
        # ================================
        def send_frames():
            """Generator function to continuously yield video frames."""
            while True:
                if self.image_fetcher is not None:
                    image_buffer = self.image_fetcher()
                    if image_buffer is None:
                        continue
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + image_buffer + b'\r\n')
                time.sleep(0.05)

        @app.route('/')
        def index():
            """Serve the main HTML page for the video stream."""
            return render_template("index.html")

        @app.route('/video_feed')
        def video_feed():
            """Serve the MJPEG video stream."""
            return Response(send_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        # ================================
        # Start Flask Server
        # ================================
        logger.info(f"Start Video Streamer: http://{ip}:{5000}")
        app.run(debug=False, threaded=True, host=ip, use_reloader=False)
