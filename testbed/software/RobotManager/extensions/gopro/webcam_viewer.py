#!/usr/bin/env python3
# stream_viewer.py

import argparse
from flask import Flask, Response, render_template_string
import cv2

HTML = """
<!doctype html>
<html>
  <head><title>Stream Viewer</title></head>
  <body style="text-align:center">
    <h1>Stream Viewer</h1>
    <img src="{{ url_for('video_feed') }}" style="max-width:100%;">
  </body>
</html>
"""


def gen_frames(src_url):
    cap = cv2.VideoCapture(src_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {src_url}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        jpg = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
    cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Proxy an RTSP/TS stream into MJPEG for the browser"
    )
    parser.add_argument(
        '--source', '-s',
        required=True,
        help="RTSP URL (e.g. rtsp://192.168.8.196:554/live) or TS URL (e.g. udp://0.0.0.0:8555)"
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8999,
        help="HTTP port to serve on (default: 8999)"
    )
    args = parser.parse_args()

    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(HTML)

    @app.route('/video_feed')
    def video_feed():
        return Response(gen_frames(args.source),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == '__main__':
    main()
