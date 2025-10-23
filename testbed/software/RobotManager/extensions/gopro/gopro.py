import enum
import threading
import time
import json
import tempfile
from base64 import b64encode
from pathlib import Path
from flask import Flask, Response, render_template_string, stream_with_context
import cv2
import requests
import subprocess

from core.utils.exit import register_exit_callback
from core.utils.logging_utils import Logger

PREVIEW_PORT = 8554  # Default port for GoPro preview stream
WEBCAM_RTSP_PORT = 554  # Default port for GoPro webcam stream

GOPRO_MEDIA_PORT = 8080  # Default port for GoPro media access
GOPRO_MEDIA_URL_PATH = "/videos/DCIM/"


class WebcamResolution(enum.IntEnum):
    """
    Enum for GoPro webcam resolutions.
    """
    RES_480P = 4
    RES_720P = 7
    RES_1080P = 12


class WebcamFOV(enum.IntEnum):
    """
    Enum for GoPro webcam field of view (FOV).
    """
    WIDE = 0
    NARROW = 2
    SUPER_VIEW = 3
    LINEAR = 4


class WebcamProtocol(enum.Enum):
    """
    Enum for GoPro webcam protocols.
    """
    RTSP = "RTSP"
    TS = "TS"


class GoPro:
    """
    GoPro controller that can load credentials from a JSON file.
    """

    def __init__(self, credentials_file: str | Path = None, keep_alive: bool = True):
        # set up logging and exit callback
        self.logger = Logger('GoPro', 'DEBUG')
        register_exit_callback(self.stop)

        # load credentials if JSON provided
        if credentials_file:
            self._load_credentials(credentials_file)

        # defaults (in case you set address/username/etc. manually)
        self._thread: threading.Thread | None = threading.Thread(
            target=self._task, name="GoProKeepAliveThread", daemon=True
        )
        self._exit = False
        self.connected = False
        self.preview_stream_running = False

    def _load_credentials(self, credentials_file: str | Path):
        """
        Load IP, username, password, and certificate (either path or PEM text)
        from a JSON file.
        """
        cred_path = Path(credentials_file)
        data = json.loads(cred_path.read_text())

        self.address = data["ip"]
        self.username = data["username"]
        self.password = data["password"]

        cert_field = data.get("certificate")
        if cert_field:
            # if it looks like PEM text, dump to a temp file
            if cert_field.strip().startswith("-----BEGIN CERTIFICATE-----"):
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
                tf.write(cert_field.encode("utf-8"))
                tf.flush()
                tf.close()
                self.certificate = Path(tf.name)
                self.logger.debug(f"Using temporary certificate file at {self.certificate}")
            else:
                # assume it's a path
                self.certificate = Path(cert_field)
        else:
            self.certificate = None
            self.logger.warning("No certificate provided; SSL verification will use defaults.")

    # ------------------------------------------------------------------------------------------------------------------
    def start(self) -> bool:
        camera_info = self.getCameraInfo()
        if not camera_info:
            self.logger.error("❌ GoPro is not reachable. Please check the connection and credentials.")
            return False

        self._thread.start()
        self.logger.info("GoPro started")
        self.connected = True
        return True

    # ------------------------------------------------------------------------------------------------------------------
    def stop(self):
        self.logger.debug("Stopping GoPro")
        if self.connected:
            self.stopWebcam()
            self.stopPreview()
            self.stopRecording()
        self._exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self.logger.info("GoPro stopped")

    # ------------------------------------------------------------------------------------------------------------------
    def _task(self):
        while not self._exit:
            try:
                self._keepAlive()
                ...
            except Exception as e:
                self.logger.error(f"❌ Error in keep-alive task: {e}")
            time.sleep(2)

    # ------------------------------------------------------------------------------------------------------------------
    def getCameraInfo(self):
        url = "/gopro/camera/info"
        response = self._sendRequest(url, method="GET")
        if response:
            self.logger.debug(f"Camera info: {response}")
            return response
        else:
            self.logger.error("❌ Failed to retrieve camera name")
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def startPreview(self, port: int = PREVIEW_PORT):
        url = "/gopro/camera/stream/start"
        params = {"port": port}
        response = self._sendRequest(url, params=params, method="GET")
        if response is not None:
            self.logger.info(f"✅ Preview stream started on port {port}")
            self.preview_stream_running = True
            return response
        else:
            self.logger.error("❌ Failed to start preview stream")
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def stopPreview(self):
        url = "/gopro/camera/stream/stop"
        response = self._sendRequest(url, method="GET")
        if response is not None:
            self.logger.info("✅ Preview stream stopped")
            self.preview_stream_running = False
            return response
        else:
            self.logger.error("❌ Failed to stop preview stream")
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def getWebcamStatus(self):
        url = "/gopro/webcam/status"
        response = self._sendRequest(url, method="GET")
        if response:
            self.logger.info("✅ Webcam status retrieved")
            self.logger.debug(f"Webcam status: {response}")
            return response
        else:
            self.logger.error("❌ Failed to retrieve webcam status")
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def startWebcam(self,
                    resolution: WebcamResolution = WebcamResolution.RES_1080P,
                    fov: WebcamFOV = WebcamFOV.LINEAR,
                    port: int = PREVIEW_PORT,
                    protocol: WebcamProtocol = WebcamProtocol.RTSP):
        url = "/gopro/webcam/start"
        params = {
            "res": resolution.value,
            "fov": fov.value,
            "port": port,
            "protocol": protocol.value,
        }
        response = self._sendRequest(url, params=params, method="GET")
        if response:
            self.logger.info(f"✅ Webcam mode started: res={resolution.name}, fov={fov.name}, "
                             f"protocol={protocol.value}, port={port}")
            return response
        else:
            self.logger.error("❌ Failed to start webcam mode")
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def stopWebcam(self):
        url = "/gopro/webcam/exit"
        response = self._sendRequest(url, method="GET")
        if response:
            self.logger.info("✅ Webcam mode stopped")
            return response
        else:
            self.logger.error("❌ Failed to stop webcam mode")
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def hostWebcamServer(self,
                         host: str = '0.0.0.0',
                         http_port: int = 8999,
                         http_path: str = '/video',
                         protocol: WebcamProtocol = WebcamProtocol.RTSP):
        if protocol == WebcamProtocol.TS:
            stream_url = (
                f"udp://0.0.0.0:{PREVIEW_PORT}"
                "?fifo_size=1000000&overrun_nonfatal=1&listen=1"
            )
            input_opts = ["-fflags", "nobuffer", "-flags", "low_delay"]
        else:
            stream_url = f"rtsp://{self.address}:{WEBCAM_RTSP_PORT}/live"
            input_opts = ["-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay"]

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            *input_opts, "-i", stream_url,
            "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5",
            "pipe:1"
        ]
        self.logger.debug("Launching FFmpeg: " + " ".join(ffmpeg_cmd))

        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 ** 8
        )

        def _log_err(pipe):
            for line in iter(pipe.readline, b""):
                self.logger.debug("FFmpeg stderr: " + line.decode(errors="ignore").strip())

        threading.Thread(target=_log_err, args=(proc.stderr,), daemon=True).start()

        cond = threading.Condition()
        frame_data = {"jpg": None, "id": 0}

        def _reader():
            buf = b""
            while not self._exit:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    s = buf.find(b"\xff\xd8")
                    e = buf.find(b"\xff\xd9", s + 2)
                    if s == -1 or e == -1:
                        break
                    jpg = buf[s:e + 2]
                    buf = buf[e + 2:]
                    with cond:
                        frame_data["jpg"] = jpg
                        frame_data["id"] += 1
                        cond.notify_all()
            proc.kill()
            self.logger.info("FFmpeg reader stopped")

        threading.Thread(target=_reader, daemon=True).start()

        HTML = f"""
           <!doctype html>
           <html><head><title>GoPro MJPEG Proxy</title></head>
           <body style="text-align:center">
             <h1>GoPro MJPEG Proxy</h1>
             <img src="{http_path}" style="max-width:100%;">
           </body></html>
           """
        app = Flask(__name__)

        @app.route('/')
        def index():
            return render_template_string(HTML)

        @app.route(http_path)
        def video_feed():
            def gen_frames():
                boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                last_id = 0
                while not self._exit:
                    with cond:
                        cond.wait_for(lambda: frame_data["id"] > last_id or self._exit)
                        if self._exit:
                            break
                        jpg = frame_data["jpg"]
                        last_id = frame_data["id"]
                    yield boundary + jpg + b"\r\n"

            return Response(
                stream_with_context(gen_frames()),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        t = threading.Thread(
            target=lambda: app.run(host=host,
                                   port=http_port,
                                   threaded=True,
                                   use_reloader=False),
            name="GoProMJPEGServer",
            daemon=True
        )
        t.start()
        self.logger.info(f"✅ MJPEG proxy at http://{host}:{http_port}{http_path}")
        return app

    # ------------------------------------------------------------------------------------------------------------------
    def getPresets(self):
        url = "/gopro/camera/presets/get"
        response = self._sendRequest(url, method="GET")
        if response:
            self.logger.info("✅ Presets retrieved")
            presets_by_group = {}
            for group in response.get("presetGroupArray", []):
                group_id = group["id"]
                group_presets = group.get("presetArray", [])
                group_name = f"Group {group_id}"
                print(f"\n📁 {group_name}:")
                for preset in group_presets:
                    pid = preset["id"]
                    pname = preset.get("customName") or f"Preset {pid}"
                    print(f"  🔹 ID: {pid} — Name: {pname}")
                presets_by_group[group_name] = group_presets
            self.logger.debug(f"Presets: {response}")
            return response
        else:
            self.logger.error("❌ Failed to retrieve presets")
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def record(self):
        url = "/gopro/camera/shutter/start"
        restart_preview = False
        if self.preview_stream_running:
            self.logger.warning("Preview stream is running, stopping it before starting recording.")
            restart_preview = True

        response = self._sendRequest(url, method="GET", timeout=10)
        if response is not None:
            self.logger.info("✅ Recording started")
            if restart_preview:
                self.logger.info("Restarting preview stream after recording start.")
            return response
        else:
            self.logger.error("❌ Failed to start recording")
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def stopRecording(self):
        url = "/gopro/camera/shutter/stop"
        response = self._sendRequest(url, method="GET")
        if response is not None:
            self.logger.info("✅ Recording stopped")
            return response
        else:
            self.logger.error("❌ Failed to stop recording")
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def deleteAllFiles(self):
        url = "/gp/gpControl/command/storage/delete/all"
        response = self._sendRequest(url, method="GET")
        if response is not None:
            self.logger.info("✅ All files deleted from GoPro")
            return response
        else:
            self.logger.error("❌ Failed to delete all files")
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def downloadLatestMedia(self, folder, file_name=None) -> bool:
        folder_gopro, file_gopro = self.getLatestMediaInfo()
        time.sleep(1)
        if folder_gopro and file_gopro:
            return self.downloadFile(folder_gopro, file_gopro, folder, file_name)
        return False

    def downloadFile(self, folder: str, filename: str, local_path: Path | str, local_name: str | None = None) -> bool:
        url = f"https://{self.address}{GOPRO_MEDIA_URL_PATH}{folder}/{filename}"
        self.logger.info(f"Downloading {filename} from GoPro")
        if local_name is None:
            local_name = f"downloaded_{filename}"
        local_file = Path(local_path) / local_name

        headers = {
            "Authorization": "Basic " + b64encode(f"{self.username}:{self.password}".encode()).decode("ascii")
        }

        try:
            r = requests.get(url, headers=headers, verify=str(self.certificate), stream=True, timeout=30)
            r.raise_for_status()
            with open(local_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            self.logger.info(f"✅ Saved as {local_file}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to download file: {e}")
            return False

    def getLatestMediaInfo(self) -> tuple[str | None, str | None]:
        url = "/gp/gpMediaList"
        response = self._sendRequest(url, method="GET")
        if response:
            media_list = response.get('media', [])
            if media_list:
                latest_media = media_list[-1]
                folder = latest_media['d']
                file = latest_media['fs'][-1]['n']
                self.logger.info(f"Latest media: Folder={folder}, File={file}")
                return folder, file
            self.logger.error("❌ No media found")
        return None, None

    # ------------------------------------------------------------------------------------------------------------------
    def _keepAlive(self):
        url = "/gopro/camera/keep_alive"
        return self._sendRequest(url, method="GET")

    # ------------------------------------------------------------------------------------------------------------------
    def _sendRequest(self, url, params=None, data=None, method="GET", timeout=5):
        BASE_URL = f"https://{self.address}"
        HEADERS = {
            "Authorization": "Basic " + b64encode(f"{self.username}:{self.password}".encode()).decode("ascii")
        }
        if not url.startswith("/"):
            url = f"/{url}"

        if method == "GET":
            try:
                r = requests.get(
                    f"{BASE_URL}{url}",
                    headers=HEADERS,
                    params=params,
                    verify=str(self.certificate),
                    timeout=timeout
                )
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                self.logger.error(f"HTTP error: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                self.logger.error(f"Error during GET request: {e}")
        return None


def main():
    # loads gopro_credentials.json from cwd
    gopro = GoPro(credentials_file="gopro_credentials.json")
    if not gopro.start():
        return

    time.sleep(3)
    gopro.startWebcam()
    time.sleep(1)
    gopro.hostWebcamServer(protocol=WebcamProtocol.RTSP)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        gopro.stop()


if __name__ == '__main__':
    main()
