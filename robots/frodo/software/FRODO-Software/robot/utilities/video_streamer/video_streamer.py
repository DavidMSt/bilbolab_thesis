"""
Video Streamer: Flask MJPEG server with drop-frame multi-client broadcasting + max_clients.

- One capture thread calls image_fetcher() once per frame.
- A broadcaster pushes each encoded MJPEG part to per-client queues (maxsize=1).
  Slow clients drop older frames and always display the newest available frame.
- Routes:
    * /video   -> raw MJPEG stream
    * /preview -> simple centered HTML viewer
    * /        -> redirects to /preview
- New:
    * max_clients: hard cap on concurrent MJPEG subscribers.
      Over-capacity requests receive a small static HTML page (no stream).
    * client-id preemption: refreshing the same client id immediately replaces the
      previous connection so you don't lose the stream at max_clients=1.
"""

# ================================
# Package Imports
# ================================
import threading
import time
import logging
import queue
from dataclasses import dataclass
from typing import Callable, Optional, Dict

from flask import Flask, Response, redirect, render_template_string, request

# ================================
# Internal Utility Imports
# ================================
from core.utils.files import relativeToFullPath
from core.utils.network import getInterfaceIP
from core.utils.logging_utils import Logger

# ================================
# Logger Setup
# ================================
logger = Logger("Video Streamer")
logger.setLevel("INFO")


# ================================
# Data model for a subscriber
# ================================
@dataclass
class Subscriber:
    q: "queue.Queue[Optional[bytes]]"
    created_at: float
    remote_addr: str
    user_agent: str
    cid: str  # stable client id


# ================================
# VideoStreamer Class Definition
# ================================
class VideoStreamer:
    """
    Flask-based MJPEG stream server with single-producer / multi-consumer (drop-frame) design.

    image_fetcher:        Callable that returns JPEG bytes (bytes-like) or None if not available.
    fetch_hz:             Target capture frequency for the background fetch loop.
    max_clients:          Maximum concurrent MJPEG subscribers allowed. New connections over the cap
                          get a static "max clients reached" page and are NOT subscribed.
    admission_grace_ms:   Small wait used as a backstop; main fix is preemption by client id.
    """

    image_fetcher: Optional[Callable]
    fetch_hz: float
    max_clients: int

    # Threads & sync
    _server_thread: threading.Thread
    _capture_thread: threading.Thread
    _stop_event: threading.Event

    # Latest frame state
    _latest_jpeg: Optional[bytes]
    _latest_mjpeg_part: Optional[bytes]  # prebuilt "--frame\r\n...JPEG...\r\n"
    _frame_counter: int
    _state_lock: threading.Lock

    # Subscribers keyed by client id
    _subs_lock: threading.Lock
    _subs_by_cid: Dict[str, Subscriber]

    # Admission grace (seconds) as a small safety net
    admission_grace_s: float

    def __init__(
            self,
            image_fetcher: Optional[Callable] = None,
            fetch_hz: float = 10.0,
            max_clients: int = 8,
            admission_grace_ms: int = 250,
            port=5000,
    ):
        self.port = port
        self.image_fetcher = image_fetcher
        self.fetch_hz = max(0.5, float(fetch_hz))
        self.max_clients = int(max(1, max_clients))
        self.admission_grace_s = max(0.0, float(admission_grace_ms) / 1000.0)

        # Latest frame state
        self._latest_jpeg = None
        self._latest_mjpeg_part = None
        self._frame_counter = 0
        self._state_lock = threading.Lock()

        # Subscriber map
        self._subs_lock = threading.Lock()
        self._subs_by_cid = {}

        # Control & threads
        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)

    # ----------------------------
    # Public API
    # ----------------------------
    def start(self):
        """Start capture + server threads."""
        logger.info("Starting VideoStreamer...")
        if not self._capture_thread.is_alive():
            self._capture_thread.start()
        if not self._server_thread.is_alive():
            self._server_thread.start()

    def stop(self):
        """Signal threads to stop (Flask will stop when process exits)."""
        self._stop_event.set()

    # ----------------------------
    # Internals: Subscription model
    # ----------------------------
    def _subscriber_count(self) -> int:
        with self._subs_lock:
            return len(self._subs_by_cid)

    def _has_capacity(self) -> bool:
        with self._subs_lock:
            return len(self._subs_by_cid) < self.max_clients

    def _subscribe_locked(self, cid: str, remote_addr: str, user_agent: str) -> Subscriber:
        """
        INTERNAL: Call with _subs_lock held and ONLY when capacity is available.
        Creates and registers a new Subscriber for cid.
        """
        q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=1)
        sub = Subscriber(q=q, created_at=time.time(), remote_addr=remote_addr, user_agent=user_agent, cid=cid)
        self._subs_by_cid[cid] = sub

        # Seed with the latest frame so first render appears fast
        with self._state_lock:
            part = self._latest_mjpeg_part
        if part:
            try:
                q.put_nowait(part)
            except queue.Full:
                pass
        return sub

    def _unsubscribe(self, cid: str):
        with self._subs_lock:
            self._subs_by_cid.pop(cid, None)

    def _preempt(self, cid: str) -> bool:
        """
        If there is an existing subscriber for this cid, enqueue a sentinel (None) to
        make its generator exit and unsubscribe. Returns True if a preemption signal
        was sent, False if nothing to preempt.
        """
        with self._subs_lock:
            sub = self._subs_by_cid.get(cid)
            if not sub:
                return False
            # Try to inject a sentinel, replacing any queued part
            try:
                sub.q.put_nowait(None)  # None signals stop
            except queue.Full:
                try:
                    sub.q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    sub.q.put_nowait(None)
                except queue.Full:
                    # If still full, skip; consumer will eventually drain
                    pass
            return True

    def _broadcast(self, part: bytes):
        """
        Deliver the new MJPEG part to every subscriber.
        If a subscriber queue is full, drop the old item and replace with the newest.
        """
        with self._subs_lock:
            subs_snapshot = list(self._subs_by_cid.values())
        for sub in subs_snapshot:
            q = sub.q
            try:
                q.put_nowait(part)
            except queue.Full:
                # Drop the stale frame and insert the newest
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(part)
                except queue.Full:
                    pass  # extremely slow client; skip this round

    def _wait_for(self, predicate, timeout_s: float, poll_interval_s: float = 0.01) -> bool:
        """Generic tiny wait helper."""
        if timeout_s <= 0:
            return bool(predicate())
        deadline = time.time() + timeout_s
        while not self._stop_event.is_set():
            if predicate():
                return True
            now = time.time()
            if now >= deadline:
                break
            time.sleep(min(poll_interval_s, max(0.0, deadline - now)))
        return bool(predicate())

    # ----------------------------
    # Internals: Capture loop
    # ----------------------------
    def _capture_loop(self):
        """Background thread: fetch frames and broadcast as MJPEG parts."""
        period = 1.0 / self.fetch_hz if self.fetch_hz > 0 else 0.05
        logger.info(f"Capture loop at ~{self.fetch_hz:.1f} FPS.")

        boundary_header = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        boundary_footer = b"\r\n"

        while not self._stop_event.is_set():
            start_t = time.time()
            try:
                buf = None
                if self.image_fetcher is not None:
                    buf = self.image_fetcher()
                if buf:
                    # Build the MJPEG part ONCE per frame (saves per-client concatenations)
                    part = boundary_header + buf + boundary_footer

                    # Update latest and notify subscribers
                    with self._state_lock:
                        self._latest_jpeg = buf
                        self._latest_mjpeg_part = part
                        self._frame_counter += 1

                    self._broadcast(part)
                else:
                    # No frame; avoid tight loop
                    time.sleep(0.001)
            except Exception as ex:
                logger.error(f"Error in image_fetcher: {ex}")
                time.sleep(0.01)

            # Pace capture
            elapsed = time.time() - start_t
            sleep_for = max(0.0, period - elapsed)
            if sleep_for:
                time.sleep(sleep_for)

        logger.info("Capture loop stopped.")

    # ----------------------------
    # Internals: Flask server
    # ----------------------------
    def _run_server(self):
        app = Flask(__name__, template_folder=relativeToFullPath("."))

        # Quiet werkzeug logs
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        ip = getInterfaceIP("wlan0")

        OVER_CAPACITY_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360">
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#111"/>
              <stop offset="1" stop-color="#222"/>
            </linearGradient>
          </defs>
          <rect width="100%" height="100%" fill="url(#g)"/>
          <g font-family="system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans'" text-anchor="middle">
            <text x="320" y="150" font-size="28" fill="#fff" opacity="0.95">Maximum number of clients reached</text>
            <text x="320" y="190" font-size="16" fill="#ddd" opacity="0.8">This is a static image; no video data is being sent.</text>
          </g>
          <rect x="20" y="20" width="600" height="320" fill="none" stroke="#444"/>
        </svg>""".encode("utf-8")

        OVER_CAPACITY_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Maximum Clients Reached</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { color-scheme: light dark; }
    body {
      margin: 0; min-height: 100vh; display: grid; place-items: center;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial, "Noto Sans";
      background: #111; color: #eee;
    }
    .card {
      max-width: 680px; padding: 18px 20px; border-radius: 16px;
      background: rgba(255,255,255,0.05); box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    }
    h1 { margin: 0 0 10px; font-size: 1.4rem; }
    p { margin: 6px 0; line-height: 1.45; opacity: 0.9; }
    code { padding: 2px 6px; background: rgba(255,255,255,0.08); border-radius: 8px; }
    .meta { margin-top: 10px; font-size: 0.9rem; opacity: 0.7; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Maximum number of clients reached</h1>
    <p>The live stream is currently at capacity. Please try again later.</p>
    <p class="meta">This is a static page; no video data is being sent.</p>
  </div>
</body>
</html>
"""

        def _make_mjpeg_stream(cid: str, q: "queue.Queue[Optional[bytes]]"):
            """Per-client generator consuming from its own queue."""
            try:
                while True:
                    part = q.get()
                    if part is None:  # sentinel -> stop
                        break
                    yield part
            except GeneratorExit:
                pass
            except Exception as ex:
                logger.error(f"Client stream error ({cid}): {ex}")
            finally:
                self._unsubscribe(cid)

        @app.route("/")
        def root():
            return redirect("/preview", code=302)

        @app.route("/preview")
        def preview():
            # Tiny viewer that injects a stable client id and points <img> to /video?cid=...
            html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Video Preview</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{ color-scheme: light dark; }}
    body {{
      margin: 0; min-height: 100vh; display: grid; place-items: center;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial, "Noto Sans";
      background: #111; color: #eee;
    }}
    .frame {{
      display: grid; place-items: center; max-width: 92vw; max-height: 86vh;
      padding: 12px; border-radius: 16px; background: rgba(255,255,255,0.05);
      box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    }}
    img {{ display: block; max-width: 90vw; max-height: 80vh; border-radius: 12px; background: #000; }}
    .meta {{ margin-top: 8px; font-size: 0.9rem; opacity: 0.8; text-align: center; }}
    a {{ color: #9ad1ff; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="frame">
    <img id="stream" alt="Live Stream" />
    <div class="meta">
      Serving <code>/video</code> (MJPEG). Open raw stream: <a id="raw" href="#">/video</a><br/>
      Clients: {self._subscriber_count()} / {self.max_clients}
    </div>
  </div>
  <script>
    (function() {{
      const key = "mjpeg_client_id";
      let cid = localStorage.getItem(key);
      if (!cid) {{
        if (window.crypto && crypto.randomUUID) {{
          cid = crypto.randomUUID();
        }} else {{
          cid = Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
        }}
        localStorage.setItem(key, cid);
      }}
      const url = "/video?cid=" + encodeURIComponent(cid);
      document.getElementById("stream").src = url;
      document.getElementById("raw").href = url;
      document.getElementById("raw").textContent = url;
    }})();
  </script>
  <noscript>
    <div style="margin-top:10px;opacity:0.75">
      JavaScript is recommended for seamless refresh behavior.
      <a href="/video">Open /video directly</a>
    </div>
  </noscript>
</body>
</html>
"""
            return render_template_string(html)

        @app.route("/video")
        def video():
            # Determine a stable client ID.
            cid = request.args.get("cid")
            rip = request.remote_addr or "?"
            ua = request.headers.get("User-Agent", "") or ""
            if not cid:
                # Fallback: fingerprint (works for raw /video without JS).
                cid = f"{rip}|{ua}"

            # If the same client connects again, preempt the old connection immediately.
            if self._preempt(cid):
                logger.info(f"Preempted existing stream for cid={cid} to hand off slot on refresh.")

            # Ensure capacity (after potential preemption)
            def _capacity_ok():
                with self._subs_lock:
                    return len(self._subs_by_cid) < self.max_clients

            if not _capacity_ok():
                # Small backstop wait (usually unnecessary with preemption)
                if not self._wait_for(_capacity_ok, timeout_s=self.admission_grace_s):
                    logger.info(f"Over-capacity from {rip}; serving static SVG image.")
                    resp = Response(OVER_CAPACITY_SVG, mimetype="image/svg+xml")
                    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                    resp.headers["Pragma"] = "no-cache"
                    resp.headers["Expires"] = "0"
                    return resp

            # Atomically subscribe
            with self._subs_lock:
                if len(self._subs_by_cid) >= self.max_clients:
                    # Another request won the race for the last slot.
                    logger.info(f"Admission race lost for cid={cid}; serving static SVG image.")
                    resp = Response(OVER_CAPACITY_SVG, mimetype="image/svg+xml")
                    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                    resp.headers["Pragma"] = "no-cache"
                    resp.headers["Expires"] = "0"
                    return resp
                sub = self._subscribe_locked(cid=cid, remote_addr=rip, user_agent=ua)

            resp = Response(
                _make_mjpeg_stream(cid, sub.q),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                direct_passthrough=True,
            )
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            resp.headers["X-Accel-Buffering"] = "no"
            return resp

        logger.info(
            f"Start Video Streamer: http://{ip}:{self.port} "
            f"(max_clients={self.max_clients}, preemption=on, admission_grace_ms={int(self.admission_grace_s * 1000)})"
        )
        app.run(debug=False, threaded=True, host=ip, port=self.port, use_reloader=False)
