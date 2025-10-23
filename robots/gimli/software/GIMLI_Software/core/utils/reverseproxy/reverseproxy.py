import asyncio
import os
import json
import signal
from typing import Dict, Optional
from dataclasses import dataclass

from aiohttp import (
    web,
    ClientSession,
    ClientConnectionError,
    WSMsgType,
    WSCloseCode,
)
from yarl import URL

# =========================
# Models & small utilities
# =========================

@dataclass
class Target:
    port: int
    base_path: str = ""       # e.g. "/app"
    host: str = "127.0.0.1"   # upstream host or IP (can be LAN IP or another machine)
    scheme: str = "http"      # "http" or "https"

def _clean_path(p: str) -> str:
    if not p:
        return ""
    s = "/" + str(p).strip().strip("/")
    return "" if s == "/" else s

def _join_paths(a: str, b: str) -> str:
    # joins two URL paths without double slashes; keeps "/" for root
    a = _clean_path(a)
    if not a:
        a = ""
    b = b or ""
    out = (a + "/" + b.lstrip("/")).rstrip("/")
    return out or "/"

def _is_ws(request: web.Request) -> bool:
    return request.headers.get("Upgrade", "").lower() == "websocket"

def _public_host_from_request(request: web.Request) -> str:
    # Return "host[:port]" as sent by the client (no scheme)
    return request.headers.get("Host", "")

def _rewrite_location(location: str, target: "Target", public_host: str) -> str:
    """
    Rewrite absolute upstream redirects back to the public host while
    respecting base_path. Leaves relative URLs alone.
    """
    try:
        u = URL(location)
    except Exception:
        return location  # not a URL we understand, leave as-is

    # Only consider absolute URLs (with host) that point to the configured upstream
    if u.host is None:
        return location

    # Compare against upstream host/port (normalize)
    upstream_host = target.host
    upstream_port = target.port
    if (u.host == upstream_host) and ((u.port or (443 if u.scheme == "https" else 80)) == upstream_port):
        # If upstream path starts with base_path, strip it so clients see clean public paths
        base = target.base_path or ""
        new_path = str(u.path)
        if base and new_path.startswith(base):
            new_path = new_path[len(base):] or "/"

        # Build a URL pointing to the public host (keep path/query/fragment)
        # We do not specify scheme here; browser will keep the current page's scheme.
        # If you terminate TLS here, this preserves https in the browser.
        try:
            # yarl requires a scheme for absolute URLs; we create a schemeless-ish by using current scheme hint.
            # We'll default to http; browsers will upgrade if current page was https.
            rebuilt = URL.build(
                scheme="http",
                host=public_host.split(":")[0],
                port=int(public_host.split(":")[1]) if ":" in public_host else None,
                path=new_path,
                query=u.query,
                fragment=u.fragment,
            )
            return str(rebuilt)
        except Exception:
            # Fallback: crude replacement
            host_only = public_host
            return f"http://{host_only}{new_path}{('?' + str(u.query)) if u.query_string else ''}{('#' + u.fragment) if u.fragment else ''}"

    return location


# =========================
# Static routes
# =========================

# EXAMPLE: app.local → http(s)://<UPSTREAM_HOST>:8400/app
# Put your real upstream host/IP in "host" if not local.
STATIC_HOSTNAME_TO_TARGET: Dict[str, Target] = {
    # "robotarm.local": Target(port=8400, base_path="/app", host="127.0.0.1", scheme="http"),
    "robotarm.local":      Target(port=8400, base_path="/app", host="robotarm.local", scheme="http"),
}



# =========================
# Persistence
# =========================

ROUTES_FILE = os.path.join(os.path.dirname(__file__), "routes.json")

# Dynamic routes in memory (overridable via API)
DYNAMIC_HOSTNAME_TO_TARGET: Dict[str, Target] = {}
HOSTS_LOCK = asyncio.Lock()


# =========================
# Validation helpers
# =========================

def is_valid_hostname(hostname: str) -> bool:
    return bool(hostname and "." in hostname and " " not in hostname)

def is_valid_port(port: int) -> bool:
    try:
        p = int(port)
        return 1 <= p <= 65535
    except Exception:
        return False

def is_valid_base_path(base_path: str) -> bool:
    return isinstance(base_path, str) and not base_path.strip().startswith(("http://", "https://"))

def is_valid_scheme(s: str) -> bool:
    return s in ("http", "https")

def is_valid_host(h: str) -> bool:
    return bool(h) and "://" not in h and " " not in h


# =========================
# Route getters / persistence
# =========================

async def get_target_for_host(hostname: str) -> Optional[Target]:
    async with HOSTS_LOCK:
        return (
            DYNAMIC_HOSTNAME_TO_TARGET.get(hostname)
            or STATIC_HOSTNAME_TO_TARGET.get(hostname)
        )

async def load_routes_from_disk():
    if os.path.exists(ROUTES_FILE):
        try:
            with open(ROUTES_FILE, "r") as f:
                data = json.load(f)

            converted: Dict[str, Target] = {}
            for host, val in data.items():
                if isinstance(val, int):
                    # Back-compat: old form { "host": 8400 }
                    converted[host] = Target(port=int(val), base_path="", host="127.0.0.1", scheme="http")
                elif isinstance(val, dict):
                    port = int(val.get("port"))
                    base_path = _clean_path(val.get("base_path", ""))
                    host_ip = val.get("host", "127.0.0.1")
                    scheme = val.get("scheme", "http")
                    converted[host] = Target(
                        port=port,
                        base_path=base_path,
                        host=host_ip,
                        scheme=scheme if scheme in ("http", "https") else "http",
                    )
                else:
                    raise ValueError(f"Unsupported route value for {host}: {val!r}")

            async with HOSTS_LOCK:
                DYNAMIC_HOSTNAME_TO_TARGET.clear()
                DYNAMIC_HOSTNAME_TO_TARGET.update(converted)
            print(f"📅 Loaded {len(DYNAMIC_HOSTNAME_TO_TARGET)} routes from disk")
        except Exception as e:
            print(f"⚠️ Failed to load routes from disk: {e}")

async def save_routes_to_disk():
    try:
        print("💾 Attempting to save routes...")
        serializable = {
            h: {"port": t.port, "base_path": t.base_path, "host": t.host, "scheme": t.scheme}
            for h, t in DYNAMIC_HOSTNAME_TO_TARGET.items()
        }
        with open(ROUTES_FILE, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"💾 Saved {len(DYNAMIC_HOSTNAME_TO_TARGET)} routes to disk")
    except Exception as e:
        print(f"⚠️ Failed to save routes: {e}")


# =========================
# Shutdown (WS cleanup)
# =========================

async def on_shutdown(app: web.Application):
    sockets = set(app.get("websockets", []))
    for ws in sockets:
        try:
            await ws.close(code=WSCloseCode.GOING_AWAY, message=b"Server shutting down")
        except Exception:
            pass


# =========================
# Reverse proxy handler
# =========================

async def handle_proxy(request: web.Request) -> web.StreamResponse:
    hostname = request.headers.get("Host", "")
    target = await get_target_for_host(hostname)

    if not target:
        return web.Response(status=502, text=f"Unknown host: {hostname}")

    # Build upstream path + query: prepend base_path
    path = request.rel_url.path
    qs = request.rel_url.query_string
    upstream_path = _join_paths(target.base_path, path)
    path_and_qs = upstream_path + (f"?{qs}" if qs else "")

    # WebSocket proxy
    if _is_ws(request):
        ws_server = web.WebSocketResponse(autoping=True, heartbeat=30)
        await ws_server.prepare(request)

        # Track for shutdown
        request.app.setdefault("websockets", set()).add(ws_server)

        ws_scheme = "wss" if target.scheme == "https" else "ws"
        ws_url = f"{ws_scheme}://{target.host}:{target.port}{path_and_qs}"
        try:
            async with ClientSession() as session:
                async with session.ws_connect(ws_url, heartbeat=30) as ws_client:

                    async def ws_forward(src, dst):
                        async for msg in src:
                            if msg.type == WSMsgType.TEXT:
                                await dst.send_str(msg.data)
                            elif msg.type == WSMsgType.BINARY:
                                await dst.send_bytes(msg.data)
                            elif msg.type == WSMsgType.PING:
                                await dst.ping()
                            elif msg.type == WSMsgType.PONG:
                                await dst.pong()
                            elif msg.type == WSMsgType.CLOSE:
                                await dst.close()
                                return

                    tasks = [
                        asyncio.create_task(ws_forward(ws_server, ws_client)),
                        asyncio.create_task(ws_forward(ws_client, ws_server)),
                    ]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending:
                        t.cancel()

                    await ws_client.close()
                    await ws_server.close()

        except ClientConnectionError as e:
            print(f"❌ Could not connect to backend WS at {ws_url}: {e}")
            await ws_server.close(code=1011, message=b"Backend WS connection failed")
        except Exception as e:
            print(f"⚠️ Unexpected error in WS proxy: {e}")
            await ws_server.close()
        finally:
            request.app.get("websockets", set()).discard(ws_server)

        return ws_server

    # Regular HTTP proxy
    target_url = f"{target.scheme}://{target.host}:{target.port}{path_and_qs}"

    # Forward headers
    headers = request.headers.copy()
    headers["X-Forwarded-Host"] = hostname
    headers["X-Forwarded-Proto"] = "https" if request.secure else "http"
    headers["X-Forwarded-For"] = request.remote or ""
    if target.base_path:
        headers["X-Forwarded-Prefix"] = target.base_path

    try:
        body_in = await request.read()
        async with ClientSession() as session:
            async with session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body_in,
                allow_redirects=False,  # we want to rewrite Location on redirects
            ) as resp:
                body = await resp.read()

                # Copy headers to mutate safely
                out_headers = resp.headers.copy()

                # Rewrite absolute Location headers pointing to upstream back to public host
                loc = out_headers.get("Location")
                if loc:
                    public_host = _public_host_from_request(request)
                    out_headers["Location"] = _rewrite_location(loc, target, public_host)

                # NOTE: If you want to strip hop-by-hop headers (Connection, etc.), do it here.
                return web.Response(
                    status=resp.status,
                    body=body,
                    headers=out_headers,
                )
    except ClientConnectionError as e:
        return web.Response(
            status=502,
            text=f"Could not connect to backend HTTP server at {target_url}: {e}",
        )
    except Exception as e:
        print(f"⚠️ Error handling HTTP proxy: {e}")
        return web.Response(status=500, text="Internal Server Error")


# =========================
# Admin / API endpoints
# =========================

async def register_host(request: web.Request) -> web.Response:
    data = await request.json()
    hostname = data.get("hostname")
    port = data.get("port")
    base_path = _clean_path(data.get("base_path", ""))
    host_ip = data.get("host", "127.0.0.1")
    scheme = data.get("scheme", "http")

    if not (is_valid_hostname(hostname) and is_valid_port(port) and is_valid_base_path(base_path) and is_valid_host(host_ip) and is_valid_scheme(scheme)):
        return web.Response(status=400, text="Invalid hostname, port, base_path, host, or scheme")

    if hostname in STATIC_HOSTNAME_TO_TARGET:
        return web.Response(status=400, text="Cannot overwrite static route")

    async with HOSTS_LOCK:
        DYNAMIC_HOSTNAME_TO_TARGET[hostname] = Target(
            port=int(port), base_path=base_path, host=host_ip, scheme=scheme
        )
        print(f"✅ Registered: {hostname} -> {scheme}://{host_ip}:{port}{base_path}")
        await save_routes_to_disk()

    return web.Response(text=f"Registered {hostname} -> {scheme}://{host_ip}:{port}{base_path}")

async def unregister_host(request: web.Request) -> web.Response:
    data = await request.json()
    hostname = data.get("hostname")

    if not is_valid_hostname(hostname):
        return web.Response(status=400, text="Invalid hostname")

    async with HOSTS_LOCK:
        removed = DYNAMIC_HOSTNAME_TO_TARGET.pop(hostname, None)
        if removed:
            print(f"❌ Unregistered: {hostname}")
            await save_routes_to_disk()
            return web.Response(text=f"Unregistered {hostname}")
        else:
            return web.Response(status=404, text="Hostname not found in dynamic routes")

async def list_routes(request: web.Request) -> web.Response:
    async with HOSTS_LOCK:
        combined = {
            **{h: {"port": t.port, "base_path": t.base_path, "host": t.host, "scheme": t.scheme, "type": "static"} for h, t in STATIC_HOSTNAME_TO_TARGET.items()},
            **{h: {"port": t.port, "base_path": t.base_path, "host": t.host, "scheme": t.scheme, "type": "dynamic"} for h, t in DYNAMIC_HOSTNAME_TO_TARGET.items()},
        }
        return web.json_response(combined)

async def static_routes(request: web.Request) -> web.Response:
    return web.json_response({h: {"port": t.port, "base_path": t.base_path, "host": t.host, "scheme": t.scheme} for h, t in STATIC_HOSTNAME_TO_TARGET.items()})


async def admin_ui(request):
    return web.Response(content_type="text/html", text="""
<!DOCTYPE html>
<html>
<head>
    <title>Reverse Proxy Admin</title>
    <style>
        body { font-family: sans-serif; padding: 2rem; background: #f9f9f9; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
        th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
        th { background-color: #eee; }
        input, button { margin: 0.5rem 0; padding: 0.5rem; font-size: 1rem; }
        .muted { color: #666; }
        .row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; max-width: 1100px; }
    </style>
</head>
<body>
    <h1>🛠 Reverse Proxy Admin</h1>
    <div>
        <h3>Add Route</h3>
        <div class="row">
          <input id="hostname" placeholder="Hostname (e.g. myapp.local)" />
          <input id="host" placeholder="Upstream host/IP (e.g. 192.168.1.42)" />
          <input id="port" type="number" placeholder="Port (e.g. 8400)" />
          <input id="base_path" placeholder="Base path (e.g. /app, optional)" />
          <select id="scheme">
            <option value="http" selected>http</option>
            <option value="https">https</option>
          </select>
        </div>
        <button onclick="registerRoute()">Register</button>
        <div class="muted">Tip: leave Upstream host blank for 127.0.0.1</div>
        <div id="message" style="color: red; margin-top: 0.5rem;"></div>
    </div>

    <h3>Active Routes</h3>
    <table id="routesTable">
        <thead><tr><th>Hostname</th><th>Upstream</th><th>Base Path</th><th>Type</th><th>Action</th></tr></thead>
        <tbody></tbody>
    </table>

<script>
async function fetchRoutes() {
    const res = await fetch('/_routes');
    const data = await res.json();
    const table = document.querySelector("#routesTable tbody");
    table.innerHTML = "";
    for (const [host, info] of Object.entries(data)) {
        const upstream = `${info.scheme}://${info.host}:${info.port}`;
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${host}</td>
            <td>${upstream}</td>
            <td>${info.base_path || "/"}</td>
            <td>${info.type}</td>
            <td>${info.type === "static" ? "-" : `<button onclick="unregisterRoute('${host}')">Remove</button>`}</td>
        `;
        table.appendChild(row);
    }
}

function showMessage(text, isError = true) {
    const msg = document.getElementById("message");
    msg.style.color = isError ? "red" : "green";
    msg.textContent = text;
}

function clearMessage() {
    document.getElementById("message").textContent = "";
}

async function registerRoute() {
    clearMessage();
    const hostname = document.getElementById("hostname").value.trim();
    const host = document.getElementById("host").value.trim() || "127.0.0.1";
    const port = parseInt(document.getElementById("port").value);
    const base_path = document.getElementById("base_path").value.trim();
    const scheme = document.getElementById("scheme").value;

    if (!hostname || !port) {
        showMessage("Please enter a valid hostname and port.");
        return;
    }

    const res = await fetch('/_register', {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hostname, host, port, base_path, scheme })
    });

    if (res.ok) {
        showMessage("✅ Route registered.", false);
        await fetchRoutes();
    } else {
        const text = await res.text();
        showMessage(`❌ ${text}`);
    }
}

async function unregisterRoute(hostname) {
    clearMessage();
    if (!confirm(`Remove ${hostname}?`)) return;

    const res = await fetch('/_unregister', {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hostname })
    });

    if (res.ok) {
        showMessage("✅ Route removed.", false);
        await fetchRoutes();
    } else {
        const text = await res.text();
        showMessage(`❌ ${text}`);
    }
}

(async () => { await fetchRoutes(); })();
</script>

</body>
</html>
""")


# =========================
# App bootstrap
# =========================

async def start_reverse_proxy() -> None:
    await load_routes_from_disk()

    app = web.Application()
    app["websockets"] = set()
    app.on_shutdown.append(on_shutdown)

    # Admin/API endpoints first
    app.router.add_post("/_register", register_host)
    app.router.add_post("/_unregister", unregister_host)
    app.router.add_get("/_routes", list_routes)
    app.router.add_get("/_static_routes", static_routes)
    app.router.add_get('/admin', admin_ui)

    # Catch-all proxy LAST
    app.router.add_route("*", "/{tail:.*}", handle_proxy)

    # Setup runner with shorter shutdown timeout
    runner = web.AppRunner(app, shutdown_timeout=5)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=80)
    await site.start()
    print("🔁 Reverse proxy listening on port 80")

    stop_event = asyncio.Event()

    def _shutdown():
        print("🛑 Gracefully shutting down...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            # Windows
            pass

    await stop_event.wait()
    await runner.cleanup()
    print("✅ Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(start_reverse_proxy())
    except PermissionError:
        print("❌ You need sudo to bind to port 80 on macOS/Linux")

"""
======================
💡 EXAMPLE USAGE (cURL)
======================

# Register a new route with non-local upstream:
curl -X POST http://localhost/_register \
     -H "Content-Type: application/json" \
     -d '{"hostname":"robotarm.local","host":"192.168.1.42","port":8400,"base_path":"/app","scheme":"http"}'

# Register a local route, HTTPS upstream, mounted at root:
curl -X POST http://localhost/_register \
     -H "Content-Type: application/json" \
     -d '{"hostname":"ui.local","host":"127.0.0.1","port":9443,"base_path":"","scheme":"https"}'

# Unregister a route:
curl -X POST http://localhost/_unregister \
     -H "Content-Type: application/json" \
     -d '{"hostname":"robotarm.local"}'

# List all routes:
curl http://localhost/_routes
"""