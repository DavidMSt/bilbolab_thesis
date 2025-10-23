#!/usr/bin/env python3
import argparse
import concurrent.futures
import ipaddress
import socket
import struct
import sys
import uuid

PTPIP_PORT = 15740

# PTP/IP packet types
PKT_INIT_CMD_REQ  = 0x00000001
PKT_INIT_CMD_ACK  = 0x00000002
PKT_INIT_EVT_REQ  = 0x00000003
PKT_INIT_EVT_ACK  = 0x00000004
PKT_OP_REQ        = 0x00000006
PKT_OP_RESP       = 0x00000007

PTP_RC_OK = 0x2001

def read_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        buf += chunk
    return buf

def recv_packet(sock):
    hdr = read_exact(sock, 4)
    total_len = struct.unpack("<I", hdr)[0]
    rest = read_exact(sock, total_len - 4)
    ptype = struct.unpack("<I", rest[:4])[0]
    return ptype, total_len, rest

def send_init_command(sock, client_name, client_guid):
    name_bytes = client_name.encode("utf-16le") + b"\x00\x00"
    version = 0x00010000
    total_len = 4 + 4 + 16 + len(name_bytes) + 4
    pkt = (
        struct.pack("<I", total_len) +
        struct.pack("<I", PKT_INIT_CMD_REQ) +
        client_guid.bytes +
        name_bytes +
        struct.pack("<I", version)
    )
    sock.sendall(pkt)
    ptype, _, rest = recv_packet(sock)
    if ptype != PKT_INIT_CMD_ACK:
        raise RuntimeError(f"Unexpected packet type {ptype:#x} (expected INIT_CMD_ACK)")
    conn_num = struct.unpack("<I", rest[4:8])[0]
    cam_name = rest[24:-4].decode("utf-16le", errors="ignore").rstrip("\x00")
    return conn_num, cam_name

def send_init_event(sock, connection_number):
    pkt = struct.pack("<III", 12, PKT_INIT_EVT_REQ, connection_number)
    sock.sendall(pkt)
    ptype, _, _ = recv_packet(sock)
    if ptype != PKT_INIT_EVT_ACK:
        raise RuntimeError(f"Unexpected packet type {ptype:#x} (expected INIT_EVT_ACK)")

def send_open_session(sock, session_id=1, txid=1):
    data_phase_no_data = 0x00000001
    op_code = 0x1002  # OpenSession
    reserved = 0x0000
    total_len = 4 + 4 + 4 + 2 + 2 + 4 + 4
    pkt = (
        struct.pack("<III", total_len, PKT_OP_REQ, data_phase_no_data) +
        struct.pack("<HHII", op_code, reserved, txid, session_id)
    )
    sock.sendall(pkt)
    ptype, _, rest = recv_packet(sock)
    if ptype != PKT_OP_RESP:
        raise RuntimeError(f"Unexpected packet type {ptype:#x} (expected OP_RESP)")
    resp_code = struct.unpack("<I", rest[4:8])[0] & 0xFFFF
    return resp_code

def probe_host(host, timeout, client_name):
    try:
        cmd = socket.create_connection((host, PTPIP_PORT), timeout=timeout)
    except ConnectionRefusedError:
        return (host, "REFUSED", None)
    except (TimeoutError, OSError):
        return (host, "NO LISTEN/UNREACHABLE", None)

    cmd.settimeout(timeout)
    try:
        conn_num, cam_name = send_init_command(cmd, client_name, uuid.uuid4())
        # open/close event channel just to prove both sides work
        evt = socket.create_connection((host, PTPIP_PORT), timeout=timeout)
        evt.settimeout(timeout)
        try:
            send_init_event(evt, conn_num)
        finally:
            evt.close()
        rc = send_open_session(cmd, session_id=1, txid=1)
        if rc == PTP_RC_OK:
            return (host, "OK", cam_name or "<camera>")
        else:
            return (host, f"PTP RESP 0x{rc:04x}", cam_name or "<camera>")
    except Exception as e:
        return (host, f"ERROR: {e}", None)
    finally:
        try:
            cmd.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        cmd.close()

def default_cidr():
    # Best-effort: pick a non-loopback IPv4 and assume /24
    try:
        # This returns multiple IPs on some OSes; choose the first non-local
        ips = socket.gethostbyname_ex(socket.gethostname())[2]
        ip = next((i for i in ips if not i.startswith("127.")), None)
        if ip:
            net = ipaddress.ip_network(ip + "/24", strict=False)
            return str(net)
    except Exception:
        pass
    # Fallback that covers many home LANs
    return "192.168.1.0/24"

def main():
    ap = argparse.ArgumentParser(description="Discover and probe PTP/IP cameras (Sony, etc.) on your LAN")
    ap.add_argument("--host", help="Probe a specific host (IP or hostname)")
    ap.add_argument("--cidr", help="CIDR to scan if --host not given (default: local /24)")
    ap.add_argument("--timeout", type=float, default=0.4, help="Socket timeout seconds")
    ap.add_argument("--threads", type=int, default=128, help="Parallel connection attempts")
    ap.add_argument("--name", default="PythonPTPIP", help="Client name shown to camera")
    args = ap.parse_args()

    if args.host:
        host, status, name = probe_host(args.host, args.timeout, args.name)
        if status == "OK":
            print(f"{host}: CONNECTED — {name}")
            sys.exit(0)
        else:
            print(f"{host}: {status}")
            sys.exit(2)

    cidr = args.cidr or default_cidr()
    print(f"Scanning {cidr} for TCP {PTPIP_PORT} ...")
    net = ipaddress.ip_network(cidr, strict=False)
    hosts = [str(h) for h in net.hosts()]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(probe_host, h, args.timeout, args.name): h for h in hosts}
        for fut in concurrent.futures.as_completed(futs):
            host, status, name = fut.result()
            if status in ("OK", "REFUSED", "PTP RESP 0x2019"):  # sample interesting statuses
                if name:
                    results.append((host, status, name))
                else:
                    results.append((host, status, ""))

    if not results:
        print("No PTP/IP endpoints found. If your camera is on, see the checklist below.")
        sys.exit(3)

    # Show all interesting hits
    for host, status, name in sorted(results):
        label = f" — {name}" if name else ""
        print(f"{host}: {status}{label}")

if __name__ == "__main__":
    main()