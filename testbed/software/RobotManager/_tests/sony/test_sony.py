#!/usr/bin/env python3
import argparse
import socket
import struct
import uuid
import sys

PTPIP_PORT = 15740

# PTP/IP packet types
PKT_INIT_CMD_REQ  = 0x00000001
PKT_INIT_CMD_ACK  = 0x00000002
PKT_INIT_EVT_REQ  = 0x00000003
PKT_INIT_EVT_ACK  = 0x00000004
PKT_OP_REQ        = 0x00000006
PKT_OP_RESP       = 0x00000007

# PTP response codes (lower 16 bits)
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
    # client_name is UTF-16LE, NUL-terminated (0x0000)
    name_bytes = client_name.encode("utf-16le") + b"\x00\x00"
    version = 0x00010000  # PTP version 1.0
    total_len = 4 + 4 + 16 + len(name_bytes) + 4
    pkt = (
        struct.pack("<I", total_len) +
        struct.pack("<I", PKT_INIT_CMD_REQ) +
        client_guid.bytes +               # 16 bytes GUID (raw)
        name_bytes +
        struct.pack("<I", version)
    )
    sock.sendall(pkt)

    ptype, total_len, rest = recv_packet(sock)
    if ptype != PKT_INIT_CMD_ACK:
        raise RuntimeError(f"Unexpected packet type {ptype:#x} (expected INIT_CMD_ACK)")

    # rest layout: [ptype(4)][conn(4)][camera_guid(16)][name(..., utf16le, nul)][version(4)]
    conn_num = struct.unpack("<I", rest[4:8])[0]
    cam_guid = rest[8:24]
    cam_name_bytes = rest[24:-4]  # exclude trailing version
    try:
        cam_name = cam_name_bytes.decode("utf-16le").rstrip("\x00")
    except UnicodeDecodeError:
        cam_name = "<unknown>"
    return conn_num, cam_name, cam_guid

def send_init_event(sock, connection_number):
    pkt = struct.pack("<III", 12, PKT_INIT_EVT_REQ, connection_number)
    sock.sendall(pkt)
    ptype, _, _ = recv_packet(sock)
    if ptype != PKT_INIT_EVT_ACK:
        raise RuntimeError(f"Unexpected packet type {ptype:#x} (expected INIT_EVT_ACK)")

def send_open_session(sock, session_id=1, txid=0):
    # Operation Request layout (PTP/IP):
    # [len:4][type:4][data_phase:4][op_code:2][reserved:2][transaction_id:4][param1..]
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
    # rest: [ptype(4)][resp_code(4)][txid(4)] ...
    resp_code = struct.unpack("<I", rest[4:8])[0] & 0xFFFF
    return resp_code

def main():
    ap = argparse.ArgumentParser(description="Probe a PTP/IP camera (Sony ZV-E10 II etc.) over LAN")
    ap.add_argument("--host", required=True, help="Camera IP or hostname on your LAN")
    ap.add_argument("--name", default="PythonPTPIP", help="Client name shown to camera")
    ap.add_argument("--timeout", type=float, default=3.0, help="Socket timeout (seconds)")
    args = ap.parse_args()

    client_guid = uuid.uuid4()

    print(f"Connecting to {args.host}:{PTPIP_PORT} ...")
    cmd = socket.create_connection((args.host, PTPIP_PORT), timeout=args.timeout)
    cmd.settimeout(args.timeout)

    try:
        print("-> Sending InitCommandRequest ...")
        conn_num, cam_name, cam_guid = send_init_command(cmd, args.name, client_guid)
        print(f"<- InitCommandAck OK: connection={conn_num}, camera='{cam_name}'")

        print("-> Opening event channel ...")
        evt = socket.create_connection((args.host, PTPIP_PORT), timeout=args.timeout)
        evt.settimeout(args.timeout)
        try:
            send_init_event(evt, conn_num)
            print("<- InitEventAck OK")
        finally:
            evt.close()

        print("-> Sending OpenSession (0x1002) ...")
        rc = send_open_session(cmd, session_id=1, txid=1)
        if rc == PTP_RC_OK:
            print("<- OperationResponse: OK (0x2001) — session opened.")
            print("\nSUCCESS: PTP/IP over your LAN works. You can now build Start/Stop REC and file transfer on top.")
        else:
            print(f"<- OperationResponse: 0x{rc:04x} — not OK (camera may require pairing / PC Remote mode).")
            sys.exit(2)

    finally:
        try:
            cmd.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        cmd.close()

if __name__ == "__main__":
    main()