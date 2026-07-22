#!/usr/bin/env python3
"""RayTrophi remote IPC TLS/security regression suite (Faz 6.6)."""

import argparse
import json
import socket
import ssl
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def exchange(host, port, ca, token, method, params=None, request_id=1):
    context = ssl.create_default_context(cafile=ca)
    payload = json.dumps({"id": request_id, "auth": token, "method": method,
                          "params": params or {}}, separators=(",", ":")).encode()
    with socket.create_connection((host, port), timeout=10) as raw:
        with context.wrap_socket(raw, server_hostname=host) as secure:
            secure.sendall(struct.pack("!I", len(payload)) + payload)
            size = struct.unpack("!I", recv_exact(secure, 4))[0]
            return json.loads(recv_exact(secure, size))


def recv_exact(sock, size):
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("connection closed before complete response")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("--port", type=int, default=7443)
    parser.add_argument("--token", required=True)
    parser.add_argument("--ca", required=True)
    parser.add_argument("--outside-path")
    parser.add_argument("--admin-token")
    parser.add_argument("--extended", action="store_true",
                        help="also run TLS downgrade, oversized and 31-second idle tests")
    args = parser.parse_args()
    failures = 0

    def check(name, predicate, detail):
        nonlocal failures
        ok = bool(predicate)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failures += not ok

    valid = exchange(args.host, args.port, args.ca, args.token, "version")
    check("valid TLS + token", valid.get("result") == "0.5.0", valid)

    invalid = exchange(args.host, args.port, args.ca, "X" * 40, "version")
    check("invalid token", invalid.get("error") == "authentication failed", invalid)

    unknown = exchange(args.host, args.port, args.ca, args.token, "internal.not_registered")
    check("default deny", "not enabled" in unknown.get("error", ""), unknown)

    batch = exchange(args.host, args.port, args.ca, args.token, "batch", {
        "calls": [{"method": "version", "params": {}},
                  {"method": "internal.not_registered", "params": {}}]})
    check("batch policy bypass blocked", "not enabled" in batch.get("error", ""), batch)

    nested = {}
    for _ in range(70):
        nested = {"value": nested}
    deep = exchange(args.host, args.port, args.ca, args.token, "version", nested)
    check("JSON nesting limit", "nesting exceeds" in deep.get("error", ""), deep)

    def concurrent_version(index):
        return exchange(args.host, args.port, args.ca, args.token, "version", request_id=100 + index)
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(concurrent_version, range(8)))
    check("8 concurrent clients", all(item.get("result") == "0.5.0" for item in concurrent), concurrent)

    reconnects = [exchange(args.host, args.port, args.ca, args.token, "version",
                           request_id=200 + index) for index in range(32)]
    check("reconnect storm", all(item.get("result") == "0.5.0" for item in reconnects),
          f"{sum(item.get('result') == '0.5.0' for item in reconnects)}/32")

    if args.outside_path:
        outside = exchange(args.host, args.port, args.ca, args.token, "project.open",
                           {"path": args.outside_path})
        check("canonical path escape blocked", "outside" in outside.get("error", ""), outside)

    if args.admin_token:
        sessions = exchange(args.host, args.port, args.ca, args.admin_token,
                            "ipc.admin.sessions.list")
        check("admin session registry", isinstance(sessions.get("result"), list), sessions)
        audit = exchange(args.host, args.port, args.ca, args.admin_token,
                         "ipc.admin.audit.list", {"maximum": 64})
        check("admin audit query", isinstance(audit.get("result"), list), audit)

    if args.extended:
        try:
            context = ssl.create_default_context(cafile=args.ca)
            with socket.create_connection((args.host, args.port), timeout=10) as raw:
                context.wrap_socket(raw, server_hostname="wrong.invalid")
            hostname_rejected = False
        except (ssl.SSLError, ssl.CertificateError):
            hostname_rejected = True
        check("wrong TLS hostname rejected", hostname_rejected, hostname_rejected)

        try:
            context = ssl.create_default_context()
            with socket.create_connection((args.host, args.port), timeout=10) as raw:
                context.wrap_socket(raw, server_hostname=args.host)
            ca_rejected = False
        except ssl.SSLError:
            ca_rejected = True
        check("untrusted CA rejected", ca_rejected, ca_rejected)

        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.load_verify_locations(args.ca)
            context.minimum_version = ssl.TLSVersion.TLSv1_1
            context.maximum_version = ssl.TLSVersion.TLSv1_1
            with socket.create_connection((args.host, args.port), timeout=10) as raw:
                context.wrap_socket(raw, server_hostname=args.host)
            tls11_rejected = False
        except (ssl.SSLError, OSError):
            tls11_rejected = True
        check("TLS 1.1 rejected", tls11_rejected, tls11_rejected)

        context = ssl.create_default_context(cafile=args.ca)
        with socket.create_connection((args.host, args.port), timeout=10) as raw:
            with context.wrap_socket(raw, server_hostname=args.host) as secure:
                secure.sendall(struct.pack("!I", 16 * 1024 * 1024 + 1))
                secure.settimeout(3)
                try:
                    oversized_closed = secure.recv(1) == b""
                except (ConnectionError, OSError, ssl.SSLError):
                    oversized_closed = True
        check("oversized frame closed", oversized_closed, oversized_closed)

        with socket.create_connection((args.host, args.port), timeout=10) as raw:
            with context.wrap_socket(raw, server_hostname=args.host) as secure:
                secure.sendall(struct.pack("!I", 64))  # deliberately incomplete body
                time.sleep(31)
                secure.settimeout(3)
                try:
                    idle_closed = secure.recv(1) == b""
                except (ConnectionError, OSError, ssl.SSLError):
                    idle_closed = True
        check("slow client idle timeout", idle_closed, idle_closed)

    print(f"\nRemote IPC security suite: {failures} FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
