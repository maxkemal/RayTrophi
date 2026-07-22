#!/usr/bin/env python3
"""RayTrophi TLS remote IPC client (length-prefixed UTF-8 JSON)."""

import argparse
import json
import os
import socket
import ssl
import struct


def read_exact(stream, size):
    chunks = []
    while size:
        chunk = stream.recv(size)
        if not chunk:
            raise ConnectionError("remote IPC connection closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def call(stream, token, request_id, method, params=None):
    request = {"id": request_id, "auth": token, "method": method,
               "params": params or {}}
    payload = json.dumps(request, separators=(",", ":")).encode("utf-8")
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError("request exceeds 16 MiB")
    stream.sendall(struct.pack("!I", len(payload)) + payload)
    response_size = struct.unpack("!I", read_exact(stream, 4))[0]
    if response_size > 16 * 1024 * 1024:
        raise ValueError("server response exceeds 16 MiB")
    return json.loads(read_exact(stream, response_size).decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="RayTrophi secure remote IPC client")
    parser.add_argument("host")
    parser.add_argument("--port", type=int, default=7443)
    parser.add_argument("--token", default=os.environ.get("RAYTROPHI_REMOTE_IPC_TOKEN"))
    parser.add_argument("--ca", help="CA/self-signed certificate PEM used to verify the server")
    parser.add_argument("--method", default="version")
    parser.add_argument("--params", default="{}", help="JSON object")
    args = parser.parse_args()
    if not args.token or len(args.token) < 32:
        parser.error("--token or RAYTROPHI_REMOTE_IPC_TOKEN must contain at least 32 characters")

    context = ssl.create_default_context(cafile=args.ca)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    with socket.create_connection((args.host, args.port), timeout=10) as raw:
        with context.wrap_socket(raw, server_hostname=args.host) as secure:
            response = call(secure, args.token, 1, args.method, json.loads(args.params))
            print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
