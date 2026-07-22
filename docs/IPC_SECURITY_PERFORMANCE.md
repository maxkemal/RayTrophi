# RayTrophi IPC Security and Performance

## Transport model

RayTrophi exposes one JSON dispatcher through two transports:

- Local Windows Named Pipe: `\\.\pipe\RayTrophiStudio`
- Optional remote TLS TCP: 4-byte network-order message length followed by UTF-8 JSON

The remote listener is disabled by default. It never falls back to plaintext.

## Local security

The pipe uses a protected DACL granting access only to the current Windows user,
LocalSystem and administrators. `PIPE_REJECT_REMOTE_CLIENTS` rejects SMB/named-pipe
remote access. `FILE_FLAG_FIRST_PIPE_INSTANCE` and a persistent pipe instance prevent
another process from racing the fixed pipe name between clients.

## Enabling remote TLS IPC

Set these process environment variables before starting RayTrophi Studio:

| Variable | Required | Default | Meaning |
|---|---:|---|---|
| `RAYTROPHI_REMOTE_IPC` | yes | `0` | Must equal `1` to enable |
| `RAYTROPHI_REMOTE_IPC_CERT` | yes | — | PEM server certificate chain |
| `RAYTROPHI_REMOTE_IPC_KEY` | yes | — | PEM private key |
| `RAYTROPHI_REMOTE_IPC_TOKEN` | yes | — | Random bearer token, minimum 32 characters |
| `RAYTROPHI_REMOTE_IPC_BIND` | no | `127.0.0.1` | IPv4 listen address |
| `RAYTROPHI_REMOTE_IPC_PORT` | no | `7443` | TCP port |
| `RAYTROPHI_REMOTE_IPC_ALLOW_FILES` | no | `0` | Enables project/import/export/render path access |
| `RAYTROPHI_REMOTE_IPC_ALLOW_SCRIPTS` | no | `0` | Enables `script.run_file` and addon enable/reload |
| `RAYTROPHI_REMOTE_IPC_ALLOW_CIDRS` | no | empty | Comma-separated IPv4 CIDRs for the environment bootstrap token |
| `RAYTROPHI_REMOTE_IPC_WORKSPACE_ROOT` | no | empty | Canonical root for remote reads/imports/scripts; empty means no path-root restriction |
| `RAYTROPHI_REMOTE_IPC_EXPORT_ROOT` | no | empty | Canonical root for remote saves/exports/renders; empty means no path-root restriction |
| `RAYTROPHI_REMOTE_IPC_TOKEN_STORE` | no | beside EXE | Protected persistent token database path |
| `RAYTROPHI_REMOTE_IPC_AUDIT_JSONL` | no | disabled | Optional bounded/rotated JSONL audit destination |

Use a certificate whose SAN matches the DNS name clients use. Keep the default loopback
binding when a VPN or TLS reverse tunnel terminates on the same host. Bind to a LAN/WAN
interface only with an OS firewall allow-list. Never publish the port without TLS certificate
verification and a randomly generated token.

Remote requests include the token in the top-level `auth` field:

```json
{"id":1,"auth":"<secret>","method":"version","params":{}}
```

Run the reference client with a trusted CA/self-signed certificate:

```powershell
python scripts/remote_ipc_client.py studio.example.com --ca studio-ca.pem --method version
```

The token can be supplied through `RAYTROPHI_REMOTE_IPC_TOKEN` or `--token`. Prefer a
secret-managed environment over a command-line token because command lines may be visible
to other local processes.

## Limits

- TLS 1.2 minimum; TLS 1.3 is used when available.
- AEAD ECDHE cipher suites only.
- Maximum request or response: 16 MiB.
- Maximum simultaneous remote clients: 8.
- Maximum sustained request rate per connection: 120 requests/second.
- Socket receive/send idle timeout: 30 seconds.
- Maximum batch size: 64 calls.

Remote file and script capabilities are separately disabled by default. Raw tokens are stored
only as SHA-256 digests in an atomically replaced, current-user ACL-protected token database.
Every normal call and every batch child passes through the same fail-closed capability policy.
Optional token CIDRs and canonical workspace/export roots prevent source-network and path
escape. Authentication comparisons are constant-time. All engine access still runs through
`rtapi` on the main thread.

## Performance

The `batch` method executes up to 64 calls in one main-thread queue hop:

```json
{
  "id": 2,
  "method": "batch",
  "params": {
    "calls": [
      {"method":"timeline.get_frame","params":{}},
      {"method":"scene.object_exists","params":{"name":"default_Cube"}}
    ]
  }
}
```

Results are returned in call order. A failed child returns an `error` entry and does not stop
later children. Batch is not a rollback transaction and nested batches are rejected.

Queued calls that exceed the 30-second main-thread deadline are cancelled before execution.
Once a mutation starts, IPC waits for its real result instead of returning an ambiguous timeout.
Long-running systems should continue using their existing asynchronous status/cancel APIs.

## Deployment checklist

1. Use a CA-issued or privately trusted certificate with the correct DNS SAN.
2. Generate a unique high-entropy token per Studio host and rotate it periodically.
3. Keep file/script capabilities off unless required.
4. Restrict the port with Windows Firewall, VPN ACLs or a private network.
5. Do not expose self-signed TLS with certificate verification disabled.
6. Run `scripts/ipc_test_client.py` locally and `scripts/remote_ipc_client.py` through TLS.
7. Run `scripts/remote_ipc_security_test.py HOST --token TOKEN --ca CA`; require `0 FAIL`.

The in-application **View → Remote IPC Control** panel exposes authoritative TLS certificate,
session, token and audit state. `Disable Remote Access` closes the listener and active remote
sessions without relying on UI visibility as a security control.

## Rotation and incident response

1. Create a replacement token with the minimum scopes/CIDRs, update clients, then revoke the old
   token. Revocation marks matching live sessions for immediate socket shutdown.
2. Install a replacement certificate/key pair, verify SAN/expiry/fingerprint in the panel, and
   restart the private listener during a controlled window.
3. On suspected compromise, use **Disable Remote Access**, revoke affected tokens, preserve the
   rotated audit JSONL, restrict the firewall/VPN rule, and rotate both token and certificate.
4. Re-enable only after the local smoke and remote security suite both report `0 FAIL`.
5. Never attach `private-key.pem`, raw tokens, or an unrestricted token store to an incident ticket.
