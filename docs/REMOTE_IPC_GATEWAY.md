# RayTrophi Remote Gateway Boundary

RayTrophiStudio's TLS IPC listener is a private DCC control plane. It may bind to
loopback, a trusted LAN, or a VPN address. It must not be exposed directly to the
public Internet.

The public edge is a separate HTTPS/WebSocket gateway process. The gateway owns
OIDC/OAuth2 or mTLS service identity, MFA/SSO integration, WAF rules, tenant and
user quotas, request normalization, and public certificate lifecycle. It maps an
approved external principal to a short-lived, least-privilege RayTrophi token and
connects to the private TLS listener.

## Required deployment path

`Client -> HTTPS/WSS gateway -> private TLS IPC -> RayTrophi main-thread queue`

- Public clients never receive the DCC bootstrap token.
- The gateway passes a correlation ID and records the RayTrophi connection/token ID.
- File paths are logical workspace-relative paths; the gateway never accepts an
  arbitrary host path.
- Uploads use quarantined object storage and malware scanning before import.
- Long operations return a job ID and are observed with status/cancel calls.
- A gateway outage does not weaken the DCC listener; RayTrophi remains fail-closed.

The versioned HTTP contract is in `remote_ipc_gateway_openapi.yaml`. WebSocket
events use the same schemas and correlation IDs. A WSS event envelope is:

```json
{
  "version": "1.0",
  "event_id": "uuid",
  "correlation_id": "uuid",
  "connection_id": "opaque",
  "type": "job.progress",
  "timestamp": 1784690000,
  "payload": {"job_id":"opaque","progress":0.5}
}
```

Allowed v1 event types are `job.started`, `job.progress`, `job.completed`,
`job.failed`, `job.cancelled`, `session.closed`, and `security.policy_denied`.
Clients resume with the last `event_id`; the gateway owns bounded replay and
backpressure. This repository intentionally
does not embed an identity provider or Internet-facing web server into the DCC.
