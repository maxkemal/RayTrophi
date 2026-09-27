"""Live IPC smoke test for bone-attached kinematic collider proxy sets.

Run from a separate terminal while RayTrophi Studio is open. The probe creates
one uniquely named set bound to a deliberately missing character, so CRUD and
unresolved-target semantics can be tested without depending on the open scene.
It removes everything it creates in a finally block.
"""

from __future__ import annotations

import os
import time

from rt_ipc import RtIpc, RtIpcError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_refusal(client: RtIpc, method: str, expected: str, **params) -> None:
    ok, result = client.try_call(method, **params)
    require(not ok, f"{method} unexpectedly succeeded: {result!r}")
    require(expected in str(result), f"{method}: expected {expected!r}, got {result!r}")


def main() -> None:
    client = RtIpc()
    set_id = None
    proxy_id = None
    name = f"__kinematic_ipc_smoke_{os.getpid()}_{time.time_ns()}"
    missing_character = "__kinematic_ipc_missing_character__"
    try:
        before = client.call("physics.collider.proxy_set.list")
        created = client.call(
            "physics.collider.proxy_set.create",
            name=name,
            target_character=missing_character,
        )
        set_id = created["id"]
        require(created["name"] == name, "created set name did not round-trip")
        require(created["revision"] == 1, "new set revision must start at 1")

        fetched = client.call("physics.collider.proxy_set.get", set_id=set_id)
        require(fetched == created, "get did not return the created set")

        require_refusal(
            client,
            "physics.collider.proxy_set.set",
            "target_node_id_not_supported",
            set_id=set_id,
            target_node_id="reserved-id",
        )
        require_refusal(
            client,
            "physics.collider.proxy_set.set",
            "invalid_contact_material",
            set_id=set_id,
            restitution=1.01,
        )
        client.call(
            "physics.collider.proxy_set.set",
            set_id=set_id,
            viewport_visible=False,
            consumer_mask=7,
        )
        fetched = client.call("physics.collider.proxy_set.get", set_id=set_id)
        require(not fetched["viewport_visible"],
                "viewport visibility did not round-trip")
        require(fetched["consumer_mask"] == 7,
                "consumer mask did not round-trip")
        require(fetched["revision"] == 2,
                "successful set update must advance revision")

        proxy = client.call(
            "physics.collider.proxy.set",
            set_id=set_id,
            name="Probe Capsule",
            bone="ProbeBone",
            shape="capsule",
            local_position=[0.1, 0.2, 0.3],
            local_axis=[0.0, 1.0, 0.0],
            radius=0.08,
            half_length=0.2,
        )
        proxy_id = proxy["id"]
        require(proxy_id > 0, "proxy id must be positive")

        fetched = client.call("physics.collider.proxy_set.get", set_id=set_id)
        require(fetched["revision"] == 3, "proxy creation must advance revision")
        require(len(fetched["proxies"]) == 1, "proxy was not stored")

        samples = client.call(
            "physics.collider.proxy_set.sample",
            set_id=set_id,
            dt=1.0 / 60.0,
        )
        require(len(samples) == 1, "sample must report every authored proxy")
        require(not samples[0]["resolved"], "missing character resolved unexpectedly")
        require(samples[0]["unresolved_reason"], "missing target needs a named reason")
        require(samples[0]["center"] == [0.0, 0.0, 0.0],
                "unresolved proxy must not publish a fabricated position")

        require_refusal(
            client,
            "physics.collider.proxy_set.auto_fit",
            "unknown_character",
            set_id=set_id,
        )

        client.call(
            "physics.collider.proxy.delete",
            set_id=set_id,
            proxy_id=proxy_id,
        )
        proxy_id = None
        client.call("physics.collider.proxy_set.delete", set_id=set_id)
        set_id = None

        after = client.call("physics.collider.proxy_set.list")
        require(after == before, "probe cleanup did not restore the original registry")
        print("PASS kinematic collider IPC CRUD/validation/unresolved-target smoke")
    finally:
        if set_id is not None:
            try:
                if proxy_id is not None:
                    client.call(
                        "physics.collider.proxy.delete",
                        set_id=set_id,
                        proxy_id=proxy_id,
                    )
                client.call("physics.collider.proxy_set.delete", set_id=set_id)
            except RtIpcError as cleanup_error:
                print(f"WARNING cleanup failed: {cleanup_error}")
        client.close()


if __name__ == "__main__":
    main()
