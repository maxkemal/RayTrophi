"""Create and leave a body-focused kinematic proxy set for viewport review."""

from __future__ import annotations

import collections
import sys

from rt_ipc import RtIpc, RtIpcError


DETAIL_TOKENS = (
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
    "eye",
    "skirt",
    "hair",
    "twist",
    "tongue",
    "jaw",
    "breast",
    "end",
)


def main() -> None:
    client = RtIpc()
    set_id = None
    character = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        characters = client.call("rig.list_characters")
        if not characters:
            raise RuntimeError("no rig character is visible")
        character = character or characters[0]
        existing = client.call("physics.collider.proxy_set.list")
        base_name = f"{character} Kinematic Preview"
        matching = next(
            (row for row in existing if row["name"] == base_name), None
        )
        created_new = matching is None
        if created_new:
            created = client.call(
                "physics.collider.proxy_set.create",
                name=base_name,
                target_character=character,
                viewport_visible=True,
            )
            set_id = created["id"]
        else:
            set_id = matching["id"]
            client.call(
                "physics.collider.proxy_set.set",
                set_id=set_id,
                target_character=character,
                viewport_visible=True,
            )
        client.call(
            "physics.collider.proxy_set.auto_fit",
            set_id=set_id,
            replace_existing=True,
            maximum_proxies=256,
        )
        authored = client.call("physics.collider.proxy_set.get", set_id=set_id)
        removed = 0
        for proxy in authored["proxies"]:
            canonical = proxy["bone"].lower()
            if any(token in canonical for token in DETAIL_TOKENS):
                client.call(
                    "physics.collider.proxy.delete",
                    set_id=set_id,
                    proxy_id=proxy["id"],
                )
                removed += 1
        authored = client.call("physics.collider.proxy_set.get", set_id=set_id)
        samples = client.call(
            "physics.collider.proxy_set.sample", set_id=set_id, dt=1.0 / 60.0
        )
        unresolved = [row for row in samples if not row["resolved"]]
        if unresolved:
            raise AssertionError(f"unresolved proxies: {unresolved!r}")
        shapes = collections.Counter(proxy["shape"] for proxy in authored["proxies"])
        names = [proxy["bone"].lower() for proxy in authored["proxies"]]
        if not any("leftfoot" in name for name in names):
            raise AssertionError("left foot proxy is missing")
        if not any("rightfoot" in name for name in names):
            raise AssertionError("right foot proxy is missing")
        print(
            f"LEFT IN SCENE set_id={set_id} name={base_name!r} "
            f"character={character!r} reused={not created_new} "
            f"proxies={len(authored['proxies'])} removed_details={removed} "
            f"shapes={dict(shapes)}"
        )
        set_id = None
    finally:
        if set_id is not None and created_new:
            try:
                client.call("physics.collider.proxy_set.delete", set_id=set_id)
            except RtIpcError as cleanup_error:
                print(f"WARNING cleanup failed: {cleanup_error}")
        client.close()


if __name__ == "__main__":
    main()
