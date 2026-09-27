"""Probe auto-fit and live bone following for Kinematic Collider Sources."""

from __future__ import annotations

import collections
import math
import os
import sys
import time

from rt_ipc import RtIpc, RtIpcError


def distance(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def main() -> None:
    client = RtIpc()
    set_id = None
    character = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        characters = client.call("rig.list_characters")
        if not characters:
            raise RuntimeError("no rig character is visible to rig.list_characters")
        if character is None:
            character = characters[0]
        if character not in characters:
            raise RuntimeError(f"unknown character {character!r}; available={characters!r}")

        bones = client.call("rig.list_bones", character=character)
        created = client.call(
            "physics.collider.proxy_set.create",
            name=f"__kinematic_rig_probe_{os.getpid()}_{time.time_ns()}",
            target_character=character,
            viewport_visible=True,
        )
        set_id = created["id"]
        fit = client.call(
            "physics.collider.proxy_set.auto_fit",
            set_id=set_id,
            maximum_proxies=64,
        )
        authored = client.call("physics.collider.proxy_set.get", set_id=set_id)
        shapes = collections.Counter(p["shape"] for p in authored["proxies"])
        first = client.call(
            "physics.collider.proxy_set.sample", set_id=set_id, dt=1.0 / 60.0
        )
        time.sleep(0.45)
        second = client.call(
            "physics.collider.proxy_set.sample", set_id=set_id, dt=1.0 / 60.0
        )

        first_by_id = {row["proxy_id"]: row for row in first}
        unresolved = [row for row in second if not row["resolved"]]
        resolved = [row for row in second if row["resolved"]]
        motion = []
        for row in resolved:
            previous = first_by_id.get(row["proxy_id"])
            if previous and previous["resolved"]:
                motion.append(
                    (distance(previous["center"], row["center"]), row["bone"])
                )
        motion.sort(reverse=True)
        moving = [entry for entry in motion if entry[0] > 1.0e-5]
        feet = [
            entry
            for entry in motion
            if "foot" in entry[1].lower() or "toe" in entry[1].lower()
        ]
        authored_names = [proxy["bone"].lower() for proxy in authored["proxies"]]
        has_left_foot = any("leftfoot" in name for name in authored_names)
        has_right_foot = any("rightfoot" in name for name in authored_names)

        print(f"character={character!r} bones={len(bones)}")
        print(
            f"auto_fit_created={fit['created_count']} stored={len(authored['proxies'])} "
            f"shapes={dict(shapes)}"
        )
        print(
            f"resolved={len(resolved)} unresolved={len(unresolved)} "
            f"moving_over_450ms={len(moving)}"
        )
        if unresolved:
            reasons = collections.Counter(row["unresolved_reason"] for row in unresolved)
            print(f"unresolved_reasons={dict(reasons)}")
        print(f"largest_motion={motion[:8]}")
        print(f"foot_motion={feet[:8]}")

        if fit["created_count"] <= 0 or len(authored["proxies"]) <= 0:
            raise AssertionError("auto-fit produced no proxies")
        if unresolved:
            raise AssertionError("one or more auto-fit proxies did not resolve")
        if not has_left_foot or not has_right_foot:
            raise AssertionError("auto-fit budget did not preserve both feet")
        if not moving:
            raise AssertionError("no proxy followed the currently playing animation")
        print("PASS kinematic rig auto-fit and live bone-follow probe")
    finally:
        if set_id is not None:
            try:
                client.call("physics.collider.proxy_set.delete", set_id=set_id)
            except RtIpcError as cleanup_error:
                print(f"WARNING cleanup failed: {cleanup_error}")
        client.close()


if __name__ == "__main__":
    main()
