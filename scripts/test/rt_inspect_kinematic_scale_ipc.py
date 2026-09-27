"""Print authored and sampled dimensions for major body proxies."""

from __future__ import annotations

import math
import sys

from rt_ipc import RtIpc


def length(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def main() -> None:
    client = RtIpc()
    try:
        requested = sys.argv[1] if len(sys.argv) > 1 else ""
        sets = client.call("physics.collider.proxy_set.list")
        selected = next(
            (row for row in sets if requested in row["name"]),
            sets[0] if sets else None,
        )
        if selected is None:
            raise RuntimeError("no kinematic proxy set")
        samples = client.call(
            "physics.collider.proxy_set.sample",
            set_id=selected["id"],
            dt=1.0 / 60.0,
        )
        sample_by_id = {row["proxy_id"]: row for row in samples}
        tokens = ("hips", "spine", "upleg", "leg", "foot", "toe", "head")
        print(f"set={selected['name']!r}")
        for proxy in selected["proxies"]:
            if not any(token in proxy["bone"].lower() for token in tokens):
                continue
            sample = sample_by_id[proxy["id"]]
            segment = length(sample["capsule_start"], sample["capsule_end"])
            print(
                f"{proxy['bone']} {proxy['shape']} "
                f"authored_radius={proxy['radius']:.6g} "
                f"half_length={proxy['half_length']:.6g} "
                f"sample_radius={sample['radius']:.6g} "
                f"sample_segment={segment:.6g} "
                f"extents={proxy['half_extents']}"
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
