"""Read the live timeline, fluid domains, and kinematic proxy coverage."""

from __future__ import annotations

import collections
import json
import math
import time

from rt_ipc import RtIpc


def main() -> None:
    client = RtIpc()
    try:
        frame = client.call("timeline.get_frame")
        domain_result = client.call("fluid.list_domains")
        domains = (
            domain_result.get("domains", [])
            if isinstance(domain_result, dict)
            else domain_result
        )
        proxy_sets = client.call("physics.collider.proxy_set.list")
        print(f"timeline_frame={frame}")
        print(f"fluid_domain_count={len(domains)}")
        for domain in domains:
            domain_name = domain if isinstance(domain, str) else domain.get("name")
            if domain_name:
                state = client.call("fluid.get", domain=domain_name)
                print(
                    f"fluid[{domain_name!r}]="
                    f"{json.dumps(state, ensure_ascii=False, sort_keys=True)}"
                )
        print(f"kinematic_sets={len(proxy_sets)}")
        for proxy_set in proxy_sets:
            set_id = proxy_set["id"]
            authored = client.call(
                "physics.collider.proxy_set.get", set_id=set_id
            )
            samples = client.call(
                "physics.collider.proxy_set.sample", set_id=set_id, dt=1.0 / 60.0
            )
            time.sleep(0.45)
            later_samples = client.call(
                "physics.collider.proxy_set.sample", set_id=set_id, dt=1.0 / 60.0
            )
            first_centers = {
                row["proxy_id"]: row["center"]
                for row in samples
                if row["resolved"]
            }
            motion = []
            for row in later_samples:
                previous = first_centers.get(row["proxy_id"])
                if previous is None or not row["resolved"]:
                    continue
                delta = math.sqrt(
                    sum(
                        (float(row["center"][axis]) - float(previous[axis])) ** 2
                        for axis in range(3)
                    )
                )
                motion.append((delta, row["bone"]))
            motion.sort(reverse=True)
            resolved = sum(bool(row["resolved"]) for row in samples)
            shapes = collections.Counter(
                proxy["shape"] for proxy in authored["proxies"]
            )
            centers = [
                row["center"] for row in samples if row["resolved"]
            ]
            sample_min = [min(point[axis] for point in centers) for axis in range(3)]
            sample_max = [max(point[axis] for point in centers) for axis in range(3)]
            feet = [
                {
                    "bone": row["bone"],
                    "center": row["center"],
                    "radius": row["radius"],
                }
                for row in samples
                if row["resolved"]
                and ("foot" in row["bone"].lower() or "toe" in row["bone"].lower())
            ]
            print(
                f"set_id={set_id} name={authored['name']!r} "
                f"target={authored['target_character']!r} "
                f"enabled={authored['enabled']} proxies={len(samples)} "
                f"resolved={resolved} shapes={dict(shapes)} "
                f"sample_bounds=({sample_min}, {sample_max})"
            )
            print(f"feet={json.dumps(feet, ensure_ascii=False)}")
            print(f"largest_motion_over_450ms={motion[:8]}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
