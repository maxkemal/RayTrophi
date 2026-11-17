"""Every droplet contact must happen AT the collider that reports it.

Studio, balloon demo prepared. Switches to live mode and stops the wet paint
from consuming contacts for one pop, then restores both. For each collision
event, the reporting collider's resolved world box (collider.list) must lie
within 0.25 m of the event position.

Catches the 2026-11-17 class: an OBB away from the origin (a ceiling at y 4.7)
"hit" every untouched drop on its first step because the local-frame round
trip was not exact; drops died around the balloon and never reached the floor.

  python scripts/test/rt_collider_event_locality_ipc.py
"""
import collections
import time

from rt_ipc import PIPE_NAME, RtIpc

SYSTEMS = ["BalloonPaintRed", "BalloonPaintBlue", "BalloonPaintYellow", "BalloonPaintGreen"]


def distance_to_box(p, mn, mx):
    d = [max(mn[k] - p[k], 0.0, p[k] - mx[k]) for k in range(3)]
    return sum(x * x for x in d) ** 0.5


def main():
    ipc = RtIpc(PIPE_NAME)
    splat = ipc.call("paint.wet_status")["splat"]
    mode = ipc.call("sim.get_mode")["mode"]
    active = [s["name"] for s in ipc.call("particle.list_systems")["systems"] if s["active"]]
    checks = []

    def check(name, ok, detail=""):
        checks.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)

    ipc.call("sim.set_mode", mode="live")
    ipc.call("paint.wet_set_splat", consume_particles=False)
    try:
        time.sleep(0.5)
        for s in SYSTEMS:
            ipc.call("particle.collision_events", system=s, max_events=65536, consume=True)
        demo = ipc.call("paint.wet_demo")
        balloon = next(b for b in demo["balloons"] if b["inflated"] and not b.get("surprise"))
        system = SYSTEMS[balloon["color"] % 4]
        ipc.call("paint.wet_demo", operation="pop", config={"balloon": balloon["id"]})
        time.sleep(3.0)
        events = ipc.call("particle.collision_events", system=system, max_events=65536,
                          consume=True)["events"]
        ipc.call("particle.set_active_system", system=system)
        colliders = ipc.call("collider.list")
        colliders = colliders if isinstance(colliders, list) else colliders.get("colliders", [])
        far = collections.Counter()
        for e in events:
            c = colliders[e["collider"]] if e["collider"] < len(colliders) else None
            if not c or not c.get("resolved_valid"):
                continue   # plane colliders have no box
            if distance_to_box(e["position"], c["resolved_world_min"], c["resolved_world_max"]) > 0.25:
                far[c["name"]] += 1
        check("1 every pop drop made contact", len(events) >= 0.9 * 1800, f"{len(events)} events")
        check("2 no contact away from its collider", not far, dict(far))
        low = sum(1 for e in events if e["position"][1] < 0.3)
        check("3 drops reach the floor", low > 0.1 * len(events), f"{low}/{len(events)} below 0.3 m")
    finally:
        ipc.call("paint.wet_set_splat", consume_particles=splat["consume_particles"])
        ipc.call("sim.set_mode", mode=mode)
        if active:
            ipc.call("particle.set_active_system", system=active[0])
    print(f"{sum(checks)}/{len(checks)} PASS")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
