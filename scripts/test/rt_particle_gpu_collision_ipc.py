"""Particle collisions on the device-resident GPU path vs the CPU reference.

Boyama Odasi step 2 (docs/dev/PLAYER_GAME_PAINT_ROOM.md). Runs in Studio on an
EMPTY scene (the startup cube is tolerated and left alone). Never launches an
application.

    python scripts/test/rt_particle_gpu_collision_ipc.py

For execution_policy auto (GPU) and cpu (reference), the same setup: a stream
falling onto a sphere collider and a floor plane collider, response die + events.
Checks: the auto run really ran on the device with the colliders
(gpu_status gpu_resident, collision_stage_on_device), every event lies on the
surface it names (floor: y = particle radius, normal up; sphere: |p - c| =
r + particle radius, normal radial), event counts of GPU and CPU agree, and in a
bounce run no particle ends below the floor.
"""
import json
import math
import sys

from rt_ipc import RtIpc

RADIUS = 0.02
SPHERE_C = (0.0, 1.0, 0.0)
SPHERE_R = 0.3
STEPS = 120
DT = 1.0 / 60.0


def setup(s, policy, response, record):
    s.call("particle.clear_systems")
    s.call("particle.add_system", name="HitTest")
    s.call("particle.set_physics", system="HitTest", execution_policy=policy,
           particle_radius=RADIUS)
    s.call("particle.add_emitter", system="HitTest", name="Stream", point=[0.0, 2.0, 0.0],
           direction=[0.0, -1.0, 0.0], speed=1.0, spread=12.0, rate_per_second=3000.0,
           lifetime_seconds=5.0)
    s.call("collider.create", name="Ball", source_mode="sphere", sphere_center=list(SPHERE_C),
           sphere_radius=SPHERE_R, restitution=0.4)
    s.call("collider.create", name="Floor", source_mode="plane", plane_y=0.0, restitution=0.4)
    s.call("particle.set_collision", system="HitTest", response=response,
           record_events=record, event_capacity=200000)
    return {name: i for i, name in enumerate(collider_names(s))}


def collider_names(s):
    listed = s.call("collider.list")
    items = listed if isinstance(listed, list) else listed.get("colliders", [])
    return [c["name"] if isinstance(c, dict) else c for c in items]


def run_case(s, policy, response, record):
    index = setup(s, policy, response, record)
    for _ in range(STEPS):
        s.call("particle.step", system="HitTest", dt=DT)
    stats = s.call("particle.stats", system="HitTest")
    events = s.call("particle.collision_events", system="HitTest", max_events=200000,
                    consume=True)
    return index, stats, events


def check_events(events, index, label, check_normals=True):
    floor, ball = index.get("Floor"), index.get("Ball")
    worst = {"floor_dy": 0.0, "floor_normal": 1.0, "ball_dr": 0.0, "ball_normal": 1.0}
    counts = {"floor": 0, "ball": 0, "other": 0}
    for e in events["events"]:
        p, n = e["position"], e["normal"]
        assert e["died"], (label, e)
        if e["collider"] == floor:
            counts["floor"] += 1
            worst["floor_dy"] = max(worst["floor_dy"], abs(p[1] - RADIUS))
            worst["floor_normal"] = min(worst["floor_normal"], n[1])
        elif e["collider"] == ball:
            counts["ball"] += 1
            d = [p[k] - SPHERE_C[k] for k in range(3)]
            r = math.sqrt(sum(x * x for x in d))
            worst["ball_dr"] = max(worst["ball_dr"], abs(r - (SPHERE_R + RADIUS)))
            radial = sum(d[k] / max(r, 1e-9) * n[k] for k in range(3))
            worst["ball_normal"] = min(worst["ball_normal"], radial)
        else:
            counts["other"] += 1
    assert counts["ball"] > 50 and counts["floor"] > 50, (label, counts)
    assert counts["other"] == 0, (label, counts)
    assert worst["floor_dy"] < 1e-3 and worst["floor_normal"] > 0.99, (label, worst)
    assert worst["ball_dr"] < 1e-3, (label, worst)
    if check_normals:
        assert worst["ball_normal"] > 0.99, (label, worst)
    return counts, worst


def main():
    # --skip-cpu-normals: only for a build before the CPU event-normal fix.
    skip_cpu_normals = "--skip-cpu-normals" in sys.argv
    s = RtIpc()
    try:
        names = s.call("scene.list_objects")
        assert set(names) <= {"Default_Cube"}, f"needs an empty Studio scene, found {names}"
        assert not s.call("particle.list_systems")["systems"], "Studio already has particle systems"

        index, gpu_stats, gpu_events = run_case(s, "auto", "die", True)
        _, cpu_stats, cpu_events = run_case(s, "cpu", "die", True)
        report = {
            "gpu": {"gpu_status": gpu_stats.get("gpu_status"),
                    "collision_stage_on_device": gpu_events["collision_stage_on_device"],
                    "gpu_colliders": gpu_events["gpu_colliders"], "pending": gpu_events["pending"],
                    "overflow": gpu_events["overflow"], "device": gpu_events["device"],
                    "alive": gpu_stats.get("alive_count")},
            "cpu": {"gpu_status": cpu_stats.get("gpu_status"), "pending": cpu_events["pending"],
                    "overflow": cpu_events["overflow"], "alive": cpu_stats.get("alive_count")},
        }
        assert gpu_stats.get("gpu_status") == "gpu_resident", ("GPU path not taken", report)
        assert gpu_events["collision_stage_on_device"] and gpu_events["gpu_colliders"] == 2, report
        assert gpu_events["device"] and cpu_stats.get("gpu_status") == "cpu_policy", report
        report["gpu"]["counts"], report["gpu"]["worst"] = check_events(gpu_events, index, "gpu")
        report["cpu"]["counts"], report["cpu"]["worst"] = check_events(
            cpu_events, index, "cpu", check_normals=not skip_cpu_normals)
        ratio = gpu_events["pending"] / max(cpu_events["pending"], 1)
        report["gpu_cpu_event_ratio"] = round(ratio, 3)
        assert 0.8 < ratio < 1.25, ("GPU and CPU contact counts disagree", report)

        # Bounce: nobody below the floor, no events recorded.
        _, bounce_stats, bounce_events = run_case(s, "auto", "bounce", False)
        # bounds_min spans EVERY live particle (the state comes home from the device).
        sample = s.call("particle.get_state_sample", system="HitTest", max_count=16)
        lowest = sample["bounds_min"][1]
        report["bounce"] = {"gpu_status": bounce_stats.get("gpu_status"),
                            "events": bounce_events["pending"], "lowest_y": round(lowest, 6),
                            "alive": sample["alive_count"]}
        assert bounce_stats.get("gpu_status") == "gpu_resident", report
        assert bounce_events["pending"] == 0, report
        assert sample["alive_count"] > 100 and lowest >= RADIUS - 1e-4,             ("particles below the floor", report)
        print(json.dumps({"result": "PASS", **report}, indent=2))
    finally:
        try:
            # Colliders first: collider.* creates a default system when none exists.
            for name in collider_names(s):
                s.call("collider.remove", name=name)
            s.call("particle.clear_systems")
        finally:
            s.close()


if __name__ == "__main__":
    sys.exit(main())
