"""Check timeline-deterministic root motion on a walking character.

Usage: rt_probe_root_motion_timeline_ipc.py <character> <kinematic_set_name> [frames]

Reads the hip position from the kinematic proxy set's Hips proxy (the same
data the fluid/granular stamp uses). Turns root motion ON for the character
and leaves it on. Checks, in order:
  1. anim.character reports a resolved root bone with non-zero cycle travel;
  2. scrub determinism: the same frame reached by different jumps gives the
     same hip position;
  3. continuity: stepping frame by frame, no per-frame jump (the old loop
     teleport) and forward travel across several loops;
  4. rewind: frame 0 after the walk equals frame 0 before it.
"""

from __future__ import annotations

import math
import sys
import time

from rt_ipc import RtIpc


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def main() -> None:
    character = sys.argv[1]
    set_name = sys.argv[2]
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 110
    client = RtIpc()
    try:
        sets = client.call("physics.collider.proxy_set.list")
        match = next((s for s in sets if s["name"] == set_name), None)
        if match is None:
            raise RuntimeError(f"kinematic set not found: {set_name!r}")
        set_id = int(match["id"])

        def hips_at(frame):
            client.call("timeline.set_frame", frame=frame)
            previous = None
            deadline = time.time() + 5.0
            # Wait until the frame loop has evaluated the pose: two equal reads.
            while time.time() < deadline:
                time.sleep(0.12)
                rows = client.call("physics.collider.proxy_set.sample", set_id=set_id, dt=1 / 24)
                hips = next(r for r in rows if r["bone"].lower().endswith("hips"))["center"]
                if previous is not None and dist(previous, hips) < 1e-6:
                    return hips
                previous = hips
            return previous

        client.call("anim.set_root_motion", character=character, enabled=True)
        info = client.call("anim.character", character=character)
        print("root motion:", info["root_motion"], "bone:", info["root_motion_resolved_bone"],
              "cycle travel (parent space):", [round(v, 4) for v in info["root_motion_cycle_travel"]],
              "valid:", info["root_motion_travel_valid"],
              "graph follows timeline:", info["graph_follows_timeline"])
        failures = []
        if not info["root_motion_travel_valid"]:
            failures.append("resolved root bone has no cycle travel; pin one with anim.set_root_motion bone=...")

        # 2. Scrub determinism.
        order = [0, 10, 50, 5, 70, 10, 0, 50]
        seen = {}
        for frame in order:
            hips = hips_at(frame)
            if frame in seen and dist(seen[frame], hips) > 1e-3:
                failures.append(f"frame {frame} gave two poses: {seen[frame]} vs {hips}")
            seen.setdefault(frame, hips)
        print("scrub:", {f: [round(v, 3) for v in seen[f]] for f in sorted(seen)})

        # 3. Continuity over several loops.
        start = hips_at(0)
        prev = start
        max_step = 0.0
        max_step_frame = 0
        for frame in range(1, frames + 1):
            hips = hips_at(frame)
            step = dist(prev, hips)
            if step > max_step:
                max_step, max_step_frame = step, frame
            prev = hips
        horizontal = math.sqrt((prev[0] - start[0]) ** 2 + (prev[2] - start[2]) ** 2)
        print(f"walk 0..{frames}: horizontal travel {horizontal:.3f} m, "
              f"height change {prev[1] - start[1]:+.3f} m, largest per-frame step "
              f"{max_step:.3f} m at frame {max_step_frame}")
        if max_step > 0.25:
            failures.append(f"per-frame jump {max_step:.3f} m at frame {max_step_frame} (loop teleport?)")
        if horizontal < 1.0:
            failures.append(f"only {horizontal:.3f} m travelled in {frames} frames")

        # 4. Rewind.
        back = hips_at(0)
        if dist(back, start) > 1e-3:
            failures.append(f"rewind to 0 landed {dist(back, start):.3f} m away")
        print("rewind offset:", round(dist(back, start), 5))

        print("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures))
    finally:
        client.close()


if __name__ == "__main__":
    main()
