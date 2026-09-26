"""Step the timeline one frame at a time over IPC and time every sim stage.

Run from a terminal (NOT the app's script workspace) with the scene open:

    python scripts/test/drive_fluid_frame_timing.py [end_frame] [slow_ms]

Unlike watching playback, each line here is exactly ONE simulated frame, so a
stall is pinned to the frame that caused it along with the per-stage split
(sim.timeline.* / sim.fluid.* / sim.collider.*) and the fluid step stats.
Frames slower than slow_ms are marked with '!!'.
"""
import sys
import time

import rt_ipc

END_FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 240
SLOW_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
FRAME_TIMEOUT_S = 120.0

SHORT = {
    "sim.timeline.update": "upd",
    "sim.timeline.step": "step",
    "sim.timeline.capture_frame": "cap",
    "sim.timeline.render_sync": "rsync",
    "sim.timeline.config_sig": "sig",
    "sim.timeline.source_poses": "pose",
    "sim.timeline.restore_frame": "rest",
    "sim.fluid.voxelize_colliders": "vox",
    "sim.fluid.thermal_cool_freeze": "cool",
    "sim.fluid.solid_overlay": "ovl",
    "sim.fluid.solid_face_weights": "fw",
    "sim.fluid.thermal_viscosity_field": "visc",
    "sim.collider.obb_resolve": "obb",
    "sim.collider.surface_cache_rebuild": "surf",
    "loop.frame": "frame",
}


def snap(c):
    return {e["name"]: (e["count"], e["total_ms"]) for e in c.call("perf.list")}


def main():
    c = rt_ipc.RtIpc()
    domain = c.call("fluid.list_domains")["domains"][0]["name"]
    frame = c.call("sim.control_state")["frame"]
    slow = []
    while frame < END_FRAME:
        target = frame + 1
        before = snap(c)
        t0 = time.perf_counter()
        c.call("timeline.set_frame", frame=target)
        while c.call("sim.control_state")["frame"] < target:
            if time.perf_counter() - t0 > FRAME_TIMEOUT_S:
                print("frame %d did not complete in %.0f s" % (target, FRAME_TIMEOUT_S))
                return
            time.sleep(0.005)
        wall = (time.perf_counter() - t0) * 1000.0
        after = snap(c)
        parts = []
        for name, short in SHORT.items():
            n0, t0_ms = before.get(name, (0, 0.0))
            n1, t1_ms = after.get(name, (0, 0.0))
            if n1 > n0 and t1_ms - t0_ms >= 0.5:
                parts.append("%s=%.0f" % (short, t1_ms - t0_ms))
        ok, st = c.try_call("fluid.step_stats", domain=domain)
        solver = ""
        if ok:
            solver = "n=%s p=%.0f up=%.0fMB dn=%.0fMB" % (
                st.get("particle_count"), st.get("pressure_ms", 0.0),
                st.get("upload_bytes", 0) / 1e6, st.get("download_bytes", 0) / 1e6)
        mark = "!!" if wall >= SLOW_MS else "  "
        print("%s f%-4d %7.0f ms | %s | %s" % (mark, target, wall, " ".join(parts), solver),
              flush=True)
        if wall >= SLOW_MS:
            slow.append((target, wall))
        frame = target
    print("\nslow frames (>= %.0f ms): %s" % (SLOW_MS, slow or "none"))


if __name__ == "__main__":
    main()
