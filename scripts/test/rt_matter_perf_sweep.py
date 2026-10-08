"""Sweep grid resolution and particle density over IPC and report where time goes.

Run from a terminal (NOT the app's script workspace) with a matter scene open
(one matter domain and at least one liquid source):

    python scripts/test/rt_matter_perf_sweep.py [frames] [warmup]

For every (voxel_size, particles_per_cell) pair the domain is reset, a warm-up
window is dropped, and then the per-section mean and worst-frame cost is read
from perf.list over a clean perf.reset window. The point is the scaling curve:
a section whose cost grows faster than particle count is the bottleneck that a
small-scene run would hide.

Wall-clock frame time measures the frame cadence, not the solver; compare the
sections (sim.*) instead. Missing sections are reported as absent, not zero.

★ This script CHANGES the open domain's voxel_size and particles_per_cell and
does not restore them. Save the project before running it, then reopen it.
"""
import sys
import time

import rt_ipc

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 10
FRAME_TIMEOUT_S = 120.0

# Coarse to fine. Halving voxel_size multiplies cells by 8, so the list is kept
# short; add a finer entry only after the coarser ones finish in budget.
VOXELS = [0.08, 0.05, 0.035]
PPCS = [4, 8]
SECTION_PREFIXES = ("sim.", "loop.frame")


def sections(c):
    return {e["name"]: e for e in c.call("perf.list")
            if e["name"].startswith(SECTION_PREFIXES)}


def run_frames(c, n):
    frame = c.call("sim.control_state")["frame"]
    worst = 0.0
    for _ in range(n):
        target = frame + 1
        t0 = time.perf_counter()
        c.call("timeline.set_frame", frame=target)
        while c.call("sim.control_state")["frame"] < target:
            if time.perf_counter() - t0 > FRAME_TIMEOUT_S:
                raise RuntimeError("frame %d did not complete" % target)
            time.sleep(0.005)
        worst = max(worst, (time.perf_counter() - t0) * 1000.0)
        frame = target
    return worst


def main():
    c = rt_ipc.RtIpc()
    domain = c.call("fluid.list_domains")["domains"][0]["name"]
    print("domain: %s  frames per point: %d (warm-up %d)" % (domain, FRAMES, WARMUP))
    for voxel in VOXELS:
        for ppc in PPCS:
            c.call("fluid.set_param", domain=domain, voxel_size=voxel,
                   particles_per_cell=ppc)
            c.call("fluid.reset")
            run_frames(c, WARMUP)
            c.call("perf.reset")
            worst_wall = run_frames(c, FRAMES)
            stats = c.try_call("fluid.step_stats", domain=domain)[1] or {}
            print("\n== voxel %.3f m  ppc %d  particles %s  up %.1f MB  worst wall %.0f ms =="
                  % (voxel, ppc, stats.get("particle_count"),
                     stats.get("upload_bytes", 0) / 1e6, worst_wall))
            rows = []
            for name, e in sections(c).items():
                count = e.get("count", 0)
                if count == 0:
                    continue
                rows.append((e.get("total_ms", 0.0) / count, e.get("max_ms", 0.0),
                             count, name))
            for mean, worst, count, name in sorted(rows, reverse=True):
                print("  %-40s mean %8.3f ms  max %8.3f ms  n=%d" % (name, mean, worst, count))
    c.close()


if __name__ == "__main__":
    main()
