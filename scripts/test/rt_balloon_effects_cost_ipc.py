"""Balloon effect cost A/B in the Player: pops 3 balloons per setting and reports
the peak live particles of each effect system and the events' simulation /
frame-interval maxima. Restores the config. ~20 s.

  python scripts/test/rt_balloon_effects_cost_ipc.py
"""
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc
ipc = RtIpc(PLAYER_PIPE_NAME)
def demo(op="status", **c): return ipc.call("paint.wet_demo", operation=op, config=c)
orig = demo()
cfg = {k: v for k, v in orig["config"].items() if k not in ("surfaces", "resolution")}
def run(label, **over):
    st = demo("configure", **dict(cfg, autostart=False, **over))
    ids = [b["id"] for b in st["balloons"] if b["inflated"]][:3]
    for i in ids: demo("pop", balloon=i)
    peak = {}
    t0 = time.time()
    while time.time() - t0 < 2.5:
        e = demo()["effects"]
        for k, v in e["systems"].items():
            peak[k] = max(peak.get(k, 0), v["alive"])
        time.sleep(0.15)
    ev = demo()["events"][-3:]
    print(label, "peak alive", peak, "sim_ms_max", [round(x["simulation_ms_max"],1) for x in ev],
          "interval_ms_max", [round(x["frame_interval_ms_max"],1) for x in ev], flush=True)
    # Stage breakdown (maxima over the events' windows; build of 2026-11-18+).
    keys = ("demo_tick_ms_max", "fx_contacts_ms_max", "fx_drop_trails_ms_max", "fx_sparkles_ms_max",
            "effects_step_ms_max", "droplet_step_ms_max", "paint_ms_max")
    print("   stages", {k.replace("_ms_max", ""): round(max(x.get(k, 0.0) for x in ev), 1)
                        for k in keys}, flush=True)
    time.sleep(1.0)
try:
    # ★ The first configure + pops after it in a fresh process carry a one-off
    # ~30 ms spike (2026-11-18: it made drop trails look like +31..36 ms). This
    # first run is the warm-up: do not compare it, but its stage line is the one
    # that says where the spike lives (start the Player fresh to see it).
    run("(FIRST - warm-up, compare stages only)")
    run("ALL ON ")
    run("no drop trails", drop_trails=False)
    run("no drop trails/sparks", drop_trails=False, drop_sparks=0.0)
    run("all effects off", drop_trails=False, drop_sparks=0.0, sparkle_trails=False, sparkles=0)
finally:
    demo("configure", **cfg)
