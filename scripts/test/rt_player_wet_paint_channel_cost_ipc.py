"""Cost of the wet paint PBR channels in the open Player workshop (measurement, no PASS/FAIL).

    python scripts/test/rt_player_wet_paint_channel_cost_ipc.py [--rounds 2] [--count 3]

Needs the pastel workshop open in the Player with the balloon demo prepared and
not running. For each configuration on ALL wet surfaces - colour only,
+roughness, +roughness+height/normal - it resets the paint, bursts `count`
balloons through paint.wet_demo (the same call rt_workshop_balloons_ipc.py
uses) and samples until the balloons are back:

  frame_cost_ms / backend_frame_ms  player.display frame averages (capture off)
  present_interval_ms_max           the worst presented frame gap in the window
  last_tick_ms                      wet paint device tick (CPU side wait included)
  copies                            colour / channel rectangles copied per second

Configurations are interleaved over the rounds so drift does not look like a
configuration effect. Restores display, capture, settings and resets the paint.
"""
import argparse
import json
import statistics
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc

CONFIGS = (("colour", False, False), ("roughness", True, False), ("roughness+normal", True, True))
RESTORE_KEYS = ("roughness_channel", "height_normal_channel", "wet_roughness", "dry_roughness",
                "height_contribution", "normal_strength")


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def summary(values):
    if not values:
        return None
    return {"mean": round(statistics.fmean(values), 3), "p95": round(pct(values, 0.95), 3),
            "max": round(max(values), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    s = RtIpc(PLAYER_PIPE_NAME)
    saved, original = {}, None
    trials = []
    try:
        demo = s.call("paint.wet_demo")
        assert demo["prepared"] and not demo["running"], "prepare the balloon demo and stop autostart"
        objects = [sf["object"] for sf in s.call("paint.wet_status")["surfaces"]]
        assert objects, "no wet paint surface"
        for obj in objects:
            settings = s.call("paint.wet_get_settings", object=obj)
            saved[obj] = {k: settings[k] for k in RESTORE_KEYS}
        original = {"display": s.call("player.display"),
                    "capture": s.call("viewport.status")["capture_enabled"]}
        s.call("viewport.capture", enabled=False)
        s.call("player.set_display", continuous=True, pacing_hz=60)

        for round_index in range(args.rounds):
            for name, roughness, normal in CONFIGS:
                s.call("paint.wet_reset")
                for obj in objects:
                    s.call("paint.wet_set_settings", object=obj, roughness_channel=roughness,
                           height_normal_channel=normal)
                time.sleep(1.0)   # channel atlas / textures settle before the burst
                st0 = s.call("paint.wet_status")
                burst = s.call("paint.wet_demo", operation="burst", config={"count": args.count})
                spawned = burst["events"][-1]["spawned"] if burst.get("events") else None
                t0 = time.monotonic()
                cost, backend, gap, tick, active = [], [], [], [], []
                while True:
                    time.sleep(0.05)
                    frame = s.call("player.display").get("frame", {})
                    st = s.call("paint.wet_status")
                    if "frame_cost_ms" in frame:
                        cost.append(frame["frame_cost_ms"])
                    if "backend_frame_ms_avg" in frame:
                        backend.append(frame["backend_frame_ms_avg"])
                    if "present_interval_ms_max" in frame:
                        gap.append(frame["present_interval_ms_max"])
                    if st["status"] != "idle":
                        tick.append(st["last_tick_ms"])
                        active.append(st["active_tiles"])
                    state = s.call("paint.wet_demo")
                    assert not state["error"], state["error"]
                    elapsed = time.monotonic() - t0
                    if elapsed > 4.0 and all(b["inflated"] for b in state["balloons"]):
                        break
                    assert elapsed < 40.0, "balloons never came back"
                st1 = s.call("paint.wet_status")
                cov = s.call("paint.wet_coverage")
                telemetry = s.call("viewport.frame_telemetry")
                trials.append({
                    "round": round_index, "config": name, "seconds": round(elapsed, 2),
                    "spawned": spawned, "samples": len(cost),
                    "frame_cost_ms": summary(cost), "backend_frame_ms": summary(backend),
                    "present_interval_ms_max": summary(gap), "wet_tick_ms": summary(tick),
                    "active_tiles_max": max(active) if active else 0,
                    "colour_copies_per_s": round((st1["publish_device_copies"] - st0["publish_device_copies"]) / elapsed, 1),
                    "channel_copies_per_s": round((st1["pbr_device_copies"] - st0["pbr_device_copies"]) / elapsed, 1),
                    "channel_dispatches": st1["pbr_dispatches"] - st0["pbr_dispatches"],
                    "host_bytes": st1["publish_bytes_downloaded"] - st0["publish_bytes_downloaded"],
                    "covered_texels": cov["covered_texels"],
                    "roughness_texels": sum(c["roughness_texels"] for c in cov["surfaces"]),
                    "normal_texels": sum(c["normal_texels"] for c in cov["surfaces"]),
                    "sources_dropped": st1["sources_dropped"] - st0["sources_dropped"],
                    "device_lost": telemetry.get("device_lost")})
                print(json.dumps(trials[-1]), flush=True)
    finally:
        try:
            if original:
                s.call("viewport.capture", enabled=original["capture"])
                s.call("player.set_display", continuous=original["display"]["continuous"],
                       pacing_hz=float(original["display"]["pacing_hz"]))
            if saved:
                s.call("paint.wet_reset")
                for obj, values in saved.items():
                    s.call("paint.wet_set_settings", object=obj, **values)
        finally:
            s.close()


if __name__ == "__main__":
    main()
