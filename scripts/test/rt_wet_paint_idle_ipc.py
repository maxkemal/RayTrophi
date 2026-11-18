"""Wet paint idles when nothing is wet: no full tick per frame, coverage cached.

Before 2026-11-18 a scene with GPU paint droplet systems ran a full wet paint
tick every frame (7.5..13 ms in the Player) even with nothing wet, because
every recording particle system counted as "contacts arriving". Needs a
prepared balloon demo; pops ONE balloon (check 5), then waits (<= 30 s) until the
paint has dried (active_tiles 0) and no drop is in flight. ~5..35 s.

  python scripts/test/rt_wet_paint_idle_ipc.py [--player]
"""
import argparse
import time

from rt_ipc import PIPE_NAME, PLAYER_PIPE_NAME, RtIpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", action="store_true")
    args = parser.parse_args()
    ipc = RtIpc(PLAYER_PIPE_NAME if args.player else PIPE_NAME)
    checks = []

    def check(name, ok, detail=""):
        checks.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)

    demo = ipc.call("paint.wet_demo", operation="stop", config={})
    # While paint is wet the room goal's periodic coverage is measured inside the
    # wet paint tick's own readback (no extra synchronize); paint.wet_coverage
    # over IPC stays exact.
    inflated = [b["id"] for b in demo["balloons"] if b["inflated"]]
    if demo["config"].get("room_goal", 0) > 0 and inflated:
        s0 = ipc.call("paint.wet_status")
        w0 = s0["coverage_in_tick"]
        ipc.call("paint.wet_demo", operation="pop", config={"balloon": inflated[0]})
        time.sleep(2.5)
        s1 = ipc.call("paint.wet_status")
        w1 = s1["coverage_in_tick"]
        exact = ipc.call("paint.wet_coverage")
        check("5 wet paint: coverage measured inside the tick, IPC coverage exact",
              w1 > w0 and exact["covered_texels"] > 0,
              f"coverage_in_tick {w0} -> {w1}, covered_texels {exact['covered_texels']}")
        # 2026-11-18: the tick reads its stats one tick later (no synchronize)
        # and publish queues its copies without waiting.
        delayed = s1["stats_delayed"] - s0["stats_delayed"]
        lost = s1["stats_readbacks_lost"] - s0["stats_readbacks_lost"]
        check("6 wet paint stats read one tick late, none lost, paint published on the device",
              delayed > 0 and lost == 0 and s1["publish_device_copies"] > s0["publish_device_copies"],
              f"stats_delayed +{delayed}, lost +{lost}, device copies "
              f"{s0['publish_device_copies']} -> {s1['publish_device_copies']}, mode {s1['publish_mode']}")
    else:
        print("SKIP 5 room_goal is 0 or no balloon inflated", flush=True)
    deadline = time.time() + 30.0
    while time.time() < deadline:
        ws = ipc.call("paint.wet_status")
        drops = sum(p["alive"] for p in ipc.call("paint.wet_demo", operation="status", config={})["pools"])
        if ws["active_tiles"] == 0 and drops == 0:
            break
        time.sleep(0.5)
    before = ipc.call("paint.wet_status")
    frames0 = ipc.call("paint.wet_demo", operation="status", config={})["observed_frames"]
    time.sleep(1.5)
    after = ipc.call("paint.wet_status")
    frames = ipc.call("paint.wet_demo", operation="status", config={})["observed_frames"] - frames0
    idle = after["idle_ticks"] - before["idle_ticks"]
    ticks = after["ticks"] - before["ticks"]
    # `ticks` counts only the ticks that dispatched; idle ones are idle_ticks.
    check("1 dry room: (almost) every tick is idle", idle > 0 and idle >= 0.9 * (idle + ticks),
          f"idle {idle}, dispatching {ticks}, {frames} frames, active_tiles {after['active_tiles']}")
    samples = []
    for _ in range(8):
        samples.append(ipc.call("paint.wet_status")["last_tick_ms"])
        time.sleep(0.1)
    check("2 an idle tick costs < 1 ms", max(samples) < 1.0, f"last_tick_ms {[round(x, 2) for x in samples]}")
    check("3 no contact source while no drop flies", after["event_sources"] == 0,
          f"event_sources {after['event_sources']}")
    c0 = after["coverage_cached"]
    ipc.call("paint.wet_coverage")
    ipc.call("paint.wet_coverage")
    c1 = ipc.call("paint.wet_status")["coverage_cached"]
    check("4 coverage of unchanged paint is cached", c1 >= c0 + 1, f"coverage_cached {c0} -> {c1}")
    if demo.get("config", {}).get("autostart"):
        ipc.call("paint.wet_demo", operation="start", config={})
    print(f"{sum(checks)}/{len(checks)} PASS")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
