"""Droplet colliders from the game design + one pop sound per balloon (Studio or Player).

Needs the pastel workshop (or any project with the four BalloonPaint* droplet
systems) open. Leaves the demo stopped and the test collider removed.

  python scripts/test/rt_balloon_collider_sound_ipc.py [--player] [--object PaintBack]
         [--sound <audio source id>]

--sound: an Audio source with a MONO asset. Without it the sound checks are skipped.
"""
import argparse
import time

from rt_ipc import PIPE_NAME, PLAYER_PIPE_NAME, RtIpc, RtIpcError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", action="store_true")
    parser.add_argument("--object", default="PaintBack")
    parser.add_argument("--sound", default="")
    args = parser.parse_args()
    ipc = RtIpc(PLAYER_PIPE_NAME if args.player else PIPE_NAME)
    checks = []

    def check(name, ok, detail=""):
        checks.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)

    def managed(result):
        return [r for r in result["colliders"]
                if r["managed"] and r["object"] == args.object]

    # 1. An SDF collider written into all four droplet systems, cooked.
    result = ipc.call("paint.wet_demo", operation="set_collider",
                      config={"object": args.object, "mode": "mesh_sdf", "sdf_resolution_mode": 0})
    rows = managed(result)
    check("1 mesh_sdf collider in all 4 droplet systems",
          len(rows) == 1 and rows[0]["mode"] == "mesh_sdf" and rows[0]["in_all_systems"], str(rows))
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30:
        rows = managed(ipc.call("paint.wet_demo"))
        if rows and rows[0]["ready"]:
            break
        time.sleep(0.2)
    check("2 SDF cooked", rows and rows[0]["ready"], f"{time.monotonic() - t0:.1f} s")

    # 2. Bad input is an error, not an abort.
    try:
        ipc.call("paint.wet_demo", operation="set_collider",
                 config={"object": args.object, "mode": "teapot"})
        check("3 unknown mode refused", False)
    except RtIpcError as e:
        check("3 unknown mode refused", "mode" in str(e))

    # 3. Sound: three pops at once -> three independent one-shots.
    if args.sound:
        demo = ipc.call("paint.wet_demo")
        config = dict(demo["config"])
        config["pop_sound"] = args.sound
        config.pop("surfaces", None)
        config.pop("resolution", None)
        demo = ipc.call("paint.wet_demo", operation="configure", config=config)
        before = demo["pop_sounds_played"]
        burst = ipc.call("paint.wet_demo", operation="burst", config={"count": 3})
        played = burst["pop_sounds_played"] - before
        check("4 three pops -> three sounds played", played == 3,
              f"played {played}, dropped {burst['pop_sounds_dropped']}, error '{burst['pop_sound_error']}'")
        status = ipc.call("audio.status")
        shots = [v for v in status.get("thunder_voices", []) if "source_id" in v]
        check("5 audio.status shows the one-shot voices", len(shots) >= 1, f"{len(shots)} live")
        positions = {tuple(round(x, 2) for x in v["position"]) for v in shots}
        check("6 one-shots sit at different balloons", len(positions) == len(shots), str(positions))
        ipc.call("paint.wet_demo", operation="stop")
    else:
        print("SKIP 4-6 (no --sound)")

    # 4. none removes it again.
    result = ipc.call("paint.wet_demo", operation="set_collider",
                      config={"object": args.object, "mode": "none"})
    check("7 none removes the collider", not managed(result))

    print(f"{sum(checks)}/{len(checks)} PASS")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
