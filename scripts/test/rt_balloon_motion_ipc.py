"""Floating balloons and the room-derived spawn volume (Player, or Studio in live mode).

Needs a prepared balloon demo with mesh balloons. Reads balloon positions twice,
2 s apart: they must have moved, stay inside the spawn volume (+ bob + margin),
and the motion cost must be reported. No config change.

  python scripts/test/rt_balloon_motion_ipc.py [--studio]
"""
import argparse
import time

from rt_ipc import PIPE_NAME, PLAYER_PIPE_NAME, RtIpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--studio", action="store_true")
    args = parser.parse_args()
    ipc = RtIpc(PIPE_NAME if args.studio else PLAYER_PIPE_NAME)
    checks = []

    def check(name, ok, detail=""):
        checks.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)

    a = ipc.call("paint.wet_demo")
    assert a["prepared"] and a["visual_mode"] == "flat_mesh", "prepare the mesh balloon demo first"
    lo, hi = a["spawn_min"], a["spawn_max"]
    check("1 spawn volume reported", all(hi[k] - lo[k] >= 0.2 for k in range(3)),
          f"{'auto' if a['spawn_auto'] else 'config'} y {lo[1]:.2f}..{hi[1]:.2f}")
    time.sleep(2.0)
    b = ipc.call("paint.wet_demo")
    moved = []
    outside = []
    amp = a["config"].get("float_amplitude", 0.08)
    for x, y in zip(a["balloons"], b["balloons"]):
        if not (x["inflated"] and y["inflated"]):
            continue
        d = sum((p - q) ** 2 for p, q in zip(x["position"], y["position"])) ** 0.5
        moved.append(d)
        for k in range(3):
            slack = 0.6 + (amp if k == 1 else 0.0)
            if y["position"][k] < lo[k] - slack or y["position"][k] > hi[k] + slack:
                outside.append((y["id"], k, round(y["position"][k], 2)))
    check("2 balloons move", moved and max(moved) > 0.02,
          f"max {max(moved) if moved else 0:.3f} m in 2 s")
    check("3 balloons stay in the room", not outside, str(outside))
    check("4 motion cost reported", b["motion_pushes"] > a["motion_pushes"],
          f"{b['motion_ms_max']:.2f} ms max, {b['motion_pushes'] - a['motion_pushes']} pushes")
    print(f"{sum(checks)}/{len(checks)} PASS")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
