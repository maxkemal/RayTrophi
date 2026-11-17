"""Room reset (manual + idle) and balloon size variety (Studio or Player).

Needs a prepared balloon demo on wet paint surfaces (the pastel workshop).
Reconfigures the demo with sizes 0.6..1.4 and a 10 s idle reset, then restores
the original config. ~25 s.

  python scripts/test/rt_balloon_reset_size_ipc.py [--player]
"""
import argparse
import math
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

    def demo(op="status", **config):
        return ipc.call("paint.wet_demo", operation=op, config=config)

    def covered():
        return ipc.call("paint.wet_coverage")["covered_texels"]

    original = demo()
    assert original["prepared"], "prepare the balloon demo first"
    config = {k: v for k, v in original["config"].items() if k not in ("surfaces", "resolution")}
    try:
        test = dict(config, size_min=0.6, size_max=1.4, idle_reset_seconds=10.0, autostart=False)
        state = demo("configure", **test)
        sizes = [b["size"] for b in state["balloons"]]
        check("1 sizes inside 0.6..1.4 and not all equal",
              all(0.6 - 1e-4 <= s <= 1.4 + 1e-4 for s in sizes) and max(sizes) - min(sizes) > 0.05,
              f"{min(sizes):.2f}..{max(sizes):.2f}")

        burst = demo("burst", count=3)
        event = burst["events"][-1]
        ppb = config.get("particles_per_balloon", 1800)
        expected = [max(32, round(ppb * b["size"] ** 2)) for b in event["balloons"]]
        actual = [b["drops"] for b in event["balloons"]]
        check("2 drops = particles_per_balloon * size^2", actual == expected, f"{actual} vs {expected}")
        check("3 spawned = sum of drops", event["spawned"] == sum(actual), str(event["spawned"]))
        time.sleep(3.0)
        before = covered()
        check("4 the burst painted", before > 0, f"{before} texels")

        # Manual reset: paint gone (second clean after 3 s), every balloon back.
        state = demo("reset")
        check("5 reset brings every balloon back", all(b["inflated"] for b in state["balloons"]))
        time.sleep(3.6)
        after = covered()
        check("6 paint cleared (incl. drops in flight)", after == 0, f"{before} -> {after}")
        check("7 resets counted, room clean", state["resets"] >= 1 and not demo()["dirty"])

        # Idle reset: one pop, then nobody for 10 s.
        resets = demo()["resets"]
        demo("burst", count=1)
        time.sleep(4.0)
        check("8 dirty after a pop", demo()["dirty"])
        time.sleep(8.0)
        state = demo()
        check("9 idle reset fired after 10 s", state["resets"] == resets + 1 and not state["dirty"],
              f"resets {resets} -> {state['resets']}, idle {state['idle_seconds']:.1f} s")
    finally:
        demo("configure", **config)

    print(f"{sum(checks)}/{len(checks)} PASS")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
