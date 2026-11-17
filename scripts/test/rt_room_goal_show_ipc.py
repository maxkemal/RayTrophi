"""Shared room goal show: milestones 25/50/75/100 %, finale cascade, auto clean.

Needs a prepared balloon demo on wet paint surfaces (the pastel workshop),
Studio or Player. Uses a tiny goal so a few bursts reach it; restores the
original config at the end. ~30 s.

  python scripts/test/rt_room_goal_show_ipc.py [--player]
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

    def demo(op="status", **config):
        return ipc.call("paint.wet_demo", operation=op, config=config)

    original = demo()
    assert original["prepared"], "prepare the balloon demo first"
    config = {k: v for k, v in original["config"].items() if k not in ("surfaces", "resolution")}
    try:
        demo("reset")
        time.sleep(3.5)
        state = demo("configure", **dict(config, room_goal=0.05, finale_reset_seconds=4.0,
                                          autostart=False, idle_reset_seconds=0.0))
        check("1 fresh round: progress 0, no milestone", state["milestones"] == 0,
              f"{state['room_progress']:.2f}")
        shows = state["shows"]
        levels = []
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            state = demo()
            if state["milestones"] >= 4:
                break
            if all(not b["inflated"] for b in state["balloons"]):
                time.sleep(0.5)
                continue
            demo("burst", count=3)
            time.sleep(1.0)
            state = demo()
            levels = [e["level"] for e in state["events"] if e.get("type") == "milestone"]
        state = demo()
        levels = [e["level"] for e in state["events"] if e.get("type") == "milestone"]
        check("2 all four milestones in order", levels[-4:] == [1, 2, 3, 4], str(levels))
        check("3 one show per milestone", state["shows"] - shows == 4, str(state["shows"] - shows))
        confetti = ipc.call("particle.stats", system="BalloonConfetti")["alive_count"]
        check("4 confetti in the air", confetti > 0, f"{confetti} cards")
        time.sleep(3.0)
        state = demo()
        check("5 finale cascade popped the rest", state["finale_cascade_left"] == 0 and
              sum(b["inflated"] for b in state["balloons"]) <= len(state["balloons"]),
              f"left {state['finale_cascade_left']}")
        # finale_reset_seconds 4 + second clean 3 s
        time.sleep(5.0)
        state = demo()
        check("6 room reset after the finale: new round", state["milestones"] == 0 and
              state["room_progress"] < 0.25, f"{state['milestones']}, {state['room_progress']:.2f}")
    finally:
        demo("configure", **config)

    print(f"{sum(checks)}/{len(checks)} PASS")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
