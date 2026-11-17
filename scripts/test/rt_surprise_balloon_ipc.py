"""Surprise balloon: gold metal look, rainbow paint, confetti that never paints.

Needs a prepared balloon demo with mesh balloons (the pastel workshop) in Studio
(material.get is Studio-only; --player skips the material checks). Reconfigures
with surprise_chance 0.5, restores the original config at the end. ~15 s.

  python scripts/test/rt_surprise_balloon_ipc.py [--player]
"""
import argparse
import time

from rt_ipc import PIPE_NAME, PLAYER_PIPE_NAME, RtIpc

GOLD = [1.0, 0.62, 0.08]


def close(a, b, eps=1e-3):
    return len(a) == len(b) and all(abs(x - y) < eps for x, y in zip(a, b))


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
        state = demo("configure", **dict(config, surprise_chance=0.5, autostart=False))
        surprises = [b for b in state["balloons"] if b["surprise"]]
        normals = [b for b in state["balloons"] if not b["surprise"]]
        check("1 some balloons are surprises at chance 0.5", 1 <= len(surprises) < len(state["balloons"]),
              f"{len(surprises)}/{len(state['balloons'])}")
        if not surprises:
            raise SystemExit(1)
        target = surprises[0]

        if not args.player and "visual_node" in target:
            color = ipc.call("material.get", object_name=target["visual_node"], param="base_color")
            metal = ipc.call("material.get", object_name=target["visual_node"], param="metallic")
            check("2 surprise mesh is gold metal", close(color, GOLD) and metal > 0.99, f"{color}, {metal}")
            if normals:
                n = normals[0]
                metal = ipc.call("material.get", object_name=n["visual_node"], param="metallic")
                color = ipc.call("material.get", object_name=n["visual_node"], param="base_color")
                check("3 normal mesh is palette colour, not metal",
                      metal < 0.01 and close(color, state["palette"][n["color"]]), f"{color}, {metal}")

        before = state["surprise_pops"]
        result = demo("pop", balloon=target["id"])
        event = result["events"][-1]
        popped = event["balloons"][0]
        ppb = config.get("particles_per_balloon", 1800)
        expected = max(32, round(ppb * popped["size"] ** 2)) * 3 // 2
        check("4 surprise pop counted, 1.5x drops", result["surprise_pops"] == before + 1 and
              popped["surprise"] and popped["drops"] == expected, f"{popped['drops']} vs {expected}")
        confetti = ipc.call("particle.stats", system="BalloonConfetti")["alive_count"]
        check("5 confetti alive", confetti > 0, f"{confetti} cards")
        time.sleep(3.0)
        coverage = ipc.call("paint.wet_coverage")
        check("6 the surprise painted", coverage["covered_texels"] > 0, f"{coverage['covered_texels']} texels")
    finally:
        demo("configure", **config)

    print(f"{sum(checks)}/{len(checks)} PASS")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
