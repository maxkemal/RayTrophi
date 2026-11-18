"""Firework sparkles + trails, and confetti/sparkles that land on the room.

Needs a prepared balloon demo whose droplet systems have colliders (the pastel
workshop / play room), in Studio or Player. Reconfigures with surprise_chance
0.5 and restores the original config at the end. ~12 s.

  python scripts/test/rt_balloon_sparkle_ipc.py [--player]
"""
import argparse
import time

from rt_ipc import PIPE_NAME, PLAYER_PIPE_NAME, RtIpc

CONFETTI, SPARKS, TRAILS = "BalloonConfetti", "BalloonSparkles", "BalloonSparkleTrails"


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

    def pop_surprise(state):
        target = next((b for b in state["balloons"] if b["surprise"] and b["inflated"]), None)
        return demo("pop", balloon=target["id"]) if target else None

    original = demo()
    assert original["prepared"], "prepare the balloon demo first"
    config = {k: v for k, v in original["config"].items() if k not in ("surfaces", "resolution")}
    try:
        state = demo("configure", **dict(config, surprise_chance=0.5, autostart=False,
                                          sparkles=120, sparkle_trails=True))
        systems = state["effects"]["systems"]
        droplet_colliders = len(state.get("colliders", []))
        check("1 confetti + sparkles mirror every droplet collider, trails none",
              droplet_colliders > 0 and systems[CONFETTI]["colliders"] == droplet_colliders and
              systems[SPARKS]["colliders"] == droplet_colliders and systems[TRAILS]["colliders"] == 0,
              f"droplet {droplet_colliders}, confetti {systems[CONFETTI]['colliders']}, "
              f"sparks {systems[SPARKS]['colliders']}, trails {systems[TRAILS]['colliders']}")

        before = state["effects"]["sparkles_spawned"]
        result = pop_surprise(state)
        if result is None:
            check("2 a surprise balloon exists at chance 0.5", False)
            raise SystemExit(1)
        effects = result["effects"]
        check("2 surprise pop throws 120 sparkles", effects["sparkles_spawned"] == before + 120 and
              effects["systems"][SPARKS]["alive"] > 0,
              f"spawned +{effects['sparkles_spawned'] - before}, alive {effects['systems'][SPARKS]['alive']}")

        y0 = effects["systems"][CONFETTI]["min_y"]
        time.sleep(0.4)
        status = demo()
        effects = status["effects"]
        moving = effects["systems"][CONFETTI]["min_y"] != y0
        if not moving:
            # The demo clock runs only in live mode (Player, or Studio live/play):
            # without it nothing moves and 3/4/7 would pass or fail meaninglessly.
            print("SKIP 3/4/7/9/10 demo clock is not running (Studio: switch to live mode)", flush=True)
        else:
            check("3 sparks leave trails", effects["trail_spawned"] > 0 and
                  effects["systems"][TRAILS]["alive"] > 0,
                  f"trail_spawned {effects['trail_spawned']}, alive {effects['systems'][TRAILS]['alive']}")

            time.sleep(1.8)
            status = demo()
            effects = status["effects"]
            systems = effects["systems"]
            # Automatic spawn volume = 0.8 m above the floor. Without colliders the
            # confetti sank 1-2 m below it by now (the sinking was silent).
            floor = status["spawn_min"][1] - 0.8 if status["spawn_auto"] else None
            min_y = systems[CONFETTI]["min_y"]
            if floor is None:
                print("SKIP 4 explicit spawn volume: floor height unknown", flush=True)
            else:
                check("4 confetti rests on the floor, not under it",
                      min_y is not None and min_y >= floor - 0.1, f"min_y {min_y}, floor ~{floor:.2f}")
            check("7 sparks hitting the room throw impact sparks, heads shed glitter",
                  effects["impacts"] > 0 and effects["impact_sparks"] == 4 * effects["impacts"] and
                  effects["glitter_spawned"] > 0,
                  f"impacts {effects['impacts']}, impact_sparks {effects['impact_sparks']}, "
                  f"glitter {effects['glitter_spawned']}")
            # Paint drops land within a second of the pop: their contacts are
            # sampled from the device event buffer and throw sparks.
            check("9 paint drop contacts throw sparks (ghost deaths)",
                  effects["drop_contacts_seen"] > 0 and effects["drop_sparks_spawned"] > 0,
                  f"contacts {effects['drop_contacts_seen']}, sparks {effects['drop_sparks_spawned']}")
            check("10 paint drops leave trails (from CPU ghosts)",
                  effects["ghosts_spawned"] > 0 and effects["drop_trail_spawned"] > 0,
                  f"ghosts {effects['ghosts_spawned']}, drop_trail_spawned {effects['drop_trail_spawned']}")

        # Glowing systems draw only as coloured billboards: an RT instance gives
        # the whole system one (white-hot orange) material over the colour.
        try:
            rt = [ipc.call("particle.get_render", system=name)["render_in_raytrace"]
                  for name in (SPARKS, TRAILS)]
            check("8 sparkles + trails are billboard-only (no RT spheres)", rt == [False, False], str(rt))
        except Exception as error:
            print(f"SKIP 8 particle.get_render unavailable here ({str(error)[:60]})", flush=True)

        state = demo("configure", **dict(config, surprise_chance=0.5, autostart=False, sparkles=0))
        before = state["effects"]["sparkles_spawned"]
        result = pop_surprise(state)
        check("5 sparkles 0 = off", result is not None and
              result["effects"]["sparkles_spawned"] == before, "")

        try:
            demo("configure", **dict(config, sparkles=500))
            check("6 sparkles 500 is refused", False, "accepted")
        except Exception as error:  # the IPC error carries the core's message
            check("6 sparkles 500 is refused", "sparkles" in str(error), str(error)[:80])
    finally:
        demo("configure", **config)

    print(f"{sum(checks)}/{len(checks)} PASS")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
