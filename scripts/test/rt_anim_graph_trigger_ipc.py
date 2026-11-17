"""Repeat one authored graph transition using one IPC trigger per attempt.

Run externally against an already open Studio (or --player). Does not push,
clone, save or rebuild a graph. Restores the initial runtime state afterward.
"""
import argparse
import json
from pathlib import Path
import time

from rt_ipc import PLAYER_PIPE_NAME, PIPE_NAME, RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", default="1")
    parser.add_argument("--from-state", default="q")
    parser.add_argument("--to-state", default="w")
    parser.add_argument("--trigger", default="1")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--player", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.cycles <= 20
    client = RtIpc(PLAYER_PIPE_NAME if args.player else PIPE_NAME)
    report = dict(status="FAIL", attempts=[])
    original = None
    node_id = None
    try:
        machines = client.call("anim.state_machines", character=args.character)
        matches = [(sm, t) for sm in machines for t in sm["transitions"]
                   if t["from"] == args.from_state and t["to"] == args.to_state
                   and t["condition"] == "trigger" and t["parameter"] == args.trigger]
        assert len(matches) == 1, "need one matching authored trigger transition"
        machine, transition = matches[0]
        original, node_id = machine["current_state"], machine["node_id"]
        assert original, "runtime state machine has not initialized"
        assert all(s["pose_connected"] for s in machine["states"]
                   if s["name"] in (args.from_state, args.to_state))
        report["transition"] = transition
        for _ in range(args.cycles):
            client.call("anim.force_state", character=args.character,
                        state=args.from_state, node_id=node_id)
            client.call("anim.trigger_graph_param", character=args.character,
                        name=args.trigger)
            samples = []
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = next(sm for sm in client.call("anim.state_machines",
                             character=args.character) if sm["node_id"] == node_id)
                samples.append(dict(current=state["current_state"],
                                    target=state["target_state"],
                                    transitioning=state["transitioning"],
                                    progress=state["transition_progress"]))
                if state["current_state"] == args.to_state and not state["transitioning"]:
                    break
                time.sleep(0.04)
            report["attempts"].append(samples)
            assert samples[-1]["current"] == args.to_state, samples[-1]
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        try:
            if original:
                client.call("anim.force_state", character=args.character,
                            state=original, node_id=node_id)
                report["restored_state"] = original
        except Exception as error:
            report.update(status="FAIL", restore_error=str(error))
            raise
        finally:
            client.close()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
