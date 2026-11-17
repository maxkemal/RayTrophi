"""Seeded 30 Hz synthetic children; never launches or builds the Player."""
import argparse
import json
import math
from pathlib import Path
import random
import time

try:
    from .schema import validate_pose
    from .transport import PIPE, connect, summary_ms, wait_frame
except ImportError:
    from schema import validate_pose
    from transport import PIPE, connect, summary_ms, wait_frame

HZ = 30
ACTIONS = ("walk", "reach", "jump", "stomp", "arms_up", "exit_enter",
           "short_occlusion", "identity_loss")


class Children:
    def __init__(self, count=4, seed=417):
        if not 1 <= count <= 10:
            raise ValueError("children must be 1..10")
        self.count = count
        rng = random.Random(seed)
        self.offsets = [rng.uniform(-0.08, 0.08) for _ in range(count)]

    def frame(self, frame):
        t = frame / HZ
        cycle = int(t // 24)
        stage = int(t % 24 // 3)
        local = t % 3
        persons = []
        for child in range(self.count):
            # All children complete every action in 24 s, even for N=1.
            if stage == 6 and 1.0 < local < 1.3:
                continue  # brief overlap, ID survives engine's 0.5 s grace
            if stage == 7 and 0.7 < local < 1.5:
                continue  # prolonged overlap, deliberate identity loss
            replacement = cycle + int(stage == 7 and local >= 1.5)
            pid = 1 + child + replacement*self.count
            x = (child - (self.count - 1)/2)*0.45 + self.offsets[child]
            z = 1.5 + 0.2*math.sin(t + child)
            lift = 0.0
            if stage == 0:
                x += 0.4*math.sin(t*1.5)
            if stage == 5:
                x += 6*math.sin(math.pi*local/3)**2
            if stage in (6, 7):
                x *= abs(local - 1.5)/1.5  # converge/overlap and separate
            if stage == 2:
                lift = 0.35*max(0.0, math.sin(local*math.pi*2))
            joints = {"pelvis": [x, 0.65 + lift, z, 1.0],
                      "neck": [x, 1.05 + lift, z, 1.0],
                      "head": [x, 1.22 + lift, z, 1.0]}
            for side, sign in (("l", -1), ("r", 1)):
                joints[f"{side}_shoulder"] = [x + sign*0.16, 1.0 + lift, z, 1.0]
                joints[f"{side}_elbow"] = [x + sign*0.23, 0.8 + lift, z, 1.0]
                joints[f"{side}_wrist"] = [x + sign*0.25, 0.65 + lift, z, 1.0]
                joints[f"{side}_hip"] = [x + sign*0.10, 0.65 + lift, z, 1.0]
                joints[f"{side}_knee"] = [x + sign*0.10, 0.35 + lift, z, 1.0]
                foot = lift
                if stage == 3 and side == "r":
                    # Raise slowly then strike in ~100 ms, repeat once a second.
                    phase = local % 1
                    foot = 0.22*min(phase/0.65, max(0, (0.85-phase)/0.1), 1)
                joints[f"{side}_ankle"] = [x + sign*0.10, foot, z, 1.0]
                if stage == 1:
                    joints[f"{side}_wrist"] = [x + sign*0.22, 1.1, z - 0.6, 1.0]
                if stage == 4:
                    joints[f"{side}_elbow"][1] = 1.22
                    joints[f"{side}_wrist"][1] = 1.55
            persons.append({"id": pid, "confidence": 1.0, "joints": joints})
        return validate_pose({"t": t, "persons": persons})


def read_path(value, path):
    for key in path.split("."):
        value = value[key] if isinstance(value, dict) else value[int(key)]
    return value


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("setup", "start", "cleanup"):
        if not isinstance(plan[key], list):
            raise ValueError(f"plan {key} must be a request list")
        for req in plan[key]:
            if not isinstance(req.get("method"), str) or not isinstance(req.get("params", {}), dict):
                raise ValueError("plan requests require method and params")
    states = plan["expected_states"]
    if not isinstance(states, list) or len(states) < 5 or states[0] != states[-1]:
        raise ValueError("expected_states must describe waiting/play/finale/reset/waiting")
    for key in ("state_path", "display_fps_path"):
        if not isinstance(plan[key], str) or not plan[key]:
            raise ValueError(f"plan requires {key}")
    if not plan["start"]:
        raise ValueError("live plan requires explicit session start requests")
    required = plan.get("required_events", [])
    if not isinstance(required, list) or any(not isinstance(event, str) for event in required):
        raise ValueError("required_events must be a list of event names")
    return plan


def run(args):
    if args.seconds < 24 or not math.isfinite(args.seconds):
        raise ValueError("seconds must be finite and >=24 to cover every scenario")
    plan = load_plan(args.session_plan) if args.session_plan else None
    if not args.dry_run and plan is None:
        raise ValueError("live mode requires --session-plan (engine lifecycle contract)")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"mode": "dry-run" if args.dry_run else "live", "seed": args.seed,
              "children": args.children, "hz": HZ, "seconds": args.seconds,
              "actions": list(ACTIONS), "samples": [], "errors": [], "frames": 0,
              "session_contract": "provided" if plan else "PENDING_ENGINE_API",
              "engine_acceptance": "NOT_RUN", "latency_scope": "pose RPC round-trip only"}
    children = Children(args.children, args.seed)
    ipc = None
    rpc_ms, generate_ms, lateness = [], [], []
    previous_t = None
    observed = []
    coverage = []
    displays = []
    message_id = 0
    stream = (args.output / "messages.jsonl").open("w", encoding="utf-8")

    def call(method, **params):
        nonlocal message_id
        message_id += 1
        stream.write(json.dumps({"id": message_id, "method": method, "params": params},
                                allow_nan=False) + "\n")
        if args.dry_run:
            return None
        return ipc.call(method, **params)

    def sample():
        values = {method: call(method) for method in
                  ("game.session", "paint.wet_coverage", "player.display")}
        report["samples"].append(values)
        state = read_path(values["game.session"], plan["state_path"])
        if not observed or observed[-1] != state:
            observed.append(state)
        wet = values["paint.wet_coverage"]
        if wet["valid_texels"] <= 0 or not 0 <= wet["ratio"] <= 1:
            raise ValueError("wet coverage requires a valid, nonempty paint atlas")
        coverage.append(wet["ratio"])
        displays.append(read_path(values["player.display"], plan["display_fps_path"]))

    try:
        if not args.dry_run:
            ipc = connect(args.pipe)
        call("input.clear_body_poses")
        for request in plan["setup"] if plan else []:
            call(request["method"], **request.get("params", {}))
        if not args.dry_run:
            sample()  # capture attract before starting
        for request in plan["start"] if plan else []:
            call(request["method"], **request.get("params", {}))
        start = time.perf_counter()
        for frame in range(math.ceil(args.seconds*HZ)):
            if not args.dry_run:
                wait_frame(start, frame)
                lateness.append(max(0, (time.perf_counter()-start-frame/HZ)*1000))
            before = time.perf_counter()
            message = children.frame(frame)
            validate_pose(message, previous_t)
            previous_t = message["t"]
            generate_ms.append((time.perf_counter()-before)*1000)
            before = time.perf_counter()
            call("input.set_body_poses", **message)
            if not args.dry_run:
                rpc_ms.append((time.perf_counter()-before)*1000)
            report["frames"] += 1
            if frame % 6 == 0 and not args.dry_run:
                sample()
            elif frame % 6 == 0:
                for method in ("game.session", "paint.wet_coverage", "player.display"):
                    call(method)
        # Empty input represents everyone leaving; replay includes the shutdown packet.
        call("input.set_body_poses", t=report["frames"]/HZ, persons=[])
        if not args.dry_run:
            sample()
        if args.dry_run:
            report["status"] = "SCHEMA_PASS_ENGINE_NOT_RUN"
        else:
            expected = plan["expected_states"]
            checks = {"session_cycle": observed == expected,
                      "paint_increased": max(coverage) > coverage[0],
                      "display_active": any(isinstance(fps, (int, float)) and
                                            math.isfinite(fps) and fps > 0 for fps in displays)}
            final_session = report["samples"][-1]["game.session"]
            if plan.get("required_events"):
                checks["pose_events_observed"] = all(
                    final_session.get("events", {}).get(name, 0) > 0
                    for name in plan["required_events"])
                checks["actions_succeeded"] = (final_session.get("actions_ok", 0) > 0 and
                                               final_session.get("actions_failed", -1) == 0)
            report.update(checks=checks, observed_states=observed,
                          engine_acceptance="PASS" if all(checks.values()) else "FAIL")
            report["status"] = report["engine_acceptance"]
    except KeyboardInterrupt:
        report["status"] = "FAIL"
        report["errors"].append("interrupted before session acceptance completed")
    except Exception as exc:
        report["status"] = "FAIL"
        report["errors"].append(str(exc))
    finally:
        if ipc is not None or args.dry_run:
            for request in [{"method": "input.clear_body_poses"}] + (plan["cleanup"] if plan else []):
                try:
                    call(request["method"], **request.get("params", {}))
                except Exception as exc:
                    report["errors"].append("cleanup: " + str(exc))
                    report["status"] = "FAIL"
            if ipc is not None:
                ipc.close()
        stream.close()
        if report.get("status") == "FAIL" and not args.dry_run:
            report["engine_acceptance"] = "FAIL"
        report["pose_generation"] = summary_ms(generate_ms)
        report["pose_rpc"] = summary_ms(rpc_ms)
        report["schedule_lateness"] = summary_ms(lateness)
        report["observed_hz"] = (report["frames"] / (time.perf_counter()-start)
                                 if not args.dry_run and "start" in locals() else None)
        (args.output / "acceptance.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return 1 if report["status"] == "FAIL" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--children", type=int, choices=range(1, 11), default=4)
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--seconds", type=float, default=36)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pipe", default=PIPE)
    parser.add_argument("--session-plan", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "out" / "fake")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
