"""Drive an authored rig/proxy set from fake body input over external IPC.

Requires a user-built C2 executable, an existing rig with IK controls, and its
existing proxy set. Does not create/load scenes, build or start applications.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

try:
    from .fake_children import Children
    from .transport import PIPE, connect, summary_ms, wait_frame
except ImportError:
    from fake_children import Children
    from transport import PIPE, connect, summary_ms, wait_frame


def run(args):
    report = {"status": "RUNNING", "checks": {}, "samples": [], "errors": [],
              "domain_acceptance": "NOT_RUN", "scope": "rig/proxy correctness; CPU timings only"}
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ipc = None
    owned = False
    bound = False

    def check(name, value):
        report["checks"][name] = bool(value)
        if not value:
            raise AssertionError(name)

    def binding():
        rows = ipc.call("anim.external_poses")["bindings"]
        return next(row for row in rows if row["character"] == args.character)

    def sample():
        return ipc.call("physics.collider.proxy_set.sample", set_id=args.proxy_set, dt=1/30)

    def profile():
        return {row["name"]: row for row in ipc.call("perf.list")
                if "voxelize_colliders" in row["name"] or "solid_face_weights" in row["name"]}

    try:
        ipc = connect(args.pipe)
        check("no_live_people", not ipc.call("input.body_poses")["persons"])
        existing = ipc.call("anim.external_poses")["bindings"]
        check("character_not_already_bound", not any(
            row["character"] == args.character for row in existing))
        original = ipc.call("physics.collider.proxy_set.get", set_id=args.proxy_set)
        check("authored_set_matches", original["target_character"] == args.character and
              original["enabled"] and bool(original["proxies"]))
        baseline_stamps = ipc.call("physics.collider.proxy_set.solver_stamps")
        before_steps = {system["system_id"]: system["steps"] for system in baseline_stamps}
        report["baseline_solver_stamps"] = baseline_stamps
        profile_before = profile()
        report["domain_profile_before"] = profile_before
        owned = True
        ipc.call("input.clear_body_poses")
        ipc.call("anim.set_external_pose", character=args.character,
                 person_id=417, config=config)
        bound = True
        check("no_person_no_ghost_proxies", all(not p["resolved"] for p in sample()))
        invalid = copy.deepcopy(config)
        invalid["min_confidence"] = True
        try:
            ipc.call("anim.set_external_pose", character=args.character,
                     person_id=417, config=invalid)
        except Exception as exc:
            check("invalid_config_rejected", str(exc).startswith("anim.set_external_pose:"))
        else:
            check("invalid_config_rejected", False)
        check("invalid_config_atomic", binding()["config"] == config)

        child = Children(1)
        rpc = []
        start = time.perf_counter()
        first_centres = {}
        moved = False
        for frame in range(math.ceil(args.seconds*30)):
            wait_frame(start, frame)
            packet = child.frame(frame)
            for person in packet["persons"]:
                person["id"] = 417
            before = time.perf_counter()
            ipc.call("input.set_body_poses", **packet)
            rpc.append((time.perf_counter()-before)*1000)
            if frame % 6 == 0:
                status = binding()
                proxies = sample()
                check("tracking", status["state"] == "tracking" and status["collider_available"])
                resolved = [p for p in proxies if p["resolved"]]
                check("resolved_proxies", bool(resolved))
                for proxy in resolved:
                    before_centre = first_centres.setdefault(proxy["proxy_id"], proxy["center"])
                    moved |= math.dist(before_centre, proxy["center"]) > 0.02
                report["samples"].append({"t": frame/30, "binding": status, "proxies": proxies})
        check("proxy_motion", moved)
        report["pose_rpc"] = summary_ms(rpc)
        stamps = ipc.call("physics.collider.proxy_set.solver_stamps")
        report["solver_stamps"] = stamps
        profile_after = profile()
        report["domain_profile_after"] = profile_after
        report["domain_profile_delta"] = {}
        for name, after in profile_after.items():
            before = profile_before.get(name, {"count": 0, "total_ms": 0})
            count = after["count"] - before["count"]
            report["domain_profile_delta"][name] = {
                "count": count,
                "mean_cpu_ms": (after["total_ms"] - before["total_ms"])/count if count > 0 else None,
                "max_ms_cumulative": after["max_ms"]}
        stamped = any(system["steps"] > before_steps.get(system["system_id"], 0) and
                      stamp["set_id"] == args.proxy_set and stamp["stamped_cells"] > 0
                      for system in stamps for stamp in system["stamps"])
        if args.require_stamps:
            check("domain_consumed_existing_proxy_set", stamped)
            report["domain_acceptance"] = "PASS"
        elif stamped:
            report["domain_acceptance"] = "STAMP_OBSERVED_NOT_REQUIRED"

        deadline = time.monotonic() + 3
        while ipc.call("input.body_poses")["persons"]:
            if time.monotonic() > deadline:
                raise TimeoutError("body did not expire")
            time.sleep(0.03)
        check("tracking_loss_no_ghost_proxies", all(not p["resolved"] for p in sample()))
        ipc.call("anim.clear_external_pose", character=args.character)
        bound = False
        check("authored_proxy_set_preserved", ipc.call(
            "physics.collider.proxy_set.get", set_id=args.proxy_set) == original)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["errors"].append(str(exc))
    finally:
        if ipc is not None:
            try:
                if owned:
                    ipc.call("input.clear_body_poses")
                if bound:
                    ipc.call("anim.clear_external_pose", character=args.character)
            except Exception as exc:
                report["status"] = "FAIL"
                report["errors"].append("cleanup: " + str(exc))
            ipc.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "errors": report["errors"], "domain": report["domain_acceptance"]}))
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--proxy-set", type=int, required=True)
    parser.add_argument("--pipe", default=PIPE)
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--require-stamps", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"out/external-rig.json")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 3 <= args.seconds <= 60 or args.proxy_set <= 0:
        parser.error("seconds must be 3..60 and proxy-set positive")
    raise SystemExit(run(args))
