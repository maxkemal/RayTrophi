"""Compare existing/default materials using real meshes and the shared balloon core.

Never builds or launches an app. Writes ignored private project copies, opens them
in the existing Player, and reopens the original project in finally. Original files
are hashed and never saved. Reopening restores the saved camera, not unsaved motion.
The fixed demo's 8m placement and four-collider limit remain explicit test limits.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/perception_bridge/out/heavy-balloon"
FIXTURE = ROOT / "docs/dev/fixtures/balloon-paint-room/balloons.rtp"
SURFACES = ["1_Walls", "1_Ground", "1_Mixer", "1_Computer"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def hashes(source):
    return {suffix: hashlib.sha256(Path(str(source) + suffix).read_bytes()).hexdigest()
            for suffix in ("", ".bin", ".shared", ".aux.json")
            if Path(str(source) + suffix).exists()}


def prepare(source, camera, variant, collider_mode):
    project = read(source)
    assert project["has_geometry"], "embedded flat geometry required"
    assert not project["particle_simulation"]["systems"], "source has particle systems"
    project["particle_simulation"] = copy.deepcopy(read(FIXTURE)["particle_simulation"])
    for system in project["particle_simulation"]["systems"]:
        template = system["colliders"][0]
        system["colliders"] = []
        for name in SURFACES:
            collider = copy.deepcopy(template)
            collider.update(name="Heavy_" + name, source_mode=collider_mode, source_name=name)
            system["colliders"].append(collider)
    for c in project["cameras"]:
        if c.get("is_active"):
            for key in ("position", "target", "up", "fov"):
                c[key] = camera[key]
    shared = read(str(source) + ".shared")
    if variant == "default":
        neutral = copy.deepcopy(read(str(FIXTURE) + ".shared")["materials"]["materials"][0])
        neutral["albedoProperty"] = {"alpha": 1.0, "color": [0.7, 0.7, 0.7], "intensity": 1.0}
        neutral["roughnessProperty"]["intensity"] = 0.65
        for index, material in enumerate(shared["materials"]["materials"]):
            value = copy.deepcopy(neutral)
            value.update(id=material["id"], name=material["name"])
            shared["materials"]["materials"][index] = value
        project["material_node_graphs"] = {}
    destination = OUT / collider_mode / variant / "heavy.rtp"
    write(destination, project)
    write(Path(str(destination) + ".shared"), shared)
    for suffix in (".bin", ".aux.json"):
        src = Path(str(source) + suffix)
        if src.exists():
            shutil.copyfile(src, Path(str(destination) + suffix))
    return destination


def collect(ipc, seconds):
    ipc.call("viewport.reset_frame_timings")
    samples = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.2)
        samples.append({"display": ipc.call("player.display"),
                        "wet": ipc.call("paint.wet_status")})
    return {"samples": samples, "timings": ipc.call("viewport.frame_timings"),
            "telemetry": ipc.call("viewport.frame_telemetry")}


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    ipc = RtIpc(PLAYER_PIPE_NAME)
    original = {"path": ipc.call("project.path"), "camera": ipc.call("camera.get"),
                "display": ipc.call("player.display"),
                "capture": ipc.call("viewport.status")["capture_enabled"]}
    source = Path(original["path"])
    assert source.name.lower() == "oyuncak.rtp", "this probe targets the inspected heavy scene"
    assert not ipc.call("paint.wet_demo")["prepared"], "active demo must not be replaced"
    assert ipc.call("paint.wet_status")["surface_count"] == 0, "existing paint must not be replaced"
    before_hashes = hashes(source)
    test_camera = (read(args.camera_from_report)["original"]["camera"]
                   if args.camera_from_report else original["camera"])
    report = {"original": original, "source_hashes_before": before_hashes,
              "test_camera": test_camera,
              "surfaces": SURFACES, "collider_mode": args.collider_mode, "variants": {},
              "limits": ["fixed demo placement: x/z about -3..3m, y about 0.8..2.8m",
                         "only four selected real mesh colliders/surfaces, not all scene objects",
                         "one selected material section per paint target; existing UV overlaps remain",
                         "all default materials retain distinct IDs; draw batching is unchanged"]}
    changed_project = False
    try:
        for variant in args.variants:
            destination = prepare(source, test_camera, variant, args.collider_mode)
            changed_project = True
            start = time.perf_counter()
            ipc.call("project.open", path=str(destination))
            result = {"open_rpc_ms": (time.perf_counter() - start) * 1000,
                      "camera": ipc.call("camera.get"), "path": str(destination)}
            report["variants"][variant] = result
            ipc.call("viewport.capture", enabled=False)
            ipc.call("player.set_display", continuous=True, pacing_hz=0)
            time.sleep(1.5)
            result["baseline"] = collect(ipc, 3)
            config = {"surfaces": SURFACES, "resolution": 512, "seed": 417,
                      "balloons": 12, "particles_per_balloon": 1800,
                      "interval_min": 2.5, "interval_max": 5.0, "autostart": False}
            result["prepared"] = ipc.call("paint.wet_demo", operation="configure", config=config)
            assert result["prepared"]["prepared"] and not result["prepared"]["running"]
            result["wet_prepared"] = ipc.call("paint.wet_status")
            result["idle_paint"] = collect(ipc, 3)
            result["trials"] = []
            for count in (1, 3, 1, 3):
                trial = {"count": count, "coverage_before": ipc.call("paint.wet_coverage")}
                burst = ipc.call("paint.wet_demo", operation="burst", config={"count": count})
                assert burst["events"][-1]["spawned"] == count * 1800, burst
                trial["burst"] = burst["events"][-1]
                trial["window"] = collect(ipc, 4)
                trial["settled_demo"] = ipc.call("paint.wet_demo")
                trial["coverage_after"] = ipc.call("paint.wet_coverage")
                trial["wet_after"] = ipc.call("paint.wet_status")
                result["trials"].append(trial)
                write(OUT / (args.collider_mode + "-report.json"), report)
                print(json.dumps({"variant": variant, "count": count,
                                  "event": trial["settled_demo"]["events"][-1],
                                  "coverage": trial["coverage_after"],
                                  "device_lost": trial["window"]["telemetry"]["device_lost"]}), flush=True)
            # Separate diagnostic controls, excluded from burst/performance results.
            # A broad unfaced dab tests atlas stamping independently of contacts.
            result["control_before"] = ipc.call("paint.wet_coverage")
            ipc.call("paint.wet_dab", position=[0.0, 0.94, 0.0], radius=0.5,
                     color=[1.0, 0.0, 0.0], amount=1.0, facing=False)
            time.sleep(0.5)
            result["control_floor_dab"] = ipc.call("paint.wet_coverage")
            ipc.call("paint.wet_dab", position=[0.0, 1.5, 0.0], radius=10.0,
                     color=[1.0, 0.0, 0.0], amount=1.0, facing=False)
            time.sleep(0.5)
            result["control_broad_dab"] = ipc.call("paint.wet_coverage")
            result["control_wet_after"] = ipc.call("paint.wet_status")
            result["camera_after"] = ipc.call("camera.get")
            result["camera_unchanged"] = result["camera_after"] == result["camera"]
            result["burst_paint_delivered"] = all(
                trial["coverage_after"]["covered_texels"] >
                trial["coverage_before"]["covered_texels"] for trial in result["trials"])
            result["sources_dropped"] = result["trials"][-1]["wet_after"]["sources_dropped"]
            result["atlas_control_pass"] = (
                result["control_broad_dab"]["covered_texels"] ==
                result["control_broad_dab"]["valid_texels"] > 0)
            print(json.dumps({"variant": variant, "control_floor": result["control_floor_dab"],
                              "control_broad": result["control_broad_dab"]}), flush=True)
            ipc.call("paint.wet_demo", operation="stop")
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        errors = []
        if changed_project:
            try:
                ipc.call("project.open", path=str(source))
            except Exception as error:
                errors.append("reopen original: " + str(error))
        for method, params in (
            ("viewport.capture", {"enabled": original["capture"]}),
            ("player.set_display", {"continuous": original["display"]["continuous"],
                                    "pacing_hz": float(original["display"]["pacing_hz"])}),
        ):
            try:
                ipc.call(method, **params)
            except Exception as error:
                errors.append(str(error))
        report["restoration_errors"] = errors
        report["source_hashes_after"] = hashes(source)
        report["original_files_unchanged"] = before_hashes == report["source_hashes_after"]
        for method in ("project.path", "camera.get", "player.display", "paint.wet_status"):
            report.setdefault("restored", {})[method] = ipc.try_call(method)
        write(OUT / (args.collider_mode + "-report.json"), report)
        ipc.close()
    assert not report["restoration_errors"], report["restoration_errors"]
    assert report["original_files_unchanged"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=("existing", "default"),
                        default=["existing", "default"])
    parser.add_argument("--collider-mode", choices=("ObjectOBB", "ObjectMeshSDF"),
                        default="ObjectOBB")
    parser.add_argument("--camera-from-report", type=Path,
                        help="Use a previous run's original camera for matched comparisons")
    run(parser.parse_args())
