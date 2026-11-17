"""Generate and exercise the balloon paint demo; never starts or builds an app.

--make-fixture copies the existing spray-room's flat geometry and analytic
colliders, removes continuous emission, and writes the startup sidecar. No live
Studio session is touched. Default run requires the updated Player already open.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/dev/.captures/player-paintroom/paintroom.rtp"
FIXTURE = ROOT / "docs/dev/fixtures/balloon-paint-room/balloons.rtp"
REPORT = ROOT / "docs/dev/.captures/balloon-paint-room/acceptance.json"
SYSTEMS = ("BalloonPaintRed", "BalloonPaintBlue", "BalloonPaintYellow", "BalloonPaintGreen")
SURFACES = ("PaintFloor", "PaintBack", "PaintLeft", "PaintRight")


def make_fixture(source=SOURCE, destination=FIXTURE):
    project = json.loads(source.read_text(encoding="utf-8"))
    systems = project["particle_simulation"]["systems"]
    assert len(systems) == 4, "source must contain the four spray systems"
    for system, name in zip(systems, SYSTEMS):
        system["name"] = name
        assert {c["source_mode"] for c in system["colliders"]} == {"ObjectAABB", "PlaneY"}
        assert not system["domains"]
        for emitter in system["emitters"]:
            emitter["rate_per_second"] = 0.0
            emitter["burst_count"] = 0
            emitter["keyframes"] = []
        system["settings"]["physics"]["collision_response"] = "die"
        system["settings"]["physics"]["record_collision_events"] = True
        system["settings"]["physics"]["collision_event_capacity"] = 65536
    project["project_name"] = "Balloon Paint Room"
    project["description"] = "Seeded single-colour balloons; physical one-shot paint bursts."
    config = {"surfaces": list(SURFACES), "resolution": 1024, "seed": 417,
              "balloons": 12, "particles_per_balloon": 1800,
              "interval_min": 2.5, "interval_max": 5.0, "autostart": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(project, separators=(",", ":")), encoding="utf-8")
    for suffix in (".bin", ".shared"):
        shutil.copyfile(Path(str(source) + suffix), Path(str(destination) + suffix))
    Path(str(destination) + ".balloons.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    print(destination)


def run(seconds):
    assert seconds >= 12.0, "use at least 12 seconds to observe random bursts"
    assert FIXTURE.exists(), "generate the fixture with --make-fixture first"
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    report = {}
    original = ipc.call("player.display")
    try:
        # Fail before replacing the user's scene when the executable is old.
        ipc.call("paint.wet_demo", operation="status")
        started = time.perf_counter()
        ipc.call("project.open", path=str(FIXTURE))
        report["open_rpc_ms"] = (time.perf_counter() - started) * 1000.0
        ready = ipc.call("paint.wet_demo", operation="stop")
        assert ready["prepared"] and not ready["running"], ready
        assert len(ready["balloons"]) == 12
        assert all(b["inflated"] and 0 <= b["color"] < 4 for b in ready["balloons"])
        wet = ipc.call("paint.wet_status")
        assert len(wet["surfaces"]) == 4
        assert all(s["valid_texels"] == 1024**2 for s in wet["surfaces"]), wet
        report["startup"] = ready
        ipc.call("player.set_display", pacing_hz=0.0, continuous=True)

        def wait_game(seconds_to_wait):
            begin = ipc.call("paint.wet_demo", operation="status")["time_s"]
            deadline = time.monotonic() + 60.0
            while True:
                time.sleep(0.05)
                state = ipc.call("paint.wet_demo", operation="status")
                assert not state["error"], state
                if state["time_s"] - begin >= seconds_to_wait:
                    return state
                assert time.monotonic() < deadline, "demo clock stopped"

        wait_game(1.5)
        report["idle_display"] = ipc.call("player.display")
        for count in (1, 3):
            before = ipc.call("paint.wet_demo", operation="status")["events_total"]
            event = ipc.call("paint.wet_demo", operation="burst", config={"count": count})
            assert event["events_total"] == before + 1, event
            assert event["events"][-1]["simultaneous"] == count, event
            assert event["events"][-1]["spawned"] == count * 1800, event
            settled = wait_game(4.0)
            event = settled["events"][-1]
            assert event["frames"] > 0, event
            report["single" if count == 1 else "triple"] = event
            runtime = ipc.call("agent.get_state_summary", include_probe=False)["runtime"]
            # Only the twelve stationary balloon visuals remain after droplet lifetime.
            assert runtime["simulation"]["particles_alive"] == 12, runtime["simulation"]
            assert not runtime["device_lost"], runtime

        # Identical validation semantics travel through the shared API/core.
        from rt_ipc import RtIpcError
        before = ipc.call("paint.wet_demo", operation="status")["events_total"]
        for invalid in (0, 4, 1.5, "2", None, 4294967297):
            try:
                ipc.call("paint.wet_demo", operation="burst", config={"count": invalid})
            except RtIpcError:
                pass
            else:
                raise AssertionError("invalid burst count was accepted")
        assert ipc.call("paint.wet_demo", operation="status")["events_total"] == before

        ipc.call("paint.wet_demo", operation="start")
        report["random"] = wait_game(seconds)
        assert report["random"]["events_total"] > before
        report["coverage"] = ipc.call("paint.wet_coverage")
        assert report["coverage"]["covered_texels"] > 0, report["coverage"]
        report["wet"] = ipc.call("paint.wet_status")
        assert report["wet"]["publish_host_downloads"] == 0, report["wet"]
        report["display"] = ipc.call("player.display")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(REPORT), "prepare_ms": ready["prepare_ms"],
                          "single": report["single"], "triple": report["triple"]}, indent=2))
    finally:
        ipc.call("player.set_display", pacing_hz=float(original.get("pacing_hz", 60.0)),
                 continuous=bool(original.get("continuous", False)))
        ipc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-fixture", action="store_true")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.make_fixture:
        make_fixture()
    else:
        run(args.seconds)


if __name__ == "__main__":
    main()
