"""Isolate startup costs in an already open Player through external IPC.

Run after the paint-room fixture exists. Never launches an application. Opens
temporary copies with/without particles, then leaves the original room painting
at 1024. RPC latency includes main-thread queue time; it is not a stage timer.
"""
import json
from pathlib import Path
import shutil
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/dev/.captures/player-paintroom/paintroom.rtp"
REPORT = ROOT / "docs/dev/.captures/player-paintroom/startup_isolation.json"
TARGETS = ("PaintFloor", "PaintBack", "PaintLeft", "PaintRight")


def main():
    source = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report = {"calls": [], "fixture": str(FIXTURE)}
    temporary = FIXTURE.with_name("startup_no_particles.rtp")
    paths = [temporary]
    empty = json.loads(json.dumps(source))
    empty["particle_simulation"]["systems"] = []
    empty["particle_simulation"]["active_system_index"] = -1
    temporary.write_text(json.dumps(empty), encoding="utf-8")
    for suffix in (".bin", ".shared"):
        target = Path(str(temporary) + suffix)
        shutil.copyfile(Path(str(FIXTURE) + suffix), target)
        paths.append(target)

    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)

    def call(label, method, **params):
        started = time.perf_counter()
        result = ipc.call(method, **params)
        elapsed = (time.perf_counter() - started) * 1000.0
        report["calls"].append({"label": label, "method": method,
                                "params": params, "rpc_ms": elapsed, "result": result})
        if method in ("project.open", "paint.wet_add_surface"):
            print(label, method, round(elapsed, 3), flush=True)
        return result

    try:
        for label, path in (("empty_a", temporary), ("particles_a", FIXTURE),
                            ("empty_b", temporary), ("particles_b", FIXTURE)):
            call(label, "paint.wet_clear_surfaces")
            call(label, "viewport.reset_frame_timings")
            call(label, "project.open", path=str(path))
            for _ in range(300):
                state = call(label, "agent.get_state_summary", include_probe=False)["runtime"]
                if state["scene_ready"] and state["project_path"] == str(path):
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError("project did not become ready")
            call(label, "player.set_display", continuous=True, pacing_hz=0.0)
            for _ in range(12):
                call(label, "player.display")
                time.sleep(0.02)
            call(label, "viewport.frame_timings")
            time.sleep(1.1)
            call(label + "_steady", "player.display")

        names = call("targets", "scene.list_objects")
        targets = [n for n in names if any(n == b or n.startswith(b + ".") for b in TARGETS)]
        assert len(targets) == 4, targets
        for resolution in (512, 1024):
            label = "paint_" + str(resolution)
            call(label, "paint.wet_clear_surfaces")
            # Clearing wet surfaces keeps material paint textures. Reload so
            # ensureTextureSet cannot silently reuse the previous resolution.
            call(label, "project.open", path=str(FIXTURE))
            call(label, "agent.get_state_summary", include_probe=False)
            time.sleep(0.3)
            for name in targets:
                info = call(label, "paint.wet_add_surface", object=name, resolution=resolution)
                assert info["valid_texels"] == resolution * resolution, info
                call(label + "_first_tick", "paint.wet_status")
            time.sleep(1.1)
            call(label + "_steady", "paint.wet_status")
            call(label + "_steady", "player.display")
    finally:
        ipc.close()
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for path in paths:
            path.unlink(missing_ok=True)
    print(REPORT)


if __name__ == "__main__":
    main()
