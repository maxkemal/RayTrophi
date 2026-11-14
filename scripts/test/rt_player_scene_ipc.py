"""External .rtp -> Vulkan Player acceptance; never launches an application.

Open the rebuilt Studio and Player first, then run:
    python scripts/test/rt_player_scene_ipc.py --project C:/scenes/simple.rtp

Use a static flat-mesh scene with a camera looking at visible geometry. This
loads the supplied project twice in Player and moves its camera between loads.
Studio is only queried. No scene is saved and no screenshot file is created.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from PIL import Image, ImageStat

from rt_ipc import PLAYER_PIPE_NAME, RtIpc, RtIpcError
from rt_player_bootstrap_ipc import check_player, check_studio


def runtime(ipc):
    return ipc.call("agent.get_state_summary", include_probe=False)["runtime"]


def wait_image(ipc, revision, minimum_render_calls=1, timeout=60):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = runtime(ipc)
        assert not latest["device_lost"], latest
        assert not latest["error"], latest
        if (latest["scene_revision"] == revision and latest["scene_ready"] and
                latest["frame_scene_revision"] == revision and
                latest["frame_available"] and latest["render_calls"] >= minimum_render_calls and
                not latest["render_pending"]):
            assert latest["telemetry_available"], latest
            if latest["async_present"]:
                assert latest["frames_consumed"] > 0, latest
            assert latest["draw_calls"] > 0, latest
            assert latest["visible_triangles"] > 0, latest
            return latest
        time.sleep(0.05)
    raise AssertionError(f"Player did not produce a settled Vulkan image: {latest}")


def check_image(ipc):
    result = ipc.call("viewport.get_screenshot")
    assert "image_base64" in result, result
    encoded = result["image_base64"]
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        image.load()
        assert image.width > 0 and image.height > 0
        variance = ImageStat.Stat(image.convert("RGB")).var
        assert max(variance) > 0.1, "Displayed frame is a uniform color"
    summary = ipc.call("agent.get_state_summary", include_probe=True)
    probe = summary["viewport"]["probe"]
    assert isinstance(probe, dict), summary
    assert probe["pixels"] > 0 and probe["nan_fraction"] == 0, probe
    assert probe["max_luminance"] > 0, probe


def run(project, player_only=False):
    if not player_only:
        check_studio()
    check_player()
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    capture_before = False
    try:
        capture_before = ipc.call("viewport.status")["capture_enabled"]
        ipc.call("viewport.capture", enabled=True)
        before = runtime(ipc)
        ipc.call("project.open", path=str(project))
        first = wait_image(ipc, before["scene_revision"] + 1)
        assert first["flat_meshes"] > 0 and first["triangles"] > 0, first
        assert first["material_count"] > 0 and first["material_buffer_ready"], first
        check_image(ipc)

        # A refused feature must not clear or replace the running scene.
        captures = Path(__file__).resolve().parents[2] / "docs/dev/.captures"
        captures.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=captures, prefix="player_rejection_") as directory:
            unsupported = Path(directory) / "unsupported_çalışma.rtp"
            unsupported.write_text(json.dumps({"fracture_recipes": [{}]}), encoding="utf-8")
            try:
                ipc.call("project.open", path=str(unsupported))
            except RtIpcError as error:
                assert "unsupported_in_player_scene: fracture_recipes" in str(error), str(error)
            else:
                raise AssertionError("Player silently accepted unsupported fracture data")
        after_rejection = runtime(ipc)
        assert after_rejection["scene_ready"], after_rejection
        assert after_rejection["scene_revision"] == first["scene_revision"], after_rejection
        assert after_rejection["project_path"] == first["project_path"], after_rejection

        camera_before = ipc.call("camera.get")
        ipc.call("camera.orbit", yaw=15.0, pitch=5.0)
        moved = wait_image(ipc, first["scene_revision"], first["render_calls"] + 1)
        assert ipc.call("camera.get") != camera_before, "Camera orbit did not change its state"
        check_image(ipc)

        ipc.call("project.open", path=str(project))
        reopened = wait_image(ipc, first["scene_revision"] + 1)
        assert reopened["flat_meshes"] == first["flat_meshes"], reopened
        assert reopened["triangles"] == first["triangles"], reopened
        restored_camera = ipc.call("camera.get")
        for key in ("position", "target", "up", "fov"):
            assert restored_camera[key] == camera_before[key], (key, restored_camera, camera_before)
        check_image(ipc)
        print(json.dumps({"result": "PASS", "first": first, "moved": moved, "reopened": reopened},
                         ensure_ascii=False, indent=2))
    finally:
        already_failed = sys.exc_info()[0] is not None
        try:
            ipc.call("viewport.capture", enabled=capture_before)
        except RtIpcError as error:
            if not already_failed:
                raise
            print(f"Capture restore failed after the primary failure: {error}", file=sys.stderr)
        finally:
            ipc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path,
                        help="Static .rtp; defaults to the installed EchoRoomsLab scene if present")
    parser.add_argument("--player-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=240,
                        help="Whole-test timeout, including an unresponsive pipe")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    selected = args.project or (Path(__file__).resolve().parents[2] /
                               "x64/Release/assets/acoustic_echo_lab/EchoRoomsLab.rtp")
    if not selected.is_file():
        parser.error("pass --project with an existing static flat-mesh .rtp")
    project = selected.resolve(strict=True)
    if args.worker:
        run(project, args.player_only)
    else:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--project", str(project)]
        if args.player_only:
            command.append("--player-only")
        try:
            result = subprocess.run(command, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired:
            raise SystemExit("FAIL: Player scene acceptance timed out")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
