"""Player material acceptance: multi-material + textured scene, Studio vs Player.

Never launches an application.

1. Fixture (Studio open; the startup cube is the only object it will remove):
       python scripts/test/rt_player_materials_ipc.py --make-fixture
2. Acceptance (Player open; Studio optional - with it the two images are compared):
       python scripts/test/rt_player_materials_ipc.py

Checks per object region (projected from known world positions is fragile, so the
fixture keeps the camera fixed and the test samples fixed screen fractions):
a red, a green, a blue cube and a checker-textured ground must show their colours
and the checker must show contrast. Images are kept in memory only.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageStat

from rt_ipc import PLAYER_PIPE_NAME, RtIpc

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "docs/dev/.captures/player-materials"
FIXTURE = FIXTURE_DIR / "materials.rtp"
CHECKER = FIXTURE_DIR / "checker.png"

CUBES = [  # name, x, colour, extra params
    ("MatRed", -1.6, [0.85, 0.08, 0.06], {"roughness": 0.4}),
    ("MatGreen", 0.0, [0.08, 0.75, 0.10], {"metallic": 0.0, "roughness": 0.8}),
    ("MatBlue", 1.6, [0.06, 0.12, 0.90], {"roughness": 0.25}),
]
CAMERA_POSITION = [0.0, 2.2, 6.0]
CAMERA_TARGET = [0.0, 0.4, 0.0]


def write_checker(path, cells=8, size=512):
    image = Image.new("RGB", (size, size))
    step = size // cells
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            dark = ((x // step) + (y // step)) % 2 == 0
            pixels[x, y] = (30, 30, 30) if dark else (235, 235, 235)
    image.save(path)


def make_fixture():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    write_checker(CHECKER)
    studio = RtIpc()
    try:
        names = studio.call("scene.list_objects")
        assert set(names) <= {"Default_Cube"}, f"Studio scene not empty: {names}"
        for name in names:
            studio.call("scene.delete", name=name)
        ground = studio.call("scene.add_primitive", type="plane", name="MatGround", size=8.0)
        material = studio.call("material.create", type="principled", name="CheckerGround")
        material = material if isinstance(material, str) else material.get("name")
        studio.call("material.assign", material_name=material, object_name=ground)
        studio.call("material.set_texture", material_name=material, path=str(CHECKER),
                    slot="albedo")
        for name, x, colour, extra in CUBES:
            cube = studio.call("scene.add_primitive", type="cube", name=name, size=1.0)
            studio.call("scene.set_transform", name=cube, translation=[x, 0.5, 0.0])
            mat = studio.call("material.create", type="principled", name=name + "Mat")
            mat = mat if isinstance(mat, str) else mat.get("name")
            studio.call("material.assign", material_name=mat, object_name=cube)
            studio.call("material.set", object_name=cube, param="base_color", value=colour)
            for param, value in extra.items():
                studio.call("material.set", object_name=cube, param=param, value=float(value))
        studio.call("camera.set_position", position=CAMERA_POSITION)
        studio.call("camera.set_target", target=CAMERA_TARGET)
        studio.call("project.save", path=str(FIXTURE))
        print(json.dumps({"fixture": str(FIXTURE), "materials": studio.call("material.list")},
                         indent=2, default=str))
    finally:
        studio.close()


def screenshot(ipc):
    encoded = ipc.call("viewport.get_screenshot")["image_base64"]
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    image.load()
    return image.convert("RGB")


def region_mean(image, fx, fy, half=0.02):
    w, h = image.size
    box = (int((fx - half) * w), int((fy - half) * h), int((fx + half) * w), int((fy + half) * h))
    crop = image.crop(box)
    pixels = list(crop.getdata())
    return [sum(p[c] for p in pixels) / len(pixels) for c in range(3)]


def wait_frame(ipc, revision, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = ipc.call("agent.get_state_summary", include_probe=False)["runtime"]
        assert not r["error"] and not r["device_lost"], r
        if r["scene_ready"] and r["scene_revision"] == revision and r["frame_available"] \
                and not r["render_pending"]:
            return r
        time.sleep(0.1)
    raise AssertionError("Player produced no settled frame")


def run(save_dir=None):
    assert FIXTURE.exists(), "run --make-fixture first"
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    before_capture = ipc.call("viewport.status")["capture_enabled"]
    try:
        ipc.call("viewport.capture", enabled=True)
        revision = ipc.call("agent.get_state_summary", include_probe=False)["runtime"]["scene_revision"]
        ipc.call("project.open", path=str(FIXTURE))
        state = wait_frame(ipc, revision + 1)
        time.sleep(0.5)
        image = screenshot(ipc)
        if save_dir:
            image.save(Path(save_dir) / "player_materials.png")
        report = {"material_count": state["material_count"],
                  "default_material_used": state["default_material_used"]}
        assert state["material_count"] >= 4 and not state["default_material_used"], report
        # Fixed camera: cubes sit left / centre / right at mid height.
        samples = {name: region_mean(image, fx, 0.52)
                   for (name, _, _, _), fx in zip(CUBES, (0.30, 0.50, 0.70))}
        report["samples"] = samples
        r, g, b = samples["MatRed"], samples["MatGreen"], samples["MatBlue"]
        assert r[0] > 1.5 * max(r[1], r[2]), ("red cube not red", report)
        assert g[1] > 1.3 * max(g[0], g[2]), ("green cube not green", report)
        assert b[2] > 1.5 * max(b[0], b[1]), ("blue cube not blue", report)
        # Checker ground: luminance spread across a strip in front of the cubes.
        strip = [sum(region_mean(image, fx, 0.85, 0.01)) / 3 for fx in
                 [0.2 + 0.03 * i for i in range(21)]]
        report["ground_strip"] = [round(v) for v in strip]
        assert max(strip) - min(strip) > 60, ("ground texture missing (no checker contrast)", report)
        # Editor reference grid must not be drawn: the band above the ground is
        # flat background in a game (measured 3.6 std with the grid, 0 without).
        band = image.convert("L").crop((0, int(0.30 * image.height), image.width,
                                        int(0.30 * image.height) + 6))
        band_std = ImageStat.Stat(band).stddev[0]
        report["grid_band_std"] = round(band_std, 2)
        assert band_std < 1.0, ("editor grid visible in the Player", report)

        # Live material edit over the Player's own IPC: red cube -> yellow.
        render_calls = ipc.call("agent.get_state_summary", include_probe=False)["runtime"]["render_calls"]
        ipc.call("material.set", object_name="MatRed", param="base_color", value=[0.9, 0.8, 0.05])
        deadline = time.monotonic() + 20
        while True:
            r = ipc.call("agent.get_state_summary", include_probe=False)["runtime"]
            if r["render_calls"] > render_calls + 2 and r["frame_available"] and not r["render_pending"]:
                break
            assert time.monotonic() < deadline, ("no frame after material.set", r)
            time.sleep(0.1)
        time.sleep(0.3)
        edited = region_mean(screenshot(ipc), 0.30, 0.52)
        report["red_after_edit"] = edited
        assert edited[1] > 1.5 * samples["MatRed"][1] and edited[0] > 80,             ("material.set did not reach the Player image", report)
        print(json.dumps({"result": "PASS", **report}, indent=2))
    finally:
        ipc.call("viewport.capture", enabled=before_capture)
        ipc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-fixture", action="store_true")
    parser.add_argument("--save-dir", help="Write the Player frame here (scratch only)")
    args = parser.parse_args()
    if args.make_fixture:
        make_fixture()
    else:
        run(args.save_dir)


if __name__ == "__main__":
    sys.exit(main())
