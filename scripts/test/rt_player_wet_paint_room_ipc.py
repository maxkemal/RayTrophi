"""Player paint room: particles of several colours paint the walls of a room
on the GPU, and what it costs (Boyama Odasi steps 2+3, PLAYER_GAME_PAINT_ROOM.md).

Never launches an application.

1. Fixture (Studio open, EMPTY scene):
       python scripts/test/rt_player_wet_paint_room_ipc.py --make-fixture
   An 8 x 8 m room: floor + back/left/right walls as planes (each its own
   material: one material holds one paint texture), a thin box behind each wall
   as its OBB collider, a floor plane collider, and four particle systems (red,
   blue, yellow, green) spraying from the middle of the room at the walls and
   the floor, response die + contact events recorded. Saved under
   docs/dev/.captures/player-paintroom/ (gitignored).
2. Run (Player open):
       python scripts/test/rt_player_wet_paint_room_ipc.py [--resolution 1024] [--seconds 10]
   Opens the fixture, measures the frame cost WITHOUT paint surfaces, adds the
   four wet paint surfaces, measures again while the particles paint, then
   reads coverage per wall and colour, and counts the paint colours on screen.

Capture stays OFF while timing (it forces a synchronous present); it is only
switched on for the final screenshot, which is kept in memory.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/dev/.captures/player-paintroom/paintroom.rtp"
HALF = 4.0           # room half size (m)
HEIGHT = 3.0
SURFACES = ["PaintFloor", "PaintBack", "PaintLeft", "PaintRight"]
SPRAYS = [  # system, colour, direction
    ("SprayRed", [0.95, 0.08, 0.06], [0.0, 0.35, -1.0]),
    ("SprayBlue", [0.06, 0.20, 0.95], [-1.0, 0.35, -0.15]),
    ("SprayYellow", [0.95, 0.85, 0.05], [1.0, 0.35, -0.15]),
    ("SprayGreen", [0.10, 0.85, 0.15], [0.25, 0.9, -0.55]),
]
CAMERA_POSITION = [0.0, 1.8, 7.5]
CAMERA_TARGET = [0.0, 1.2, -1.0]


def named(result):
    return result if isinstance(result, str) else result.get("name")


def make_fixture():
    studio = RtIpc()
    try:
        names = studio.call("scene.list_objects")
        assert set(names) <= {"Default_Cube"}, f"Studio scene not empty: {names}"
        assert not studio.call("particle.list_systems")["systems"], "Studio has particle systems"
        for name in names:
            studio.call("scene.delete", name=name)

        def surface(name, translation, rotation, albedo):
            obj = studio.call("scene.add_primitive", type="plane", name=name, size=2 * HALF)
            studio.call("scene.set_transform", name=obj, translation=translation, rotation=rotation)
            mat = named(studio.call("material.create", type="principled", name=name + "Mat"))
            studio.call("material.assign", material_name=mat, object_name=obj)
            studio.call("material.set", object_name=obj, param="base_color", value=albedo)
            studio.call("material.set", object_name=obj, param="roughness", value=0.8)
            return obj

        def blocker(name, translation, scale):
            obj = studio.call("scene.add_primitive", type="cube", name=name, size=1.0)
            studio.call("scene.set_transform", name=obj, translation=translation, scale=scale)
            return obj

        made = {}
        made["floor"] = surface("PaintFloor", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.82, 0.82, 0.80])
        # Plane normal is +Y: +90 about X faces +Z (back wall looks into the room),
        # -90 / +90 about Z face +X / -X (left / right walls).
        made["back"] = surface("PaintBack", [0.0, HEIGHT / 2, -HALF], [90.0, 0.0, 0.0], [0.86, 0.86, 0.84])
        made["left"] = surface("PaintLeft", [-HALF, HEIGHT / 2, 0.0], [0.0, 0.0, -90.0], [0.86, 0.86, 0.84])
        made["right"] = surface("PaintRight", [HALF, HEIGHT / 2, 0.0], [0.0, 0.0, 90.0], [0.86, 0.86, 0.84])
        boxes = [blocker("BackWallBox", [0.0, HEIGHT / 2, -HALF - 0.11], [2 * HALF, HEIGHT, 0.2]),
                 blocker("LeftWallBox", [-HALF - 0.11, HEIGHT / 2, 0.0], [0.2, HEIGHT, 2 * HALF]),
                 blocker("RightWallBox", [HALF + 0.11, HEIGHT / 2, 0.0], [0.2, HEIGHT, 2 * HALF])]

        for system, colour, direction in SPRAYS:
            studio.call("particle.add_system", name=system)
            studio.call("particle.set_physics", system=system, execution_policy="auto",
                        particle_radius=0.03)
            emitter = studio.call("particle.add_emitter", system=system, name=system + "Nozzle",
                                  point=[0.0, 1.2, 0.5], direction=direction, speed=7.0,
                                  spread=14.0, rate_per_second=1500.0, lifetime_seconds=3.0)
            studio.call("particle.set_appearance", system=system,
                        profile_id=int(emitter["appearance_profile_id"]),
                        color_ramp=[[0.0] + colour, [1.0] + colour])
            studio.call("particle.set_system_emitter_only", system=system, emitter_only=False)
            studio.call("particle.set_collision", system=system, response="die",
                        record_events=True, event_capacity=65536)
            # Colliders belong to the ACTIVE system (collider.* scripts the
            # active runtime): each spray gets its own copies.
            studio.call("particle.set_active_system", system=system)
            for box in boxes:
                # aabb, not obb: an OBB of a scaled/translated box resolved to a
                # box around the room centre (2026-11-15) - particles died at
                # the nozzle. Walls are axis-aligned, so AABB is exact here.
                studio.call("collider.create", name=box + "Collider", source_mode="aabb",
                            source_object=box, restitution=0.1)
            studio.call("collider.create", name="FloorCollider", source_mode="plane",
                        plane_y=0.0, restitution=0.1)
        # collider.create made a default system while none existed (known trap).
        for system in studio.call("particle.list_systems")["systems"]:
            if system.get("name") not in {name for name, _, _ in SPRAYS}:
                studio.call("particle.remove_system", system=system.get("name"))
        studio.call("camera.set_position", position=CAMERA_POSITION)
        studio.call("camera.set_target", target=CAMERA_TARGET)
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        studio.call("project.save", path=str(FIXTURE))
        print(json.dumps({"fixture": str(FIXTURE), "objects": studio.call("scene.list_objects"),
                          "systems": [s.get("name") for s in studio.call("particle.list_systems")["systems"]]},
                         indent=2))
    finally:
        studio.close()


def runtime(ipc):
    return ipc.call("agent.get_state_summary", include_probe=False)["runtime"]


def frame_row(ipc, seconds):
    time.sleep(1.2)
    deadline = time.monotonic() + seconds
    last = None
    paint = []
    while time.monotonic() < deadline:
        time.sleep(0.5)
        last = ipc.call("player.display")
        st = ipc.call("paint.wet_status")
        paint.append((st["last_tick_ms"], st["active_tiles"], st["ticked_tiles"], st["stamped_texels"]))
    frame = last["frame"]
    row = {k: round(float(frame.get(k, 0.0)), 2) for k in (
        "rendered_fps", "frame_cost_ms", "capacity_fps", "backend_frame_ms_avg",
        "cpu_record_ms_avg", "upload_ms_avg", "advance_ms_avg", "present_ms_avg")}
    row["render"] = last["render"]
    if paint:
        row["paint_tick_ms_max"] = round(max(p[0] for p in paint), 3)
        row["paint_tick_ms_avg"] = round(sum(p[0] for p in paint) / len(paint), 3)
        row["active_tiles_max"] = max(p[1] for p in paint)
        row["ticked_tiles_max"] = max(p[2] for p in paint)
        row["stamped_texels_max"] = max(p[3] for p in paint)
    return row


def colour_counts(ipc):
    from PIL import Image
    ipc.call("viewport.capture", enabled=True)
    try:
        time.sleep(0.8)
        encoded = ipc.call("viewport.get_screenshot")["image_base64"]
    finally:
        ipc.call("viewport.capture", enabled=False)
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    image.thumbnail((640, 640))
    counts = {"red": 0, "blue": 0, "yellow": 0, "green": 0}
    for r, g, b in image.getdata():
        if r > g + 70 and r > b + 70:
            counts["red"] += 1
        elif b > r + 70 and b > g + 40:
            counts["blue"] += 1
        elif r > b + 70 and g > b + 70:
            counts["yellow"] += 1
        elif g > r + 60 and g > b + 60:
            counts["green"] += 1
    return counts


def run(resolution, seconds):
    assert FIXTURE.exists(), "run --make-fixture first (Studio)"
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    original = ipc.call("player.display")
    report = {"resolution": resolution}
    try:
        ipc.call("viewport.capture", enabled=False)
        # Wet paint is process state: surfaces of an earlier run on the same
        # project survive project.open, and the baseline would not be paint-free.
        ipc.call("paint.wet_clear_surfaces")
        ipc.call("paint.wet_set_splat", consume_particles=True, particle_system="")
        revision = runtime(ipc)["scene_revision"]
        ipc.call("project.open", path=str(FIXTURE))
        deadline = time.monotonic() + 90
        while not (runtime(ipc)["scene_ready"] and runtime(ipc)["scene_revision"] == revision + 1):
            assert time.monotonic() < deadline, runtime(ipc)
            time.sleep(0.1)
        ipc.call("player.set_display", pacing_hz=0.0, continuous=True)
        time.sleep(2.0)  # sprays reach steady state
        sim = runtime(ipc)["simulation"]
        report["particles"] = {"systems": sim.get("particle_systems"), "alive": sim.get("particles_alive")}
        report["no_paint"] = frame_row(ipc, seconds / 2)
        assert report["no_paint"].get("active_tiles_max", 0) == 0, ("baseline is not paint-free", report)

        surfaces = {}
        # Real names: a deleted name stays taken in Studio, so the fixture may
        # carry ".001" suffixes.
        present = ipc.call("scene.list_objects")
        names = [n for n in present if any(n == b or n.startswith(b + ".") for b in SURFACES)]
        assert len(names) == len(SURFACES), ("paint surfaces missing", present)
        for name in names:
            info = ipc.call("paint.wet_add_surface", object=name, resolution=resolution)
            surfaces[name] = {"valid": info["valid_texels"], "tiles": info["tiles"]}
        ipc.call("paint.wet_set_splat", radius=0.08, amount=0.9, particle_system="")
        report["surfaces"] = surfaces
        report["painting"] = frame_row(ipc, seconds)
        st = ipc.call("paint.wet_status")
        report["paint"] = {k: st[k] for k in (
            "status", "compute_context", "publish_mode", "publish_device_copies",
            "publish_host_downloads", "publish_bytes_downloaded", "device_event_dispatches",
            "host_events_applied", "sources_dropped", "tiles")}
        cov = ipc.call("paint.wet_coverage")
        report["coverage"] = {c["object"]: {"ratio": round(c["ratio"], 4), "wet": c["wet_texels"],
                                            "mean_color": [round(v, 2) for v in c["mean_color"]]}
                              for c in cov["surfaces"]}
        report["coverage_total"] = round(cov["ratio"], 4)
        report["screen_colours"] = colour_counts(ipc)
        report["runtime_publishes"] = runtime(ipc)["simulation"].get("wet_paint_publishes")
        print(json.dumps(report, indent=2))
    finally:
        try:
            ipc.call("player.set_display", pacing_hz=float(original.get("pacing_hz", 60.0)),
                     continuous=bool(original.get("continuous", False)))
        finally:
            ipc.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-fixture", action="store_true")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.make_fixture:
        make_fixture()
    else:
        run(args.resolution, args.seconds)


if __name__ == "__main__":
    main()
