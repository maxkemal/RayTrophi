"""Create/listen to a persistent echo lab through external Studio IPC.

Preserves existing geometry/audio. Creates a separate saved project on setup.
Use --room hard|soft|corridor to move the listener and audition that room.
Use --dry for an A/B without the live acoustic processing.
"""
import argparse
import json
import math
from pathlib import Path
import random
import struct
import time
import wave

from rt_ipc import RtIpc

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "x64/Release/assets/acoustic_echo_lab"
STATE = FOLDER / "session.json"
BANDS = (125, 250, 500, 1000, 2000, 4000)
ROOMS = {
    "hard": {"center": 100, "size": [60, 24, 80], "listener": [100, 3, 0]},
    "soft": {"center": 220, "size": [60, 24, 80], "listener": [220, 3, 0]},
    "corridor": {"center": 340, "size": [12, 12, 110], "listener": [340, 3, 0]},
}


def sound(path, kind, onset):
    """Dry mono fixtures; no baked-in reverb. Six-second cycles isolate echoes."""
    rate = 48000
    rng = random.Random(173 + int(onset * 10))
    samples = []
    smooth = 0
    for index in range(rate * 6):
        t = index / rate - onset
        value = 0
        if 0 <= t < 0.18:
            attack = min(1, t / 0.0008)
            if kind == "clap":
                noise = rng.uniform(-1, 1)
                smooth += 0.12 * (noise - smooth)
                value = (noise - smooth) * math.exp(-t * 160) * attack
            elif kind == "wood":
                value = sum(math.sin(2 * math.pi * f * t) * math.exp(-t * decay)
                            for f, decay in ((330, 70), (730, 95), (1330, 130))) / 3
                value *= attack
            else:
                value = math.sin(2 * math.pi * (250 * t + 14000 * t * t))
                value *= math.sin(math.pi * min(t / 0.09, 1)) ** 2 if t < 0.09 else 0
        samples.append(value)
    peak = max(abs(value) for value in samples)
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(b"".join(struct.pack("<h", round(value / peak * 19000))
                                 for value in samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--room", choices=ROOMS, default="hard")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    c = RtIpc()
    FOLDER.mkdir(parents=True, exist_ok=True)
    try:
        if args.setup:
            if STATE.exists():
                raise RuntimeError("Lab already exists; omit --setup to change rooms")
            state = {"before": {"project": c.call("project.path"),
                                "camera": c.call("camera.get"), "audio": c.call("audio.list"),
                                "status": c.call("audio.status"),
                                "preview": c.call("audio.acoustics_preview_status"),
                                "render": c.call("render.get_settings")},
                     "objects": [], "profiles": [], "assets": [], "sources": {},
                     "rooms": ROOMS, "completed": False}

            def remember():
                STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

            remember()
            for voice in state["before"]["status"]["voices"]:
                if voice["playing"]:
                    c.call("audio.pause", id=voice["id"])

            def cube(label, position, scale, profile):
                name = c.call("scene.add_primitive", type="cube", name="EchoLab " + label, size=1)
                state["objects"].append(name)
                remember()
                c.call("scene.set_transform", name=name, translation=position, scale=scale)
                c.call("substance.set_object", object=name, binding={"substance": profile})

            for key, room in ROOMS.items():
                profile = "EchoLab " + key
                c.call("substance.derive", name=profile, based_on="Stone")
                state["profiles"].append(profile)
                remember()
                fields = {"acoustic_defined": True,
                          "acoustic_scattering": 0.02 if key != "soft" else 0.3}
                absorption = [0.04] * 6 if key != "soft" else [0.6, 0.75, 0.88, 0.95, 0.98, 0.99]
                for band, loss in zip(BANDS, absorption):
                    fields[f"acoustic_absorption_{band}"] = loss
                    fields[f"acoustic_transmission_{band}"] = 0
                c.call("substance.set", name=profile, fields=fields)
                x = room["center"]
                width, height, length = room["size"]
                for label, position, scale in (
                    ("floor", [x, -0.2, 0], [width, 0.4, length]),
                    ("ceiling", [x, height + 0.2, 0], [width, 0.4, length]),
                    ("left", [x - width / 2 - 0.2, height / 2, 0], [0.4, height, length]),
                    ("right", [x + width / 2 + 0.2, height / 2, 0], [0.4, height, length]),
                    ("front", [x, height / 2, length / 2 + 0.2], [width, height, 0.4]),
                    ("back", [x, height / 2, -length / 2 - 0.2], [width, height, 0.4]),
                ):
                    cube(key + " " + label, position, scale, profile)
                if key == "corridor":
                    for index, z in enumerate((-32, -16, 18, 34)):
                        cube(f"corridor side baffle {index}",
                             [x + (-4.5 if index % 2 else 4.5), 6, z], [3, 12, 0.5], profile)

            for kind, onset in (("clap", 0.1), ("wood", 2.1), ("chirp", 4.1)):
                path = FOLDER / (kind + "_dry_spaced_mono.wav")
                sound(path, kind, onset)
                asset = c.call("audio.asset_import", path=str(path),
                               author="Procedural diagnostic fixture", source="rt_setup_echo_rooms.py",
                               tags=["dry", "echo-test", "mono", "six-second-cycle"])["id"]
                state["assets"].append(asset)
                for key, room in ROOMS.items():
                    if key == "soft" and kind != "clap":
                        continue
                    if key == "corridor" and kind == "wood":
                        continue
                    offset = {"clap": [4, 3, 8], "wood": [-5, 3, -7], "chirp": [2, 4, -12]}[kind]
                    position = [room["center"] + offset[0], offset[1], offset[2]]
                    source = c.call("audio.create", name=f"EchoLab {key} {kind}", asset_id=asset,
                                    position=position, spatial=True, loop=True, gain=0.45,
                                    min_distance=4, max_distance=150, rolloff=1)["id"]
                    state["sources"].setdefault(key, []).append(source)
                    remember()
            state["completed"] = True
            remember()
        else:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        assert state["completed"], "Setup was interrupted; inspect session.json"
        available = {e["id"]: e for e in c.call("audio.list")["emitters"]}
        for ids in state["sources"].values():
            for source in ids:
                assert available[source]["name"].startswith("EchoLab "), "Open the lab project first"
                c.call("audio.stop", id=source)
        c.call("audio.playback_update", mode="continuous")
        listener = ROOMS[args.room]["listener"]
        c.call("camera.set_position", position=listener)
        c.call("camera.set_target", target=[listener[0] + 4, listener[1], listener[2] + 8])
        c.call("camera.set_fov", fov=75)
        # Solid mode can retain an older raster AS after geometry edits.
        c.call("viewport.set_shading", mode="material")
        c.call("render.set_settings", debug_view=0)
        for source in state["sources"][args.room]:
            c.call("audio.acoustics_preview", id=source, enabled=not args.dry,
                   spectral=True, reflections=True)
        c.call("audio.select", id=state["sources"][args.room][0])
        c.call("audio.workspace_update", acoustic_band=-1, acoustic_all_sources=True)
        started = c.call("batch", calls=[{"method": "audio.play", "params": {"id": source}}
                                         for source in state["sources"][args.room]])
        assert all("result" in item for item in started), started
        deadline = time.monotonic() + 15
        while not args.dry:
            current = {entry["id"]: entry for entry in c.call("audio.acoustics_preview_status")}
            if all(current[source].get("solver", {}).get("state") == "ready"
                   for source in state["sources"][args.room]):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Acoustic GPU scene is not ready; inspect preview status")
            time.sleep(0.2)
        report = {"room": args.room, "dry": args.dry,
                  "preview": c.call("audio.acoustics_preview_status"),
                  "status": c.call("audio.status"), "mixer": c.call("audio.mixer_get")}
        (FOLDER / (args.room + ("_dry" if args.dry else "_wet") + "_report.json")).write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        if args.setup:
            c.call("project.save", path=str(FOLDER / "EchoRoomsLab.rtp"))
        print(json.dumps({"room": args.room, "dry": args.dry,
                          "playing": [v["id"] for v in report["status"]["voices"] if v["playing"]],
                          "sources": [{"id": e["id"], "status": e["status"],
                                       "delays_ms": [round(p["excess_delay_seconds"] * 1000, 2)
                                                     for p in e["reflection_paths"]]}
                                      for e in report["preview"]
                                      if e["id"] in state["sources"][args.room]]}, indent=2))
    finally:
        c.close()


if __name__ == "__main__":
    main()
