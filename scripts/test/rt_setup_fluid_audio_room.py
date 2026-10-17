"""Build a saved acoustic room around the existing fluid fixture via external IPC.

Run once with Studio open. Preserves a before-project and incremental session
checkpoint. Does not launch Studio, compile, or replace the fluid solver settings.
The water loop is an existing CC0 recording driven by measured solver activity;
the separate dry clap is a manual reflection reference, not a splash detector.
"""
import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="Restore session-only acoustics after opening the saved room")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    folder = root / "x64/Release/assets/fluid_audio_room"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "session.json"
    if args.resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        c = RtIpc()
        try:
            assert Path(c.call("project.path")).resolve() == Path(state["project"]).resolve(), \
                "Open FluidAudioRoom.rtp first"
            emitters = {e["id"]: e for e in c.call("audio.list")["emitters"]}
            for voice in state["sources"].values():
                assert voice in emitters, "Saved room source missing"
            for voice in state["sources"].values():
                c.call("audio.acoustics_preview", id=voice, enabled=True,
                       spectral=True, reflections=True)
            c.call("viewport.set_shading", mode="material")
            print(json.dumps(c.call("audio.acoustics_preview_status"), indent=2))
        finally:
            c.close()
        return
    assert not checkpoint.exists(), "Already configured; use --resume after reopening"
    c = RtIpc()
    try:
        domains = c.call("fluid.list_domains")["domains"]
        assert len(domains) == 1, "This fixture expects one existing fluid domain"
        domain = domains[0]
        assert domain["domain_min"] == [-2.5, -2.5, -2.5]
        assert domain["domain_max"] == [2.5, 2.5, 2.5]
        objects = c.call("scene.list_objects")
        assert objects == ["Default_Cube"], "Expected the user's single-cube fluid fixture"
        before_audio = c.call("audio.list")
        assert not before_audio["emitters"], "Preserve existing audio; use a fresh fixture"
        row = next(r for r in c.call("audio.domains")
                   if r["domain_name"] == domain["name"] and
                   r["domain_channel"] == "fluid_flow")
        state = {"before": {"project": c.call("project.path"), "audio": before_audio,
                            "camera": c.call("camera.get"), "domain": domain,
                            "flows": c.call("flow_source.list"),
                            "cube_transform": c.call("scene.get_transform", name="Default_Cube"),
                            "cube_substance": c.call("substance.get_object", object="Default_Cube")},
                 "objects": [], "sources": {}, "assets": [], "completed": False}

        def remember():
            checkpoint.write_text(json.dumps(state, indent=2), encoding="utf-8")

        remember()
        c.call("project.save", path=str(folder / "FluidBeforeAudioRoom.rtp"))
        profile = "Fluid Audio Room - hard diagnostic"
        c.call("substance.derive", name=profile, based_on="Stone")
        state["profile"] = profile
        remember()
        fields = {"acoustic_defined": True, "acoustic_scattering": 0.02}
        for band, absorption in zip((125, 250, 500, 1000, 2000, 4000),
                                    (0.04, 0.04, 0.05, 0.06, 0.08, 0.10)):
            fields[f"acoustic_absorption_{band}"] = absorption
            fields[f"acoustic_transmission_{band}"] = 0
        c.call("substance.set", name=profile, fields=fields)
        # Floor matches the closed fluid grid bottom; no new physics collider.
        c.call("scene.set_transform", name="Default_Cube",
               translation=[0, -2.8, 0], scale=[6, 0.6, 6])
        c.call("substance.set_object", object="Default_Cube", binding={"substance": profile})
        for name, position, scale in (
            ("floor", [0, -3.1, 0], [16.6, 0.4, 60.6]),
            ("ceiling", [0, 9.2, 0], [16.6, 0.4, 60.6]),
            ("left", [-8.2, 3, 0], [0.4, 12, 60.6]),
            ("right", [8.2, 3, 0], [0.4, 12, 60.6]),
            ("front", [0, 3, 30.2], [16, 12, 0.4]),
            ("back", [0, 3, -30.2], [16, 12, 0.4]),
        ):
            obj = c.call("scene.add_primitive", type="cube", name="FluidRoom " + name, size=1)
            state["objects"].append(obj)
            remember()
            c.call("scene.set_transform", name=obj, translation=position, scale=scale)
            c.call("substance.set_object", object=obj, binding={"substance": profile})
        pack = root / "RayTrophiStudio/assets/audio/cc0_starter"
        manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
        meta = next(a for a in manifest["assets"] if a["file"] == "water_flow_mono.wav")
        water = c.call("audio.asset_import", path=str(pack / meta["file"]),
                       author=meta["author"], license=meta["license"], source=meta["source_page"],
                       tags=["water", "recorded", "physics-control-test"])["id"]
        state["assets"].append(water)
        remember()
        source = c.call("audio.create", name="Water - solver flow", asset_id=water,
                        position=[0, 0, 0], loop=True, spatial=True, gain=0.45,
                        min_distance=5, max_distance=100, rolloff=0.7,
                        reaction={"driver": "domain_contact", "system_id": row["system_id"],
                                  "domain_name": domain["name"], "domain_channel": "fluid_flow",
                                  "input_metric": "power_w", "input_min": 0,
                                  "input_max": 1000, "gain_min": 0, "gain_max": 1,
                                  "response_seconds": 0.15, "follow_position": True,
                                  "automatic": True})["id"]
        state["sources"]["water"] = source
        remember()
        from rt_setup_echo_rooms import sound
        clap_path = folder / "clap_dry_reference.wav"
        sound(clap_path, "clap", 0.1)
        clap = c.call("audio.asset_import", path=str(clap_path),
                      author="Procedural diagnostic fixture", tags=["dry", "reference"])["id"]
        state["assets"].append(clap)
        remember()
        reference = c.call("audio.create", name="Clap - echo reference", asset_id=clap,
                           position=[1, 1.2, 1], spatial=True, loop=True, gain=0.4,
                           min_distance=5, max_distance=100, rolloff=0.7)["id"]
        state["sources"]["reference"] = reference
        remember()
        for voice in state["sources"].values():
            c.call("audio.acoustics_preview", id=voice, enabled=True,
                   spectral=True, reflections=True)
        c.call("audio.telemetry_update", enabled=True)
        c.call("audio.playback_update", mode="timeline")
        c.call("camera.set_position", position=[0, 1.7, -10])
        c.call("camera.set_target", target=[0, -0.3, 0])
        c.call("camera.set_fov", fov=65)
        c.call("viewport.set_shading", mode="material")
        c.call("render.set_settings", debug_view=0)
        c.call("audio.select", id=source)
        state["project"] = str(folder / "FluidAudioRoom.rtp")
        state["completed"] = True
        remember()
        c.call("project.save", path=state["project"])
        print(json.dumps({"project": state["project"], "sources": state["sources"],
                          "checkpoint": str(checkpoint)}, indent=2))
    finally:
        c.close()


if __name__ == "__main__":
    main()
