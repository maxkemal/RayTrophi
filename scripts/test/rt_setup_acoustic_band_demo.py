"""Populate an empty new project with an acoustic comparison through external IPC.

Requires an empty project and does not save it. Profiles deliberately
exaggerate band differences; these are diagnostic fixtures, not measured presets.
"""
import json
from pathlib import Path
import time

from rt_ipc import RtIpc


def main():
    client = RtIpc()
    if client.call("scene.list_objects"):
        client.close()
        raise RuntimeError("Open a new Empty project before running this setup")
    tag = str(time.time_ns())
    folder = Path(__file__).resolve().parents[2] / "docs/dev/.captures"
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / ("acoustic_demo_" + tag + ".json")
    state = {
        "project": client.call("project.path"),
        "camera": client.call("camera.get"),
        "workspace": client.call("audio.workspace_get"),
        "objects": [], "profiles": [], "source": None,
    }

    def remember():
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    remember()
    bands = (125, 250, 500, 1000, 2000, 4000)
    fixtures = [
        ("Reflective control", [0.05] * 6, [0.01] * 6, 0.45),
        ("High band absorber", [0.05, 0.15, 0.4, 0.7, 0.9, 0.97], [0.01] * 6, 0.8),
        ("Low band transmitter", [0.05] * 6, [0.8, 0.6, 0.3, 0.1, 0.02, 0.005], 0.45),
    ]
    try:
        profiles = []
        for label, absorption, transmission, scattering in fixtures:
            name = "Diagnostic " + label + " " + tag
            client.call("substance.derive", name=name, based_on="Stone")
            state["profiles"].append(name)
            remember()
            fields = {"acoustic_defined": True, "acoustic_scattering": scattering}
            fields.update({f"acoustic_absorption_{b}": v
                           for b, v in zip(bands, absorption)})
            fields.update({f"acoustic_transmission_{b}": v
                           for b, v in zip(bands, transmission)})
            client.call("substance.set", name=name, fields=fields)
            profiles.append(name)

        def cube(label, position, scale, profile):
            name = client.call("scene.add_primitive", type="cube",
                               name="AcousticDemo " + label + " " + tag, size=1)
            state["objects"].append(name)
            remember()
            client.call("scene.set_transform", name=name,
                        translation=position, scale=scale)
            client.call("substance.set_object", object=name,
                        binding={"substance": profile})

        origin = 0
        cube("Floor", [origin, -0.15, 0], [22, 0.3, 16], profiles[0])
        for x, profile, label in zip((-6, 0, 6), profiles,
                                     ("Reflector", "Absorber", "Transmitter")):
            cube(label, [origin + x, 2, 0], [4, 4, 0.25], profile)
        source = client.call("audio.create", name="Acoustic band diagnostic",
                             position=[origin, 2, 4], spatial=True,
                             gain=1, min_distance=2, max_distance=40, rolloff=1)
        state["source"] = source["id"]
        remember()
        client.call("audio.select", id=state["source"])
        client.call("audio.workspace_update", acoustic_band=0,
                    acoustic_all_sources=False, acoustic_reflection_samples=2)
        client.call("camera.set_position", position=[origin + 13, 10, -16])
        client.call("camera.set_target", target=[origin, 1, 0])
        client.call("camera.set_fov", fov=55)
        client.call("render.set_settings", debug_view=15)
        time.sleep(1)
        print(json.dumps({"restore_state": str(state_path),
                          "debug": client.call("audio.acoustics_debug",
                                               id=state["source"])}, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
