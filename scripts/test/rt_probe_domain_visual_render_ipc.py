"""External visual acceptance on a dedicated paused field-sample domain.

Restores camera, shading, capture and authoring settings; deletes temporary meshes.
Captures are temporary verification artifacts under ignored docs/dev/.captures/.
"""
import argparse
import base64
import json
import time
import uuid
from pathlib import Path

from rt_ipc import RtIpc


def main(domain, substance, output):
    call = RtIpc().call
    assert not call("sim.control_state")["playing"]
    camera = call("camera.get")
    shading = call("viewport.shading")
    capture = call("viewport.status")["capture_enabled"]
    settings = call("fluid.visual_samples", domain=domain)["settings"]
    geometry = call("fluid.get_substance_splat_geometry", domain=domain, substance=substance)
    output.mkdir(parents=True, exist_ok=True)
    created = []
    report = {}

    def shot(label, mode):
        call("viewport.set_shading", mode=mode)
        time.sleep(0.6)
        deadline = time.monotonic() + 30
        while True:
            status = call("viewport.status")
            ready = status["frame_available"] and status["shading"] == mode
            if mode == "rendered":
                ready = ready and status["samples"] >= 4
            if ready:
                image = call("viewport.get_screenshot")
                if image.get("image_base64"):
                    payload = image["image_base64"].split(",")[-1]
                    (output / (label + ".jpg")).write_bytes(base64.b64decode(payload))
                    break
            if time.monotonic() >= deadline:
                raise RuntimeError("viewport capture timeout: " + repr(status))
            time.sleep(0.15)
        report[label] = {
            "viewport": status,
            "instances": call("render.instance_stats"),
            "visual": call("fluid.visual_samples", domain=domain)["stats"],
            "near": call("fluid.get", domain=domain)["grain_mesh_near"],
        }
        print(label, json.dumps(report[label]), flush=True)

    try:
        call("fluid.set_visual_samples", domain=domain, settings={
            "enabled": True, "target_volume": 0.00005,
            "max_samples": 20000, "working_budget_mb": 1024,
        })
        call("camera.set_position", position=[1.25, 1.0, 1.5])
        call("camera.set_target", target=[0.0, 0.4, 0.0])
        call("viewport.capture", enabled=True)
        call("fluid.set_substance_splat_geometry", domain=domain, substance=substance,
             mode="sphere", objects=[])
        for mode in ("solid", "material", "rendered"):
            shot("sphere_" + mode, mode)
        for index in range(2):
            name = call("scene.add_primitive", type="cube",
                        name="T_FieldVisual_" + uuid.uuid4().hex[:8], size=0.5)
            created.append(name)
            call("scene.set_transform", name=name, translation=[index * 2, -100, 0],
                 scale=[1, 0.5 + index * 0.4, 0.7])
        call("fluid.set_substance_splat_geometry", domain=domain, substance=substance,
             mode="mesh_grains", objects=created, source_materials=True,
             near_distance=1000000.0, mesh_budget=20000)
        for mode in ("material", "rendered"):
            shot("mesh_" + mode, mode)
    finally:
        call("fluid.set_substance_splat_geometry", domain=domain, substance=substance, **geometry)
        call("fluid.set_visual_samples", domain=domain, settings=settings)
        call("camera.set_position", position=camera["position"])
        call("camera.set_target", target=camera["target"])
        call("viewport.set_shading", mode=shading["mode"], matcap_preset=shading["matcap_preset"])
        call("viewport.capture", enabled=capture)
        for name in reversed(created):
            call("scene.delete", name=name)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--substance", default="Water")
    parser.add_argument("--output", type=Path,
                        default=Path("docs/dev/.captures/domain-visual-live"))
    args = parser.parse_args()
    main(args.domain, args.substance, args.output.resolve())
