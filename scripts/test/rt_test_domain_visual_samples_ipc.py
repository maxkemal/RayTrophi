"""User-built Studio acceptance on a dedicated paused, seeded Splat domain.

Appearance settings are restored in finally; no simulation time is advanced.
Uses the external named-pipe client, never an embedded pipe call.
"""
import argparse
import json
import time
import uuid
from pathlib import Path

from rt_ipc import RtIpc, RtIpcError


def mesh_case(call, domain, substance):
    original = call("fluid.get_substance_splat_geometry", domain=domain, substance=substance)
    created = []
    try:
        for index in range(2):
            name = call("scene.add_primitive", type="cube",
                        name="T_FieldMesh_" + uuid.uuid4().hex[:8], size=0.5)
            created.append(name)
            call("scene.set_transform", name=name, translation=[index * 2, -100, 0],
                 scale=[1, 0.6 + index * 0.3, 0.7])
        call("fluid.set_substance_splat_geometry", domain=domain, substance=substance,
             mode="mesh_grains", objects=created, source_materials=True,
             near_distance=1000000.0, mesh_budget=1000)
        deadline = time.monotonic() + 30
        while True:
            near = call("fluid.get", domain=domain)["grain_mesh_near"]
            if near["status"] == "active" and near["near_instances"] > 0:
                assert near["near_instances"] <= 1000
                break
            if time.monotonic() >= deadline:
                raise AssertionError("sample mesh palette did not become active: " + repr(near))
            time.sleep(0.1)
        print("PASS shared sample mesh palette", json.dumps(near))
    finally:
        call("fluid.set_substance_splat_geometry", domain=domain, substance=substance, **original)
        for name in reversed(created):
            call("scene.delete", name=name)


def python_case(call, domain):
    expected = call("fluid.visual_samples", domain=domain, limit=16)["items"]
    root = Path(__file__).resolve().parents[2]
    directory = root / "docs/dev/.captures" / ("field-python-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    wrapper = directory / "check.py"
    source = root / "scripts/test/rt_test_domain_visual_python.py"
    try:
        wrapper.write_text("import runpy\n" +
                           "runpy.run_path(" + repr(str(source)) + ")[\"run\"](" +
                           repr(domain) + ", " + repr(expected) + ")\n", encoding="utf-8")
        call("script.run_file", path=str(wrapper))
    finally:
        wrapper.unlink(missing_ok=True)
        directory.rmdir()


def main(domain, target_volume, steps=0, dt=1 / 120, mesh_substance=None, python=False):
    call = RtIpc().call
    assert not call("sim.control_state")["playing"], "Pause the dedicated test scene"
    baseline = call("fluid.visual_samples", domain=domain)
    physical = call("fluid.state_digest", domain=domain)
    assert physical["particles"] > 0, "Seed the dedicated domain first"
    original = baseline["settings"]
    try:
        initial = call("fluid.set_visual_samples", domain=domain, settings={
            "enabled": True, "target_volume": target_volume, "max_samples": 250000,
            "working_budget_mb": 1024,
        })
        assert initial["stats"]["current"], initial["stats"]
        assert initial["stats"]["samples"] > 0, "Use a continuum body routed to Splat"
        assert initial["stats"]["samples"] <= 250000
        first = call("fluid.visual_samples", domain=domain, limit=256)
        again = call("fluid.visual_samples", domain=domain, limit=256)
        assert first["items"] == again["items"], "paused redraw drift or identity churn"
        assert all(item["volume"] > 0 for item in first["items"])
        if python:
            python_case(call, domain)
        saved = first["settings"]
        for invalid in ({"target_volume": -1}, {"max_samples": True},
                        {"max_samples": -1}, {"max_substeps": 65537}, {"typo": 1}):
            try:
                call("fluid.set_visual_samples", domain=domain, settings=invalid)
            except RtIpcError:
                pass
            else:
                raise AssertionError("invalid settings accepted: " + repr(invalid))
            assert call("fluid.visual_samples", domain=domain)["settings"] == saved
        try:
            call("fluid.visual_samples", domain=domain, limit=257)
        except RtIpcError:
            pass
        else:
            raise AssertionError("unbounded inspection accepted")
        lower = call("fluid.set_visual_samples", domain=domain, settings={"max_samples": 1000})
        assert lower["stats"]["current"], lower["stats"]
        assert lower["stats"]["samples"] <= 1000
        volume_before = initial["stats"]["represented_volume"] + initial["stats"]["unrepresented_volume"]
        volume_after = lower["stats"]["represented_volume"] + lower["stats"]["unrepresented_volume"]
        assert abs(volume_after - volume_before) <= max(1e-10, abs(volume_before) * 1e-6)
        after = call("fluid.state_digest", domain=domain)
        for key in ("particles", "position_hash", "velocity_hash"):
            assert after[key] == physical[key], "visual settings changed physics: " + key
        if mesh_substance:
            mesh_case(call, domain, mesh_substance)
        for _ in range(steps):
            previous_time = call("fluid.visual_samples", domain=domain)["stats"]["physical_time"]
            call("fluid.step", dt=dt)
            advanced = call("fluid.visual_samples", domain=domain)["stats"]
            assert advanced["current"], advanced
            assert advanced["physical_time"] > previous_time
        print(json.dumps({"before": initial["stats"], "reduced_budget": lower["stats"]}, indent=2))
    finally:
        call("fluid.set_visual_samples", domain=domain, settings=original)
    print("PASS paused field samples: shared authoring, stable IDs, budget, validation, physics isolation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--target-volume", type=float, default=0.00001)
    parser.add_argument("--steps", type=int, default=0,
                        help="Optional physical steps of the whole dedicated test scene")
    parser.add_argument("--dt", type=float, default=1 / 120)
    parser.add_argument("--mesh-substance", help="Also test a temporary two-object mesh palette")
    parser.add_argument("--python", action="store_true", help="Also compare embedded Python with IPC")
    args = parser.parse_args()
    main(args.domain, args.target_volume, args.steps, args.dt, args.mesh_substance, args.python)
