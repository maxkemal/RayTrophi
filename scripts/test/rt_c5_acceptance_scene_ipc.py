"""External IPC C5 acceptance fixture. Setup edits the open scene; no save/build.

python scripts/test/rt_c5_acceptance_scene_ipc.py --setup
python scripts/test/rt_c5_acceptance_scene_ipc.py --steps 10
After setup the three time-limited sources can be reset/replayed in the UI.
"""
import argparse
import json
import math
from pathlib import Path
from rt_ipc import RtIpc
from rt_domain_material import domain_material  # noqa: E402

DOMAIN = "C5_WaterSand_Acceptance"
LOG = Path(__file__).resolve().parents[2] / "docs/dev/matter_c5_acceptance_live_2026-10-05.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--log", type=Path, default=LOG)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--voxel", type=float, default=0.1)
    parser.add_argument("--water-limit", type=int, default=360,
                        help="Reference Water count at voxel 0.1; scaled with voxel volume")
    parser.add_argument("--balance", action="store_true",
                        help="Check a source-free window against its starting inventory")
    parser.add_argument("--wet", action="store_true",
                        help="With --setup, enable C6 wet physics and appearance (new build)")
    args = parser.parse_args()
    if not math.isfinite(args.dt) or not 0 < args.dt <= 0.1:
        parser.error("--dt must be finite and in (0, 0.1]")
    if not math.isfinite(args.voxel) or not 0.05 <= args.voxel <= 0.2:
        parser.error("--voxel must be finite and in [0.05, 0.2]")
    if args.steps < 0 or args.water_limit < 0:
        parser.error("Steps and water limit must be nonnegative")
    if args.balance and args.setup:
        parser.error("--balance requires an already primed source-free scene")
    if args.wet and not args.setup:
        parser.error("--wet requires --setup to avoid resetting an existing wet runtime")
    client = RtIpc()
    log = args.log
    records = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and ("__error" in result or result.get("ok") is False):
            raise RuntimeError((method, result))
        return result

    def record(kind, **extra):
        data = {"kind": kind, **extra}
        records.append(data)
        log.write_text(json.dumps(records, indent=2, ensure_ascii=True), encoding="utf-8")
        return data

    try:
        if args.setup:
            before = record("original_scene", control=call("sim.control_state"),
                            domains=call("fluid.list_domains"), sources=call("flow_source.list"))
            assert not before["control"]["playing"], "Pause before setup"
            for source in before["sources"]:
                if source["enabled"]:
                    call("flow_source.update", name=source["name"], enabled=False)
            for domain in before["domains"]["domains"]:
                call("fluid.set_param", domain=domain["name"], enabled=False, visible=False)
            if DOMAIN not in [d["name"] for d in before["domains"]["domains"]]:
                call("fluid.create_domain", name=DOMAIN, type="matter",
                     domain_min=[-1.6, 0, -1.6], domain_max=[1.6, 3.2, 1.6], voxel_size=0.1)
            material = domain_material(call, 'T: c5_acceptance_scene', 'Sand',
                granular_cohesion=0.0, granular_young_modulus=50000.0, granular_tensile_cutoff=0.0, granular_damage_rate=0.0)
            call("fluid.set_param", domain=DOMAIN, enabled=True, visible=True,
                 voxel_size=args.voxel,
                 backend="vulkan", boundary="closed", default_substance=material,
                 solid_phase=False, thermal_liquid_enabled=False, render_mode="particles")
            materials = {m["name"] for m in call("material.list")}
            for substance, model, color, roughness in [
                ("Sand", "granular", [0.63, 0.40, 0.16], 0.85),
                ("Water", "fluid", [0.03, 0.25, 0.62], 0.18)]:
                material = "C5 " + substance + " Preview"
                if material not in materials:
                    material = call("material.create", type="substance:" + substance, name=material)
                call("material.set_param", material_name=material, param="base_color", value=color)
                call("material.set_param", material_name=material, param="roughness", value=roughness)
                call("fluid.set_substance_material", domain=DOMAIN, substance=substance,
                     constitutive_model=model, phase="liquid", representation="splat", material=material)
            if not call("scene.list_objects"):
                ground = call("scene.add_primitive", type="plane", name="Acceptance Ground", size=4.0)
                call("scene.set_transform", name=ground, translation=[0, -0.08, 0])
                material = call("material.create", type="principled", name="C5 Neutral Ground")
                call("material.set_param", material_name=material, param="base_color", value=[0.19, 0.21, 0.24])
                call("material.set_param", material_name=material, param="roughness", value=0.9)
                call("material.assign", object_name=ground, material_name=material)
            if not call("lights.list"):
                call("lights.add", type="point", position=[-1, 4, 3])
            call("camera.set_target", target=[0.05, 0.25, 0])
            call("camera.set_position", position=[2.6, 2.1, 3.6])
            call("camera.set_fov", fov=35)
            call("select.clear")
            call("fluid.set_pore_exchange", domain=DOMAIN, enabled=True, porosity=0.35,
                 permeability_m2=1e-8, viscosity_pa_s=0.001, gravity_m_s2=9.81, drainage_scale=1.0)
            supported = call("fluid.matter_models", domain=DOMAIN)["pore_exchange"]["settings"]
            if args.wet or "wet_response_enabled" in supported:
                call("fluid.set_pore_exchange", domain=DOMAIN, wet_response_enabled=args.wet,
                     wet_appearance_enabled=args.wet)
            existing = {s["name"] for s in call("flow_source.list")}
            parcel_scale = (0.1 / args.voxel) ** 3
            for name, substance, model, position, radius, rate, limit in [
                ("C5_Sand_Burst", "Sand", "granular", [-0.5, 0.45, 0], 0.26, 1200, 360),
                ("C5_Water_Burst", "Water", "fluid", [-0.5, 0.65, 0], 0.26, 1200, args.water_limit),
                ("C5_Dry_Sand_Control", "Sand", "granular", [1.05, 0.35, 0], 0.12, 200, 60)]:
                rate = round(rate * parcel_scale)
                limit = round(limit * parcel_scale)
                call("flow_source.update" if name in existing else "flow_source.create",
                     name=name, domain=DOMAIN, enabled=limit > 0, phase="liquid", source_mode="point",
                     fluid_substance=substance, position=position,
                     radius=radius, velocity=[0, 0, 0], fluid_velocity_spread=0.0,
                     fluid_particles_per_second=rate, fluid_temperature_override=True,
                     fluid_temperature_kelvin=293.15, use_time_limit=True,
                     start_time=0.0, end_time=0.3, use_particle_limit=True,
                     max_emitted_particles=limit)
            call("timeline.set_frame", frame=0)
            call("fluid.reset")
            record("fixture_config", domain=call("fluid.get", domain=DOMAIN),
                   sources=call("flow_source.list"), control=call("sim.control_state"),
                   dt=args.dt, reference_water_limit=args.water_limit, parcel_scale=parcel_scale)
        control_start = call("sim.control_state")
        assert not control_start["playing"], "Pause before stepping"
        baseline = call("fluid.matter_models", domain=DOMAIN) if args.balance else None
        if baseline:
            record("balance_start", inventory=baseline, control=control_start, dt=args.dt)
        maximum_drift = 0.0

        def check_sample(sample):
            nonlocal maximum_drift
            assert call("sim.control_state") == control_start, "Control changed; discard this window"
            assert not sample["mixed_execution"]["step_held"], sample
            assert not sample["pore_exchange"]["held"], sample
            if baseline is None:
                return
            def masses(inventory):
                pore = inventory["pore_exchange"]["pore_water_kg"]
                return inventory["models"][0]["mass_kg"] + pore, inventory["models"][1]["mass_kg"] - pore
            start_water, start_dry = masses(baseline)
            water, dry = masses(sample)
            maximum_drift = max(maximum_drift, abs(water - start_water))
            assert abs(dry - start_dry) <= max(1e-6, start_dry * 1e-7), "Dry source still active or mass drift"
            assert abs(water - start_water) <= max(1e-5, start_water * 1e-6), "Water source still active or mass drift"

        for index in range(args.steps):
            call("fluid.step", dt=args.dt)
            if (index + 1) % 10 == 0:
                sample = call("fluid.matter_models", domain=DOMAIN)
                check_sample(sample)
                record("interval_sample", step_in_batch=index + 1, inventory=sample,
                       step_stats=call("fluid.step_stats", domain=DOMAIN),
                       control=call("sim.control_state"))
        control_before = call("sim.control_state")
        inventory = call("fluid.matter_models", domain=DOMAIN)
        if args.steps:
            check_sample(inventory)
        info = call("fluid.get", domain=DOMAIN)
        data = record("snapshot", requested_manual_steps=args.steps,
                      control_before=control_before, inventory=inventory,
                      control_after=call("sim.control_state"),
                      particle_count=info["particle_count"], substances=info["substances"],
                      step_stats=call("fluid.step_stats", domain=DOMAIN),
                      sources=call("flow_source.list"))
        if baseline:
            record("balance_result", maximum_sampled_water_drift_kg=maximum_drift,
                   steps=args.steps, dt=args.dt, control=data["control_after"])
        pore = inventory["pore_exchange"]
        print(json.dumps({"steps": args.steps, "count": info["particle_count"],
                          "models": inventory["models"], "substances": info["substances"],
                          "pore": pore, "execution": inventory["mixed_execution"],
                          "wet": inventory.get("wet_response"),
                          "maximum_sampled_water_drift_kg": maximum_drift if baseline else None,
                          "control": data["control_after"]}, ensure_ascii=True))
    finally:
        client.close()


if __name__ == "__main__":
    main()
