"""Phase 3 gate: ordinary flame is not a structural pressure event.

A localized high-pressure pulse is queued by the gas domain and consumed by the
same fracture impulse path as Jolt contacts. Full gas pressure fields never need
a CPU readback.
"""

import rt

DOMAIN = "Phase03_PressureGas"
FLAME = "Phase03_NormalFlame"
TARGET = "Phase03_PressureTarget"

for source in list(rt.flow_source.list()):
    if source["name"] == FLAME:
        rt.flow_source.remove(FLAME)
try:
    rt.fluid.remove_domain(DOMAIN)
except RuntimeError:
    pass
if rt.scene.exists(TARGET):
    rt.scene.delete(TARGET)

target = rt.scene.add_primitive("cube", TARGET, 0.65)
rt.scene.set_transform(target, translation=(0.8, 0.75, 0.0),
                       scale=(1.0, 0.8, 0.8))
rt.gas.create_domain(name=DOMAIN, domain_min=(-2.0, 0.0, -1.5),
                     domain_max=(2.0, 2.5, 1.5), voxel_size=0.14)
rt.gas.set_settings(DOMAIN, fire_enabled=True, ignition_temperature=0.22,
                    burn_rate=2.0, heat_release=2.5)
rt.flow_source.create(
    FLAME, domain=DOMAIN, source_mode="point", position=(-0.7, 0.45, 0.0),
    velocity=(0.0, 0.4, 0.0), radius=0.3, density=0.04,
    temperature=7.0, fuel=0.8, falloff=1.2,
)

rt.physics.make_fracture_group(
    TARGET, [target], break_impulse=4.0, integrity_weakening=False)
baseline = rt.gas.structural_impulse_stats()

# A normal open flame may heat/burn, but it is not an explosion and must not
# create structural pressure events or break the target.
for _ in range(30):
    rt.fluid.step(1.0 / 30.0)
    rt.physics.step(1.0 / 30.0)
after_flame = rt.gas.structural_impulse_stats()
assert after_flame["queued"] == baseline["queued"], (baseline, after_flame)
assert rt.physics.fracture_group(TARGET)["broken_count"] == 0

sequence = rt.gas.pressure_pulse(
    DOMAIN, center=(0.0, 0.75, 0.0), radius=2.0,
    peak_pressure_kpa=500.0, duration_seconds=0.02, coupling=1.0)
rt.physics.step(1.0 / 30.0)

final_stats = rt.gas.structural_impulse_stats()
target_info = rt.physics.fracture_group(TARGET)
assert sequence > baseline["queued"]
assert final_stats["consumed"] > after_flame["consumed"], final_stats
assert final_stats["last_max_impulse"] > target_info["base_break_impulse"], (
    final_stats, target_info)
assert target_info["broken_count"] == 1, (final_stats, target_info)

print({"result": "PASS", "phase": 3, "normal_flame_events": 0,
       "pulse_sequence": sequence, "pressure": final_stats,
       "target": target_info})
