"""RayTrophi production gas-domain API test.

Run from RayTrophi's embedded Python console:
    exec(open("scripts/test_gas_system.py", encoding="utf-8").read())

The script validates creation, GPU backend selection, settings round-trip,
manual stepping, clearing and lifecycle calls. It intentionally leaves the
domain in the scene for inspection.
"""

import rt


DOMAIN = "Script_GasSystem_Test"
STEP_COUNT = 10
DT = 1.0 / 60.0


def remove_if_present(name):
    try:
        rt.gas.remove_domain(name)
    except RuntimeError:
        pass


remove_if_present(DOMAIN)

created = rt.gas.create_domain(
    name=DOMAIN,
    domain_min=(-2.0, 0.0, -2.0),
    domain_max=(2.0, 4.0, 2.0),
    voxel_size=0.10,
)

# "gpu" is the production auto route: CUDA when available, otherwise Vulkan
# Compute, with deterministic CPU fallback if neither GPU backend is usable.
rt.gas.set_param(
    DOMAIN,
    backend="gpu",
    boundary="open",
    render_mode="volume",
    enabled=True,
    visible=True,
)

rt.gas.set_settings(
    DOMAIN,
    quality_profile="interactive",
    resource_budget_mb=512,
    enforce_resource_budget=True,
    use_sparse_tiles=True,
    render_to_nanovdb=True,
    fire_enabled=False,
    buoyancy_heat=1.0,
    buoyancy_density=0.08,
    vorticity=0.35,
    turbulence_strength=0.20,
    turbulence_scale=1.25,
    turbulence_octaves=3,
    turbulence_lacunarity=2.0,
    turbulence_persistence=0.5,
    turbulence_speed=0.5,
)

before = rt.gas.get(DOMAIN)
settings = rt.gas.get_settings(DOMAIN)

# rt 0.5.0's get() response omitted "type"; use create_domain() as fallback.
domain_type = before.get("type", created.get("type"))
assert domain_type == "gas", {"created": created, "before": before}
assert before["backend"] == "gpu", before
assert settings["quality_profile"] == "interactive", settings
assert settings["resource_budget_mb"] == 512, settings

for _ in range(STEP_COUNT):
    rt.gas.step(DT)

after = rt.gas.get(DOMAIN)
assert after["name"] == DOMAIN, after

print("[gas-system-test] created:", created)
print("[gas-system-test] before:", before)
print("[gas-system-test] settings:", settings)
print(f"[gas-system-test] stepped {STEP_COUNT} x {DT:.6f}s")
print("[gas-system-test] after:", after)
print("[gas-system-test] PASS")
print("Not: Yoğunluk üretmek için domain'e UI'dan bir smoke flow source ekleyin.")
