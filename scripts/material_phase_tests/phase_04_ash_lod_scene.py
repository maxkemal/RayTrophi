"""Phase 4 gate: ash detail scales with distance and obeys a hard budget."""

import rt

rt.particle.clear()
rt.debris.configure(enabled=True, max_particles=20, particles_per_kg=100.0,
                    near_distance=10.0, far_lod_scale=0.25,
                    lifetime_seconds=8.0)
before = rt.debris.stats()

near_spawned = rt.debris.emit_ash(
    center=(-0.5, 1.0, 0.0), mass_kg=0.10,
    velocity=(0.0, 0.4, 0.0), camera_distance=2.0, seed=11)
far_spawned = rt.debris.emit_ash(
    center=(0.5, 1.0, 0.0), mass_kg=0.10,
    velocity=(0.0, 0.4, 0.0), camera_distance=30.0, seed=22)
budget_spawned = rt.debris.emit_ash(
    center=(0.0, 1.0, 0.0), mass_kg=1.0,
    velocity=(0.0, 0.7, 0.0), camera_distance=2.0, seed=33)

after = rt.debris.stats()
delta_requested = after["requested_particles"] - before["requested_particles"]
delta_spawned = after["spawned_particles"] - before["spawned_particles"]
delta_lod = after["lod_reduced_particles"] - before["lod_reduced_particles"]
delta_rejected = (after["budget_rejected_particles"] -
                  before["budget_rejected_particles"])

assert near_spawned == 10, near_spawned
assert far_spawned == 3, far_spawned
assert budget_spawned == 7, budget_spawned
assert delta_requested == 120, delta_requested
assert delta_spawned == 20, delta_spawned
assert delta_lod == 7, delta_lod
assert delta_rejected == 93, delta_rejected
assert after["alive_particles"] == 20, after
assert after["alive_particles"] <= after["max_particles"], after

# ★ The budget caps VISUAL DETAIL, never mass. Rejecting 93 detail particles must
# not delete the kilograms they stood for: whatever is not carried by a live
# particle has to be sitting in the reservoir, waiting for the next event.
emitted_mass = 0.10 + 0.10 + 1.0
accounted_mass = ((after["accepted_mass_kg"] - before["accepted_mass_kg"]) +
                  after["reservoir_mass_kg"] - before["reservoir_mass_kg"])
assert abs(accounted_mass - emitted_mass) <= 1e-4, (
    accounted_mass, emitted_mass, after)

# A budget-starved event must park its mass rather than return it to nobody.
starved = rt.debris.emit_ash(
    center=(0.0, 1.2, 0.0), mass_kg=0.5,
    velocity=(0.0, 0.5, 0.0), camera_distance=2.0, seed=44)
starved_stats = rt.debris.stats()
assert starved == 0, starved
assert abs(starved_stats["reservoir_mass_kg"] -
           (after["reservoir_mass_kg"] + 0.5)) <= 1e-4, starved_stats

print({"result": "PASS", "phase": 4,
       "mass_accounted": accounted_mass,
       "reservoir_after_starved": starved_stats["reservoir_mass_kg"],
       "near_spawned": near_spawned, "far_spawned": far_spawned,
       "budget_spawned": budget_spawned,
       "delta": {"requested": delta_requested, "spawned": delta_spawned,
                 "lod_reduced": delta_lod, "budget_rejected": delta_rejected},
       "debris": after})
