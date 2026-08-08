"""
Match -> gasoline -> spreading fire, built entirely from the scripting API.

The beats, and which system owns each one:

  1. A lit match is thrown.        Jolt rigid body + a gas flow source parented
                                   to the match's TIP (parent-local offset).
  2. It falls naturally.           Physics drives the transform; the flame and
                                   the ember emitter ride it because they are
                                   parented, not keyframed to a path.
  3. A moving hose pours petrol.   A fluid flow source parented to the hose,
                                   with a KEYED flow rate (valve open/close).
  4. The pool catches and spreads. sim_fluid_surface_combustion: the liquid
                                   flashes -> vapour enters the gas fuel field
                                   -> the gas burns -> its heat flashes the
                                   NEXT surface cell. The spread is emergent,
                                   there is no flame-front model to author.
  5. Objects in the room ignite.   Material State Field substances on their
                                   colliders (pyrolysis in real Kelvin).

Run from the Scripts panel, or:  rt.project.open(...) then exec this file.
"""

import rt

# When this file is selected through "Install Addon...", RayTrophi wraps it in
# an addon package and executes it only after the user enables the addon.
bl_info = {
    "name": "Match & Gasoline Fire",
    "description": "Builds the match, petrol spill and spreading-fire simulation test scene.",
    "author": "RayTrophi Studio",
    "version": (1, 0, 0),
    "category": "Simulation",
    "location": "Scripts > Addons",
    "warning": "Enabling rebuilds the test scene and clears existing particle systems.",
}

addon_settings = {
    "fps": 30,
    "gas_domain": "FireGas",
    "fluid_domain": "Petrol",
}

FPS = 30.0


def f(seconds):
    """Frame index for a time in seconds (keys are authored in frames)."""
    return int(round(seconds * FPS))


# ---------------------------------------------------------------------------
# 0. Stage geometry
#
# ★ add_primitive RENAMES on collision and returns the name it actually used.
# Re-running the script without deleting first would leave "Match" from the
# previous run in the scene and create "Match.001" — and every parented emitter
# would then follow the OLD, motionless cube while the visible one falls. Wipe
# first, and bind to the returned names, never to the requested ones.
# ---------------------------------------------------------------------------
STAGE = ["Floor", "Match", "Hose", "Crate", "Pallet"]
for name in STAGE:
    if rt.scene.exists(name):
        rt.scene.delete(name)

# ★★ WIPE THE SIMULATION SIDE TOO, not just the meshes.
#
# Deleting the stage objects leaves every flow source, emitter, collider and
# grid domain in the scene untouched. Anything a UI preset created keeps
# running alongside this script's setup, and the two are indistinguishable in
# the viewport. The IgnitedFuelJet preset in particular ships:
#   - "Fuel Jet Pilot": a permanently-burning gas source at (1.35, 0.28, 0)
#     -> the mystery flame that sits burning in the middle of the floor;
#   - "Fuel Nozzle": a liquid jet at (0, 1.45, 0) doing 5200 particles/sec
#     with a 42000 budget and an 8-second time limit -> it dumps its whole
#     budget in one go and then goes dead, which is exactly the "the water
#     empties out all at once and then the hose moves around empty" symptom.
# Neither belongs to this script, so neither can be tuned by editing it.
#
# ★★ AND THEY CANNOT BE REMOVED ONE BY ONE EITHER. rt.flow_source.list(),
# .remove(), rt.particle.clear_emitters() and rt.collider.* are all scoped to
# the ACTIVE particle system. A preset creates its OWN system, so once anything
# else becomes active the preset's pilot is invisible to list() and unreachable
# by remove() — which is why it survived every attempt to delete it. clear_systems()
# is the scene-wide wipe; the first add below re-creates a clean system.
print("Before wipe:", rt.particle.list_systems())
rt.particle.clear_systems()

FLOOR = rt.scene.add_primitive("plane", "Floor", 12.0)
rt.scene.set_transform(FLOOR, translation=(0.0, 0.0, 0.0))

MATCH = rt.scene.add_primitive("cube", "Match", 0.09)
rt.scene.set_transform(MATCH, translation=(-1.2, 1.55, 0.0), rotation=(0.0, 0.0, 25.0))

HOSE = rt.scene.add_primitive("cube", "Hose", 0.18)
rt.scene.set_transform(HOSE, translation=(-2.6, 1.15, 0.0))

# Things that should catch fire once the pool is burning.
CRATE = rt.scene.add_primitive("cube", "Crate", 0.55)
rt.scene.set_transform(CRATE, translation=(1.9, 0.55, 0.6))
PALLET = rt.scene.add_primitive("cube", "Pallet", 0.42)
rt.scene.set_transform(PALLET, translation=(2.4, 0.42, -0.9))

print("Stage objects:", FLOOR, MATCH, HOSE, CRATE, PALLET)


# ---------------------------------------------------------------------------
# 1. Domains
#
# ONE gas domain covers everything. There is no gas<->gas coupling, so a match
# with its own little domain could never hand its fire to the pool. The box
# must also reach the height the match is released from, or the flame is born
# outside the grid and nothing ignites.
#
# The fluid domain is a wide, floor-hugging slab: petrol spreads, it does not
# pile up.
# ---------------------------------------------------------------------------
rt.gas.create_domain(
    name="FireGas",
    domain_min=(-4.0, 0.0, -3.0),
    domain_max=(4.0, 3.2, 3.0),
    voxel_size=0.05,
)
rt.gas.set_settings(
    "FireGas",
    fire_enabled=True,           # also provisions the Fuel channel automatically
    ignition_temperature=0.24,
    burn_rate=2.6,
    heat_release=2.9,
    smoke_generation=0.5,
    flame_dissipation=1.9,       # a small flame must survive long enough to spread
    fire_max_temperature=9.0,
    fire_expansion=0.09,
    buoyancy_heat=0.9,
    buoyancy_density=0.03,
    # Vorticity/turbulence are what carry the flame ACROSS the pool once it is
    # lit — they mix hot gas sideways instead of letting buoyancy take all of it
    # straight up. They do NOT help the first ignition: stirring a cell that is
    # already below the ignition threshold only cools it faster.
    vorticity=0.55,
    turbulence_strength=0.40,
    turbulence_scale=2.2,
    turbulence_octaves=4,
    turbulence_speed=1.1,
)
# Without this the fire simulates correctly and renders as flat grey: the
# default smoke preset carries no blackbody emission.
rt.gas.set_shader(
    "FireGas",
    preset="fire",
    blackbody_intensity=6.5,
    temperature_min=780.0,
    temperature_max=1900.0,
    density_multiplier=1.45,
    # A match flame is a handful of voxels at very low density. At the usual
    # 0.014 cutoff it is culled outright, so the match looks unlit until the
    # pool catches and the fire is big enough to clear the threshold.
    density_cutoff=0.006,
)

rt.fluid.create_domain(
    name="Petrol",
    domain_min=(-4.0, 0.0, -3.0),
    domain_max=(4.0, 0.9, 3.0),
    voxel_size=0.05,
)
rt.fluid.set_param("Petrol", boundary="closed", preset="oil", viscosity=0.14)
rt.fluid.set_combustion(
    "Petrol",
    enabled=True,
    chemistry_preset="gasoline",
    auto_ignite=False,            # the match is the ignition source, not a cheat
    # Petrol flashes off a spark, so the threshold is low and the evaporation
    # rate high: vapour has to be sitting above the surface BEFORE the flame
    # arrives, otherwise the flame passes over dry liquid and nothing happens.
    ignition_temperature=0.24,
    evaporation_rate=0.45,
    surface_fuel_capacity=4.5,
    heat_release=2.6,
    smoke_yield=0.46,
    surface_cooling=0.16,         # cooling this high stalls the spread front
)


# ---------------------------------------------------------------------------
# 2. The match: rigid body + collider + its own flammability
# ---------------------------------------------------------------------------
# ★ TWO SEPARATE COLLISION SYSTEMS. rt.collider.* is the particle/liquid
# solver's collider list; it does NOT stop a Jolt rigid body. The match
# falls straight through a "FloorCollider" unless the floor also has a
# STATIC Jolt body of its own.
rt.physics.add_body(FLOOR, kind="rigid", motion_type="static", shape="box")
rt.physics.add_body(MATCH, kind="rigid", motion_type="dynamic", shape="box", mass=0.02)

rt.collider.create(
    "MatchCollider",
    source_mode="obb",
    source_object=MATCH,
    restitution=0.15,
    friction=0.6,
    msf_substance="Wood (Oak)",
    msf_override_ignition=True,
    msf_ignition_kelvin=520.0,     # a match head lights well below seasoned oak
    msf_burn_rate_scale=2.4,       # and burns away fast
    msf_mask_resolution=64,
)

# The tip flame. Parented, so the offset is in the MATCH's space: +Y in local
# space stays the tip no matter how the match tumbles.
#
# ★ RADIUS IS THE IGNITION LEVER, not temperature. The liquid->gas coupling
# samples the gas cell sitting on the petrol SURFACE. A physically-sized match
# flame is smaller than one 5cm voxel, so it happily heats a cell BESIDE that
# surface cell and never hands over a single degree — the fire looks like it
# refuses to catch no matter how hot the source is. 0.30 spans ~6 voxels, which
# is what actually makes contact. (The IgnitedFuelJet preset carries the same
# note on its pilot, at 0.65 for a 10cm grid.)
rt.flow_source.create(
    "MatchFlame",
    domain="FireGas",
    source_mode="point",
    parent_object=MATCH,
    position=(0.0, 0.11, 0.0),
    velocity=(0.0, 0.55, 0.0),
    velocity_space="local",
    inherit_velocity=1.0,
    radius=0.30,
    density=0.06,
    temperature=8.0,
    fuel=2.4,
    falloff=1.4,
    # Lower than the preset's pilot on purpose: a strong coupling on a source
    # this small blows its own hot gas out of the cell it just heated.
    velocity_coupling=5.0,
)

# ── No ember particle emitter ────────────────────────────────────────────────
# There used to be a "MatchEmbers" particle emitter here, parented to the match,
# depositing fuel/heat into the gas grid so the match could ignite petrol it had
# not physically touched yet. It is REDUNDANT: MatchFlame above already injects
# temperature and fuel into the same gas domain, at a radius chosen to reach the
# liquid surface. Two sources doing one job only made the scene harder to read —
# and the emitter's viewport marker was the stray dot in the middle of the floor.
#
# The particle->grid deposit contract itself is untouched and still available:
# add an emitter with override_grid_deposit=True when you want sub-voxel sparks
# that carry fire (see rt.particle.add_emitter). This scenario just doesn't need
# one to work.


# ---------------------------------------------------------------------------
# 3. The hose: keyframed motion + a keyed valve
#
# The nozzle is parented, so it follows the hose without any of its own keys.
# What IS keyed is the flow rate — the valve opens, runs, then shuts.
# ---------------------------------------------------------------------------
# ★ ORDER OF EVENTS. The match is a DYNAMIC rigid body from frame 0, so it
# lands in well under a second — there is no way to "hold" it and no keyable
# motion type. So the beat is the honest one: the match is already lit when it
# is dropped, it lands FIRST at roughly (-1.2, 0, 0), and the hose then walks
# the petrol INTO the burning match. Ignition happens when the spreading pool
# reaches the flame, not when a timer says so.
rt.anim.insert_key(HOSE, "location", f(0.0), (2.6, 1.15, -1.4))
rt.anim.insert_key(HOSE, "location", f(1.6), (1.4, 1.15, 0.6))
rt.anim.insert_key(HOSE, "location", f(3.4), (0.1, 1.15, -0.4))
rt.anim.insert_key(HOSE, "location", f(5.2), (-1.1, 1.15, 0.15))

rt.flow_source.create(
    "FuelNozzle",
    domain="Petrol",
    source_mode="point",
    parent_object=HOSE,
    position=(0.0, -0.14, 0.0),
    velocity=(0.0, -1.6, 0.0),
    velocity_space="local",
    inherit_velocity=1.0,          # the stream arcs with the hose, not straight down
    radius=0.10,
    fluid_particles_per_second=1800.0,
    fluid_velocity_spread=0.14,
    # ★ NO PARTICLE BUDGET. A budget is a hard cliff, not a rate: the source
    # runs at full rate until the counter is spent and then stops dead, so the
    # hose empties itself early and then wanders around dry. The valve keys
    # below are the only thing that should decide when pouring stops. The
    # domain's own max_particles still backstops the memory cost.
    use_particle_limit=False,
)
# Valve curve: opens once the hose is over the floor, shuts before the flame
# front arrives, so what burns is a settled pool and not a live jet.
rt.flow_source.key("FuelNozzle", f(0.0), flow_rate=0.0)
rt.flow_source.key("FuelNozzle", f(0.6), flow_rate=1800.0)
rt.flow_source.key("FuelNozzle", f(5.0), flow_rate=1800.0)
rt.flow_source.key("FuelNozzle", f(5.4), flow_rate=0.0)


# ---------------------------------------------------------------------------
# 4. The match burns down
#
# It is lit from the first frame (it is dropped already burning). These keys
# only govern how long it keeps feeding the gas: once the pool has caught, the
# liquid's own surface fuel owns the fire and the match is irrelevant.
# ---------------------------------------------------------------------------
rt.flow_source.key("MatchFlame", f(0.0), enabled=True, temperature=8.0, fuel=2.4)
rt.flow_source.key("MatchFlame", f(7.5), temperature=8.0, fuel=2.4)
rt.flow_source.key("MatchFlame", f(9.5), temperature=0.0, fuel=0.0)


# ---------------------------------------------------------------------------
# 5. Room objects: what they are MADE of decides whether they catch
# ---------------------------------------------------------------------------
rt.collider.create(
    "CrateCollider",
    source_mode="mesh_sdf",
    source_object=CRATE,
    msf_substance="Wood (Oak)",
    msf_mask_resolution=128,
)
rt.collider.create(
    "PalletCollider",
    source_mode="mesh_sdf",
    source_object=PALLET,
    msf_substance="Wood (Oak)",
    msf_burn_rate_scale=1.6,   # a thin pallet catches faster than a solid crate
    msf_mask_resolution=128,
)
rt.collider.create(
    "FloorCollider",
    source_mode="plane",
    plane_y=0.0,
    friction=0.7,
)

rt.timeline.set_frame(0)
print("Scene built. Substances available:", rt.msf.substances())

# ★ PLAY THIS ON THE TIMELINE, NOT IN LIVE UPDATE.
# Every key below is authored in TIMELINE frames. Live Update free-runs the
# solver at the UI refresh rate with the playhead parked, so a curve keyed
# across frames 18..150 is consumed in a second or two of wall time: the hose
# empties itself at once and the match's flame is already past its fade-out key.
for src in rt.flow_source.list():
    print("  flow source %-12s parent=%-8s pos=%s rate=%.0f" % (
        src["name"], src["parent_object"] or "-", src["position"],
        src["fluid_particles_per_second"]))
print("Scrub or bake the timeline; watch particle.stats() deposit counters if "
      "the fire never starts.")
