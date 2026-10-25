# Existing Snow scene: external IPC probe, 2026-10-23

User-authorized measurement of the open Snow seed + collider scene. No build,
application launch, material tuning, project save, or scene replacement.

## Procedure

The scene was paused at frame 0 with no live particles. Domain `Physics Domain 1`
used Snow, Vulkan, closed bounds +/-2.5 m, voxel 0.1 m, capacity 50,000.
The existing point source was disabled. Seeded its existing box
(-1.125, 0.375, -1.125) to (1.125, 2.125, 1.125), passing the reported PPC 2.
Runtime subsequently reported PPC 8, requested 87,552 and emitted 50,000;
seed_truncated=true, initial seeded top approximately 1.400 m. This discrepancy
needs investigation before claiming exact reproduction of the user's seed.

Stepped 180 times at dt=1/120 s (1.5 simulated seconds), sampling every 10 steps.
The scene remains paused at the measured end state. No reset or save was issued.
Raw observations: `snow_scene_probe_20261023.json`.

## Results

- All 50,000 parcels remained granular/MPM; fluid count stayed zero.
- Temperature stayed 263.15 K; thermal simulation was disabled. No actual melting.
- Requested and effective Young modulus both 140,000 Pa; no stiffness cap.
- Compaction hardening 10, limit 0.025; cohesion/tensile cutoff both 500 Pa.
- Direct particle plastic-volume state: final mean 1.20234, maximum 3.16947.
  This is an internal compactness variable, not independently measured density.
- Direct particle damage: final mean 0.01856, maximum 0.23271.
- Domain summary granular_max_damage remained zero despite nonzero attr.stats
  damage. Do not use those summary counters to conclude no fracture activity.
- Fracture-history attribute was unavailable; its zero-looking response is not
  evidence of zero fracture history.
- Horizontal RMS radius increased from 0.97588 m at step 10 to 1.08966 m at
  step 180, approximately 11.7 percent. Bounds include vertically falling edges;
  these moments alone cannot establish correct block fracture or support.
- Dry mass stayed approximately 2,500 kg. No sampled step held.
- Final kinetic energy approximately 49.01 J; not a completed repose test.
- Working set 266,320,000 bytes (254 MiB), not total application/VRAM usage.
- Sampled step calls approximately 0.55-0.62 seconds host wall time, including IPC
  and application work; not an isolated solver GPU timing.
- Runtime marked stiffness below estimated load. This is a load heuristic,
  not a proof of instability or the cause of spreading.
- A Sand output binding still activates DEM readiness, but there were zero DEM
  owners. The runtime status calls this a liquid lane despite MPM owners; that
  label must not be read as evidence that the Snow became liquid.

## Interpretation and next check

The active Snow path does compact and accumulate damage; there is no evidence
of a missing Snow preset assignment or thermal melting in this run. The current
measurement cannot distinguish calibrated physical yielding from constitutive,
contact, or resolution errors. Before tuning or replacing the model: reconcile
seed PPC/cap truncation and summary damage reporting, then perform identical-seed
compression, cohesion-control, and supported-block edge-release comparisons.
