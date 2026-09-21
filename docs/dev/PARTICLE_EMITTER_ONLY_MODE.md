# Particle Emitter-Only Mode

Particle systems now default to **Emitter Only (Gas / Fluid)**. In this mode,
carrier particles continue to simulate and deposit density, temperature, fuel,
and liquid state into their domains, but the carriers themselves are not drawn.

The mode suppresses both particle presentation paths:

- RayFusion/Solid camera-facing billboards
- OptiX/Vulkan ray-traced particle instances

Gas and fluid domain rendering is unchanged. Debug overlays remain available for
authoring because they are an explicit diagnostic view, not scene presentation.

The Simulation > Particle System > System tab exposes the choice between
`Emitter Only (Gas / Fluid)` and `Visible Particles`. The same core setting is
available through:

```python
rt.particle.set_system_emitter_only("0", True)
rt.particle.list_systems()
```

IPC uses `particle.set_system_emitter_only` with `system` and `emitter_only`.
Systems may be addressed by list index or exact name.

New systems default to emitter-only. Older project files without the new field
load as visible-particle systems to preserve their previous appearance.

Visible built-in particles now share one material per system. The former color
bucket path created multiple global materials and duplicate primitive sources;
billboards still use their per-particle start/end colors directly.

## Verification

Without building, run the IPC descriptor/capability audits and Python syntax
checks. After the user builds:

1. Add a gas or fluid particle system and keep `Emitter Only` selected.
2. Play the simulation in RayFusion: the domain must render, carrier billboards
   must not render, and emission/deposition must continue.
3. Switch to a ray-traced render: no particle primitive instances should appear.
4. Select `Visible Particles`: billboards and optional RT geometry should return.
5. Save/reload and verify the selected usage mode persists.
6. Run `scripts/rt_api_smoke_test.py` and `scripts/ipc_test_client.py`.
