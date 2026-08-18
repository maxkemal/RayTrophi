# Vulkan RT Gas + Fluid Surface Layering

> **Durum:** REFERANS — Uygulandi ve dogrulandi; katmanlama sozlesmesinin kaydi.

> ★ Bu belge katmanlama **sozlesmesini** anlatir. Ayni konfigurasyonun uzun sure
> acik kalan arizasi — cakisik kutularda isinin ilerleyememesi, siyah bant ve
> maliyet patlamasi — ayri bir postmortemde:
> [VOLUME_BOX_REENTRY_POSTMORTEM.md](VOLUME_BOX_REENTRY_POSTMORTEM.md).
> Buradaki hakem (arbiter) mantigi dogru calisirken de o ariza olusabiliyordu.

## Status

Implemented and validated for coincident live gas and Fluid `SurfaceSDF`
domains. The reference scene is the embedded **Burning Fuel Spill** preset.

## Original failure

Vulkan RT represented both domains as procedural AABB instances. When their
bounds overlapped, traversal selected only one closest-hit program:

- gas first: its no-scatter continuation advanced to the gas AABB exit and
  skipped the nested liquid surface, producing a dark rectangular band;
- SurfaceSDF first: its redirect or empty-AABB continuation skipped the gas;
- forcing a second complete AABB pass rendered only one layer in common cases
  and made path tracing unnecessarily expensive.

OptiX did not show the defect because its renderer explicitly sorts VDB
intervals per ray and composites them front-to-back.

## Vulkan solution

The Vulkan path keeps the existing production gas marcher and adds an ordered
handoff at a real liquid boundary:

1. Coincident participating media win the equal-distance AABB entry.
2. A live gas closest-hit searches active SurfaceSDF volumes for the nearest
   actual `density = 0.5` crossing along the current ray.
3. Gas integration is clamped to that crossing instead of marching to the
   outer domain boundary.
4. Gas/fog and SurfaceSDF instances use separate TLAS mask bits.
5. The next trace excludes gas for one pass and invokes the SurfaceSDF
   closest-hit at the boundary.
6. The handoff does not consume a GI bounce.
7. Solid probes exclude both procedural volume classes.

The candidate scan is capped at 16 active volumes, matching the current OptiX
per-ray VDB sorting limit. Ordinary single VDB, cloud, and gas scenes retain
their established fast path.

## Related GPU allocation fix

Fluid particle position, velocity, and affine buffers must not be allocated at
the exact live particle count. Continuous emitters change that count often and
previously forced all three buffers to be destroyed and recreated repeatedly.
They now retain spare capacity and grow geometrically. This avoids transient
allocation failures that could push the APIC step onto its CPU fallback path.

Whitewater foam is independent:

- `FoamParams::enabled` is false by default;
- the embedded Burning Fuel Spill preset does not enable foam;
- `Fluid::stepFoam` is currently CPU-side and can add CPU cost when explicitly
  enabled, but it does not share the three APIC particle buffers.

## Timeline cache and GPU-resident gas

The live Vulkan gas solver keeps velocity and scalar fields resident on the
device. Copying `gridDomainStates()` directly therefore does not create a valid
timeline snapshot: its CPU vectors may contain only an old host publication or
the latest authored source deposit. Replaying such a frame invalidates the
live device addresses and leaves the renderer with an empty or black domain.

`captureGridDomainStatesForCache()` now performs one batched readback of the
authoritative gas velocity, density, temperature, fuel, and interaction fields
into the cache copy. It never mutates the live state or clears GPU residency.
The snapshot also recomputes occupied-density bounds so replay can immediately
publish the correct NanoVDB fallback before the next live solve.

## Invariants for future changes

- Never resolve overlapping gas and SurfaceSDF by advancing directly to the
  outer AABB exit.
- Never solve layering by running two complete coincident AABB marches.
- Preserve separate TLAS mask bits for gas/fog and SurfaceSDF.
- A gas-to-surface handoff must not consume the authored path-bounce budget.
- Keep procedural volume masks out of solid-geometry probes.
- Recompile every Vulkan RT shader after changing `RayPayload`.

## Embedded production preset direction

Keep production presets embedded until the scripting path can reproduce their
state without per-frame authoring work or resets.

Recommended set:

1. **Burning Fuel Spill**
   - finite oil pool, SurfaceSDF liquid, open gas domain;
   - surface evaporation, ignition, smoke and flame attachment;
   - reference preset for coincident gas/SDF rendering.
2. **Ignited Fuel Jet**
   - continuous liquid flow source from a nozzle;
   - ground accumulation followed by delayed ignition;
   - demonstrates moving APIC particles, continuous buffer growth and
     liquid-to-gas combustion.
   - implemented as an embedded additive production preset; uses a finite
     eight-second fuel emission and a short pilot pulse rather than a
     permanently injected decorative flame.
3. **Flaming Wall Impact**
   - directed combustible liquid jet and a collider wall;
   - impact splash, retained surface fuel and secondary wall fire;
   - reference for collider coupling and deposited heat.
4. **Molotov Pool**
   - short liquid burst followed by spreading surface fire;
   - finite emission, cooling and burn-out instead of filling the domain.
5. **Burning Waterfall**
   - elevated continuous liquid sheet entering a catch basin;
   - exercises gravity, thin SurfaceSDF features and gas rising through a
     moving liquid layer.

Each preset should remain additive, use unique domain/source names, avoid
deleting existing scene systems, explicitly select Vulkan compute, and leave
foam disabled unless the preset is specifically intended to demonstrate
whitewater.
