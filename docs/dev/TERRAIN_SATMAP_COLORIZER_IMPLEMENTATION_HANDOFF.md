# Terrain SatMap Colorizer - Implementation Handoff

Date: 2026-08-21

## Completed phases

- SatMap ColorRamp evaluates Height, Slope, Flow, Soil, Grass, Snow, Ice,
  Meltwater and Avalanche semantic inputs on the terrain paint grid.
- Missing Slope is derived from height, Flow from the terrain erosion result and
  Soil from inverse hardness. An explicit graph connection always overrides a
  fallback.
- Snow/Ice is a protected, highest-priority overlay. Meltwater and Avalanche move
  clean snow gradually toward wet/dirty snow; the base SatMap cannot repaint full
  snow coverage.
- Eight terrain presets configure the height ramp, separate multi-stop
  Slope/Flow/Soil/Grass ramps, blend weights and preset-specific percentile distribution.
- Explicit Biome Grass overrides the fallback. When it is absent and Auto Derive
  is enabled, SatMap derives vegetation suitability from Soil, Slope and Flow.
- Terrain presets expose an optional `add_satmap` flag through the shared core,
  Python and IPC paths. The node editor's Setups menu mirrors it with an
  `Include SatMap Colorizer` preference and also offers a standalone setup action.
- SatMap, Auto Splat and Surface Composer produce their texture decisions at the
  independent paint resolution. Low-resolution physical masks are bilinearly
  sampled; procedural noise, patchiness and color detail are generated directly
  on the target grid.
- SatMap output uses the same Y orientation contract as Splat Output.
- SatMap settings use schema-v10 JSON serialization with migration for older preset
  distributions. Percentile limits are soft histogram shoulders; upper/lower
  terrain tails remain distinct instead of clamping to one color.
- Schema v9 rebalances every named preset around neutral substrate/rock/soil
  height colors. Vegetation color is owned by the separate Grass ramp, avoiding
  the former double-green response when a real Biome Grass mask is connected.
- Schema v10 reserves bright low-chroma colors for protected Snow/Ice. Named
  preset substrate composition is checked again after detail and all mask
  overlays, so height alone cannot turn a summit white. Custom ramps remain
  unrestricted.
- Preset ramps contain denser warm/cool color variation. Slope color uses a
  smooth limited overlay, preserving height and procedural detail on steep faces.
- Neutral near-white colors are removed from height/slope rock ramps. White is
  reserved for protected Snow/Ice inputs; steep terrain uses multiple weathered,
  fractured and hard-rock color bands instead.
- Exposed steep rock receives a second decorrelated paint-grid breakup field.
  Soil influence fades with steepness, preventing Soil and Slope overlays from
  converging into one color when Snow is disconnected.
- Paint resolution uses one core API exposed through both Python (`rt.terrain`) and
  IPC. The SatMap property smoke test is
  `scripts/test/rt_test_terrain_satmap_properties.py`.
- `SatMap Blend` composes two RGBA SatMaps through a scalar mask at paint
  resolution. It supports opacity, mask shaping and inversion, and can be
  chained for independently authored Soil, Flow, Grass and Rock color layers.
- Four layer-oriented ColorRamp presets (`Layer: Soil`, `Layer: Flow`,
  `Layer: Grass`, `Layer: Rock`) disable implicit mask derivation and internal
  overlays. Their primary scalar creates local color variation while the
  external Blend mask controls coverage.
- `Grass Mask` is now a dedicated reusable node rather than an inverted slope
  approximation. It combines Height with optional Soil, Flow, Slope, Wetness
  and Hardness, evaluates patch detail on the paint grid and provides
  Temperate, Lush, Alpine, Arid and Boreal presets.
- ColorRamp, Blend, Grass Mask and SatMap Output all participate in terrain
  graph save/load reconstruction; their authored properties use the same JSON
  reflection path exposed to Python and IPC.
- The legacy Curvature Mask now adaptively normalizes signed curvature instead
  of comparing tiny normalized-height Laplacian values directly against 0..1.
  `Surface Detail Masks` reuses the corrected concavity signal and emits Cavity,
  Mud and Moss masks, with Humid/Temperate/Arid/Alpine recipes, directly on the
  paint grid.
- Dedicated `Layer: Mud`, `Layer: Moss` and `Layer: Cavity` SatMap presets pair
  with those outputs through SatMap Blend without consuming material channels.
- The data-driven library in `assets/terrain/satmap_presets` currently ships
  eight recipes and 32 authored layers. Recipes split thin/wide and high/low
  flow using Hydraulic Channel Width and Height; distinguish slope, exposure,
  concavity and convexity; and compose mixed field/paint-resolution conditions
  through `Paint Mask Combine`.
- The Setups menu exposes the library. Python provides
  `rt.terrain.list_satmap_presets()` / `apply_satmap_preset()`, with matching
  `terrain.list_satmap_presets` / `terrain.apply_satmap_preset` IPC methods.
  Missing semantic fields skip only their dependent layers and return explicit
  warnings instead of silently substituting unrelated masks.

## Architectural boundary

Hydraulic Erosion and Snow remain field-resolution physical simulations by design.
Their results are sampled by texture nodes at paint resolution instead of running
an impractical 4K physical simulation. Real paint-domain evaluation is implemented
in SatMap ColorRamp, Auto Splat and Surface Composer. A graph-wide
`EvaluationContext::targetResolution` for every generic Noise/Math node remains a
separate future architecture change.

## One-pass build verification

1. Build RayTrophi Studio once with the new `TerrainPaintEvaluation.cpp`,
   `TerrainSatMapNodes.cpp` and `TerrainSatMapSetup.cpp` project entries.
2. Create a terrain with 1024 field/mesh resolution and 4096 paint resolution.
3. In SatMap debug view, inspect Height, Slope, Flow, Soil, Grass and Snow alignment.
4. Return debug view to `Final Color`; with Alpine, confirm no isolated yellow
   summit patch or percentile wrap remains.
5. Connect Snow/Ice, then vary Meltwater/Avalanche. Full snow must stay protected;
   only transition regions should become wet or dirty snow.
6. Connect Auto Splat and Surface Composer to Splat Output separately. Confirm the
   output contains crisp 4K procedural variation instead of enlarged 1K blocks.
7. Chain two ColorRamps through SatMap Blend using Soil and Flow masks. Confirm
   each layer remains confined to its mask and the final Blend feeds SatMap Output.
8. Save and reload the project. Confirm ColorRamp presets/stops, SatMap Blend,
   Grass Mask, blend values, percentiles, detail settings and debug selection survive.
9. Run `scripts/test/rt_test_terrain_satmap_properties.py` in the Studio Python
   environment to verify scripting/property parity.
