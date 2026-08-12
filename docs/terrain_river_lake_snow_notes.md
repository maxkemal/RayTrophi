# Terrain River, Lake and Snow Notes

## 2026-08-11 — Snowmelt water budget and river visibility

- `Snow Layer` keeps its existing normalized `Meltwater` mask and additionally
  publishes `Meltwater Depth` in physical metres.
- `Watershed Analysis` routes this physical water depth through the drainage
  network and publishes cumulative `Runoff Volume` in cubic metres.
- When `Runoff Volume` is connected to `Lake Basin`, the lake is no longer
  assumed to be full up to its spill elevation. The available runoff volume is
  compared with the terrain-derived basin capacity and the water surface is
  solved from the basin's volume–elevation relationship.
- Therefore snow amount, melt amount and meltwater location can affect lake
  water quantity, flooded area, depth and surface level.
- The visual runoff trace is not included in physical storage because it can
  count the same moving water in multiple cells. Only the final liquid-water
  depth contributes to the conserved volume input.
- Existing graphs remain compatible: all new pins were appended. A `Lake Basin`
  without a connected physical runoff input retains the previous behavior and
  fills geometric basins to their spill level.
- Generated river spline control points retain their hydraulically calculated
  water level, but are clamped against the final terrain heightmap with a small
  clearance. This prevents river surfaces from passing below the terrain while
  avoiding the slope distortion caused by enabling generic `Follow Terrain`.

The changes were reviewed with static source and diff checks. Project build and
runtime scene validation are intentionally left to the project owner.
