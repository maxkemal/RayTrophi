# Terrain Surface Relief and Structural Hardness

Date: 2026-08-28

## Outcome

The uniform `Ground Detail` displacement was removed from `Noise Generator`.
Two focused terrain nodes now own the missing responsibilities:

- `TerrainV2.StructuralHardness` derives effective erosion resistance from the
  source height, slope, convexity and local sharpness. Optional Lithology is the
  intrinsic substrate and optional authored hardness remains an explicit final
  override.
- `TerrainV2.SurfaceRelief` adds resolution-safe rock fabric, soft soil relief
  and flow-aligned rills after erosion. It outputs Height plus reusable Rock
  Detail, Rills and Soil Detail masks.

Both types are registered with the shared node service. UI, Python and IPC use
the same constructors, JSON properties, validation and evaluation code.

## Flow audit

Hydraulic Erosion already publishes the data Surface Relief needs:

- `Discharge`: raw `m3/s`, not a pre-clamped mask.
- `Drainage Area`: raw square metres.
- `Flow Direction`: a two-channel unit vector field populated by both the CPU
  LEM path and GPU readback, then normalized at the node boundary.

The scalar `Flow` node remains the authority for magnitude/classification.
It deliberately does not pretend to carry direction. Surface Relief therefore
reads Flow for strength and Hydraulic Erosion's vector output for orientation.
When direction is absent it falls back to the local downhill height gradient.

The ready `snowy_mountain_valley` graph wires:

```text
Noise -> Structural Hardness -> Hydraulic Hardness
  |              |                    |
  +----------> Hydraulic Erosion -----+--> Slope / Flow / Soil
                                             |
                                             v
                                      Surface Relief -> Snow -> Height Output
```

Surface Relief intentionally does not feed its micro displacement back into
Hydraulic Erosion. Doing so would create a graph cycle and let sub-metre detail
rebuild the macro drainage network. The added-slope limit keeps this terminal
stage in its intended scale range.

## Hardness audit

The old `TerrainManager::autoGenerateHardness` path had three problems:

1. Absolute elevation was treated as hardness, so every summit became hard
   independent of lithology or exposure.
2. One shared random generator was called inside an OpenMP loop, making the
   result non-deterministic and data-racy.
3. The UI action and graph Lithology path used different business logic.

The UI action now calls the same deterministic structural core as the node.
Steepness is treated as evidence of exposed bedrock, not as intrinsic rock
type. Lithology supplies intrinsic hardness; convexity/sharpness modify surface
exposure; fractures locally lower resistance. Hydraulic Erosion already
multiplies incision by hardness on CPU, Vulkan and CUDA paths, so no separate
erosion implementation was added.

## Geometry gaps found

- `Terrain Detail` is still a generic uniform three-band displacement. It is
  retained for authored legacy graphs, but built-in noise terrain no longer
  uses uniform ground detail as its surface solution.
- Flow direction is authoritative only when Hydraulic Erosion or another
  vector producer is connected. Height-only graphs use a downhill fallback;
  they cannot express lake spill routing or erosion history.
- Structural Hardness is a pre-erosion exposure estimate. A fully iterative
  model that uncovers new strata during erosion needs solver-owned evolving
  material state rather than a graph feedback loop.
- Surface Relief runs on the field grid, while final geometry may use a coarser
  mesh grid. Its band limit uses the coarser of field and mesh spacing so it
  does not author frequencies the final TriangleMesh cannot represent.
- Surface detail masks for SatMap remain paint-domain material masks. They are
  not geometry and are intentionally separate from `Surface Relief`.

## User verification after build

1. Apply `snowy_mountain_valley`; confirm the graph contains Structural
   Hardness before Hydraulic Erosion and Surface Relief after it.
2. Preview Structural Hardness: steep exposed ridges should be resistant, but
   fracture lines and low soil-covered slopes should remain softer.
3. Preview Surface Relief Rills: they should follow the hydraulic vector field,
   not a global texture direction.
4. Set rock/soil/rill amplitudes to zero; Height must exactly match its input.
5. Test a coarse mesh over a high-resolution field; the node's effective
   feature readout should rise with mesh cell size.
6. Run `scripts/test/rt_test_terrain_surface_relief.py` inside Studio.
7. Run `scripts/test/rt_test_terrain_surface_relief_ipc.py` externally while
   Studio is open.
