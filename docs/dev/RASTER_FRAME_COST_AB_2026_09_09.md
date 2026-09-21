# Raster material cost: live A/B, 2026-09-09

User authorized driving the already open foliage/water scene. No application
launch, project build, shader compilation, or renderer implementation changes.

## Method

1680 x 945, **balanced**, scene lighting, RT shadows and depth prepass active.
Do not compare absolute timings with the earlier handoff's performance preset.
Camera X alternates by 0.0001 world units to trigger dirty frames; eight warmup
writes precede each timing reset. Standard arms collect 19 reported frames;
the transmission verification repeat collects 13. Raw stage timings and applied
settings are saved under `tmp/`.

## Results (GPU milliseconds)

| Arm | Visible triangles | Draws | Frame | Main | Transmission |
|---|---:|---:|---:|---:|---:|
| Initial baseline | 29,497,832 | 34 | 376.68 | 206.84 | 148.30 |
| World solid color | 29,327,572 | 34 | 201.15 | 94.70 | 85.22 |
| Nishita restored | 29,327,572 | 34 | 372.00 | 204.16 | 146.51 |
| Water transmission scalars zero | 29,327,572 | 34 | 377.36 | 207.07 | 148.73 |
| Water scalars restored | 29,327,572 | 34 | 370.75 | 203.59 | 145.84 |
| Solid viewport shading | 66,041,127 | 34 | 25.02 | 24.09 | SKIPPED |
| Opacity baseline | 29,327,572 | 34 | 380.79 | 208.84 | 150.16 |
| Four needle materials: opacity zero | 29,327,572 | 34 | 176.54 | 103.78 | 56.45 |
| Needle opacity restored | 29,327,572 | 34 | 378.07 | 207.75 | 148.80 |
| Transmission verification baseline | 29,327,572 | 34 | 380.79 | 209.05 | 149.98 |
| Transmission zero, per-write API readback | 29,327,572 | 34 | 382.29 | 209.93 | 150.39 |
| Transmission restored again | 29,327,572 | 34 | 368.24 | 202.25 | 145.01 |

Initial baseline triangle count drifted; the world-solid/restored pair matches.
Solid shading reports a different geometry workload, so it is not a pure
same-geometry shader comparison. Its speed nevertheless contradicts treating
raw triangle count alone as an explanation for the material-mode cost.

CPU recording in the first material arms is approximately 0.69–1.10 ms.
Nishita background sky stage is only about 0.094 ms (solid world: 0.045 ms).
The large world-mode difference lives in main/transmission, not background sky.

## Interpretation and limits

- World solid color reduces frame cost by about 46% versus the matching restored
  Nishita arm. This changes environment radiance, world sun eligibility, and
  shader branches together: it is not an isolated sky lookup benchmark.
- Zeroing four foliage opacity scalars reduces frame cost by about 53% and the
  restored arm returns within 3 ms. Geometry submission stays identical. This
  removes needle fragments, including their shading, refraction, visibility and
  depth effects; it does not isolate discard cost from lighting or transmission.
- Forty materials have authored transmission 1, all water; four needle materials
  have opacity textures, scalar opacity 1, scalar transmission 0. No transmission
  textures were returned by the inventory. Intermediate needle alpha can still
  trigger the shader's legacy opacity-to-transmission rule.
- Zeroing authored transmission does **not** stop replay in the running build.
  Setter/getter roundtrips succeed in the repeat, but this is not proof of the
  effective per-fragment GPU classification or persistent terrain-generated
  values. Do not conclude that refraction is cheap from this arm.
- Opacity textures are embedded names, not reloadable file paths. Their bindings
  were not cleared. Scalar opacity zero is a contribution test, not an opaque
  cutout replacement test.
- Source and running shader binaries were not established to be identical.
  In particular, source already contains sky IBL and early replay-discard work.
- Skipped stages can carry a nonzero numeric timing in this running build
  (solid mode reported transmission 0.84 ms with frames_ran=0). Respect the ran
  flag, not the number alone.

## Next engineering target

Final IPC readback: **44 values checked, zero mismatches**; Nishita and material
shading restored. Lighting reports `world_ibl_ready=true`, source `sky`,
`world_ibl_fallback=false`. Therefore missing sky IBL is not demonstrated by
this test; inspect remaining sky consumers and actual shader branch selection.

Measure effective replay classification before
changing depth semantics. Check the legacy partial-alpha transmission path and
avoid replaying provably opaque meshes. Any EQUAL/early-tests variant must also
preserve partial-alpha coverage, material graph behavior, and the opaque-phase
back-depth imageAtomicMax side effect; invariant vertex positions alone are
insufficient. No claim of a specific post-fix frame time is supported yet.

## Artifacts

- `scripts/ipc/Inspect-RasterMaterials.ps1`: capture scalar values/texture names.
- `scripts/ipc/Probe-RasterMaterialCost.ps1`: same-camera tests, optional
  `-OpacityOnly` / `-TransmissionOnly`, restoration in finally.
- `scripts/ipc/Verify-RasterProbeRestore.ps1`: 44 changed-value readbacks and
  current world/shading/lighting state.
- `tmp/raster-material-snapshot.json`, `tmp/raster-material-cost.json`,
  `tmp/raster-opacity-cost.json`, `tmp/raster-transmission-cost.json`,
  `tmp/raster-probe-restored.json`.

Scripts use the captured scene-specific snapshot: recapture before a different
scene or later edits. Run only one IPC client at a time. Material setters mark
the project modified; restoring values does not clear that flag. No project
save was requested or performed.
