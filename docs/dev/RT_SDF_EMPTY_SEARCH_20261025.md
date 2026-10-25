# SDF exterior empty-space search — 2026-10-25

Implements the SDF search optimization identified by
[the near-camera investigation](RT_CLOSE_VOLUME_REENTRY_20261025.md).
No build or application launch was performed. GPU speed and visual acceptance
remain unverified until the user builds these changes.

## Implementation and safety boundary

`surface_sdf_empty.glsl` provides one raw-proxy hierarchy proof for both the main
SurfaceSDF search and the gas-to-SDF ordering search. The original search loops
and refinement equations were extracted into `surface_sdf_main_search.glsl` and
`surface_sdf_arbiter_search.glsl`; the large closest-hit shader only wires them.

An exterior segment can advance over multiple original sample positions when:

- The source is a NanoVDB SurfaceSDF and the previous sample is below iso=0.5.
- A constant tile/background value, or a certified live-producer leaf maximum,
  proves every raw trilinear sample in that region below 0.5 with a rounding
  margin. Arbitrary imported leaves without a complete inactive-value bound
  retain normal sampling. Missing statistics/non-finite bounds also fall back.
- The entire interpolation footprint remains in the region: its final voxel
  slab must use normal sampling because it can see a nonempty neighbor.
- Advancement stays inside the existing interval and iteration budget. It uses
  whole original step intervals and re-samples the retained endpoint, preserving
  the next crossing bracket and the original four refinement steps.

Fog density multiplier, remap and cutoff are never used to classify geometry.
Pores and opacity masks only subtract from the current raw field, so a proven
exterior upper bound remains safe with these modifiers. Interior/exit searches
are deliberately unchanged: holes may exist inside a raw-full region.
The remaining iteration budget and per-consumer hysteresis/threshold rules are
preserved. Unknown or unsupported storage is a normal-search fallback, never a
claim that the surface is absent.

Each search caches its most recent raw hierarchy bound, including rejected
bounds. A dense leaf therefore does not repeatedly load its maximum on every
small step. The cache is local to one field/ray/search and is not reused across
reflections, refractions, shader invocations or publication changes.

Solid/splat distance probes, gas overlap selection, SurfaceSDF handoff masks,
foam integration, normal sampling and shading are unchanged. This patch does
not share a final crossing result between the two searches: their original step
and start-side policies differ, so that would need a separate correctness change.

## Canonical diagnostics

The existing Volume Performance controls and `render.volume_counters` enable the
same backend instrumentation; default remains off. The panel and copyable report
now show six independent fields, also available via both Python
`rt.render.volume_stats()` and IPC `render.volume_stats`:

| Fields | Meaning |
| --- | --- |
| `sdf_main_samples`, `sdf_arbiter_samples` | Actual initial, retained endpoint and refinement proxy samples in each search |
| `sdf_main_queries`, `sdf_arbiter_queries` | Uncached raw-region classifications, not hardware memory transaction counts |
| `sdf_main_skipped`, `sdf_arbiter_skipped` | Original sample positions omitted by a successful proof; retained endpoint excluded |

These do not redefine gas `density_samples` or `empty_segments_skipped`. They
are not total SDF reads: normal, foam, shadow and SSS consumers are outside this
search-only scope. Region filtering, opt-in behavior, reset/refresh semantics,
multi-device summation and existing validation/error behavior remain shared.
There are no new controls or UI-only operations.

The instrumentation SSBO appends six uint32 words: 184 -> 208 bytes. Existing
word offsets, including enabled and normalized region, are unchanged. C++, GLSL,
API copying, Python and IPC mappings were updated together. No scene geometry
representation or scene serialization changed.

## Executed non-build validation

- `check_surface_sdf_empty.py`: 6 checks, including 400 randomized field/ray cases
  under both search policies, adjacent dense cells, interior holes, crossing
  refinement equivalence, read reduction, counter ABI and public surfaces.
- Existing RT optical checks: 22 passed.
- Existing whole-volume empty rejection checks: 7 passed.
- Volume material checks: 10 passed.
- Matter phase source contracts: passed.
- IPC descriptor consistency and capability audit: passed (691 methods).

Reference/source checks do not compile or execute the Vulkan shader.

## User build and live checklist

1. **Build C++ and shaders together.** This is not a shader-only update because
   the instrumentation buffer grew. Regenerate at least `volume_closesthit.spv`,
   `closesthit.spv` and `raygen.spv`, which include the shared instrumentation
   layout, then restart with the rebuilt application. Do not load the new shader
   into the old binary with its smaller instrumentation buffer.
2. Keep the close camera, frame 28, routes, materials, resolution and sample
   settings fixed. From an external terminal with Studio open and paused:

   ```powershell
   python scripts/test/rt_probe_volume_reentry_ipc.py --output docs/dev/rt_volume_close_reentry_after_sdf_20261025.json
   ```

   The probe takes equal GPU sample windows separately from counters and restores
   render settings. Verify `sdf_counters_available` is true, then compare GPU
   sample indices 0..7 against `rt_volume_close_reentry_20261025.json`. The old
   report has no SDF counters; absent old fields are not zero measured SDF work.
3. Check empty SDF exterior around a small surface, a gas layer before the
   surface, and embedded solids/splats closer than the SDF. Both outside and
   inside camera positions must preserve the closest visible surface and gas.
4. Check adjacent leaf/tile boundaries, negative grid coordinates, grazing rays,
   thin disconnected surfaces, refraction and porosity/opacity holes. Interior
   exits must retain their old surface; unsupported bounds must remain correct.
5. Recheck optical density=0: SDF geometry must remain, while truly inert gas
   still takes the prior whole-instance early rejection. Check foam separately;
   its integration has not been replaced by an SDF proxy emptiness proof.
6. Disable counters for ordinary timing/use. Evaluate main and ordering sample
   reductions separately. A dense surface may legitimately yield few skips;
   faster rendering or zero surface-search cost is not assumed in advance.
