# Compact mixture material response, 2026-10-30

## Change

The compiler diagnosis reproduced SSA ID exhaustion and process stack overflow
after merge-return/exhaustive-inlining expansion. This first source change
reduces aggregate copies and separates the mixture shadow consumer's call tree.
It is not yet a verified resolution of full optimized compilation.

`volume_mixture_material_sample.glsl` now owns canonical mixture sampling and
graph evaluation. Instead of copying a 672-byte `VkVolumeInstance`, applying
graph overrides to it, and passing that full aggregate into response evaluation,
the code passes the volume index and `MfOpticalResponse`: three colors and five
scalars (14 scalar components). This is a shader-local record, not a GPU buffer
ABI or a claim about actual register allocation/physical copy bytes.

Transforms, ramp arrays and acceleration data remain in the existing read-only
volume SSBO. The response and sampling modules have no `VkVolumeInstance`
variables or parameters. `mfDensity` takes the canonical index, and
`mfInitializeNano` takes only the resolved grid address. Lower-level legacy
density, temperature and thermal helpers still take full instances; this change
does not claim to remove all aggregate copies from the monolithic shader.

Camera and graph-capable shadow sampling call the same `mfSampleMaterial`.
Both use `mfExtinctionTerms` for the colored optical/similarity equations.
Camera emission is added by `mfOpticalTerms`; shadow `mfShadowExtinction` never
calls that emission consumer. Prepared NanoVDB shadow stepping is unchanged.
This is a real change in data interfaces and call edges, not only text extraction.
It is not a new shader stage, callable, hit group or full pipeline split.

## Preserved behavior

- All nine graph-write bits retain their previous defaults, masks and clamps.
  Graphs still run with zero raw density and may produce nonzero density there.
- Shadow graphs still receive temperature, coordinates, authored input colors,
  time and all the same material-program inputs. Noise/non-NanoVDB/further-media
  fallbacks remain enabled. Density sample counters and membership/bounds gates
  remain in their previous order.
- Independent scatter/absorption colors, black absorption's neutral fallback,
  phase/similarity threshold and event ordinal retain their operation order.
- Plain-color emission uses graph-overridden emission color/intensity and the
  existing authored response. Thermal/ramp emission still uses the canonical
  `volumeEmissionAt` with the graph-adjusted density and sampled temperature.
  That helper's thermal branch reads blackbody/ramp strength, not the graph's
  plain-color emission fields; passing the canonical instance preserves this
  existing behavior. No thermal equation/ramp implementation is duplicated.
- No march budget, sample positions, random draws, cache policy, SSBO layout,
  descriptor/SBT/payload ABI or authoring operation changes. Existing scripting
  and IPC controls continue through their existing implementation; this internal
  refactor introduces no new user-facing API.

## Non-build verification

67 tests passed across compact response, shared walk, overlap cache, RT optical
contracts, adaptive mixture, step budgets, volume materials and empty rejection.
The new numerical oracle covers all 512 graph-write masks across walking on/off
and pre/post similarity events, with independent colors and zero-input-density
graph cases. Source checks verify the extinction-only consumer has no edge to
the emission response and both consumers retain common graph evaluation.
These are Python math/source checks, not execution of the GLSL shader.

The light-cache and march-LOD source-audit scripts also passed. The modified RT
optical test was synchronized to its existing Release test mirror to satisfy its
mirror check; no shader, application binary or compiled artifact was deployed.
The agent ran no build, shader compiler, optimizer pass or application launch.

## User acceptance sequence

1. From the repository root run the same isolated diagnosis against the changed
   source. It writes only scratch outputs and preserves the installed shader:

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/test/diagnose_volume_compiler.ps1
   ```

2. Share its output directory. Compare reference/inline sizes and static counts
   with the diagnosed baseline, and check strict full-optimization success plus
   validation. `Auto` fallback to `-O0` must not count as optimizer acceptance.
   If full optimization still fails, keep the validated reference and continue
   reducing the diagnosed call tree; do not install a failed checkpoint.
3. After a successful diagnostic, install only through the existing user shader
   workflow. The baseline mode remains:

   ```powershell
   .\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off
   ```

   The subsequent all-functions boundary experiment passed offline checks but
   crashed at RT driver compilation. Modular deployment is disabled; retain
   `Off`. `Full` still requests the original, crashing direct glslc optimization
   path. No C++ build is required by this source change. For image acceptance,
   compare the previous
   shader at equal camera/frame/seed/sample count, including plain emission,
   thermal ramps, graph-modified extinction, zero-density graph output, noise,
   random walk and more than two overlapping contributors.
4. Once image parity is accepted, repeat the paused hidden/A/B/A+B/A+B timing
   matrix with counters off, followed by a separate counter pass. This change
   does not repair the separate persistent-light-cache policy difference and
   makes no measured FPS/GPU-time improvement claim.
