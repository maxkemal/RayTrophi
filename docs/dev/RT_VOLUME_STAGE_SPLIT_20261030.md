# SurfaceSDF / fog RT stage split, 2026-10-30

## State

Source implementation and non-build checks are complete. The user built it;
external IPC verified split RT ready (driver creation 205.364 s). Initial live
GPU medians were 139.676 / 136.845 ms, but historical gas hashes differ, so no
controlled speed comparison is established. Image parity remains pending. See
[the live cost check](RT_VOLUME_STAGE_SPLIT_COST_20261030.md). The agent
did not build, launch Studio or replace installed shaders. Implementation checks
used no live IPC; the later cost check temporarily changed measurement controls
and restored them as recorded in its report. The accepted legacy Off shader
remains the default. The driver-rejected all-functions Modular route stays disabled.

This change creates two separately compiled closest-hit modules from explicit
compile-time branches of the canonical source:

- `volume_fog_closesthit.spv`: ordinary fog, overlapping mixture and volume random
  walk. It has no SurfaceSDF material/shading branch in its entry call tree.
- `volume_surface_closesthit.spv`: SurfaceSDF search, foam and surface BSDF.
  It excludes the fog integration body, mixture/cache modules and volume random
  walk definitions. Shared SDF field queries remain available to fog for handoff.

The default `volume_closesthit.spv` still compiles both paths for rollback.
There is no global DontInline annotation or increased RT recursion depth.
Both stages retain their original shadow/probe trace capabilities; shader
recursion remains at the pipeline's existing value of two.

## Source boundaries and data

Sampling, random walk, SurfaceSDF shading and fog shading moved to four focused
`volume_hit_sampling.glsl`, `volume_hit_random_walk.glsl`,
`volume_surface_shading.glsl` and `volume_fog_shading.glsl` modules.
The entry source and all extracted files are below 2,000 lines. At the first
extraction checkpoint, rebuilding the original file from the extracted blocks
matched its pre-extraction SHA-256 exactly:
`79a10357ac9cb4689d2c02a9c12e9b971835aaa7bdde05c6fd1b9fba12cd689b`.
The later changes are stage guards, dispatch and host integration; the moved
equations, RNG calls and march budgets were not rewritten.

Two build invocations use `RT_VOLUME_FOG_STAGE` and `RT_VOLUME_SURFACE_STAGE`.
This is a real translation-unit/pipeline boundary, in addition to source
extraction. Just moving includes would not reduce the reachable shading tree.
The canonical volume SSBO, 672-byte instance layout and RayPayload layout are
unchanged. No per-face geometry collections or new authoring operations exist.
Existing UI, scripting and IPC material/visibility services remain authoritative.

## Foreground gas correctness

A liquid AABB can win BVH traversal before a gas box lying inside it, while
the true liquid boundary is farther away. Legacy `selectForegroundGas` then
switches to the gas integration path. Static SBT routing by source type alone
would lose that path in a surface-only shader.

The split SurfaceSDF stage runs that same selection first. When gas wins, it
returns a reserved `volumeTraversalMask` marker before shading, RNG draws or
cloud compositing. Raygen consumes it immediately after the first trace and
before reading the result or charging the outer pass/bounce budget. It issues
one gas-only trace with the original origin, direction, tMin/tMax and seed.
The canonical fog overlap resolver still chooses and sums the contributors.
The surface stage skips cloud compositing on this signal, leaving one cloud
prefix application in the final fog stage. A gas-only mask cannot select SDF,
so this redispatch does not recurse or loop. Photon traces exclude domains and
need no new route. A sentinel cannot alias positive solid-distance encoding.

This can add an actual traversal for the SDF-box-first case; it is not free GPU
work, even though it consumes no path budget. Independent shader invocations
also do not share invocation-local density caches. Performance must be measured.

## Host routing and consistency

The normal fog/legacy volume hit group stays at offset 1. SurfaceSDF is appended
after the existing optional hair/sphere groups, preserving those offsets.
CPU instance packing and both GPU instance preparation paths use the same
`getVolumeRoutedSbtOffset` policy for reserved domain masks. Installation refreshes
the TLAS and pending GPU metadata before tracing; mode changes refresh domain
masks through the existing volume update path. Volume slot/custom-index identity
is unchanged. The shared volume intersection module serves both hit groups.

Shader-set hashing includes the new stage. Pipeline status reports actual
`volume_fog_closesthit`, `volume_surface_closesthit`, `volume_intersection` stages.
The volume-free A/B omission clears both closest-hit modules before hashing.
Missing/invalid paired modules fail selection/creation; a fog-only pipeline is
never installed while a paired surface module has failed creation.

Runtime selection is process-start opt-in `RT_VK_SPLIT_VOLUME=1`. With it absent
or not exactly 1, the existing legacy stage remains selected. The three coupled
modules (raygen, fog, surface) have a consistency manifest checked against the
actual loaded words. Its word-wise FNV stamp detects accidental mixed deployment;
it is not a security signature. Missing/mismatched files or manifest log a warning
and keep the legacy shader. A failed or partial shader deployment cannot silently
pair the new surface handoff with an old raygen.

## Non-build checks

88 Python tests passed, including seven new stage/routing contracts and existing
SDF, mixture, material, empty rejection and pipeline A/B checks. These cover
stage entry-tree exclusion, foreground signal/cloud/RNG ordering, mask/interval
preservation, coincident contributor visit order, sentinel separation, all
hair/sphere SBT combinations, CPU/GPU packing hooks, source registration and
opt-in/pair checks. They are source/math oracles, not execution of C++ or GLSL.

Light-cache and march-LOD source audits passed. The consistency-manifest mock
checks passed creation, replacement, known checksum, malformed-input rejection
and old-stamp preservation. Existing ten compiler-policy mocks and the seven
Modular staging mocks plus the Modular-refusal check passed. Split and legacy
build plans passed `-ValidateOnly`, invoking no compiler. Source-audit readers
now read the extracted implementations; a stale SDF ABI check was adjusted to
compare the full actual GPU/CPU scalar layout and asserted byte count instead
of an outdated fixed 52-field/208-byte assumption (the existing layout has 64
fields/256 bytes; this change did not alter it).

## User build and first driver acceptance

Keep current work saved, then close Studio before replacing the executable or
raygen. Build C++ with the usual user workflow; the new focused backend module
is registered in the Visual Studio project and is covered by the existing CMake
source glob. The agent must not perform that build.

From the repository root, build/deploy only the three split-related shaders:

```powershell
.\RayTrophiStudio\compile_shaders.bat -SplitVolumeOnly -VolumeOptimization Off
```

This preserves the working legacy volume SPV and unrelated shaders, validates
the new stages and raygen, and stamps only a complete deployed set.
It also prints a `Pre-split shader backup` path under ignored captures before
compilation, retaining prior raygen, paired shaders and stamp for source and
each existing runtime directory. Keep this directory through driver acceptance.
Start the new executable from a PowerShell with the opt-in set (use the binary just built):

```powershell
$env:RT_VK_SPLIT_VOLUME = '1'
& .\x64\Release\RayTrophiStudio.exe
```

Require driver pipeline ready and the three split stage names in
`render.rt_pipeline_status` through the external `scripts/test/rt_ipc.py` client.
The startup log must say `Split volume stages selected`, not missing/mixed
fallback. Do not run this IPC client inside the embedded script workspace.
No full `-O` experiment is required for this first structural acceptance.

Next compare equal camera/frame/seed/sample-count images for gas-only, SDF-only,
overlapping gas in front of/inside SDF, camera inside both, gas-hidden/SDF-hidden,
coincident faces, solid geometry, foam, SSS/refraction and absorption. Exercise
material mode changes, empty/rebirth, hide/show, timeline rewind and transformed
volumes, plus hair and procedural spheres to validate unchanged group offsets.
Finally repeat the fixed hidden/A/B/A+B/A+B GPU timing matrix with counters off,
and collect work counters in a separate pass. Check physical field hashes and
restore visibility/instrumentation as in the original matrix.

Rollback is process-start only: close Studio, unset the flag or set it to `0`,
then relaunch the same new binary. It loads the retained legacy volume module.
If new raygen itself fails, restore its previous artifact as well; no driver
acceptance is claimed until the user completes these steps.
