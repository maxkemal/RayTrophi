# RT gas: shared camera-cell traversal

## Compiler optimization experiment

Update, 2026-10-30: SSA rewrite failure was reproduced after excessive inlining.
An all-functions DontInline variant passed offline optimization/validation but
subsequently crashed during the user's RT driver compilation. Modular deployment
is now disabled, and the validated unoptimized reference was restored to source
and Release shader paths. Use `Off`; do not install the protected diagnostic
artifact. The earlier `Full`/`Auto` commands below retain their original failing
direct-optimizer behavior for historical experiments. See
[the compiler diagnosis](RT_VOLUME_COMPILER_DIAGNOSIS_20261030.md) for evidence
and rollback details. Offline success was not driver/GPU acceptance.

The user observed only a small improvement after the shared-walk shader build.
The performance problem remains open. The user subsequently reported a long
`-O` attempt ending in access violation (-1073741819 / 0xC0000005), followed
by a validator layout error on the `-O0` fallback. Default is now `Off` / `-O0`
to avoid repeating the expensive failing optimizer on routine builds.
Validation now passes `--scalar-block-layout`, matching the shader's required
GL_EXT_scalar_block_layout. Inspection also found the runtime feature had been
disabled by an old timing-bisect toggle. VulkanBackend now requests the supported
scalarBlockLayout feature and attaches it through its existing pNext wiring.
That correction requires a user C++ build as well as the shader workflow.
Pipeline creation time must be rechecked; no performance benefit is asserted.
The existing SPV passed a real read-only spirv-val check with scalar layout.
No buffer offsets or shader math changed for this validation correction.
Explicit `Auto` mode tries `-O`, validates staged SPIR-V, then publishes.
A compiler crash/nonzero exit or validation failure retries once with `-O0`.
Publication errors do not trigger an optimization fallback. If neither attempt
succeeds, the previous SPV remains installed. This contains failures; it does
not assert that the optimizer will succeed or that valid SPIR-V proves image parity.

From the repository root, user-only build commands:

```powershell
.\RayTrophiStudio\compile_shaders.bat -VolumeOnly
# Explicit optimizer experiment with fallback (known to be slow/fail on this SDK):
.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Auto
# Strict optimized A/B (fails instead of silently using the reference):
.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Full
# Reference build:
.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off
```

The final `OK` line includes the actual flags. `VolumeOnly` publishes/deploys
only the volume shader and skips unrelated stale-file cleanup. Restart/reload
the shader through the application's normal workflow before the next capture.
The mutex and atomic staged publication remain in use. Mode `Auto` is not
performance evidence when its final line reports `-O0`.

`scripts/test/check_shader_optimization_policy.ps1` extracts driver functions
through the PowerShell AST and tests them with mock compiler/validator functions;
it never runs glslc, spirv-val, optimization or a project build. Ten cases cover
success, stack failure, validator failure, explicit reference, strict failure,
both attempts failing, publication failure and retry exhaustion. Temporary mock
files are deleted from ignored workspace scratch. Plan validation (`-ValidateOnly`)
also does not invoke the compiler. Real optimized compilation and GPU/image
acceptance remain with the user.

Source implementation, 2026-10-29. Shader build and GPU acceptance pending.

## Scope

The first specialization covers exactly two active Matter fog fields with
identical camera index/local rays, bounds and intersection intervals. Both
must be NanoVDB, adaptive, without density noise or a material program.
Random walk, a third participating medium, unequal support, or unequal ray
coordinates retain the general integrator. Overlap alone never proves alignment.
Aligned fields from the same simulation domain satisfy the intended use case;
the shader proves the required geometric equivalence rather than guessing a
domain identity from volume IDs or names.

No raw density merge, temperature average, new authoring controls, or GPU ABI
changes are introduced. Existing UI/Python/IPC material operations remain the
canonical controls. This is an internal rendering specialization.

## Work shared

`volume_mixture_shared_walk.glsl` selects a common integer cell and exit, sums
the two independent conservative slope bounds, and prepares a common midpoint
and trilinear fractions. Each field retains its own grid corners, density
remap/curve, cutoff, multiplier, emission response and temperature lookup.
Sparse traversal still requires both fields to prove emptiness.

The prepared densities bypass the general material-program evaluator during
camera coefficient gathering. `volume_mixture_optical_terms.glsl` contains the
same canonical optical response for both the specialized path and the general
graph-capable fallback. Graph overrides feed that common response. Summation
and the scattering contributor reservoir stay in their original order; the
fast path preserves camera density sample accounting.

The existing camera march, optical error target, refinement floor, authored
step ceiling, coverage fallback, opacity termination, light sampling and shadow
march are preserved. This does not yet implement a common shadow lattice,
share temperature VDB storage, optimize three or more fields, or move volume
integration into a separate compute shader. No speedup is claimed before GPU
measurement; register allocation and driver compilation remain relevant risks.

## Non-build checks

`python scripts/test/check_volume_shared_walk.py` passes four tests: 2,000
randomized aligned two-field partitions with negative rays and independent
materials, a complete reference/shared march with transmission and nonlinear
independent thermal emission, fallback wiring/lifetime, and common optical
response/reservoir wiring. The Python oracle is not execution of GLSL.

Eight existing volume check scripts also pass: mixture adaptive, step budget,
material, optical blocks, light cache, march LOD, grid residency and empty
rejection. In total, 43 unit tests and two additional source-audit scripts pass.

## User build and acceptance

Rebuild RT shaders through the existing workflow (`volume_closesthit.rchit`
includes the new modules); no C++ change is required for this optimization.
After installing the new shader, compare the current paused scene without
changing its camera, frame, material settings, resolution or physical fields.
Use the external IPC client; keep timing counters disabled for GPU captures,
and obtain work counters in a separate pass. Report warm median and cold peak
separately because clock ramp-up affects peaks.

Compare first / second / both-visible configurations and repeat the both case.
Check images at equal seed and sample count against the previous shader.
Exercise independent absorption colors, temperature ramps, multiplier zero,
different curves, and transformed aligned fields. Verify fallback images for
three contributors, unequal boxes/transforms, graphs/noise, adaptive off and
random walk. GPU speed and image parity remain unverified until that build.
