# Volume compiler diagnosis, 2026-10-30

## Conclusion and limits

### Performance priority returned to the large two-gas overlap penalty

Small split/legacy deltas are not the principal problem. The confirmed source
policy gap was that mixture fixed-light visibility never consumed the persistent
light cache used by ordinary fog. A guarded RGB-preserving consumer and authored
shadow-count fill floor are now implemented; 96 non-build tests passed. No new
shader build or performance acceptance is claimed. See
[the focused mixture change](RT_GAS_MIXTURE_LIGHT_CACHE_20261030.md).

### Legacy restart measured in the same cost band; cross-run fields still differ

The user reopened legacy mode; IPC verified the legacy hit stage installed.
Fixed-index GPU medians were 139.464 / 144.266 ms versus split 139.676 /
136.845 ms. Camera/frame/material/settings match, but density/temperature hashes
changed again across reopening. No causal performance percentage or speedup is
established, and no large split-only penalty is demonstrated by these runs.
Both modes are driver-accepted. Further restart-based A/B without frozen voxel
identity should not be treated as valid. Details and restored-state evidence are
in [the live cost report](RT_VOLUME_STAGE_SPLIT_COST_20261030.md).

### Split driver accepted; initial timing comparison is inconclusive

The user built and opened the split version. External IPC verified both split
hit stages plus the shared intersection stage installed and RT ready.
Two independently rearmed fixed-index captures measured 139.676 / 136.845 ms.
The older 123 ms matrix has different gas voxel hashes/density sums, despite
matching camera/frame/material settings, so it does not isolate a split-stage
regression. A same-binary legacy control with identical field hashes is pending.
See [the live measurement report](RT_VOLUME_STAGE_SPLIT_COST_20261030.md).
All measurement controls were restored; no build or app launch was performed.

### First real shader-stage split implemented, acceptance pending

See [SurfaceSDF / fog RT stage split](RT_VOLUME_STAGE_SPLIT_20261030.md).
SurfaceSDF and fog/mixture now have separate compiled closest-hit variants and
hit groups, with same-pass foreground-gas redispatch to preserve the existing
SDF-box-first behavior. Existing hair/sphere offsets remain stable, CPU/GPU
instance packing shares one routing policy, and a coupled-stage stamp prevents
mixing old raygen with the new handoff. Selection is opt-in; the working legacy
Off artifacts have not been replaced by the agent. 88 Python non-build tests,
source audits and mocked build/stamp checks passed. User C++/shader compilation,
driver acceptance and image/timing validation remain pending.

### Recovery confirmed by the user

After the unoptimized reference was restored, the user reports driver compilation
completed without problems and RT is open. Recovery is therefore accepted by
the user for the restored Off/reference module. No additional application launch,
live setting change or performance capture was performed by the agent.

Current accepted state: Off/reference shader runs; direct Full optimization
still fails offline, and the all-functions DontInline/Modular variant passed
offline validation but failed driver compilation and remains disabled.
This restores usability; it does not resolve the original optimization failure
or the two-overlapping-gas runtime cost. Subsequent work must create bounded
functional shader/stage interfaces with driver acceptance, followed by the
fixed-scene image/timing comparison, rather than re-enable the rejected variant.

### Driver compilation rejection: Modular withdrawn and reference restored

After deployment the user reported Studio crashes during RT driver compilation.
The current `x64/Release/StartupCrash.log` records two startup exceptions at
11:09:52 and 11:10:41 on 2026-10-30, code `0x80000003`. This log has no module
or stack attribution. `StartupCrashSignal.dmp` is dated 2026-10-22 and must not
be presented as a dump of this crash. The stale agent startup Python log is
also not evidence about this RT failure.

SPIR-V validation and successful offline optimization did not establish driver
compatibility. The all-functions DontInline variant failed runtime acceptance;
it is not an accepted production fix. Exact driver failure mechanism is still
unknown. In particular, no evidence establishes whether DontInline itself,
retained call/parameter shapes, optimization changes, or another interaction
caused the crash. A valid module can still trigger a driver compiler defect.

The production build entrypoint now refuses Modular mode before any compiler
or deployment activity. Default Off remains unchanged. Scratch diagnosis tools
remain available, but no further all-functions variant should be installed.
The seven helper mock cases remain unit coverage of staging semantics, with an
additional entrypoint test for refusal before tools run.

No Studio process was running at rollback. The latest unoptimized reference
from `volume-compiler-7f756b709f6f4b99a9c65de3fef9655b/reference.spv` was checked
with read-only spirv-val (scalar layout) and restored to both source and Release
shader paths without any compilation or app launch. Its source/include hashes
match the current shader sources. Both restored files have SHA-256:
`a1246e4592d8456a5042bd8e00efda8b987ccbb685f2adfebd8f860d89a7b6aa`.
The rejected source/Release modules and current crash log were preserved in
ignored `docs/dev/.captures/rt-driver-rollback-30474d8101054d07a01422e98926723d`.
The shader-set cache keys include module contents, so no blind cache deletion
was performed. Restart/RT driver acceptance of this restored reference is still
for the user to verify.

Next structural work should bound actual per-stage functionality and hot data
interfaces rather than impose DontInline on every function. The original
single-stage shader still combines SurfaceSDF/BSDF, ordinary fog, mixture,
random walk and graph evaluation; introducing real stage boundaries also needs
SBT/TLAS/payload regression coverage. Do not call the problem solved on the
basis of validator or file-size acceptance alone.

### User build and deployment accepted

The user subsequently ran the normal workflow with `-VolumeOptimization Modular`
and reported the final validated Modular OK line and successful deployment.
Read-only verification found source `volume_closesthit.spv` and the existing
`x64/Release/shaders/volume_closesthit.spv` have the same SHA-256,
`a0c2cbba87d5a2befa3c50f196000b1be8fdea44aa80c610613a998d859d8d69`,
matching the successful diagnostic artifact. The installed source module retains
326 functions / 325 DontInline controls, 42,172 function instructions and
922,772 bytes. This closes Modular shader build/publication acceptance for this
input/toolchain. Direct unprotected `Full` optimization remains a known failing
path, and Modular remains explicit rather than the default.

The agent did not launch Studio or change live application state. Pipeline
creation, appearance parity and the paused-scene GPU cost matrix remain pending;
successful shader deployment does not establish that a running process has
reloaded it or that rendering is faster.

### Protected full optimization passed; explicit build mode delivered

User run `volume-compiler-f0c41b9662f84bcfa41d28029bc350b6` passed every phase,
including full optimization (4.726 s), full validation (1.996 s) and inspection.
The optimized module is 922,772 bytes, 42,172 function instructions, ID bound
157,613, with 326 functions and 325 DontInline controls retained. Its SHA-256 is
`a0c2cbba87d5a2befa3c50f196000b1be8fdea44aa80c610613a998d859d8d69`.
This demonstrates a validated optimization route around the reproduced compiler
failure on this input/toolchain. It is not a measured GPU speed improvement.

The normal shader workflow now offers explicit `Modular` mode:

```powershell
.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Modular
```

This builds a private validated `-O0` reference, retains function boundaries,
validates that module, runs full standalone `spirv-opt -O`, validates the final
module, then uses the existing atomic publication and deployment workflow.
The helper lives in `RayTrophiStudio/compile_volume_modular.ps1`. Any failed
stage preserves the previously installed shader, and private intermediates are
cleaned. There is no fallback to `-O0` in Modular mode. Its final OK line reports
the actual multi-stage route. Python and spirv-opt are additional dependencies
only for this explicitly selected mode; the default remains Off until image/GPU
acceptance. Existing Auto/Full behavior is unchanged for controlled comparisons.

Seven mock build cases passed: success, frontend failure, boundary-transform
failure, boundary-validation failure, optimizer failure, output-validation
failure and publication failure. They verify operation order, old-module
preservation and intermediate cleanup without a real compiler/optimizer.
The ten existing publication-policy mock cases also passed. Agent execution
remains limited to mocks, read-only inspection and plan validation; the user
performs the actual build/install. Next acceptance is driver pipeline creation,
equal-seed image parity and the fixed-scene gas cost matrix from the roadmap.

### Compact response did not resolve full optimization; preserve IR functions

User run `volume-compiler-7f756b709f6f4b99a9c65de3fef9655b`: reference,
reference validation, inline and inline validation passed; full optimization
failed with stack overflow (`-1073741571`) after 35.640 s, again preceded by
ID-overflow messages. The source response refactor is not a compiler fix.
Reference counts changed from 1,023,604 bytes / 50,413 function instructions
to 1,026,840 bytes / 50,593 instructions, with the largest function unchanged
at 9,113 instructions. No reduction in optimizer input size was established.

The next implementation addresses the established lost function boundaries
directly. `scripts/shaders/preserve_spirv_functions.py` changes only the
Function Control operand of non-entry `OpFunction` instructions: it clears
`Inline` and sets `DontInline`. No IDs, types, operations, branches, descriptors,
payloads or sampling equations change. The entry point is excluded. Input is
immutable and outputs require a fresh file. There is no implicit shader install.

This uses a standard SPIR-V control rather than a new GPU stage/SBT layout.
The [Khronos inliner source](https://github.com/KhronosGroup/SPIRV-Tools/blob/main/source/opt/inline_pass.cpp)
explicitly excludes DontInline functions, and
[Khronos change history](https://github.com/KhronosGroup/SPIRV-Tools/blob/main/CHANGES)
records support under issue #3858 (2020-era changes). Exact installed-SDK
behavior must still pass the local experiment; a current upstream source is not
an execution test of the installed binary. Function-call retention may affect
driver optimization/GPU speed, so this remains a diagnostic candidate rather
than an unmeasured production default.

`diagnose_volume_compiler.ps1 -ReferenceModule ... -PreserveFunctions` now:

1. Copies the already-compiled reference to fresh ignored scratch and validates it.
2. Applies the Function Control-only transformation and validates that output.
3. Runs the same inline/full experiments on the protected copy; validates each
   successful output and records read-only instruction/function counts.

No source recompilation is needed for this next experiment:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/test/diagnose_volume_compiler.ps1 -ReferenceModule docs/dev/.captures/volume-compiler-7f756b709f6f4b99a9c65de3fef9655b/reference.spv -PreserveFunctions
```

Four synthetic non-build tests passed for exact changed-word limits, Inline/
DontInline exclusivity, preservation of other flags, entry exclusion,
idempotence and malformed-input rejection. A dry-run on the current reference
identified 325 non-entry functions out of 326, keeping byte count and ID bound
unchanged. The agent ran only that read-only dry-run and `-ValidateOnly`; no
real transformed module, compiler/optimizer pass or installed shader was produced.
Use the returned `inline-inspection.stdout.log` and `full-inspection.stdout.log`
to check whether the compiler actually retained boundaries and reduced expansion.

### First targeted source reduction delivered

See [Compact mixture material response](RT_VOLUME_COMPACT_RESPONSE_20261030.md).
Mixture graph overrides now travel in a small response instead of a mutated full
volume copy; density/NanoVDB setup interfaces carry an index/address, and the
graph-capable shadow consumer terminates at common extinction response rather
than the camera emission consumer. 67 non-build tests and two source audits
passed. User shader compilation/optimizer acceptance and GPU parity are pending;
the diagnosed stack-overflow failure is not yet claimed fixed.

### Stack overflow reproduced in SSA rewrite despite compaction

Latest user run: `volume-passes-600adfc10c6f4891be36d226c74374c9`.
The pass-15 input, ID compaction and compacted-input validation all passed.
Compaction reduced the ID bound from 2,364,690 to 1,158,470 without changing
the 24,413,088-byte module length. SSA rewrite then emitted ID-overflow messages
and crashed after 26.553 seconds with `-1073741571` (`0xC00000FD`, process stack
overflow). No output from that failed phase should be used.

This reproduces the actual stack-overflow failure class in the standalone
optimizer on a valid, compacted module. It is no longer only an inference from
the historical glslc exit code. It does not prove the historical crash used the
same internal call stack, or identify the recursive optimizer routine without
a debugger dump. Compaction alone is demonstrably insufficient on this input.

The actionable diagnosis is excessive IR expansion through merge-return and
exhaustive inlining, then further aggregate expansion, followed by SSA ID
exhaustion/invalid output or process stack overflow in this SDK. These are
compiler-process failures; there is no evidence here of GPU traversal recursion
exhaustion. Do not raise GPU recursion depth or install an invalid checkpoint.

No further blind `-O`, compaction-only or higher-timeout run is needed to justify
source restructuring. Keep the working validated `-O0` shader as the baseline.
The agent has not compiled/optimized or installed any shader during these audits.

Recommended implementation order, for the next change:

1. Reduce the graph passed to optimization rather than just moving text into
   includes. Inventory entry-point responsibilities: SurfaceSDF, ordinary fog,
   overlapping fog, random walk, and material evaluation. Choose actual shader
   specialization/stage boundaries against existing SBT/payload integration;
   a new include is not such a boundary.
2. Shrink hot sampling interfaces from full 672-byte `VkVolumeInstance` values
   toward canonical index/reference access and small optical parameter records.
   Preserve intentional graph overrides in a small local response record instead
   of making/mutating an entire instance copy. Do not change the SSBO ABI merely
   to repair compiler complexity.
3. Keep density/extinction-only shadow and bounds helpers independent from the
   full emission/graph evaluation tree where semantics permit. Graphs that can
   change extinction must retain correct graph evaluation; do not silently
   remove that capability or reduce quality to make the shader compile.
4. Change one boundary at a time; compare static SPIR-V growth and obtain a
   user build with staged validation before measuring the same paused gas scene.
   Any new specialization must preserve fallback coverage and independent media.
5. Address the distinct mixture light-cache/adaptive cost after obtaining a
   stable compiler path. GPU image parity and timing remain acceptance gates;
   neither a valid module nor a smaller module proves faster/correct rendering.

The exact stage split remains a design decision, not an implementation already
delivered by this diagnosis. Over-2000-line source files must receive only minimal
integration wiring under workspace rules. No UI/scripting/IPC capability is
being introduced by the diagnostic tooling.

### First invalid checkpoint identified: SSA rewrite after expansion

User checkpoint run: `volume-passes-977a11ae8ce04c3eab92b6bee5a98032`.
Passes 1 through 15 and their validations passed. Pass 16 `ssa-rewrite`
returned zero after 354.669 seconds, but emitted repeated ID-overflow errors.
Its output failed validation:

```text
error: line 156169: ID '4192130[%4192130]' has not been defined
  OpStore %40053 %4192130
```

This is independently reproduced invalid optimizer output from valid input,
not a reported GLSL syntax error. A zero optimizer exit is insufficient.
The first bad boundary in this checkpoint experiment is SSA rewrite. It is
not yet proven to be the exact crash boundary inside a single-process `-O`.

The upstream expansion is now measurable:

| Checkpoint | Bytes | ID bound |
|---|---:|---:|
| Reference | 1,023,604 | 39,423 |
| 03 merge-return | 1,043,928 | 39,980 |
| 04 exhaustive inlining, after merge-return | 32,408,312 | 1,574,205 |
| 11 scalar-replacement=100 | 27,204,688 | 2,364,690 |
| 15 dead-code elimination (valid) | 24,413,088 | 2,364,690 |
| 16 SSA rewrite (invalid) | 21,942,388 | 4,193,967 |

The pass-15 module contains one function (`main`), 1,399,222 function
instructions, 111,367 local variable declarations and no function calls.
Inlining alone previously grew the module only to 3,629,796 bytes and retained
322 functions. The pipeline's preceding merge-return changes what exhaustive
inlining can expand; it is therefore incorrect to clear inlining interactions
based on the isolated inline-only test. This strongly supports IR expansion
followed by SSA ID exhaustion as the mechanism of the current failure.
Historical compiler stack overflow remains separately unproven.

Next controlled experiment: resume from the valid pass-15 artifact, compact IDs
before each remaining SSA rewrite, validate compaction, and continue the remaining
passes. This tests a diagnostic workaround without changing GLSL or installing
SPIR-V. It may still fail if SSA needs more IDs than compaction recovers.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/test/diagnose_volume_optimizer_passes.ps1 -ReferenceModule docs/dev/.captures/volume-passes-977a11ae8ce04c3eab92b6bee5a98032/15-eliminate-dead-code-aggressive.spv -StartPass 16 -CompactBeforeSsa
```

The agent ran only the corresponding `-ValidateOnly` plan check. The experiment
records resume/compaction settings in `experiment.json`, compaction size/ID
counts in `module-sizes.jsonl`, and normal phase outcomes/logs alongside them.
Passing all remaining validation does not establish GPU image or speed parity.
The structural fix should reduce expansion at real function/stage boundaries
or aggregate interfaces; textual include splitting alone cannot do that.

### Confirmed standalone optimizer failure from the second user run

User-run artifacts: `volume-compiler-a35acc4780794a718d88aa79f2e7bac4`.
Reference compilation and validation passed. Inlining alone passed in 19.293 s
and its output passed validation. Standalone full `-O` failed after 396.278 s
with `-1073741819` (`0xC0000005`, access violation).
The full stderr repeatedly reports `ID overflow. Try running compact-ids.`.
This establishes optimizer ID exhaustion diagnostics before the crash on this
input/toolchain; the triggering pass and underlying compiler defect remain open.
It does not establish that the historical stack overflow had the same cause.

Read-only inspection: inlining grew the module from 1,023,604 to 3,629,796 bytes,
and the ID bound from 39,423 to 169,292. The inline artifact has 211,288 function
instructions versus 50,413 in the reference. Unused function definitions remain;
these totals are not execution counts. Inlining alone being valid does not clear
its interactions with later transforms or the preceding merge-return transform.

Next user-only checkpoint test (reuses the existing reference; no source build
or deployment):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/test/diagnose_volume_optimizer_passes.ps1 -ReferenceModule docs/dev/.captures/volume-compiler-a35acc4780794a718d88aa79f2e7bac4/reference.spv
```

The runner applies the 44 passes advertised by this SDK's `-O` help in order,
validates every output, and stops at the first failure. It records module bytes
and ID bounds after each successful checkpoint. Serializing between passes can
alter ID allocation/pass interaction, so passing every checkpoint would not
disprove the single-process crash. No automatic compact-ids workaround is applied:
first identify the failing boundary, then compare a controlled compaction run.
Agent verification ran only `-ValidateOnly`; 44 checkpoints parsed successfully
and no optimizer/validator/compiler process was invoked.

The historical `0xC00000FD` is a compiler-process stack overflow, not evidence
of GPU ray-stack exhaustion. Later `-O` attempts also reported `0xC0000005`
(access violation). Successful `-O0` builds narrow suspicion toward the
optimization-dependent path; neither record identifies the failing pass.
Do not label this a confirmed recursive shader bug or confirmed inliner bug.

No shader compilation, project build, optimizer pass, application launch or
live IPC mutation was performed for this diagnosis. Existing workspace edits
were preserved. The only new files are this report and a read-only SPIR-V
inspection script. Compiler version/help queries and the ten existing mock
publication-policy cases ran successfully; the latter use no real compiler.

## Evidence

Historical records: `MATTER_VOLUME_STREAMS_PLAN_20261024.md` (compiler crash,
stack overflow, then user-confirmed successful `-O0` build), and
`GAS_RT_SHARED_CELL_WALK.md` (later optimized access violation).

Current `compile_shaders.ps1` defaults to `Off`, compiling volume with `-O0`.
`Full` requires `-O` success; `Auto` can silently change the experiment to
`-O0`, although its final line reports the actual flags. Use neither automatic
fallback nor deployment for root-cause isolation below.

Installed SDK reported by `VULKAN_SDK`:
`E:\RayTrophi_projesi\external_dependencies\1.3.275.0`.
Read-only `glslc --version` reports shaderc v2023.8, spirv-tools v2023.6
(`v2022.4-391-gf0cc85ef`), glslang `11.1.0-882-ga91631b2`.
This identifies the local toolchain; it does not establish a known upstream bug.

Existing source-tree `volume_closesthit.spv`:

- SHA-256: `1b283b0c092a2c048bfc8512f764edbb80005d1383577a11bb16542d9aa24429`
- 1,023,604 bytes, SPIR-V 1.4, 322 functions, 1,084 static function-call sites.
- No cycles in its `OpFunctionCall` graph. This does not inspect `traceRayEXT`
  recursion across stages or internal recursion inside the compiler itself.
- `shadeVolumeHit`: 9,113 instructions, 1,227 function-local variable declarations,
  184 call sites.
- `evalMaterialProgramWithObjectRandom`: 4,000 instructions, 356 local declarations.
- `shadeFogMixture`: 1,260 instructions; `mfEvaluate`: 591 instructions.

These are static artifact counts, not executed instruction counts, GPU register
counts, simultaneously live variables, stack bytes or component timings.
The artifact's source and optimization-flag provenance are not established.
Reproduce with `python scripts/test/inspect_volume_spirv.py`.

## Ranked source suspects

1. **Large entry-point call tree and aggregate expansion.** `main` calls
   `shadeVolumeHit`, which combines SurfaceSDF shading, ordinary fog, mixture,
   random walk, material programs and caches. Source `VkVolumeInstance` includes
   transforms, ramps and several arrays; C++ asserts a 672-byte ABI. Numerous
   functions take it by value. This is a credible source of optimizer expansion,
   not proof of physical 672-byte GPU copies at each call. Old 256/576/624-byte
   comments are stale; do not use them to infer an ABI mismatch.
2. **Material-program fallback remains reachable from nested sampling.**
   `shadeFogMixture -> mfSum -> mfEvaluate -> evalMaterialProgram` and
   `mfDirect -> mfShadow -> mfEvaluate -> evalMaterialProgram` coexist.
   The prepared NanoVDB shadow branch avoids that evaluator, but the fallback
   at `volume_mixture.glsl:437` retains it. Runtime material/fast-path conditions
   do not remove the general branch from the compiler's input. The volume
   shader already reduces `MP_REGISTER_COUNT` to 12; claiming it uses the
   interpreter's default 32 would be incorrect.
3. **NanoVDB traversal is reused across many sampling/bound paths.** The artifact
   contains 30 call sites to `sampleIsoField`, 16 to `sampleDensityAcc`, and 13
   to `volumeNanoRay`. Exhaustive inlining plus subsequent aggregate/SSA passes
   deserves isolation before changing sampling math.

Moving these functions into textual includes alone leaves the same compilation
unit. Splitting hit groups/callables/compute stages changes execution contracts;
that is a later architecture choice requiring image and performance acceptance.

## Separate runtime finding

The cost matrix's 123 ms two-volume run does not prove optimizer failure causes
the runtime regression. In source, `generalMixture` branches into
`shadeFogMixture` and returns before the ordinary fog lighting path. That
ordinary path calls `volumeLightCacheFetch`; mixture `mfDirect` instead keeps
its own step-strided visibility and calls `mfShadow`. This is a concrete policy
difference consistent with near-zero persistent light-cache hits, not proof
that the cache rejected requests or that shadows explain all excess time.
An earlier shadow-zero ablation still left substantial cost. Keep compiler
diagnosis and overlap/adaptive/cache performance experiments separate.

## User-only phase isolation (no installed shader replacement)

### Runner correction after first user attempt

The first user run in `volume-compiler-e5ea682129d54aae868e2bbd84e4f98c`
returned null exit codes for both `--version` and the reference compile.
Version stdout was present, compiler error logs were empty, and a 1,023,604-byte
reference module existed. This is an inconclusive run caused by runner exit-code
collection, not evidence that `-O0` failed. Optimizer stages were never reached.
The runner now owns a `System.Diagnostics.Process` from Start through ExitCode
retrieval and drains stdout/stderr asynchronously. A null exit code after exit
is classified separately as `runner-error`.
`check_volume_diagnostic_runner.ps1` passed real short-lived PowerShell-child
checks for exit 0, exit 7, both output streams and timeout termination, without
invoking any shader tool. The corrected compiler experiment still needs the
user to rerun the command below.

The automated runner now executes the sequence below, validates successful
optimizer outputs, records source/include/tool hashes and per-phase exit codes,
and limits each child process to 600 seconds. It does not publish/deploy shaders.
Run from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/test/diagnose_volume_compiler.ps1
```

Return the printed `outcome.json` path. Adjacent stdout/stderr logs contain
compiler diagnostics. `-TimeoutSeconds 1200` permits a longer experiment;
a timeout is recorded separately from a compiler failure.
The agent ran only `-ValidateOnly`: input/tool existence and PowerShell parsing
passed, with no compiler/optimizer process or output directory created.
The detailed manual equivalent follows for inspection.

Run from repository root in an external PowerShell. These commands compile or
optimize only scratch outputs; the agent has **not** run them. Keep the current
working SPV. Use a fresh directory to prevent stale outputs masquerading as
success. Stop after a failed reference compile or validation.

```powershell
$diag = Join-Path (Get-Location) ('docs/dev/.captures/volume-compiler-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $diag | Out-Null
$bin = Join-Path $env:VULKAN_SDK 'Bin'
$src = (Resolve-Path 'RayTrophiStudio/source/shaders/volume_closesthit.rchit').Path
& "$bin/glslc.exe" --version 2>&1 | Out-File "$diag/versions.txt"
Get-FileHash -Algorithm SHA256 $src | Out-File "$diag/source-hash.txt"

# A: frontend + ordinary SPIR-V generation, without requested optimization.
& "$bin/glslc.exe" $src --target-env=vulkan1.3 --target-spv=spv1.4 -O0 -o "$diag/reference.spv" 2>&1 | Out-File "$diag/reference.log"
$referenceExit = $LASTEXITCODE
if ($referenceExit -ne 0) { throw "Reference compile failed: $referenceExit; $diag" }
& "$bin/spirv-val.exe" --target-env vulkan1.3 --scalar-block-layout "$diag/reference.spv" 2>&1 | Out-File "$diag/validation.log"
if ($LASTEXITCODE -ne 0) { throw "Reference validation failed; $diag" }

# B: exhaustive inlining only, from the same valid reference.
& "$bin/spirv-opt.exe" --target-env=vulkan1.3 --scalar-block-layout --inline-entry-points-exhaustive "$diag/reference.spv" -o "$diag/inline.spv" 2>&1 | Out-File "$diag/inline.log"
$inlineExit = $LASTEXITCODE

# C: full standalone optimizer chain, again from the original reference.
& "$bin/spirv-opt.exe" --target-env=vulkan1.3 --scalar-block-layout -O "$diag/reference.spv" -o "$diag/optimized.spv" 2>&1 | Out-File "$diag/optimized.log"
$optimizedExit = $LASTEXITCODE
"reference=$referenceExit inline=$inlineExit optimized=$optimizedExit" | Out-File "$diag/outcome.txt"
Write-Output $diag
Get-Content "$diag/outcome.txt"
```

Interpretation:

- A fails: the current source/toolchain also fails outside requested `-O`;
  do not continue attributing the current crash solely to optimization.
- A passes, B crashes: exhaustive inlining is independently sufficient to
  reproduce a failure on this module/toolchain. Next reduce the entry call tree
  or aggregate parameters in an isolated diagnostic variant.
- B passes, C crashes: target later passes or their interactions. A follow-up
  `spirv-opt --print-all -O` capture can identify the last emitted pass boundary;
  output can be very large and a crash boundary still needs a minimal reproducer.
- B and C pass: standalone SPIRV-Tools does not reproduce the integrated glslc
  failure. Investigate integrated pass setup/frontend output/tool differences;
  do not assume identical optimizer input or behavior.

If an experiment hangs, interrupt that process and record it as interrupted,
not as a crash or success. Do not install any of these scratch modules for a
GPU timing comparison yet. The next code change should follow this phase
result; no sampling-quality reduction or broad architecture rewrite is justified
by static counts alone.
