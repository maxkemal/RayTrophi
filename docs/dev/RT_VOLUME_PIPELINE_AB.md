# Volume shader attachment A/B on a volume-free scene

2026-10-30. User C++ build and the first live volume-free A/B are complete.
No build, shader compilation or application launch was performed by the agent.

## First live result

External IPC verified the full pipeline contained both volume stages and the
omitted variant contained neither; all other installed stages matched. At
1680 x 945, fixed GPU samples 8 through 39, counters off, warm medians were:

| Pipeline | GPU median |
|---|---:|
| Full | 99.145424 ms |
| Volume hit group omitted | 98.818768 ms |

The 0.329% reduction is too small to support the proposed large attachment
penalty in this scene. It does not explain the reported roughly 40% volume-free
regression, nor compare optimized SPIR-V against `-O0`.

An initial omitted capture measured 95.89 ms but was rejected for comparison
because camera.target differed after reopening. The final capture temporarily
restored the baseline target, matched the recorded camera/settings/frame/project
context and dimensions, then restored the user's current target successfully.
The final omitted capture peak was 103.526464 ms; baseline peak 109.623008 ms.
Timing/counter states were restored. No scene or material edits were made.
Reports are ignored JSON artifacts under `docs/dev/.captures/`:
`rt-volume-full-20261030.json` and `rt-volume-omitted-matched-20261030.json`.
Normal operation still requires restarting without `RT_VK_AB_NO_VOLUME=1`.

This experiment tests whether including the heavy volume hit group affects
volume-free camera RT dispatches. It does not establish that `-O0` alone causes
the reported regression, and does not replace multiple-volume profiling.

## Design

Developer-only process-start environment variable `RT_VK_AB_NO_VOLUME=1`
clears the volume closest-hit and intersection modules before shader-set hashing
and pipeline creation. All other shader vectors, compiler flags, descriptor
layouts and pipeline creation settings remain unchanged. The existing optional
hit-group logic builds the corresponding SBT and correct hair/sphere offsets.
The shader-set hash includes each stage's length and contents, so full and
omitted variants cannot be mistaken for an already-installed identical pipeline.
Existing cache-stamp validation also distinguishes the variants.

With the variable absent or anything except exactly `1`, normal behavior is
preserved. It is read once per process: restarting Studio is required to change
variants. The startup log identifies omission. Existing Python/IPC
`render.rt_pipeline_status` exposes the actual installed stage list; no new
authoring operation or UI control is added.

An omitted-volume viewport refuses RT dispatch when either the published volume
count or TLAS volume-slot count is nonzero. Use a saved scene with no volume,
gas or SurfaceSDF objects, including hidden ones. Do not use the existing gas
scene for this experiment. Ordinary meshes, materials and lights are allowed.
The separate cloud callable and volume code embedded in other stages remain
unchanged, so a negative result does not clear those other paths of suspicion.

## User sequence

Build C++ once. No shader rebuild is needed for this A/B control. Use the same
new executable, saved project and SPV files for both runs; do not rebuild or
edit the scene between captures. Preserve any current work before closing
Studio. Commands below run from the repository root; the Release executable
path exists in this workspace, but use the exact binary you just built.

First run, normal pipeline, launch from this PowerShell so the variable is inherited:

```powershell
$env:RT_VK_AB_NO_VOLUME = '0'
& .\x64\Release\RayTrophiStudio.exe
```

Open the saved volume-free scene, select Vulkan Rendered, Pause, allow at least
40 progressive samples, and wait for the pipeline to be ready. Keep the camera,
viewport size and all render settings fixed. In another external terminal:

```powershell
New-Item -ItemType Directory -Force docs/dev/.captures | Out-Null
python scripts/test/rt_probe_volume_pipeline_ab.py --variant with-volume --output docs/dev/.captures/rt-volume-full.json
```

Close Studio completely. In the launching PowerShell, change the variable and
launch the same executable, then reopen the exact same saved scene/settings:

```powershell
$env:RT_VK_AB_NO_VOLUME = '1'
& .\x64\Release\RayTrophiStudio.exe
```

Wait for Vulkan Rendered / ready / Pause, then in the external terminal:

```powershell
python scripts/test/rt_probe_volume_pipeline_ab.py --variant without-volume --output docs/dev/.captures/rt-volume-omitted.json --compare docs/dev/.captures/rt-volume-full.json
```

The collector checks actual installed stages and captures fixed sample indices
8 through 39 with volume counters disabled. It records warm median/mean and
the observed capture peak separately, and restores timing/counter settings.
It rejects changed frame/camera/settings, resolution or other installed stages.
Context identity does not hash scene geometry: keeping the saved scene and
shader/binary files identical is still part of the experiment. If the difference
is small or clocks/other applications were active, repeat the full run (A-B-A).

After the test, close Studio and remove the variable from the launching shell:

```powershell
Remove-Item Env:RT_VK_AB_NO_VOLUME -ErrorAction SilentlyContinue
```

No persistent application setting or project state is changed by this mode.
Compare GPU dispatch timings, not FPS or pipeline creation duration. A reduction
supports an attachment/pipeline resource effect; it does not identify a specific
register spill or optimizer pass. Similar times point toward other changes.

## Non-build verification

`python scripts/test/check_volume_pipeline_ab.py`: six tests cover fixed warm
sample selection, instrumentation restoration on success/failure, wrong-stage
and volume-scene rejection, comparison identity, stage omission before hashing,
and the pre-dispatch scene guard. C++/GPU execution remains unverified.
