# RT volume empty rejection — 2026-10-25

Implements exact early rejection following
[the live diagnosis](RT_ZERO_OPTICS_GAS_DIAGNOSIS_20261025.md).
Shader sources changed; no shader/application build or live acceptance run was
performed. The open application still runs its previously compiled shaders.

## Changes

- `volume_optical_activity.glsl` provides one metadata-only inactivity proof for
  intersection and overlapping-medium membership. Known gas/fog sources are
  rejected if density multiplier is exactly zero and no graph is bound, or if
  both extinction coefficients are exactly zero with emission off and no graph.
  No epsilon removes weak smoke. Unknown sources, clouds, SurfaceSDF and graph
  bindings retain the existing path. Existing non-graph emission responses are
  density-gated, including thermal/ramp response at zero density.
- `volume_intersection.rint` rejects only the inert instance before reporting
  an intersection. The original ray traversal then finds other gas, a triangle
  or SurfaceSDF normally. This avoids entering the gas closest-hit stage and its
  nested solid probe on camera, reflected and transmitted rays alike.
- `volume_overlap_selection.glsl` uses the same proof, so a rejected instance
  cannot return as a selected foreground gas or companion inside a live medium.
  General-mixture membership/evaluation also excludes it before field reads.
- Optical density samplers return zero before NanoVDB/dense field reads when
  their multiplier is zero. This is only an input-density shortcut: graph
  evaluation is retained and can still synthesize nonzero density. Raw SDF reads
  explicitly bypass optical controls and do not take this shortcut.
- NanoVDB constant tiles are skipped using their actual value after optical
  remap/cutoff, rather than their active bit. Active-zero tiles can be skipped;
  inactive nonzero backgrounds or remaps that create density cannot. Both leaf
  and tile proofs stop before a trilinear footprint touches a neighbor.
  Non-finite bounds, material noise, graphs and geometry retain normal stepping.

The shared SSBO ABI, stable volume indices, ray masks, nearest-solid probe,
gas-to-SDF handoff, quadrature lattice and authored bounce/step budgets are
unchanged. Empty-cell advances remain clamped by the existing solid/SDF-clipped
interval. Overlapping fields still require a joint emptiness proof; no primary
field alone authorizes skipping a nonempty companion.

This change uses exact emptiness proofs. It does not introduce a lower-quality
secondary-ray budget or replace the existing surface-distance search architecture.
No new authoring control or separate UI/API/IPC behavior is introduced.

## Executed non-build checks

```text
python scripts/test/check_volume_empty_rejection.py        7 passed
python scripts/test/check_rt_gas_optical_contracts.py      22 passed
python scripts/test/check_volume_material_contracts.py    10 passed
python scripts/test/check_matter_phase_contracts.py        PASS
```

The new checks interpret the production metadata predicate without a shader
compiler, test all traversal permutations of inert gas/active gas/SDF/triangle,
and verify trilinear support, negative tile coordinates, weak media and graph/raw
SDF guards. These are semantic/reference and source checks, not Vulkan execution.

## User build and acceptance checklist

1. Run the normal shader build so BOTH `volume_intersection.spv` and
   `volume_closesthit.spv` are regenerated with the new includes. Keep the current
   application binary when using the existing shader reload workflow, or perform
   your normal application build/restart. No new C++ module is required.
2. With the original Nuclear Gas scene paused at frame 129 and Plane_1 visible,
   retain density/scattering/absorption zero, camera and resolution fixed. Run
   from an external terminal (not the embedded script workspace):

   ```powershell
   python scripts/test/rt_probe_gas_zero_optics_ipc.py --domain 'Nuclear Gas' --output docs/dev/rt_gas_zero_optics_after_20261025.json
   python scripts/test/rt_probe_gas_bounce_cost_ipc.py --domain 'Nuclear Gas' --counts-only --output docs/dev/rt_gas_zero_optics_counts_after_20261025.json
   ```

   Scripts restore settings and disable their instrumentation. No simulation
   stepping occurs. Compare equal samples 0..63 to the saved before report;
   expect this domain's camera density reads/solid probes to disappear. Other
   scene volumes can legitimately retain their own counts. Do not expect the
   plane's independent surface/lighting cost to disappear.
3. Check the plane and solid geometry inside, in front of and behind the domain,
   with the camera outside and inside it. Check reflected/transmitted views and
   rays grazing the domain entry. No black box, missing plane or free-pass loop.
4. Restore positive gas density: check a visible gas layer before an opaque
   solid and before/after an overlapping liquid/solid SurfaceSDF. Set the SDF's
   fog density multiplier to zero: its geometric surface must remain visible.
5. Overlap inert gas with nonzero fog/gas and SurfaceSDF. Swap publication order;
   the nonzero medium and nearest surface must still render in the same order.
6. Check a Volume Graph that writes nonzero density with input multiplier zero,
   and emitting media with positive density but zero scattering/absorption.
   Their graph/emission paths must remain selected; this patch does not change
   their existing emission integration equations.
7. Check sparse cells next to a dense leaf/tile, active-zero tiles, inactive
   nonzero VDB backgrounds, negative coordinates, remap-low below zero and
   material noise. Fine edges must not disappear or shift with shadow budget.

No post-build speedup or image equivalence is claimed until these live checks
have been performed.
