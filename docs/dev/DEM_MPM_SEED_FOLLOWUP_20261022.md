# MPM / DEM seed follow-up, 2026-10-22

Live external named-pipe test against the user's running build. No build or
application launch by Codex. Full measurements and prior scene settings:
[dem_mpm_seed_followup_20261022.json](dem_mpm_seed_followup_20261022.json).

Original Physics Domain 1: 49,600 DEM grains at paused frame 94. Forced autosave
before editing; this preserves project configuration, not a live solver checkpoint.
The original domain remains configured but disabled; its live carriers were cleared
by authoring invalidation/reset. No source or domain was deleted.

Test domain: MPM Sand + DEM Stones R2, bounds (-.6,-2.5,-.6) to (.6,-.9,.6),
requested voxel .06 m. Test MPM Sand R2 derives from Sand, transport=mpm.
MPM bed: (-.45,-2.44,-.45) to (.45,-2.14,.45), PPC 8, 12,288 carriers.
DEM Stone Drop R2 emits Gravel: (0,-1.55,0), radius .18 m source, velocity
(0,-.5,0), 120/s, total 12. Physical DEM radius .045 m; max substeps 2048,
contact resolution 24. Dry test, settle-on-seed off, no fluid carriers.

144 steps at dt=1/120 s (1.2 s), sampled each 12 steps:

- All steps completed; no held step; owners remained 12,288 MPM + 12 DEM.
- Shared geometry/contact clock enabled; sampled frame had 36 transport steps
  and 6 continuum grid refreshes. P2G/G2P reported GPU execution.
- First sampled nonzero contact at step 36. Final frame reported 10,035 contact
  events accumulated over its substeps, not unique pairs or a whole-run total.
- Maximum sampled contact momentum residual 2.552e-7 N s; final 4.608e-8 N s.
- Reported working set 42,844,384 bytes (~40.9 MiB), not total application VRAM.
- All six recorded regression gates pass. No restitution/material values tuned.

This establishes birth ownership, continued shared transport, and reciprocal
contact impulse accounting. It does not establish correct penetration, long-term
support/repose, or dt/voxel convergence. Those require isolated controls and
convergence measurements; appearance alone is insufficient.

Left paused at frame 0 after reset, MPM bed reseeded and DEM source ready.
Play starts the drop. The original domain is disabled. No explicit user project
save was forced after test edits; use normal Save to keep this fixture.
