# Compiled adaptive SDF inspection — 2026-10-10

External IPC inspection of the already open application. Paused frame 22, 9166 particles, 128 cubed surface, voxel 0.0390625, refinement 1, smoothing 2, anisotropy disabled. No solver settings, camera or timeline were changed. Capture was enabled temporarily and restored.

Last published surface build 14.0447 ms; smoothing 1.2671 ms; gather 1.5524 ms; scheduled cells 79872. These are a single published build, not a median benchmark. Pre-capture viewport frame telemetry 1.7901 ms is a paused-frame measurement, not live simulation throughput. Different frame and camera from the earlier frame-60 image prohibit direct before/after comparisons.

Image sdf_adaptive_compiled_frame22.jpg shows a continuous falling column with visible small contour protrusions. The bulk surface remains lumpy; smooth-body acceptance is still open. User reports improved detail and no disappearing small parts during playback. Temporal disappearance has not been independently verified by this single-frame capture. Cache valid=false, so no cached temporal seek was performed to avoid unintentional resimulation.

Material-specific reconstruction and analytic sub-voxel drops remain pending. No builds or application launch performed by Codex.
