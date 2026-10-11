# Live gas RT scatter parity

Reported behavior: uncached gas produces bright noisy pixels in RT when scattering
is positive. Replaying cached frames or setting scattering to zero removes them.

The Vulkan RT volume table previously published a live-only emissive block list.
`volume_closesthit.rchit` selects one random block per ray, samples a random point
inside it, and multiplies its emitted intensity by the number of stored emitters.
That estimate contributes only inside the positive-scattering branch. Cached
volumes do not publish this list. This is a concrete lighting-path difference
consistent with the report; runtime attribution has not been verified.

The table now publishes zero emissive-list address and capacity. Both paths use
the existing scene/sky scattering and direct volume emission. Density, temperature,
flame fields and majorant skipping are unchanged. This deliberately removes the
additional live-only fire-to-smoke lighting estimate; it does not implement a
converged replacement or guarantee identical cached/live flame fields.

## User build and verification

1. Build the project normally. This change is C++ volume-table publication; no
   shader source was changed for this fix.
2. In the same campfire scene, clear its simulation cache through the normal UI
   and play in Vulkan RT with a positive scatter coefficient. Check several
   frames for the reported bright noisy gas patches.
3. Replay the resulting cached frames with the same camera, lighting and material
   settings. Compare the same frame, especially the surrounding smoke brightness.
4. Repeat with scattering zero. The flame's direct emission should remain visible.
5. Check Solid and RayFusion for regressions. Their volume shaders do not consume
   the emissive block list.

Static inspection confirms the only assignments of these Vulkan table fields
publish zero and the NEE shader requires a nonzero list address. No build, shader
compilation, application launch or live IPC verification was performed by Codex.
