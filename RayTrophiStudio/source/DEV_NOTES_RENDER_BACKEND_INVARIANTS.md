# Render Backend Invariants (Do Not Break)

This file documents critical runtime rules that fixed hard Vulkan/undo issues.

## 1) Transform Undo/Redo in Vulkan

- Path: `SceneCommand::TransformCommand::applyState()`
- Rule: For Vulkan TLAS mode, use `updateObjectTransform(nodeName, matrix)` and then `resetAccumulation()`.
- Do not replace this with full rebuild logic for transform-only updates.
- Reason: Full rebuild flow here caused frozen viewport and ghost/stale frame artifacts until backend switch.

## 2) Deferred GPU Refit Block

- Path: `Main.cpp`, `if (g_gpu_refit_pending)`.
- Rule: Keep this as transform-only sync (`updateInstanceTransforms`, camera/lights update, reset accumulation).
- Do not call `rebuildAccelerationStructure()` in this block for Vulkan.
- Reason: Vulkan `rebuildAccelerationStructure()` is a scene reset path, not lightweight refit.

## 3) Interactive Render Gate

- Path: `Main.cpp`, `if (start_render)`.
- Rule: Block interactive render only when `rendering_in_progress && ui_ctx.is_animation_mode`.
- Do not block on `rendering_in_progress` alone.
- Reason: Non-animation paths may temporarily keep this flag true and stall interactive Vulkan render.

## 4) For New Features (Hair/Terrain/Water/Skin/VDB)

- If update is transform-only, prefer TLAS/instance update paths.
- If topology changes (add/remove geometry, BLAS content changes), use full rebuild paths.
- After backend-side scene mutations, always call `resetAccumulation()`.
- During backend switch/load/import, avoid direct GPU mutation from UI thread; prefer deferred flags.
