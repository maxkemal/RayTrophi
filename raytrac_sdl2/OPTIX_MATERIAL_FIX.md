# OptiX Material Mismatch Fix - Critical Design Notes

## Problem Summary
When objects are deleted from the scene, OptiX material indexing becomes misaligned, causing:
- Textures/materials shifting to wrong surfaces
- Visual artifacts and incorrect rendering
- GPU crashes in extreme cases

## Root Cause
The issue occurs because:
1. **Scene Load**: `buildFromData()` creates material index buffer matching triangle count
   ```
   Triangles: 4 (tri0, tri1, tri2, tri3)
   mat_indices: [0, 1, 1, 2]  // 4 elements
   ```

2. **Object Deletion**: Triangle count decreases but material index buffer stays same size
   ```
   Triangles: 3 (tri0, tri2, tri3) - tri1 deleted
   mat_indices: [0, 1, 1, 2]  // STILL 4 elements! ❌
   
   Result:
   - Triangle 0 uses mat_indices[0] = 0 ✅
   - Triangle 1 uses mat_indices[1] = 1 ✅
   - Triangle 2 uses mat_indices[2] = 1 ❌ (should be 2!)
   ```

## Solution: Full Rebuild on Deletion
**File**: `scene_ui.cpp`, deletion handler (~line 2037)

```cpp
if (new_end != objs.end()) {
    objs.erase(new_end, objs.end());
    deleted = true;
    
    // CRITICAL: Material index buffer MUST be regenerated
    // When triangles are deleted, the index buffer shrinks
    // If we don't rebuild, index buffer size != triangle count
    // This causes material shifting and rendering artifacts
    if (ctx.optix_gpu_ptr) {
        ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
    }
}
```

## Why Not Use updateGeometry()?
`updateGeometry()` only updates:
- ✅ Vertex positions
- ✅ Normal vectors
- ✅ OptiX BVH (refit/rebuild)

But it does NOT update:
- ❌ Material index buffer (`d_material_indices`)
- ❌ SBT records count
- ❌ Material-to-triangle mapping

## Current Implementation Details

### Renderer::rebuildOptiXGeometry()
**File**: `Renderer.cpp` (~line 2330)

**What it does:**
1. Converts current Hittable objects to Triangle list
2. Calls `assimpLoader.convertTrianglesToOptixData()` to regenerate:
   - Vertex/normal/UV buffers
   - **Material index buffer** (critical!)
   - Texture references
3. Calls `optix_gpu_ptr->buildFromData()` to rebuild:
   - OptiX BVH
   - SBT records
   - Material bindings

**Performance**: ~200-500ms for medium scenes (acceptable for infrequent deletions)

## When to Use Each Method

### Use `buildFromData()` (via rebuildOptiXGeometry):
- ✅ Object deletion
- ✅ Object addition (if implemented)
- ✅ Material changes (if implemented)
- ✅ Scene topology changes

### Use `updateGeometry()`:
- ✅ Object transformation (position/rotation/scale)
- ✅ Animation playback
- ✅ Any change that only affects vertex positions

## Future Considerations

### Optimization Opportunities:
1. **Lazy Rebuild**: Set flag on deletion, rebuild before next render
   - Batches multiple deletions into single rebuild
   - Trades immediate feedback for performance
   - **Current status**: Removed due to user feedback (immediate visual update preferred)

2. **Partial Update**: Only regenerate affected material indices
   - More complex implementation
   - Risk of edge cases
   - Not recommended unless profiling shows rebuild is bottleneck

3. **Material ID Remapping**: Keep SBT, remap IDs after deletion
   - Very complex
   - High risk of bugs
   - Not worth the effort for current use case

### DO NOT:
- ❌ Use `updateGeometry()` for deletion (causes material mismatch)
- ❌ Skip material index regeneration (causes rendering artifacts)
- ❌ Modify SBT without rebuilding material indices
- ❌ Change triangle count without updating material index buffer

## Testing Checklist
When modifying this code, verify:
- [ ] Delete single object → materials stay correct
- [ ] Delete multiple objects → no material shifting
- [ ] Delete object with different material → no crashes
- [ ] Render after deletion → correct textures on all surfaces
- [ ] Multiple delete operations → no accumulating errors

## Related Files
- `scene_ui.cpp`: Deletion handler (line ~2037)
- `Renderer.cpp`: `rebuildOptiXGeometry()` (line ~2330)
- `OptixWrapper.cpp`: `buildFromData()` (line ~444), `updateGeometry()` (line ~1240)
- `Renderer.h`: `rebuildOptiXGeometry()` declaration (line ~126)

## Last Updated
2025-12-18 - Initial implementation with full rebuild on deletion

---
**IMPORTANT**: This is a STABLE, WORKING solution. Any changes must maintain material index synchronization!
