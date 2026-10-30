#ifndef VOLUME_MIXTURE_CELL_CACHE_GLSL
#define VOLUME_MIXTURE_CELL_CACHE_GLSL

// Per-ray scalar bounds for the two prepared NanoVDB contributors. Cache the
// cell's directional bound, not sampled density/radiance or a chosen step.
// Straight camera rays keep direction and material fixed for the invocation.
// Walk segments bypass this cache because their direction changes per event.
ivec3 mfBoundCellA;
ivec3 mfBoundCellB;
vec2 mfBoundValueA;
vec2 mfBoundValueB;
bool mfBoundValidA;
bool mfBoundValidB;

void mfResetCellBounds() {
    mfBoundValidA = false;
    mfBoundValidB = false;
}

bool mfLoadCellBounds(uint index, ivec3 cell, out vec2 value) {
    value = vec2(0.0);
    if (mfWalking) return false;
    if (index == mfCacheIndexA && mfBoundValidA && all(equal(cell, mfBoundCellA))) {
        value = mfBoundValueA;
        return true;
    }
    if (index == mfCacheIndexB && mfBoundValidB && all(equal(cell, mfBoundCellB))) {
        value = mfBoundValueB;
        return true;
    }
    return false;
}

void mfStoreCellBounds(uint index, ivec3 cell, vec2 value) {
    if (mfWalking) return;
    if (index == mfCacheIndexA) {
        mfBoundCellA = cell;
        mfBoundValueA = value;
        mfBoundValidA = true;
    } else if (index == mfCacheIndexB) {
        mfBoundCellB = cell;
        mfBoundValueB = value;
        mfBoundValidB = true;
    }
}

#endif
