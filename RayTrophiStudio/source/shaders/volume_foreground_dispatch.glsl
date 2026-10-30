#ifndef VOLUME_FOREGROUND_DISPATCH_GLSL
#define VOLUME_FOREGROUND_DISPATCH_GLSL

// Return-only marker: negative/NaN float encoding cannot alias the positive
// distance encoded by plSolidHandoffWord. The next trace restores the real mask.
const uint VOLUME_FOREGROUND_GAS_DISPATCH = 0xFFFFFFFEu;

#ifdef VOLUME_FOREGROUND_DISPATCH_RAYGEN
void volumeFinishForegroundDispatch(vec3 origin, float nearT, vec3 direction, float farT) {
    if (payload.volumeTraversalMask != VOLUME_FOREGROUND_GAS_DISPATCH) {
        return;
    }
    // The SDF stage requested this before RNG, shading or cloud compositing.
    // Redispatch within the current raygen pass: no bounce/pass-budget charge,
    // no origin offset, no change to seed or the original ray interval.
    payload.volumeTraversalMask = 0x02u;
    traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT, 0x02u, 0, 0, 0,
        origin, nearT, direction, farT, 0);
}
#endif

#endif
