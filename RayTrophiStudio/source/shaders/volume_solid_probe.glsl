#ifndef VOLUME_SOLID_PROBE_GLSL
#define VOLUME_SOLID_PROBE_GLSL

// Accept candidates without terminating traversal: Vulkan can then prune the
// ray interval to the committed hit and keep searching for a closer surface.
// .w remains the mode sentinel; .z is the hit flag, independent of miss output.
bool volumeSolidProbeCandidate(inout vec4 state, float distance) {
    if (floatBitsToUint(state.w) != 0xC17D5EEDu) return false;
    state.x = state.z > 0.5 ? min(state.x, distance) : distance;
    state.z = 1.0;
    return true;
}
#endif
