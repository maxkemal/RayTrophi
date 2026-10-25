#ifndef VOLUME_OPTICAL_ACTIVITY_GLSL
#define VOLUME_OPTICAL_ACTIVITY_GLSL

// Metadata-only proof shared by intersection and overlap membership. Never
// disable a slot: the same domain may publish an independent SurfaceSDF or
// another participating field. Reject only this instance, on every ray type.
bool volumeOpticallyInert(VkVolumeInstance v) {
    bool knownMedium = v.source_type == 0 || v.source_type == 5 || v.source_type == 6;
    if (!knownMedium || v.volume_type == 3) return false;
    // A bound graph may synthesize density, extinction or emission. Treat even
    // an unresolved binding conservatively; the intersection stage has no
    // material-program descriptors. Also retain non-finite metadata.
    if (!(v._reserved[1] <= 0.5)) return false;
    // All current non-graph emission responses are zero at zero density.
    // Exact zero only: weak media must retain their authored optical depth.
    if (v.density_multiplier == 0.0) return true;
    // Do not assume emission is coupled to extinction. Preserve an emitting
    // medium even if the ordinary extinction coefficients are both zero.
    return v.scatter_coefficient == 0.0 && v.absorption_coefficient == 0.0 &&
        v.emission_mode == 0;
}

#endif
