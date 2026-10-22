#ifndef RT_GRID_SCALAR_TRACE
#define RT_GRID_SCALAR_TRACE

// Coordinates are MAC grid coordinates; h is already folded into signedStep.
vec3 gridScalarVelocity(vec3 p) {
    return vec3(sample_vx(p.x, p.y - 0.5, p.z - 0.5),
                sample_vy(p.x - 0.5, p.y, p.z - 0.5),
                sample_vz(p.x - 0.5, p.y - 0.5, p.z));
}

vec3 gridScalarTrace(vec3 p, float signedStep) {
    vec3 midpoint = p - gridScalarVelocity(p) * (0.5 * signedStep);
    return p - gridScalarVelocity(midpoint) * signedStep;
}

#endif
