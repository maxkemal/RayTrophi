#ifndef RT_ACOUSTIC_SURFACE_DEBUG_GLSL
#define RT_ACOUSTIC_SURFACE_DEBUG_GLSL

// Selected-source direct-field proxy. This is deliberately separate from the
// material-transmission solver: no optical transmission is reused as acoustics.
// Source layout matches Audio/AcousticRenderDebug.h, scalar stride 4.
vec3 acousticSurfaceDebug(vec3 point) {
    if (cam.acousticSource[7] < 0.5) {
        return vec3(0.12);
    }
    vec3 source = vec3(cam.acousticSource[0], cam.acousticSource[1], cam.acousticSource[2]);
    vec3 toSource = source - point;
    float distance = length(toSource);
    float minimum = max(cam.acousticSource[4], 0.001);
    float maximum = max(cam.acousticSource[5], minimum);
    float attenuation = minimum / (minimum + cam.acousticSource[6] *
        (clamp(distance, minimum, maximum) - minimum));
    float amplitude = cam.acousticSource[3] * attenuation;
    vec3 direction = vec3(cam.acousticSource[8], cam.acousticSource[9], cam.acousticSource[10]);
    float directionLength = length(direction);
    float inner = cam.acousticSource[11];
    float outer = cam.acousticSource[12];
    if (distance > 1e-6 && directionLength > 1e-6 && outer < PI) {
        float angle = acos(clamp(dot(direction / directionLength, -toSource / distance),
            -1.0, 1.0));
        float blend = outer > inner ? clamp((angle - inner) / (outer - inner), 0.0, 1.0) :
            (angle > inner ? 1.0 : 0.0);
        amplitude *= mix(1.0, cam.acousticSource[13], blend);
    }
    float energy = amplitude * amplitude * max(cam.debugExposure, 0.0);
    if (energy <= 0.0 || isnan(energy) || isinf(energy)) {
        return vec3(0.0);
    }
    float epsilon = max(0.0001, distance * 1e-6);
    if (distance > 4.0 * epsilon) {
        shadowPayload = vec4(1.0, 1.0, 1.0, 0.0);
        // Force opaque: skip optical alpha/transmission any-hit logic. Use the
        // renderer's authored-solid visibility bit, excluding gas/fog (0x02),
        // transient splats (0x04) and SurfaceSDF (0x08). Ordinary procedural
        // solids remain eligible. No optional primitive-culling feature needed.
        uint flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT |
            gl_RayFlagsSkipClosestHitShaderEXT;
        traceRayEXT(topLevelAS, flags, 0x01, 0, 1, 1,
            point + toSource / distance * epsilon, epsilon,
            toSource / distance, distance - 3.0 * epsilon, 1);
        if (shadowPayload.w < 0.5) {
            return vec3(0.0);
        }
    }
    float decibels = 10.0 * log2(max(energy, 1e-6)) / log2(10.0);
    return dvViridis(clamp((decibels + 60.0) / 60.0, 0.0, 1.0));
}
#endif
