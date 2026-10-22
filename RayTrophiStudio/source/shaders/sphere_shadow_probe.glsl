// Shared shadow any-hit is also attached to the procedural sphere hit group.
// Dispatch before triangle instance/vertex reads: sphere primitive IDs index
// binding 18, not a TriangleMesh face stream.
#include "foam_sphere.glsl"

void sphereShadowProbe() {
    if ((gl_IncomingRayFlagsEXT & gl_RayFlagsSkipClosestHitShaderEXT) == 0u) {
        return; // camera/bounce ray: accept; sphere closest-hit owns its BSDF
    }
    if (volumeSolidProbeCandidate(shadowPayload, gl_HitTEXT)) {
        return; // accept, keep traversal searching for the nearest solid
    }
    if (floatBitsToUint(shadowPayload.w) == 0x555B0DEDu) {
        if (shadowPayload.z > 0.5 && gl_HitTEXT >= shadowPayload.x) {
            ignoreIntersectionEXT;
        }
        vec3 point = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        vec3 normal = normalize(point - foamSpheres[gl_PrimitiveID].centerRadius.xyz);
        normal /= max(abs(normal.x) + abs(normal.y) + abs(normal.z), 1e-8);
        vec2 oct = normal.xy;
        if (normal.z < 0.0) {
            oct = (vec2(1.0) - abs(oct.yx)) *
                mix(vec2(-1.0), vec2(1.0), greaterThanEqual(oct, vec2(0.0)));
        }
        shadowPayload.x = gl_HitTEXT;
        shadowPayload.y = uintBitsToFloat(packSnorm2x16(oct));
        shadowPayload.z = 1.0;
        ignoreIntersectionEXT;
    }
    // Bevel probes did not gather sphere normals before this wiring change.
    // Preserve that behaviour, and never reinterpret the packed probe as tint.
    if ((floatBitsToUint(shadowPayload.w) & 0x80000000u) != 0u) {
        ignoreIntersectionEXT;
    }
    // Previously a sphere accepted a shadow intersection without any-hit, so
    // it was opaque to direct-light visibility regardless of its surface BSDF.
    shadowPayload.w = 0.0;
    terminateRayEXT;
}
