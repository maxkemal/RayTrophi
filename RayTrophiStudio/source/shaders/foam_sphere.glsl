// Foam / fluid-splat point spheres: one AABB BLAS for the whole set, one
// record per sphere (binding 18), indexed by gl_PrimitiveID. Shared by
// sphere_intersection.rint and closesthit.rchit.
#ifndef FOAM_SPHERE_GLSL
#define FOAM_SPHERE_GLSL

// Hit kind reported by sphere_intersection.rint. closesthit.rchit serves both
// the triangle hit group (kinds 0xFE / 0xFF) and the sphere group, and tells
// them apart by this value: it used to be compiled twice (-DSPHERE_HIT), a
// whole second closest-hit compile in the RT pipeline.
#define FOAM_SPHERE_HIT_KIND 1u

struct FoamSphereGPU {
    vec4 centerRadius;   // xyz = centre (world space), w = radius
    uint matId;          // foam material index (Spray / Foam / Bubble)
    float objectRandom;  // stable logical-sphere value for Object Info -> Random
    uint _p1;
    uint _p2;
};

layout(set = 0, binding = 18, scalar) readonly buffer FoamSphereSSBO {
    FoamSphereGPU foamSpheres[];
};

#endif // FOAM_SPHERE_GLSL
