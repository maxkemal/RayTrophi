/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          hair_intersect.cuh
 * Description:   GPU Hair Curve Intersection for OptiX
 *                Custom intersection for linear/bezier hair curves
 * =========================================================================
 */
#pragma once

#include <optix.h>
#include <optix_device.h>
#include "vec3_utils.cuh"

namespace HairGPU {

// ============================================================================
// Hair Curve Segment Data (uploaded to GPU)
// ============================================================================

struct HairSegment {
    float3 p0, p1;      // Endpoints
    float r0, r1;       // Radii at endpoints
    uint32_t strandID;
    uint16_t materialID;
    float v0, v1;       // Parametric position along strand
    float2 rootUV;      // UV on scalp
};

// ============================================================================
// Ray-Cylinder Intersection (Hair Approximation)
// ============================================================================

/**
 * @brief Intersect ray with tapered cylinder (hair segment)
 * 
 * Uses simplified cylinder intersection for performance.
 * For higher quality, use proper ribbon or curve intersection.
 * 
 * @param rayOrigin Ray origin
 * @param rayDir Ray direction (normalized)
 * @param p0 Cylinder start point
 * @param p1 Cylinder end point
 * @param r0 Radius at start
 * @param r1 Radius at end
 * @param tMin Minimum t
 * @param tMax Maximum t
 * @param outT Hit distance
 * @param outU Parameter along segment [0,1]
 * @param outV Parameter around cylinder [0,1]
 * @return true if hit
 */
__device__ __forceinline__ bool intersectHairCylinder(
    const float3& rayOrigin,
    const float3& rayDir,
    const float3& p0,
    const float3& p1,
    float r0,
    float r1,
    float tMin,
    float tMax,
    float& outT,
    float& outU,
    float& outV
) {
    // Axis of cylinder
    float3 axis = p1 - p0;
    float axisLen = length(axis);
    if (axisLen < 1e-6f) return false;
    
    float3 axisDir = axis / axisLen;
    
    // Transform to cylinder-local space
    float3 d = rayOrigin - p0;
    
    // Project ray onto plane perpendicular to axis
    float3 rayDirPerp = rayDir - dot(rayDir, axisDir) * axisDir;
    float3 dPerp = d - dot(d, axisDir) * axisDir;
    
    // Average radius for simplified intersection
    float avgRadius = (r0 + r1) * 0.5f;
    float avgRadius2 = avgRadius * avgRadius;
    
    // Quadratic coefficients for circle intersection
    float a = dot(rayDirPerp, rayDirPerp);
    float b = 2.0f * dot(rayDirPerp, dPerp);
    float c = dot(dPerp, dPerp) - avgRadius2;
    
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) return false;
    
    float sqrtDisc = sqrtf(discriminant);
    float invA = 1.0f / (2.0f * a);
    
    // Try both roots
    float t1 = (-b - sqrtDisc) * invA;
    float t2 = (-b + sqrtDisc) * invA;
    
    // Choose valid root
    float t = t1;
    if (t < tMin) t = t2;
    if (t < tMin || t > tMax) return false;
    
    // Check if hit is within cylinder length
    float3 hitPoint = rayOrigin + rayDir * t;
    float3 hitLocal = hitPoint - p0;
    float u = dot(hitLocal, axisDir) / axisLen;
    
    if (u < 0.0f || u > 1.0f) return false;
    
    // Check against tapered radius at u
    float radiusAtU = r0 * (1.0f - u) + r1 * u;
    float distToAxis = length(hitLocal - axisDir * (u * axisLen));
    
    if (distToAxis > radiusAtU * 1.1f) return false;  // Small tolerance
    
    outT = t;
    outU = u;
    
    // V parameter (angle around cylinder, for anisotropic shading)
    float3 perpDir = normalize(hitLocal - axisDir * (u * axisLen));
    outV = atan2f(perpDir.y, perpDir.x) / (2.0f * M_PIf) + 0.5f;
    
    return true;
}

// ============================================================================
// OptiX Intersection Program for Hair
// ============================================================================

#ifdef OPTIX_HAIR_INTERSECTION

extern "C" __global__ void __intersection__hair() {
    // Get primitive ID (segment index)
    const uint32_t primID = optixGetPrimitiveIndex();
    
    // Get segment data from SBT
    const HairSegment* segments = 
        reinterpret_cast<const HairSegment*>(optixGetSbtDataPointer());
    const HairSegment& seg = segments[primID];
    
    // Get ray in object space
    float3 rayOrigin = optixGetObjectRayOrigin();
    float3 rayDir = optixGetObjectRayDirection();
    float tMin = optixGetRayTmin();
    float tMax = optixGetRayTmax();
    
    float t, u, v;
    if (intersectHairCylinder(rayOrigin, rayDir, seg.p0, seg.p1,
                               seg.r0, seg.r1, tMin, tMax, t, u, v)) {
        // Report hit
        optixReportIntersection(
            t,                          // Hit distance
            0,                          // Hit kind
            __float_as_uint(u),         // Attribute 0: u along segment
            __float_as_uint(v),         // Attribute 1: v around cylinder
            seg.strandID,               // Attribute 2: strand ID
            seg.materialID,             // Attribute 3: material ID
            __float_as_uint(seg.v0 + u * (seg.v1 - seg.v0))  // Attribute 4: v along strand
        );
    }
}

// ============================================================================
// OptiX Closest Hit for Hair
// ============================================================================

extern "C" __global__ void __closesthit__hair() {
    // Get hit attributes
    float u = __uint_as_float(optixGetAttribute_0());
    float v = __uint_as_float(optixGetAttribute_1());
    uint32_t strandID = optixGetAttribute_2();
    uint16_t materialID = static_cast<uint16_t>(optixGetAttribute_3());
    float vStrand = __uint_as_float(optixGetAttribute_4());
    
    // Get segment data
    const uint32_t primID = optixGetPrimitiveIndex();
    const HairSegment* segments = 
        reinterpret_cast<const HairSegment*>(optixGetSbtDataPointer());
    const HairSegment& seg = segments[primID];
    
    // Compute tangent (axis direction)
    float3 tangent = normalize(seg.p1 - seg.p0);
    
    // Compute hit position
    float3 hitPos = seg.p0 + (seg.p1 - seg.p0) * u;
    
    // Compute shading normal (perpendicular to tangent, facing ray)
    float3 rayDir = optixGetWorldRayDirection();
    float3 normal = normalize(rayDir - dot(rayDir, tangent) * tangent);
    normal = -normal;  // Face toward camera
    
    // Get payload and fill in hair-specific data
    OptixHitResult* payload = getPayload();
    
    payload->t = optixGetRayTmax();
    payload->position = hitPos;
    payload->normal = normal;
    payload->is_hair = true;
    payload->hair_tangent = tangent;
    payload->hair_v = vStrand;
    payload->hair_u = v;
    payload->uv = seg.rootUV;
    payload->material_id = materialID;
    payload->prim_id = primID;
}

// ============================================================================
// OptiX Any Hit for Hair (Alpha/Transparency)
// ============================================================================

extern "C" __global__ void __anyhit__hair_shadow() {
    // For hair shadows, we can optionally thin out shadows
    // based on density to avoid overly dark areas
    
    // Get strand V (root to tip)
    float vStrand = __uint_as_float(optixGetAttribute_4());
    
    // Tips cast less shadow (thinner)
    float shadowOpacity = 1.0f - vStrand * 0.3f;
    
    // Stochastic shadow (optional - for soft hair shadows)
    // unsigned int seed = optixGetPrimitiveIndex() + optixGetPayload_0();
    // if (random_float(seed) > shadowOpacity) optixIgnoreIntersection();
}

#endif // OPTIX_HAIR_INTERSECTION

// ============================================================================
// Compute Hair Normal for Shading (CPU-compatible)
// ============================================================================

__device__ __host__ inline float3 computeHairShadingNormal(
    const float3& tangent,
    const float3& viewDir
) {
    // Normal perpendicular to tangent, facing viewer
    float3 normal = viewDir - dot(viewDir, tangent) * tangent;
    float len = length(normal);
    if (len > 1e-6f) {
        return normal / len;
    }
    // Fallback: any perpendicular direction
    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    if (fabsf(dot(tangent, up)) > 0.99f) {
        up = make_float3(1.0f, 0.0f, 0.0f);
    }
    return normalize(cross(tangent, up));
}

} // namespace HairGPU
