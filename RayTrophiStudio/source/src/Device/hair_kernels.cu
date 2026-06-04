/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          hair_kernels.cu
 * Description:   CUDA Kernels for Hair/Fur Rendering
 *                OptiX programs for hair intersection and shading
 * =========================================================================
 */

// OptiX 7.x+ headers
#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>

// Project headers
#include "vec3_utils.cuh"
#include "random_utils.cuh"
#include "hair_bsdf.cuh"
#include "hair_intersect.cuh"

// Enable OptiX programs
#define OPTIX_HAIR_INTERSECTION

// ============================================================================
// Launch Parameters (shared with host)
// ============================================================================

struct HairLaunchParams {
    // Hair geometry buffers
    float4* hairVertices;       // x,y,z,radius
    uint32_t* strandOffsets;    // Start index for each strand
    uint16_t* materialIDs;      // Per-strand material
    float2* rootUVs;            // Per-strand root UV
    uint32_t numStrands;
    uint32_t numVertices;
    
    // Hair material
    HairGPU::GpuHairMaterial material;
    
    // Output
    OptixTraversableHandle traversable;
};

extern "C" {
    __constant__ HairLaunchParams hairParams;
}

// ============================================================================
// OptiX Intersection Program for Hair Curves
// ============================================================================

extern "C" __global__ void __intersection__hair_curve() {
    // Get primitive index
    const uint32_t primIdx = optixGetPrimitiveIndex();
    
    // Get ray parameters
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDir = optixGetWorldRayDirection();
    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();
    
    // Get strand and segment info
    // primIdx encodes: strandID * maxSegmentsPerStrand + segmentIndex
    // Or use a lookup table
    
    // For now, assume linear indexing into segments
    uint32_t strandID = 0;
    uint32_t segmentStart = 0;
    
    // Find which strand this segment belongs to
    for (uint32_t s = 0; s < hairParams.numStrands; ++s) {
        uint32_t nextOffset = hairParams.strandOffsets[s + 1];
        if (primIdx < nextOffset - 1) {
            strandID = s;
            segmentStart = hairParams.strandOffsets[s];
            break;
        }
    }
    
    uint32_t localSegment = primIdx - segmentStart;
    uint32_t v0Idx = segmentStart + localSegment;
    uint32_t v1Idx = v0Idx + 1;
    
    // Get segment endpoints
    float4 vert0 = hairParams.hairVertices[v0Idx];
    float4 vert1 = hairParams.hairVertices[v1Idx];
    
    float3 p0 = make_float3(vert0.x, vert0.y, vert0.z);
    float3 p1 = make_float3(vert1.x, vert1.y, vert1.z);
    float r0 = vert0.w;
    float r1 = vert1.w;
    
    // Perform intersection
    float t, u, v;
    bool hit = HairGPU::intersectHairCylinder(
        rayOrigin, rayDir, p0, p1, r0, r1,
        tMin, tMax, t, u, v
    );
    
    if (hit) {
        // Calculate V coordinate along strand
        uint32_t strandStart = hairParams.strandOffsets[strandID];
        uint32_t strandEnd = hairParams.strandOffsets[strandID + 1];
        uint32_t strandPoints = strandEnd - strandStart;
        
        float vCoord = (float(localSegment) + u) / float(strandPoints - 1);
        
        // Report hit with attributes
        // Attributes: u (around), v (along strand), strandID
        uint32_t attr0 = __float_as_uint(u);
        uint32_t attr1 = __float_as_uint(vCoord);
        
        optixReportIntersection(t, 0, attr0, attr1, strandID);
    }
}

// ============================================================================
// OptiX Closest Hit Program for Hair
// ============================================================================

extern "C" __global__ void __closesthit__hair() {
    // Get hit info
    const float t = optixGetRayTmax();
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDir = optixGetWorldRayDirection();
    
    // Get attributes from intersection
    const float u = __uint_as_float(optixGetAttribute_0());
    const float v = __uint_as_float(optixGetAttribute_1());
    const uint32_t strandID = optixGetAttribute_2();
    
    // Calculate hit position
    float3 hitPos = rayOrigin + t * rayDir;
    
    // Get segment for tangent calculation
    uint32_t primIdx = optixGetPrimitiveIndex();
    uint32_t strandStart = hairParams.strandOffsets[strandID];
    uint32_t localSegment = primIdx - strandStart;
    uint32_t v0Idx = strandStart + localSegment;
    
    float4 vert0 = hairParams.hairVertices[v0Idx];
    float4 vert1 = hairParams.hairVertices[v0Idx + 1];
    
    float3 p0 = make_float3(vert0.x, vert0.y, vert0.z);
    float3 p1 = make_float3(vert1.x, vert1.y, vert1.z);
    
    // Hair tangent
    float3 tangent = normalize(p1 - p0);
    
    // Normal perpendicular to tangent and view
    float3 wo = -rayDir;
    float3 normal = normalize(cross(cross(tangent, wo), tangent));
    

    
    // Evaluate hair BSDF
    // This would be called during shading, typically in raygen or anyhit
    
    // Pack hit info into payload
    // Payload structure depends on your raygen implementation
    unsigned int p0_payload = optixGetPayload_0();
    unsigned int p1_payload = optixGetPayload_1();
    
    // Store hit distance
    optixSetPayload_0(__float_as_uint(t));
    
    // Store hit type (1 = hair)
    optixSetPayload_1(1);
    
    // Additional payloads for normal, tangent, UV, etc.
    optixSetPayload_2(__float_as_uint(normal.x));
    optixSetPayload_3(__float_as_uint(normal.y));
    optixSetPayload_4(__float_as_uint(normal.z));
    optixSetPayload_5(__float_as_uint(tangent.x));
    optixSetPayload_6(__float_as_uint(tangent.y));
    optixSetPayload_7(__float_as_uint(tangent.z));
}

// ============================================================================
// OptiX Any Hit Program for Hair (shadow rays)
// ============================================================================

extern "C" __global__ void __anyhit__hair_shadow() {
    // Hair is mostly opaque, but tips can be transparent
    const float v = __uint_as_float(optixGetAttribute_1());
    
    // Tip transparency (thinner = more transparent)
    float opacity = 1.0f - v * 0.3f;
    
    // Stochastic transparency
    unsigned int seed = optixGetPayload_0();
    PCGState rng;
    rng.state = seed;
    float rnd = pcg_float(&rng);
    optixSetPayload_0(rng.state);
    
    if (rnd > opacity) {
        // Transparent - continue ray
        optixIgnoreIntersection();
    }
    // Else: opaque - terminate ray (default behavior)
}

// ============================================================================
// Hair BVH Build Kernel
// ============================================================================

extern "C" __global__ void buildHairAABBs(
    const float4* vertices,
    const uint32_t* strandOffsets,
    uint32_t numSegments,
    OptixAabb* aabbs
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSegments) return;
    
    // Find segment vertices
    // This is simplified - actual would need proper segment-to-vertex mapping
    float4 v0 = vertices[idx];
    float4 v1 = vertices[idx + 1];
    
    float3 p0 = make_float3(v0.x, v0.y, v0.z);
    float3 p1 = make_float3(v1.x, v1.y, v1.z);
    float r0 = v0.w;
    float r1 = v1.w;
    float maxR = fmaxf(r0, r1);
    
    // Compute AABB with radius padding
    OptixAabb aabb;
    aabb.minX = fminf(p0.x, p1.x) - maxR;
    aabb.minY = fminf(p0.y, p1.y) - maxR;
    aabb.minZ = fminf(p0.z, p1.z) - maxR;
    aabb.maxX = fmaxf(p0.x, p1.x) + maxR;
    aabb.maxY = fmaxf(p0.y, p1.y) + maxR;
    aabb.maxZ = fmaxf(p0.z, p1.z) + maxR;
    
    aabbs[idx] = aabb;
}
