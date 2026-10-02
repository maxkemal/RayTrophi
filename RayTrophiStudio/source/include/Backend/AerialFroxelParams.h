#pragma once
// Parameter block of shaders/atmosphere_aerial_froxel.comp (binding 4, std430)
// and the camera image plane both Vulkan backends build it from.
//
// ★★ ONE definition, for the same reason as AtmosphereLutParams.h: Vulkan RT
//   and RayFusion run the froxel shader on two different VkDevices, and "they
//   agree" can only mean "they were fed by the same function". The image plane
//   is here too because the RT raygen projects hit directions back onto the
//   plane its push constants describe -- if RayFusion built its froxel from a
//   different plane formula, the two would disagree off-centre and nowhere
//   else, which is exactly the kind of difference nobody reports.

#include <cmath>
#include <cstdint>
#include <cstring>
#include "World.h"
#include "Backend/IBackend.h"

struct AerialFroxelParamsGPU {
    float origin[4];      // xyz camera position, w = 1 when the Nishita LUTs light the medium
    float lowerLeft[4];   // xyz image-plane lower-left corner, w = 1 when air scattering is on
    float horizontal[4];  // xyz image-plane width vector, w = 1 when height fog is on
    float vertical[4];    // xyz image-plane height vector
    float fogA[4];        // density (1/m), height (m), falloff (1/m), distance (m)
    float fogB[4];        // albedo rgb, anisotropy g
};
static_assert(sizeof(AerialFroxelParamsGPU) == 96, "froxel params ABI changed (shader FroxelParams)");

// Perspective image plane: origin, lower-left corner, width and height vectors.
// The RT raygen's push constants ARE this plane at plane_distance =
// focus distance (DoF needs the plane there); the froxel only uses ratios on
// it, so any distance gives the same cells.
struct CameraImagePlane {
    Vec3 origin;
    Vec3 lowerLeft;
    Vec3 horizontal;
    Vec3 vertical;
};

inline CameraImagePlane makeCameraImagePlane(const Backend::CameraParams& cam, float aspect, float plane_distance) {
    const float fov = cam.fov > 1.0f ? cam.fov : 60.0f;
    const float h_half = std::tan(fov * 0.5f * 3.14159265358979f / 180.0f);
    const float viewport_height = 2.0f * h_half;
    const float viewport_width = aspect * viewport_height;

    Vec3 lookFrom = cam.origin;
    Vec3 lookAt = cam.lookAt;
    Vec3 vup = cam.up;
    // Safety fallback for an empty/default camera.
    if ((lookFrom - lookAt).length() < 0.0001f) {
        lookFrom = Vec3(0, 0, 5);
        lookAt = Vec3(0, 0, 0);
        vup = Vec3(0, 1, 0);
    }

    const Vec3 camW = (lookFrom - lookAt).normalize();
    const Vec3 camU = vup.cross(camW).normalize();
    const Vec3 camV = camW.cross(camU);

    CameraImagePlane p;
    p.origin = lookFrom;
    p.horizontal = camU * viewport_width * plane_distance;
    p.vertical = camV * viewport_height * plane_distance;
    p.lowerLeft = lookFrom - p.horizontal * 0.5f - p.vertical * 0.5f - camW * plane_distance;
    return p;
}

// Does this world need the froxel at all? Air haze is a toggle; height fog has
// its own switch and a density that can be zero.
inline bool aerialFroxelWanted(const WorldData& w) {
    return w.advanced.aerial_perspective != 0 ||
           (w.nishita.fog_enabled != 0 && w.nishita.fog_density > 0.0f);
}

inline AerialFroxelParamsGPU makeAerialFroxelParamsGPU(const WorldData& w,
                                                       const CameraImagePlane& plane,
                                                       bool lut_lit) {
    const NishitaSkyParams& n = w.nishita;
    AerialFroxelParamsGPU p{};
    p.origin[0] = plane.origin.x;       p.origin[1] = plane.origin.y;       p.origin[2] = plane.origin.z;
    p.origin[3] = lut_lit ? 1.0f : 0.0f;
    p.lowerLeft[0] = plane.lowerLeft.x; p.lowerLeft[1] = plane.lowerLeft.y; p.lowerLeft[2] = plane.lowerLeft.z;
    p.lowerLeft[3] = w.advanced.aerial_perspective ? 1.0f : 0.0f;
    p.horizontal[0] = plane.horizontal.x; p.horizontal[1] = plane.horizontal.y; p.horizontal[2] = plane.horizontal.z;
    p.horizontal[3] = n.fog_enabled ? 1.0f : 0.0f;
    p.vertical[0] = plane.vertical.x;   p.vertical[1] = plane.vertical.y;   p.vertical[2] = plane.vertical.z;
    p.vertical[3] = 0.0f;
    p.fogA[0] = n.fog_density;
    p.fogA[1] = n.fog_height;
    p.fogA[2] = n.fog_falloff;
    p.fogA[3] = n.fog_distance;
    p.fogB[0] = n.fog_albedo.x;
    p.fogB[1] = n.fog_albedo.y;
    p.fogB[2] = n.fog_albedo.z;
    p.fogB[3] = n.fog_anisotropy;
    return p;
}

// FNV-1a over raw bytes: "same bytes in, same froxel out" -- the froxel is
// rebuilt only when this changes (camera move, atmosphere or fog edit).
inline uint64_t hashAerialFroxelInputs(const void* a, size_t a_size, const void* b, size_t b_size) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* pa = static_cast<const uint8_t*>(a);
    for (size_t i = 0; i < a_size; ++i) { h ^= pa[i]; h *= 1099511628211ull; }
    const uint8_t* pb = static_cast<const uint8_t*>(b);
    for (size_t i = 0; i < b_size; ++i) { h ^= pb[i]; h *= 1099511628211ull; }
    return h ? h : 1ull;
}
