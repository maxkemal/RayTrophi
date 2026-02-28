/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          VolumetricRenderer.h
* =========================================================================
*/
#ifndef VOLUMETRIC_RENDERER_H
#define VOLUMETRIC_RENDERER_H

#include "Vec3.h"
#include "Ray.h"
#include "scene_data.h"
namespace Backend { class IBackend; }
#include "World.h" // Needed for WorldData

// Forward declarations
class Hittable;

/**
 * @brief Volumetric Rendering Subsystem
 * 
 * Handles all volumetric calculations including:
 * - Volumetric God Rays (CPU)
 * - Volumetric Shadow Transmittance
 * - Syncing Volumetric data (VDB/Gas) to GPU buffers
 */
class VolumetricRenderer {
public:
    /**
     * @brief Calculate atmospheric volumetric god rays (crepuscular rays)
     */
    static Vec3 calculateGodRays(const SceneData& scene, const WorldData& world_data, const Ray& ray, float maxDistance, const Hittable* bvh, class AtmosphereLUT* lut = nullptr);

    /**
     * @brief Calculate sun transmittance through solid and volumetric objects
     */
    static float determineSunTransmittance(const Vec3& origin, const Vec3& sunDir, float maxDist, const Hittable* bvh, const SceneData& scene, const WorldData& world_data);

    /**
     * @brief Sync all volumetric data (VDB & Gas) to Optix GPU buffers
     */
    static void syncVolumetricData(SceneData& scene, Backend::IBackend* backend);
    
    /**
     * @brief Apply atmospheric aerial perspective and height fog to a color based on distance
     */
    static Vec3 applyAerialPerspective(const SceneData& scene, const WorldData& world_data, const Vec3& origin, const Vec3& dir, float dist, const Vec3& color, class AtmosphereLUT* lut);

private:
    // Internal helpers
    static float getCloudTransmittance(const SceneData& scene, const Vec3& p, const Vec3& sunDir);
};

#endif // VOLUMETRIC_RENDERER_H
