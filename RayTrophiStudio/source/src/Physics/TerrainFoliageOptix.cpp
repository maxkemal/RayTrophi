#include <thread>
#include <future>
#include "TerrainManager.h"
#include "TerrainSystem.h"
#include "TriangleMesh.h"
#include "OptixWrapper.h"
#include "OptixAccelManager.h"
#include "globals.h"
#include <random>
#include <algorithm>

void TerrainManager::updateFoliage(TerrainObject* terrain, OptixWrapper* optix) {
    if (!terrain || !optix) {
        SCENE_LOG_WARN("[Foliage] updateFoliage called with null terrain or optix");
        return;
    }

    OptixAccelManager* accel = optix->getAccelManager();
    if (!accel) {
        SCENE_LOG_WARN("[Foliage] No AccelManager available");
        return;
    }

    SCENE_LOG_INFO("[Foliage] Updating foliage for terrain " + std::to_string(terrain->id) +
                   ", layers: " + std::to_string(terrain->foliageLayers.size()));

    // 1. Clear existing foliage instances for this terrain
    clearFoliage(terrain, optix);

    int totalSpawned = 0;

    // 2. Iterate through each foliage layer
    for (size_t layerIdx = 0; layerIdx < terrain->foliageLayers.size(); ++layerIdx) {
        auto& slayer = terrain->foliageLayers[layerIdx];

        if (!slayer.enabled) {
            SCENE_LOG_INFO("[Foliage] Layer " + std::to_string(layerIdx) + " disabled, skipping");
            continue;
        }
        if (slayer.meshPath.empty()) {
            SCENE_LOG_INFO("[Foliage] Layer " + std::to_string(layerIdx) +
                           " has no meshPath, skipping");
            continue;
        }

        // Skip if mesh not assigned (meshId -1)
        if (slayer.meshId == -1) {
            SCENE_LOG_WARN("[Foliage] Layer " + std::to_string(layerIdx) +
                           " meshId is -1 (path: " + slayer.meshPath + "), skipping");
            continue;
        }

        SCENE_LOG_INFO("[Foliage] Layer " + std::to_string(layerIdx) +
                       ": meshId=" + std::to_string(slayer.meshId) +
                       ", density=" + std::to_string(slayer.density) + ", path=" + slayer.meshPath);

        // 3. Scatter Logic (Multi-threaded)
        int targetCount = slayer.density;
        if (targetCount <= 0) {
            SCENE_LOG_WARN("[Foliage] Layer " + std::to_string(layerIdx) +
                           " density is 0, skipping");
            continue;
        }

        int maxAttempts = targetCount * 5; // Avoid infinite loop

        // Cache splat texture dimensions for faster access
        int splatW = 0, splatH = 0;
        Texture* splatTex = terrain->splatMap.get();
        if (splatTex && splatTex->is_loaded()) {
            splatW = splatTex->width;
            splatH = splatTex->height;
        }

        // Border keep-out (two heightmap cells): the procedural edge-falloff lip
        // pins the outermost rows to the falloff target, which otherwise collects
        // a line of instances along the terrain rim.
        const float legacyCellX =
            terrain->heightmap.scale_xz / (float)(std::max)(1, terrain->heightmap.width - 1);
        const float legacyCellZ =
            terrain->heightmap.scale_xz / (float)(std::max)(1, terrain->heightmap.height - 1);
        const float edgeMarginUV = terrain->heightmap.scale_xz > 1e-6f
                                       ? std::clamp(2.0f * (std::max)(legacyCellX, legacyCellZ) /
                                                        terrain->heightmap.scale_xz,
                                                    0.0f, 0.49f)
                                       : 0.0f;

        // Structure to hold pre-computed instance data
        struct InstanceData {
            float transform[12];
            std::string name;
            bool valid = false;
        };

        // Multi-threaded instance data generation
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4;
        size_t attempts_per_thread = (maxAttempts + num_threads - 1) / num_threads;

        std::vector<std::vector<InstanceData>> thread_results(num_threads);
        std::vector<std::future<void>> futures;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t attempt_start = t * attempts_per_thread;
            size_t attempt_end = std::min(attempt_start + attempts_per_thread, (size_t)maxAttempts);
            if (attempt_start >= attempt_end)
                continue;

            futures.push_back(std::async(std::launch::async, [&, t, attempt_start, attempt_end]() {
                // Thread-local RNG with unique seed
                std::mt19937 rng(12345 + terrain->id * 7 + slayer.meshId * 3 + (int)t * 1000);
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);

                // Skip to thread's start position
                for (size_t skip = 0; skip < attempt_start; ++skip) {
                    dist(rng);
                    dist(rng);
                    dist(rng);
                    dist(rng); // Consume random numbers
                }

                std::vector<InstanceData>& results = thread_results[t];
                results.reserve((attempt_end - attempt_start) / 5); // Estimate

                for (size_t i = attempt_start; i < attempt_end; ++i) {
                    // Random position on terrain (0..1)
                    float u = dist(rng);
                    float v = dist(rng);

                    if (edgeMarginUV > 0.0f && (u < edgeMarginUV || u > 1.0f - edgeMarginUV ||
                                                v < edgeMarginUV || v > 1.0f - edgeMarginUV))
                        continue;

                    // Grid coords for heightmap
                    int gx = (int)(u * (terrain->heightmap.width - 1));
                    int gy = (int)(v * (terrain->heightmap.height - 1));

                    // Check Mask
                    float maskValue = 0.0f;
                    if (splatTex && splatW > 0 && splatH > 0 && !splatTex->pixels.empty()) {
                        int sampleX = std::clamp((int)(u * (float)(splatW - 1)), 0, splatW - 1);
                        int sampleY =
                            std::clamp((int)((1.0f - v) * (float)(splatH - 1)), 0, splatH - 1);
                        int pIdx = sampleY * splatW + sampleX;
                        if (pIdx >= 0 && pIdx < (int)splatTex->pixels.size()) {
                            const auto& p = splatTex->pixels[pIdx];
                            int ch = slayer.targetMaskLayerId;
                            if (ch == 0)
                                maskValue = p.r / 255.0f;
                            else if (ch == 1)
                                maskValue = p.g / 255.0f;
                            else if (ch == 2)
                                maskValue = p.b / 255.0f;
                            else if (ch == 3)
                                maskValue = p.a / 255.0f;
                        }
                    } else {
                        maskValue = 1.0f;
                    }

                    if (maskValue < slayer.maskThreshold)
                        continue;

                    // Compute World Position
                    float x = u * terrain->heightmap.scale_xz;
                    float z = v * terrain->heightmap.scale_xz;

                    float h_norm =
                        terrain->heightmap.getHeight(gx, gy) / terrain->heightmap.scale_y;
                    float y = h_norm * terrain->heightmap.scale_y;

                    Vec3 pos(x, y, z);
                    if (terrain->transform) {
                        pos = terrain->transform->getFinal().multiplyVector(Vec4(pos, 1.0f)).xyz();
                    }

                    // Transform Generation
                    float scaleC = slayer.scaleRange.x +
                                   (slayer.scaleRange.y - slayer.scaleRange.x) * dist(rng);
                    float rotY = slayer.rotationRange.x +
                                 (slayer.rotationRange.y - slayer.rotationRange.x) * dist(rng);

                    float rad = rotY * 3.14159265f / 180.0f;
                    float c = cosf(rad);
                    float s = sinf(rad);

                    InstanceData inst;
                    inst.transform[0] = scaleC * c;
                    inst.transform[1] = 0.0f;
                    inst.transform[2] = scaleC * s;
                    inst.transform[3] = pos.x;
                    inst.transform[4] = 0.0f;
                    inst.transform[5] = scaleC;
                    inst.transform[6] = 0.0f;
                    inst.transform[7] = pos.y;
                    inst.transform[8] = scaleC * -s;
                    inst.transform[9] = 0.0f;
                    inst.transform[10] = scaleC * c;
                    inst.transform[11] = pos.z;
                    inst.name = "Foliage_" + std::to_string(terrain->id) + "_" +
                                std::to_string(slayer.meshId) + "_" + std::to_string(i);
                    inst.valid = true;

                    results.push_back(inst);
                }
            }));
        }

        // Wait for all threads
        for (auto& f : futures)
            f.get();

        // Merge results and add to GPU (serial - GPU API not thread-safe)
        int spawnedCount = 0;
        for (auto& results : thread_results) {
            for (auto& inst : results) {
                if (spawnedCount >= targetCount)
                    break;
                if (!inst.valid)
                    continue;

                int instId = accel->addInstance(slayer.meshId, inst.transform, 0,
                                                InstanceType::Mesh, inst.name);
                if (instId >= 0) {
                    slayer.instanceIds.push_back(instId);
                    spawnedCount++;
                }
            }
            if (spawnedCount >= targetCount)
                break;
        }

        SCENE_LOG_INFO("[Foliage] Layer " + std::to_string(layerIdx) + " spawned " +
                       std::to_string(spawnedCount) + "/" + std::to_string(targetCount) +
                       " instances");
        totalSpawned += spawnedCount;
    }

    SCENE_LOG_INFO("[Foliage] Total spawned: " + std::to_string(totalSpawned) + " instances");

    // Trigger TLAS rebuild
    if (totalSpawned > 0) {
        accel->buildTLAS();
        SCENE_LOG_INFO("[Foliage] TLAS rebuild triggered");
    }
}

void TerrainManager::clearFoliage(TerrainObject* terrain, OptixWrapper* optix) {
    if (!terrain || !optix)
        return;
    OptixAccelManager* accel = optix->getAccelManager();
    if (!accel)
        return;

    for (auto& slayer : terrain->foliageLayers) {
        for (int id : slayer.instanceIds) {
            accel->removeInstance(id);
        }
        slayer.instanceIds.clear();
    }
}

void TerrainManager::reapplyAllFoliage(OptixWrapper* optix) {
    if (!optix)
        return;
    for (auto& t : terrains) {
        // Only update if foliage layers exist and are enabled
        bool hasFoliage = false;
        for (const auto& l : t.foliageLayers)
            if (l.enabled && l.density > 0)
                hasFoliage = true;

        if (hasFoliage) {
            updateFoliage(&t, optix);
        }
    }
}
