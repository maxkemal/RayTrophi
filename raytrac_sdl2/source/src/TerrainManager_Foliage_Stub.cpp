void TerrainManager::updateFoliage(TerrainObject* terrain, SceneData& scene) {
    if (!terrain) return;

    // Ensure OptiX Accel Manager is available
    if (!scene.optixAccel) return;

    // 1. Clear existing foliage instances for this terrain
    // (This is a simplified approach; ideally we'd track and update only changed ones, 
    // but full rebuild per scatter is safer for v1)
    clearFoliage(terrain, scene);

    // 2. Iterate through each foliage layer
    for (auto& slayer : terrain->foliageLayers) {
        if (!slayer.enabled || slayer.meshPath.empty()) continue;

        // Load mesh if needed (getting BLAS ID)
        // Note: We need a way to load a mesh "headless" or use existing SceneLoader
        // For now, let's assume the mesh is loaded via Assimp or we use a primitive
        // Ideally SceneLoader or Renderer should handle this. 
        // We will assume the meshId is already valid OR we need a way to load it.
        // Let's defer actual loading to the UI layer for now, assuming UI sets meshId?
        // NO, UI sets path, Backend should ensure it's loaded.
        
        // TODO: Implement Mesh Loading on demand. 
        // For V1, we will skip if meshId is invalid (-1). 
        // The UI will handle loading the mesh into the system and setting the meshId.
        if (slayer.meshId == -1) continue; 

        // 3. Scatter Logic
        int targetCount = slayer.density; // This is a "density" rating, actually target count
        if (targetCount <= 0) continue;

        // Random generator
        std::mt19937 rng(12345 + terrain->id); // Fixed seed for stability
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        int spawnedCount = 0;
        int maxAttempts = targetCount * 5; // Avoid infinite loop

        for (int i = 0; i < maxAttempts; ++i) {
            if (spawnedCount >= targetCount) break;

            // Random position on terrain (0..1)
            float u = dist(rng);
            float v = dist(rng);

            // Convert to world coords
            float x = (u - 0.5f) * terrain->heightmap.scale_xz;
            float z = (v - 0.5f) * terrain->heightmap.scale_xz;
            
            // Grid coords for masking
            int gx = (int)(u * (terrain->heightmap.width - 1));
            int gy = (int)(v * (terrain->heightmap.height - 1));

            // Check Mask
            float maskValue = 0.0f;
            if (terrain->splatMap) {
                // Read from CPU splat map (assumes 4-channel float or byte)
                // We don't have direct access to splat pixels here easily without reading texture data.
                // Assuming SplatMap keeps a CPU copy? 
                // Texture class might not keep CPU copy if uploaded. 
                // Fallback: Use manual splat layers if stored in 'layers' weights? 
                // Nope, we use Splat Texture.
                
                // WORKAROUND: If splat map is on GPU only, we can't read it easily on CPU.
                // WE MUST ensure we have CPU access. 
                // For now, let's skip mask check or assume uniform for testing.
                maskValue = 1.0f; 
            } else {
                 maskValue = 1.0f; // No mask = global scatter
            }

            // Height check
            float h_norm = terrain->heightmap.getHeight(gx, gy) / terrain->heightmap.scale_y;
            float h_world = h_norm * terrain->heightmap.scale_y;

            if (maskValue < slayer.maskThreshold) continue;

            // Spawn
            // Transform
            float scale = slayer.scaleRange.x + (slayer.scaleRange.y - slayer.scaleRange.x) * dist(rng);
            float rotY = slayer.rotationRange.x + (slayer.rotationRange.y - slayer.rotationRange.x) * dist(rng);
            
            // Build Transform Matrix
            // ... (Math logic) ...
            
            // Add Instance
            // float transform[12] = ...;
            // int instId = scene.optixAccel->addInstance(slayer.meshId, transform, 0); // 0 = Material offset?
            // slayer.instanceIds.push_back(instId);
            
            spawnedCount++;
        }
    }
    
    // Trigger TLAS rebuild
    scene.optixAccel->buildTLAS();
}

void TerrainManager::clearFoliage(TerrainObject* terrain, SceneData& scene) {
    if (!terrain || !scene.optixAccel) return;
    for (auto& slayer : terrain->foliageLayers) {
        for (int id : slayer.instanceIds) {
            scene.optixAccel->removeInstance(id);
        }
        slayer.instanceIds.clear();
    }
}
