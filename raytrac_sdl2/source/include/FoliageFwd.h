#pragma once

#include <string>
#include <vector>
#include <memory>
#include "Vec2.h"
#include "Vec3.h"

// Forward declarations
struct SceneData;
class Triangle;

// A single foliage layer definition (e.g. "Pine Trees", "Grass Clumps")
struct TerrainFoliageLayer {
    std::string name = "New Foliage";
    bool enabled = true;
    
    // Mesh settings
    std::string meshPath;        // Path to .obj/.fbx or selected node name
    int meshId = -1;             // OptiX BLAS ID (may be invalidated by rebuild)
    
    // Source triangles for cloning (like scatter brush approach)
    std::vector<std::shared_ptr<Triangle>> sourceTriangles;
    Vec3 meshCenter = Vec3(0, 0, 0);  // Calculated center for proper placement
    
    // Distribution settings
    int density = 1000;          // Target count (max instances)
    int targetMaskLayerId = 0;   // Which splat channel drives this? (0=R, 1=G, 2=B, 3=A)
    float maskThreshold = 0.2f;  // Minimum mask value to spawn
    
    // Transform variation
    Vec2 scaleRange = Vec2(0.8f, 1.2f);
    Vec2 rotationRange = Vec2(0.0f, 360.0f); // Y-axis rotation in degrees
    float alignToNormal = 0.0f;  // 0.0 = Up(Y), 1.0 = Align to Surface Normal
    
    // Runtime data
    std::vector<int> instanceIds; // IDs of spawned instances in OptixAccelManager
    
    // Helper to clear runtime instances
    void clearInstances() {
        instanceIds.clear();
    }
    
    // Calculate mesh center from source triangles
    void calculateMeshCenter() {
        if (sourceTriangles.empty()) return;
        meshCenter = Vec3(0, 0, 0);
        int vertexCount = 0;
        for (const auto& tri : sourceTriangles) {
            if (!tri) continue;
            meshCenter = meshCenter + tri->getV0();
            meshCenter = meshCenter + tri->getV1();
            meshCenter = meshCenter + tri->getV2();
            vertexCount += 3;
        }
        if (vertexCount > 0) {
            meshCenter = meshCenter / (float)vertexCount;
        }
    }
    
    // Check if source is ready for scattering
    bool hasValidSource() const {
        return !sourceTriangles.empty();
    }
};
