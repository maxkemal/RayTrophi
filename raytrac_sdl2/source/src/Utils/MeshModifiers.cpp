#include "MeshModifiers.h"
#include <map>

namespace MeshModifiers {

    std::vector<std::shared_ptr<Triangle>> SubdivideSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels) {
        std::vector<std::shared_ptr<Triangle>> currentMesh = inputMesh;

        for (int level = 0; level < levels; ++level) {
            std::vector<std::shared_ptr<Triangle>> nextMesh;
            nextMesh.reserve(currentMesh.size() * 4);

            for (const auto& tri : currentMesh) {
                if (!tri) continue;

                Vec3 v0 = tri->getVertexPosition(0);
                Vec3 v1 = tri->getVertexPosition(1);
                Vec3 v2 = tri->getVertexPosition(2);

                Vec3 n0 = tri->getVertexNormal(0);
                Vec3 n1 = tri->getVertexNormal(1);
                Vec3 n2 = tri->getVertexNormal(2);

                auto [uv0, uv1, uv2] = tri->getUVCoordinates();
                
                uint16_t matID = tri->getMaterialID();
                auto transform = tri->getTransformHandle();
                std::string nodeName = tri->getNodeName();

                // Midpoints
                Vec3 m01 = (v0 + v1) * 0.5f;
                Vec3 m12 = (v1 + v2) * 0.5f;
                Vec3 m20 = (v2 + v0) * 0.5f;

                // Normals (Slerp-like for exactness, but simple lerp + normalize is fine here)
                Vec3 nm01 = (n0 + n1).normalize();
                Vec3 nm12 = (n1 + n2).normalize();
                Vec3 nm20 = (n2 + n0).normalize();

                // UVs
                Vec2 uvm01 = (uv0 + uv1) * 0.5f;
                Vec2 uvm12 = (uv1 + uv2) * 0.5f;
                Vec2 uvm20 = (uv2 + uv0) * 0.5f;

                // Create 4 new triangles
                std::shared_ptr<Triangle> t1 = std::make_shared<Triangle>(
                    v0, m01, m20, n0, nm01, nm20, uv0, uvm01, uvm20, matID);
                std::shared_ptr<Triangle> t2 = std::make_shared<Triangle>(
                    m01, v1, m12, nm01, n1, nm12, uvm01, uv1, uvm12, matID);
                std::shared_ptr<Triangle> t3 = std::make_shared<Triangle>(
                    m12, v2, m20, nm12, n2, nm20, uvm12, uv2, uvm20, matID);
                std::shared_ptr<Triangle> t4 = std::make_shared<Triangle>(
                    m01, m12, m20, nm01, nm12, nm20, uvm01, uvm12, uvm20, matID);

                t1->setTransformHandle(transform);
                t2->setTransformHandle(transform);
                t3->setTransformHandle(transform);
                t4->setTransformHandle(transform);

                t1->setNodeName(nodeName);
                t2->setNodeName(nodeName);
                t3->setNodeName(nodeName);
                t4->setNodeName(nodeName);

                t1->update_bounding_box();
                t2->update_bounding_box();
                t3->update_bounding_box();
                t4->update_bounding_box();

                nextMesh.push_back(t1);
                nextMesh.push_back(t2);
                nextMesh.push_back(t3);
                nextMesh.push_back(t4);
            }
            currentMesh = std::move(nextMesh);
        }

        return currentMesh;
    }

    // Helper structure for vertex hashing/smoothing
    struct VertexData {
        Vec3 accumulatedPos;
        int count;
        Vec3 computedNormal;
    };

    std::vector<std::shared_ptr<Triangle>> SmoothSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels, float smoothAngle) {
        // 1. Linearly subdivide first to get the topology density
        auto currentMesh = SubdivideSubD(inputMesh, levels);

        // We do a simple Laplacian smoothing iteration
        // To do this, we need to build an adjacency or vertex map
        // Due to lack of half-edge data structure, we'll use a position-based point cloud smoothing
        const float epsilon = 1e-4f;
        
        // A simple Lambda to hash a vec3 to a string or int for map lookup
        auto hashVec = [epsilon](const Vec3& v) -> std::string {
            int x = static_cast<int>(std::round(v.x / epsilon));
            int y = static_cast<int>(std::round(v.y / epsilon));
            int z = static_cast<int>(std::round(v.z / epsilon));
            return std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z);
        };

        // Smooth iterations
        int smoothIterations = 1; // 1 iter for basic smoothing
        for (int iter = 0; iter < smoothIterations; ++iter) {
            std::map<std::string, VertexData> vertexMap;

            // 1. Accumulate positions from neighboring triangles (Laplacian approximation)
            for (const auto& tri : currentMesh) {
                if (!tri) continue;
                for (int i = 0; i < 3; ++i) {
                    Vec3 pos = tri->getVertexPosition(i);
                    std::string key = hashVec(pos);
                    
                    // Add vertices of THIS triangle to the neighbors of the current vertex
                    for(int j=0; j<3; ++j) {
                        if (i == j) continue;
                        vertexMap[key].accumulatedPos = vertexMap[key].accumulatedPos + tri->getVertexPosition(j);
                        vertexMap[key].count++;
                    }
                }
            }

            // 2. Apply smoothed positions
            for (auto& tri : currentMesh) {
                if (!tri) continue;
                for (int i = 0; i < 3; ++i) {
                    Vec3 pos = tri->getVertexPosition(i);
                    std::string key = hashVec(pos);
                    
                    if (vertexMap[key].count > 0) {
                        Vec3 avgPos = vertexMap[key].accumulatedPos * (1.0f / (float)vertexMap[key].count);
                        // Interpolate between original and average based on smoothAngle (0.0 to 1.0)
                        Vec3 newPos = pos * (1.0f - smoothAngle) + avgPos * smoothAngle;
                        tri->setVertexPosition(i, newPos);
                    }
                }
            }

            // 3. Recompute Normals for smoothed geometry
            std::map<std::string, VertexData> normalMap;
            // Face normals to vertex normals
            for (auto& tri : currentMesh) {
                if (!tri) continue;
                Vec3 v0 = tri->getVertexPosition(0);
                Vec3 v1 = tri->getVertexPosition(1);
                Vec3 v2 = tri->getVertexPosition(2);
                
                Vec3 faceNormal = ((v1 - v0).cross(v2 - v0)).normalize();
                
                for(int i=0; i<3; ++i) {
                    Vec3 pos = tri->getVertexPosition(i);
                    std::string key = hashVec(pos);
                    normalMap[key].computedNormal = (normalMap[key].computedNormal + faceNormal);
                }
            }
            
            // Apply smoothed normals
            for (auto& tri : currentMesh) {
               if (!tri) continue;
               for (int i = 0; i < 3; ++i) {
                   Vec3 pos = tri->getVertexPosition(i);
                   std::string key = hashVec(pos);
                   Vec3 smoothNormal = normalMap[key].computedNormal.normalize();
                   tri->setVertexNormal(i, smoothNormal);
               }
               tri->update_bounding_box();
            }
        }

        return currentMesh;
    }

    void ModifierData::serialize(nlohmann::json& j) const {
        j["name"] = name;
        j["type"] = static_cast<int>(type);
        j["enabled"] = enabled;
        j["levels"] = levels;
        j["smoothAngle"] = smoothAngle;
    }

    void ModifierData::deserialize(const nlohmann::json& j) {
        if (j.contains("name")) name = j["name"].get<std::string>();
        if (j.contains("type")) type = static_cast<ModifierType>(j["type"].get<int>());
        if (j.contains("enabled")) enabled = j["enabled"].get<bool>();
        if (j.contains("levels")) levels = j["levels"].get<int>();
        if (j.contains("smoothAngle")) smoothAngle = j["smoothAngle"].get<float>();
    }

    std::vector<std::shared_ptr<Triangle>> ModifierStack::evaluate(const std::vector<std::shared_ptr<Triangle>>& baseMesh) const {
        std::vector<std::shared_ptr<Triangle>> currentMesh = baseMesh;

        // Apply modifiers sequentially
        for (const auto& mod : modifiers) {
            if (!mod.enabled) continue;

            if (mod.type == ModifierType::FlatSubdivision) {
                currentMesh = SubdivideSubD(currentMesh, mod.levels);
            } else if (mod.type == ModifierType::SmoothSubdivision) {
                currentMesh = SmoothSubD(currentMesh, mod.levels, mod.smoothAngle);
            }
        }

        return currentMesh;
    }

    void ModifierStack::serialize(nlohmann::json& j) const {
        j["modifiers"] = nlohmann::json::array();
        for (const auto& mod : modifiers) {
            nlohmann::json modJson;
            mod.serialize(modJson);
            j["modifiers"].push_back(modJson);
        }
    }

    void ModifierStack::deserialize(const nlohmann::json& j) {
        modifiers.clear();
        if (j.contains("modifiers") && j["modifiers"].is_array()) {
            for (const auto& modJson : j["modifiers"]) {
                ModifierData mod;
                mod.deserialize(modJson);
                modifiers.push_back(mod);
            }
        }
    }

}
