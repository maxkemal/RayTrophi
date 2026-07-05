#include "InstanceManager.h"
#include "Triangle.h"
#include "scene_data.h"
#include "EmbreeBVH.h"
#include "OptixWrapper.h"
#include "Backend/IBackend.h"
#include "FoliageWindSystem.h"
#include "TerrainManager.h"
#include "globals.h"
#include <cstdint>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <future>
#include <cstring>

// ═══════════════════════════════════════════════════════════════════════════════
// GROUP MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

int InstanceManager::createGroup(const std::string& name, const std::string& source_node_name,
                                  const std::vector<std::shared_ptr<Triangle>>& source_triangles) {
    // Allowed empty for multi-source Foliage Layers
    
    InstanceGroup group;
    int group_id = next_group_id++; // Get ID before creating group
    group.id = group_id;
    group.name = name;
    group.source_node_name = source_node_name;
    group.source_triangles = source_triangles;
    group.gpu_dirty = true;
    
    groups.push_back(std::move(group));
    
    SCENE_LOG_INFO("[InstanceManager] Created group '" + name + "' (ID: " + 
                   std::to_string(group_id) + ") with " + 
                   std::to_string(source_triangles.size()) + " source triangles");
    
    return group_id;
}

void InstanceManager::deleteGroup(int group_id) {
    auto it = std::find_if(groups.begin(), groups.end(),
        [group_id](const InstanceGroup& g) { return g.id == group_id; });
    
    if (it != groups.end()) {
        SCENE_LOG_INFO("[InstanceManager] Deleted group '" + it->name + "'");
        groups.erase(it);
    }
}

InstanceGroup* InstanceManager::getGroup(int group_id) {
    auto it = std::find_if(groups.begin(), groups.end(),
        [group_id](const InstanceGroup& g) { return g.id == group_id; });
    return (it != groups.end()) ? &(*it) : nullptr;
}

const InstanceGroup* InstanceManager::getGroup(int group_id) const {
    auto it = std::find_if(groups.begin(), groups.end(),
        [group_id](const InstanceGroup& g) { return g.id == group_id; });
    return (it != groups.end()) ? &(*it) : nullptr;
}

InstanceGroup* InstanceManager::findGroupByName(const std::string& name) {
    auto it = std::find_if(groups.begin(), groups.end(),
        [&name](const InstanceGroup& g) { return g.name == name; });
    return (it != groups.end()) ? &(*it) : nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BRUSH OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Simple Möller-Trumbore ray-triangle intersection utility local to this file
static float local_RayTriangle(const Vec3& orig, const Vec3& dir,
                               const Vec3& v0, const Vec3& v1, const Vec3& v2)
{
    const float EPS = 1e-8f;
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 h = Vec3::cross(dir, edge2);
    float a = Vec3::dot(edge1, h);
    if (fabsf(a) < EPS) return -1.f;
    float f = 1.f / a;
    Vec3 s = orig - v0;
    float u = f * Vec3::dot(s, h);
    if (u < 0.f || u > 1.f) return -1.f;
    Vec3 q = Vec3::cross(s, edge1);
    float v = f * Vec3::dot(dir, q);
    if (v < 0.f || u + v > 1.f) return -1.f;
    float t = f * Vec3::dot(edge2, q);
    return (t > EPS) ? t : -1.f;
}

int InstanceManager::paintInstances(int group_id, const Vec3& center, const Vec3& normal,
                                     float brush_radius, float density_multiplier, SceneData* scene) {
    InstanceGroup* group = getGroup(group_id);
    if (!group) return 0;
    
    // Calculate number of instances based on density
    float area = 3.14159f * brush_radius * brush_radius;
    float density = group->brush_settings.density * density_multiplier;
    int target_count = static_cast<int>(area * density);
    
    if (target_count <= 0) return 0;
    
    // Minimum distance between instances (based on density)
    float min_distance = 1.0f / sqrtf(density);
    
    // Generate poisson-distributed points
    std::vector<Vec3> points = generatePoissonPoints(center, brush_radius, min_distance);
    
    // Add instances at each point (check against existing instances)
    int added = 0;
    // If target is mesh, collect candidate triangles from scene
    std::vector<std::shared_ptr<Triangle>> target_tris;
    bool use_mesh_projection = false;
    if (group->target_type == InstanceGroup::TargetType::MESH && !group->target_node_name.empty() && scene) {
        use_mesh_projection = true;
        for (auto& obj : scene->world.objects) {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                if (tri->getNodeName() == group->target_node_name) target_tris.push_back(tri);
            } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                // Flat-aware: the scatter target surface may be ONE flat SoA TriangleMesh now.
                if (tm->nodeName == group->target_node_name && tm->geometry) {
                    const size_t nTris = tm->num_triangles();
                    for (size_t t = 0; t < nTris; ++t)
                        target_tris.push_back(std::make_shared<Triangle>(tm, static_cast<uint32_t>(t)));
                }
            }
        }
    }

    for (const Vec3& pos : points) {
        // Check if too close to any existing instance
        bool too_close = false;
        for (const auto& existing : group->instances) {
            float dist = (existing.position - pos).length();
            if (dist < min_distance * 0.8f) {  // 80% of min_distance for existing
                too_close = true;
                break;
            }
        }
        
        if (!too_close) {
            // Re-sample terrain height and normal for each point to follow slope
            Vec3 surface_normal = normal;
            Vec3 surface_pos = pos;

            std::shared_ptr<Triangle> best_tri;
            Vec3 best_v0(0,0,0), best_v1(0,0,0), best_v2(0,0,0);
            if (use_mesh_projection && !target_tris.empty()) {
                // Cast a short ray downwards from above the point to find mesh surface
                Vec3 ray_orig = Vec3(pos.x, pos.y + 1000.0f, pos.z);
                Vec3 ray_dir = Vec3(0, -1, 0);
                float closest_t = 1e20f;
                Vec3 best_n(0,1,0);
                bool hit_found = false;
                for (const auto& tri : target_tris) {
                    Vec3 v0 = tri->getV0();
                    Vec3 v1 = tri->getV1();
                    Vec3 v2 = tri->getV2();
                    float t = local_RayTriangle(ray_orig, ray_dir, v0, v1, v2);
                    if (t > 0 && t < closest_t) {
                        closest_t = t;
                        Vec3 fn = Vec3::cross(v1 - v0, v2 - v0);
                        float fn_len = fn.length();
                        best_n = (fn_len > 1e-6f) ? fn / fn_len : Vec3(0,1,0);
                        if (best_n.y < 0.f) best_n = -best_n;
                        hit_found = true;
                        best_tri = tri;
                        best_v0 = v0; best_v1 = v1; best_v2 = v2;
                    }
                }
                if (hit_found) {
                    surface_pos = ray_orig + ray_dir * closest_t;
                    surface_normal = best_n;
                } else if (TerrainManager::getInstance().hasActiveTerrain()) {
                    surface_pos.y = TerrainManager::getInstance().sampleHeight(pos.x, pos.z);
                    surface_normal = TerrainManager::getInstance().sampleNormal(pos.x, pos.z);
                }
            } else {
                if (TerrainManager::getInstance().hasActiveTerrain()) {
                    surface_pos.y = TerrainManager::getInstance().sampleHeight(pos.x, pos.z);
                    surface_normal = TerrainManager::getInstance().sampleNormal(pos.x, pos.z);
                }
            }

            // Faz 8b Field bridge: gate/scale MESH-target placement by per-vertex float
            // attributes on the hit triangle's parent mesh (barycentric sample at the
            // actual hit point), mirroring what ScatterInstancesNode does in the Geo-DAG.
            // Density and scale read INDEPENDENT named attributes (Blender vertex-group-
            // per-slot style) so e.g. a paint mask can gate density while a separate
            // "edge falloff" attribute drives scale. Legacy standalone facades (no
            // parentMesh, e.g. non-flat objects) have no attribute storage to sample —
            // sampled value stays 1.0 (unmasked) for them, same as "attribute not found".
            const std::string& density_attr_name = group->brush_settings.density_mask_attribute;
            const std::string& scale_attr_name = group->brush_settings.scale_mask_attribute;
            auto sampleFieldAttribute = [&](const std::string& attrName) -> float {
                if (attrName.empty() || !best_tri || !best_tri->parentMesh || !best_tri->parentMesh->geometry) return 1.0f;
                const auto& mgeom = *best_tri->parentMesh->geometry;
                const float* attr = mgeom.get_attribute_data<float>(attrName);
                const int faceIdx = best_tri->getFaceIndex();
                if (!attr || faceIdx < 0 || static_cast<size_t>(faceIdx) * 3 + 2 >= mgeom.indices.size()) return 1.0f;
                const Vec3 e0 = best_v1 - best_v0;
                const Vec3 e1 = best_v2 - best_v0;
                const Vec3 e2 = surface_pos - best_v0;
                const float d00 = e0.dot(e0), d01 = e0.dot(e1), d11 = e1.dot(e1);
                const float d20 = e2.dot(e0), d21 = e2.dot(e1);
                const float denom = d00 * d11 - d01 * d01;
                if (std::fabs(denom) <= 1e-12f) return 1.0f;
                const float bw = (d11 * d20 - d01 * d21) / denom;
                const float bv = (d00 * d21 - d01 * d20) / denom;
                const float bu = 1.0f - bv - bw;
                const uint32_t i0 = mgeom.indices[static_cast<size_t>(faceIdx) * 3 + 0];
                const uint32_t i1 = mgeom.indices[static_cast<size_t>(faceIdx) * 3 + 1];
                const uint32_t i2 = mgeom.indices[static_cast<size_t>(faceIdx) * 3 + 2];
                return bu * attr[i0] + bv * attr[i1] + bw * attr[i2];
            };
            if (!density_attr_name.empty() && sampleFieldAttribute(density_attr_name) <= 0.001f) {
                continue;  // mask gates placement, same rejection idea as splat/exclusion below
            }

            // If group requests splat-map masking (terrain channels), respect it: only add when mask allows
            int splat_ch = group->brush_settings.splat_map_channel;
            int excl_ch = group->brush_settings.exclusion_channel;
            float excl_th = group->brush_settings.exclusion_threshold;

            if (splat_ch >= 0) {
                float m = TerrainManager::getInstance().sampleSplatChannel(surface_pos.x, surface_pos.z, splat_ch);
                if (m <= 0.001f) {
                    continue; // not allowed on this channel
                }
            }
            if (excl_ch >= 0) {
                float em = TerrainManager::getInstance().sampleSplatChannel(surface_pos.x, surface_pos.z, excl_ch);
                if (em >= excl_th) {
                    continue; // excluded by mask
                }
            }

            InstanceTransform t = group->generateRandomTransform(surface_pos, surface_normal);
            if (!scale_attr_name.empty() && group->brush_settings.scale_mask_influence > 0.0f) {
                const float scale_sample = sampleFieldAttribute(scale_attr_name);
                const float f = 1.0f - group->brush_settings.scale_mask_influence * (1.0f - scale_sample);
                t.scale = t.scale * f;
            }
            group->addInstance(t);
            added++;
        }
        
        if (added >= target_count) break;
    }
    
    return added;
}

int InstanceManager::eraseInstances(int group_id, const Vec3& center, float brush_radius) {
    InstanceGroup* group = getGroup(group_id);
    if (!group) return 0;
    
    size_t before = group->instances.size();
    group->removeInstancesInRadius(center, brush_radius);
    return static_cast<int>(before - group->instances.size());
}

int InstanceManager::eraseAllGroupsInRadius(const Vec3& center, float brush_radius) {
    int total_removed = 0;
    for (auto& group : groups) {
        size_t before = group.instances.size();
        group.removeInstancesInRadius(center, brush_radius);
        total_removed += static_cast<int>(before - group.instances.size());
    }
    return total_removed;
}

// ═══════════════════════════════════════════════════════════════════════════════
// POISSON DISK SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════

std::vector<Vec3> InstanceManager::generatePoissonPoints(const Vec3& center, float radius,
                                                          float min_distance, int max_attempts) {
    std::vector<Vec3> points;
    std::vector<Vec3> active;
    
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_angle(0.0f, 2.0f * 3.14159f);
    
    // Start with center point
    points.push_back(center);
    active.push_back(center);
    
    while (!active.empty()) {
        // Pick random active point
        std::uniform_int_distribution<size_t> idx_dist(0, active.size() - 1);
        size_t idx = idx_dist(rng);
        Vec3 point = active[idx];
        
        bool found = false;
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            // Generate random point at distance [min_distance, 2*min_distance]
            float angle = dist_angle(rng);
            float r = min_distance * (1.0f + dist01(rng));
            
            Vec3 new_point = point + Vec3(cosf(angle) * r, 0, sinf(angle) * r);
            
            // Check if within brush radius
            float dist_from_center = (new_point - center).length();
            if (dist_from_center > radius) continue;
            
            // Check if far enough from all existing points
            bool too_close = false;
            for (const Vec3& p : points) {
                if ((new_point - p).length() < min_distance) {
                    too_close = true;
                    break;
                }
            }
            
            if (!too_close) {
                points.push_back(new_point);
                active.push_back(new_point);
                found = true;
                break;
            }
        }
        
        if (!found) {
            // Remove from active list
            active.erase(active.begin() + idx);
        }
    }
    
    return points;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU SYNCHRONIZATION (Placeholder - will integrate with OptixWrapper)
// ═══════════════════════════════════════════════════════════════════════════════

FoliageWindUpdateStats InstanceManager::updateWind(float time, SceneData& scene, Backend::IBackend* backend) {
    return FoliageWindSystem::update(scene, time, backend);
}

void InstanceManager::syncToGPU(OptixWrapper* optix) {
    // TODO: Implement when integrating with OptixWrapper
    // For each dirty group:
    //   - Register/update BLAS
    //   - Create/update TLAS instances
    for (auto& group : groups) {
        if (group.gpu_dirty) {
            // Will call optix->registerInstanceGroup() etc.
            group.gpu_dirty = false;
        }
    }
}

void InstanceManager::rebuildAllTLAS(OptixWrapper* optix) {
    markAllDirty();
    syncToGPU(optix);
}

void InstanceManager::markAllDirty() {
    for (auto& group : groups) {
        group.gpu_dirty = true;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════

size_t InstanceManager::getTotalInstanceCount() const {
    // NOTE: every caller uses this as the "foliage tail" length appended to
    // SceneData::world.objects (to skip it during picking / selection / save).
    // Transient groups (particle render bridges) are batched scatter that is
    // NEVER added to world.objects, so counting them here would make callers skip
    // that many *real* scene objects -> they become unselectable / vanish from the
    // hierarchy whenever particles exist. Exclude transient groups.
    size_t total = 0;
    for (const auto& group : groups) {
        if (group.transient) continue;
        total += group.instances.size();
    }
    return total;
}

size_t InstanceManager::getTotalTriangleCount() const {
    size_t total = 0;
    for (const auto& group : groups) {
        if (group.transient) continue;
        total += group.getTriangleCount();
    }
    return total;
}

void InstanceManager::clearAll() {
    groups.clear();
    next_group_id = 1;
    SCENE_LOG_INFO("[InstanceManager] Cleared all instance groups");
}

// ─────────────────────────────────────────────────────────────────────────
// SERIALIZATION
// ─────────────────────────────────────────────────────────────────────────

using json = nlohmann::json;

#include "ParallelBVHNode.h"

namespace {
    constexpr const char* kInstanceBinaryCodecRaw = "raw";
    constexpr const char* kInstanceBinaryCodecPacked16 = "packed16";

    struct PreparedInstanceBlob {
        std::string codec = kInstanceBinaryCodecPacked16;
        std::vector<char> bytes;
        uint32_t count = 0;
        Vec3 position_min = Vec3(0.0f);
        Vec3 position_max = Vec3(0.0f);
        Vec3 rotation_min = Vec3(0.0f);
        Vec3 rotation_max = Vec3(0.0f);
        Vec3 scale_min = Vec3(1.0f);
        Vec3 scale_max = Vec3(1.0f);
    };

    struct PendingInstanceBlobRead {
        size_t group_index = 0;
        std::string codec = kInstanceBinaryCodecPacked16;
        std::vector<char> bytes;
        Vec3 position_min = Vec3(0.0f);
        Vec3 position_max = Vec3(0.0f);
        Vec3 rotation_min = Vec3(0.0f);
        Vec3 rotation_max = Vec3(0.0f);
        Vec3 scale_min = Vec3(1.0f);
        Vec3 scale_max = Vec3(1.0f);
    };

    constexpr size_t instanceRecordSize() {
        return sizeof(Vec3) * 3 + sizeof(int32_t);
    }

    constexpr size_t packed16RecordSize() {
        return sizeof(uint16_t) * 10;
    }

    float quantizeNormalizedToFloat(uint16_t value) {
        return static_cast<float>(value) / 65535.0f;
    }

    uint16_t quantizeFloatToU16(float value, float min_value, float max_value) {
        const float range = max_value - min_value;
        if (std::fabs(range) <= 1e-12f) {
            return 0;
        }
        const float normalized = std::clamp((value - min_value) / range, 0.0f, 1.0f);
        return static_cast<uint16_t>(std::lround(normalized * 65535.0f));
    }

    float dequantizeU16ToFloat(uint16_t value, float min_value, float max_value) {
        const float range = max_value - min_value;
        if (std::fabs(range) <= 1e-12f) {
            return min_value;
        }
        return min_value + quantizeNormalizedToFloat(value) * range;
    }

    void updateBounds(Vec3& min_v, Vec3& max_v, const Vec3& value) {
        min_v.x = std::min(min_v.x, value.x);
        min_v.y = std::min(min_v.y, value.y);
        min_v.z = std::min(min_v.z, value.z);
        max_v.x = std::max(max_v.x, value.x);
        max_v.y = std::max(max_v.y, value.y);
        max_v.z = std::max(max_v.z, value.z);
    }

    json vec3ToJsonArray(const Vec3& v) {
        return json::array({v.x, v.y, v.z});
    }

    bool readVec3JsonArray(simdjson::simdjson_result<simdjson::dom::element> el_result, Vec3& out) {
        simdjson::dom::element el;
        if (el_result.get(el)) return false;
        simdjson::dom::array arr;
        if (el.get_array().get(arr)) return false;
        auto it = arr.begin();
        if (it == arr.end()) return false;
        double x = 0.0;
        if ((*it).get(x)) return false;
        ++it;
        if (it == arr.end()) return false;
        double y = 0.0;
        if ((*it).get(y)) return false;
        ++it;
        if (it == arr.end()) return false;
        double z = 0.0;
        if ((*it).get(z)) return false;
        out = Vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        return true;
    }

    PreparedInstanceBlob prepareInstanceBlob(const InstanceGroup& group) {
        PreparedInstanceBlob result;
        result.count = static_cast<uint32_t>(group.instances.size());
        result.bytes.resize(sizeof(uint32_t) + static_cast<size_t>(result.count) * packed16RecordSize());

        char* write_ptr = result.bytes.data();
        std::memcpy(write_ptr, &result.count, sizeof(result.count));
        write_ptr += sizeof(result.count);

        if (group.instances.empty()) {
            return result;
        }

        result.position_min = group.instances.front().position;
        result.position_max = group.instances.front().position;
        result.rotation_min = group.instances.front().rotation;
        result.rotation_max = group.instances.front().rotation;
        result.scale_min = group.instances.front().scale;
        result.scale_max = group.instances.front().scale;

        bool can_pack = true;
        for (const auto& inst : group.instances) {
            updateBounds(result.position_min, result.position_max, inst.position);
            updateBounds(result.rotation_min, result.rotation_max, inst.rotation);
            updateBounds(result.scale_min, result.scale_max, inst.scale);
            if (inst.source_index < 0 || inst.source_index > 65535) {
                can_pack = false;
            }
        }

        if (!can_pack) {
            result.codec = kInstanceBinaryCodecRaw;
            result.bytes.resize(sizeof(uint32_t) + static_cast<size_t>(result.count) * instanceRecordSize());
            write_ptr = result.bytes.data();
            std::memcpy(write_ptr, &result.count, sizeof(result.count));
            write_ptr += sizeof(result.count);
            for (const auto& inst : group.instances) {
                std::memcpy(write_ptr, &inst.position, sizeof(Vec3));
                write_ptr += sizeof(Vec3);
                std::memcpy(write_ptr, &inst.rotation, sizeof(Vec3));
                write_ptr += sizeof(Vec3);
                std::memcpy(write_ptr, &inst.scale, sizeof(Vec3));
                write_ptr += sizeof(Vec3);
                const int32_t source_index = static_cast<int32_t>(inst.source_index);
                std::memcpy(write_ptr, &source_index, sizeof(source_index));
                write_ptr += sizeof(source_index);
            }
            return result;
        }

        for (const auto& inst : group.instances) {
            const uint16_t px = quantizeFloatToU16(inst.position.x, result.position_min.x, result.position_max.x);
            const uint16_t py = quantizeFloatToU16(inst.position.y, result.position_min.y, result.position_max.y);
            const uint16_t pz = quantizeFloatToU16(inst.position.z, result.position_min.z, result.position_max.z);
            const uint16_t rx = quantizeFloatToU16(inst.rotation.x, result.rotation_min.x, result.rotation_max.x);
            const uint16_t ry = quantizeFloatToU16(inst.rotation.y, result.rotation_min.y, result.rotation_max.y);
            const uint16_t rz = quantizeFloatToU16(inst.rotation.z, result.rotation_min.z, result.rotation_max.z);
            const uint16_t sx = quantizeFloatToU16(inst.scale.x, result.scale_min.x, result.scale_max.x);
            const uint16_t sy = quantizeFloatToU16(inst.scale.y, result.scale_min.y, result.scale_max.y);
            const uint16_t sz = quantizeFloatToU16(inst.scale.z, result.scale_min.z, result.scale_max.z);
            const uint16_t source_index = static_cast<uint16_t>(inst.source_index);

            std::memcpy(write_ptr, &px, sizeof(px)); write_ptr += sizeof(px);
            std::memcpy(write_ptr, &py, sizeof(py)); write_ptr += sizeof(py);
            std::memcpy(write_ptr, &pz, sizeof(pz)); write_ptr += sizeof(pz);
            std::memcpy(write_ptr, &rx, sizeof(rx)); write_ptr += sizeof(rx);
            std::memcpy(write_ptr, &ry, sizeof(ry)); write_ptr += sizeof(ry);
            std::memcpy(write_ptr, &rz, sizeof(rz)); write_ptr += sizeof(rz);
            std::memcpy(write_ptr, &sx, sizeof(sx)); write_ptr += sizeof(sx);
            std::memcpy(write_ptr, &sy, sizeof(sy)); write_ptr += sizeof(sy);
            std::memcpy(write_ptr, &sz, sizeof(sz)); write_ptr += sizeof(sz);
            std::memcpy(write_ptr, &source_index, sizeof(source_index)); write_ptr += sizeof(source_index);
        }

        return result;
    }

    bool decodeRawInstanceBlobToGroup(const std::vector<char>& raw_bytes, InstanceGroup& group) {
        if (raw_bytes.size() < sizeof(uint32_t)) {
            return false;
        }

        const char* read_ptr = raw_bytes.data();
        uint32_t count = 0;
        std::memcpy(&count, read_ptr, sizeof(count));
        read_ptr += sizeof(count);

        const size_t expected_size = sizeof(uint32_t) + static_cast<size_t>(count) * instanceRecordSize();
        if (raw_bytes.size() < expected_size) {
            return false;
        }

        group.instances.clear();
        group.instances.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            InstanceTransform inst;
            std::memcpy(&inst.position, read_ptr, sizeof(Vec3));
            read_ptr += sizeof(Vec3);
            std::memcpy(&inst.rotation, read_ptr, sizeof(Vec3));
            read_ptr += sizeof(Vec3);
            std::memcpy(&inst.scale, read_ptr, sizeof(Vec3));
            read_ptr += sizeof(Vec3);
            int32_t source_index = 0;
            std::memcpy(&source_index, read_ptr, sizeof(source_index));
            read_ptr += sizeof(source_index);
            inst.source_index = static_cast<int>(source_index);
            group.instances.push_back(inst);
        }

        group.initial_instances = group.instances;
        return true;
    }

    bool decodePacked16InstanceBlobToGroup(const PendingInstanceBlobRead& pending, InstanceGroup& group) {
        if (pending.bytes.size() < sizeof(uint32_t)) {
            return false;
        }

        const char* read_ptr = pending.bytes.data();
        uint32_t count = 0;
        std::memcpy(&count, read_ptr, sizeof(count));
        read_ptr += sizeof(count);

        const size_t expected_size = sizeof(uint32_t) + static_cast<size_t>(count) * packed16RecordSize();
        if (pending.bytes.size() < expected_size) {
            return false;
        }

        group.instances.clear();
        group.instances.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            uint16_t px = 0, py = 0, pz = 0;
            uint16_t rx = 0, ry = 0, rz = 0;
            uint16_t sx = 0, sy = 0, sz = 0;
            uint16_t source_index = 0;

            std::memcpy(&px, read_ptr, sizeof(px)); read_ptr += sizeof(px);
            std::memcpy(&py, read_ptr, sizeof(py)); read_ptr += sizeof(py);
            std::memcpy(&pz, read_ptr, sizeof(pz)); read_ptr += sizeof(pz);
            std::memcpy(&rx, read_ptr, sizeof(rx)); read_ptr += sizeof(rx);
            std::memcpy(&ry, read_ptr, sizeof(ry)); read_ptr += sizeof(ry);
            std::memcpy(&rz, read_ptr, sizeof(rz)); read_ptr += sizeof(rz);
            std::memcpy(&sx, read_ptr, sizeof(sx)); read_ptr += sizeof(sx);
            std::memcpy(&sy, read_ptr, sizeof(sy)); read_ptr += sizeof(sy);
            std::memcpy(&sz, read_ptr, sizeof(sz)); read_ptr += sizeof(sz);
            std::memcpy(&source_index, read_ptr, sizeof(source_index)); read_ptr += sizeof(source_index);

            InstanceTransform inst;
            inst.position = Vec3(
                dequantizeU16ToFloat(px, pending.position_min.x, pending.position_max.x),
                dequantizeU16ToFloat(py, pending.position_min.y, pending.position_max.y),
                dequantizeU16ToFloat(pz, pending.position_min.z, pending.position_max.z));
            inst.rotation = Vec3(
                dequantizeU16ToFloat(rx, pending.rotation_min.x, pending.rotation_max.x),
                dequantizeU16ToFloat(ry, pending.rotation_min.y, pending.rotation_max.y),
                dequantizeU16ToFloat(rz, pending.rotation_min.z, pending.rotation_max.z));
            inst.scale = Vec3(
                dequantizeU16ToFloat(sx, pending.scale_min.x, pending.scale_max.x),
                dequantizeU16ToFloat(sy, pending.scale_min.y, pending.scale_max.y),
                dequantizeU16ToFloat(sz, pending.scale_min.z, pending.scale_max.z));
            inst.source_index = static_cast<int>(source_index);
            group.instances.push_back(inst);
        }

        group.initial_instances = group.instances;
        return true;
    }
}

json InstanceManager::serialize(std::ostream* binaryOut) {
    json root = json::array();

    std::vector<PreparedInstanceBlob> prepared_blobs;
    if (binaryOut) {
        std::vector<std::future<PreparedInstanceBlob>> futures;
        futures.reserve(groups.size());
        for (const auto& group : groups) {
            if (group.transient) continue;  // particle render bridges are never saved
            const InstanceGroup* group_ptr = &group;
            futures.push_back(std::async(std::launch::async, [group_ptr]() {
                return prepareInstanceBlob(*group_ptr);
            }));
        }

        prepared_blobs.reserve(groups.size());
        for (auto& future : futures) {
            prepared_blobs.push_back(future.get());
        }
    }
    
    size_t blob_idx = 0;  // index into prepared_blobs (skips transient groups)
    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
        const auto& group = groups[group_idx];
        // Transient groups (e.g. particle render bridges) are runtime-only and
        // rebuilt every frame; never persist them.
        if (group.transient) continue;
        json j_group;
        j_group["id"] = group.id;
        j_group["name"] = group.name;
        
        // Group Settings
        // Note: Using correct member names from InstanceGroup.h
        j_group["settings"] = {
            {"density", group.brush_settings.density},
            {"target_count", group.brush_settings.target_count},
            {"seed", group.brush_settings.seed},
            {"min_distance", group.brush_settings.min_distance},
            {"scale_min", group.brush_settings.scale_min},
            {"scale_max", group.brush_settings.scale_max},
            {"rotation_random_y", group.brush_settings.rotation_random_y},
            {"rotation_random_xz", group.brush_settings.rotation_random_xz},
            {"y_offset_min", group.brush_settings.y_offset_min},
            {"y_offset_max", group.brush_settings.y_offset_max},
            {"align_to_normal", group.brush_settings.align_to_normal},
            {"normal_influence", group.brush_settings.normal_influence},
            {"splat_map_channel", group.brush_settings.splat_map_channel},
            {"exclusion_channel", group.brush_settings.exclusion_channel},
            {"exclusion_threshold", group.brush_settings.exclusion_threshold},
            {"slope_max", group.brush_settings.slope_max},
            {"height_min", group.brush_settings.height_min},
            {"height_max", group.brush_settings.height_max},
            {"curvature_min", group.brush_settings.curvature_min},
            {"curvature_max", group.brush_settings.curvature_max},
            {"curvature_step", group.brush_settings.curvature_step},
            {"allow_ridges", group.brush_settings.allow_ridges},
            {"allow_flats", group.brush_settings.allow_flats},
            {"allow_gullies", group.brush_settings.allow_gullies},
            {"slope_direction_angle", group.brush_settings.slope_direction_angle},
            {"slope_direction_influence", group.brush_settings.slope_direction_influence},
            {"use_global_settings", group.brush_settings.use_global_settings},
            {"density_mask_attribute", group.brush_settings.density_mask_attribute},
            {"scale_mask_attribute", group.brush_settings.scale_mask_attribute},
            {"scale_mask_influence", group.brush_settings.scale_mask_influence}
        };

        // Wind Settings
        j_group["wind"] = {
            {"enabled", group.wind_settings.enabled},
            {"speed", group.wind_settings.speed},
            {"strength", group.wind_settings.strength},
            {"direction", {group.wind_settings.direction.x, group.wind_settings.direction.y, group.wind_settings.direction.z}},
            {"turbulence", group.wind_settings.turbulence},
            {"wave_size", group.wind_settings.wave_size},
            {"use_source_profiles", group.wind_settings.use_source_profiles},
            {"allow_gpu_deform", group.wind_settings.allow_gpu_deform},
            {"gpu_deform_max_distance", group.wind_settings.gpu_deform_max_distance},
            {"gpu_deform_max_instances", group.wind_settings.gpu_deform_max_instances}
        };
        
        // Source Settings
        json j_sources = json::array();
        for (const auto& src : group.sources) {
            json j_src;
            j_src["name"] = src.name;
            j_src["weight"] = src.weight;
            j_src["settings"] = {
                {"scale_min", src.settings.scale_min},
                {"scale_max", src.settings.scale_max},
                {"rotation_random_y", src.settings.rotation_random_y},
                {"rotation_random_xz", src.settings.rotation_random_xz},
                {"y_offset_min", src.settings.y_offset_min},
                {"y_offset_max", src.settings.y_offset_max},
                {"align_to_normal", src.settings.align_to_normal},
                {"normal_influence", src.settings.normal_influence},
                {"wind_strength_scale", src.settings.wind_strength_scale},
                {"wind_speed_scale", src.settings.wind_speed_scale},
                {"wind_turbulence_scale", src.settings.wind_turbulence_scale},
                {"wind_bend_limit_scale", src.settings.wind_bend_limit_scale},
                {"wind_phase_offset", src.settings.wind_phase_offset}
            };
            j_sources.push_back(j_src);
        }
        j_group["sources"] = j_sources;
        
        // Instances
        if (binaryOut) {
            j_group["instances_storage"] = "binary";
            const std::streampos startPos = binaryOut->tellp();
            j_group["instances_binary_offset"] = static_cast<long long>(startPos);
            const auto& blob = prepared_blobs[blob_idx++];
            j_group["instances_count"] = blob.count;
            j_group["instances_binary_codec"] = blob.codec;
            if (blob.codec == kInstanceBinaryCodecPacked16) {
                j_group["instances_position_min"] = vec3ToJsonArray(blob.position_min);
                j_group["instances_position_max"] = vec3ToJsonArray(blob.position_max);
                j_group["instances_rotation_min"] = vec3ToJsonArray(blob.rotation_min);
                j_group["instances_rotation_max"] = vec3ToJsonArray(blob.rotation_max);
                j_group["instances_scale_min"] = vec3ToJsonArray(blob.scale_min);
                j_group["instances_scale_max"] = vec3ToJsonArray(blob.scale_max);
            }

            if (!blob.bytes.empty()) {
                binaryOut->write(blob.bytes.data(), static_cast<std::streamsize>(blob.bytes.size()));
            }

            const std::streampos endPos = binaryOut->tellp();
            j_group["instances_binary_size"] = static_cast<long long>(endPos - startPos);
        } else {
            json j_insts = json::array();
            for (const auto& inst : group.instances) {
                // Save as compact array [x,y,z, rx,ry,rz, sx,sy,sz, src_idx]
                // Note: Rotation is Euler Vec3 (x,y,z), NOT Quaternion
                j_insts.push_back({
                    inst.position.x, inst.position.y, inst.position.z,
                    inst.rotation.x, inst.rotation.y, inst.rotation.z,
                    inst.scale.x, inst.scale.y, inst.scale.z,
                    inst.source_index
                });
            }
            j_group["instances"] = j_insts;
        }
        root.push_back(j_group);
    }
    
    return root;
}

void InstanceManager::deserializeBinaryInstances(simdjson::dom::element el, std::istream& binaryIn) {
    simdjson::dom::array arr;
    if (el.get_array().get(arr)) return;

    std::vector<PendingInstanceBlobRead> pending_reads;
    pending_reads.reserve(groups.size());

    size_t group_index = 0;
    for (simdjson::dom::element j_group : arr) {
        if (group_index >= groups.size()) break;

        std::string storage = "json";
        std::string_view storage_sv;
        if (!j_group["instances_storage"].get(storage_sv)) {
            storage = std::string(storage_sv);
        }
        if (storage != "binary") {
            ++group_index;
            continue;
        }

        int64_t offset_value = 0;
        simdjson::dom::element offset_el;
        if (j_group["instances_binary_offset"].get(offset_el) || offset_el.get(offset_value)) {
            ++group_index;
            continue;
        }

        int64_t size_value = 0;
        simdjson::dom::element size_el;
        if (j_group["instances_binary_size"].get(size_el) || size_el.get(size_value) || size_value < 0) {
            ++group_index;
            continue;
        }

        std::string codec = kInstanceBinaryCodecRaw;
        std::string_view codec_sv;
        if (!j_group["instances_binary_codec"].get(codec_sv)) {
            codec = std::string(codec_sv);
        }

        binaryIn.clear();
        binaryIn.seekg(static_cast<std::streamoff>(offset_value), std::ios::beg);
        if (!binaryIn.good()) {
            ++group_index;
            continue;
        }

        PendingInstanceBlobRead pending;
        pending.group_index = group_index;
        pending.codec = codec;
        if (codec == kInstanceBinaryCodecPacked16) {
            readVec3JsonArray(j_group["instances_position_min"], pending.position_min);
            readVec3JsonArray(j_group["instances_position_max"], pending.position_max);
            readVec3JsonArray(j_group["instances_rotation_min"], pending.rotation_min);
            readVec3JsonArray(j_group["instances_rotation_max"], pending.rotation_max);
            readVec3JsonArray(j_group["instances_scale_min"], pending.scale_min);
            readVec3JsonArray(j_group["instances_scale_max"], pending.scale_max);
        }
        pending.bytes.resize(static_cast<size_t>(size_value));
        if (!pending.bytes.empty()) {
            binaryIn.read(pending.bytes.data(), static_cast<std::streamsize>(pending.bytes.size()));
        }
        if (!binaryIn.good() && !pending.bytes.empty()) {
            ++group_index;
            continue;
        }
        pending_reads.push_back(std::move(pending));
        ++group_index;
    }

    std::vector<std::future<void>> futures;
    futures.reserve(pending_reads.size());
    for (auto& pending : pending_reads) {
        PendingInstanceBlobRead* pending_ptr = &pending;
        futures.push_back(std::async(std::launch::async, [this, pending_ptr]() {
            if (pending_ptr->codec == kInstanceBinaryCodecPacked16) {
                decodePacked16InstanceBlobToGroup(*pending_ptr, groups[pending_ptr->group_index]);
            } else {
                decodeRawInstanceBlobToGroup(pending_ptr->bytes, groups[pending_ptr->group_index]);
            }
        }));
    }

    for (auto& future : futures) {
        future.get();
    }
}

#include "scene_ui.h" 

void InstanceManager::deserialize(const json& j, SceneData& scene) {
    clearAll();
    
    if (!j.is_array()) return;
    
    for (const auto& j_group : j) {
        InstanceGroup group;
        group.id = j_group.value("id", next_group_id++);
        group.name = j_group.value("name", "Foliage Layer");
        if (group.id >= next_group_id) next_group_id = group.id + 1;
        
        // Settings
        if (j_group.contains("settings")) {
            auto& s = j_group["settings"];
            group.brush_settings.density = s.value("density", 1.0f);
            group.brush_settings.target_count = s.value("target_count", 1000);
            group.brush_settings.seed = s.value("seed", 1234);
            group.brush_settings.min_distance = s.value("min_distance", 0.5f);
            
            group.brush_settings.scale_min = s.value("scale_min", 0.8f);
            group.brush_settings.scale_max = s.value("scale_max", 1.2f);
            group.brush_settings.rotation_random_y = s.value("rotation_random_y", 360.0f);
            group.brush_settings.rotation_random_xz = s.value("rotation_random_xz", 15.0f);
            group.brush_settings.y_offset_min = s.value("y_offset_min", 0.0f);
            group.brush_settings.y_offset_max = s.value("y_offset_max", 0.0f);
            group.brush_settings.align_to_normal = s.value("align_to_normal", true);
            group.brush_settings.normal_influence = s.value("normal_influence", 1.0f);

            group.brush_settings.splat_map_channel = s.value("splat_map_channel", -1);
            group.brush_settings.exclusion_channel = s.value("exclusion_channel", -1);
            group.brush_settings.exclusion_threshold = s.value("exclusion_threshold", 0.5f);
            group.brush_settings.slope_max = s.value("slope_max", 45.0f);
            group.brush_settings.height_min = s.value("height_min", -10.0f);
            group.brush_settings.height_max = s.value("height_max", 10.0f);
            group.brush_settings.curvature_min = s.value("curvature_min", -2.0f);
            group.brush_settings.curvature_max = s.value("curvature_max", 2.0f);
            group.brush_settings.curvature_step = s.value("curvature_step", 1);
            group.brush_settings.allow_ridges = s.value("allow_ridges", true);
            group.brush_settings.allow_flats = s.value("allow_flats", true);
            group.brush_settings.allow_gullies = s.value("allow_gullies", true);
            group.brush_settings.slope_direction_angle = s.value("slope_direction_angle", 0.0f);
            group.brush_settings.slope_direction_influence = s.value("slope_direction_influence", 0.0f);
            group.brush_settings.use_global_settings = s.value("use_global_settings", false);
            group.brush_settings.density_mask_attribute = s.value("density_mask_attribute", s.value("mask_attribute", std::string()));
            group.brush_settings.scale_mask_attribute = s.value("scale_mask_attribute", s.value("mask_attribute", std::string()));
            group.brush_settings.scale_mask_influence = s.value("scale_mask_influence", s.value("mask_scale_influence", 1.0f));
        }

        // Wind Settings
        if (j_group.contains("wind")) {
            auto& w = j_group["wind"];
            group.wind_settings.enabled = w.value("enabled", false);
            group.wind_settings.speed = w.value("speed", 1.0f);
            group.wind_settings.strength = w.value("strength", 0.1f);
            if (w.contains("direction")) {
                group.wind_settings.direction = Vec3(w["direction"][0], w["direction"][1], w["direction"][2]);
            }
            group.wind_settings.turbulence = w.value("turbulence", 1.5f);
            group.wind_settings.wave_size = w.value("wave_size", 50.0f);
            group.wind_settings.use_source_profiles = w.value("use_source_profiles", true);
            group.wind_settings.allow_gpu_deform = w.value("allow_gpu_deform", true);
            group.wind_settings.gpu_deform_max_distance = w.value("gpu_deform_max_distance", 35.0f);
            group.wind_settings.gpu_deform_max_instances = w.value("gpu_deform_max_instances", 32);
        }
        
        // Sources
        if (j_group.contains("sources")) {
            for (const auto& j_src : j_group["sources"]) {
                ScatterSource src;
                src.name = j_src.value("name", "");
                src.weight = j_src.value("weight", 1.0f);
                
                if (j_src.contains("settings")) {
                    auto& s = j_src["settings"];
                    src.settings.scale_min = s.value("scale_min", 0.8f);
                    src.settings.scale_max = s.value("scale_max", 1.2f);
                    src.settings.rotation_random_y = s.value("rotation_random_y", 360.0f);
                    src.settings.rotation_random_xz = s.value("rotation_random_xz", 5.0f); // Note: correct member name
                    src.settings.y_offset_min = s.value("y_offset_min", 0.0f);
                    src.settings.y_offset_max = s.value("y_offset_max", 0.0f);
                    src.settings.align_to_normal = s.value("align_to_normal", true);
                    src.settings.normal_influence = s.value("normal_influence", 0.8f);
                    src.settings.wind_strength_scale = s.value("wind_strength_scale", 1.0f);
                    src.settings.wind_speed_scale = s.value("wind_speed_scale", 1.0f);
                    src.settings.wind_turbulence_scale = s.value("wind_turbulence_scale", 1.0f);
                    src.settings.wind_bend_limit_scale = s.value("wind_bend_limit_scale", 1.0f);
                    src.settings.wind_phase_offset = s.value("wind_phase_offset", 0.0f);
                }
                
                // Re-link triangles from Scene (flat-aware: a scatter source may now be ONE flat
                // SoA TriangleMesh, not a per-face facade soup — materialize its facades).
                for (const auto& obj : scene.world.objects) {
                    if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                        if (tri->getNodeName() == src.name) src.triangles.push_back(tri);
                    } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                        if (tm->nodeName == src.name && tm->geometry) {
                            const size_t nTris = tm->num_triangles();
                            for (size_t t = 0; t < nTris; ++t)
                                src.triangles.push_back(std::make_shared<Triangle>(tm, static_cast<uint32_t>(t)));
                        }
                    }
                }
                src.computeCenter();
                group.sources.push_back(src);
            }
        }
        
        // Instances
        if (j_group.contains("instances")) {
            const auto& j_insts = j_group["instances"];
            group.instances.reserve(j_insts.size());
            
            for (const auto& j_i : j_insts) {
                if (j_i.is_array() && j_i.size() >= 10) { // Expect 10 floats now
                    InstanceTransform inst;
                    inst.position = Vec3(j_i[0], j_i[1], j_i[2]);
                    inst.rotation = Vec3(j_i[3], j_i[4], j_i[5]); // Euler
                    inst.scale = Vec3(j_i[6], j_i[7], j_i[8]);
                    inst.source_index = j_i[9];
                    group.instances.push_back(inst);
                }
            }
            // Populate backup for wind/reset
            group.initial_instances = group.instances;
        }
        groups.push_back(group);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FAST SIMDJSON DESERIALIZE — avoids sjsonToNlohmann round-trip
// ═══════════════════════════════════════════════════════════════════════════════

static double sj_double(simdjson::dom::element el, double def = 0.0) {
    double v; return el.get(v) ? def : v;
}
static bool sj_bool(simdjson::dom::element el, bool def = false) {
    bool v; return el.get(v) ? def : v;
}
static int64_t sj_int(simdjson::dom::element el, int64_t def = 0) {
    int64_t v; return el.get(v) ? def : v;
}
static std::string sj_str(simdjson::dom::element el, const char* def = "") {
    std::string_view sv; return el.get(sv) ? std::string(def) : std::string(sv);
}

void InstanceManager::deserializeFast(simdjson::dom::element el, SceneData& scene) {
    clearAll();

    simdjson::dom::array arr;
    if (el.get_array().get(arr)) return;

    // Build a name→triangles lookup once instead of scanning scene.world.objects per source.
    // Flat migration: a scatter source mesh may now live in world.objects as ONE flat SoA
    // TriangleMesh (open-flat collapse / emit_flat) rather than a per-face Triangle facade soup.
    // Record those by name too and materialize their facades ON DEMAND only for names that are
    // actually referenced as a scatter source below — so a 2M-tri flat mesh that is NOT a source
    // never pays the materialize. Without this, the source re-link (tri_by_name lookup) misses the
    // flat mesh and the whole foliage layer loads with zero source triangles → nothing renders.
    // flat_by_name maps to a VECTOR of TriangleMesh, not one — a multi-material import
    // gives every material's TriangleMesh the SAME nodeName by design (one logical
    // object; see [[bugfix_multimaterial_import_nodename_collision]]). A single-value
    // map here overwrote all but the last material, so a reloaded multi-material
    // scatter source only ever got one material's worth of geometry back — "her
    // materyal tek üçgenle dönüyor" (user report, 2026-07-03): each OTHER material
    // that used to resolve here fell through to returning nothing at all, and any
    // caller treating a near-empty facade list as "no source" (or one stray leftover
    // representative facade elsewhere) is what read as "single triangle".
    std::unordered_map<std::string, std::vector<std::shared_ptr<Triangle>>> tri_by_name;
    std::unordered_map<std::string, std::vector<std::shared_ptr<TriangleMesh>>> flat_by_name;
    for (const auto& obj : scene.world.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            tri_by_name[tri->getNodeName()].push_back(tri);
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (!tm->nodeName.empty()) flat_by_name[tm->nodeName].push_back(tm);
        }
    }
    auto resolveSourceTriangles =
        [&tri_by_name, &flat_by_name](const std::string& name) -> std::vector<std::shared_ptr<Triangle>> {
        auto it = tri_by_name.find(name);
        if (it != tri_by_name.end()) return it->second;
        auto fit = flat_by_name.find(name);
        if (fit != flat_by_name.end() && !fit->second.empty()) {
            std::vector<std::shared_ptr<Triangle>> facades;
            for (const auto& tm : fit->second) {
                if (!tm || !tm->geometry) continue;
                const size_t nTris = tm->num_triangles();
                facades.reserve(facades.size() + nTris);
                for (size_t t = 0; t < nTris; ++t)
                    facades.push_back(std::make_shared<Triangle>(tm, static_cast<uint32_t>(t)));
            }
            if (!facades.empty()) {
                tri_by_name[name] = facades; // cache so repeated source refs reuse the same facades
                return facades;
            }
        }
        return {};
    };

    for (simdjson::dom::element j_group : arr) {
        InstanceGroup group;
        { simdjson::dom::element v; if (!j_group["id"].get(v)) group.id = (int)sj_int(v); else group.id = next_group_id++; }
        { simdjson::dom::element v; if (!j_group["name"].get(v)) group.name = sj_str(v, "Foliage Layer"); else group.name = "Foliage Layer"; }
        if (group.id >= next_group_id) next_group_id = group.id + 1;

        // Settings
        simdjson::dom::element s_el;
        if (!j_group["settings"].get(s_el)) {
            auto& bs = group.brush_settings;
            simdjson::dom::element v;
            if (!s_el["density"].get(v)) bs.density = (float)sj_double(v, 1.0);
            if (!s_el["target_count"].get(v)) bs.target_count = (int)sj_int(v, 1000);
            if (!s_el["seed"].get(v)) bs.seed = (int)sj_int(v, 1234);
            if (!s_el["min_distance"].get(v)) bs.min_distance = (float)sj_double(v, 0.5);
            if (!s_el["scale_min"].get(v)) bs.scale_min = (float)sj_double(v, 0.8);
            if (!s_el["scale_max"].get(v)) bs.scale_max = (float)sj_double(v, 1.2);
            if (!s_el["rotation_random_y"].get(v)) bs.rotation_random_y = (float)sj_double(v, 360.0);
            if (!s_el["rotation_random_xz"].get(v)) bs.rotation_random_xz = (float)sj_double(v, 15.0);
            if (!s_el["y_offset_min"].get(v)) bs.y_offset_min = (float)sj_double(v, 0.0);
            if (!s_el["y_offset_max"].get(v)) bs.y_offset_max = (float)sj_double(v, 0.0);
            if (!s_el["align_to_normal"].get(v)) bs.align_to_normal = sj_bool(v, true);
            if (!s_el["normal_influence"].get(v)) bs.normal_influence = (float)sj_double(v, 1.0);
            if (!s_el["splat_map_channel"].get(v)) bs.splat_map_channel = (int)sj_int(v, -1);
            if (!s_el["exclusion_channel"].get(v)) bs.exclusion_channel = (int)sj_int(v, -1);
            if (!s_el["exclusion_threshold"].get(v)) bs.exclusion_threshold = (float)sj_double(v, 0.5);
            if (!s_el["slope_max"].get(v)) bs.slope_max = (float)sj_double(v, 45.0);
            if (!s_el["height_min"].get(v)) bs.height_min = (float)sj_double(v, -10.0);
            if (!s_el["height_max"].get(v)) bs.height_max = (float)sj_double(v, 10.0);
            if (!s_el["curvature_min"].get(v)) bs.curvature_min = (float)sj_double(v, -2.0);
            if (!s_el["curvature_max"].get(v)) bs.curvature_max = (float)sj_double(v, 2.0);
            if (!s_el["curvature_step"].get(v)) bs.curvature_step = (int)sj_int(v, 1);
            if (!s_el["allow_ridges"].get(v)) bs.allow_ridges = sj_bool(v, true);
            if (!s_el["allow_flats"].get(v)) bs.allow_flats = sj_bool(v, true);
            if (!s_el["allow_gullies"].get(v)) bs.allow_gullies = sj_bool(v, true);
            if (!s_el["slope_direction_angle"].get(v)) bs.slope_direction_angle = (float)sj_double(v, 0.0);
            if (!s_el["slope_direction_influence"].get(v)) bs.slope_direction_influence = (float)sj_double(v, 0.0);
            if (!s_el["use_global_settings"].get(v)) bs.use_global_settings = sj_bool(v, false);
            { simdjson::dom::element mv;
              if (!s_el["density_mask_attribute"].get(mv)) bs.density_mask_attribute = sj_str(mv);
              else if (!s_el["mask_attribute"].get(mv)) bs.density_mask_attribute = sj_str(mv); }
            { simdjson::dom::element mv;
              if (!s_el["scale_mask_attribute"].get(mv)) bs.scale_mask_attribute = sj_str(mv);
              else if (!s_el["mask_attribute"].get(mv)) bs.scale_mask_attribute = sj_str(mv); }
            if (!s_el["scale_mask_influence"].get(v)) bs.scale_mask_influence = (float)sj_double(v, 1.0);
            else if (!s_el["mask_scale_influence"].get(v)) bs.scale_mask_influence = (float)sj_double(v, 0.0);
        }

        // Wind Settings
        simdjson::dom::element w_el;
        if (!j_group["wind"].get(w_el)) {
            auto& ws = group.wind_settings;
            simdjson::dom::element v;
            if (!w_el["enabled"].get(v)) ws.enabled = sj_bool(v, false);
            if (!w_el["speed"].get(v)) ws.speed = (float)sj_double(v, 1.0);
            if (!w_el["strength"].get(v)) ws.strength = (float)sj_double(v, 0.1);
            simdjson::dom::array dir_arr;
            if (!w_el["direction"].get(dir_arr) && dir_arr.size() >= 3) {
                auto it = dir_arr.begin();
                float dx = (float)sj_double(*it); ++it;
                float dy = (float)sj_double(*it); ++it;
                float dz = (float)sj_double(*it);
                ws.direction = Vec3(dx, dy, dz);
            }
            if (!w_el["turbulence"].get(v)) ws.turbulence = (float)sj_double(v, 1.5);
            if (!w_el["wave_size"].get(v)) ws.wave_size = (float)sj_double(v, 50.0);
            if (!w_el["use_source_profiles"].get(v)) ws.use_source_profiles = sj_bool(v, true);
            if (!w_el["allow_gpu_deform"].get(v)) ws.allow_gpu_deform = sj_bool(v, true);
            if (!w_el["gpu_deform_max_distance"].get(v)) ws.gpu_deform_max_distance = (float)sj_double(v, 35.0);
            if (!w_el["gpu_deform_max_instances"].get(v)) ws.gpu_deform_max_instances = (int)sj_int(v, 32);
        }

        // Sources
        simdjson::dom::array src_arr;
        if (!j_group["sources"].get(src_arr)) {
            for (simdjson::dom::element j_src : src_arr) {
                ScatterSource src;
                { simdjson::dom::element v; if (!j_src["name"].get(v)) src.name = sj_str(v); }
                { simdjson::dom::element v; if (!j_src["weight"].get(v)) src.weight = (float)sj_double(v, 1.0); }

                simdjson::dom::element ss_el;
                if (!j_src["settings"].get(ss_el)) {
                    auto& ss = src.settings;
                    simdjson::dom::element v;
                    if (!ss_el["scale_min"].get(v)) ss.scale_min = (float)sj_double(v, 0.8);
                    if (!ss_el["scale_max"].get(v)) ss.scale_max = (float)sj_double(v, 1.2);
                    if (!ss_el["rotation_random_y"].get(v)) ss.rotation_random_y = (float)sj_double(v, 360.0);
                    if (!ss_el["rotation_random_xz"].get(v)) ss.rotation_random_xz = (float)sj_double(v, 5.0);
                    if (!ss_el["y_offset_min"].get(v)) ss.y_offset_min = (float)sj_double(v, 0.0);
                    if (!ss_el["y_offset_max"].get(v)) ss.y_offset_max = (float)sj_double(v, 0.0);
                    if (!ss_el["align_to_normal"].get(v)) ss.align_to_normal = sj_bool(v, true);
                    if (!ss_el["normal_influence"].get(v)) ss.normal_influence = (float)sj_double(v, 0.8);
                    if (!ss_el["wind_strength_scale"].get(v)) ss.wind_strength_scale = (float)sj_double(v, 1.0);
                    if (!ss_el["wind_speed_scale"].get(v)) ss.wind_speed_scale = (float)sj_double(v, 1.0);
                    if (!ss_el["wind_turbulence_scale"].get(v)) ss.wind_turbulence_scale = (float)sj_double(v, 1.0);
                    if (!ss_el["wind_bend_limit_scale"].get(v)) ss.wind_bend_limit_scale = (float)sj_double(v, 1.0);
                    if (!ss_el["wind_phase_offset"].get(v)) ss.wind_phase_offset = (float)sj_double(v, 0.0);
                }

                // Re-link triangles from scene using pre-built lookup (flat-aware: materializes
                // a flat TriangleMesh source's facades on demand).
                src.triangles = resolveSourceTriangles(src.name);
                src.computeCenter();
                group.sources.push_back(std::move(src));
            }
        }

        // Instances — the hot path. Parse directly from simdjson arrays.
        simdjson::dom::array inst_arr;
        if (!j_group["instances"].get(inst_arr)) {
            group.instances.reserve(inst_arr.size());
            for (simdjson::dom::element j_i : inst_arr) {
                simdjson::dom::array vals;
                if (j_i.get_array().get(vals) || vals.size() < 10) continue;
                InstanceTransform inst;
                auto it = vals.begin();
                inst.position.x = (float)sj_double(*it); ++it;
                inst.position.y = (float)sj_double(*it); ++it;
                inst.position.z = (float)sj_double(*it); ++it;
                inst.rotation.x = (float)sj_double(*it); ++it;
                inst.rotation.y = (float)sj_double(*it); ++it;
                inst.rotation.z = (float)sj_double(*it); ++it;
                inst.scale.x    = (float)sj_double(*it); ++it;
                inst.scale.y    = (float)sj_double(*it); ++it;
                inst.scale.z    = (float)sj_double(*it); ++it;
                inst.source_index = (int)sj_int(*it);
                group.instances.push_back(inst);
            }
            group.initial_instances = group.instances;
        }
        groups.push_back(std::move(group));
    }
}

#include "HittableInstance.h"

void InstanceManager::rebuildSceneObjects(SceneData& scene) {
    if (groups.empty()) return;

    const auto total_start = std::chrono::steady_clock::now();
    size_t total_bvh_build_ms = 0;
    size_t total_spawn_ms = 0;
    size_t total_cleanup_ms = 0;
    size_t total_instances_requested = 0;
    
    std::string instance_prefix = "_inst_"; // Prefix to identify instances

    for (auto& group : groups) {
        // Transient groups (particle render bridges) are GPU-batched scatter owned
        // by the runtime; they must NOT be CPU-expanded into world.objects or have
        // their source triangles re-centered/baked here. Doing so would shift the
        // selection list and corrupt their live geometry.
        if (group.transient) continue;
        const auto group_bvh_start = std::chrono::steady_clock::now();
        // 1. Ensure BVHs are built for all sources
        //
        // Per-source builds are independent (each source owns its own centered
        // triangles, EmbreeBVH and local bbox), so we can run the heavy work in
        // parallel. For foliage-heavy projects the source list typically holds
        // 5-30 unique prefabs (trees, grass, rocks), each with thousands of
        // triangles — serializing them was the dominant cost on project load.
        auto buildOneSource = [](ScatterSource& source) {
            if (source.triangles.empty()) return;

            if (source.bvh) {
                if (!source.has_local_bbox) {
                    source.has_local_bbox = source.bvh->bounding_box(0, 0, source.local_bbox);
                }
                return;
            }

            // Determine Bounds to find Center (Pivot)
            Vec3 mesh_bbox_min(1e9, 1e9, 1e9);
            Vec3 mesh_bbox_max(-1e9, -1e9, -1e9);

            bool has_geo = false;
            for (const auto& src_tri : source.triangles) {
                Matrix4x4 transform = src_tri->getTransformMatrix();

                Vec3 v0 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(0), 1.0f)).xyz();
                Vec3 v1 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(1), 1.0f)).xyz();
                Vec3 v2 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(2), 1.0f)).xyz();

                Vec3 v[3] = { v0, v1, v2 };
                for(int k=0; k<3; k++) {
                    mesh_bbox_min = Vec3::min(mesh_bbox_min, v[k]);
                    mesh_bbox_max = Vec3::max(mesh_bbox_max, v[k]);
                }
                has_geo = true;
            }

            if (!has_geo) return;

            Vec3 mesh_center = (mesh_bbox_min + mesh_bbox_max) * 0.5f;
            mesh_center.y = mesh_bbox_min.y; // Pivot at Bottom-Center (Crucial for foliage)

            auto centered_tris = std::make_shared<std::vector<std::shared_ptr<Triangle>>>();
            centered_tris->reserve(source.triangles.size());

            std::vector<std::shared_ptr<Hittable>> source_hittables;
            source_hittables.reserve(source.triangles.size());

            for (const auto& src_tri : source.triangles) {
                Matrix4x4 transform = src_tri->getTransformMatrix();

                Vec3 v0 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(0), 1.0f)).xyz() - mesh_center;
                Vec3 v1 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(1), 1.0f)).xyz() - mesh_center;
                Vec3 v2 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(2), 1.0f)).xyz() - mesh_center;

                Vec3 n0 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexNormal(0), 0.0f)).xyz().normalize();
                Vec3 n1 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexNormal(1), 0.0f)).xyz().normalize();
                Vec3 n2 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexNormal(2), 0.0f)).xyz().normalize();

                auto new_tri = std::make_shared<Triangle>(
                    v0, v1, v2,
                    n0, n1, n2,
                    src_tri->t_ref(0), src_tri->t_ref(1), src_tri->t_ref(2),
                    src_tri->getMaterial()
                );
                new_tri->setNodeName(source.name + "_BAKED");

                centered_tris->push_back(new_tri);
                source_hittables.push_back(new_tri);
            }

            source.centered_triangles_ptr = centered_tris;

            auto embree = std::make_shared<EmbreeBVH>();
            embree->build(source_hittables);
            source.bvh = embree;
            source.has_local_bbox = source.bvh->bounding_box(0, 0, source.local_bbox);
            SCENE_LOG_INFO("[InstanceManager] Built Centered BVH (Baked Transform) for restored source: " + source.name);
        };

        // Collect sources that actually need heavy work (missing BVH). Sources
        // whose BVH already exists fall back to the quick bbox refresh.
        std::vector<size_t> heavy_source_indices;
        heavy_source_indices.reserve(group.sources.size());
        for (size_t si = 0; si < group.sources.size(); ++si) {
            auto& source = group.sources[si];
            if (source.triangles.empty()) continue;
            if (!source.bvh) {
                heavy_source_indices.push_back(si);
            } else if (!source.has_local_bbox) {
                source.has_local_bbox = source.bvh->bounding_box(0, 0, source.local_bbox);
            }
        }

        if (heavy_source_indices.size() >= 2) {
            std::vector<std::future<void>> source_futures;
            source_futures.reserve(heavy_source_indices.size());
            for (size_t si : heavy_source_indices) {
                source_futures.push_back(std::async(std::launch::async,
                    [&buildOneSource, &group, si]() {
                        buildOneSource(group.sources[si]);
                    }));
            }
            for (auto& f : source_futures) f.get();
        } else {
            for (size_t si : heavy_source_indices) {
                buildOneSource(group.sources[si]);
            }
        }
        total_bvh_build_ms += static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - group_bvh_start).count());
        
        // 2. Spawn Instances (Multi-threaded)
        size_t num_instances = group.instances.size();
        total_instances_requested += num_instances;
        
        // Clear old links
        group.active_hittables.clear();
        group.active_hittables.resize(num_instances);
        
        // Pre-allocate space in scene objects
        size_t start_offset = scene.world.objects.size();
        scene.world.objects.resize(start_offset + num_instances);
        
        // Parallel instance creation
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        size_t chunk_size = (num_instances + num_threads - 1) / num_threads;
        
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);
        const auto spawn_start = std::chrono::steady_clock::now();
        
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, num_instances);
            if (start >= end) continue;
            
            futures.push_back(std::async(std::launch::async, 
                [&, start, end, start_offset, instance_prefix]() {
                    for (size_t i = start; i < end; ++i) {
                        const auto& inst = group.instances[i];
                        
                        int src_idx = inst.source_index;
                        if (src_idx < 0 || src_idx >= (int)group.sources.size()) src_idx = 0;
                        if (group.sources.empty()) continue;
                        
                        auto& source = group.sources[src_idx];
                        if (!source.bvh) continue;
                        
                        Matrix4x4 mat = inst.toMatrix();
                        // USE ID-BASED PREFIX for robust identification and to avoid collision with other layers of same name
                        std::string name = "_inst_gid" + std::to_string(group.id) + "_" + std::to_string(i);

                        std::shared_ptr<HittableInstance> hit_inst;
                        if (source.has_local_bbox) {
                            hit_inst = std::make_shared<HittableInstance>(
                                source.bvh,
                                source.centered_triangles_ptr,
                                mat,
                                name,
                                source.local_bbox
                            );
                        } else {
                            hit_inst = std::make_shared<HittableInstance>(
                                source.bvh,
                                source.centered_triangles_ptr,
                                mat,
                                name
                            );
                        }
                        
                        scene.world.objects[start_offset + i] = hit_inst;
                        group.active_hittables[i] = hit_inst;
                    }
                }));
        }
        
        // Wait for all threads to complete
        for (auto& f : futures) f.get();
        total_spawn_ms += static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - spawn_start).count());
        
        // Remove any failed instances (nullptrs)
        const auto cleanup_start = std::chrono::steady_clock::now();
        auto& objs = scene.world.objects;
        objs.erase(std::remove(objs.begin(), objs.end(), nullptr), objs.end());
        total_cleanup_ms += static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - cleanup_start).count());
    }
    SCENE_LOG_INFO("[InstanceManager] Rebuilt " + std::to_string(scene.world.objects.size()) + " scene objects from foliage data.");
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - total_start).count();
    SCENE_LOG_INFO("[Perf] InstanceManager::rebuildSceneObjects total: " + std::to_string(total_ms) +
                   " ms | requested instances: " + std::to_string(total_instances_requested) +
                   " | source BVH build: " + std::to_string(total_bvh_build_ms) +
                   " ms | instance spawn: " + std::to_string(total_spawn_ms) +
                   " ms | cleanup: " + std::to_string(total_cleanup_ms) + " ms");
}
