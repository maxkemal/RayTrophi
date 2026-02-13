#include "InstanceManager.h"
#include "Triangle.h"
#include "scene_data.h"
#include "EmbreeBVH.h"
#include "OptixWrapper.h"
#include "TerrainManager.h"
#include "globals.h"
#include <random>
#include <cmath>
#include <algorithm>

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

int InstanceManager::paintInstances(int group_id, const Vec3& center, const Vec3& normal,
                                     float brush_radius, float density_multiplier) {
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
            
            if (TerrainManager::getInstance().hasActiveTerrain()) {
                surface_pos.y = TerrainManager::getInstance().sampleHeight(pos.x, pos.z);
                surface_normal = TerrainManager::getInstance().sampleNormal(pos.x, pos.z);
            }

            InstanceTransform t = group->generateRandomTransform(surface_pos, surface_normal);
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

void InstanceManager::updateWind(float time, SceneData& scene) {
    for (auto& group : groups) {
        if (group.wind_settings.enabled) {
            group.updateWind(time);
        }
    }
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
    size_t total = 0;
    for (const auto& group : groups) {
        total += group.instances.size();
    }
    return total;
}

size_t InstanceManager::getTotalTriangleCount() const {
    size_t total = 0;
    for (const auto& group : groups) {
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

json InstanceManager::serialize() {
    json root = json::array();
    
    for (const auto& group : groups) {
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
            {"use_global_settings", group.brush_settings.use_global_settings}
        };

        // Wind Settings
        j_group["wind"] = {
            {"enabled", group.wind_settings.enabled},
            {"speed", group.wind_settings.speed},
            {"strength", group.wind_settings.strength},
            {"direction", {group.wind_settings.direction.x, group.wind_settings.direction.y, group.wind_settings.direction.z}},
            {"turbulence", group.wind_settings.turbulence},
            {"wave_size", group.wind_settings.wave_size}
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
                {"normal_influence", src.settings.normal_influence}
            };
            j_sources.push_back(j_src);
        }
        j_group["sources"] = j_sources;
        
        // Instances
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
        root.push_back(j_group);
    }
    
    return root;
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
                }
                
                // Re-link triangles from Scene
                 for (const auto& obj : scene.world.objects) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (tri && tri->getNodeName() == src.name) {
                        src.triangles.push_back(tri);
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

#include "HittableInstance.h"
// #include "BVHNode.h" // Removed

void InstanceManager::rebuildSceneObjects(SceneData& scene) {
    if (groups.empty()) return;
    
    std::string instance_prefix = "_inst_"; // Prefix to identify instances
    
    for (auto& group : groups) {
        // 1. Ensure BVHs are built for all sources
        for (auto& source : group.sources) {
            if (source.triangles.empty()) continue;
            
            // Build BVH if missing (reloaded)
            if (!source.bvh) {
                // Determine Bounds to find Center (Pivot)
                Vec3 mesh_bbox_min(1e9, 1e9, 1e9);
                Vec3 mesh_bbox_max(-1e9, -1e9, -1e9);
                
                // Calculate bounds from source triangles
                bool has_geo = false;
                for (const auto& src_tri : source.triangles) {
                    Matrix4x4 transform = src_tri->getTransformMatrix();
                    
                    // Transform vertices to World Space (Bake Transform)
                    // Use Original Vertices to avoid Double Transformation
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
                
                if (!has_geo) continue;

                Vec3 mesh_center = (mesh_bbox_min + mesh_bbox_max) * 0.5f;
                mesh_center.y = mesh_bbox_min.y; // Pivot at Bottom-Center (Crucial for foliage)

                // Create Centered Copies (Local Space Prefab)
                auto centered_tris = std::make_shared<std::vector<std::shared_ptr<Triangle>>>();
                centered_tris->reserve(source.triangles.size());
                
                std::vector<std::shared_ptr<Hittable>> source_hittables;
                source_hittables.reserve(source.triangles.size());
                
                for (const auto& src_tri : source.triangles) {
                    Matrix4x4 transform = src_tri->getTransformMatrix();
                    
                    // Convert World (Baked) -> Local (Centered)
                    // Use Original Vertices
                    Vec3 v0 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(0), 1.0f)).xyz() - mesh_center;
                    Vec3 v1 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(1), 1.0f)).xyz() - mesh_center;
                    Vec3 v2 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(2), 1.0f)).xyz() - mesh_center;
                    
                    // Transform Normal (Rotation only)
                    Vec3 n0 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexNormal(0), 0.0f)).xyz().normalize();
                    Vec3 n1 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexNormal(1), 0.0f)).xyz().normalize();
                    Vec3 n2 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexNormal(2), 0.0f)).xyz().normalize();
                    
                    auto new_tri = std::make_shared<Triangle>(
                        v0, v1, v2, 
                        n0, n1, n2,
                        src_tri->t0, src_tri->t1, src_tri->t2,
                        src_tri->getMaterial()
                    );
                    // CRITICAL: Append suffix to prevent OptixWrapper from reusing the BLAS 
                    // of the original Source Mesh (which is unbaked/lying down).
                    // This ensures instances get their own BAKED geometry on GPU.
                    new_tri->setNodeName(source.name + "_BAKED"); 
                    
                    centered_tris->push_back(new_tri);
                    source_hittables.push_back(new_tri);
                }
                
                source.centered_triangles_ptr = centered_tris;
                
                // Build BVH (EmbreeBVH) over LOCAL geometry
                auto embree = std::make_shared<EmbreeBVH>();
                embree->build(source_hittables);
                source.bvh = embree;
                SCENE_LOG_INFO("[InstanceManager] Built Centered BVH (Baked Transform) for restored source: " + source.name);
            }
        }
        
        // 2. Spawn Instances (Multi-threaded)
        size_t num_instances = group.instances.size();
        
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
                        
                        auto hit_inst = std::make_shared<HittableInstance>(
                            source.bvh, 
                            source.centered_triangles_ptr, 
                            mat, 
                            name
                        );
                        
                        scene.world.objects[start_offset + i] = hit_inst;
                        group.active_hittables[i] = hit_inst;
                    }
                }));
        }
        
        // Wait for all threads to complete
        for (auto& f : futures) f.get();
        
        // Remove any failed instances (nullptrs)
        auto& objs = scene.world.objects;
        objs.erase(std::remove(objs.begin(), objs.end(), nullptr), objs.end());
    }
    SCENE_LOG_INFO("[InstanceManager] Rebuilt " + std::to_string(scene.world.objects.size()) + " scene objects from foliage data.");
}
