#include "InstanceGroup.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include "HittableInstance.h" // Added for wind update
#include "TerrainSystem.h"
#include "Texture.h"
#include "Transform.h"
#include <random>
#include <algorithm>
#include <map>
#include <cmath>
#include "globals.h" // For SCENE_LOG_INFO

// ═══════════════════════════════════════════════════════════════════════════════
// SCATTER SOURCE
// ═══════════════════════════════════════════════════════════════════════════════

ScatterSource::ScatterSource(const std::string& n, const std::vector<std::shared_ptr<Triangle>>& tris)
    : name(n), triangles(tris) {
    computeCenter();
}

ScatterSource::ScatterSource(
    const std::string& n,
    const std::vector<std::shared_ptr<TriangleMesh>>& meshes)
    : name(n), flat_meshes(meshes) {
    computeCenter();
}

void ScatterSource::computeCenter() {
    mesh_center = Vec3(0, 0, 0);
    has_local_bbox = false;

    if (!flat_meshes.empty()) {
        Vec3 bboxMin(1e9f, 1e9f, 1e9f);
        Vec3 bboxMax(-1e9f, -1e9f, -1e9f);
        bool hasBounds = false;
        for (const auto& mesh : flat_meshes) {
            if (!mesh) continue;
            AABB bounds;
            if (!mesh->bounding_box(0.0f, 0.0f, bounds)) continue;
            bboxMin = Vec3::min(bboxMin, bounds.min);
            bboxMax = Vec3::max(bboxMax, bounds.max);
            hasBounds = true;
        }
        if (hasBounds) {
            mesh_center = (bboxMin + bboxMax) * 0.5f;
            mesh_center.y = bboxMin.y;
            local_bbox = AABB(bboxMin, bboxMax);
            has_local_bbox = true;
        }
        return;
    }

    int vertex_count = 0;
    Vec3 bboxMin(1e9f, 1e9f, 1e9f);
    Vec3 bboxMax(-1e9f, -1e9f, -1e9f);
    
    for (const auto& tri : triangles) {
        mesh_center = mesh_center + tri->getV0();
        mesh_center = mesh_center + tri->getV1();
        mesh_center = mesh_center + tri->getV2();
        bboxMin = Vec3::min(bboxMin, Vec3::min(tri->getV0(), Vec3::min(tri->getV1(), tri->getV2())));
        bboxMax = Vec3::max(bboxMax, Vec3::max(tri->getV0(), Vec3::max(tri->getV1(), tri->getV2())));
        vertex_count += 3;
    }
    
    if (vertex_count > 0) {
        mesh_center = mesh_center / (float)vertex_count;
        local_bbox = AABB(bboxMin, bboxMax);
        has_local_bbox = true;
    }
}

size_t ScatterSource::sourceTriangleCount() const {
    if (!flat_meshes.empty()) {
        size_t total = 0;
        for (const auto& mesh : flat_meshes) {
            if (mesh) total += mesh->num_triangles();
        }
        return total;
    }
    return triangles.size();
}

// ═══════════════════════════════════════════════════════════════════════════════
// INSTANCE TRANSFORM
// ═══════════════════════════════════════════════════════════════════════════════

Matrix4x4 InstanceTransform::toMatrix() const {
    Matrix4x4 result;
    
    // Build TRS matrix: Translation * Rotation * Scale
    float cx = cosf(rotation.x * 3.14159f / 180.0f);
    float sx = sinf(rotation.x * 3.14159f / 180.0f);
    float cy = cosf(rotation.y * 3.14159f / 180.0f);
    float sy = sinf(rotation.y * 3.14159f / 180.0f);
    float cz = cosf(rotation.z * 3.14159f / 180.0f);
    float sz = sinf(rotation.z * 3.14159f / 180.0f);
    
    // Rotation matrix (Y * X * Z order - typical for games)
    result.m[0][0] = (cy * cz + sy * sx * sz) * scale.x;
    result.m[0][1] = (cz * sy * sx - cy * sz) * scale.x;
    result.m[0][2] = (cx * sy) * scale.x;
    result.m[0][3] = position.x;
    
    result.m[1][0] = (cx * sz) * scale.y;
    result.m[1][1] = (cx * cz) * scale.y;
    result.m[1][2] = (-sx) * scale.y;
    result.m[1][3] = position.y;
    
    result.m[2][0] = (cy * sx * sz - cz * sy) * scale.z;
    result.m[2][1] = (cy * cz * sx + sy * sz) * scale.z;
    result.m[2][2] = (cx * cy) * scale.z;
    result.m[2][3] = position.z;
    
    result.m[3][0] = 0.0f;
    result.m[3][1] = 0.0f;
    result.m[3][2] = 0.0f;
    result.m[3][3] = 1.0f;
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// INSTANCE GROUP
// ═══════════════════════════════════════════════════════════════════════════════

void InstanceGroup::addInstance(const InstanceTransform& transform) {
    instances.push_back(transform);
    initial_instances.push_back(transform); // Store Rest Pose
    gpu_dirty = true;
}

void InstanceGroup::removeInstancesInRadius(const Vec3& center, float radius) {
    float radius_sq = radius * radius;
    
    // Synchronized compaction to keep instances and initial_instances in sync
    size_t write_idx = 0;
    bool has_initial = initial_instances.size() == instances.size();

    for (size_t read_idx = 0; read_idx < instances.size(); ++read_idx) {
        const auto& t = instances[read_idx];
        float dx = t.position.x - center.x;
        float dz = t.position.z - center.z;
        
        // Condition to KEEP: 2D Distance >= Radius (Outside circle)
        // Note: Using 2D check ensures instances with Y-offsets or on slopes are captured correctly.
        if ((dx*dx + dz*dz) >= radius_sq) {
            if (write_idx != read_idx) {
                instances[write_idx] = instances[read_idx];
                if (has_initial) initial_instances[write_idx] = initial_instances[read_idx];
            }
            write_idx++;
        }
    }
    
    if (write_idx < instances.size()) {
        instances.resize(write_idx);
        if (has_initial) initial_instances.resize(write_idx);
        gpu_dirty = true;
    }
}

void InstanceGroup::clearInstances() {
    instances.clear();
    initial_instances.clear();
    tlas_instance_ids.clear();
    gpu_dirty = true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MESH SURFACE SAMPLER
// ═══════════════════════════════════════════════════════════════════════════════

void MeshSurfaceSampler::build(const std::vector<std::shared_ptr<Triangle>>& triangles) {
    source_tris = &triangles;
    cdf.clear();
    total_area = 0.f;
    if (triangles.empty()) return;

    cdf.reserve(triangles.size());
    for (const auto& tri : triangles) {
        if (!tri) { cdf.push_back(total_area); continue; }
        Vec3 e1 = tri->getV1() - tri->getV0();
        Vec3 e2 = tri->getV2() - tri->getV0();
        float area = e1.cross(e2).length() * 0.5f;
        total_area += area;
        cdf.push_back(total_area);
    }

    // Normalise to [0..1] so binary search works with uniform(0,1)
    if (total_area > 0.f)
        for (auto& v : cdf) v /= total_area;
}

MeshSurfaceSampler::Sample MeshSurfaceSampler::sample(std::mt19937& rng) const {
    Sample s;
    if (!source_tris || cdf.empty()) return s;

    std::uniform_real_distribution<float> dist(0.f, 1.f);

    // Pick triangle proportional to its area
    float r = dist(rng);
    int idx = (int)(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
    idx = std::clamp(idx, 0, (int)source_tris->size() - 1);

    const auto& tri = (*source_tris)[idx];
    if (!tri) return s;

    // Uniform random point on triangle (Osada et al. 2002)
    //   P = (1-√r1)*v0 + √r1*(1-r2)*v1 + √r1*r2*v2
    float r1 = sqrtf(dist(rng));
    float r2 = dist(rng);
    float u = 1.f - r1;
    float v = r1 * (1.f - r2);
    float w = r1 * r2;

    Vec3 v0 = tri->getV0();
    Vec3 v1 = tri->getV1();
    Vec3 v2 = tri->getV2();

    s.position = v0 * u + v1 * v + v2 * w;
    s.triangle = tri;
    s.bary_u = u; s.bary_v = v; s.bary_w = w;

    // Face normal
    Vec3 fn = (v1 - v0).cross(v2 - v0);
    float len = fn.length();
    s.normal = (len > 1e-6f) ? fn / len : Vec3(0, 1, 0);

    // Always point toward the upper hemisphere so scatter placement is consistent
    if (s.normal.y < 0.f) s.normal = -s.normal;

    return s;
}

namespace {
    // Faz 8b Field bridge: barycentric-sample a named per-vertex float attribute on a
    // triangle's parent flat mesh. Empty name / non-flat facade (no parentMesh) / attribute
    // not found all resolve to 1.0 (unmasked) - same "missing = no-op" contract as the brush's
    // paintInstances sampler and ScatterInstancesNode.
    float sampleTriangleFieldAttribute(const std::shared_ptr<Triangle>& tri, float bu, float bv, float bw,
                                        const std::string& attrName, float missingValue = 1.0f) {
        if (attrName.empty() || !tri || !tri->parentMesh || !tri->parentMesh->geometry) return missingValue;
        const auto& mgeom = *tri->parentMesh->geometry;
        const float* attr = mgeom.get_attribute_data<float>(attrName);
        const int faceIdx = tri->getFaceIndex();
        if (!attr || faceIdx < 0 || static_cast<size_t>(faceIdx) * 3 + 2 >= mgeom.indices.size()) return missingValue;
        const uint32_t i0 = mgeom.indices[static_cast<size_t>(faceIdx) * 3 + 0];
        const uint32_t i1 = mgeom.indices[static_cast<size_t>(faceIdx) * 3 + 1];
        const uint32_t i2 = mgeom.indices[static_cast<size_t>(faceIdx) * 3 + 2];
        return bu * attr[i0] + bv * attr[i1] + bw * attr[i2];
    }
}

int InstanceGroup::scatterFillMesh(const std::vector<std::shared_ptr<Triangle>>& surfaceTriangles) {
    if (surfaceTriangles.empty()) return 0;

    MeshSurfaceSampler mss;
    mss.build(surfaceTriangles);
    if (!mss.isValid()) return 0;

    std::mt19937 rng(static_cast<uint32_t>(brush_settings.seed));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    const int count = brush_settings.target_count;
    int spawned = 0, attempts = 0;
    const int max_attempts = (std::max)(count, 0) * 50;

    const float min_dist_sq = brush_settings.min_distance * brush_settings.min_distance;
    const bool check_overlap = brush_settings.min_distance > 0.01f;
    const float cell_size = brush_settings.min_distance > 0.1f ? brush_settings.min_distance : 1.0f;
    std::map<std::pair<int, int>, std::vector<Vec3>> grid;

    while (spawned < count && attempts < max_attempts) {
        ++attempts;
        MeshSurfaceSampler::Sample s = mss.sample(rng);
        if (!s.triangle) continue;

        const float slope_deg = acosf(std::clamp(s.normal.y, -1.0f, 1.0f)) * 57.2958f;
        if (slope_deg > brush_settings.slope_max) continue;
        if (s.position.y < brush_settings.height_min || s.position.y > brush_settings.height_max) continue;

        if (!brush_settings.density_mask_attribute.empty()) {
            const float dval = sampleTriangleFieldAttribute(s.triangle, s.bary_u, s.bary_v, s.bary_w,
                                                              brush_settings.density_mask_attribute);
            if (dist01(rng) > dval) continue;   // rejection sampling — mask = density
        }
        if (!brush_settings.exclusion_mask_attribute.empty()) {
            const float eval = sampleTriangleFieldAttribute(s.triangle, s.bary_u, s.bary_v, s.bary_w,
                                                              brush_settings.exclusion_mask_attribute, -1.0f);
            if (eval >= 0.0f && eval >= brush_settings.exclusion_threshold) continue;
        }

        if (check_overlap) {
            const int cx = static_cast<int>(std::floor(s.position.x / cell_size));
            const int cz = static_cast<int>(std::floor(s.position.z / cell_size));
            bool collision = false;
            for (int ddx = -1; ddx <= 1 && !collision; ++ddx)
                for (int ddz = -1; ddz <= 1 && !collision; ++ddz) {
                    auto git = grid.find({ cx + ddx, cz + ddz });
                    if (git == grid.end()) continue;
                    for (const auto& gp : git->second) {
                        if ((gp - s.position).length_squared() < min_dist_sq) { collision = true; break; }
                    }
                }
            if (collision) continue;
            grid[{cx, cz}].push_back(s.position);
        }

        InstanceTransform inst = generateRandomTransform(s.position, s.normal);
        if (!brush_settings.scale_mask_attribute.empty() && brush_settings.scale_mask_influence > 0.0f) {
            const float sval = sampleTriangleFieldAttribute(s.triangle, s.bary_u, s.bary_v, s.bary_w,
                                                              brush_settings.scale_mask_attribute);
            const float f = 1.0f - brush_settings.scale_mask_influence * (1.0f - sval);
            inst.scale = inst.scale * f;
        }
        addInstance(inst);
        ++spawned;
    }
    return spawned;
}

int InstanceGroup::scatterFillTerrain(TerrainObject* terrain) {
    if (!terrain || terrain->heightmap.width < 3 || terrain->heightmap.height < 3 ||
        terrain->heightmap.data.empty() || brush_settings.target_count <= 0) return 0;

    const int width = terrain->heightmap.width;
    const int height = terrain->heightmap.height;
    const float terrainScale = terrain->heightmap.scale_xz;
    const float cellX = terrainScale / static_cast<float>((std::max)(1, width - 1));
    const float cellZ = terrainScale / static_cast<float>((std::max)(1, height - 1));
    std::mt19937 rng(static_cast<uint32_t>(brush_settings.seed));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    auto sampleNamedField = [&](const std::string& fieldName, float u, float v) -> float {
        if (fieldName.empty()) return 1.0f;
        const auto it = terrain->analysisFields.find(fieldName);
        const size_t expected = static_cast<size_t>(width) * height;
        if (it == terrain->analysisFields.end() || !it->second || it->second->size() != expected) {
            return -1.0f; // fail open, matching brush and legacy Scatter
        }
        const auto& field = *it->second;
        const float gx = std::clamp(u, 0.0f, 1.0f) * (width - 1);
        const float gz = std::clamp(v, 0.0f, 1.0f) * (height - 1);
        const int x0 = static_cast<int>(gx), z0 = static_cast<int>(gz);
        const int x1 = (std::min)(x0 + 1, width - 1);
        const int z1 = (std::min)(z0 + 1, height - 1);
        const float ax = gx - x0, az = gz - z0;
        auto at = [&](int x, int z) { return field[static_cast<size_t>(z) * width + x]; };
        const float a = at(x0, z0) * (1.0f - ax) + at(x1, z0) * ax;
        const float b = at(x0, z1) * (1.0f - ax) + at(x1, z1) * ax;
        return std::clamp(a * (1.0f - az) + b * az, 0.0f, 1.0f);
    };

    auto sampleSplat = [&](int channel, float u, float v) -> float {
        if (channel < 0 || channel > 3 || !terrain->splatMap ||
            !terrain->splatMap->is_loaded() || terrain->splatMap->pixels.empty()) return -1.0f;
        const int texW = terrain->splatMap->width;
        const int texH = terrain->splatMap->height;
        if (texW <= 0 || texH <= 0) return -1.0f;
        const int x = std::clamp(static_cast<int>(u * (texW - 1)), 0, texW - 1);
        const int y = std::clamp(static_cast<int>((1.0f - v) * (texH - 1)), 0, texH - 1);
        const size_t pixelIndex = static_cast<size_t>(y) * texW + x;
        if (pixelIndex >= terrain->splatMap->pixels.size()) return -1.0f;
        const auto& pixel = terrain->splatMap->pixels[pixelIndex];
        if (channel == 0) return pixel.r / 255.0f;
        if (channel == 1) return pixel.g / 255.0f;
        if (channel == 2) return pixel.b / 255.0f;
        return pixel.a / 255.0f;
    };

    const float minDistanceSq = brush_settings.min_distance * brush_settings.min_distance;
    const bool checkOverlap = brush_settings.min_distance > 0.01f;
    const float cellSize = brush_settings.min_distance > 0.1f ? brush_settings.min_distance : 1.0f;
    std::map<std::pair<int, int>, std::vector<Vec3>> occupied;
    const int target = brush_settings.target_count;
    const int maxAttempts = target * 100;
    int spawned = 0;

    for (int attempt = 0; spawned < target && attempt < maxAttempts; ++attempt) {
        const float u = dist01(rng);
        const float v = dist01(rng);

        if (!brush_settings.density_mask_attribute.empty()) {
            const float densityField = sampleNamedField(brush_settings.density_mask_attribute, u, v);
            if (densityField >= 0.0f && dist01(rng) > densityField) continue;
        }
        const float exclusionField = sampleNamedField(brush_settings.exclusion_mask_attribute, u, v);
        if (!brush_settings.exclusion_mask_attribute.empty() && exclusionField >= 0.0f &&
            exclusionField >= brush_settings.exclusion_threshold) continue;

        const float splat = sampleSplat(brush_settings.splat_map_channel, u, v);
        if (splat >= 0.0f && (splat < 0.2f || dist01(rng) > splat)) continue;
        const float exclusionSplat = sampleSplat(brush_settings.exclusion_channel, u, v);
        if (exclusionSplat >= brush_settings.exclusion_threshold) continue;

        const float gx = u * (width - 1);
        const float gz = v * (height - 1);
        const int x0 = static_cast<int>(gx), z0 = static_cast<int>(gz);
        const int x1 = (std::min)(x0 + 1, width - 1);
        const int z1 = (std::min)(z0 + 1, height - 1);
        const float fx = gx - x0, fz = gz - z0;
        const float h00 = terrain->heightmap.getHeight(x0, z0);
        const float h10 = terrain->heightmap.getHeight(x1, z0);
        const float h01 = terrain->heightmap.getHeight(x0, z1);
        const float h11 = terrain->heightmap.getHeight(x1, z1);
        const float localHeight = (h00 * (1.0f - fx) + h10 * fx) * (1.0f - fz) +
                                  (h01 * (1.0f - fx) + h11 * fx) * fz;
        if (localHeight < brush_settings.height_min || localHeight > brush_settings.height_max) continue;

        int step = (std::max)(1, brush_settings.curvature_step);
        if (width <= step * 2 || height <= step * 2) continue;
        const int sx = std::clamp(static_cast<int>(gx + 0.5f), step, width - 1 - step);
        const int sz = std::clamp(static_cast<int>(gz + 0.5f), step, height - 1 - step);
        const auto normalizedHeight = [&](int x, int z) {
            return terrain->heightmap.data[static_cast<size_t>(z) * width + x];
        };
        const float hl = normalizedHeight(sx - step, sz);
        const float hr = normalizedHeight(sx + step, sz);
        const float hu = normalizedHeight(sx, sz - step);
        const float hd = normalizedHeight(sx, sz + step);
        const float hc = normalizedHeight(sx, sz);
        const float dx = ((hr - hl) * terrain->heightmap.scale_y) / (2.0f * cellX * step);
        const float dz = ((hd - hu) * terrain->heightmap.scale_y) / (2.0f * cellZ * step);
        const float slopeDegrees = std::atan(std::sqrt(dx * dx + dz * dz)) * 57.2958f;
        if (slopeDegrees > brush_settings.slope_max) continue;

        if (brush_settings.slope_direction_influence > 0.01f && slopeDegrees > 2.0f) {
            float aspect = std::atan2(-dx, -dz) * 57.2958f;
            if (aspect < 0.0f) aspect += 360.0f;
            float difference = std::fabs(aspect - brush_settings.slope_direction_angle);
            if (difference > 180.0f) difference = 360.0f - difference;
            const float directionalWeight = (std::max)(0.0f, std::cos(difference * 0.0174533f));
            const float probability = (1.0f - brush_settings.slope_direction_influence) +
                brush_settings.slope_direction_influence * directionalWeight;
            if (dist01(rng) > probability) continue;
        }

        const float curvature = ((hl + hr + hu + hd) - 4.0f * hc) /
            static_cast<float>(step * step) * 1000.0f;
        const bool ridge = curvature < brush_settings.curvature_min;
        const bool gully = curvature > brush_settings.curvature_max;
        if ((ridge && !brush_settings.allow_ridges) ||
            (gully && !brush_settings.allow_gullies) ||
            (!ridge && !gully && !brush_settings.allow_flats)) continue;

        Vec3 worldPosition(u * terrainScale, localHeight, v * terrainScale);
        Vec3 worldNormal(-dx, 1.0f, -dz);
        worldNormal = worldNormal.normalize();
        if (terrain->transform) {
            terrain->transform->updateFinal();
            worldPosition = terrain->transform->final.transform_point(worldPosition);
            worldNormal = terrain->transform->getNormalTransform().transform_vector(worldNormal).normalize();
        }

        if (checkOverlap) {
            const int cellKeyX = static_cast<int>(std::floor(worldPosition.x / cellSize));
            const int cellKeyZ = static_cast<int>(std::floor(worldPosition.z / cellSize));
            bool collision = false;
            for (int ox = -1; ox <= 1 && !collision; ++ox) {
                for (int oz = -1; oz <= 1 && !collision; ++oz) {
                    const auto it = occupied.find({cellKeyX + ox, cellKeyZ + oz});
                    if (it == occupied.end()) continue;
                    for (const Vec3& point : it->second) {
                        if ((point - worldPosition).length_squared() < minDistanceSq) {
                            collision = true;
                            break;
                        }
                    }
                }
            }
            if (collision) continue;
            occupied[{cellKeyX, cellKeyZ}].push_back(worldPosition);
        }

        InstanceTransform instance = generateRandomTransform(worldPosition, worldNormal);
        const float scaleField = sampleNamedField(brush_settings.scale_mask_attribute, u, v);
        if (!brush_settings.scale_mask_attribute.empty() && scaleField >= 0.0f &&
            brush_settings.scale_mask_influence > 0.0f) {
            const float factor = 1.0f - brush_settings.scale_mask_influence * (1.0f - scaleField);
            instance.scale = instance.scale * factor;
        }
        addInstance(instance);
        ++spawned;
    }
    return spawned;
}

InstanceTransform InstanceGroup::generateRandomTransform(const Vec3& position, const Vec3& normal) const {
    // Thread-local RNG for performance
    thread_local std::mt19937 rng(std::random_device{}());
    
    InstanceTransform t;
    t.position = position;

    // 1. SELECT SOURCE FIRST
    if (!sources.empty()) {
        if (sources.size() == 1) {
            t.source_index = 0;
        } else {
            float total_weight = 0.0f;
            for (const auto& src : sources) total_weight += src.weight;
            
            if (total_weight <= 0.001f) {
                std::uniform_int_distribution<int> idx_dist(0, (int)sources.size() - 1);
                t.source_index = idx_dist(rng);
            } else {
                std::uniform_real_distribution<float> w_dist(0.0f, total_weight);
                float r = w_dist(rng);
                float current_w = 0.0f;
                int selected_idx = 0;
                for (int i = 0; i < (int)sources.size(); ++i) {
                    current_w += sources[i].weight;
                    if (r <= current_w) {
                        selected_idx = i;
                        break;
                    }
                }
                t.source_index = selected_idx;
            }
        }
    } else {
        t.source_index = 0;
    }
    
    // 2. DETERMINE SETTINGS (Global or Local)
    float scale_min, scale_max;
    float rot_y, rot_xz;
    float y_off_min, y_off_max;
    bool align;
    float normal_inf;
    
    // Use local settings if available and not overridden
    if (!brush_settings.use_global_settings && !sources.empty() && t.source_index < sources.size()) {
        const auto& set = sources[t.source_index].settings;
        scale_min = set.scale_min;
        scale_max = set.scale_max;
        rot_y = set.rotation_random_y;
        rot_xz = set.rotation_random_xz;
        y_off_min = set.y_offset_min;
        y_off_max = set.y_offset_max;
        align = set.align_to_normal;
        normal_inf = set.normal_influence;
    } else {
        // Fallback to global brush settings
        scale_min = brush_settings.scale_min;
        scale_max = brush_settings.scale_max;
        rot_y = brush_settings.rotation_random_y;
        rot_xz = brush_settings.rotation_random_xz;
        y_off_min = brush_settings.y_offset_min;
        y_off_max = brush_settings.y_offset_max;
        align = brush_settings.align_to_normal;
        normal_inf = brush_settings.normal_influence;
    }
    
    // 3. GENERATE TRANSFORM
    
    // Random scale
    std::uniform_real_distribution<float> scale_dist(scale_min, scale_max);
    float uniform_scale = scale_dist(rng);
    t.scale = Vec3(uniform_scale, uniform_scale, uniform_scale);
    
    // Random Y rotation
    std::uniform_real_distribution<float> rot_y_dist(0.0f, rot_y);
    t.rotation.y = rot_y_dist(rng);
    
    // Random tilt (X/Z rotation)
    if (rot_xz > 0.0f) {
        std::uniform_real_distribution<float> tilt_dist(-rot_xz, rot_xz);
        t.rotation.x = tilt_dist(rng);
        t.rotation.z = tilt_dist(rng);
    }
    
    // Generate random Y offset
    if (y_off_min != 0.0f || y_off_max != 0.0f) {
        std::uniform_real_distribution<float> offset_dist(y_off_min, y_off_max);
        t.position.y += offset_dist(rng);
    }
    
    // Align to surface normal
    if (align && normal.length() > 0.01f) {
        Vec3 up(0, 1, 0);
        Vec3 n = normal.normalize();
        
        float influence = normal_inf;
        Vec3 target_up = up * (1.0f - influence) + n * influence;
        target_up = target_up.normalize();
        
        t.rotation.x += asinf(-target_up.z) * 180.0f / 3.14159f * influence;
        t.rotation.z += asinf(target_up.x) * 180.0f / 3.14159f * influence;
    }

    return t;
}

void InstanceGroup::updateWind(float time) {
    if (!wind_settings.enabled) return;

    // ═══════════════════════════════════════════════════════════════════════════
    // WIND PARAMETERS
    // ═══════════════════════════════════════════════════════════════════════════
    
    float speed = wind_settings.speed;
    float strength = wind_settings.strength;
    float turbulence = wind_settings.turbulence;
    float wave_size = wind_settings.wave_size;
    if (wave_size < 0.1f) wave_size = 50.0f;
    
    // Normalize wind direction
    Vec3 dir = wind_settings.direction;
    float len = dir.length();
    if (len > 0.001f) dir = dir / len;
    else return;

    // ═══════════════════════════════════════════════════════════════════════════
    // ENHANCED WIND PARAMETERS
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Constant lean: How much the tree bends TOWARDS wind direction
    float lean_amount = strength * 0.6f;  // 60% goes to constant lean
    
    // Oscillation: The back-and-forth sway
    float sway_amount = strength * 0.4f;  // 40% goes to dynamic sway
    
    // Maximum bend angle (prevents unnatural over-bending)
    const float max_bend_angle = 25.0f;  // degrees
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ROTATION AXIS (perpendicular to wind and UP)
    // ═══════════════════════════════════════════════════════════════════════════
    
    Vec3 up(0, 1, 0);
    Vec3 axis = dir.cross(up);
    float axis_len = axis.length();
    if (axis_len < 0.001f) {
        axis = Vec3(1, 0, 0);  // Fallback for vertical wind
    } else {
        axis = axis / axis_len;
    }
    float ax = axis.x, ay = axis.y, az = axis.z;

    // ═══════════════════════════════════════════════════════════════════════════
    // LAZY INIT: Capture initial state if missing
    // ═══════════════════════════════════════════════════════════════════════════
    
    if (initial_instances.empty() && !instances.empty()) {
        initial_instances = instances;
    }
    if (initial_instances.size() != instances.size()) {
        initial_instances = instances;
    }

    // Iterate instances
    size_t count = std::min({instances.size(), active_hittables.size(), initial_instances.size()});

    for (size_t i = 0; i < count; ++i) {
        // Check if weak pointer is valid
        if (active_hittables[i].expired()) continue;
        
        auto hittable_ptr = active_hittables[i].lock();
        if (!hittable_ptr) continue;

        auto inst = std::dynamic_pointer_cast<HittableInstance>(hittable_ptr);
        if (!inst) continue;

        const auto& initial = initial_instances[i];
        
        // ═══════════════════════════════════════════════════════════════════════════
        // MULTI-FREQUENCY OSCILLATION
        // Creates natural, organic movement by combining multiple wave frequencies
        // ═══════════════════════════════════════════════════════════════════════════
        
        // Phase based on world position (creates wave propagation effect)
        float pos_phase = (initial.position.x * dir.x + initial.position.z * dir.z) / wave_size;
        float t_phase = time * speed;
        
        // Primary wave: Slow, large movement
        float wave_primary = sinf(pos_phase + t_phase) * 1.0f;
        
        // Secondary wave: Faster, smaller movement (flutter)
        float wave_secondary = sinf(pos_phase * 2.3f + t_phase * 1.7f) * 0.35f;
        
        // Tertiary wave: High frequency micro-movement (turbulence)
        float wave_tertiary = sinf(pos_phase * 4.1f + t_phase * 2.9f * turbulence) * 0.15f;
        
        // Combined oscillation (-1 to +1 range)
        float oscillation = (wave_primary + wave_secondary + wave_tertiary) / 1.5f;
        
        // ═══════════════════════════════════════════════════════════════════════════
        // DIRECTIONAL BENDING
        // Constant lean towards wind + oscillation around that lean
        // ═══════════════════════════════════════════════════════════════════════════
        
        // Total angle = constant lean + dynamic oscillation
        float total_angle = lean_amount + oscillation * sway_amount;
        
        // Clamp to max bend angle
        total_angle = std::max(-max_bend_angle, std::min(max_bend_angle, total_angle));
        
        // ═══════════════════════════════════════════════════════════════════════════
        // BUILD ROTATION MATRIX (Axis-Angle)
        // ═══════════════════════════════════════════════════════════════════════════
        
        float rad = total_angle * 3.14159f / 180.0f;
        float c = cosf(rad);
        float s = sinf(rad);
        float t_val = 1.0f - c;

        Matrix4x4 swayMat;
        swayMat.m[0][0] = t_val*ax*ax + c;    swayMat.m[0][1] = t_val*ax*ay - az*s; swayMat.m[0][2] = t_val*ax*az + ay*s; swayMat.m[0][3] = 0;
        swayMat.m[1][0] = t_val*ax*ay + az*s; swayMat.m[1][1] = t_val*ay*ay + c;    swayMat.m[1][2] = t_val*ay*az - ax*s; swayMat.m[1][3] = 0;
        swayMat.m[2][0] = t_val*ax*az - ay*s; swayMat.m[2][1] = t_val*ay*az + ax*s; swayMat.m[2][2] = t_val*az*az + c;    swayMat.m[2][3] = 0;
        swayMat.m[3][0] = 0;                  swayMat.m[3][1] = 0;                  swayMat.m[3][2] = 0;                  swayMat.m[3][3] = 1;

        // ═══════════════════════════════════════════════════════════════════════════
        // APPLY TRANSFORM: Translate * Sway * LocalRotationScale
        // ═══════════════════════════════════════════════════════════════════════════
        
        Matrix4x4 baseMat = initial.toMatrix();
        
        // Remove translation to get Orientation * Scale
        Matrix4x4 localMat = baseMat;
        localMat.m[0][3] = 0; localMat.m[1][3] = 0; localMat.m[2][3] = 0;

        // Translation matrix
        Matrix4x4 T; 
        T.m[0][3] = initial.position.x; 
        T.m[1][3] = initial.position.y; 
        T.m[2][3] = initial.position.z;

        // Final = Translate * Sway * Local
        Matrix4x4 TSway = T * swayMat; 
        Matrix4x4 finalMat = TSway * localMat;

        // Update Instance
        inst->setTransform(finalMat);
    }
}
