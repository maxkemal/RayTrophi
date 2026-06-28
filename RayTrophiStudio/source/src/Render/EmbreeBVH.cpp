#include "EmbreeBVH.h"
#include "HittableInstance.h"
#include "VDBVolume.h" // Add VDB support
#include "VDBVolumeManager.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <Volumetric.h>

// Static member initialization
RTCDevice EmbreeBVH::device = nullptr;

namespace {
constexpr int kMaxCpuVolumeMarchSteps = 4096;

float compute_safe_volume_step(float requested_step, float t_enter, float t_exit) {
    const float distance = std::max(0.0f, t_exit - t_enter);
    const float min_step = 0.01f;
    const float base_step = std::max(requested_step, min_step);
    if (distance <= 0.0f) {
        return base_step;
    }

    // Prevent pathological hangs on very large scenes by keeping march count bounded.
    return std::max(base_step, distance / float(kMaxCpuVolumeMarchSteps));
}

bool advance_embree_tmin(float hit_tfar, float& t_min, float t_max, const char* context) {
    const float next_t_min = hit_tfar + 0.001f;
    if (!std::isfinite(next_t_min) || next_t_min <= t_min) {
        SCENE_LOG_WARN(std::string(context) + " Abort: non-progressing transparent/volume skip.");
        return false;
    }

    t_min = next_t_min;
    return t_min < t_max;
}

bool resolve_embree_surface_hit(const EmbreeBVH& bvh,
                                const RTCRayHit& rayhit,
                                const TriangleData*& out_tri,
                                Material*& out_mat,
                                Vec2& out_uv,
                                bool& out_hit_instance) {
    out_tri = nullptr;
    out_mat = nullptr;
    out_uv = Vec2(0.0f, 0.0f);

    const unsigned top_inst_id = rayhit.hit.instID[0];
    out_hit_instance = (top_inst_id != RTC_INVALID_GEOMETRY_ID &&
                        top_inst_id < bvh.instance_objects.size() &&
                        bvh.instance_objects[top_inst_id] != nullptr);

    if (!out_hit_instance) {
        if (rayhit.hit.geomID != bvh.triangle_geom_id ||
            rayhit.hit.primID >= bvh.triangle_data.size()) {
            return false;
        }

        out_tri = &bvh.triangle_data[rayhit.hit.primID];
    } else {
        const auto& inst = bvh.instance_objects[top_inst_id];
        if (!inst) {
            return false;
        }

        auto child_bvh = std::dynamic_pointer_cast<EmbreeBVH>(inst->mesh);
        if (!child_bvh || rayhit.hit.primID >= child_bvh->triangle_data.size()) {
            return false;
        }

        out_tri = &child_bvh->triangle_data[rayhit.hit.primID];
    }

    const float u = rayhit.hit.u;
    const float v = rayhit.hit.v;
    const float w = 1.0f - u - v;
    if (out_tri->original_ptr) {
        const auto [t0, t1, t2] = out_tri->original_ptr->getUVCoordinates();
        out_uv = t0 * w + t1 * u + t2 * v;
    } else {
        out_uv = Vec2(0.0f);
    }
    out_mat = out_tri->getMaterial();
    return true;
}
}

EmbreeBVH::EmbreeBVH() {
    // Lazily create device once
    if (!device) {
        device = rtcNewDevice(nullptr);
        // Error handling?
        if (!device) {
             SCENE_LOG_ERROR("Failed to create Embree device!");
        }
    }
    
    // Create scene (unique per BVH instance)
    if (device) {
        scene = rtcNewScene(device);
        rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_MEDIUM); 
        rtcSetSceneFlags(scene, RTC_SCENE_FLAG_DYNAMIC); 
    }
}

EmbreeBVH::~EmbreeBVH() {
    if (scene) {
        rtcReleaseScene(scene);
        scene = nullptr;
    }
    // DO NOT release device here! It is shared.
}

void EmbreeBVH::shutdown() {
    if (device) {
        rtcReleaseDevice(device);
        device = nullptr;
    }
}

void EmbreeBVH::build(const std::vector<std::shared_ptr<Hittable>>& objects) {
    auto build_start = std::chrono::high_resolution_clock::now();

    // Reset state
    clearGeometry(); 
    triangle_geom_id = RTC_INVALID_GEOMETRY_ID;
    vdb_geom_id = RTC_INVALID_GEOMETRY_ID;
    instance_objects.clear();
    vdb_objects.clear();
    active_mesh_groups.clear();

    if (!device) {
        SCENE_LOG_ERROR("[EmbreeBVH::build] Device is null!");
        return;
    }
    
    // Create new scene if needed
    if (!scene) {
        scene = rtcNewScene(device);
        rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_MEDIUM);
        rtcSetSceneFlags(scene, RTC_SCENE_FLAG_DYNAMIC | RTC_SCENE_FLAG_ROBUST);
    }

    // 1. Separate objects
    cached_triangles.clear();
    std::vector<std::shared_ptr<HittableInstance>> local_instances;
    
    cached_triangles.reserve(objects.size());
    local_instances.reserve(objects.size());
    vdb_objects.reserve(objects.size());

    for (const auto& obj : objects) {
        if (!obj->visible) continue;

        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            cached_triangles.push_back(tri);
        }
        else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            local_instances.push_back(inst);
        }
        else if (auto vdb = std::dynamic_pointer_cast<VDBVolume>(obj)) {
            vdb_objects.push_back(vdb.get());
        }
    }

    // 2. Build Triangle Geometry (if any)
    if (!cached_triangles.empty()) {
        // Group triangles by parentMesh to exploit contiguous flat buffers
        struct TempMeshGroup {
            TriangleMesh* mesh = nullptr;
            std::vector<std::shared_ptr<Triangle>> tris;
        };
        std::vector<TempMeshGroup> temp_groups;
        std::unordered_map<TriangleMesh*, size_t> mesh_to_group;
        std::vector<std::shared_ptr<Triangle>> standalone_tris;

        for (const auto& tri : cached_triangles) {
            if (!tri) continue;
            if (tri->parentMesh) {
                auto it = mesh_to_group.find(tri->parentMesh.get());
                if (it != mesh_to_group.end()) {
                    temp_groups[it->second].tris.push_back(tri);
                } else {
                    mesh_to_group[tri->parentMesh.get()] = temp_groups.size();
                    temp_groups.push_back({tri->parentMesh.get(), {tri}});
                }
            } else {
                standalone_tris.push_back(tri);
            }
        }

        // Calculate total vertices and triangles to allocate
        size_t total_vertices = 0;
        size_t total_triangles = cached_triangles.size();

        for (const auto& group : temp_groups) {
            total_vertices += group.mesh->num_vertices();
        }
        total_vertices += standalone_tris.size() * 3;

        RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

        Vec3* vertex_buffer = (Vec3*)rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
            sizeof(Vec3), total_vertices
        );
        unsigned* index_buffer = (unsigned*)rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
            sizeof(unsigned) * 3, total_triangles
        );

        triangle_data.resize(total_triangles);

        size_t vertex_offset = 0;
        size_t tri_offset = 0;

        // 2a. Populate grouped meshes using zero-copy memcpy
        for (const auto& group : temp_groups) {
            size_t vCount = group.mesh->num_vertices();
            if (vCount > 0 && group.mesh->geometry) {
                const Vec3* src_positions = group.mesh->geometry->get_attribute_data<Vec3>("P");
                if (src_positions) {
                    memcpy(vertex_buffer + vertex_offset, src_positions, vCount * sizeof(Vec3));
                }

                // Add to active_mesh_groups for fast refitting
                Matrix4x4 initial_xform = group.mesh->transform ? group.mesh->transform->getFinal() : Matrix4x4::identity();
                active_mesh_groups.push_back({group.mesh, vertex_offset, vCount, initial_xform});
            }

            // Populate indices and triangle_data in parallel
            size_t group_tris_count = group.tris.size();
            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static)
            for (int i = 0; i < (int)group_tris_count; ++i) {
                const auto& tri = group.tris[i];
                uint32_t faceIdx = tri->faceIndex;
                size_t local_tri_offset = tri_offset + i;
                if (group.mesh->geometry && faceIdx * 3 + 2 < group.mesh->geometry->indices.size()) {
                    index_buffer[local_tri_offset * 3 + 0] = group.mesh->geometry->indices[faceIdx * 3 + 0] + static_cast<unsigned>(vertex_offset);
                    index_buffer[local_tri_offset * 3 + 1] = group.mesh->geometry->indices[faceIdx * 3 + 1] + static_cast<unsigned>(vertex_offset);
                    index_buffer[local_tri_offset * 3 + 2] = group.mesh->geometry->indices[faceIdx * 3 + 2] + static_cast<unsigned>(vertex_offset);
                } else {
                    index_buffer[local_tri_offset * 3 + 0] = 0;
                    index_buffer[local_tri_offset * 3 + 1] = 0;
                    index_buffer[local_tri_offset * 3 + 2] = 0;
                }

                triangle_data[local_tri_offset] = {
                    tri->getMaterialID(),
                    tri->terrain_id,
                    tri.get()
                };
            }
            tri_offset += group_tris_count;
            vertex_offset += vCount;
        }

        // Save standalone offsets for fast refits
        standalone_vertex_offset = vertex_offset;
        standalone_tri_offset = tri_offset;

        // 2b. Populate standalone triangles (fallback path)
        if (!standalone_tris.empty()) {
            size_t standalone_count = standalone_tris.size();
            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static)
            for (int i = 0; i < (int)standalone_count; ++i) {
                const auto& tri = standalone_tris[i];
                size_t local_v_offset = vertex_offset + i * 3;
                size_t local_t_offset = tri_offset + i;

                // Vertices
                vertex_buffer[local_v_offset + 0] = tri->getVertexPosition(0);
                vertex_buffer[local_v_offset + 1] = tri->getVertexPosition(1);
                vertex_buffer[local_v_offset + 2] = tri->getVertexPosition(2);

                // Indices
                index_buffer[local_t_offset * 3 + 0] = static_cast<unsigned>(local_v_offset + 0);
                index_buffer[local_t_offset * 3 + 1] = static_cast<unsigned>(local_v_offset + 1);
                index_buffer[local_t_offset * 3 + 2] = static_cast<unsigned>(local_v_offset + 2);

                // TriangleData
                triangle_data[local_t_offset] = {
                    tri->getMaterialID(),
                    tri->terrain_id,
                    tri.get()
                };
            }
            
            tri_offset += standalone_count;
            vertex_offset += standalone_count * 3;
        }

        rtcSetGeometryMask(geom, 0x01); // Mask 1 for Surfaces
        rtcCommitGeometry(geom);
        triangle_geom_id = rtcAttachGeometry(scene, geom);
        rtcReleaseGeometry(geom);
    }

    // 3. Build Instances
    if (!local_instances.empty()) {
        for (const auto& inst : local_instances) {
            auto child_bvh = std::dynamic_pointer_cast<EmbreeBVH>(inst->mesh);
            if (!child_bvh) {
                 continue; 
            }

            RTCGeometry inst_geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryInstancedScene(inst_geom, child_bvh->getRTCScene());
            
            float transform[12];
            // Column 0 (X axis)
            transform[0] = inst->transform.m[0][0];
            transform[1] = inst->transform.m[1][0];
            transform[2] = inst->transform.m[2][0];
            
            // Column 1 (Y axis)
            transform[3] = inst->transform.m[0][1];
            transform[4] = inst->transform.m[1][1];
            transform[5] = inst->transform.m[2][1];
            
            // Column 2 (Z axis)
            transform[6] = inst->transform.m[0][2];
            transform[7] = inst->transform.m[1][2];
            transform[8] = inst->transform.m[2][2];
             
            // Column 3 (Translation)
            transform[9]  = inst->transform.m[0][3];
            transform[10] = inst->transform.m[1][3];
            transform[11] = inst->transform.m[2][3];

            rtcSetGeometryTransform(inst_geom, 0, RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR, transform);
            
            rtcSetGeometryMask(inst_geom, 0xFFFFFFFF); // Instances pass everything
            rtcCommitGeometry(inst_geom);
            unsigned geomID = rtcAttachGeometry(scene, inst_geom);
            rtcReleaseGeometry(inst_geom);
            
            // Store mapping
            if (instance_objects.size() <= geomID) {
                instance_objects.resize(geomID + 1, nullptr);
            }
            instance_objects[geomID] = inst;
        }
    }

    // 4. Build VDB Volumes (User Geometry)
    if (!vdb_objects.empty()) {
        RTCGeometry vdb_geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(vdb_geom, vdb_objects.size());
        rtcSetGeometryUserData(vdb_geom, this);
        
        rtcSetGeometryBoundsFunction(vdb_geom, userBoundsFunc, nullptr);
        rtcSetGeometryIntersectFunction(vdb_geom, userIntersectFunc);
        rtcSetGeometryOccludedFunction(vdb_geom, userOccludedFunc);
        
        rtcSetGeometryMask(vdb_geom, 0x02); // Mask 2 for Volumes
        rtcCommitGeometry(vdb_geom);
        vdb_geom_id = rtcAttachGeometry(scene, vdb_geom);
        rtcReleaseGeometry(vdb_geom);
    }

    rtcCommitScene(scene);
}

// ... existing methods ...

// ═══════════════════════════════════════════════════════════════════════════
// EMBREE USER GEOMETRY CALLBACKS (For VDB Volumes)
// ═══════════════════════════════════════════════════════════════════════════

void EmbreeBVH::userBoundsFunc(const RTCBoundsFunctionArguments* args) {
    const EmbreeBVH* bvh = (const EmbreeBVH*)args->geometryUserPtr;
    const VDBVolume* vdb = bvh->vdb_objects[args->primID];
    
    AABB bounds = vdb->getWorldBounds();
    RTCBounds* rtc_bounds = args->bounds_o;
    
    rtc_bounds->lower_x = bounds.min.x;
    rtc_bounds->lower_y = bounds.min.y;
    rtc_bounds->lower_z = bounds.min.z;
    rtc_bounds->upper_x = bounds.max.x;
    rtc_bounds->upper_y = bounds.max.y;
    rtc_bounds->upper_z = bounds.max.z;
}

void EmbreeBVH::userIntersectFunc(const RTCIntersectFunctionNArguments* args) {
    int* valid = args->valid;
    EmbreeBVH* bvh = (EmbreeBVH*)args->geometryUserPtr;
    unsigned int primID = args->primID;
    unsigned int geomID = args->geomID;
    
    RTCRayHitN* rayhit = args->rayhit;
    RTCRayN* ray = RTCRayHitN_RayN(rayhit, args->N);
    RTCHitN* hit = RTCRayHitN_HitN(rayhit, args->N);
    
    // Safety check for context
    unsigned int instID = RTC_INVALID_GEOMETRY_ID;
    if (args->context) instID = args->context->instID[0];

    for (unsigned int i=0; i < args->N; i++) {
        if (!valid[i]) continue;
        
        // Retrieve Ray properties via accessors
        float tfar = RTCRayN_tfar(ray, args->N, i);
        float tnear = RTCRayN_tnear(ray, args->N, i);
        
        Vec3 origin(
            RTCRayN_org_x(ray, args->N, i),
            RTCRayN_org_y(ray, args->N, i),
            RTCRayN_org_z(ray, args->N, i)
        );
        Vec3 dir(
            RTCRayN_dir_x(ray, args->N, i),
            RTCRayN_dir_y(ray, args->N, i),
            RTCRayN_dir_z(ray, args->N, i)
        );
        
        Ray r(origin, dir);
        const VDBVolume* vdb = bvh->vdb_objects[primID];
        // Particles/splat-sphere fluid mode keeps the volume registered for the GPU
        // SSBO mapping but it must NOT occlude/shadow on the CPU, or its AABB masks
        // the discrete splat spheres inside the domain. Checked at runtime (no rebuild
        // needed when the render mode toggles).
        if (vdb && vdb->cpu_render_skip) continue;

        float t_enter, t_exit;
        // Pass -infinity instead of tnear to detect if we are inside the box (t_enter < tnear)
        if (vdb->intersectTransformedAABB(r, -std::numeric_limits<float>::infinity(), tfar, t_enter, t_exit)) {
            const bool starts_inside = (t_enter < tnear);
            float reported_hit = starts_inside ? t_exit : t_enter;
            bool is_inside_or_enter = starts_inside ? (t_exit > tnear) : (t_enter >= tnear);

            if (is_inside_or_enter && reported_hit < tfar) {
                // Update Ray (shorten to hit)
                RTCRayN_tfar(ray, args->N, i) = reported_hit;
                
                // Update Hit
                RTCHitN_geomID(hit, args->N, i) = geomID;
                RTCHitN_primID(hit, args->N, i) = primID;
                RTCHitN_instID(hit, args->N, i, 0) = instID;
                
                // Set geometric normal
                RTCHitN_Ng_x(hit, args->N, i) = 0.0f;
                RTCHitN_Ng_y(hit, args->N, i) = 1.0f;
                RTCHitN_Ng_z(hit, args->N, i) = 0.0f;
            }
        }
    }
}

void EmbreeBVH::userOccludedFunc(const RTCOccludedFunctionNArguments* args) {
    int* valid = args->valid;
    EmbreeBVH* bvh = (EmbreeBVH*)args->geometryUserPtr;
    unsigned int primID = args->primID;
    
    RTCRayN* ray = args->ray;
    
    for (unsigned int i=0; i < args->N; i++) {
        if (!valid[i]) continue;
        
        float tfar = RTCRayN_tfar(ray, args->N, i);
        // Note: For occlusion, tfar is -inf if occluded. 
        // But incoming ray has valid tfar (dist to light).
        // If we hit, we set tfar to -inf.
        if (tfar < 0.0f) continue; // Already occluded
        
        float tnear = RTCRayN_tnear(ray, args->N, i);
        
        Vec3 origin(
            RTCRayN_org_x(ray, args->N, i),
            RTCRayN_org_y(ray, args->N, i),
            RTCRayN_org_z(ray, args->N, i)
        );
        Vec3 dir(
            RTCRayN_dir_x(ray, args->N, i),
            RTCRayN_dir_y(ray, args->N, i),
            RTCRayN_dir_z(ray, args->N, i)
        );
        
        Ray r(origin, dir);
        const VDBVolume* vdb = bvh->vdb_objects[primID];
        // Particles/splat-sphere fluid mode keeps the volume registered for the GPU
        // SSBO mapping but it must NOT occlude/shadow on the CPU, or its AABB masks
        // the discrete splat spheres inside the domain. Checked at runtime (no rebuild
        // needed when the render mode toggles).
        if (vdb && vdb->cpu_render_skip) continue;

        float t_enter, t_exit;
        if (vdb->intersectTransformedAABB(r, tnear, tfar, t_enter, t_exit)) {
            // Stochastic Ray Marching for Shadow
            float step_size = 0.5f; 
            if (vdb->volume_shader) step_size = vdb->volume_shader->quality.step_size * 2.0f;
            step_size = compute_safe_volume_step(step_size, t_enter, t_exit);
            
            const auto& mgr = VDBVolumeManager::getInstance();
            int vid = vdb->getVDBVolumeID();
            float density_mult = (vdb->volume_shader ? vdb->volume_shader->density.multiplier : 1.0f) * vdb->density_scale;

            float t = t_enter + ((float)rand() / RAND_MAX) * step_size;
            float transmittance = 1.0f;
            int steps = 0;
            // Use local temporary ray for marching to avoid modifying original r
            // Note: r is already world space
            
            while (t < t_exit && steps < kMaxCpuVolumeMarchSteps) {
                Vec3 pos = r.at(t);
                Vec3 local_pos = vdb->getInverseTransform().transform_point(pos);
                
                float density = mgr.sampleDensityCPU(vid, local_pos.x, local_pos.y, local_pos.z);
                
                if (density > 0.001f) {
                    float sigma_t = density * density_mult;
                    transmittance *= exp(-sigma_t * step_size);
                }
                
                if (transmittance < 0.01f) break; // Fully blocked
                t += step_size;
                ++steps;
            }

            // Stochastic Test
            if (Vec3::random_float() > transmittance) {
                // Occluded!
                RTCRayN_tfar(ray, args->N, i) = -INFINITY;
            } else {
                // Transparent - Continue (do nothing)
            }
            continue; // Move to next potential hit (or finish if we set tfar=-inf, Embree might stop)
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Clear all geometry from the scene (for full rebuild)
// ═══════════════════════════════════════════════════════════════════════════
void EmbreeBVH::clearGeometry() {
    // Release the current scene and create a new empty one
    if (scene) {
        rtcReleaseScene(scene);
        scene = rtcNewScene(device);
        rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_MEDIUM);
        rtcSetSceneFlags(scene, RTC_SCENE_FLAG_DYNAMIC);
    }
    triangle_data.clear();
    cached_triangles.clear(); // [NEW] Önbelleği de temizle
    active_mesh_groups.clear();
    
    // CRITICAL FIX: Clear instance mappings to prevent stale pointers after rebuild
    instance_objects.clear();
    triangle_geom_id = 0xFFFFFFFF; // Reset to invalid
}

void EmbreeBVH::updateGeometryFromTriangles() {
    // Tek geometry'li yapıdaysak geometry ID sabittir (örneğin 0)
    RTCGeometry geom = rtcGetGeometry(scene, 0); // ID = 0 çünkü tek sefer attach ettik

    Vec3* vertex_buffer = (Vec3*)rtcGetGeometryBufferData(geom, RTC_BUFFER_TYPE_VERTEX, 0);
    for (size_t i = 0; i < triangle_data.size(); ++i) {
        const auto& tri = triangle_data[i];
        if (tri.original_ptr) {
            vertex_buffer[i * 3 + 0] = tri.original_ptr->getVertexPosition(0);
            vertex_buffer[i * 3 + 1] = tri.original_ptr->getVertexPosition(1);
            vertex_buffer[i * 3 + 2] = tri.original_ptr->getVertexPosition(2);
        }
    }

    rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0);
    rtcCommitGeometry(geom);
    rtcCommitScene(scene);
}
// Bu her karede çağrılır - OPTIMIZE WITH REFIT
// Bu her karede çağrılır - OPTIMIZE WITH REFIT
void EmbreeBVH::updateGeometryFromTrianglesFromSource(const std::vector<std::shared_ptr<Hittable>>& objects) {
    bool geometry_committed = false;

    // Topology changes require a full rebuild. Refit is only safe when the dense
    // triangle set still matches Embree's existing vertex/index buffers.
    if (cached_triangles.size() != triangle_data.size()) {
        extern bool g_bvh_rebuild_pending;
        g_bvh_rebuild_pending = true;
        return;
    }

    // 1. Update Triangle Geometry (Deformable)
    if (triangle_geom_id != RTC_INVALID_GEOMETRY_ID) {
        RTCGeometry geom = rtcGetGeometry(scene, triangle_geom_id);
        if (geom) {
            Vec3* vertex_buffer = (Vec3*)rtcGetGeometryBufferData(geom, RTC_BUFFER_TYPE_VERTEX, 0);
            if (vertex_buffer) {
                rtcSetGeometryBuildQuality(geom, RTC_BUILD_QUALITY_REFIT);

                int any_dirty = 0;

                // 1. Grouped fast-path: zero-copy memcpy deformed vertices in parallel (only if transform changed)
                if (!active_mesh_groups.empty()) {
                    size_t group_count = active_mesh_groups.size();
                    #pragma omp parallel for num_threads(get_omp_threads_limit()) reduction(+:any_dirty)
                    for (int i = 0; i < (int)group_count; ++i) {
                        auto& group = active_mesh_groups[i];
                        Matrix4x4 current_xform = group.mesh->transform ? group.mesh->transform->getFinal() : Matrix4x4::identity();
                        if (!(group.last_xform == current_xform)) {
                            const Vec3* src_positions = group.mesh->geometry->get_attribute_data<Vec3>("P");
                            if (src_positions) {
                                memcpy(vertex_buffer + group.vertex_offset, src_positions, group.vertex_count * sizeof(Vec3));
                                any_dirty += 1;
                            }
                            group.last_xform = current_xform;
                        }
                    }
                }

                // 2. Standalone fallback path: update vertices for standalone triangles
                if (standalone_tri_offset < cached_triangles.size()) {
                    size_t standalone_count = cached_triangles.size() - standalone_tri_offset;
                    #pragma omp parallel for num_threads(get_omp_threads_limit()) reduction(+:any_dirty)
                    for (int i = 0; i < (int)standalone_count; ++i) {
                        const auto& tri = cached_triangles[standalone_tri_offset + i];
                        if (tri->vertexPositionsDirty) {
                            any_dirty += 1;
                            size_t local_v_offset = standalone_vertex_offset + i * 3;
                            vertex_buffer[local_v_offset + 0] = tri->getVertexPosition(0);
                            vertex_buffer[local_v_offset + 1] = tri->getVertexPosition(1);
                            vertex_buffer[local_v_offset + 2] = tri->getVertexPosition(2);
                            tri->vertexPositionsDirty = false;
                        }
                    }
                } else if (active_mesh_groups.empty()) {
                    // Fallback when active_mesh_groups is empty and standalone_tri_offset is not set/invalid
                    size_t valid_tri_count = cached_triangles.size();
                    if (valid_tri_count > 0) {
                        #pragma omp parallel for num_threads(get_omp_threads_limit()) reduction(+:any_dirty)
                        for (int i = 0; i < (int)valid_tri_count; ++i) {
                            const auto& tri = cached_triangles[i];
                            if (tri->vertexPositionsDirty) {
                                any_dirty += 1;
                                vertex_buffer[i * 3 + 0] = tri->getVertexPosition(0);
                                vertex_buffer[i * 3 + 1] = tri->getVertexPosition(1);
                                vertex_buffer[i * 3 + 2] = tri->getVertexPosition(2);
                                tri->vertexPositionsDirty = false;
                            }
                        }
                    }
                }

                if (any_dirty > 0) {
                    rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0);
                    rtcCommitGeometry(geom);
                    geometry_committed = true;
                }
            }
        }
    }

    // 2. Update Instance Transforms (Rigid Body)
    if (!instance_objects.empty()) {
        for (size_t geomID = 0; geomID < instance_objects.size(); ++geomID) {
            const auto& inst = instance_objects[geomID];
            if (!inst) continue;
            if (geomID == triangle_geom_id) continue;

            RTCGeometry inst_geom = rtcGetGeometry(scene, (unsigned)geomID);
            if (!inst_geom) continue;

            float transform[12];
            // Column 0 (X axis)
            transform[0] = inst->transform.m[0][0];
            transform[1] = inst->transform.m[1][0];
            transform[2] = inst->transform.m[2][0];
            
            // Column 1 (Y axis)
            transform[3] = inst->transform.m[0][1];
            transform[4] = inst->transform.m[1][1];
            transform[5] = inst->transform.m[2][1];
            
            // Column 2 (Z axis)
            transform[6] = inst->transform.m[0][2];
            transform[7] = inst->transform.m[1][2];
            transform[8] = inst->transform.m[2][2];
             
            // Column 3 (Translation)
            transform[9]  = inst->transform.m[0][3];
            transform[10] = inst->transform.m[1][3];
            transform[11] = inst->transform.m[2][3];

            rtcSetGeometryTransform(inst_geom, 0, RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR, transform);
            rtcCommitGeometry(inst_geom);
            geometry_committed = true;
        }
    }

    if (geometry_committed) {
         rtcCommitScene(scene);
    }
}

bool EmbreeBVH::occluded(const Ray& ray, float t_min, float t_max) const {
    RTCRayHit rayhit = {};
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.flags = RTC_RAY_QUERY_FLAG_NONE;
    int skip_guard = 0;

    while (true) {
        if (++skip_guard > 1024) {
            SCENE_LOG_WARN("[EmbreeBVH::occluded] Abort: transparency/volume skip guard exceeded.");
            return false;
        }

        rayhit.ray.org_x = ray.origin.x;
        rayhit.ray.org_y = ray.origin.y;
        rayhit.ray.org_z = ray.origin.z;
        rayhit.ray.dir_x = ray.direction.x;
        rayhit.ray.dir_y = ray.direction.y;
        rayhit.ray.dir_z = ray.direction.z;
        rayhit.ray.tnear = t_min;
        rayhit.ray.tfar = t_max;
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit, &args);

        if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) return false;

        const TriangleData* tri = nullptr;
        Material* mat = nullptr;
        Vec2 uv(0, 0);
        bool hit_instance = false;
        const bool resolved_surface = resolve_embree_surface_hit(*this, rayhit, tri, mat, uv, hit_instance);

        // 1. Triangle Alpha Test
        if (resolved_surface && !hit_instance) {
            if (mat && mat->isTransparent()) {
                float opacity = mat->get_opacity(uv);
                if (opacity <= 0.5f) {
                    if (!advance_embree_tmin(rayhit.ray.tfar, t_min, t_max, "[EmbreeBVH::occluded]")) {
                        return false;
                    }
                    continue; // Skip and check behind
                }
            }
            return true; // Opaque hit
        }
        else if (resolved_surface && hit_instance) {
            if (mat && mat->isTransparent()) {
                float opacity = mat->get_opacity(uv);
                if (opacity <= 0.5f) {
                    if (!advance_embree_tmin(rayhit.ray.tfar, t_min, t_max, "[EmbreeBVH::occluded]")) {
                        return false;
                    }
                    continue; // Skip and check behind
                }
            }
            return true; // Opaque hit
        }
        // 2. Volume Shadowing
        else if (rayhit.hit.geomID == vdb_geom_id) {
            unsigned int prim_id = rayhit.hit.primID;
            if (prim_id < vdb_objects.size()) {
                const VDBVolume* vdb = vdb_objects[prim_id];
                float t_enter, t_exit;
                if (vdb->intersectTransformedAABB(ray, t_min, t_max, t_enter, t_exit)) {
                    float step_size = 0.5f; 
                    if (vdb->volume_shader) step_size = vdb->volume_shader->quality.step_size * 2.0f;
                    step_size = compute_safe_volume_step(step_size, t_enter, t_exit);
                    auto& mgr = VDBVolumeManager::getInstance();
                    int vid = vdb->getVDBVolumeID();
                    Matrix4x4 inv = vdb->getInverseTransform();
                    float density_mult = (vdb->volume_shader ? vdb->volume_shader->density.multiplier : 1.0f) * vdb->density_scale;
                    
                    float t = t_enter + Vec3::random_float() * step_size;
                    float transmittance = 1.0f;
                    int steps = 0;
                    while (t < t_exit && steps < kMaxCpuVolumeMarchSteps) {
                        Vec3 local_pos = inv.transform_point(ray.at(t));
                        float d = mgr.sampleDensityCPU(vid, local_pos.x, local_pos.y, local_pos.z);
                        if (d > 0.001f) transmittance *= exp(-d * density_mult * step_size);
                        if (transmittance < 0.01f) break;
                        t += step_size;
                        ++steps;
                    }

                    if (Vec3::random_float() > transmittance) return true; // Blocked
                    else {
                        if (!advance_embree_tmin(t_exit, t_min, t_max, "[EmbreeBVH::occluded]")) {
                            return false;
                        }
                        continue; // Pass through volume
                    }
                }
            }
        }
        return true; // Fallback
    }
}


bool EmbreeBVH::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec, bool ignore_volumes) const {
    RTCRayHit rayhit = {};
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);

    // STOCHASTIC ALPHA TEST LOOP
    Material* mat = nullptr;
    Vec2 uv(0,0);
    const TriangleData* resolved_tri = nullptr;
    bool resolved_hit_instance = false;
    int skip_guard = 0;
    while (true) {
        if (++skip_guard > 1024) {
            SCENE_LOG_WARN("[EmbreeBVH::hit] Abort: transparency skip guard exceeded.");
            return false;
        }

        rayhit.ray.org_x = ray.origin.x;
        rayhit.ray.org_y = ray.origin.y;
        rayhit.ray.org_z = ray.origin.z;
        rayhit.ray.dir_x = ray.direction.x;
        rayhit.ray.dir_y = ray.direction.y;
        rayhit.ray.dir_z = ray.direction.z;
        rayhit.ray.tnear = t_min;
        rayhit.ray.tfar = t_max;
        rayhit.ray.mask = ignore_volumes ? 0x01 : 0xFFFFFFFF; // Masking handles volumes!
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit, &args);

        if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) return false;

        mat = nullptr;
        resolved_tri = nullptr;
        resolved_hit_instance = false;
        resolve_embree_surface_hit(*this, rayhit, resolved_tri, mat, uv, resolved_hit_instance);

        if (mat && mat->isTransparent()) {
            float opacity = mat->get_opacity(uv);
            if (opacity <= 0.5f) {
                // Transparent: continue ray from this point
                if (!advance_embree_tmin(rayhit.ray.tfar, t_min, t_max, "[EmbreeBVH::hit]")) {
                    return false;
                }
                continue;
            }
        }
        
        break; // Hit is opaque enough or it's a volume
    }

    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        const unsigned top_inst_id = rayhit.hit.instID[0];
        const bool hit_instance = resolved_hit_instance;
        
        // [New Logic]
        if (rayhit.hit.geomID == vdb_geom_id) {
            // Hit VDB Volume (User Geometry)
            unsigned int prim_id = rayhit.hit.primID;
            if (prim_id < vdb_objects.size()) {
                const VDBVolume* vdb = vdb_objects[prim_id];
                rec.t = rayhit.ray.tfar;
                rec.point = ray.at(rec.t);
                rec.vdb_volume = vdb;
                rec.normal = Vec3(0.0f, 1.0f, 0.0f); // Dummy normal (not used for volume)
                rec.set_face_normal(ray, rec.normal);
                return true;
            }
        }
        else if (!hit_instance && rayhit.hit.geomID == triangle_geom_id) {
            // Hit Local Triangle
            int prim_id = rayhit.hit.primID;
            const TriangleData& tri = resolved_tri ? *resolved_tri : triangle_data[prim_id];

            float u = rayhit.hit.u;
            float v = rayhit.hit.v;
            float w = 1.0f - u - v;

            if (tri.original_ptr) {
                rec.normal = (tri.original_ptr->getVertexNormal(0) * w + 
                              tri.original_ptr->getVertexNormal(1) * u + 
                              tri.original_ptr->getVertexNormal(2) * v).normalize();
                const auto [t0, t1, t2] = tri.original_ptr->getUVCoordinates();
                rec.u = t0.u * w + t1.u * u + t2.u * v;
                rec.v = t0.v * w + t1.v * u + t2.v * v;
            } else {
                rec.normal = Vec3(0.0f, 1.0f, 0.0f);
                rec.u = 0.0f;
                rec.v = 0.0f;
            }
            rec.interpolated_normal = rec.normal;
            rec.uv = Vec2(rec.u, rec.v); 
            rec.t = rayhit.ray.tfar;
            rec.point = ray.at(rec.t); 
            rec.set_face_normal(ray, rec.normal);
            rec.materialPtr = mat; // Reuse the alpha-test lookup; shared_ptr is resolved later
            rec.materialID = tri.materialID;
            rec.terrain_id = tri.terrain_id;
            rec.is_instance_hit = false; // Local geometry (e.g. Terrain)
            rec.triangle = tri.original_ptr; // Restore identity
            if (tri.original_ptr) {          // Faz 1: (mesh, faceIndex) handle
                rec.tri_mesh = tri.original_ptr->parentMesh.get();
                rec.tri_face = tri.original_ptr->faceIndex;
            }
            return true;
        }
        else if (hit_instance) {
            // Hit Instance
            rec.is_instance_hit = true; // Mark as instance for filtering
            const auto& inst = instance_objects[top_inst_id];
            if (inst) {
                // We need to fetch the child data
                auto child_bvh = std::dynamic_pointer_cast<EmbreeBVH>(inst->mesh);
                if (child_bvh) {
                     int prim_id = rayhit.hit.primID;
                     if (prim_id < child_bvh->triangle_data.size()) {
                          const TriangleData& tri = resolved_tri ? *resolved_tri : child_bvh->triangle_data[prim_id];

                          float u = rayhit.hit.u;
                          float v = rayhit.hit.v;
                          float w = 1.0f - u - v;
                          
                          // Interpolate Local Normal
                          Vec3 local_normal(0.0f, 1.0f, 0.0f);
                          if (tri.original_ptr) {
                              local_normal = (tri.original_ptr->getVertexNormal(0) * w + 
                                              tri.original_ptr->getVertexNormal(1) * u + 
                                              tri.original_ptr->getVertexNormal(2) * v).normalize();
                              const auto [t0, t1, t2] = tri.original_ptr->getUVCoordinates();
                              rec.u = t0.u * w + t1.u * u + t2.u * v;
                              rec.v = t0.v * w + t1.v * u + t2.v * v;
                          } else {
                              rec.u = 0.0f;
                              rec.v = 0.0f;
                          }
                          
                          // Transform Normal to World using Instance Transform
                          rec.normal = inst->transform.transform_vector(local_normal).normalize();
                          rec.interpolated_normal = rec.normal;
                          rec.uv = Vec2(rec.u, rec.v); 
                          rec.t = rayhit.ray.tfar;
                          rec.point = ray.at(rec.t); 
                          rec.set_face_normal(ray, rec.normal);
                          rec.materialPtr = mat; // Reuse the alpha-test lookup; shared_ptr is resolved later
                          rec.materialID = tri.materialID;
                          rec.terrain_id = tri.terrain_id;
                          rec.triangle = tri.original_ptr; // Restore identity (Source Mesh)
                          if (tri.original_ptr) {          // Faz 1: (mesh, faceIndex) handle
                              rec.tri_mesh = tri.original_ptr->parentMesh.get();
                              rec.tri_face = tri.original_ptr->faceIndex;
                          }
                          return true;
                     }
                }
            }
        }
    }
    return false;
}
void EmbreeBVH::clearAndRebuild(const std::vector<std::shared_ptr<Hittable>>& objects) {
    rtcReleaseScene(scene);
    scene = rtcNewScene(device);
    build(objects);
}

// OptixGeometryData icin tam donusumlu export metodu
OptixGeometryData EmbreeBVH::exportToOptixData() const {
    OptixGeometryData data;

    for (const auto& tri : triangle_data) {
        if (!tri.original_ptr) continue;
        uint32_t base_index = static_cast<uint32_t>(data.vertices.size());

        const Vec3& v0 = tri.original_ptr->getVertexPosition(0);
        const Vec3& v1 = tri.original_ptr->getVertexPosition(1);
        const Vec3& v2 = tri.original_ptr->getVertexPosition(2);
        
        const Vec3& n0 = tri.original_ptr->getVertexNormal(0);
        const Vec3& n1 = tri.original_ptr->getVertexNormal(1);
        const Vec3& n2 = tri.original_ptr->getVertexNormal(2);
        
        const auto [t0, t1, t2] = tri.original_ptr->getUVCoordinates();

        // Vertexler
        data.vertices.push_back(make_float3(v0.x, v0.y, v0.z));
        data.vertices.push_back(make_float3(v1.x, v1.y, v1.z));
        data.vertices.push_back(make_float3(v2.x, v2.y, v2.z));

        // Index
        data.indices.push_back(make_uint3(base_index, base_index + 1, base_index + 2));

        // Normaller
        data.normals.push_back(make_float3(n0.x, n0.y, n0.z));
        data.normals.push_back(make_float3(n1.x, n1.y, n1.z));
        data.normals.push_back(make_float3(n2.x, n2.y, n2.z));

        // UV
        data.uvs.push_back(make_float2(t0.x, t0.y));
        data.uvs.push_back(make_float2(t1.x, t1.y));
        data.uvs.push_back(make_float2(t2.x, t2.y));

        // Material
        GpuMaterial gpuMat = {};  // Zero-initialize all fields
        auto material = tri.getMaterialShared();
        if (material) {
            Vec3 albedoColor = material->getPropertyValue(material->albedoProperty, Vec2(0.5f, 0.5f));
            gpuMat.albedo = make_float3(albedoColor.x, albedoColor.y, albedoColor.z);
            gpuMat.opacity = material->get_opacity(Vec2(0.5f, 0.5f));
            gpuMat.roughness = material->getPropertyValue(material->roughnessProperty, Vec2(0.5f, 0.5f)).y;
            gpuMat.metallic = material->getPropertyValue(material->metallicProperty, Vec2(0.5f, 0.5f)).z;
            gpuMat.transmission = material->getPropertyValue(material->transmissionProperty, Vec2(0.5f, 0.5f)).x;
            Vec3 emissionColor = material->getEmission(Vec2(0.5f, 0.5f), Vec3(0,0,0));
            gpuMat.emission = make_float3(emissionColor.x, emissionColor.y, emissionColor.z);
            gpuMat.ior = material->getIOR();
            
            // SSS defaults (will be overwritten by PrincipledBSDF if available)
            gpuMat.subsurface = 0.0f;
            gpuMat.subsurface_color = make_float3(1.0f, 0.8f, 0.6f);
            gpuMat.subsurface_radius = make_float3(1.0f, 0.2f, 0.1f);
            gpuMat.subsurface_scale = 0.05f;
            gpuMat.subsurface_anisotropy = 0.0f;
            gpuMat.subsurface_ior = 1.4f;
            gpuMat.sss_use_random_walk = 1;
            gpuMat.sss_max_steps = 6;
            
            // Clear Coat defaults
            gpuMat.clearcoat = 0.0f;
            gpuMat.clearcoat_roughness = 0.03f;
            
            // Translucent default
            gpuMat.translucent = 0.0f;
            
            // Other defaults
            gpuMat.anisotropic = 0.0f;
            gpuMat.sheen = 0.0f;
            gpuMat.sheen_tint = 0.0f;
        }
        else {
            // Fallback pink material - properly initialize all fields
            gpuMat.albedo = make_float3(1.0f, 0.0f, 1.0f);
            gpuMat.opacity = 1.0f;
            gpuMat.roughness = 0.5f;
            gpuMat.metallic = 0.0f;
            gpuMat.transmission = 0.0f;
            gpuMat.emission = make_float3(0.0f, 0.0f, 0.0f);
            gpuMat.ior = 1.5f;
            
            // SSS defaults
            gpuMat.subsurface = 0.0f;
            gpuMat.subsurface_color = make_float3(1.0f, 0.8f, 0.6f);
            gpuMat.subsurface_radius = make_float3(1.0f, 0.2f, 0.1f);
            gpuMat.subsurface_scale = 0.05f;
            gpuMat.subsurface_anisotropy = 0.0f;
            gpuMat.subsurface_ior = 1.4f;
            
            // Clear Coat defaults
            gpuMat.clearcoat = 0.0f;
            gpuMat.clearcoat_roughness = 0.03f;
            
            // Translucent default
            gpuMat.translucent = 0.0f;
            
            // Other defaults
            gpuMat.anisotropic = 0.0f;
            gpuMat.sheen = 0.0f;
            gpuMat.sheen_tint = 0.0f;
        }

        // Benzersiz materyal kontrolu (basit ekle, hash'e gerek yok burada)
        int matIndex = -1;
        for (int m = 0; m < data.materials.size(); ++m) {
            if (memcmp(&data.materials[m], &gpuMat, sizeof(GpuMaterial)) == 0) {
                matIndex = m;
                break;
            }
        }
        if (matIndex == -1) {
            data.materials.push_back(gpuMat);
            matIndex = static_cast<int>(data.materials.size()) - 1;
        }
        data.material_indices.push_back(matIndex);
    }

    return data;
}
