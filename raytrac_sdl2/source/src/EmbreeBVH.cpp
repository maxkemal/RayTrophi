#include "EmbreeBVH.h"
#include "HittableInstance.h"
#include "VDBVolume.h" // Add VDB support
#include <cassert>
#include <chrono>
#include <Volumetric.h>

// Static member initialization
RTCDevice EmbreeBVH::device = nullptr;

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

    if (!device) {
        SCENE_LOG_ERROR("[EmbreeBVH::build] Device is null!");
        return;
    }
    
    // Create new scene if needed (clearGeometry usually does this but let's be safe)
    if (!scene) {
        scene = rtcNewScene(device);
        rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_MEDIUM);
        rtcSetSceneFlags(scene, RTC_SCENE_FLAG_DYNAMIC | RTC_SCENE_FLAG_ROBUST); // ROBUST important for instancing
    }

    // 1. Separate objects
    std::vector<std::shared_ptr<Triangle>> local_triangles;
    std::vector<std::shared_ptr<HittableInstance>> local_instances;
    
    local_triangles.reserve(objects.size());
    local_instances.reserve(objects.size());
    vdb_objects.reserve(objects.size());

    for (const auto& obj : objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            local_triangles.push_back(tri);
        }
        else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            local_instances.push_back(inst);
        }
        else if (auto vdb = std::dynamic_pointer_cast<VDBVolume>(obj)) {
            vdb_objects.push_back(vdb.get());
        }
    }

    // 2. Build Triangle Geometry (if any)
    if (!local_triangles.empty()) {
        size_t tri_count = local_triangles.size();
        triangle_data.clear();
        triangle_data.reserve(tri_count);

        RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

        Vec3* vertex_buffer = (Vec3*)rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
            sizeof(Vec3), tri_count * 3
        );
        unsigned* index_buffer = (unsigned*)rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
            sizeof(unsigned) * 3, tri_count
        );

        // Fill buffers
        #pragma omp parallel for
        for (int i = 0; i < (int)tri_count; ++i) {
            const auto& tri = local_triangles[i];
            
            // Vertices
            vertex_buffer[i * 3 + 0] = tri->getVertexPosition(0);
            vertex_buffer[i * 3 + 1] = tri->getVertexPosition(1);
            vertex_buffer[i * 3 + 2] = tri->getVertexPosition(2);

            // Indices
            index_buffer[i * 3 + 0] = i * 3 + 0;
            index_buffer[i * 3 + 1] = i * 3 + 1;
            index_buffer[i * 3 + 2] = i * 3 + 2;
        }

        // Fill triangle_data serial (or parallel if safe)
        for (const auto& tri : local_triangles) {
            triangle_data.push_back({
                tri->getVertexPosition(0), tri->getVertexPosition(1), tri->getVertexPosition(2),
                tri->getVertexNormal(0), tri->getVertexNormal(1), tri->getVertexNormal(2),
                tri->t0, tri->t1, tri->t2,
                tri->getMaterialID(),
                tri.get() // Store original pointer
            });
        }

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
            
            // Set Transform (Embree uses column-major float[12] for 3x4 affine)
            // Matrix4x4 is likely row-major or we need to check.
            // Matrix4x4 typically: m[row][col]
            // Embree expects: 
            // xx yx zx tx
            // xy yy zy ty
            // xz yz zz tz
            
            // Standard format for rtcSetGeometryTransform:
            // "The transformation matrix is passed as a pointer to an array of 12 floats in column-major order."
            
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
            unsigned geomID = rtcAttachGeometry(scene, inst_geom);
            rtcReleaseGeometry(inst_geom);
            
            // Store mapping
            if (instance_objects.size() <= geomID) {
                instance_objects.resize(geomID + 1, nullptr);
            }
            instance_objects[geomID] = inst.get();
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
        
        float t_enter, t_exit;
        // Pass -infinity instead of tnear to detect if we are inside the box (t_enter < tnear)
        if (vdb->intersectTransformedAABB(r, -std::numeric_limits<float>::infinity(), tfar, t_enter, t_exit)) {
            
            float reported_hit = t_enter;
            if (reported_hit < tnear) reported_hit = tnear;
            
            // Check if valid interval exists (exit must be AFTER near plane)
            bool is_inside_or_enter = (t_exit > tnear);

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
        
        float t_enter, t_exit;
        if (vdb->intersectTransformedAABB(r, tnear, tfar, t_enter, t_exit)) {
            // CRITICAL FIX: VDB volumes are semi-transparent!
            // DO NOT set tfar = -infinity (full occlusion)
            // Instead, let Renderer handle volumetric shadow ray marching
            // by NOT reporting occlusion here.
            // 
            // The shadow ray will pass through and Renderer's 
            // calculate_direct_lighting_single_light() will do proper
            // density-based transmittance calculation.
            //
            // If we set tfar = -inf, object surfaces behind VDB would be
            // completely shadowed (solid box shadow).
            // By skipping, the light reaches surfaces and VDB shadow
            // is computed via ray marching in the main hit path.
            
            // NOTE: This means EmbreeBVH.occluded() won't report VDB as blocker
            // but the main hit() path will still find VDB and render it correctly.
            // Shadow rays in Renderer use bvh->hit(), not bvh->occluded().
            
            continue; // Don't occlude - let ray pass through for volumetric handling
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
        vertex_buffer[i * 3 + 0] = tri.v0;
        vertex_buffer[i * 3 + 1] = tri.v1;
        vertex_buffer[i * 3 + 2] = tri.v2;
    }

    rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0);
    rtcCommitGeometry(geom);
    rtcCommitScene(scene);
}
// Bu her karede çağrılır - OPTIMIZE WITH REFIT
// Bu her karede çağrılır - OPTIMIZE WITH REFIT
void EmbreeBVH::updateGeometryFromTrianglesFromSource(const std::vector<std::shared_ptr<Hittable>>& objects) {
    bool geometry_committed = false;

    // 1. Update Triangle Geometry (Deformable)
    if (triangle_geom_id != RTC_INVALID_GEOMETRY_ID) {
        RTCGeometry geom = rtcGetGeometry(scene, triangle_geom_id);
        if (geom) {
            Vec3* vertex_buffer = (Vec3*)rtcGetGeometryBufferData(geom, RTC_BUFFER_TYPE_VERTEX, 0);
            if (vertex_buffer) {
                // RTC_BUILD_QUALITY_REFIT tells Embree to update existing BVH structure
                rtcSetGeometryBuildQuality(geom, RTC_BUILD_QUALITY_REFIT);

                // FIX: Ensure perfectly aligned indexing between 'objects' and Embree buffer.
                // Since 'objects' might contain non-Triangle items (which are skipped in build),
                // we must first extract all valid Triangles to match the dense buffer layout.
                std::vector<std::shared_ptr<Triangle>> active_triangles;
                active_triangles.reserve(objects.size());
                
                for (const auto& obj : objects) {
                    if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                        active_triangles.push_back(tri);
                    }
                }
                
                size_t valid_tri_count = active_triangles.size();
                if (valid_tri_count > 0) {
                     // Parallelize vertex update on the DENSE list (safe indexing)
                    #pragma omp parallel for
                    for (int i = 0; i < (int)valid_tri_count; ++i) {
                        const auto& tri = active_triangles[i];
                        
                        // Direct write to mapped Embree buffer
                        vertex_buffer[i * 3 + 0] = tri->getVertexPosition(0);
                        vertex_buffer[i * 3 + 1] = tri->getVertexPosition(1);
                        vertex_buffer[i * 3 + 2] = tri->getVertexPosition(2);

                        // Update shadow cache in TriangleData if necessary
                        if (i < triangle_data.size()) {
                            triangle_data[i].v0 = tri->getVertexPosition(0);
                            triangle_data[i].v1 = tri->getVertexPosition(1);
                            triangle_data[i].v2 = tri->getVertexPosition(2);

                            if (triangle_data[i].n0 != Vec3()) {
                                triangle_data[i].n0 = tri->getVertexNormal(0);
                                triangle_data[i].n1 = tri->getVertexNormal(1);
                                triangle_data[i].n2 = tri->getVertexNormal(2);
                            }
                        }
                    }

                    rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0);
                    rtcCommitGeometry(geom);
                    geometry_committed = true;
                }
            }
        }
    }

    // 2. Update Instance Transforms (Rigid Body)
    if (!instance_objects.empty()) {
        // Iterate over stored instances and update their transforms
        for (size_t geomID = 0; geomID < instance_objects.size(); ++geomID) {
            const HittableInstance* inst = instance_objects[geomID];
            if (!inst) continue;
            
            // Skip if this slot matches triangle_geom_id (though instance_objects should be null there usually? 
            // no, instance_objects is sparse or we loop carefully?
            // current implementation: instance_objects resizes to Max ID.
            // If triangle_geom_id is 0, instance_objects[0] is likely null/garbage if not initialized carefully?
            // In build(), we resize instance_objects but only write to instance slots.
            // Initialize with nullptr.
            if (geomID == triangle_geom_id) continue;

            RTCGeometry inst_geom = rtcGetGeometry(scene, (unsigned)geomID);
            if (!inst_geom) continue;

            // Check if it's actually an instance geometry? 
            // Assume yes based on our tracking.

            // Update Transform
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

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.flags = RTC_RAY_QUERY_FLAG_NONE;

    rtcIntersect1(scene, &rayhit, &args);

    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        
        // Use geometry ID to distinguish triangle buffer vs instances
        if (rayhit.hit.geomID == triangle_geom_id) {
            // Local Triangles
            int prim_id = rayhit.hit.primID;
            const TriangleData& tri = triangle_data[prim_id];

            float u = rayhit.hit.u;
            float v = rayhit.hit.v;
            float w = 1.0f - u - v;

            Vec2 uv = tri.t0 * w + tri.t1 * u + tri.t2 * v;
            auto material = tri.getMaterialShared();
            float opacity = material ? material->get_opacity(uv) : 1.0f;
            
            // Basic alpha testing (approx)
             if (Vec3::random_float() > opacity) return false;
            // Note: Full volumetric check omitted here for brevity or needs to be consistent
            return true;
        } 
        else if (rayhit.hit.geomID < instance_objects.size()) {
            // Instance
            const HittableInstance* inst = instance_objects[rayhit.hit.geomID];
            if (inst) {
                 // Occluded? Yes.
                 // If we need alpha testing, we have to look up the material.
                 // For now, assume opaque or return true for shadow to be safe/fast.
                 // Correct logic requires recursing or peeking the child BVH.
                 auto child_bvh = std::dynamic_pointer_cast<EmbreeBVH>(inst->mesh);
                 if (child_bvh) {
                      int prim_id = rayhit.hit.primID;
                      if (prim_id < child_bvh->triangle_data.size()) {
                            const TriangleData& tri = child_bvh->triangle_data[prim_id];
                            float u = rayhit.hit.u;
                            float v = rayhit.hit.v;
                            float w = 1.0f - u - v;
                            Vec2 uv = tri.t0 * w + tri.t1 * u + tri.t2 * v;
                            auto material = tri.getMaterialShared();
                            float opacity = material ? material->get_opacity(uv) : 1.0f;
                            if (Vec3::random_float() > opacity) return false;
                      }
                 }
                 return true;
            }
        }
    }
    return false;
}
void EmbreeBVH::buildFromTriangleData(const std::vector<TriangleData>& triangles) {
    triangle_data = triangles; // Store triangle data

    if (triangles.empty()) return;

    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Allocate buffers directly
    Vec3* vertex_buffer = (Vec3*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        sizeof(Vec3), triangles.size() * 3
    );

    unsigned* index_buffer = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        sizeof(unsigned) * 3, triangles.size()
    );

    // Write directly to buffers (no intermediate vectors!)
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& tri = triangles[i];
        unsigned base_idx = static_cast<unsigned>(i * 3);

        // Vertices
        vertex_buffer[base_idx + 0] = tri.v0;
        vertex_buffer[base_idx + 1] = tri.v1;
        vertex_buffer[base_idx + 2] = tri.v2;

        // Indices
        index_buffer[i * 3 + 0] = base_idx + 0;
        index_buffer[i * 3 + 1] = base_idx + 1;
        index_buffer[i * 3 + 2] = base_idx + 2;
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);

    rtcCommitScene(scene);
}

bool EmbreeBVH::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
    RTCRayHit rayhit = {};
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

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);

    rtcIntersect1(scene, &rayhit, &args);

    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        
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
        else if (rayhit.hit.geomID == triangle_geom_id) {
            // Hit Local Triangle
            int prim_id = rayhit.hit.primID;
            const TriangleData& tri = triangle_data[prim_id];

            float u = rayhit.hit.u;
            float v = rayhit.hit.v;
            float w = 1.0f - u - v;

            rec.normal = (tri.n0 * w + tri.n1 * u + tri.n2 * v).normalize();
            rec.u = tri.t0.u * w + tri.t1.u * u + tri.t2.u * v;
            rec.v = tri.t0.v * w + tri.t1.v * u + tri.t2.v * v;
            rec.uv = Vec2(rec.u, rec.v); 
            rec.t = rayhit.ray.tfar;
            rec.point = ray.at(rec.t); 
            rec.set_face_normal(ray, rec.normal);
            rec.material = tri.getMaterialShared();
            rec.materialID = tri.materialID;
            rec.is_instance_hit = false; // Local geometry (e.g. Terrain)
            rec.triangle = tri.original_ptr; // Restore identity
            return true;
        } 
        else if (rayhit.hit.geomID < instance_objects.size()) {
            // Hit Instance
            rec.is_instance_hit = true; // Mark as instance for filtering
            const HittableInstance* inst = instance_objects[rayhit.hit.geomID];
            if (inst) {
                // We need to fetch the child data
                auto child_bvh = std::dynamic_pointer_cast<EmbreeBVH>(inst->mesh);
                if (child_bvh) {
                     int prim_id = rayhit.hit.primID;
                     if (prim_id < child_bvh->triangle_data.size()) {
                         const TriangleData& tri = child_bvh->triangle_data[prim_id];

                         float u = rayhit.hit.u;
                         float v = rayhit.hit.v;
                         float w = 1.0f - u - v;
                         
                          // Interpolate Local Normal
                         Vec3 local_normal = (tri.n0 * w + tri.n1 * u + tri.n2 * v).normalize();
                         
                         // Transform Normal to World using Instance Transform
                         // Assuming uniform scale, transform_vector (rotation) is sufficient.
                         // For non-uniform, use inverse transpose (not easily available without computing it)
                         // But we can key off inv_transform if available or just use transform_vector
                         rec.normal = inst->transform.transform_vector(local_normal).normalize();

                         rec.u = tri.t0.u * w + tri.t1.u * u + tri.t2.u * v;
                         rec.v = tri.t0.v * w + tri.t1.v * u + tri.t2.v * v;
                         rec.uv = Vec2(rec.u, rec.v); 
                         rec.t = rayhit.ray.tfar;
                         rec.point = ray.at(rec.t); 
                         rec.set_face_normal(ray, rec.normal);
                         rec.material = tri.getMaterialShared();
                         rec.materialID = tri.materialID;
                         rec.triangle = tri.original_ptr; // Restore identity (Source Mesh)
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
        uint32_t base_index = static_cast<uint32_t>(data.vertices.size());

        // Vertexler
        data.vertices.push_back(make_float3(tri.v0.x, tri.v0.y, tri.v0.z));
        data.vertices.push_back(make_float3(tri.v1.x, tri.v1.y, tri.v1.z));
        data.vertices.push_back(make_float3(tri.v2.x, tri.v2.y, tri.v2.z));

        // Index
        data.indices.push_back(make_uint3(base_index, base_index + 1, base_index + 2));

        // Normaller
        data.normals.push_back(make_float3(tri.n0.x, tri.n0.y, tri.n0.z));
        data.normals.push_back(make_float3(tri.n1.x, tri.n1.y, tri.n1.z));
        data.normals.push_back(make_float3(tri.n2.x, tri.n2.y, tri.n2.z));

        // UV
        data.uvs.push_back(make_float2(tri.t0.x, tri.t0.y));
        data.uvs.push_back(make_float2(tri.t1.x, tri.t1.y));
        data.uvs.push_back(make_float2(tri.t2.x, tri.t2.y));

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
