#include "EmbreeBVH.h"
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
        // Configure scene for optimal BVH build performance
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
    // [VERBOSE] SCENE_LOG_INFO("[EmbreeBVH::build] Build started"); // Runs on every manipulation

    // First pass: Count triangles
    size_t tri_count = 0;
    for (const auto& obj : objects) {
        if (std::dynamic_pointer_cast<Triangle>(obj)) {
            tri_count++;
        }
    }

    if (tri_count == 0) {
        SCENE_LOG_WARN("[EmbreeBVH::build] No triangles found");
        return;
    }

    // [VERBOSE] SCENE_LOG_INFO("[EmbreeBVH::build] Triangle count = " + std::to_string(tri_count));

    // Pre-allocate all vectors to avoid reallocation overhead
    triangle_data.clear();
    triangle_data.reserve(tri_count);

    // Create geometry
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    if (!geom) {
        SCENE_LOG_ERROR("[EmbreeBVH::build] Failed to create geometry");
        return;
    }

    // Allocate Embree buffers directly (no intermediate vectors!)
    Vec3* vertex_buffer = (Vec3*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        sizeof(Vec3), tri_count * 3
    );
    if (!vertex_buffer) {
        SCENE_LOG_ERROR("[EmbreeBVH::build] Failed to allocate vertex buffer");
        rtcReleaseGeometry(geom);
        return;
    }

    unsigned* index_buffer = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        sizeof(unsigned) * 3, tri_count
    );
    if (!index_buffer) {
        SCENE_LOG_ERROR("[EmbreeBVH::build] Failed to allocate index buffer");
        rtcReleaseGeometry(geom);
        return;
    }

    // Second pass: Fill buffers directly (single pass, no redundant copies)
    size_t tri_idx = 0;
    for (const auto& obj : objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (!tri) continue;

        unsigned base_idx = static_cast<unsigned>(tri_idx * 3);

        // Write directly to Embree's vertex buffer
        vertex_buffer[base_idx + 0] = tri->getVertexPosition(0);
        vertex_buffer[base_idx + 1] = tri->getVertexPosition(1);
        vertex_buffer[base_idx + 2] = tri->getVertexPosition(2);

        // Write directly to Embree's index buffer
        index_buffer[tri_idx * 3 + 0] = base_idx + 0;
        index_buffer[tri_idx * 3 + 1] = base_idx + 1;
        index_buffer[tri_idx * 3 + 2] = base_idx + 2;

        // Store triangle data (single copy)
        triangle_data.push_back({
            tri->getVertexPosition(0), tri->getVertexPosition(1), tri->getVertexPosition(2),
            tri->getVertexNormal(0), tri->getVertexNormal(1), tri->getVertexNormal(2),
            tri->t0, tri->t1, tri->t2,
            tri->getMaterialID()
        });

        tri_idx++;
    }

    auto data_prep_end = std::chrono::high_resolution_clock::now();
    auto data_prep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(data_prep_end - build_start).count();
    // [VERBOSE] SCENE_LOG_INFO("[EmbreeBVH::build] Data preparation took " + std::to_string(data_prep_ms) + " ms");

    // Commit geometry
    rtcCommitGeometry(geom);
    // [VERBOSE] SCENE_LOG_INFO("[EmbreeBVH::build] Geometry committed");

    unsigned geomID = rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    // [VERBOSE] SCENE_LOG_INFO("[EmbreeBVH::build] Geometry attached with ID = " + std::to_string(geomID));

    // Commit scene (this is where BVH construction happens)
    auto bvh_start = std::chrono::high_resolution_clock::now();
    rtcCommitScene(scene);
    auto bvh_end = std::chrono::high_resolution_clock::now();
    
    auto bvh_build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(bvh_end - bvh_start).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(bvh_end - build_start).count();
    
    // [VERBOSE] Build timing logs - disabled to reduce log volume during manipulation
    // SCENE_LOG_INFO("[EmbreeBVH::build] BVH construction took " + std::to_string(bvh_build_ms) + " ms");
    // SCENE_LOG_INFO("[EmbreeBVH::build] Total build time: " + std::to_string(total_ms) + " ms");
    // SCENE_LOG_INFO("[EmbreeBVH::build] BVH build completed");
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
void EmbreeBVH::updateGeometryFromTrianglesFromSource(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (objects.empty() || triangle_data.empty()) return;
    
    RTCGeometry geom = rtcGetGeometry(scene, 0); // 0 → geometry ID
    if (!geom) return;

    Vec3* vertex_buffer = (Vec3*)rtcGetGeometryBufferData(
        geom, RTC_BUFFER_TYPE_VERTEX, 0 // 0 → buffer slot
    );
    if (!vertex_buffer) return;

    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIMIZATION: Use REFIT quality for faster BVH updates during animation
    // ═══════════════════════════════════════════════════════════════════════════
    // RTC_BUILD_QUALITY_REFIT tells Embree to update the existing BVH structure
    // instead of rebuilding from scratch. This is 2-5x faster for animations
    // where only vertex positions change but topology remains the same.
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
    if (valid_tri_count == 0) return;

    // Parallelize vertex update on the DENSE list (safe indexing)
    #pragma omp parallel for
    for (int i = 0; i < (int)valid_tri_count; ++i) {
        const auto& tri = active_triangles[i];
        
        // Direct write to mapped Embree buffer
        vertex_buffer[i * 3 + 0] = tri->getVertexPosition(0);
        vertex_buffer[i * 3 + 1] = tri->getVertexPosition(1);
        vertex_buffer[i * 3 + 2] = tri->getVertexPosition(2);

        // Update shadow cache in TriangleData if necessary
        // Note: Check bounds for triangle_data just in case
        // (Assuming triangle_data resized in build() to match tri count)
         if (i < triangle_data.size()) { // Safe but maybe slow inside loop?
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
    rtcCommitScene(scene);
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
        int prim_id = rayhit.hit.primID;
        const TriangleData& tri = triangle_data[prim_id];

        float u = rayhit.hit.u;
        float v = rayhit.hit.v;
        float w = 1.0f - u - v;

        Vec2 uv = tri.t0 * w + tri.t1 * u + tri.t2 * v;

        auto material = tri.getMaterialShared();
        float opacity = material ? material->get_opacity(uv) : 1.0f;
       
        if (material && material->type() == MaterialType::Volumetric) {
            double distance = rayhit.ray.tfar;
            const auto* volume = static_cast<const Volumetric*>(material.get());

            // Yönlü ışığın geçtiği noktadaki örnekleme
            Vec3 sample_point = ray.origin + ray.direction * distance;
            double density = volume->calculate_density(sample_point);
            double variability = 0.5 + 0.5 * volume->noise->noise(sample_point * 2.0); // 0.5–1.0 arası
            double attenuation = exp(-density * distance * variability);


            // Probabilistik karar: daha yoğun → daha düşük pass olasılığı
            if (Vec3::random_float() < attenuation)
                return false; // Işık geçti, zayıf gölge
        }

        if (Vec3::random_float() > opacity)
            return false; // ışık geçti
        return true; // gölgede
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
        int prim_id = rayhit.hit.primID;
        const TriangleData& tri = triangle_data[prim_id];

        // Barycentric koordinatları doğrudan kullan
        float u = rayhit.hit.u;
        float v = rayhit.hit.v;
        float w = 1.0f - u - v;

       
        rec.normal = (tri.n0 * w + tri.n1 * u + tri.n2 * v).normalize();
        rec.u = tri.t0.u * w + tri.t1.u * u + tri.t2.u * v;
        rec.v = tri.t0.v * w + tri.t1.v * u + tri.t2.v * v;
        rec.uv = Vec2(rec.u, rec.v); // Gerekliyse
        rec.t = rayhit.ray.tfar;
        rec.point = ray.at(rec.t);
        rec.set_face_normal(ray, rec.normal);
        rec.material = tri.getMaterialShared();
        rec.materialID = tri.materialID;

        return true;
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
