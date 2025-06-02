﻿#include "EmbreeBVH.h"
#include <cassert>
#include <Volumetric.h>

EmbreeBVH::EmbreeBVH() {
    device = rtcNewDevice(nullptr);
    scene = rtcNewScene(device);
}

EmbreeBVH::~EmbreeBVH() {
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

void EmbreeBVH::build(const std::vector<std::shared_ptr<Hittable>>& objects) {
    std::vector<Vec3> vertices;
    std::vector<std::array<unsigned, 3>> indices;
    triangle_data.clear();

    unsigned vert_offset = 0;

    for (const auto& obj : objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (!tri) continue;

        // Vertex buffer'a üçgenin köşe noktalarını ekle
        vertices.push_back(tri->v0);
        vertices.push_back(tri->v1);
        vertices.push_back(tri->v2);

        // Index buffer (3'lü indeks)
        indices.push_back({ vert_offset, vert_offset + 1, vert_offset + 2 });

        triangle_data.push_back({
       tri->v0, tri->v1, tri->v2,
       tri->n0, tri->n1, tri->n2,
       tri->t0, tri->t1, tri->t2,
       tri->mat_ptr
            });

        vert_offset += 3;
    }

    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Vertex buffer
    Vec3* vertex_buffer = (Vec3*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        sizeof(Vec3), vertices.size()
    );
    std::memcpy(vertex_buffer, vertices.data(), sizeof(Vec3) * vertices.size());

    // Index buffer
    unsigned* index_buffer = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        sizeof(unsigned) * 3, indices.size()
    );
    std::memcpy(index_buffer, indices.data(), sizeof(unsigned) * 3 * indices.size());

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);

    rtcCommitScene(scene);

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
// Bu her karede çağrılır
void EmbreeBVH::updateGeometryFromTrianglesFromSource(const std::vector<std::shared_ptr<Hittable>>& objects) {
    RTCGeometry geom = rtcGetGeometry(scene, 0); // 0 → geometry ID

    Vec3* vertex_buffer = (Vec3*)rtcGetGeometryBufferData(
        geom, RTC_BUFFER_TYPE_VERTEX, 0 // 0 → buffer slot
    );

    for (size_t i = 0; i < objects.size(); ++i) {
        auto tri = std::dynamic_pointer_cast<Triangle>(objects[i]);
        if (!tri) continue;

        vertex_buffer[i * 3 + 0] = tri->v0;
        vertex_buffer[i * 3 + 1] = tri->v1;
        vertex_buffer[i * 3 + 2] = tri->v2;

        // TriangleData içindeki gölge bilgisi de güncellensin
        triangle_data[i].v0 = tri->v0;
        triangle_data[i].v1 = tri->v1;
        triangle_data[i].v2 = tri->v2;

        if (triangle_data[i].n0 != Vec3()) {
            triangle_data[i].n0 = tri->n0;
            triangle_data[i].n1 = tri->n1;
            triangle_data[i].n2 = tri->n2;
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

        float opacity = tri.material->get_opacity(uv);
       
        if (tri.material->type() == MaterialType::Volumetric) {
            double distance = rayhit.ray.tfar;
            const auto* volume = static_cast<const Volumetric*>(tri.material.get());

            // Yönlü ışığın geçtiği noktadaki örnekleme
            Vec3 sample_point = ray.origin + ray.direction * distance;
            double density = volume->calculate_density(sample_point);
            double variability = 0.5 + 0.5 * volume->noise->noise(sample_point * 2.0); // 0.5–1.0 arası
            double attenuation = exp(-density * distance * variability);


            // Probabilistik karar: daha yoğun → daha düşük pass olasılığı
            if (Vec3::random_double() < attenuation)
                return false; // Işık geçti, zayıf gölge
        }


        if (opacity < 1.0)
            return false; // ışık geçti
      
       
        return true; // gölgede
    }

    return false;
}
void EmbreeBVH::buildFromTriangleData(const std::vector<TriangleData>& triangles) {
    triangle_data = triangles; // Kopyala

    std::vector<Vec3> vertices;
    std::vector<std::array<unsigned, 3>> indices;

    unsigned vert_offset = 0;
    for (const auto& tri : triangles) {
        vertices.push_back(tri.v0);
        vertices.push_back(tri.v1);
        vertices.push_back(tri.v2);

        indices.push_back({ vert_offset, vert_offset + 1, vert_offset + 2 });
        vert_offset += 3;
    }

    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    Vec3* vertex_buffer = (Vec3*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        sizeof(Vec3), vertices.size()
    );
    std::memcpy(vertex_buffer, vertices.data(), sizeof(Vec3) * vertices.size());

    unsigned* index_buffer = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        sizeof(unsigned) * 3, indices.size()
    );
    std::memcpy(index_buffer, indices.data(), sizeof(unsigned) * 3 * indices.size());

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);

    rtcCommitScene(scene);
}

bool EmbreeBVH::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
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
        const TriangleData& tri = triangle_data[prim_id];  // doğrudan index
        float u = rayhit.hit.u;
        float v = rayhit.hit.v;
        float w = 1.0f - u - v;

        Vec3 normal = (tri.n0 * w + tri.n1 * u + tri.n2 * v).normalize();
        Vec2 uv = tri.t0 * w + tri.t1 * u + tri.t2 * v;
        rec.uv = uv;
		rec.u = uv.u;
		rec.v = uv.v;
        rec.t = rayhit.ray.tfar;
        rec.point = ray.at(rec.t);
		rec.normal = normal;
	
        rec.set_face_normal(ray, normal);
        rec.material = tri.material;    

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
        GpuMaterial gpuMat;
        if (tri.material) {
            Vec3 albedoColor = tri.material->getPropertyValue(tri.material->albedoProperty, Vec2(0.5f, 0.5f));
            gpuMat.albedo = make_float3(albedoColor.x, albedoColor.y, albedoColor.z);
            gpuMat.roughness = tri.material->getPropertyValue(tri.material->roughnessProperty, Vec2(0.5f, 0.5f)).y;
            gpuMat.metallic = tri.material->getPropertyValue(tri.material->metallicProperty, Vec2(0.5f, 0.5f)).z;
            gpuMat.transmission = tri.material->getPropertyValue(tri.material->transmissionProperty, Vec2(0.5f, 0.5f)).x;
            gpuMat.ior = tri.material->getIOR();
        }
        else {
            gpuMat = { make_float3(1, 0, 1), 0.5f, 0.0f, 0.0f, 1.5f }; // fallback pembe
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