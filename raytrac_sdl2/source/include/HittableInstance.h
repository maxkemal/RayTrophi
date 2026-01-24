/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          HittableInstance.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include "Hittable.h"
#include "Matrix4x4.h"
#include "AABB.h"
#include <memory>
#include <vector>
#include "Vec3.h"
class HittableInstance : public Hittable {
public:
	std::shared_ptr<Hittable> mesh;
    // Helper for OptiX to access original geometry for BLAS building
    std::shared_ptr<std::vector<std::shared_ptr<class Triangle>>> source_triangles;

	Matrix4x4 transform;
	Matrix4x4 inv_transform;
	std::string node_name;

	AABB bbox;
    bool visible = true; // Visibility flag for incremental updates
    std::vector<int> optix_instance_ids; // OptiX Instance IDs for fast updates (supports multi-material splits)
    // We also need to store material assignment if the instance overrides it?
    // For now, assume material is in the mesh.

	HittableInstance(std::shared_ptr<Hittable> m, std::shared_ptr<std::vector<std::shared_ptr<class Triangle>>> tris, const Matrix4x4& t, const std::string& name)
		: mesh(m), source_triangles(tris), transform(t), node_name(name) {
		updateBounds();
	}

    void setTransform(const Matrix4x4& t) {
        transform = t;
        calculateInverse(); // Use helper or inline
        updateBounds();
    }

private:
    void calculateInverse() {
        inv_transform = transform.inverse();
    }

    void updateBounds() {
        inv_transform = transform.inverse(); // Ensure inverse is up to date

		// Compute world space AABB
        // Get local AABB
        AABB local_box;
        if (!mesh || !mesh->bounding_box(0, 0, local_box)) {
            // singular/empty mesh or null
            bbox = AABB(transform.getTranslation(), transform.getTranslation());
            return;
        }

        // Transform all 8 corners
        Vec3 min = local_box.min;
        Vec3 max = local_box.max;

        Vec3 corners[8];
        corners[0] = Vec3(min.x, min.y, min.z);
        corners[1] = Vec3(max.x, min.y, min.z);
        corners[2] = Vec3(min.x, max.y, min.z);
        corners[3] = Vec3(max.x, max.y, min.z);
        corners[4] = Vec3(min.x, min.y, max.z);
        corners[5] = Vec3(max.x, min.y, max.z);
        corners[6] = Vec3(min.x, max.y, max.z);
        corners[7] = Vec3(max.x, max.y, max.z);

        Vec3 new_min(1e9, 1e9, 1e9);
        Vec3 new_max(-1e9, -1e9, -1e9);

        for (int i = 0; i < 8; i++) {
            // Apply Transform
            Vec3 p = transform.transform_point(corners[i]);
            new_min = (Vec3::min)(new_min, p);
            new_max = (Vec3::max)(new_max, p);
        }
        bbox = AABB(new_min, new_max);
    }

public:

	virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes = false) const override {
		// Transform Hit Logic
        // 1. Transform ray to object space
        Vec3 origin = inv_transform.transform_point(r.origin);
        Vec3 direction = inv_transform.transform_vector(r.direction);
        
        Ray object_ray(origin, direction);

        // 2. Hit in object space
        if (!mesh->hit(object_ray, t_min, t_max, rec, ignore_volumes))
            return false;

        // 3. Transform result back to world space
        rec.point = transform.transform_point(rec.point);
        rec.normal = transform.transform_vector(rec.normal).normalize();
        
        // Re-orient face normal
        rec.set_face_normal(r, rec.normal);
        
        return true;
	}

    virtual bool occluded(const Ray& r, float t_min, float t_max) const override {
        // Transform Ray
        Vec3 origin = inv_transform.transform_point(r.origin);
        Vec3 direction = inv_transform.transform_vector(r.direction);
        Ray object_ray(origin, direction);
        
        // Forward to mesh (which might be a VDB with stochastic transparency)
        return mesh->occluded(object_ray, t_min, t_max);
    }

    virtual void hit_packet(const RayPacket& r, float t_min, float t_max, HitRecordPacket& rec, bool ignore_volumes = false) const override {
        if (!visible) return;

        // 1. Transform ray packet to local space
        RayPacket local_ray;
        
        __m256 m00 = _mm256_set1_ps(inv_transform.m[0][0]);
        __m256 m01 = _mm256_set1_ps(inv_transform.m[0][1]);
        __m256 m02 = _mm256_set1_ps(inv_transform.m[0][2]);
        __m256 m03 = _mm256_set1_ps(inv_transform.m[0][3]);
        
        __m256 m10 = _mm256_set1_ps(inv_transform.m[1][0]);
        __m256 m11 = _mm256_set1_ps(inv_transform.m[1][1]);
        __m256 m12 = _mm256_set1_ps(inv_transform.m[1][2]);
        __m256 m13 = _mm256_set1_ps(inv_transform.m[1][3]);
        
        __m256 m20 = _mm256_set1_ps(inv_transform.m[2][0]);
        __m256 m21 = _mm256_set1_ps(inv_transform.m[2][1]);
        __m256 m22 = _mm256_set1_ps(inv_transform.m[2][2]);
        __m256 m23 = _mm256_set1_ps(inv_transform.m[2][3]);
        
        local_ray.orig_x.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m00, r.orig_x.data), _mm256_mul_ps(m01, r.orig_y.data)), _mm256_add_ps(_mm256_mul_ps(m02, r.orig_z.data), m03));
        local_ray.orig_y.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m10, r.orig_x.data), _mm256_mul_ps(m11, r.orig_y.data)), _mm256_add_ps(_mm256_mul_ps(m12, r.orig_z.data), m13));
        local_ray.orig_z.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m20, r.orig_x.data), _mm256_mul_ps(m21, r.orig_y.data)), _mm256_add_ps(_mm256_mul_ps(m22, r.orig_z.data), m23));
        
        local_ray.dir_x.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m00, r.dir_x.data), _mm256_mul_ps(m01, r.dir_y.data)), _mm256_mul_ps(m02, r.dir_z.data));
        local_ray.dir_y.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m10, r.dir_x.data), _mm256_mul_ps(m11, r.dir_y.data)), _mm256_mul_ps(m12, r.dir_z.data));
        local_ray.dir_z.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m20, r.dir_x.data), _mm256_mul_ps(m21, r.dir_y.data)), _mm256_mul_ps(m22, r.dir_z.data));
        
        local_ray.update_derived_data();
        
        // 2. Hit in local space
        HitRecordPacket local_rec;
        mesh->hit_packet(local_ray, t_min, t_max, local_rec, ignore_volumes);
        
        // 3. Transform back to world space
        __m256 hit_mask = local_rec.mask;
        
        __m256 w00 = _mm256_set1_ps(transform.m[0][0]);
        __m256 w01 = _mm256_set1_ps(transform.m[0][1]);
        __m256 w02 = _mm256_set1_ps(transform.m[0][2]);
        __m256 w03 = _mm256_set1_ps(transform.m[0][3]);
        
        __m256 w10 = _mm256_set1_ps(transform.m[1][0]);
        __m256 w11 = _mm256_set1_ps(transform.m[1][1]);
        __m256 w12 = _mm256_set1_ps(transform.m[1][2]);
        __m256 w13 = _mm256_set1_ps(transform.m[1][3]);
        
        __m256 w20 = _mm256_set1_ps(transform.m[2][0]);
        __m256 w21 = _mm256_set1_ps(transform.m[2][1]);
        __m256 w22 = _mm256_set1_ps(transform.m[2][2]);
        __m256 w23 = _mm256_set1_ps(transform.m[2][3]);
        
        local_rec.p_x.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w00, local_rec.p_x.data), _mm256_mul_ps(w01, local_rec.p_y.data)), _mm256_add_ps(_mm256_mul_ps(w02, local_rec.p_z.data), w03));
        local_rec.p_y.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w10, local_rec.p_x.data), _mm256_mul_ps(w11, local_rec.p_y.data)), _mm256_add_ps(_mm256_mul_ps(w12, local_rec.p_z.data), w13));
        local_rec.p_z.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w20, local_rec.p_x.data), _mm256_mul_ps(w21, local_rec.p_y.data)), _mm256_add_ps(_mm256_mul_ps(w22, local_rec.p_z.data), w23));
        
        local_rec.normal_x.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w00, local_rec.normal_x.data), _mm256_mul_ps(w01, local_rec.normal_y.data)), _mm256_mul_ps(w02, local_rec.normal_z.data));
        local_rec.normal_y.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w10, local_rec.normal_x.data), _mm256_mul_ps(w11, local_rec.normal_y.data)), _mm256_mul_ps(w12, local_rec.normal_z.data));
        local_rec.normal_z.data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(w20, local_rec.normal_x.data), _mm256_mul_ps(w21, local_rec.normal_y.data)), _mm256_mul_ps(w22, local_rec.normal_z.data));
        
        Vec3SIMD::normalize_8x(local_rec.normal_x, local_rec.normal_y, local_rec.normal_z, local_rec.normal_x, local_rec.normal_y, local_rec.normal_z);
        
        rec.merge_if_closer(local_rec, hit_mask);
    }

	virtual bool bounding_box(float time0, float time1, AABB& output_box) const override {
		output_box = bbox;
		return true;
	}
};

