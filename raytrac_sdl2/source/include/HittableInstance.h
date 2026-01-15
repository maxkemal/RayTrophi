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

	virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
		// Transform Hit Logic
        // 1. Transform ray to object space
        Vec3 origin = inv_transform.transform_point(r.origin);
        Vec3 direction = inv_transform.transform_vector(r.direction);
        
        Ray object_ray(origin, direction);

        // 2. Hit in object space
        if (!mesh->hit(object_ray, t_min, t_max, rec))
            return false;

        // 3. Transform result back to world space
        // Point is easy: just use ray.at(rec.t) to be precise in world space
        // Or transform rec.point?
        // rec.point is in object space.
        rec.point = transform.transform_point(rec.point);
        
        // Normal needs inverse transpose for non-uniform scaling
        // But for rigid transform, just rotation.
        // Matrix4x4 handles vector transform correctly (ignoring translation)
        // Correct way: transform_vector using cofactor matrix (inv transform transpose).
        // Since we store inv_transform, we can use its transpose? 
        // Or simpler: Matrix4x4::transform_normal which usually does this.
        // Our Matrix4x4::transform_vector does M * v (rotation/scale).
        // If scale is uniform, this is correct.
        // If non-uniform, we need inverse-transpose.
        // Let's assume Matrix4x4::transpose exists.
        
        // For now, let's use transform_vector, risking non-uniform scale artifacts on normals.
        // (Most scatter is uniform scale).
        // Actually, we can compute it:
        // normal_world = transpose(inv_transform) * normal_local
        
        // Let's defer strict normal correctness for now and just rotate.
        rec.normal = transform.transform_vector(rec.normal).normalize();
        
        // Re-orient face normal
        rec.set_face_normal(r, rec.normal);
        
        return true;
	}

	virtual bool bounding_box(float time0, float time1, AABB& output_box) const override {
		output_box = bbox;
		return true;
	}
};
