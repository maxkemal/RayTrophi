#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.h"
#include "AABB.h"
#include <vector>
#include <memory> // std::shared_ptr kullanýmý için
#include "Vec2.h"

class Material; // Ýleri bildirim
class Texture;
class Triangle;

struct HitRecord {
    Vec3 point;
    Vec3 normal;
    Vec3 neighbor_normal;
    bool has_neighbor_normal = false;
    Vec3 interpolated_normal;
    Vec3 face_normal;
  
    std::shared_ptr<Material>  material;
    float t;
    float u;
    float v;
    bool front_face;
  
    const Triangle* triangle = nullptr;  // Triangle referansý 
    // Yeni alanlar
    Vec3 tangent;    // Teðet vektör
    Vec3 bitangent;  // Binormal vektör
    bool has_tangent = false; // Teðet vektörün var olup olmadýðýný kontrol
    Vec2 uv; // UV koordinatlarý eklendi
  
    inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = Vec3::dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
   
    // Konstruktör
    HitRecord() : t(0), front_face(false), u(0), v(0) {}
};

class Hittable {
public:
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
 
    virtual bool bounding_box(float time0, float time1, AABB& output_box) const = 0;
    virtual ~Hittable() = default;
    virtual void collect_neighbor_normals(const AABB& query_box, Vec3& neighbor_normal,
        int& neighbor_count, const std::shared_ptr<Material>& current_material) const {
        // Varsayýlan implementasyon: hiçbir þey yapma
    }
    virtual bool occluded(const Ray& ray, float t_min, float t_max) const {
        HitRecord dummy;
        return hit(ray, t_min, t_max, dummy);
    }

    std::vector<std::shared_ptr<Hittable>> objects; // objects üyesi eklendi
private:
    std::vector<std::shared_ptr<Hittable>> hittables;
};

#endif // HITTABLE_H
