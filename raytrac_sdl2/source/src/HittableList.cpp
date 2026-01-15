#include "HittableList.h"

HittableList::HittableList() {}

HittableList::HittableList(std::shared_ptr<Hittable> object) {
    add(object);
}

void HittableList::clear() {
    objects.clear();
}

void HittableList::add(std::shared_ptr<Hittable> object) {
    objects.push_back(object);
}

void HittableList::reserve(size_t n) {
    objects.reserve(n);
}

size_t HittableList::size() const {
    return objects.size();
}

bool HittableList::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    if (bvh_root) {
        return bvh_root->hit(r, t_min, t_max, rec);
    }
    else {
        HitRecord temp_rec;
        bool hit_anything = false;
        auto closest_so_far = t_max;

        for (const auto& object : objects) {
            if (object && object->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
}

bool HittableList::bounding_box(float time0, float time1, AABB& output_box) const {
    if (objects.empty()) return false;

    AABB temp_box;
    bool first_box = true;

    for (const auto& object : objects) {
        if (!object) continue;
        if (!object->bounding_box(time0, time1, temp_box)) return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;
}

void HittableList::build_bvh() {
    // Implement BVH construction logic here if needed
    // Example: bvh_root = build_bvh(objects, 0.0, 1.0);
}
