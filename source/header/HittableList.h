#ifndef HITTABLELIST_H
#define HITTABLELIST_H

#include "Hittable.h"
#include "ParallelBVHNode.h"
#include <memory>
#include <vector>


class HittableList : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;
    std::shared_ptr<ParallelBVHNode> bvh_root;

    HittableList();
    HittableList(std::shared_ptr<Hittable> object);

    void clear();
    void add(std::shared_ptr<Hittable> object);
    void reserve(size_t n);
    size_t size() const;

    std::shared_ptr<Hittable> getHittable(size_t index) const {
        return hittables[index]; // hittables bir vektör veya benzeri bir veri yapýsý olmalý
    }
    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override;
    virtual bool bounding_box(double time0, double time1, AABB& output_box) const override;

    void build_bvh();
private:
    std::vector<std::shared_ptr<Hittable>> hittables; // Hittable objelerin listesi
  
};

#endif // HITTABLELIST_H
