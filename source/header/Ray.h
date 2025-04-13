#ifndef RAY_H
#define RAY_H

#include "Vec3.h"

class Ray {
public:
    Vec3 origin;
    Vec3 direction;

    Ray() {}
    Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}

    Vec3 at(float t) const { return Vec3(origin) + t * Vec3(direction); }
};

#endif // RAY_H

