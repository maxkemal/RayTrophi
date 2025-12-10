#ifndef PLANE_H
#define PLANE_H

#include "Vec3.h"

struct Plane {
    Vec3 normal;
    float distance;

    // Default constructor (initializes with zero normal and distance)
    Plane() : normal(Vec3()), distance(0) {}

    // Constructor with parameters
    Plane(const Vec3& n, float d) : normal(n), distance(d) {}

    // Function to compute the signed distance from the plane to a point
    float distanceToPoint(const Vec3& point) const {
        return normal.dot(point) + distance;
    }
};

#endif // PLANE_H
