#ifndef RAY_H
#define RAY_H

#ifdef __CUDACC__  // CUDA derleyicisiyle derleniyorsa
#include <cuda_runtime.h>
#include "vec3_utils.cuh"
#include "random_utils.cuh" // curand tabanlý random_float()

struct Ray {
    float3 origin;
    float3 direction;

    __device__ __host__ Ray() {}
    __device__ __host__ Ray(const float3& o, const float3& d) : origin(o), direction(d) {}

    __device__ __host__ float3 at(float t) const {
        return origin + t * direction;
    }
};

#else  // CPU derlemesi
#include "Vec3.h"

class Ray {
public:
    Vec3 origin;
    Vec3 direction;

    Ray() {}
    Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}

    Vec3 at(float t) const { return origin + t * direction; }
};

#endif  // __CUDACC__

#endif  // RAY_H
