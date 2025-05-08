// hit_result.h
#pragma once
#include <vector_types.h>

struct OptixHitResult {
    bool hasHit;
    float3 hitPoint;
    float3 normal;
    float t;
};
