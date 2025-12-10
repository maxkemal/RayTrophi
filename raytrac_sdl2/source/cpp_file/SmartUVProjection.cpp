#include "SmartUVProjection.h"
#include <iostream>
#include <limits>
#include <cmath>

SmartUVProjection::SmartUVProjection(std::vector<std::shared_ptr<Triangle>>& triangles)
    : triangles(triangles) {}

void SmartUVProjection::calculateBoundingBox() {
    if (triangles.empty()) {
        std::cerr << "Error: No triangles to calculate bounding box" << std::endl;
        return;
    }

    minBound = Vec3(std::numeric_limits<float>::max());
    maxBound = Vec3(std::numeric_limits<float>::lowest());

    for (const auto& triangle : triangles) {
        if (!triangle) continue; // Null pointer kontrolü
        minBound = Vec3::min(minBound, triangle->v0);
        minBound = Vec3::min(minBound, triangle->v1);
        minBound = Vec3::min(minBound, triangle->v2);

        maxBound = Vec3::max(maxBound, triangle->v0);
        maxBound = Vec3::max(maxBound, triangle->v1);
        maxBound = Vec3::max(maxBound, triangle->v2);
    }

    std::cout << "Bounding Box Min: " << minBound.x << ", " << minBound.y << ", " << minBound.z << std::endl;
    std::cout << "Bounding Box Max: " << maxBound.x << ", " << maxBound.y << ", " << maxBound.z << std::endl;
}

void SmartUVProjection::apply(ProjectionType type) {
    std::cout << "Entering apply method with type: " << static_cast<int>(type) << std::endl;

    if (triangles.empty()) {
        std::cerr << "Warning: No Triangle objects found for UV projection." << std::endl;
        return;
    }

    calculateBoundingBox();
    switch (type) {
    case ProjectionType::PLANAR:
        calculatePlanarProjection();
        break;
    case ProjectionType::SPHERICAL:
        calculateSphericalProjection();
        break;
        // Diðer projeksiyon türleri burada eklenebilir
    default:
        std::cerr << "Error: Unsupported projection type" << std::endl;
        return;
    }
    normalizeUVCoordinates();
}

void SmartUVProjection::calculatePlanarProjection() {
    Vec3 avgNormal = calculateAverageNormal();
    Vec3 tangent, bitangent;
    createOrthonormalBasis(avgNormal, tangent, bitangent);

    Vec3 boxSize = maxBound - minBound;
    float maxDimension = std::max({ boxSize.x, boxSize.y, boxSize.z });

    for (auto& triangle : triangles) {
        if (!triangle) continue; // Null pointer kontrolü
        triangle->t0 = calculateUVForVertex(triangle->v0, minBound, tangent, bitangent, boxSize, maxDimension);
        triangle->t1 = calculateUVForVertex(triangle->v1, minBound, tangent, bitangent, boxSize, maxDimension);
        triangle->t2 = calculateUVForVertex(triangle->v2, minBound, tangent, bitangent, boxSize, maxDimension);
    }
}

void SmartUVProjection::calculateSphericalProjection() {
    for (auto& triangle : triangles) {
        if (!triangle) continue; // Null pointer kontrolü
        triangle->t0 = calculateSphericalUV(triangle->v0);
        triangle->t1 = calculateSphericalUV(triangle->v1);
        triangle->t2 = calculateSphericalUV(triangle->v2);
    }
}

Vec2 SmartUVProjection::calculateUVForVertex(const Vec3& vertex, const Vec3& minBound,
    const Vec3& tangent, const Vec3& bitangent,
    const Vec3& boxSize, double maxDimension) {
    Vec3 localPos = vertex - minBound;
    Vec2 uv;
    uv.x = Vec3::dot(localPos,tangent) / maxDimension;
    uv.y = Vec3::dot(localPos, bitangent) / maxDimension;

    // Normalize UV coordinates to [0, 1] range
    uv.x = std::clamp(uv.x, 0.0, 1.0);
    uv.y = std::clamp(uv.y, 0.0, 1.0);

    return uv;
}

Vec2 SmartUVProjection::calculateSphericalUV(const Vec3& vertex) {
    Vec3 normVertex = vertex.normalize();
    double u = 0.5 + (atan2(normVertex.z, normVertex.x) / (2.0 * M_PI));
    double v = 0.5 - (asin(normVertex.y) / M_PI);
    return Vec2(u, v);
}

Vec3 SmartUVProjection::calculateAverageNormal() {
    Vec3 avgNormal(0, 0, 0);
    int validTriangleCount = 0;

    for (const auto& triangle : triangles) {
        if (!triangle) continue;

        Vec3 edge1 = triangle->v1 - triangle->v0;
        Vec3 edge2 = triangle->v2 - triangle->v0;
        Vec3 normal = Vec3::cross(edge1, edge2).normalize();

        if (normal.length() > 1e-6) {
            avgNormal += normal;
            validTriangleCount++;
        }
    }

    if (validTriangleCount == 0) {
        std::cerr << "Error: No valid triangles found for normal calculation" << std::endl;
        return Vec3(0, 1, 0);  // Return a default up vector
    }

    avgNormal = avgNormal / validTriangleCount;

    if (avgNormal.length() < 1e-6) {
        std::cerr << "Warning: Average normal is near-zero, using default up vector" << std::endl;
        return Vec3(0, 1, 0);
    }

    return avgNormal.normalize();
}

void SmartUVProjection::createOrthonormalBasis(const Vec3& normal, Vec3& tangent, Vec3& bitangent) {
    if (std::abs(normal.x) > std::abs(normal.y)) {
        tangent = Vec3(normal.z, 0, -normal.x).normalize();
    }
    else {
        tangent = Vec3(0, -normal.z, normal.y).normalize();
    }
    bitangent = normal.cross(tangent).normalize();

    // Ensure no `nan` values
    if (std::isnan(tangent.x) || std::isnan(tangent.y) || std::isnan(tangent.z) ||
        std::isnan(bitangent.x) || std::isnan(bitangent.y) || std::isnan(bitangent.z)) {
        std::cerr << "Error: `nan` values detected in tangent or bitangent calculation" << std::endl;
        tangent = Vec3(1, 0, 0); // Default tangent
        bitangent = Vec3(0, 1, 0); // Default bitangent
    }
}

void SmartUVProjection::normalizeUVCoordinates() {
    Vec2 minUV(std::numeric_limits<float>::max());
    Vec2 maxUV(std::numeric_limits<float>::lowest());

    // Find min and max UV coordinates
    for (const auto& triangle : triangles) {
        if (!triangle) continue;
        minUV = Vec2::min(minUV, triangle->t0);
        minUV = Vec2::min(minUV, triangle->t1);
        minUV = Vec2::min(minUV, triangle->t2);
        maxUV = Vec2::max(maxUV, triangle->t0);
        maxUV = Vec2::max(maxUV, triangle->t1);
        maxUV = Vec2::max(maxUV, triangle->t2);
    }

    Vec2 uvRange = maxUV - minUV;

    // Normalize UV coordinates
    for (auto& triangle : triangles) {
        if (!triangle) continue;
        triangle->t0 = (triangle->t0 - minUV) / uvRange;
        triangle->t1 = (triangle->t1 - minUV) / uvRange;
        triangle->t2 = (triangle->t2 - minUV) / uvRange;
    }
}

Vec2 SmartUVProjection::normalizeUV(const Vec2& uv) {
    return Vec2(fmod(uv.x, 1.0f), fmod(uv.y, 1.0f));
}
