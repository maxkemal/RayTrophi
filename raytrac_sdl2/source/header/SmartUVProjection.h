#ifndef SMART_UV_PROJECTION_H
#define SMART_UV_PROJECTION_H

#include <vector>
#include <memory>
#include "Triangle.h"
#include "Vec2.h"
#include "Vec3.h"
#include "PrincipledBSDF.h"

enum class ProjectionType {
    PLANAR,
    SPHERICAL
    // Diðer projeksiyon türleri buraya eklenebilir
};

class SmartUVProjection {
public:

    SmartUVProjection(std::vector<std::shared_ptr<Triangle>>& triangles);
    void apply();

    void calculateBoundingBox();

    void apply(ProjectionType type);

    void calculateUVProjection();

private:
    Vec3 minBound;  // Bounding box'ýn minimum noktasý
    Vec3 maxBound;  // Bounding box'ýn maksimum noktasý
    std::vector<std::shared_ptr<Triangle>>& triangles;
    std::vector<Vec2> calculatedUVs;
    std::vector<std::vector<std::shared_ptr<Triangle>>> groupTriangles();
    void projectGroup(const std::vector<std::shared_ptr<Triangle>>& group);
    void optimizeUVIslands();
    void normalizeUVCoordinates();
    Vec2 normalizeUV(const Vec2& uv);
    Vec2 calculateUV(const Vec3& vertex, const Vec3& tangent, const Vec3& bitangent);
    void applyUVProjection(Triangle* triangle, const Vec3& tangent, const Vec3& bitangent);
    Vec3 calculateAverageNormal(const std::vector<std::shared_ptr<Triangle>>& triangles);
    void calculatePlanarProjection();
    void calculateSphericalProjection();
    Vec2 calculateUVForVertex(const Vec3& vertex, const Vec3& minBound, const Vec3& tangent, const Vec3& bitangent, const Vec3& boxSize, double maxDimension);
    Vec2 calculateSphericalUV(const Vec3& vertex);
    void applyCalculatedUVs();
    Vec3 calculateAverageNormal();
    void createOrthonormalBasis(const Vec3& normal, Vec3& tangent, Vec3& bitangent);
    void calculateUVCoordinates(const Vec3& vertex, const Vec3& tangent, const Vec3& bitangent, Vec2& uv);
};

#endif // SMART_UV_PROJECTION_H