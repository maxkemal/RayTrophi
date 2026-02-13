/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          HairSystem.cpp
 * Author:        Kemal Demirtaş
 * Description:   Main hair/fur management system implementation
 *                Handles strand generation, interpolation, and BVH building
 * =========================================================================
 */

#include "Hair/HairSystem.h"
#include "Triangle.h"
#include "Vec3.h"
#include "Vec2.h"

#include <embree4/rtcore.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <Quaternion.h>



namespace Hair {

// ============================================================================
// Constants
// ============================================================================

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;

// ============================================================================
// Random Number Generator (thread-safe)
// ============================================================================

class ThreadSafeRNG {
public:
    ThreadSafeRNG(uint32_t seed = 42) : gen(seed), dist(0.0f, 1.0f) {}
    
    float random() { return dist(gen); }
    float random(float min, float max) { return min + (max - min) * dist(gen); }
    
    void seed(uint32_t s) { gen.seed(s); }
    
private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

// ============================================================================
// HairSystem Implementation
// ============================================================================

HairSystem::HairSystem() {
    // Embree scene will be created on demand
}

HairSystem::~HairSystem() {
    if (m_embreeScene) {
        rtcReleaseScene(m_embreeScene);
        m_embreeScene = nullptr;
    }
}

std::vector<std::string> HairSystem::getGroomNames() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    std::vector<std::string> names;
    names.reserve(m_grooms.size());
    for (const auto& [name, groom] : m_grooms) {
        names.push_back(name);
    }
    return names;
}


// ============================================================================
// Strand Generation
// ============================================================================

void HairSystem::generateOnMesh(
    const std::vector<std::shared_ptr<Triangle>>& triangles,
    const HairGenerationParams& params,
    const std::string& groomName
) {
    if (triangles.empty()) return;
    
    // Create or get existing groom
    HairGroom& groom = m_grooms[groomName];
    groom.name = groomName;
    groom.params = params;
    groom.guides.clear();
    groom.interpolated.clear();
    groom.isDirty = true;
    m_statsDirty = true;
    // Material is already default initialized in HairGroom struct, 
    // but we could explicitly set it if needed (e.g. from a global default).

    
    // Store bound mesh name and initial transform
    groom.boundMeshName = triangles[0]->getNodeName();
    groom.boundMeshPtr = triangles[0]; // Cache for transform updates
    groom.boundTriangles = triangles;  // Store all triangles for skinning lookups
    // Hair is generated in world space, so start with identity transform
    // Store the initial mesh transform for delta calculation when mesh moves
    groom.transform.setIdentity();
    groom.initialMeshTransform = triangles[0]->getTransformMatrix();
    
    // Calculate total surface area for uniform distribution
    std::vector<float> triangleAreas;
    triangleAreas.reserve(triangles.size());
    float totalArea = 0.0f;
    
    for (const auto& tri : triangles) {
        Vec3 v0 = tri->getV0();
        Vec3 v1 = tri->getV1();
        Vec3 v2 = tri->getV2();
        
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        float area = 0.5f * Vec3::cross(edge1, edge2).length();
        
        triangleAreas.push_back(area);
        totalArea += area;
    }
    
    if (totalArea < 1e-6f) return;
    
    // Build CDF for importance sampling triangles by area
    std::vector<float> cdf;
    cdf.reserve(triangles.size());
    float cumulative = 0.0f;
    for (float area : triangleAreas) {
        cumulative += area / totalArea;
        cdf.push_back(cumulative);
    }
    
    // Generate guide strands
    ThreadSafeRNG rng(12345 + static_cast<uint32_t>(groomName.length()));
    groom.guides.reserve(params.guideCount);
    
    for (uint32_t i = 0; i < params.guideCount; ++i) {
        // Sample triangle by area
        float r = rng.random();
        auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
        size_t triIndex = std::distance(cdf.begin(), it);
        triIndex = std::min(triIndex, triangles.size() - 1);
        
        const auto& tri = triangles[triIndex];
        
        // Sample point on triangle (barycentric)
        float u = rng.random();
        float v = rng.random();
        if (u + v > 1.0f) {
            u = 1.0f - u;
            v = 1.0f - v;
        }
        
        Vec3 normal;
        Vec2 rootUV;
        Vec3 rootPos = sampleTriangleSurface(*tri, u, v, normal, rootUV);
        
        // Create strand
        HairStrand strand;
        strand.strandID = i;
        strand.materialID = params.defaultMaterialID;
        strand.meshMaterialID = tri->getMaterialID();
        strand.type = StrandType::GUIDE;
        strand.rootUV = rootUV;
        strand.randomSeed = rng.random();
        
        // Calculate strand length with variation
        float lengthVariation = 1.0f + (rng.random() - 0.5f) * 2.0f * params.lengthVariation;
        float strandLength = params.length * lengthVariation;
        
        strand.baseRootPos = rootPos;
        strand.baseLength = strandLength;
        strand.rootNormal = normal;
        
        strand.points.resize(params.pointsPerStrand);
        strand.groomedPositions.resize(params.pointsPerStrand);
        for (uint32_t p = 0; p < params.pointsPerStrand; ++p) {
            float t = static_cast<float>(p) / (params.pointsPerStrand - 1);
            strand.groomedPositions[p] = rootPos + normal * (t * strandLength);
        }
        
        // Skinning Data Initialization
        strand.triangleIndex = (uint32_t)triIndex;
        strand.barycentricUV = Vec2(u, v);
        // Store the initial shape relative to the world (or bind pose if that's what we are in)
        // Ideally, we'd store it relative to the triangle's local frame, but simply storing 
        // the "Rest Shape" points allows us to re-calculate the deformation delta.
        strand.restGroomedPositions = strand.groomedPositions;
        
        groom.guides.push_back(std::move(strand));
    }

    
    // Apply initial styling to the newly created guides
    restyleGroom(groomName);
    
    m_bvhDirty = true;
}


void HairSystem::generateFur(
    const std::vector<std::shared_ptr<Triangle>>& triangles,
    const HairGenerationParams& undercoatParams,
    const HairGenerationParams& guardParams,
    const std::string& groomName
) {
    // Generate undercoat (short, dense)
    HairGenerationParams undercoat = undercoatParams;
    undercoat.defaultMaterialID = undercoatParams.defaultMaterialID;
    
    std::string undercoatName = groomName + "_undercoat";
    generateOnMesh(triangles, undercoat, undercoatName);
    
    // Mark as undercoat type
    if (m_grooms.count(undercoatName)) {
        for (auto& strand : m_grooms[undercoatName].guides) {
            strand.type = StrandType::FUR_UNDERCOAT;
        }
    }
    
    // Generate guard hairs (long, sparse)
    HairGenerationParams guard = guardParams;
    guard.defaultMaterialID = guardParams.defaultMaterialID;
    
    std::string guardName = groomName + "_guard";
    generateOnMesh(triangles, guard, guardName);
    
    // Mark as guard type
    if (m_grooms.count(guardName)) {
        for (auto& strand : m_grooms[guardName].guides) {
            strand.type = StrandType::FUR_GUARD;
        }
    }
}

bool HairSystem::importAlembic(const std::string& filepath, const std::string& groomName) {
    // TODO: Implement Alembic import using Alembic library
    // For now, return false (not implemented)
    (void)filepath;
    (void)groomName;
    return false;
}

// ============================================================================
// Child Strand Interpolation
// ============================================================================

void HairSystem::interpolateChildren(HairGroom& groom) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (groom.guides.empty() || groom.params.interpolatedPerGuide == 0) return;
    
    ThreadSafeRNG rng(54321 + static_cast<uint32_t>(groom.name.length()));
    
    groom.interpolated.clear();
    groom.interpolated.reserve(groom.guides.size() * groom.params.interpolatedPerGuide);
    
    uint32_t childID = static_cast<uint32_t>(groom.guides.size());
    
    for (const auto& guide : groom.guides) {
        for (uint32_t c = 0; c < groom.params.interpolatedPerGuide; ++c) {
            HairStrand child;
            child.strandID = childID++;
            child.materialID = guide.materialID;
            child.meshMaterialID = guide.meshMaterialID;
            child.type = StrandType::INTERPOLATED;
            child.randomSeed = rng.random();
            child.rootRadius = guide.rootRadius * rng.random(0.7f, 1.0f);
            // Offset child root from guide root
            float offsetRadius = groom.params.childRadius; 
            float offsetAngle = rng.random() * TWO_PI;
            Vec3 offset(
                cosf(offsetAngle) * offsetRadius,
                0.0f,
                sinf(offsetAngle) * offsetRadius
            );
            
            child.rootUV = guide.rootUV;
            child.points.resize(guide.points.size());
            
            for (size_t p = 0; p < guide.points.size(); ++p) {
                const HairPoint& guidePoint = guide.points[p];
                HairPoint& childPoint = child.points[p];
                
                childPoint.v_coord = guidePoint.v_coord;
                float t = childPoint.v_coord;

                // better Clumpiness Model:
                float clump = groom.params.clumpiness * guide.clumpScale;

                
                // clumpShape determines the power of the effect along the strand
                float spreadFactor;
                if (clump >= 0.0f) {
                    // tips pull in: (1 - t^2) * (1 - clump) + (t^2) * 0
                    // actually simpler: 1.0 - pow(t, 2.0) * clump
                    spreadFactor = 1.0f - powf(t, 2.0f) * clump;
                } else {
                    // tips flare out: 1.0 + pow(t, 2.0) * abs(clump)
                    spreadFactor = 1.0f + powf(t, 2.0f) * fabsf(clump);
                }

                childPoint.position = guidePoint.position + offset * spreadFactor;
                
                // Add unique child-only variation (high frequency)
                if (groom.params.frizz > 0.0f) {
                    Vec3 j(
                        (rng.random() - 0.5f),
                        (rng.random() - 0.5f),
                        (rng.random() - 0.5f)
                    );
                    childPoint.position = childPoint.position + j * (groom.params.frizz * t * 0.01f);
                }
                
                childPoint.radius = guidePoint.radius * rng.random(0.7f, 1.0f);
            }

            
            groom.interpolated.push_back(std::move(child));
        }
    }
}

// ============================================================================
// BVH Building (Embree Curves)
// ============================================================================

void HairSystem::buildBVH() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Get Embree device (assuming it's available globally or passed in)
    // For now, we'll create a local device - in production, share with EmbreeBVH
    static RTCDevice embreeDevice = nullptr;
    if (!embreeDevice) {
        embreeDevice = rtcNewDevice(nullptr);
    }
    
    if (m_embreeScene) {
        rtcReleaseScene(m_embreeScene);
    }
    
    m_embreeScene = rtcNewScene(embreeDevice);
    rtcSetSceneFlags(m_embreeScene, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(m_embreeScene, RTC_BUILD_QUALITY_HIGH);
    
    // Clear buffers and geom mapping for fresh build
    m_smoothTangents.clear();
    m_segMap.clear();
    m_geomToGroom.clear();
    m_geomToTangentOffset.clear();


    
    // Add hair geometry for each groom
    for (auto& [name, groom] : m_grooms) {
        // ... (rest of the logic remains similar but with mapping data)
        std::vector<const HairStrand*> allStrands;
        for (const auto& s : groom.guides) allStrands.push_back(&s);
        for (const auto& s : groom.interpolated) allStrands.push_back(&s);
        
        if (allStrands.empty()) continue;
        
        const Matrix4x4& transform = groom.transform;
        
        // Calculate average scale for radius adjustment
        float sx = Vec3(transform.m[0][0], transform.m[1][0], transform.m[2][0]).length();
        float sy = Vec3(transform.m[0][1], transform.m[1][1], transform.m[2][1]).length();
        float sz = Vec3(transform.m[0][2], transform.m[1][2], transform.m[2][2]).length();
        float avgScale = (sx + sy + sz) / 3.0f;

        // Count total segments and vertices safely
        size_t totalSegmentsCount = 0;
        size_t totalVerticesCount = 0;
        for (const auto* strand : allStrands) {
            size_t n = strand->points.size();
            if (groom.params.useBSpline) {
                if (n >= 4) {
                    totalSegmentsCount += (n - 1); 
                    totalVerticesCount += (n + 2);
                }
            } else {
                if (n >= 2) {
                    totalSegmentsCount += (n - 1);
                    totalVerticesCount += n;
                }
            }
        }
        
        if (totalSegmentsCount == 0) continue;
        
        RTCGeometryType curveType = groom.params.useBSpline ? RTC_GEOMETRY_TYPE_ROUND_BSPLINE_CURVE : RTC_GEOMETRY_TYPE_ROUND_LINEAR_CURVE;
        RTCGeometry hairGeo = rtcNewGeometry(embreeDevice, curveType);
        
        if (groom.params.useBSpline) {
            rtcSetGeometryTessellationRate(hairGeo, (float)std::pow(2, groom.params.subdivisions));
        }
        
        float* vertices = (float*)rtcSetNewGeometryBuffer(hairGeo, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(float)*4, totalVerticesCount);
        unsigned int* indices = (unsigned int*)rtcSetNewGeometryBuffer(hairGeo, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT, sizeof(unsigned int), totalSegmentsCount);
        
        size_t vertexOffset = 0;
        size_t segmentOffset = 0;
        size_t tangentStartIndex = m_smoothTangents.size();
        size_t mapStartIndex = m_segMap.size();
        
        m_smoothTangents.resize(tangentStartIndex + totalSegmentsCount * 3); 
        m_segMap.resize(mapStartIndex + totalSegmentsCount);
        
        uint32_t groomNameHash = 0;
        for(char c : name) groomNameHash = groomNameHash * 31 + c;

        for (size_t sIdx = 0; sIdx < allStrands.size(); ++sIdx) {
            const auto* strand = allStrands[sIdx];
            size_t n = strand->points.size();
            
            if (groom.params.useBSpline) {
                if (n < 4) continue;
                
                size_t strandVertexStart = vertexOffset;
                const auto& pts = strand->points;
                
                Vec3 p0 = transform.transform_point(pts[0].position);
                Vec3 p1 = transform.transform_point(pts[1].position);
                Vec3 p_minus1 = p0 * 2.0f - p1;
                
                vertices[vertexOffset * 4 + 0] = p_minus1.x;
                vertices[vertexOffset * 4 + 1] = p_minus1.y;
                vertices[vertexOffset * 4 + 2] = p_minus1.z;
                vertices[vertexOffset * 4 + 3] = pts[0].radius * avgScale;
                vertexOffset++;
                
                for (const auto& point : pts) {
                    Vec3 worldPos = transform.transform_point(point.position);
                    vertices[vertexOffset * 4 + 0] = worldPos.x;
                    vertices[vertexOffset * 4 + 1] = worldPos.y;
                    vertices[vertexOffset * 4 + 2] = worldPos.z;
                    vertices[vertexOffset * 4 + 3] = point.radius * avgScale;
                    vertexOffset++;
                }
                
                Vec3 pn = transform.transform_point(pts[n-1].position);
                Vec3 pn_minus1 = transform.transform_point(pts[n-2].position);
                Vec3 p_plus1 = pn * 2.0f - pn_minus1;
                
                vertices[vertexOffset * 4 + 0] = p_plus1.x;
                vertices[vertexOffset * 4 + 1] = p_plus1.y;
                vertices[vertexOffset * 4 + 2] = p_plus1.z;
                vertices[vertexOffset * 4 + 3] = pts[n-1].radius * avgScale;
                vertexOffset++;
                
                for (size_t i = 0; i < n - 1; ++i) {
                    indices[segmentOffset] = static_cast<unsigned int>(strandVertexStart + i);
                    
                    Vec3 cp0(vertices[(strandVertexStart + i) * 4 + 0], vertices[(strandVertexStart + i) * 4 + 1], vertices[(strandVertexStart + i) * 4 + 2]);
                    Vec3 cp1(vertices[(strandVertexStart + i + 1) * 4 + 0], vertices[(strandVertexStart + i + 1) * 4 + 1], vertices[(strandVertexStart + i + 1) * 4 + 2]);
                    Vec3 cp2(vertices[(strandVertexStart + i + 2) * 4 + 0], vertices[(strandVertexStart + i + 2) * 4 + 1], vertices[(strandVertexStart + i + 2) * 4 + 2]);
                    Vec3 cp3(vertices[(strandVertexStart + i + 3) * 4 + 0], vertices[(strandVertexStart + i + 3) * 4 + 1], vertices[(strandVertexStart + i + 3) * 4 + 2]);
                    
                    // Exact B-Spline tangents for continuity
                    Vec3 T0 = (cp2 - cp0);
                    Vec3 T1 = (cp2 - cp1);
                    Vec3 T2 = (cp3 - cp1);
                    
                    // Normalize all to ensure quadratic interpolation weighting is purely directional
                    float l0 = T0.length(); if (l0 > 1e-6f) T0 = T0 / l0; else T0 = (cp2 - cp1).normalize();
                    float l1 = T1.length(); if (l1 > 1e-6f) T1 = T1 / l1; else T1 = (cp2 - cp1).normalize();
                    float l2 = T2.length(); if (l2 > 1e-6f) T2 = T2 / l2; else T2 = (cp2 - cp1).normalize();

                    m_smoothTangents[tangentStartIndex + segmentOffset * 3 + 0] = T0;
                    m_smoothTangents[tangentStartIndex + segmentOffset * 3 + 1] = T1;
                    m_smoothTangents[tangentStartIndex + segmentOffset * 3 + 2] = T2;
                    
                    SegmentMap& map = m_segMap[mapStartIndex + segmentOffset];
                    map.globalStrandID = groomNameHash + (uint32_t)sIdx;
                    map.localStrandIdx = (uint32_t)sIdx;
                    map.vStart = static_cast<float>(i) / static_cast<float>(n - 1);
                    map.vStep = 1.0f / static_cast<float>(n - 1);
                    
                    segmentOffset++;
                }
                
            } else {
                if (n < 2) continue;
                
                size_t strandVertexStart = vertexOffset;
                const auto& pts = strand->points;
                
                for (const auto& point : pts) {
                    Vec3 worldPos = transform.transform_point(point.position);
                    vertices[vertexOffset * 4 + 0] = worldPos.x;
                    vertices[vertexOffset * 4 + 1] = worldPos.y;
                    vertices[vertexOffset * 4 + 2] = worldPos.z;
                    vertices[vertexOffset * 4 + 3] = point.radius * avgScale;
                    vertexOffset++;
                }
                
                // Linear indexing
                for (size_t i = 0; i < n - 1; ++i) {
                    indices[segmentOffset] = static_cast<unsigned int>(strandVertexStart + i);
                    
                    Vec3 v0 = transform.transform_point(pts[std::max((int)i - 1, 0)].position);
                    Vec3 v1 = transform.transform_point(pts[i].position);
                    Vec3 v2 = transform.transform_point(pts[i + 1].position);
                    size_t idx_v3 = static_cast<size_t>(std::min<int>(static_cast<int>(i) + 2, static_cast<int>(n) - 1));
                    Vec3 v3 = transform.transform_point(pts[idx_v3].position);
                    
                    m_smoothTangents[tangentStartIndex + segmentOffset * 3 + 0] = (v2 - v0).normalize();
                    m_smoothTangents[tangentStartIndex + segmentOffset * 3 + 1] = (v2 - v1).normalize();
                    m_smoothTangents[tangentStartIndex + segmentOffset * 3 + 2] = (v3 - v1).normalize();
                    
                    SegmentMap& map = m_segMap[mapStartIndex + segmentOffset];
                    map.globalStrandID = groomNameHash + (uint32_t)sIdx;
                    map.localStrandIdx = (uint32_t)sIdx;
                    map.vStart = static_cast<float>(i) / static_cast<float>(n - 1);
                    map.vStep = 1.0f / static_cast<float>(n - 1);
                    
                    segmentOffset++;
                }
            }
        }
        
        rtcCommitGeometry(hairGeo);
        unsigned int geomID = rtcAttachGeometry(m_embreeScene, hairGeo);
        m_geomToGroom[geomID] = name;
        m_geomToTangentOffset[geomID] = tangentStartIndex;
        rtcReleaseGeometry(hairGeo);


    }
    
    rtcCommitScene(m_embreeScene);
    m_bvhDirty = false;
}

// ============================================================================
// Ray Intersection
// ============================================================================

bool HairSystem::intersect(
    const Vec3& rayOrigin,
    const Vec3& rayDir,
    float tMin,
    float tMax,
    HairHitInfo& hitInfo
) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_embreeScene || m_grooms.empty()) return false;
    
    RTCRayHit rayhit;
    rayhit.ray.org_x = rayOrigin.x;
    rayhit.ray.org_y = rayOrigin.y;
    rayhit.ray.org_z = rayOrigin.z;
    rayhit.ray.dir_x = rayDir.x;
    rayhit.ray.dir_y = rayDir.y;
    rayhit.ray.dir_z = rayDir.z;
    rayhit.ray.tnear = tMin;
    rayhit.ray.tfar = tMax;
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    
    rtcIntersect1(m_embreeScene, &rayhit);
    
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return false;
    }
    
    hitInfo.t = rayhit.ray.tfar;
    hitInfo.position = rayOrigin + rayDir * hitInfo.t;
    
    // Precise Quadratic Tangent Interpolation
    Vec3 tangent(0, 1, 0);
    uint32_t primID = rayhit.hit.primID;
    unsigned int geomID = rayhit.hit.geomID;
    float uParam = rayhit.hit.u;
    
    auto tIt = m_geomToTangentOffset.find(geomID);
    if (tIt != m_geomToTangentOffset.end()) {
        size_t tangentIdxStart = tIt->second; 
        size_t mapIdxStart = tangentIdxStart / 3;
        
        if (mapIdxStart + primID < m_segMap.size()) {
            const SegmentMap& map = m_segMap[mapIdxStart + primID];
            
            // Tangent evaluation
            size_t tIdx = tangentIdxStart + primID * 3;
            const Vec3& T0 = m_smoothTangents[tIdx + 0];
            const Vec3& T1 = m_smoothTangents[tIdx + 1];
            const Vec3& T2 = m_smoothTangents[tIdx + 2];
            
            // Quadratic Bezier derivative: B'(u) = (1-u)^2 * T0 + 2u(1-u) * T1 + u^2 * T2
            float u = uParam;
            tangent = (T0 * (1.0f - u) * (1.0f - u) + T1 * 2.0f * u * (1.0f - u) + T2 * u * u).normalize();
            
            // V-Coordinate evaluation
            hitInfo.v = map.vStart + u * map.vStep;
            hitInfo.strandID = map.globalStrandID;

            // Fetch the rootUV for texture mapping using local index
            auto itGroom = m_geomToGroom.find(rayhit.hit.geomID);
            if (itGroom != m_geomToGroom.end()) {
                const HairGroom& groom = m_grooms.at(itGroom->second);
                uint32_t localIdx = map.localStrandIdx;
                if (localIdx < groom.guides.size()) {
                    hitInfo.rootUV = groom.guides[localIdx].rootUV;
                    hitInfo.meshMaterialID = groom.guides[localIdx].meshMaterialID;
                } else {
                    size_t interpIdx = localIdx - groom.guides.size();
                    if (interpIdx < groom.interpolated.size()) {
                        hitInfo.rootUV = groom.interpolated[interpIdx].rootUV;
                        hitInfo.meshMaterialID = groom.interpolated[interpIdx].meshMaterialID;
                    }
                }
            }
        }
    }

    // Robust Geometric Normal (Ng from Embree represents the radial vector from axis to hit)
    Vec3 viewDir = -rayDir;
    Vec3 Ng(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
    
    // EMBREE STABILIZATION: Force Ng to always point towards the viewer.
    // This is critical for avoiding shadow-ray artifacts and 'black-flipping' segments.
    if (Vec3::dot(Ng, viewDir) < 0) Ng = Ng * -1.0f;
    Ng = Ng.normalize();
    
    // Robust Azimuthal Offset Calculation (h)
    // We use the camera-right vector in the plane perpendicular to the tangent.
    Vec3 camRight = Vec3::cross(tangent, viewDir);
    float crLen = camRight.length();
    
    if (crLen > 1e-6f) {
        camRight = camRight / crLen;
        // h is the signed distance of the ray from the cylinder axis in the plane of the cross section.
        // It's equivalent to the projection of the radial normal (Ng) onto the 'side' vector.
        float h = Vec3::dot(Ng, camRight);
        hitInfo.u = std::clamp(h, -1.0f, 1.0f);
    } else {
        // Degenerate case: Looking exactly down the strand.
        // h is undefined, so we default to 0 (looking at the center line).
        hitInfo.u = 0.0f; 
    }
    
    hitInfo.tangent = tangent;
    hitInfo.normal = Ng; // Use actual radial normal for shading/offset calculations
    
    auto it = m_geomToGroom.find(rayhit.hit.geomID);
    if (it != m_geomToGroom.end()) {
        const HairGroom& groom = m_grooms.at(it->second);
        hitInfo.material = groom.material;
        hitInfo.groomName = it->second;
    }


    
    return true;
}

bool HairSystem::intersectVolumetric(
    const Vec3& rayOrigin,
    const Vec3& rayDir,
    float tMin,
    float tMax,
    float searchRadius,
    HairHitInfo& hitInfo
) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_embreeScene) return false;

    float bestMetric = 1e30f;
    float searchRadiusSq = searchRadius * searchRadius;
    bool found = false;

    Vec3 D = rayDir.normalize();

    // Helper: Closest point on segment AB to Ray (O, D)
    auto ClosestPointSegmentRay = [&](const Vec3& A, const Vec3& B, const Vec3& O, const Vec3& D, 
                                    float& outT, float& outV_segment) {
        Vec3 u = B - A;
        Vec3 v = D;
        Vec3 w = A - O;
        float a = Vec3::dot(u, u);
        float b = Vec3::dot(u, v);
        float c = Vec3::dot(v, v);
        float d = Vec3::dot(u, w);
        float e = Vec3::dot(v, w);
        float D_denom = a * c - b * b;
        float sc, tc;

        if (std::abs(D_denom) < 1e-12f) {
            sc = 0.0f;
            if (std::abs(b) > std::abs(c) && std::abs(b) > 1e-12f) tc = d / b;
            else if (std::abs(c) > 1e-12f) tc = e / c;
            else tc = 0.0f;
        } else {
            sc = (b * e - c * d) / D_denom;
            tc = (a * e - b * d) / D_denom;
        }

        sc = std::clamp(sc, 0.0f, 1.0f);
        outV_segment = sc;
        outT = tc;
        return A + u * sc;
    };

    for (const auto& [name, groom] : m_grooms) {
        Matrix4x4 l2w = groom.transform;
        
        for (const auto& strand : groom.guides) {
            if (strand.groomedPositions.size() < 2) continue;

            // Optional: Quick bounding box/root check for performance
            Vec3 rootWorld = l2w.transform_point(strand.baseRootPos);
            if ((rootWorld - rayOrigin).length() > (tMax + strand.baseLength)) continue;

            for (size_t i = 0; i < strand.groomedPositions.size() - 1; ++i) {
                Vec3 pA = l2w.transform_point(strand.groomedPositions[i]);
                Vec3 pB = l2w.transform_point(strand.groomedPositions[i+1]);
                
                float t, v_seg;
                Vec3 pOnSeg = ClosestPointSegmentRay(pA, pB, rayOrigin, D, t, v_seg);
                
                if (t < tMin || t > tMax) continue;

                Vec3 pOnRay = rayOrigin + D * t;
                float distSq = (pOnSeg - pOnRay).length_squared();

                if (distSq < searchRadiusSq) {
                    // Metric: Prioritize Centerness (distSq) but respect Depth (t)
                    // We use a small depth bias to prevent jittering between similar layers
                    float metric = distSq + (t * 0.00001f); 
                    
                    if (metric < bestMetric) {
                        bestMetric = metric;
                        
                        hitInfo.t = t;
                        hitInfo.position = pOnSeg;
                        hitInfo.groomName = name;
                        hitInfo.material = groom.material;
                        hitInfo.strandID = strand.strandID;
                        
                        float baseV = (float)i / (strand.groomedPositions.size() - 1);
                        float stepV = 1.0f / (strand.groomedPositions.size() - 1);
                        hitInfo.v = baseV + v_seg * stepV;
                        
                        hitInfo.tangent = (pB - pA).normalize();
                        Vec3 normalVec = pOnRay - pOnSeg;
                        float nLen = normalVec.length();
                        if (nLen > 1e-8f) hitInfo.normal = normalVec / nLen;
                        else hitInfo.normal = Vec3(0, 1, 0);

                        found = true;
                    }
                }
            }
        }
    }

    return found;
}

// ============================================================================
// Fast Shadow Occlusion Test
// ============================================================================

bool HairSystem::occluded(
    const Vec3& rayOrigin,
    const Vec3& rayDir,
    float tMin,
    float tMax
) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_embreeScene) return false;
    
    RTCRay ray;
    ray.org_x = rayOrigin.x;
    ray.org_y = rayOrigin.y;
    ray.org_z = rayOrigin.z;
    ray.dir_x = rayDir.x;
    ray.dir_y = rayDir.y;
    ray.dir_z = rayDir.z;
    ray.tnear = tMin;
    ray.tfar = tMax;
    ray.mask = -1;
    ray.flags = 0;
    
    // rtcOccluded1 is faster than rtcIntersect1 for shadow rays
    // It only returns whether there's ANY hit, not the details
    rtcOccluded1(m_embreeScene, &ray);
    
    // If tfar becomes negative, ray was occluded
    return (ray.tfar < 0.0f);
}

// ============================================================================
// GPU Data Preparation
// ============================================================================


HairGPUData HairSystem::prepareGPUData() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    HairGPUData gpuData;
    
    std::vector<HairStrand> allStrands;
    for (const auto& [name, groom] : m_grooms) {
        allStrands.insert(allStrands.end(), groom.guides.begin(), groom.guides.end());
        allStrands.insert(allStrands.end(), groom.interpolated.begin(), groom.interpolated.end());
    }
    
    gpuData.buildFromStrands(allStrands);
    return gpuData;
}

// ============================================================================
// OptiX GPU Data Export
// ============================================================================

bool HairSystem::getOptiXCurveData(
    std::vector<float>& outVertices4,
    std::vector<unsigned int>& outIndices,
    std::vector<uint32_t>& outStrandIDs,
    std::vector<float>& outTangents3,
    std::vector<float>& outRootUVs2,
    size_t& outVertexCount,
    size_t& outSegmentCount,
    bool includeInterpolated
) const {
    outVertices4.clear();
    outIndices.clear();
    outStrandIDs.clear();
    outTangents3.clear();
    outRootUVs2.clear();
    outVertexCount = 0;
    outSegmentCount = 0;

    bool globalUseBSpline = false;
    
    for (const auto& [name, groom] : m_grooms) {
        std::vector<float> v4;
        std::vector<unsigned int> indices;
        std::vector<uint32_t> strands;
        std::vector<float> tangents;
        std::vector<float> uvs;
        size_t vc = 0, sc = 0;
        HairMaterialParams dummyParams;
        int dummyID = 0;
        int dummyMeshID = -1;
        
        bool isBSpline = getOptiXCurveDataByGroom(name, v4, indices, strands, tangents, uvs, vc, sc, dummyParams, dummyID, dummyMeshID, includeInterpolated);
        if (isBSpline) globalUseBSpline = true;
        
        // Offset indices for global buffer
        unsigned int indexOffset = (unsigned int)outVertexCount;
        for (auto& idx : indices) idx += indexOffset;
        
        outVertices4.insert(outVertices4.end(), v4.begin(), v4.end());
        outIndices.insert(outIndices.end(), indices.begin(), indices.end());
        outStrandIDs.insert(outStrandIDs.end(), strands.begin(), strands.end());
        outTangents3.insert(outTangents3.end(), tangents.begin(), tangents.end());
        outRootUVs2.insert(outRootUVs2.end(), uvs.begin(), uvs.end());
        
        outVertexCount += vc;
        outSegmentCount += sc;
    }
    
    return globalUseBSpline;
}
bool HairSystem::getOptiXCurveDataByGroom(
    const std::string& groomName,
    std::vector<float>& outVertices4,
    std::vector<unsigned int>& outIndices,
    std::vector<uint32_t>& outStrandIDs,
    std::vector<float>& outTangents3,
    std::vector<float>& outRootUVs2,
    size_t& outVertexCount,
    size_t& outSegmentCount,
    HairMaterialParams& outMatParams,
    int& outMatID,
    int& outMeshMatID,
    bool includeInterpolated
) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(groomName);
    if (it == m_grooms.end()) return false;
    
    const auto& groom = it->second;
    outMatParams = groom.material;
    outMatID = groom.params.defaultMaterialID;
    
    // Get representative mesh material ID from the first strand
    outMeshMatID = -1;
    if (!groom.guides.empty()) {
        outMeshMatID = groom.guides[0].meshMaterialID;
    } else if (!groom.interpolated.empty()) {
        outMeshMatID = groom.interpolated[0].meshMaterialID;
    }
    
    const Matrix4x4& transform = it->second.transform;
    
    // Calculate average scale for radius adjustment
    float sx = Vec3(transform.m[0][0], transform.m[1][0], transform.m[2][0]).length();
    float sy = Vec3(transform.m[0][1], transform.m[1][1], transform.m[2][1]).length();
    float sz = Vec3(transform.m[0][2], transform.m[1][2], transform.m[2][2]).length();
    float avgScale = (sx + sy + sz) / 3.0f;

    bool useBSpline = groom.params.useBSpline;
    
    outVertices4.clear();
    outIndices.clear();
    outStrandIDs.clear();
    outTangents3.clear();
    outRootUVs2.clear();
    outVertexCount = 0;
    outSegmentCount = 0;

    std::vector<const HairStrand*> allStrands;
    for (const auto& s : groom.guides) allStrands.push_back(&s);
    if (includeInterpolated) {
        for (const auto& s : groom.interpolated) allStrands.push_back(&s);
    }
    
    for (const auto* strand : allStrands) {
        size_t nPoints = strand->points.size();
        if (nPoints < 2) continue;
        
        size_t strandStartIdx = outVertexCount;
        
        if (useBSpline) {
            // Triple endpoints for B-Splines
            Vec3 p0 = transform.transform_point(strand->points[0].position);
            float r0 = std::max(strand->points[0].radius * avgScale, 1e-5f);
            for (int j = 0; j < 2; ++j) {
                outVertices4.push_back(p0.x); outVertices4.push_back(p0.y); outVertices4.push_back(p0.z); outVertices4.push_back(r0);
                outVertexCount++;
            }

            for (const auto& pt : strand->points) {
                Vec3 p = transform.transform_point(pt.position);
                float r = std::max(pt.radius * avgScale, 1e-5f);
                outVertices4.push_back(p.x); outVertices4.push_back(p.y); outVertices4.push_back(p.z); outVertices4.push_back(r);
                outVertexCount++;
            }

            Vec3 pn = transform.transform_point(strand->points.back().position);
            float rn = std::max(strand->points.back().radius * avgScale, 1e-5f);
            for (int j = 0; j < 2; ++j) {
                outVertices4.push_back(pn.x); outVertices4.push_back(pn.y); outVertices4.push_back(pn.z); outVertices4.push_back(rn);
                outVertexCount++;
            }

            for (size_t i = 0; i < nPoints + 1; ++i) {
                outIndices.push_back(static_cast<unsigned int>(strandStartIdx + i));
                outSegmentCount++;
                
                size_t idx0 = (i < 1) ? 0 : (i - 1);
                size_t idx1 = (i >= nPoints) ? (nPoints - 1) : i;
                Vec3 v0 = transform.transform_point(strand->points[idx0].position);
                Vec3 v1 = transform.transform_point(strand->points[idx1].position);
                Vec3 tangent = (v1 - v0);
                if (tangent.length_squared() < 1e-8f && nPoints > 1) {
                     tangent = transform.transform_point(strand->points[1].position) - transform.transform_point(strand->points[0].position);
                }
                tangent = (tangent).normalize();
                outTangents3.push_back(tangent.x); outTangents3.push_back(tangent.y); outTangents3.push_back(tangent.z);
                outStrandIDs.push_back(strand->strandID);
                outRootUVs2.push_back(strand->rootUV.u); outRootUVs2.push_back(strand->rootUV.v);
            }
        } else {
            for (const auto& pt : strand->points) {
                Vec3 p = transform.transform_point(pt.position);
                float r = std::max(pt.radius * avgScale, 1e-5f);
                outVertices4.push_back(p.x); outVertices4.push_back(p.y); outVertices4.push_back(p.z); outVertices4.push_back(r);
                outVertexCount++;
            }

            for (size_t i = 0; i < nPoints - 1; ++i) {
                outIndices.push_back(static_cast<unsigned int>(strandStartIdx + i));
                outSegmentCount++;
                
                Vec3 p0 = transform.transform_point(strand->points[i].position);
                Vec3 p1 = transform.transform_point(strand->points[i+1].position);
                Vec3 delta = p1 - p0;
                Vec3 tangent = delta.length_squared() > 1e-9f ? delta.normalize() : Vec3(0, 1, 0);
                outTangents3.push_back(tangent.x); outTangents3.push_back(tangent.y); outTangents3.push_back(tangent.z);
                outStrandIDs.push_back(strand->strandID);
                outRootUVs2.push_back(strand->rootUV.u); outRootUVs2.push_back(strand->rootUV.v);
            }
        }
    }
    
    return useBSpline;
}

// ============================================================================
// Accessors
// ============================================================================

void HairSystem::refreshStats() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_totalStrandCount = 0;
    m_totalPointCount = 0;
    for (const auto& [name, groom] : m_grooms) {
        size_t strandsInGroom = groom.guides.size() + groom.interpolated.size();
        m_totalStrandCount += strandsInGroom;
        m_totalPointCount += strandsInGroom * groom.params.pointsPerStrand;
    }
    m_statsDirty = false;
}

size_t HairSystem::getTotalStrandCount() const {
    if (m_statsDirty) refreshStats();
    return m_totalStrandCount;
}

size_t HairSystem::getTotalPointCount() const {
    if (m_statsDirty) refreshStats();
    return m_totalPointCount;
}

HairGroom* HairSystem::getGroom(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(name);
    return (it != m_grooms.end()) ? &it->second : nullptr;
}

const HairGroom* HairSystem::getGroom(const std::string& name) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(name);
    return (it != m_grooms.end()) ? &it->second : nullptr;
}

void HairSystem::clearAll() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_grooms.clear();
    m_geomToGroom.clear();
    m_geomToTangentOffset.clear();
    m_smoothTangents.clear();
    m_statsDirty = true;
    
    // Release Embree scene to free memory
    if (m_embreeScene) {
        rtcReleaseScene(m_embreeScene);
        m_embreeScene = nullptr;
    }
    m_bvhDirty = true;
}

void HairSystem::removeGroom(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(name);
    if (it != m_grooms.end()) {
        m_grooms.erase(it);
        m_statsDirty = true;
        
        // Also clear geometry mappings that might reference this groom
        // This is a brutal clear, but safe since buildBVH will rebuild everything
        if (m_grooms.empty()) {
             m_geomToGroom.clear();
             m_geomToTangentOffset.clear();
             m_smoothTangents.clear();
        }
        
        m_bvhDirty = true;
    }
}


// ===================================
// styling logic moved below to restyleGroom
// ===================================


// ============================================================================
// Internal Helpers
// ============================================================================

Vec3 HairSystem::sampleTriangleSurface(
    const Triangle& tri,
    float u, float v,
    Vec3& outNormal,
    Vec2& outUV
) const {
    Vec3 v0 = tri.getV0();
    Vec3 v1 = tri.getV1();
    Vec3 v2 = tri.getV2();
    
    Vec3 n0 = tri.getN0();
    Vec3 n1 = tri.getN1();
    Vec3 n2 = tri.getN2();
    
    // Barycentric interpolation
    float w = 1.0f - u - v;
    
    Vec3 position = v0 * w + v1 * u + v2 * v;
    outNormal = (n0 * w + n1 * u + n2 * v).normalize();
    
    // UV interpolation (using public members t0, t1, t2)
    Vec2 uv0 = tri.t0;
    Vec2 uv1 = tri.t1;
    Vec2 uv2 = tri.t2;
    outUV.x = uv0.x * w + uv1.x * u + uv2.x * v;
    outUV.y = uv0.y * w + uv1.y * u + uv2.y * v;
    
    return position;
}

void HairSystem::applyGravityToStrand(HairStrand& strand, float gravity) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (strand.points.empty()) return;
    
    // Simple gravity: bend downward based on distance from root
    float strandLength = strand.length();
    
    for (size_t i = 1; i < strand.points.size(); ++i) {
        float t = strand.points[i].v_coord;
        float gravityOffset = t * t * gravity * strandLength;
        strand.points[i].position.y -= gravityOffset;
    }
}

void HairSystem::applyCurlToStrand(HairStrand& strand, float frequency, float radius) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (strand.points.empty() || frequency <= 0.0f) return;
    if (radius <= 0.0f) return;
    
    // Get tangent direction
    Vec3 tangent = (strand.points.back().position - strand.points.front().position).normalize();
    Vec3 bitangent = Vec3::cross(tangent, Vec3(0, 1, 0)).normalize();
    if (bitangent.length() < 0.1f) {
        bitangent = Vec3::cross(tangent, Vec3(1, 0, 0)).normalize();
    }
    Vec3 normal = Vec3::cross(bitangent, tangent);
    
    for (size_t i = 1; i < strand.points.size(); ++i) {
        float t = strand.points[i].v_coord;
        float angle = t * frequency * TWO_PI + strand.randomSeed * TWO_PI;
        
        Vec3 offset = bitangent * (cosf(angle) * radius * t) +
                      normal * (sinf(angle) * radius * t);
        
        strand.points[i].position = strand.points[i].position + offset;
    }
}

// ============================================================================
// Transform & Binding
// ============================================================================

void HairSystem::updateGroomTransform(const std::string& groomName, const Matrix4x4& currentMeshTransform) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(groomName);
    if (it == m_grooms.end()) return;
    
    HairGroom& groom = it->second;
    
    // [FIX] Skinned grooms handle their own transform (Identity + Vertex Deformation)
    // Applying rigid transform on top would cause Double Transformation.
    if (!groom.boundTriangles.empty() && groom.boundTriangles[0]->hasSkinData()) return;
    
    // Calculate delta transform: currentTransform * inverse(initialTransform)
    // This gives us the relative movement from when hair was generated
    Matrix4x4 initialInverse = groom.initialMeshTransform.inverse();
    Matrix4x4 deltaTransform = currentMeshTransform * initialInverse;
    
    // Check if transform actually changed
    bool transformChanged = false;
    for (int i = 0; i < 4 && !transformChanged; ++i) {
        for (int j = 0; j < 4 && !transformChanged; ++j) {
            if (std::abs(groom.transform.m[i][j] - deltaTransform.m[i][j]) > 1e-6f) {
                transformChanged = true;
            }
        }
    }
    
    if (transformChanged) {
        groom.transform = deltaTransform;
        groom.isDirty = true;
        m_bvhDirty = true;
    }
}

Vec3 HairSystem::getTransformedPosition(const HairStrand& strand, size_t pointIndex, const Matrix4x4& transform) const {
    if (pointIndex >= strand.points.size()) return Vec3(0, 0, 0);
    
    const Vec3& localPos = strand.points[pointIndex].position;
    return transform.transform_point(localPos);
}

void HairSystem::markDirty(const std::string& groomName) {
    auto it = m_grooms.find(groomName);
    if (it != m_grooms.end()) {
        it->second.isDirty = true;
        m_bvhDirty = true;
    }
}

void HairSystem::updateAllTransforms(const std::vector<std::shared_ptr<Hittable>>& sceneObjects) {
    if (m_grooms.empty()) return;

    // 1. First pass: Update from cached pointers where possible
    std::unordered_set<std::string> meshesToFind;
    for (auto& [name, groom] : m_grooms) {
        if (groom.boundMeshName.empty()) continue;

        // Check if we have skinning data
        if (!groom.boundTriangles.empty() && groom.boundTriangles[0]->hasSkinData()) {
            updateSkinnedGroom(name);
            continue; // Skip rigid transform update
        }

        if (auto mesh = groom.boundMeshPtr.lock()) {
            // Check if this mesh is actually the one we expect (same name)
            // This handles cases where mesh objects might be replaced but pointers linger
            if (mesh->getNodeName() == groom.boundMeshName) {
                updateFromMeshTransform(groom.boundMeshName, mesh->getTransformMatrix());
                continue; 
            }
        }
        
        // If we reach here, we need to find the mesh in sceneObjects
        meshesToFind.insert(groom.boundMeshName);
    }

    if (meshesToFind.empty()) return;

    // 2. Second pass: Scan scene only for grooms that aren't cached yet
    for (const auto& obj : sceneObjects) {
        if (meshesToFind.empty()) break; 

        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri) {
            const std::string& name = tri->getNodeName();
            auto it = meshesToFind.find(name);
            if (it != meshesToFind.end()) {
                // Found it! Cache it for next frame
                for (auto& [gName, g] : m_grooms) {
                    if (g.boundMeshName == name) {
                        g.boundMeshPtr = tri;
                    }
                }
                updateFromMeshTransform(name, tri->getTransformMatrix());
                meshesToFind.erase(it);
            }
        }
    }
}

void HairSystem::updateFromMeshTransform(const std::string& meshName, const Matrix4x4& meshTransform) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Find all grooms bound to this mesh and update them
    for (auto& [groomName, groom] : m_grooms) {
        if (groom.boundMeshName == meshName) {
            updateGroomTransform(groomName, meshTransform);
        }
    }
}

HairGroom* HairSystem::getGroomByMesh(const std::string& meshName) {
    for (auto& [groomName, groom] : m_grooms) {
        if (groom.boundMeshName == meshName) {
            return &groom;
        }
    }
    return nullptr;
}

void HairSystem::addStrandsAtPosition(const std::string& groomName, const Vec3& position, 
                                       const Vec3& normal, float radius, int count) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(groomName);
    if (it == m_grooms.end()) return;
    
    HairGroom& groom = it->second;
    
    // Transform world position back to hair local space
    // Hair is stored in world space from initial generation, but if mesh moved
    // we need to transform brush position to match hair's original coordinate space
    Matrix4x4 inverseTransform = groom.transform.inverse();
    Vec3 localPosition = inverseTransform.transform_point(position);
    Vec3 localNormal = inverseTransform.transform_vector(normal).normalize();
    
    // Simple random generator
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    
    // Create a coordinate frame from normal
    Vec3 up = localNormal;
    Vec3 right = Vec3::cross(up, Vec3(0, 1, 0)).normalize();
    if (right.length() < 0.1f) {
        right = Vec3::cross(up, Vec3(1, 0, 0)).normalize();
    }
    Vec3 forward = Vec3::cross(right, up).normalize();
    
    for (int i = 0; i < count; ++i) {
        // Random position within radius
        float angle = dist01(gen) * 6.283f;
        float r = dist01(gen) * radius;
        Vec3 offset = right * (std::cos(angle) * r) + forward * (std::sin(angle) * r);
        Vec3 rootPos = localPosition + offset;
        
        // Create strand
        HairStrand strand;
        strand.rootNormal = localNormal;
        strand.strandID = static_cast<uint32_t>(groom.guides.size() + i);
        strand.type = StrandType::GUIDE;
        strand.randomSeed = dist01(gen);
        
        float length = groom.params.length * (1.0f - groom.params.lengthVariation * dist01(gen));
        
        strand.baseRootPos = rootPos;
        strand.baseLength = length;
        strand.rootNormal = localNormal;
        
        strand.materialID = groom.guides.empty() ? (uint16_t)0 : groom.guides[0].materialID;
        strand.points.resize(groom.params.pointsPerStrand);
        strand.groomedPositions.resize(groom.params.pointsPerStrand);
        for (uint32_t p = 0; p < groom.params.pointsPerStrand; ++p) {
            float t = static_cast<float>(p) / (groom.params.pointsPerStrand - 1);
            strand.groomedPositions[p] = rootPos + localNormal * (t * length);
        }
        
        groom.guides.push_back(std::move(strand));
    }

    
    restyleGroom(groomName);
    groom.isDirty = true;
    m_bvhDirty = true;
    m_statsDirty = true;
}


void HairSystem::removeStrandsAtPosition(const std::string& groomName, const Vec3& position, 
                                          float radius) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(groomName);
    if (it == m_grooms.end()) return;
    
    HairGroom& groom = it->second;
    
    // Transform world position to local space
    Matrix4x4 inverseTransform = groom.transform.inverse();
    Vec3 localPosition = inverseTransform.transform_point(position);
    
    float radiusSq = radius * radius;
    
    // Remove guides within radius (comparing in local space)
    auto removeFunc = [&](std::vector<HairStrand>& strands) {
        strands.erase(
            std::remove_if(strands.begin(), strands.end(), 
                [&](const HairStrand& s) {
                    Vec3 diff = s.rootPosition() - localPosition;
                    return (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z) < radiusSq;
                }
            ),
            strands.end()
        );
    };
    
    size_t oldGuideCount = groom.guides.size();
    size_t oldInterpCount = groom.interpolated.size();
    
    removeFunc(groom.guides);
    removeFunc(groom.interpolated);
    
    if (groom.guides.size() != oldGuideCount || groom.interpolated.size() != oldInterpCount) {
        groom.isDirty = true;
        m_bvhDirty = true;
        m_statsDirty = true;
    }
}

void HairSystem::restyleGroom(const std::string& name, const Physics::ForceFieldManager* forceManager, float time) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    auto it = m_grooms.find(name);
    if (it == m_grooms.end()) return;
    
    HairGroom& groom = it->second;
    const HairGenerationParams& params = groom.params;
    
    // 1. Update all guide strands based on new styling params
    for (auto& strand : groom.guides) {
        strand.points.resize(params.pointsPerStrand);
        
        Vec3 rootPos = strand.baseRootPos;
        Vec3 normal = strand.rootNormal;
        float strandLength = strand.baseLength;
        
        // Initialize groomedPositions if they don't exist or if point count changed
        if (strand.groomedPositions.size() != params.pointsPerStrand) {
            strand.groomedPositions.resize(params.pointsPerStrand);
            float div = (params.pointsPerStrand > 1) ? (float)(params.pointsPerStrand - 1) : 1.0f;
            for (uint32_t p = 0; p < params.pointsPerStrand; ++p) {
                float t = static_cast<float>(p) / div;
                strand.groomedPositions[p] = rootPos + normal * (t * strandLength);
            }
        }
        
        float div = (params.pointsPerStrand > 1) ? (float)(params.pointsPerStrand - 1) : 1.0f;
        for (uint32_t p = 0; p < params.pointsPerStrand; ++p) {
            float t = static_cast<float>(p) / div;
            HairPoint& point = strand.points[p];
            point.v_coord = t;
            
            // Re-apply the same logic as generateOnMesh
            point.position = strand.groomedPositions[p];

            
            if (params.waveAmplitude > 0.0f && params.waveFrequency > 0.0f) {
                float wave = sinf(t * params.waveFrequency * TWO_PI + strand.randomSeed * TWO_PI);
                Vec3 bitangent = (std::abs(normal.y) < 0.9f) ? Vec3::cross(normal, Vec3(0,1,0)) : Vec3::cross(normal, Vec3(1,0,0));
                bitangent.normalize();
                point.position = point.position + bitangent * (wave * params.waveAmplitude * t);
            }

            if (params.curlRadius > 0.0f && params.curlFrequency > 0.0f) {
                float angle = t * params.curlFrequency * TWO_PI + strand.randomSeed * TWO_PI;
                Vec3 bitangent = (std::abs(normal.y) < 0.9f) ? Vec3::cross(normal, Vec3(0,1,0)) : Vec3::cross(normal, Vec3(1,0,0));
                bitangent.normalize();
                Vec3 binormal = Vec3::cross(normal, bitangent).normalize();
                point.position = point.position + (bitangent * cosf(angle) + binormal * sinf(angle)) * (params.curlRadius * t);
            }

            if (params.roughness > 0.0f) {
                Vec3 noise(
                    sinf(t * 3.0f + strand.randomSeed * 10.0f),
                    sinf(t * 4.2f + 1.5f + strand.randomSeed * 11.0f),
                    sinf(t * 2.7f + 0.8f + strand.randomSeed * 9.0f)
                );
                point.position = point.position + noise * (params.roughness * t * strandLength * 0.2f);
            }

            if (params.frizz > 0.0f) {
                ThreadSafeRNG rng(strand.strandID + 12345);
                Vec3 jitter((rng.random()-0.5f)*2,(rng.random()-0.5f)*2,(rng.random()-0.5f)*2);
                point.position = point.position + jitter * (params.frizz * t * strandLength * 0.05f);
            }

            if (params.gravity > 0.0f) {
                point.position.y -= t * t * params.gravity * strandLength;
            }

            // --- Force Field Interaction ---
            if (forceManager && params.forceInfluence > 0.0f) {
                // Evaluate force in world space
                Vec3 currentWorldPos = groom.transform.transform_point(point.position);
                // System filter: is_gas=false, is_particle=true, is_cloth=true, is_rigidbody=false
                Vec3 force = forceManager->evaluateAtFiltered(currentWorldPos, time, Vec3(0,0,0), false, true, true, false);
                
                // Bend the strand based on force (quadratic influence from root to tip)
                // We use local space offset
                point.position = point.position + force * (params.forceInfluence * t * t * 0.1f);
            }

            
            point.radius = params.rootRadius * (1.0f - t) + params.tipRadius * t;
        }
    }

    

    // [FIX] Skinned grooms have their positions updated to current World Space directly.
    // So we must ensure their transform is Identity to prevent double transformation.
    if (!groom.boundTriangles.empty() && groom.boundTriangles[0]->hasSkinData()) {
        groom.transform.setIdentity();
    }

    // 2. Interpolate children
    interpolateChildren(groom);
    
   // m_bvhDirty = true;
}

void HairSystem::regenerateInterpolated(const std::string& groomName) {
    restyleGroom(groomName);
}

void HairSystem::setGravity(const std::string& groomName, float gravity) {
    if (HairGroom* g = getGroom(groomName)) {
        g->params.gravity = gravity;
        restyleGroom(groomName);
    }
}

void HairSystem::setClumpiness(const std::string& groomName, float clump) {
    if (HairGroom* g = getGroom(groomName)) {
        g->params.clumpiness = clump;
        restyleGroom(groomName);
    }
}

void HairSystem::bakeGroomToRest(const std::string& groomName) {
    if (groomName.empty()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(groomName);
    if (it == m_grooms.end()) return;
    HairGroom& groom = it->second;

    // Helper to get position/normal from barycentrics (Copy from updateSkinnedGroom logic)
    auto getPos = [](const Triangle& tri, float u, float v, bool original) {
        float w = 1.0f - u - v;
        if (original) {
            return tri.getOriginalVertexPosition(0) * w + 
                   tri.getOriginalVertexPosition(1) * u + 
                   tri.getOriginalVertexPosition(2) * v;
        } else {
            return tri.getVertexPosition(0) * w + 
                   tri.getVertexPosition(1) * u + 
                   tri.getVertexPosition(2) * v;
        }
    };
    auto getNorm = [](const Triangle& tri, float u, float v, bool original) {
        float w = 1.0f - u - v;
        if (original) {
            return (tri.getOriginalVertexNormal(0) * w + 
                    tri.getOriginalVertexNormal(1) * u + 
                    tri.getOriginalVertexNormal(2) * v).normalize();
        } else {
            return (tri.getVertexNormal(0) * w + 
                    tri.getVertexNormal(1) * u + 
                    tri.getVertexNormal(2) * v).normalize();
        }
    };

    for (auto& strand : groom.guides) {
        if (!strand.groomedPositions.empty()) {
            // Ensure rest capacity
            if (strand.restGroomedPositions.size() != strand.groomedPositions.size()) {
                strand.restGroomedPositions.resize(strand.groomedPositions.size());
            }

            // Case 1: Skinned Mesh
            if (!groom.boundTriangles.empty() && strand.triangleIndex < groom.boundTriangles.size()) {
                const auto& tri = *groom.boundTriangles[strand.triangleIndex];
                
                // 1. Calculate Old Frame (Bind Pose)
                // [FIX] P0 must be World Space at Generation time, so apply Initial Transform
                Vec3 rawP0 = getPos(tri, strand.barycentricUV.u, strand.barycentricUV.v, true);
                Vec3 rawN0 = getNorm(tri, strand.barycentricUV.u, strand.barycentricUV.v, true);
                Vec3 P0 = groom.initialMeshTransform.transform_point(rawP0);
                Vec3 N0 = groom.initialMeshTransform.transform_vector(rawN0).normalize();
                
                // 2. Calculate New Frame (Current Pose)
                Vec3 P1 = getPos(tri, strand.barycentricUV.u, strand.barycentricUV.v, false);
                Vec3 N1 = getNorm(tri, strand.barycentricUV.u, strand.barycentricUV.v, false);
                
                // 3. Compute Rotation (align N0 to N1)
                Quaternion rot = Quaternion::rotationBetween(N0, N1);
                Matrix4x4 R = rot.toMatrix();
                // We need Inverse Rotation for baking: New -> Old
                // R^{-1} = R^T (for rotation matrices)
                
                for (size_t i = 0; i < strand.groomedPositions.size(); ++i) {
                    // Forward: New = P1 + R * (Old - P0)
                    // Inverse: Old - P0 = R^T * (New - P1)
                    //          Old = P0 + R^T * (New - P1)
                    
                    Vec3 currentPt = strand.groomedPositions[i];
                    Vec3 offsetInCurrent = currentPt - P1;
                    Vec3 offsetInRest = R.transpose().transform_vector(offsetInCurrent);
                    strand.restGroomedPositions[i] = P0 + offsetInRest;
                }
            } else {
                // Case 2: Rigid / Static (Just Copy)
                strand.restGroomedPositions = strand.groomedPositions;
            }
        }
    }
}


// ============================================================================
// Serialization Implementation
// ============================================================================

void to_json(nlohmann::json& j, const HairGenerationParams& p) {
    j = nlohmann::json{
        {"guideCount", p.guideCount},
        {"interpolatedPerGuide", p.interpolatedPerGuide},
        {"pointsPerStrand", p.pointsPerStrand},
        {"length", p.length},
        {"lengthVariation", p.lengthVariation},
        {"rootRadius", p.rootRadius},
        {"tipRadius", p.tipRadius},
        {"clumpiness", p.clumpiness},
        {"childRadius", p.childRadius},
        {"curlFrequency", p.curlFrequency},
        {"curlRadius", p.curlRadius},
        {"waveFrequency", p.waveFrequency},
        {"waveAmplitude", p.waveAmplitude},
        {"frizz", p.frizz},
        {"roughness", p.roughness},
        {"gravity", p.gravity},
        {"defaultMaterialID", p.defaultMaterialID},

        {"useTangentShading", p.useTangentShading},
        {"useBSpline", p.useBSpline},
        {"subdivisions", p.subdivisions},
        {"forceInfluence", p.forceInfluence}
    };
}


void from_json(const nlohmann::json& j, HairGenerationParams& p) {
    p.guideCount = j.value("guideCount", 1000u);
    p.interpolatedPerGuide = j.value("interpolatedPerGuide", 4u);
    p.pointsPerStrand = j.value("pointsPerStrand", 8u);
    p.length = j.value("length", 0.1f);
    p.lengthVariation = j.value("lengthVariation", 0.2f);
    p.rootRadius = j.value("rootRadius", 0.001f);
    p.tipRadius = j.value("tipRadius", 0.0001f);
    p.clumpiness = j.value("clumpiness", 0.5f);
    p.childRadius = j.value("childRadius", 0.01f);
    p.curlFrequency = j.value("curlFrequency", 0.0f);
    p.curlRadius = j.value("curlRadius", 0.01f);
    p.waveFrequency = j.value("waveFrequency", 0.0f);
    p.waveAmplitude = j.value("waveAmplitude", 0.0f);
    p.frizz = j.value("frizz", 0.0f);
    p.roughness = j.value("roughness", 0.0f);
    p.gravity = j.value("gravity", 0.0f);
    p.defaultMaterialID = j.value("defaultMaterialID", (uint16_t)0);
    p.useTangentShading = j.value("useTangentShading", true);
    p.useBSpline = j.value("useBSpline", true);
    p.subdivisions = j.value("subdivisions", 2u);
    p.forceInfluence = j.value("forceInfluence", 1.0f);

}


static nlohmann::json mat4ToJson(const Matrix4x4& m) {
    nlohmann::json j = nlohmann::json::array();
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            j.push_back(m.m[i][k]);
    return j;
}

static Matrix4x4 jsonToMat4(const nlohmann::json& j) {
    Matrix4x4 m;
    if (j.is_array() && j.size() == 16) {
        int idx = 0;
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 4; ++k)
                m.m[i][k] = j[idx++];
    }
    return m;
}

nlohmann::json HairSystem::serialize(std::ostream* binaryOut) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    nlohmann::json j;
    nlohmann::json grooms_arr = nlohmann::json::array();

    for (const auto& [name, groom] : m_grooms) {
        nlohmann::json g;
        g["name"] = name;
        g["boundMeshName"] = groom.boundMeshName;
        g["params"] = groom.params;
        g["transform"] = mat4ToJson(groom.transform);
        g["initialMeshTransform"] = mat4ToJson(groom.initialMeshTransform);
        g["material"] = groom.material;

        if (binaryOut) {
            // Binary Serialization (Fast)
            g["storage"] = "binary";
            
            // Record start offset
            std::streampos startPos = binaryOut->tellp();
            g["binaryOffset"] = (long long)startPos;
            
            // Write Guides
            uint32_t numGuides = (uint32_t)groom.guides.size();
            binaryOut->write(reinterpret_cast<const char*>(&numGuides), sizeof(numGuides));
            for (const auto& strand : groom.guides) {
                // Write Metadata
                binaryOut->write(reinterpret_cast<const char*>(&strand.strandID), sizeof(strand.strandID));
                binaryOut->write(reinterpret_cast<const char*>(&strand.materialID), sizeof(strand.materialID));
                int type = (int)strand.type;
                binaryOut->write(reinterpret_cast<const char*>(&type), sizeof(type));
                binaryOut->write(reinterpret_cast<const char*>(&strand.rootRadius), sizeof(strand.rootRadius));
                binaryOut->write(reinterpret_cast<const char*>(&strand.tipRadius), sizeof(strand.tipRadius));
                binaryOut->write(reinterpret_cast<const char*>(&strand.randomSeed), sizeof(strand.randomSeed));
                binaryOut->write(reinterpret_cast<const char*>(&strand.rootUV), sizeof(strand.rootUV));
                binaryOut->write(reinterpret_cast<const char*>(&strand.rootNormal), sizeof(strand.rootNormal));
                binaryOut->write(reinterpret_cast<const char*>(&strand.baseRootPos), sizeof(strand.baseRootPos));
                binaryOut->write(reinterpret_cast<const char*>(&strand.baseLength), sizeof(strand.baseLength));
                binaryOut->write(reinterpret_cast<const char*>(&strand.clumpScale), sizeof(strand.clumpScale));
                binaryOut->write(reinterpret_cast<const char*>(&strand.meshMaterialID), sizeof(strand.meshMaterialID));
                
                // Write Points
                uint32_t numPoints = (uint32_t)strand.points.size();
                binaryOut->write(reinterpret_cast<const char*>(&numPoints), sizeof(numPoints));
                if (numPoints > 0) {
                    binaryOut->write(reinterpret_cast<const char*>(strand.points.data()), numPoints * sizeof(HairPoint));
                }

                // Write Groomed Positions (Usually control points)
                uint32_t numGroomed = (uint32_t)strand.groomedPositions.size();
                binaryOut->write(reinterpret_cast<const char*>(&numGroomed), sizeof(numGroomed));
                if (numGroomed > 0) {
                    binaryOut->write(reinterpret_cast<const char*>(strand.groomedPositions.data()), numGroomed * sizeof(Vec3));
                }
            }

            // Write Interpolated (Optional - can be regenerated, but saving preserves exact state)
            uint32_t numInterpolated = (uint32_t)groom.interpolated.size();
            binaryOut->write(reinterpret_cast<const char*>(&numInterpolated), sizeof(numInterpolated));
            // Same binary format for interpolated strands...
            // Optimization: Maybe skip saving interpolated and just regenerate?
            // BUT: If user Painted heavily, regeneration might lose detail if not deterministic.
            // Let's save them for safety.
            for (const auto& strand : groom.interpolated) {
                // ... (Copy/Paste same logic or helper function?)
                // Helper would be better but let's keep it inline for this specific task
                binaryOut->write(reinterpret_cast<const char*>(&strand.strandID), sizeof(strand.strandID));
                binaryOut->write(reinterpret_cast<const char*>(&strand.materialID), sizeof(strand.materialID));
                int type = (int)strand.type;
                binaryOut->write(reinterpret_cast<const char*>(&type), sizeof(type));
                binaryOut->write(reinterpret_cast<const char*>(&strand.rootRadius), sizeof(strand.rootRadius));
                binaryOut->write(reinterpret_cast<const char*>(&strand.tipRadius), sizeof(strand.tipRadius));
                binaryOut->write(reinterpret_cast<const char*>(&strand.randomSeed), sizeof(strand.randomSeed));
                binaryOut->write(reinterpret_cast<const char*>(&strand.rootUV), sizeof(strand.rootUV));
                binaryOut->write(reinterpret_cast<const char*>(&strand.rootNormal), sizeof(strand.rootNormal));
                binaryOut->write(reinterpret_cast<const char*>(&strand.baseRootPos), sizeof(strand.baseRootPos));
                binaryOut->write(reinterpret_cast<const char*>(&strand.baseLength), sizeof(strand.baseLength));
                binaryOut->write(reinterpret_cast<const char*>(&strand.clumpScale), sizeof(strand.clumpScale));
                binaryOut->write(reinterpret_cast<const char*>(&strand.meshMaterialID), sizeof(strand.meshMaterialID));
                
                uint32_t numPoints = (uint32_t)strand.points.size();
                binaryOut->write(reinterpret_cast<const char*>(&numPoints), sizeof(numPoints));
                if (numPoints > 0) binaryOut->write(reinterpret_cast<const char*>(strand.points.data()), numPoints * sizeof(HairPoint));

                uint32_t numGroomed = (uint32_t)strand.groomedPositions.size();
                binaryOut->write(reinterpret_cast<const char*>(&numGroomed), sizeof(numGroomed));
                if (numGroomed > 0) binaryOut->write(reinterpret_cast<const char*>(strand.groomedPositions.data()), numGroomed * sizeof(Vec3));
            }

            std::streampos endPos = binaryOut->tellp();
            g["binarySize"] = (long long)(endPos - startPos);

        } else {
            // Legacy JSON Fallback
            g["storage"] = "json";
            g["guides"] = groom.guides;
            g["interpolated"] = groom.interpolated;
        }

        grooms_arr.push_back(g);
    }

    j["grooms"] = grooms_arr;
    return j;
}

void HairSystem::deserialize(const nlohmann::json& j, std::istream* binaryIn) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    clearAll();

    if (j.contains("grooms")) {
        for (const auto& g : j["grooms"]) {
            std::string name = g.value("name", "undefined");
            HairGroom& groom = m_grooms[name];
            
            groom.name = name;
            groom.boundMeshName = g.value("boundMeshName", "");
            if (g.contains("params")) groom.params = g["params"];
            if (g.contains("transform")) groom.transform = jsonToMat4(g["transform"]);
            if (g.contains("initialMeshTransform")) groom.initialMeshTransform = jsonToMat4(g["initialMeshTransform"]);
            if (g.contains("material")) groom.material = g["material"];
            
            std::string storage = g.value("storage", "json");

            if (storage == "binary" && binaryIn) {
                // Load from binary stream
                long long offset = g.value("binaryOffset", 0LL);
                binaryIn->seekg(offset);
                
                auto readStrand = [&](HairStrand& s) {
                    binaryIn->read(reinterpret_cast<char*>(&s.strandID), sizeof(s.strandID));
                    binaryIn->read(reinterpret_cast<char*>(&s.materialID), sizeof(s.materialID));
                    int type;
                    binaryIn->read(reinterpret_cast<char*>(&type), sizeof(type));
                    s.type = (StrandType)type;
                    binaryIn->read(reinterpret_cast<char*>(&s.rootRadius), sizeof(s.rootRadius));
                    binaryIn->read(reinterpret_cast<char*>(&s.tipRadius), sizeof(s.tipRadius));
                    binaryIn->read(reinterpret_cast<char*>(&s.randomSeed), sizeof(s.randomSeed));
                    binaryIn->read(reinterpret_cast<char*>(&s.rootUV), sizeof(s.rootUV));
                    binaryIn->read(reinterpret_cast<char*>(&s.rootNormal), sizeof(s.rootNormal));
                    binaryIn->read(reinterpret_cast<char*>(&s.baseRootPos), sizeof(s.baseRootPos));
                    binaryIn->read(reinterpret_cast<char*>(&s.baseLength), sizeof(s.baseLength));
                    binaryIn->read(reinterpret_cast<char*>(&s.clumpScale), sizeof(s.clumpScale));
                    binaryIn->read(reinterpret_cast<char*>(&s.meshMaterialID), sizeof(s.meshMaterialID));
                    
                    uint32_t numPoints;
                    binaryIn->read(reinterpret_cast<char*>(&numPoints), sizeof(numPoints));
                    if (numPoints > 0) {
                        s.points.resize(numPoints);
                        binaryIn->read(reinterpret_cast<char*>(s.points.data()), numPoints * sizeof(HairPoint));
                    }

                    uint32_t numGroomed;
                    binaryIn->read(reinterpret_cast<char*>(&numGroomed), sizeof(numGroomed));
                    if (numGroomed > 0) {
                        s.groomedPositions.resize(numGroomed);
                        binaryIn->read(reinterpret_cast<char*>(s.groomedPositions.data()), numGroomed * sizeof(Vec3));
                    }
                };

                // Read Guides
                uint32_t numGuides;
                binaryIn->read(reinterpret_cast<char*>(&numGuides), sizeof(numGuides));
                groom.guides.resize(numGuides);
                for(uint32_t i=0; i<numGuides; ++i) readStrand(groom.guides[i]);

                // Read Interpolated
                uint32_t numInterpolated;
                binaryIn->read(reinterpret_cast<char*>(&numInterpolated), sizeof(numInterpolated));
                groom.interpolated.resize(numInterpolated);
                for(uint32_t i=0; i<numInterpolated; ++i) readStrand(groom.interpolated[i]);

            } else {
                // Legacy JSON Load
                if (g.contains("guides")) {
                    groom.guides = g["guides"].get<std::vector<HairStrand>>();
                }
                if (g.contains("interpolated")) {
                    groom.interpolated = g["interpolated"].get<std::vector<HairStrand>>();
                }
            }
            
            groom.isDirty = true;
            m_statsDirty = true;
        }
    }

    m_bvhDirty = true;
    m_statsDirty = true;
}

// ============================================================================
// Skinning / Animation Update
// ============================================================================

void HairSystem::updateSkinnedGroom(const std::string& groomName) {
    if (groomName.empty()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(groomName);
    if (it == m_grooms.end()) return;
    
    HairGroom& groom = it->second;
    if (groom.boundTriangles.empty()) return;

    bool anyChange = false;

    // Helper to get position from barycentrics
    auto getPos = [](const Triangle& tri, float u, float v, bool original) {
        float w = 1.0f - u - v;
        if (original) {
            return tri.getOriginalVertexPosition(0) * w + 
                   tri.getOriginalVertexPosition(1) * u + 
                   tri.getOriginalVertexPosition(2) * v;
        } else {
            return tri.getVertexPosition(0) * w + 
                   tri.getVertexPosition(1) * u + 
                   tri.getVertexPosition(2) * v;
        }
    };
    
    // Helper to get normal from barycentrics
    auto getNorm = [](const Triangle& tri, float u, float v, bool original) {
        float w = 1.0f - u - v;
        if (original) {
            return (tri.getOriginalVertexNormal(0) * w + 
                    tri.getOriginalVertexNormal(1) * u + 
                    tri.getOriginalVertexNormal(2) * v).normalize();
        } else {
            return (tri.getVertexNormal(0) * w + 
                    tri.getVertexNormal(1) * u + 
                    tri.getVertexNormal(2) * v).normalize();
        }
    };

    // Update guides based on skin deformation
    for (auto& strand : groom.guides) {
        if (strand.triangleIndex >= groom.boundTriangles.size()) continue;
        
        const auto& tri = *groom.boundTriangles[strand.triangleIndex];
        
        // 1. Calculate Old Frame (Bind Pose) - Apply Initial Transform to match World Space Rest positions
        Vec3 rawP0 = getPos(tri, strand.barycentricUV.u, strand.barycentricUV.v, true);
        Vec3 rawN0 = getNorm(tri, strand.barycentricUV.u, strand.barycentricUV.v, true);
        Vec3 P0 = groom.initialMeshTransform.transform_point(rawP0);
        Vec3 N0 = groom.initialMeshTransform.transform_vector(rawN0).normalize();
        
        // 2. Calculate New Frame (Current Pose)
        Vec3 P1 = getPos(tri, strand.barycentricUV.u, strand.barycentricUV.v, false);
        Vec3 N1 = getNorm(tri, strand.barycentricUV.u, strand.barycentricUV.v, false);
        
        // 3. Compute Rotation (align N0 to N1)
        Quaternion rot = Quaternion::rotationBetween(N0, N1);
        Matrix4x4 R = rot.toMatrix();
        
        // 4. Update all control points
        // Use restGroomedPositions as the source of truth for the shape
        if (strand.restGroomedPositions.empty()) {
            strand.restGroomedPositions = strand.groomedPositions;
        }
        
        // Just in case size changed (e.g. painting added points)
        if (strand.restGroomedPositions.size() != strand.groomedPositions.size()) {
             strand.restGroomedPositions = strand.groomedPositions;
             continue; // Skip update this frame, wait for next where size matches or re-init
        }

        for (size_t i = 0; i < strand.groomedPositions.size(); ++i) {
            // Transform point: New = P1 + R * (Old - P0)
            Vec3 oldPt = strand.restGroomedPositions[i];
            Vec3 localOffset = oldPt - P0;
            Vec3 revolvedOffset = R.transform_vector(localOffset); 
            strand.groomedPositions[i] = P1 + revolvedOffset;
        }
        anyChange = true;
    }

    if (anyChange) {
        // Re-interpolate children to follow new guide positions
        interpolateChildren(groom);
        
        // Important: Update BVH
        m_bvhDirty = true;
        
        // Reset transform since we baked the deformation into the points
        // NOTE: This assumes groomedPositions are now in World Space
        groom.transform.setIdentity(); 
    }
}

// ============================================================================
// GPU Grooming / Brushing Implementation
// ============================================================================

} // namespace Hair
