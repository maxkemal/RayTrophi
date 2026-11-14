// Studio CPU ray queries; excluded from Player.
#include "Hair/HairSystem.h"
#include <embree4/rtcore.h>
#include <algorithm>
#include <cmath>

namespace Hair {
void HairSystem::buildBVHImpl(bool includeInterpolated) {
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
    // MEDIUM, not HIGH. HIGH buys spatial splits, which pay off for triangles with bad
    // aspect ratios and very little for curve primitives — but it costs a large multiple
    // of the build time, and this scene is rebuilt on every groom edit (brush drag runs
    // it at 15 Hz) even when Vulkan/OptiX is doing the actual rendering and the only
    // consumer left is brush picking.
    rtcSetSceneBuildQuality(m_embreeScene, RTC_BUILD_QUALITY_MEDIUM);

    // Clear buffers and geom mapping for fresh build
    m_smoothTangents.clear();
    m_segMap.clear();
    m_geomToGroom.clear();
    m_geomToTangentOffset.clear();

    // Add hair geometry for each groom
    for (auto& [name, groom] : m_grooms) {
        if (!groom.isVisible)
            continue;
        // ... (rest of the logic remains similar but with mapping data)
        std::vector<const HairStrand*> allStrands;
        for (const auto& s : groom.guides)
            allStrands.push_back(&s);
        if (includeInterpolated) {
            for (const auto& s : groom.interpolated)
                allStrands.push_back(&s);
        }

        if (allStrands.empty())
            continue;

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

        if (totalSegmentsCount == 0)
            continue;

        RTCGeometryType curveType = groom.params.useBSpline ? RTC_GEOMETRY_TYPE_ROUND_BSPLINE_CURVE
                                                            : RTC_GEOMETRY_TYPE_ROUND_LINEAR_CURVE;
        RTCGeometry hairGeo = rtcNewGeometry(embreeDevice, curveType);

        if (groom.params.useBSpline) {
            rtcSetGeometryTessellationRate(hairGeo, (float)std::pow(2, groom.params.subdivisions));
        }

        float* vertices =
            (float*)rtcSetNewGeometryBuffer(hairGeo, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4,
                                            sizeof(float) * 4, totalVerticesCount);
        unsigned int* indices = (unsigned int*)rtcSetNewGeometryBuffer(
            hairGeo, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT, sizeof(unsigned int),
            totalSegmentsCount);

        size_t vertexOffset = 0;
        size_t segmentOffset = 0;
        size_t tangentStartIndex = m_smoothTangents.size();
        size_t mapStartIndex = m_segMap.size();

        m_smoothTangents.resize(tangentStartIndex + totalSegmentsCount * 3);
        m_segMap.resize(mapStartIndex + totalSegmentsCount);

        uint32_t groomNameHash = 0;
        for (char c : name)
            groomNameHash = groomNameHash * 31 + c;

        for (size_t sIdx = 0; sIdx < allStrands.size(); ++sIdx) {
            const auto* strand = allStrands[sIdx];
            size_t n = strand->points.size();

            if (groom.params.useBSpline) {
                if (n < 4)
                    continue;

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

                Vec3 pn = transform.transform_point(pts[n - 1].position);
                Vec3 pn_minus1 = transform.transform_point(pts[n - 2].position);
                Vec3 p_plus1 = pn * 2.0f - pn_minus1;

                vertices[vertexOffset * 4 + 0] = p_plus1.x;
                vertices[vertexOffset * 4 + 1] = p_plus1.y;
                vertices[vertexOffset * 4 + 2] = p_plus1.z;
                vertices[vertexOffset * 4 + 3] = pts[n - 1].radius * avgScale;
                vertexOffset++;

                for (size_t i = 0; i < n - 1; ++i) {
                    indices[segmentOffset] = static_cast<unsigned int>(strandVertexStart + i);

                    Vec3 cp0(vertices[(strandVertexStart + i) * 4 + 0],
                             vertices[(strandVertexStart + i) * 4 + 1],
                             vertices[(strandVertexStart + i) * 4 + 2]);
                    Vec3 cp1(vertices[(strandVertexStart + i + 1) * 4 + 0],
                             vertices[(strandVertexStart + i + 1) * 4 + 1],
                             vertices[(strandVertexStart + i + 1) * 4 + 2]);
                    Vec3 cp2(vertices[(strandVertexStart + i + 2) * 4 + 0],
                             vertices[(strandVertexStart + i + 2) * 4 + 1],
                             vertices[(strandVertexStart + i + 2) * 4 + 2]);
                    Vec3 cp3(vertices[(strandVertexStart + i + 3) * 4 + 0],
                             vertices[(strandVertexStart + i + 3) * 4 + 1],
                             vertices[(strandVertexStart + i + 3) * 4 + 2]);

                    // Exact B-Spline tangents for continuity
                    Vec3 T0 = (cp2 - cp0);
                    Vec3 T1 = (cp2 - cp1);
                    Vec3 T2 = (cp3 - cp1);

                    // Normalize all to ensure quadratic interpolation weighting is purely
                    // directional
                    float l0 = T0.length();
                    if (l0 > 1e-6f)
                        T0 = T0 / l0;
                    else
                        T0 = (cp2 - cp1).normalize();
                    float l1 = T1.length();
                    if (l1 > 1e-6f)
                        T1 = T1 / l1;
                    else
                        T1 = (cp2 - cp1).normalize();
                    float l2 = T2.length();
                    if (l2 > 1e-6f)
                        T2 = T2 / l2;
                    else
                        T2 = (cp2 - cp1).normalize();

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
                if (n < 2)
                    continue;

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
                    size_t idx_v3 = static_cast<size_t>(
                        std::min<int>(static_cast<int>(i) + 2, static_cast<int>(n) - 1));
                    Vec3 v3 = transform.transform_point(pts[idx_v3].position);

                    m_smoothTangents[tangentStartIndex + segmentOffset * 3 + 0] =
                        (v2 - v0).normalize();
                    m_smoothTangents[tangentStartIndex + segmentOffset * 3 + 1] =
                        (v2 - v1).normalize();
                    m_smoothTangents[tangentStartIndex + segmentOffset * 3 + 2] =
                        (v3 - v1).normalize();

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
    // Do NOT clear m_bvhDirty here — that latch is the GPU-resync signal and is owned by
    // buildBVH()/isBVHDirty(). This function only builds the CPU scene; its own staleness
    // flag (m_cpuBvhDirty) was already cleared by ensureBVH() before calling in.

    // Clear per-groom dirty flags now that they are in the BVH.
    for (auto& [name, groom] : m_grooms) {
        groom.isDirty = false;
    }
}

bool HairSystem::intersect(const Vec3& rayOrigin, const Vec3& rayDir, float tMin, float tMax,
                           HairHitInfo& hitInfo) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Materialize the deferred CPU BVH — this is the only place it is actually needed.
    ensureBVH();
    if (!m_embreeScene || m_grooms.empty())
        return false;

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
            tangent = (T0 * (1.0f - u) * (1.0f - u) + T1 * 2.0f * u * (1.0f - u) + T2 * u * u)
                          .normalize();

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
    if (Vec3::dot(Ng, viewDir) < 0)
        Ng = Ng * -1.0f;
    Ng = Ng.normalize();

    // Robust Azimuthal Offset Calculation (h)
    // We use the camera-right vector in the plane perpendicular to the tangent.
    Vec3 camRight = Vec3::cross(tangent, viewDir);
    float crLen = camRight.length();

    if (crLen > 1e-6f) {
        camRight = camRight / crLen;
        // h is the signed distance of the ray from the cylinder axis in the plane of the cross
        // section. It's equivalent to the projection of the radial normal (Ng) onto the 'side'
        // vector.
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
        hitInfo.groomName = it->second;

        // Use shared material if assigned, otherwise fallback to per-groom material
        if (!groom.materialName.empty()) {
            auto matIt = m_materials.find(groom.materialName);
            if (matIt != m_materials.end()) {
                hitInfo.material = matIt->second;
            } else {
                hitInfo.material = groom.material;
            }
        } else {
            hitInfo.material = groom.material;
        }
    }

    return true;
}

bool HairSystem::intersectVolumetric(const Vec3& rayOrigin, const Vec3& rayDir, float tMin,
                                     float tMax, float searchRadius, HairHitInfo& hitInfo) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Materialize the deferred CPU BVH — this is the only place it is actually needed.
    ensureBVH();
    if (!m_embreeScene)
        return false;

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
            if (std::abs(b) > std::abs(c) && std::abs(b) > 1e-12f)
                tc = d / b;
            else if (std::abs(c) > 1e-12f)
                tc = e / c;
            else
                tc = 0.0f;
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
        if (!groom.isVisible)
            continue;
        Matrix4x4 l2w = groom.transform;

        for (const auto& strand : groom.guides) {
            if (strand.groomedPositions.size() < 2)
                continue;

            // Optional: Quick bounding box/root check for performance
            Vec3 rootWorld = l2w.transform_point(strand.baseRootPos);
            if ((rootWorld - rayOrigin).length() > (tMax + strand.baseLength))
                continue;

            for (size_t i = 0; i < strand.groomedPositions.size() - 1; ++i) {
                Vec3 pA = l2w.transform_point(strand.groomedPositions[i]);
                Vec3 pB = l2w.transform_point(strand.groomedPositions[i + 1]);

                float t, v_seg;
                Vec3 pOnSeg = ClosestPointSegmentRay(pA, pB, rayOrigin, D, t, v_seg);

                if (t < tMin || t > tMax)
                    continue;

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

                        // Use shared material if assigned
                        if (!groom.materialName.empty()) {
                            auto matIt = m_materials.find(groom.materialName);
                            if (matIt != m_materials.end()) {
                                hitInfo.material = matIt->second;
                            } else {
                                hitInfo.material = groom.material;
                            }
                        } else {
                            hitInfo.material = groom.material;
                        }

                        hitInfo.strandID = strand.strandID;

                        float baseV = (float)i / (strand.groomedPositions.size() - 1);
                        float stepV = 1.0f / (strand.groomedPositions.size() - 1);
                        hitInfo.v = baseV + v_seg * stepV;

                        hitInfo.tangent = (pB - pA).normalize();
                        Vec3 normalVec = pOnRay - pOnSeg;
                        float nLen = normalVec.length();
                        if (nLen > 1e-8f)
                            hitInfo.normal = normalVec / nLen;
                        else
                            hitInfo.normal = Vec3(0, 1, 0);

                        found = true;
                    }
                }
            }
        }
    }

    return found;
}

bool HairSystem::occluded(const Vec3& rayOrigin, const Vec3& rayDir, float tMin, float tMax) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Materialize the deferred CPU BVH — this is the only place it is actually needed.
    ensureBVH();
    if (!m_embreeScene)
        return false;

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

void HairSystem::releaseCpuBVH() {
    if (m_embreeScene) {
        rtcReleaseScene(m_embreeScene);
        m_embreeScene = nullptr;
    }
}

} // namespace Hair
