
#include "Hair/HairSystem.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include "Quaternion.h"
#include <mutex>

namespace Hair {

void HairSystem::updateSkinnedGroom(const std::string& groomName, const std::vector<Matrix4x4>& boneMatrices) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(groomName);
    if (it == m_grooms.end()) return;
    
    HairGroom& groom = it->second;
    if (groom.bound.empty()) return;

    // --- CHANGE DETECTION ---
    // If we have bone matrices, compare with cached to avoid setting dirty flags every frame
    static std::unordered_map<std::string, std::vector<Matrix4x4>> lastMatricesMap;
    auto& lastMatrices = lastMatricesMap[groomName];
    
    bool matricesChanged = false;
    if (lastMatrices.size() != boneMatrices.size()) {
        matricesChanged = true;
    } else {
        // Optimized check: check translation and diagonal of each bone matrix
        for (size_t i = 0; i < boneMatrices.size(); ++i) {
            const auto& m1 = boneMatrices[i];
            const auto& m2 = lastMatrices[i];
            if (std::abs(m1.m[3][0] - m2.m[3][0]) > 0.0001f ||
                std::abs(m1.m[3][1] - m2.m[3][1]) > 0.0001f ||
                std::abs(m1.m[3][2] - m2.m[3][2]) > 0.0001f ||
                std::abs(m1.m[0][0] - m2.m[0][0]) > 0.0001f ||
                std::abs(m1.m[1][1] - m2.m[1][1]) > 0.0001f ||
                std::abs(m1.m[2][2] - m2.m[2][2]) > 0.0001f) {
                matricesChanged = true;
                break;
            }
        }
    }
    
    // Force field liveness: while the field time changes (animated/simulated field) keep
    // ticking so the strands respond every frame even if the skeleton is momentarily still;
    // a static field time lets a static pose stop (bones already drive motion during play).
    static std::unordered_map<std::string, float> lastEnvTimeMap;
    float& lastEnvTime = lastEnvTimeMap[groomName];
    const bool forceWake = (m_envForce != nullptr) && (lastEnvTime != m_envTime);

    // Also check if groom itself is dirty (e.g. from UI parameter change)
    if (!matricesChanged && !groom.isDirty && !forceWake) return;

    lastMatrices = boneMatrices;
    lastEnvTime  = m_envTime;
    groom.isDirty = false; // We are processing it now

    bool anyChange = false;

    // Lock the scalp meshes once, and hoist the node-level matrices out of the loop:
    // rootMat is the SAME for every strand of a groom (it is the mesh node transform),
    // so inverting/transposing it per strand was pure per-frame waste.
    const auto boundLocked = groom.bound.lockMeshes();
    const int boundCount   = (int)groom.bound.size();
    Matrix4x4 rootMat = Matrix4x4::identity();
    {
        Triangle probe;
        if (const Triangle* t0 = groom.bound.resolve(boundLocked, 0, probe))
            rootMat = t0->getTransformMatrix();
    }
    const Matrix4x4 rootNormMat = rootMat.inverse().transpose();

    // Each guide is independent (reads the scalp, writes only its own points), so this
    // parallelises cleanly. Signed loop variable: MSVC only implements OpenMP 2.0.
    const int guideCount = (int)groom.guides.size();
    #pragma omp parallel for schedule(static)
    for (int gi = 0; gi < guideCount; ++gi) {
        HairStrand& strand = groom.guides[gi];
        // Unsigned compare on purpose: an unbound strand carries 0xFFFFFFFF, which a
        // signed compare would read as -1 and wrongly accept.
        if (strand.triangleIndex >= (uint32_t)boundCount) continue;
        Triangle scratch;   // per-thread stack facade
        const Triangle* triPtr = groom.bound.resolve(boundLocked, strand.triangleIndex, scratch);
        if (!triPtr) continue;
        const Triangle& tri = *triPtr;

        float u = strand.barycentricUV.u;
        float v = strand.barycentricUV.v;
        float w = 1.0f - u - v;

        // 1. Calculate BIND FRAME (B0)
        Vec3 rawV0 = tri.getOriginalVertexPosition(0);
        Vec3 rawV1 = tri.getOriginalVertexPosition(1);
        Vec3 rawV2 = tri.getOriginalVertexPosition(2);
        
        Vec3 rawP0 = rawV0 * w + rawV1 * u + rawV2 * v;
        Vec3 rawN0 = (tri.getOriginalVertexNormal(0) * w + 
                      tri.getOriginalVertexNormal(1) * u + 
                      tri.getOriginalVertexNormal(2) * v).normalize();
        
        // Consistent tangent in local bind space
        Vec3 rawT0 = (rawV1 - rawV0).normalize();

        // ORIGIN of the bind frame must be the point the rest pose was authored around,
        // not a freshly measured one. restGroomedPositions comes from generation, where
        // the root was sampled off the scalp's CURRENT surface via sampleTriangleSurface;
        // re-deriving the origin here from getOriginalVertexPosition + initialMeshTransform
        // lands in a different space whenever the scalp was posed (or the node transform
        // is not identity), and the constant difference is added to every strand — which
        // is exactly the "hair follows the object but sits far away" symptom. baseRootPos
        // is the generation-time root and is by construction in the same space as the rest
        // pose, so (rest - P0) stays the small root-relative shape offset it is meant to be.
        // The rotation part was already correct, so the measured frame axes are kept.
        Vec3 P0 = strand.baseRootPos;
        Vec3 N0 = groom.initialMeshTransform.transform_vector(rawN0).normalize();
        Vec3 T0 = groom.initialMeshTransform.transform_vector(rawT0).normalize();
        
        Vec3 B0 = Vec3::cross(N0, T0).normalize();
        T0 = Vec3::cross(B0, N0).normalize();

        // 2. Calculate SKINNED FRAME (B1)
        Matrix4x4 skinMat = Matrix4x4::zero();
        bool hasBones = false;
        
        for (int vi = 0; vi < 3; ++vi) {
            float bary = (vi == 0) ? w : ((vi == 1) ? u : v);
            const auto& weights = tri.getSkinBoneWeights(vi);
            for (const auto& [boneIdx, weight] : weights) {
                if (boneIdx < (int)boneMatrices.size() && weight > 1e-6f) {
                    const Matrix4x4& m = boneMatrices[boneIdx];
                    float blendedWeight = weight * bary;
                    for(int r=0; r<4; ++r) {
                        for(int c=0; c<4; ++c) {
                            skinMat.m[r][c] += m.m[r][c] * blendedWeight;
                        }
                    }
                    hasBones = true;
                }
            }
        }
        
        if (!hasBones) {
            skinMat = Matrix4x4::identity();
        }

        Matrix4x4 normalMat = skinMat.inverse().transpose();

        Vec3 P1 = (rootMat * skinMat).transform_point(rawP0);
        Vec3 N1 = (rootNormMat * normalMat).transform_vector(rawN0).normalize();
        Vec3 T1 = (rootMat * skinMat).transform_vector(rawT0).normalize();
        
        Vec3 B1 = Vec3::cross(N1, T1).normalize();
        T1 = Vec3::cross(B1, N1).normalize();

        // 3. Construct Frame-to-Frame Transformation Matrix (R)
        // We want R that transforms (X0, Y0, Z0) to (X1, Y1, Z1)
        // Basis matrices (as columns): M0 = [T0 B0 N0], M1 = [T1 B1 N1]
        // R = M1 * M0^-1 = M1 * M0^T (since orthonormal)
        Matrix4x4 R;
        R.m[0][0] = T1.x*T0.x + B1.x*B0.x + N1.x*N0.x;
        R.m[0][1] = T1.x*T0.y + B1.x*B0.y + N1.x*N0.y;
        R.m[0][2] = T1.x*T0.z + B1.x*B0.z + N1.x*N0.z;
        R.m[1][0] = T1.y*T0.x + B1.y*B0.x + N1.y*N0.x;
        R.m[1][1] = T1.y*T0.y + B1.y*B0.y + N1.y*N0.y;
        R.m[1][2] = T1.y*T0.z + B1.y*B0.z + N1.y*N0.z;
        R.m[2][0] = T1.z*T0.x + B1.z*B0.x + N1.z*N0.x;
        R.m[2][1] = T1.z*T0.y + B1.z*B0.y + N1.z*N0.y;
        R.m[2][2] = T1.z*T0.z + B1.z*B0.z + N1.z*N0.z;
        R.m[3][3] = 1.0f;

        if (strand.groomedPositions.size() != strand.restGroomedPositions.size()) {
            strand.restGroomedPositions = strand.groomedPositions;
        }

        for (size_t i = 0; i < strand.groomedPositions.size(); ++i) {
            Vec3 oldPt = strand.restGroomedPositions[i];
            Vec3 localOffset = oldPt - P0;
            Vec3 revolvedOffset = R.transform_vector(localOffset); 
            strand.groomedPositions[i] = P1 + revolvedOffset;
        }
    }
    anyChange = guideCount > 0;

    if (anyChange) {
        restyleGroom(groomName, m_envForce, m_envTime);   // apply the scene force field in the Verlet
        m_bvhDirty = true;
    }
}



} // namespace Hair
