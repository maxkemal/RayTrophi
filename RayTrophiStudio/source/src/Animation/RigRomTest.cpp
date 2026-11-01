#include "Animation/RigRomTest.h"
#include "Animation/AnimationData.h"
#include "Animation/NodeHierarchy.h"
#include "Animation/RigAnatomy.h"
#include "Animation/RigBindMath.h"
#include "Animation/RigJointRules.h"
#include "Animation/RigPoseAuthoringMath.h"
#include "Animation/RigPosePreview.h"
#include "DNA/GeometryDetail.h"
#include "Quaternion.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace RigAuthoring {
namespace {
constexpr float kPi = 3.14159265359f;
// A triangle whose area falls below half of its rest area is reported as pinched (collapse,
// candy-wrapper twist); above twice its rest area as stretched.
constexpr float kPinch = .5f;
constexpr float kStretch = 2.f;
constexpr size_t kMaxMotions = 512;

struct Motion {
    std::string bone;
    std::string label;
    Vec3 axis;
    float degrees = 0;
};

struct Triangle {
    uint32_t v[3];
    float restArea = 0;
    Vec3 restNormal;
    int dominant = -1; // Skin index carrying most of the triangle's weight.
};

struct MeshCache {
    const RomTestMesh *mesh = nullptr;
    std::vector<Triangle> triangles;
};

struct MotionResult {
    float minRatio = std::numeric_limits<float>::infinity();
    float maxRatio = 0;
    size_t flipped = 0, pinched = 0, stretched = 0;
    std::string worstMesh;
    uint32_t worstVertex = 0;
    int worstBone = -1;
};

Vec3 origin(const Matrix4x4 &matrix) {
    return Vec3(matrix.m[0][3], matrix.m[1][3], matrix.m[2][3]);
}

size_t index(const RayTrophi::NodeHierarchy &hierarchy, const std::string &key) {
    return static_cast<size_t>(hierarchy.find(key) - hierarchy.nodes.data());
}

Vec3 unit(const Vec3 &value, const Vec3 &fallback) {
    return value.length_squared() > 1e-12f ? value.normalize() : fallback;
}

Quaternion axisAngle(const Vec3 &axis, float radians) {
    const float sine = std::sin(radians * .5f);
    return Quaternion(std::cos(radians * .5f), axis.x * sine, axis.y * sine, axis.z * sine);
}

Matrix4x4 rotated(const Matrix4x4 &source, const Quaternion &delta) {
    Vec3 position, scale;
    Quaternion rotation;
    RayTrophi::decomposeTRS(source, position, rotation, scale);
    rotation = delta * rotation;
    rotation.normalize();
    return Matrix4x4::translation(position) * rotation.toMatrix();
}

bool startsWith(const std::string &value, const char *prefix) {
    return value.rfind(prefix, 0) == 0;
}

// Limb motions follow the anatomy: root swing/spread/twist, the middle joint folds away from
// its bend hint (the knee moves forward, so the shin folds back), the end joint twists and
// flexes. Spine, neck and head flex, side-bend and twist about the character frame.
bool buildMotions(const RayTrophi::NodeHierarchy &h, const RigAnatomy &anatomy,
                  const std::vector<PreviewJoint> &joints, float scale,
                  std::vector<Motion> &motions, std::string &error) {
    auto position = [&](const std::string &bone) { return origin(joints[index(h, bone)].world); };
    const Vec3 up(0, 1, 0);
    Vec3 forward(0, 0, 0);
    for (const auto &chain : anatomy.chains) {
        const bool leg = chain.name.size() >= 4 &&
                         chain.name.compare(chain.name.size() - 4, 4, "_leg") == 0;
        if (!leg || chain.bones.size() < 4)
            continue;
        auto flat = position(chain.bones[3]) - position(chain.bones[2]);
        flat -= up * Vec3::dot(flat, up);
        if (flat.length_squared() > 1e-12f)
            forward += flat.normalize();
    }
    forward = unit(forward, Vec3(0, 0, 1));
    const Vec3 left = unit(Vec3::cross(up, forward), Vec3(1, 0, 0));
    auto add = [&](const std::string &bone, const std::string &label, const Vec3 &axis,
                   float degrees) { motions.push_back({bone, label, axis, degrees * scale}); };

    std::vector<IKControl> limbs;
    std::string limbError;
    if (!buildLimbIKControls(anatomy, h, limbs, limbError))
        limbs.clear(); // A rig with no limb chains still gets its spine tested.
    for (const auto &limb : limbs) {
        const auto a = position(limb.root), b = position(limb.mid), t = position(limb.tip);
        const auto dir = unit(b - a, up);
        const auto hint = limb.bendHint.length_squared() > 0 ? limb.bendHint.normalize() : forward;
        const auto swing = unit(Vec3::cross(dir, hint), left);
        const auto spread = unit(Vec3::cross(dir, swing), forward);
        add(limb.root, limb.name + ".swing+", swing, 60);
        add(limb.root, limb.name + ".swing-", swing, -60);
        add(limb.root, limb.name + ".spread+", spread, 45);
        add(limb.root, limb.name + ".spread-", spread, -45);
        add(limb.root, limb.name + ".twist+", dir, 45);
        add(limb.root, limb.name + ".twist-", dir, -45);
        const auto lower = unit(t - b, dir);
        add(limb.mid, limb.name + ".bend", unit(Vec3::cross(lower, hint * -1.f), swing), 110);
        add(limb.tip, limb.name + ".end_twist+", lower, 60);
        add(limb.tip, limb.name + ".end_twist-", lower, -60);
        add(limb.tip, limb.name + ".end_flex+", swing, 45);
        add(limb.tip, limb.name + ".end_flex-", swing, -45);
    }
    for (const auto &role : anatomy.roles) {
        if (!startsWith(role.role, "spine") && !startsWith(role.role, "neck") &&
            role.role != "head")
            continue;
        add(role.bone, role.role + ".flex+", left, 30);
        add(role.bone, role.role + ".flex-", left, -30);
        add(role.bone, role.role + ".side+", forward, 25);
        add(role.bone, role.role + ".side-", forward, -25);
        add(role.bone, role.role + ".twist+", up, 35);
        add(role.bone, role.role + ".twist-", up, -35);
    }
    if (motions.empty()) {
        error = "rig_rom_no_motions";
        return false;
    }
    if (motions.size() > kMaxMotions) {
        error = "rig_rom_limit";
        return false;
    }
    return true;
}

// Rotates one bone about its own origin by a rig-space delta, then applies the joint limits so
// the test never asks for more than the rig allows.
bool posed(const RayTrophi::NodeHierarchy &h, const RigAnatomy &anatomy,
           const std::vector<PreviewJoint> &joints, const Motion &motion,
           RayTrophi::NodeHierarchy &output, bool &limited, std::string &error) {
    auto pose = h;
    auto &node = pose.nodes[index(pose, motion.bone)];
    Quaternion parent;
    if (node.parent >= 0) {
        parent = Quaternion::fromMatrix(joints[static_cast<size_t>(node.parent)].world);
        parent.normalize();
    }
    const auto delta = axisAngle(motion.axis, motion.degrees * kPi / 180.f);
    node.localBind = rotated(node.localBind, parent.conjugate() * delta * parent);
    std::vector<std::string> hits;
    if (!constrainJointPose(h, anatomy.joints, pose, output, hits, error))
        return false;
    limited = !hits.empty();
    return true;
}

bool cacheMesh(const RomTestMesh &mesh, MeshCache &cache, std::string &error) {
    const auto *geometry = mesh.geometry;
    const auto *positions = geometry ? geometry->get_positions_orig() : nullptr;
    if (!positions || geometry->get_positions_orig_count() < mesh.vertices ||
        geometry->skin_weights.size() < mesh.vertices) {
        error = "rig_rom_missing_bind_data";
        return false;
    }
    const auto &indices = geometry->indices;
    if (indices.size() % 3 != 0) {
        error = "rig_rom_invalid_indices";
        return false;
    }
    cache.mesh = &mesh;
    cache.triangles.reserve(indices.size() / 3);
    for (size_t f = 0; f + 2 < indices.size(); f += 3) {
        Triangle triangle;
        for (int k = 0; k < 3; ++k) {
            triangle.v[k] = indices[f + static_cast<size_t>(k)];
            if (triangle.v[k] >= mesh.vertices) {
                error = "rig_rom_invalid_indices";
                return false;
            }
        }
        const auto cross = Vec3::cross(positions[triangle.v[1]] - positions[triangle.v[0]],
                                       positions[triangle.v[2]] - positions[triangle.v[0]]);
        triangle.restArea = cross.length() * .5f;
        if (!std::isfinite(triangle.restArea) || triangle.restArea <= 1e-14f)
            continue; // Degenerate at rest: no ratio to measure.
        triangle.restNormal = cross.normalize();
        std::vector<std::pair<int, float>> totals;
        for (const auto vertex : triangle.v) {
            for (const auto &influence : geometry->skin_weights[vertex]) {
                auto found = std::find_if(totals.begin(), totals.end(), [&](const auto &entry) {
                    return entry.first == influence.first;
                });
                if (found == totals.end())
                    totals.push_back(influence);
                else
                    found->second += influence.second;
            }
        }
        float best = 0;
        for (const auto &entry : totals) {
            if (entry.second > best) {
                best = entry.second;
                triangle.dominant = entry.first;
            }
        }
        cache.triangles.push_back(triangle);
    }
    return true;
}
} // namespace

bool runRomTest(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                const std::unordered_map<std::string, int> &boneIndex,
                const std::vector<RomTestMesh> &meshes, const RomTestSettings &settings,
                nlohmann::json &report, AnimationData *clip, std::string &error) {
    error.clear();
    if (!std::isfinite(settings.scale) || settings.scale < .1f || settings.scale > 1.5f ||
        settings.framesPerMotion < 4 || settings.framesPerMotion > 120 ||
        !std::isfinite(settings.fps) || settings.fps < 1 || settings.fps > 120) {
        error = "rig_rom_invalid_settings";
        return false;
    }
    if (meshes.empty()) {
        error = "rig_rom_requires_bound_mesh";
        return false;
    }
    std::vector<PreviewJoint> rest;
    if (!sampleRigPose(hierarchy, nullptr, 0, rest, error))
        return false;
    std::vector<Motion> motions;
    if (!buildMotions(hierarchy, anatomy, rest, settings.scale, motions, error))
        return false;

    // Skin index -> hierarchy joint, and the inverse rest of every joint.
    int maxSkin = -1;
    for (const auto &entry : boneIndex)
        maxSkin = (std::max)(maxSkin, entry.second);
    std::vector<int> skinToJoint(static_cast<size_t>(maxSkin + 1), -1);
    std::vector<std::string> skinNames(static_cast<size_t>(maxSkin + 1));
    for (const auto &entry : boneIndex) {
        if (entry.second < 0 || !hierarchy.find(entry.first))
            continue;
        skinToJoint[static_cast<size_t>(entry.second)] =
            static_cast<int>(index(hierarchy, entry.first));
        skinNames[static_cast<size_t>(entry.second)] = entry.first;
    }
    std::vector<Matrix4x4> restInverse(rest.size());
    for (size_t j = 0; j < rest.size(); ++j) {
        if (!bindAffineInverse(rest[j].world, restInverse[j])) {
            error = "rig_rom_invalid_rest";
            return false;
        }
    }
    std::vector<MeshCache> caches(meshes.size());
    size_t triangleCount = 0, vertexCount = 0;
    for (size_t m = 0; m < meshes.size(); ++m) {
        if (!cacheMesh(meshes[m], caches[m], error))
            return false;
        triangleCount += caches[m].triangles.size();
        vertexCount += meshes[m].vertices;
    }
    if (vertexCount * motions.size() > 400000000ull) {
        error = "rig_rom_limit";
        return false;
    }

    std::vector<std::string> bones;
    for (const auto &node : hierarchy.nodes)
        bones.push_back(node.uniqueName);
    const int half = settings.framesPerMotion / 2;
    if (clip) {
        clip->rigAuthoring = true;
        clip->duration = 1;
        clip->ticksPerSecond = settings.fps;
        clip->startFrame = 0;
        clip->endFrame = static_cast<int>(motions.size()) * settings.framesPerMotion;
    }

    size_t unknownInfluences = 0;
    size_t pinchedMotions = 0, foldedMotions = 0, stretchedMotions = 0, limitedMotions = 0;
    float overallMin = std::numeric_limits<float>::infinity();
    auto rows = nlohmann::json::array();
    std::vector<Matrix4x4> skin(skinToJoint.size());
    std::vector<Vec3> deformed;
    for (size_t i = 0; i < motions.size(); ++i) {
        const auto &motion = motions[i];
        RayTrophi::NodeHierarchy pose;
        bool limited = false;
        if (!posed(hierarchy, anatomy, rest, motion, pose, limited, error))
            return false;
        std::vector<PreviewJoint> joints;
        if (!sampleRigPose(pose, nullptr, 0, joints, error))
            return false;
        for (size_t s = 0; s < skinToJoint.size(); ++s) {
            const int joint = skinToJoint[s];
            if (joint >= 0)
                skin[s] = joints[static_cast<size_t>(joint)].world *
                          restInverse[static_cast<size_t>(joint)];
        }
        MotionResult result;
        for (const auto &cache : caches) {
            const auto *geometry = cache.mesh->geometry;
            const auto *positions = geometry->get_positions_orig();
            deformed.assign(cache.mesh->vertices, Vec3(0, 0, 0));
            for (size_t v = 0; v < cache.mesh->vertices; ++v) {
                Vec3 sum(0, 0, 0);
                float total = 0;
                for (const auto &influence : geometry->skin_weights[v]) {
                    if (influence.first < 0 ||
                        static_cast<size_t>(influence.first) >= skinToJoint.size() ||
                        skinToJoint[static_cast<size_t>(influence.first)] < 0) {
                        ++unknownInfluences;
                        continue;
                    }
                    sum += skin[static_cast<size_t>(influence.first)].transform_point(positions[v]) *
                           influence.second;
                    total += influence.second;
                }
                deformed[v] = total > 1e-8f ? sum * (1.f / total) : positions[v];
            }
            for (const auto &triangle : cache.triangles) {
                const auto cross = Vec3::cross(deformed[triangle.v[1]] - deformed[triangle.v[0]],
                                               deformed[triangle.v[2]] - deformed[triangle.v[0]]);
                const float ratio = cross.length() * .5f / triangle.restArea;
                if (ratio < result.minRatio) {
                    result.minRatio = ratio;
                    result.worstMesh = cache.mesh->name;
                    result.worstVertex = triangle.v[0];
                    result.worstBone = triangle.dominant;
                }
                result.maxRatio = (std::max)(result.maxRatio, ratio);
                result.pinched += ratio < kPinch ? 1 : 0;
                result.stretched += ratio > kStretch ? 1 : 0;
                // Fold-over: the deformed normal turned against where the dominant bone alone
                // would have carried the rest normal.
                if (triangle.dominant >= 0 &&
                    static_cast<size_t>(triangle.dominant) < skinToJoint.size() &&
                    skinToJoint[static_cast<size_t>(triangle.dominant)] >= 0) {
                    const auto expected = skin[static_cast<size_t>(triangle.dominant)]
                                              .transform_vector(triangle.restNormal);
                    result.flipped += Vec3::dot(cross, expected) < 0 ? 1 : 0;
                }
            }
        }
        pinchedMotions += result.pinched ? 1 : 0;
        foldedMotions += result.flipped ? 1 : 0;
        stretchedMotions += result.stretched ? 1 : 0;
        limitedMotions += limited ? 1 : 0;
        overallMin = (std::min)(overallMin, result.minRatio);
        const std::string worstBone =
            result.worstBone >= 0 && static_cast<size_t>(result.worstBone) < skinNames.size()
                ? skinNames[static_cast<size_t>(result.worstBone)]
                : std::string();
        rows.push_back({{"motion", motion.label},
                        {"bone", motion.bone},
                        {"degrees", motion.degrees},
                        {"limited", limited},
                        {"min_area_ratio", std::isfinite(result.minRatio) ? result.minRatio : 1.f},
                        {"max_area_ratio", result.maxRatio},
                        {"pinched_triangles", result.pinched},
                        {"stretched_triangles", result.stretched},
                        {"folded_triangles", result.flipped},
                        {"worst", {{"mesh", result.worstMesh},
                                   {"vertex", result.worstVertex},
                                   {"bone", worstBone}}},
                        {"frame_peak", static_cast<int>(i) * settings.framesPerMotion + half}});
        if (clip) {
            const double start = double(i) * settings.framesPerMotion / settings.fps;
            if (!insertPoseKeys(*clip, hierarchy, bones, start, error) ||
                !insertPoseKeys(*clip, pose, bones, start + double(half) / settings.fps, error))
                return false;
        }
    }
    if (clip && !insertPoseKeys(*clip, hierarchy, bones,
                                double(motions.size()) * settings.framesPerMotion / settings.fps,
                                error))
        return false;

    std::vector<size_t> order(rows.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return rows[a]["min_area_ratio"].get<float>() < rows[b]["min_area_ratio"].get<float>();
    });
    auto worst = nlohmann::json::array();
    for (size_t i = 0; i < order.size() && i < 10; ++i)
        worst.push_back(rows[order[i]]);
    report = {{"kind", "rom_test_v1"},
              {"motions", motions.size()},
              {"meshes", meshes.size()},
              {"vertices", vertexCount},
              {"triangles", triangleCount},
              {"scale", settings.scale},
              {"frames_per_motion", settings.framesPerMotion},
              {"clip_frames", static_cast<int>(motions.size()) * settings.framesPerMotion + 1},
              {"thresholds", {{"pinch_area_ratio", kPinch}, {"stretch_area_ratio", kStretch}}},
              {"summary",
               {{"min_area_ratio", std::isfinite(overallMin) ? overallMin : 1.f},
                {"pinched_motions", pinchedMotions},
                {"stretched_motions", stretchedMotions},
                {"folded_motions", foldedMotions},
                {"limited_motions", limitedMotions},
                {"unknown_influences", unknownInfluences}}},
              {"worst", worst},
              {"rows", rows}};
    return true;
}
} // namespace RigAuthoring
