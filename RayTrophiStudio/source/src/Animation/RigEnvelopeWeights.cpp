#include "Animation/RigEnvelopeWeights.h"
#include "Animation/RigBindMath.h"
#include "Animation/RigPoseAuthoring.h"
#include "Animation/RigPosePreview.h"
#include "Animation/SkinWeightContract.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include "scene_data.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace RigAuthoring {
namespace {
struct EnvelopeSegment {
    SkinSegment segment;
    double startRadius = 0;
    double endRadius = 0;
    double falloff = 2;
};

struct InfluenceCount {
    size_t nonzero = 0;
    size_t significant = 0;
};

Vec3 origin(const Matrix4x4 &matrix) {
    return Vec3(matrix.m[0][3], matrix.m[1][3], matrix.m[2][3]);
}

bool validSettings(const EnvelopeWeightSettings &settings) {
    return std::isfinite(settings.torsoRadius) && settings.torsoRadius >= .02f &&
           settings.torsoRadius <= .5f && std::isfinite(settings.limbRadius) &&
           settings.limbRadius >= .01f && settings.limbRadius <= .3f &&
           std::isfinite(settings.extremityRadius) && settings.extremityRadius >= .005f &&
           settings.extremityRadius <= .2f && std::isfinite(settings.falloff) &&
           settings.falloff >= .5f && settings.falloff <= 8.f;
}

bool validProfile(const EnvelopeBoneProfile &profile) {
    return !profile.bone.empty() && std::isfinite(profile.startRadius) &&
           profile.startRadius >= .005f && profile.startRadius <= .5f &&
           std::isfinite(profile.endRadius) && profile.endRadius >= .005f &&
           profile.endRadius <= .5f && std::isfinite(profile.startExtension) &&
           profile.startExtension >= 0.f && profile.startExtension <= .75f &&
           std::isfinite(profile.endExtension) && profile.endExtension >= 0.f &&
           profile.endExtension <= .75f && std::isfinite(profile.falloff) &&
           profile.falloff >= .5f && profile.falloff <= 8.f;
}

std::string categoryForRole(const std::string &role) {
    if (role.find("_hand.") != std::string::npos) {
        return "digit";
    }
    if (role.find("hand") != std::string::npos || role.find("foot") != std::string::npos ||
        role.find("toe") != std::string::npos || role.find("paw") != std::string::npos) {
        return "extremity";
    }
    if (role == "pelvis" || role == "head" || role.find("spine") != std::string::npos ||
        role.find("neck") != std::string::npos) {
        return "torso";
    }
    return "limb";
}

float radiusFractionForRole(const std::string &role, const EnvelopeWeightSettings &settings) {
    const auto category = categoryForRole(role);
    if (category == "digit")
        return (std::max)(.005f, settings.extremityRadius * .3f);
    if (category == "extremity")
        return settings.extremityRadius;
    if (category == "torso")
        return settings.torsoRadius;
    return settings.limbRadius;
}

double radiusForRole(const std::string &role, const EnvelopeWeightSettings &settings,
                     double height) {
    return radiusFractionForRole(role, settings) * height;
}

VertexInfluences weights(const Vec3 &point, const std::vector<EnvelopeSegment> &segments,
                         bool &fallback) {
    std::unordered_map<int, double> scores;
    const EnvelopeSegment *nearest = nullptr;
    double nearestDistance = std::numeric_limits<double>::infinity();
    for (const auto &value : segments) {
        const auto delta = value.segment.end - value.segment.start;
        const auto relative = point - value.segment.start;
        const double lengthSquared = delta.length_squared();
        const double t = lengthSquared > 0
                             ? std::clamp(double(relative.dot(delta)) / lengthSquared, 0.0, 1.0)
                             : 0.0;
        const auto closest = value.segment.start + delta * static_cast<float>(t);
        const double distanceSquared = (point - closest).length_squared();
        if (distanceSquared < nearestDistance) {
            nearestDistance = distanceSquared;
            nearest = &value;
        }
        const double distance = std::sqrt(distanceSquared);
        const double radius = value.startRadius + (value.endRadius - value.startRadius) * t;
        if (distance >= radius) {
            continue;
        }
        const double score =
            std::pow((std::max)(0.0, 1.0 - distance / radius), value.falloff);
        auto found = scores.find(value.segment.bone);
        if (found == scores.end()) {
            scores.emplace(value.segment.bone, score);
        } else {
            found->second = (std::max)(found->second, score);
        }
    }
    fallback = scores.empty();
    if (fallback && nearest) {
        scores.emplace(nearest->segment.bone, 1.0);
    }
    std::vector<std::pair<int, double>> ranked(scores.begin(), scores.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    if (ranked.size() > 4) {
        ranked.resize(4);
    }
    VertexInfluences result;
    for (const auto &entry : ranked) {
        result.emplace_back(entry.first, static_cast<float>(entry.second));
    }
    canonicalizeInfluences(result);
    if (result.empty() && nearest) {
        fallback = true;
        result.emplace_back(nearest->segment.bone, 1.f);
    }
    return result;
}

bool build(const SceneData &scene, const std::string &character,
           const EnvelopeWeightSettings &settings, uint64_t expectedRevision, bool checkRevision,
           const std::vector<EnvelopeBoneProfile> *profileOverride, EnvelopeWeightState &state,
           nlohmann::json &report, std::string &error) {
    report = nullptr;
    error.clear();
    state = {};
    if (!validSettings(settings)) {
        error = "rig_envelope_invalid_settings";
        return false;
    }
    const SceneData::ImportedModelContext *model = nullptr;
    for (const auto &candidate : scene.importedModelContexts) {
        if (candidate.importName == character) {
            if (model) {
                error = "ambiguous_character";
                return false;
            }
            model = &candidate;
        }
    }
    if (!model) {
        error = "unknown_character";
        return false;
    }
    if (checkRevision && model->rigRevision != expectedRevision) {
        error = "rig_edit_stale_revision";
        return false;
    }
    if (model->rigBoundMeshes.empty()) {
        error = "rig_envelope_requires_authored_binding";
        return false;
    }
    if (model->rigRevision == (std::numeric_limits<uint64_t>::max)()) {
        error = "rig_revision_overflow";
        return false;
    }
    if (model->nodeHierarchy.size() > 4096 || model->rigBoundMeshes.size() > 256) {
        error = "rig_envelope_limit";
        return false;
    }
    const auto &profiles = profileOverride ? *profileOverride : model->rigEnvelopeProfiles;
    if (profiles.size() > 4096) {
        error = "rig_envelope_limit";
        return false;
    }
    std::unordered_map<std::string, const EnvelopeBoneProfile *> profileByBone;
    for (const auto &profile : profiles) {
        if (!validProfile(profile) || !profileByBone.emplace(profile.bone, &profile).second) {
            error = "rig_envelope_invalid_profile";
            return false;
        }
    }
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(model->nodeHierarchy, nullptr, 0, joints, error)) {
        return false;
    }
    if (joints.empty()) {
        error = "rig_envelope_invalid_height";
        return false;
    }
    float minimum = origin(joints.front().world).y;
    float maximum = minimum;
    for (const auto &joint : joints) {
        minimum = (std::min)(minimum, origin(joint.world).y);
        maximum = (std::max)(maximum, origin(joint.world).y);
    }
    const double height = double(maximum) - minimum;
    if (!std::isfinite(height) || height <= 1e-6) {
        error = "rig_envelope_invalid_height";
        return false;
    }
    std::unordered_map<std::string, std::string> roles;
    std::unordered_set<std::string> nonDeforming;
    for (const auto &role : model->rigAnatomy.roles) {
        roles[role.bone] = role.role;
        if (role.role == "root") {
            nonDeforming.insert(role.bone);
        }
    }
    std::vector<EnvelopeSegment> segments;
    std::unordered_set<std::string> segmentOwners;
    for (size_t i = 0; i < joints.size(); ++i) {
        const auto parent = model->nodeHierarchy.nodes[i].parent;
        if (parent < 0 || nonDeforming.count(joints[static_cast<size_t>(parent)].name)) {
            continue;
        }
        const auto &bone = joints[static_cast<size_t>(parent)].name;
        segmentOwners.insert(bone);
        const auto index = scene.boneData.boneNameToIndex.find(bone);
        if (index == scene.boneData.boneNameToIndex.end() ||
            index->second > static_cast<unsigned int>((std::numeric_limits<int>::max)())) {
            error = "rig_bind_invalid_indices";
            return false;
        }
        const auto role = roles.find(bone);
        const std::string roleName = role == roles.end() ? std::string() : role->second;
        EnvelopeSegment value;
        value.segment = {static_cast<int>(index->second),
                         origin(joints[static_cast<size_t>(parent)].world),
                         origin(joints[i].world)};
        const double defaultRadius = radiusForRole(roleName, settings, height);
        value.startRadius = defaultRadius;
        value.endRadius = defaultRadius;
        value.falloff = settings.falloff;
        const auto baseDelta = value.segment.end - value.segment.start;
        const double baseLength = std::sqrt(baseDelta.length_squared());
        if (!std::isfinite(baseLength) || baseLength <= 1e-9) {
            error = "rig_bind_zero_length_segment";
            return false;
        }
        const auto profile = profileByBone.find(bone);
        if (profile != profileByBone.end()) {
            const auto *custom = profile->second;
            const auto direction = baseDelta * static_cast<float>(1.0 / baseLength);
            value.segment.start -=
                direction * static_cast<float>(custom->startExtension * baseLength);
            value.segment.end += direction * static_cast<float>(custom->endExtension * baseLength);
            value.startRadius = custom->startRadius * height;
            value.endRadius = custom->endRadius * height;
            value.falloff = custom->falloff;
        }
        if (segmentDistanceSquared(value.segment.end, {value.segment.bone, value.segment.start,
                                                       value.segment.start}) <= 0 ||
            !std::isfinite(value.startRadius) || !std::isfinite(value.endRadius) ||
            value.startRadius <= 0 || value.endRadius <= 0) {
            error = "rig_bind_zero_length_segment";
            return false;
        }
        segments.push_back(value);
    }
    if (segments.empty()) {
        error = "rig_bind_requires_segments";
        return false;
    }
    for (const auto &profile : profiles) {
        if (!segmentOwners.count(profile.bone)) {
            error = "rig_envelope_bone_has_no_segment";
            return false;
        }
    }
    std::unordered_map<int, InfluenceCount> counts;
    size_t vertexCount = 0;
    size_t fallbackCount = 0;
    for (const auto &meshName : model->rigBoundMeshes) {
        std::shared_ptr<TriangleMesh> mesh;
        for (const auto &object : scene.world.objects) {
            auto candidate = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (candidate && candidate->nodeName == meshName) {
                if (mesh) {
                    error = "ambiguous_mesh_name";
                    return false;
                }
                mesh = std::move(candidate);
            }
        }
        if (!mesh || !mesh->geometry) {
            error = "rig_envelope_missing_bound_mesh";
            return false;
        }
        if (mesh->num_vertices() > 2000000 - vertexCount) {
            error = "rig_envelope_limit";
            return false;
        }
        vertexCount += mesh->num_vertices();
        auto geometry = std::make_shared<DNA::GeometryDetail>(*mesh->geometry);
        const auto *positions = geometry->get_positions_orig();
        if (!positions || geometry->get_positions_orig_count() < mesh->num_vertices()) {
            error = "rig_envelope_missing_bind_positions";
            return false;
        }
        geometry->skin_weights.assign(mesh->num_vertices(), {});
        for (size_t vertex = 0; vertex < mesh->num_vertices(); ++vertex) {
            bool fallback = false;
            auto row = weights(positions[vertex], segments, fallback);
            if (row.empty()) {
                error = "rig_bind_empty_weights";
                return false;
            }
            fallbackCount += fallback ? 1 : 0;
            for (const auto &influence : row) {
                auto &count = counts[influence.first];
                ++count.nonzero;
                count.significant += influence.second > .05f ? 1 : 0;
            }
            geometry->skin_weights[vertex] = std::move(row);
        }
        geometry->last_skinned_pose_hash = 0;
        state.parts.push_back({std::move(mesh), std::move(geometry)});
    }
    std::vector<std::pair<std::string, InfluenceCount>> orderedCounts;
    for (const auto &entry : counts) {
        std::string bone;
        for (const auto &name : scene.boneData.boneNameToIndex) {
            if (name.second == static_cast<unsigned int>(entry.first)) {
                bone = name.first;
                break;
            }
        }
        orderedCounts.emplace_back(std::move(bone), entry.second);
    }
    std::sort(orderedCounts.begin(), orderedCounts.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    auto influenceCounts = nlohmann::json::array();
    for (const auto &entry : orderedCounts) {
        influenceCounts.push_back({{"bone", entry.first},
                                   {"nonzero_vertices", entry.second.nonzero},
                                   {"significant_vertices", entry.second.significant}});
    }
    state.character = character;
    state.settings = settings;
    state.profiles = profiles;
    state.revision = model->rigRevision + 1;
    report = {{"character", character},
              {"rig_revision", model->rigRevision},
              {"algorithm", "anatomical_capsule_v1"},
              {"vertex_count", vertexCount},
              {"part_count", state.parts.size()},
              {"segment_count", segments.size()},
              {"fallback_vertices", fallbackCount},
              {"torso_radius", settings.torsoRadius},
              {"limb_radius", settings.limbRadius},
              {"extremity_radius", settings.extremityRadius},
              {"falloff", settings.falloff},
              {"influence_counts", std::move(influenceCounts)}};
    return true;
}
} // namespace

bool validEnvelopeWeightSettings(const EnvelopeWeightSettings &settings) {
    return validSettings(settings);
}

bool envelopeSegmentViews(const SceneData &scene, const std::string &character,
                          const EnvelopeWeightSettings &settings,
                          std::vector<EnvelopeSegmentView> &segments, std::string &error) {
    segments.clear();
    error.clear();
    if (!validSettings(settings)) {
        error = "rig_envelope_invalid_settings";
        return false;
    }
    const SceneData::ImportedModelContext *model = nullptr;
    for (const auto &candidate : scene.importedModelContexts) {
        if (candidate.importName == character) {
            if (model) {
                error = "ambiguous_character";
                return false;
            }
            model = &candidate;
        }
    }
    if (!model) {
        error = "unknown_character";
        return false;
    }
    if (model->rigBoundMeshes.empty()) {
        error = "rig_envelope_requires_authored_binding";
        return false;
    }
    std::vector<PreviewJoint> rest;
    if (!sampleRigPose(model->nodeHierarchy, nullptr, 0, rest, error) || rest.empty()) {
        if (error.empty())
            error = "rig_envelope_invalid_height";
        return false;
    }
    float minimum = origin(rest.front().world).y;
    float maximum = minimum;
    for (const auto &joint : rest) {
        minimum = (std::min)(minimum, origin(joint.world).y);
        maximum = (std::max)(maximum, origin(joint.world).y);
    }
    const double height = double(maximum) - minimum;
    if (!std::isfinite(height) || height <= 1e-6) {
        error = "rig_envelope_invalid_height";
        return false;
    }
    Matrix4x4 placement = model->rigSceneTransform;
    for (const auto &object : scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || !mesh->transform)
            continue;
        if (std::find(model->rigBoundMeshes.begin(), model->rigBoundMeshes.end(),
                      mesh->nodeName) != model->rigBoundMeshes.end()) {
            placement = mesh->transform->base;
            break;
        }
    }
    std::unordered_map<std::string, Matrix4x4> displayed;
    for (const auto &joint : rest)
        displayed.emplace(joint.name, placement * joint.world);
    const bool posing = scene.rigView.pose.active && scene.rigView.pose.character == character;
    if (posing) {
        RayTrophi::NodeHierarchy pose;
        std::vector<PreviewJoint> joints;
        if (!currentPoseHierarchy(scene, character, pose, error, true) ||
            !sampleRigPose(pose, nullptr, 0, joints, error)) {
            return false;
        }
        for (const auto &joint : joints)
            displayed[joint.name] = placement * joint.world;
    } else {
        const auto poseView = scene.rigView.pose_views.find(character);
        const bool restView =
            poseView != scene.rigView.pose_views.end() && poseView->second == "rest";
        if (!restView) {
            for (const auto &joint : model->rigJointGlobals)
                displayed[joint.first] = placement * joint.second;
        }
    }
    std::unordered_map<std::string, std::string> roles;
    std::unordered_set<std::string> nonDeforming;
    for (const auto &role : model->rigAnatomy.roles) {
        roles[role.bone] = role.role;
        if (role.role == "root")
            nonDeforming.insert(role.bone);
    }
    std::unordered_map<std::string, const EnvelopeBoneProfile *> profiles;
    for (const auto &profile : model->rigEnvelopeProfiles)
        profiles.emplace(profile.bone, &profile);
    for (size_t index = 0; index < model->nodeHierarchy.nodes.size(); ++index) {
        const auto &child = model->nodeHierarchy.nodes[index];
        if (child.parent < 0)
            continue;
        if (static_cast<size_t>(child.parent) >= model->nodeHierarchy.nodes.size()) {
            error = "rig_bind_invalid_hierarchy";
            return false;
        }
        const auto &parentNode = model->nodeHierarchy.nodes[static_cast<size_t>(child.parent)];
        if (nonDeforming.count(parentNode.uniqueName))
            continue;
        const auto parent = displayed.find(parentNode.uniqueName);
        const auto end = displayed.find(child.uniqueName);
        if (parent == displayed.end() || end == displayed.end())
            continue;
        const auto role = roles.find(parentNode.uniqueName);
        const std::string roleName = role == roles.end() ? std::string() : role->second;
        EnvelopeSegmentView view;
        view.bone = parentNode.uniqueName;
        view.category = categoryForRole(roleName);
        view.start = origin(parent->second);
        view.end = origin(end->second);
        const float defaultRadius = static_cast<float>(radiusForRole(roleName, settings, height));
        view.startRadius = defaultRadius;
        view.endRadius = defaultRadius;
        const auto profile = profiles.find(view.bone);
        if (profile != profiles.end() && validProfile(*profile->second)) {
            const auto delta = view.end - view.start;
            const float length = std::sqrt(delta.length_squared());
            if (length > 1e-6f) {
                const auto direction = delta * (1.f / length);
                view.start -= direction * (profile->second->startExtension * length);
                view.end += direction * (profile->second->endExtension * length);
                view.startRadius = profile->second->startRadius * static_cast<float>(height);
                view.endRadius = profile->second->endRadius * static_cast<float>(height);
            }
        }
        if ((view.end - view.start).length_squared() > 1e-12f)
            segments.push_back(std::move(view));
    }
    if (segments.empty()) {
        error = "rig_bind_requires_segments";
        return false;
    }
    return true;
}

bool getEnvelopeBoneProfile(const SceneData &scene, const std::string &character,
                            const std::string &bone, EnvelopeBoneProfile &profile,
                            uint64_t &revision, bool &overridden, std::string &error) {
    error.clear();
    for (const auto &model : scene.importedModelContexts) {
        if (model.importName != character)
            continue;
        if (model.rigBoundMeshes.empty()) {
            error = "rig_envelope_requires_authored_binding";
            return false;
        }
        for (const auto &custom : model.rigEnvelopeProfiles) {
            if (custom.bone == bone) {
                profile = custom;
                revision = model.rigRevision;
                overridden = true;
                return true;
            }
        }
        bool ownsSegment = false;
        for (const auto &node : model.nodeHierarchy.nodes) {
            if (node.parent >= 0 &&
                static_cast<size_t>(node.parent) < model.nodeHierarchy.nodes.size() &&
                model.nodeHierarchy.nodes[static_cast<size_t>(node.parent)].uniqueName == bone) {
                ownsSegment = true;
                break;
            }
        }
        if (!ownsSegment) {
            error = "rig_envelope_bone_has_no_segment";
            return false;
        }
        std::string roleName;
        for (const auto &role : model.rigAnatomy.roles)
            if (role.bone == bone) {
                roleName = role.role;
                break;
            }
        EnvelopeWeightSettings settings{model.rigEnvelopeTorsoRadius,
                                        model.rigEnvelopeLimbRadius,
                                        model.rigEnvelopeExtremityRadius,
                                        model.rigEnvelopeFalloff};
        profile = {};
        profile.bone = bone;
        profile.startRadius = radiusFractionForRole(roleName, settings);
        profile.endRadius = profile.startRadius;
        profile.falloff = settings.falloff;
        revision = model.rigRevision;
        overridden = false;
        return true;
    }
    error = "unknown_character";
    return false;
}

bool stageEnvelopeBoneProfile(const SceneData &scene, const std::string &character,
                              const EnvelopeBoneProfile &profile, uint64_t expectedRevision,
                              EnvelopeWeightState &state, nlohmann::json &report,
                              std::string &error) {
    if (!validProfile(profile)) {
        error = "rig_envelope_invalid_profile";
        return false;
    }
    for (const auto &model : scene.importedModelContexts) {
        if (model.importName != character)
            continue;
        EnvelopeBoneProfile current;
        uint64_t revision = 0;
        bool overridden = false;
        if (!getEnvelopeBoneProfile(scene, character, profile.bone, current, revision, overridden,
                                    error)) {
            return false;
        }
        if (revision != expectedRevision) {
            error = "rig_edit_stale_revision";
            return false;
        }
        auto profiles = model.rigEnvelopeProfiles;
        auto found = std::find_if(profiles.begin(), profiles.end(), [&](const auto &candidate) {
            return candidate.bone == profile.bone;
        });
        if (found == profiles.end())
            profiles.push_back(profile);
        else
            *found = profile;
        EnvelopeWeightSettings settings{model.rigEnvelopeTorsoRadius,
                                        model.rigEnvelopeLimbRadius,
                                        model.rigEnvelopeExtremityRadius,
                                        model.rigEnvelopeFalloff};
        if (!build(scene, character, settings, expectedRevision, true, &profiles, state, report,
                   error)) {
            return false;
        }
        report["bone_profile"] = {{"bone", profile.bone},
                                  {"start_radius", profile.startRadius},
                                  {"end_radius", profile.endRadius},
                                  {"start_extension", profile.startExtension},
                                  {"end_extension", profile.endExtension},
                                  {"falloff", profile.falloff}};
        return true;
    }
    error = "unknown_character";
    return false;
}

bool previewEnvelopeWeights(const SceneData &scene, const std::string &character,
                            const EnvelopeWeightSettings &settings, nlohmann::json &report,
                            std::string &error) {
    EnvelopeWeightState state;
    return build(scene, character, settings, 0, false, nullptr, state, report, error);
}

bool stageEnvelopeWeights(const SceneData &scene, const std::string &character,
                          const EnvelopeWeightSettings &settings, uint64_t expectedRevision,
                          EnvelopeWeightState &state, nlohmann::json &report, std::string &error) {
    return build(scene, character, settings, expectedRevision, true, nullptr, state, report, error);
}
} // namespace RigAuthoring
