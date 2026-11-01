#include "Animation/RigAutoLandmarks.h"
#include "Animation/RigFitting.h"
#include "Animation/RigMeshSpace.h"
#include "Animation/RigPreflight.h"
#include "Quaternion.h"
#include "Transform.h"
#include "TriangleMesh.h"
#include "scene_data.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace RigAuthoring {
namespace {
// Below this a joint is listed for the user to check.
constexpr float kLowConfidence = .5f;
constexpr size_t kMaxPoints = 300000;

struct Mark {
    Vec3 p;
    float confidence = .5f;
    std::string source = "template";
};

struct Cloud {
    std::vector<Vec3> points;
    Vec3 lo, hi;
    float height = 0;
    float centerX = 0;
};

bool readVec(const nlohmann::json &value, Vec3 &out) {
    if (!value.is_array() || value.size() != 3)
        return false;
    for (int i = 0; i < 3; ++i)
        if (!value[i].is_number())
            return false;
    out = Vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
    return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z);
}

Vec3 transformPoint(const Matrix4x4 &m, const Vec3 &p) {
    return Vec3(m.m[0][0] * p.x + m.m[0][1] * p.y + m.m[0][2] * p.z + m.m[0][3],
                m.m[1][0] * p.x + m.m[1][1] * p.y + m.m[1][2] * p.z + m.m[1][3],
                m.m[2][0] * p.x + m.m[2][1] * p.y + m.m[2][2] * p.z + m.m[2][3]);
}

// Same fit space as fitSetup: viewport flat positions with the mesh's final transform.
bool buildCloud(const SceneData &scene, const std::string &name, Cloud &cloud, std::string &error) {
    auto mesh = resolveFitMesh(scene, name, error);
    if (!mesh || !mesh->geometry)
        return false;
    const auto transform = mesh->transform ? mesh->transform->getFinal() : Matrix4x4::identity();
    const auto *positions = meshSpacePositions(*mesh->geometry, mesh->hasSkinWeights());
    const size_t count = mesh->num_vertices();
    if (!positions || !count) {
        error = "mesh_has_no_positions";
        return false;
    }
    const size_t stride = (std::max)(size_t(1), (count + kMaxPoints - 1) / kMaxPoints);
    const float inf = std::numeric_limits<float>::infinity();
    cloud.lo = Vec3(inf, inf, inf);
    cloud.hi = Vec3(-inf, -inf, -inf);
    double sumX = 0;
    for (size_t i = 0; i < count; i += stride) {
        const auto p = transformPoint(transform, positions[i]);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;
        cloud.points.push_back(p);
        cloud.lo = Vec3((std::min)(cloud.lo.x, p.x), (std::min)(cloud.lo.y, p.y),
                        (std::min)(cloud.lo.z, p.z));
        cloud.hi = Vec3((std::max)(cloud.hi.x, p.x), (std::max)(cloud.hi.y, p.y),
                        (std::max)(cloud.hi.z, p.z));
        sumX += p.x;
    }
    cloud.height = cloud.hi.y - cloud.lo.y;
    if (cloud.points.empty() || !std::isfinite(cloud.height) || cloud.height <= 1e-6f) {
        error = "rig_fit_requires_height";
        return false;
    }
    cloud.centerX = static_cast<float>(sumX / double(cloud.points.size()));
    return true;
}

// Centroid of the mesh points in a thin slab through c perpendicular to d, within radius of c,
// projected back into that plane. This puts a joint in the middle of its limb or torso section.
bool slabCenter(const Cloud &cloud, const Vec3 &c, const Vec3 &dir, float radius, float thickness,
                Vec3 &out, size_t &count, float sideSign = 0.f) {
    const auto d = dir.normalize();
    Vec3 sum(0, 0, 0);
    count = 0;
    for (const auto &p : cloud.points) {
        if (sideSign != 0.f && (p.x - cloud.centerX) * sideSign < 0)
            continue;
        const auto rel = p - c;
        const float along = Vec3::dot(rel, d);
        if (std::fabs(along) > thickness)
            continue;
        const auto planar = rel - d * along;
        if (planar.length_squared() > radius * radius)
            continue;
        sum += planar;
        ++count;
    }
    if (count == 0)
        return false;
    out = c + sum * (1.f / float(count));
    return true;
}

float slabConfidence(size_t count, float moved, float radius) {
    const float density = (std::min)(1.f, float(count) / 24.f);
    const float drift =
        moved < .6f * radius ? 1.f : (std::max)(0.f, 1.f - (moved - .6f * radius) / (.6f * radius));
    return density * drift;
}

class Placer {
  public:
    Placer(const Cloud &cloud, std::unordered_map<std::string, Mark> &marks,
           const std::unordered_map<std::string, std::string> &roleBones)
        : cloud(cloud), marks(marks), roleBones(roleBones), H(cloud.height) {}

    std::string bone(const std::string &role) const {
        const auto found = roleBones.find(role);
        return found == roleBones.end() ? std::string() : found->second;
    }
    bool has(const std::string &role) const {
        const auto name = bone(role);
        return !name.empty() && marks.count(name);
    }
    Mark &at(const std::string &role) { return marks.at(bone(role)); }

    void set(const std::string &role, const Vec3 &p, float confidence, const char *source) {
        if (!has(role))
            return;
        auto &mark = at(role);
        mark.p = p;
        mark.confidence = confidence;
        mark.source = source;
    }

    // Re-centres an existing mark in its cross-section and scores the move.
    void center(const std::string &role, const Vec3 &dir, float radiusFraction,
                float sideSign = 0.f) {
        if (!has(role))
            return;
        auto &mark = at(role);
        const float radius = radiusFraction * H;
        Vec3 centred;
        size_t count = 0;
        if (!slabCenter(cloud, mark.p, dir, radius, .015f * H, centred, count, sideSign)) {
            mark.confidence = .2f;
            mark.source = "no_section";
            return;
        }
        const float moved = (centred - mark.p).length();
        mark.confidence = slabConfidence(count, moved, radius);
        mark.p = centred;
        mark.source = "centred";
    }

    const Cloud &cloud;
    std::unordered_map<std::string, Mark> &marks;
    const std::unordered_map<std::string, std::string> &roleBones;
    float H;
};

void placeHumanoid(Placer &placer, const std::unordered_map<std::string, Vec3> &templ,
                   const SceneData::ImportedModelContext &model,
                   const std::unordered_set<std::string> &handled, nlohmann::json &warnings,
                   std::string &forwardCheck) {
    const auto &cloud = placer.cloud;
    const float H = placer.H;
    const Vec3 up(0, 1, 0);

    // Facing check: the feet should extend from the lower legs toward +Z (fit axes).
    double footZ = 0, shinZ = 0;
    size_t footN = 0, shinN = 0;
    for (const auto &p : cloud.points) {
        const float y = p.y - cloud.lo.y;
        if (y < .04f * H) {
            footZ += p.z;
            ++footN;
        } else if (y > .12f * H && y < .3f * H) {
            shinZ += p.z;
            ++shinN;
        }
    }
    forwardCheck = "unknown";
    if (footN > 10 && shinN > 10) {
        const double delta = footZ / footN - shinZ / shinN;
        forwardCheck = delta > .005 * H ? "+Z" : (delta < -.005 * H ? "-Z" : "unknown");
    }

    if (placer.has("head.tip")) {
        Vec3 sum(0, 0, 0);
        size_t n = 0;
        for (const auto &p : cloud.points)
            if (p.y > cloud.hi.y - .02f * H) {
                sum += p;
                ++n;
            }
        if (n)
            placer.set("head.tip", Vec3(sum.x / n, cloud.hi.y, sum.z / n), n >= 3 ? .95f : .6f,
                       "detected");
    }
    if (placer.has("root")) {
        auto &root = placer.at("root");
        placer.set("root", Vec3(cloud.centerX, cloud.lo.y, root.p.z), .8f, "detected");
    }
    for (const char *role : {"head", "neck", "neck.upper", "spine.lower", "spine.middle",
                             "spine.chest", "spine.upper"})
        placer.center(role, up, .1f);
    placer.center("pelvis", up, .12f);

    for (int sideIndex = 0; sideIndex < 2; ++sideIndex) {
        const float sign = sideIndex == 0 ? 1.f : -1.f;
        const std::string side = sideIndex == 0 ? "left" : "right";
        const std::string leg = side + "_leg", arm = side + "_arm";

        // Feet: the lowest band of points on this side.
        float heelZ = std::numeric_limits<float>::infinity(), toeZ = -heelZ;
        double sumX = 0;
        size_t n = 0;
        for (const auto &p : cloud.points) {
            if (p.y - cloud.lo.y > .05f * H || (p.x - cloud.centerX) * sign < .01f * H)
                continue;
            heelZ = (std::min)(heelZ, p.z);
            toeZ = (std::max)(toeZ, p.z);
            sumX += p.x;
            ++n;
        }
        if (n >= 10) {
            const float footX = static_cast<float>(sumX / n);
            const float length = toeZ - heelZ;
            const float conf = length > .08f * H && length < .22f * H ? .9f : .4f;
            placer.set(leg + ".toe_tip", Vec3(footX, cloud.lo.y + .02f * H, toeZ - .005f * H),
                       conf, "detected");
            placer.set(leg + ".toe", Vec3(footX, cloud.lo.y + .02f * H, heelZ + .72f * length),
                       conf * .9f, "detected");
            placer.set(leg + ".ankle", Vec3(footX, cloud.lo.y + .045f * H, heelZ + .22f * length),
                       conf, "detected");
            placer.center(leg + ".ankle", up, .06f, sign);
        } else {
            warnings.push_back(side + " foot not found near the ground");
        }
        placer.center(leg + ".upper", up, .07f, sign);
        if (placer.has(leg + ".upper") && placer.has(leg + ".ankle") && placer.has(leg + ".lower")) {
            const auto hip = placer.at(leg + ".upper").p, ankle = placer.at(leg + ".ankle").p;
            placer.set(leg + ".lower", hip + (ankle - hip) * .489f, .5f, "estimated");
            placer.center(leg + ".lower", ankle - hip, .07f, sign);
        }

        // Hands: the most lateral point above the hips on this side, T or A pose alike.
        const Vec3 *tip = nullptr;
        float lateral = -std::numeric_limits<float>::infinity();
        for (const auto &p : cloud.points) {
            if (p.y - cloud.lo.y < .35f * H)
                continue;
            const float value = (p.x - cloud.centerX) * sign;
            if (value > lateral) {
                lateral = value;
                tip = &p;
            }
        }
        if (!tip || !placer.has(arm + ".upper")) {
            warnings.push_back(side + " hand not found");
            continue;
        }
        Vec3 sum(0, 0, 0);
        size_t near = 0;
        for (const auto &p : cloud.points)
            if ((p - *tip).length_squared() < (.03f * H) * (.03f * H)) {
                sum += p;
                ++near;
            }
        const Vec3 handTip = sum * (1.f / float((std::max)(near, size_t(1))));
        const float armConf = lateral > .25f * H ? .9f : .35f;
        if (lateral <= .25f * H)
            warnings.push_back(side + " arm is close to the body; hand tip guessed");
        placer.set(arm + ".hand_tip", handTip, armConf, "detected");
        const Vec3 shoulderTemplate = placer.at(arm + ".upper").p;
        placer.center(arm + ".upper", handTip - shoulderTemplate, .07f, sign);
        const Vec3 shoulder = placer.at(arm + ".upper").p;
        const Vec3 armDir = handTip - shoulder;
        placer.set(arm + ".lower", shoulder + armDir * .4416f, armConf * .8f, "estimated");
        placer.center(arm + ".lower", armDir, .06f, sign);
        placer.set(arm + ".hand", shoulder + armDir * .805f, armConf * .8f, "estimated");
        placer.center(arm + ".hand", armDir, .05f, sign);

        // Fingers and other hand descendants: carry the template hand into the detected arm
        // direction and length. Marked just under the threshold so the hands get a look.
        const auto handBone = placer.bone(arm + ".hand"), tipBone = placer.bone(arm + ".hand_tip"),
                   upperBone = placer.bone(arm + ".upper");
        if (handBone.empty() || tipBone.empty() || upperBone.empty() || !templ.count(handBone) ||
            !templ.count(tipBone) || !templ.count(upperBone))
            continue;
        const Vec3 templArm = templ.at(tipBone) - templ.at(upperBone);
        if (templArm.length_squared() < 1e-12f || armDir.length_squared() < 1e-12f)
            continue;
        auto rotation = Quaternion::rotationBetween(templArm.normalize(), armDir.normalize());
        rotation.normalize();
        const float scale = armDir.length() / templArm.length();
        const Vec3 hand = placer.at(arm + ".hand").p;
        std::vector<int> stack;
        const int handIndex =
            static_cast<int>(model.nodeHierarchy.find(handBone) - model.nodeHierarchy.nodes.data());
        for (int child : model.nodeHierarchy.nodes[static_cast<size_t>(handIndex)].children)
            stack.push_back(child);
        while (!stack.empty()) {
            const int index = stack.back();
            stack.pop_back();
            const auto &node = model.nodeHierarchy.nodes[static_cast<size_t>(index)];
            for (int child : node.children)
                stack.push_back(child);
            if (handled.count(node.uniqueName) || node.uniqueName == tipBone ||
                !placer.marks.count(node.uniqueName) || !templ.count(node.uniqueName))
                continue;
            auto &mark = placer.marks.at(node.uniqueName);
            mark.p = hand + rotation.rotate(templ.at(node.uniqueName) - templ.at(handBone)) * scale;
            mark.confidence = .45f;
            mark.source = "estimated";
        }
    }
}

void placeGeneric(Placer &placer, const SceneData::ImportedModelContext &model) {
    for (size_t i = 0; i < model.nodeHierarchy.size(); ++i) {
        const auto &node = model.nodeHierarchy.nodes[i];
        if (node.parent < 0 || !placer.marks.count(node.uniqueName))
            continue;
        const auto &parent = model.nodeHierarchy.nodes[static_cast<size_t>(node.parent)].uniqueName;
        if (!placer.marks.count(parent))
            continue;
        auto &mark = placer.marks.at(node.uniqueName);
        const auto dir = mark.p - placer.marks.at(parent).p;
        if (dir.length_squared() < 1e-12f)
            continue;
        const float radius = .08f * placer.H;
        Vec3 centred;
        size_t count = 0;
        if (!slabCenter(placer.cloud, mark.p, dir, radius, .015f * placer.H, centred, count)) {
            mark.confidence = .2f;
            mark.source = "no_section";
            continue;
        }
        mark.confidence = slabConfidence(count, (centred - mark.p).length(), radius);
        mark.p = centred;
        mark.source = "centred";
    }
}
} // namespace

bool autoLandmarks(const SceneData &scene, const std::string &character, const std::string &mesh,
                   nlohmann::json &output, std::string &error) {
    output = nullptr;
    nlohmann::json setup;
    if (!fitSetup(scene, character, mesh, setup, error))
        return false;
    const SceneData::ImportedModelContext *model = nullptr;
    for (const auto &candidate : scene.importedModelContexts)
        if (candidate.importName == character)
            model = &candidate;
    if (!model) {
        error = "unknown_character";
        return false;
    }
    Cloud cloud;
    if (!buildCloud(scene, mesh, cloud, error))
        return false;

    std::unordered_map<std::string, Mark> marks;
    std::unordered_map<std::string, Vec3> templ;
    for (const auto &item : setup["landmarks"].items()) {
        Vec3 p;
        if (!readVec(item.value(), p)) {
            error = "rig_fit_invalid_landmark";
            return false;
        }
        marks[item.key()].p = p;
        templ[item.key()] = p;
    }
    std::unordered_map<std::string, std::string> roleBones, boneRoles;
    for (const auto &role : model->rigAnatomy.roles) {
        roleBones[role.role] = role.bone;
        boneRoles[role.bone] = role.role;
    }
    std::unordered_set<std::string> derived;
    for (const auto &rule : model->rigAnatomy.fitRules)
        derived.insert(rule.bone);

    Placer placer(cloud, marks, roleBones);
    auto warnings = nlohmann::json::array();
    std::string forwardCheck = "not_checked";
    const bool humanoid = model->rigAnatomy.family == "humanoid" && placer.has("pelvis") &&
                          placer.has("left_leg.ankle") && placer.has("right_leg.ankle");
    if (humanoid) {
        std::unordered_set<std::string> handled;
        for (const auto &role : model->rigAnatomy.roles)
            if (role.role.find("_hand.") == std::string::npos)
                handled.insert(role.bone);
        placeHumanoid(placer, templ, *model, handled, warnings, forwardCheck);
        if (forwardCheck == "-Z") {
            warnings.push_back("The mesh appears to face -Z; fit axes assume +Z forward. Rotate "
                               "the mesh or rig, then run again.");
            for (auto &entry : marks)
                entry.second.confidence = (std::min)(entry.second.confidence, .3f);
        }
    } else {
        placeGeneric(placer, *model);
    }

    // Derived joints follow their fit rules, exactly as previewFit resolves them.
    for (const auto &rule : model->rigAnatomy.fitRules) {
        if (!marks.count(rule.start) || !marks.count(rule.end) || !marks.count(rule.bone))
            continue;
        const auto &a = marks.at(rule.start), &b = marks.at(rule.end);
        auto &mark = marks.at(rule.bone);
        mark.p = a.p + (b.p - a.p) * rule.position;
        mark.confidence = (std::min)(a.confidence, b.confidence);
        mark.source = "fit_rule";
    }

    auto landmarks = nlohmann::json::object();
    auto joints = nlohmann::json::array();
    auto low = nlohmann::json::array();
    float total = 0;
    for (const auto &node : model->nodeHierarchy.nodes) {
        const auto found = marks.find(node.uniqueName);
        if (found == marks.end())
            continue;
        const auto &mark = found->second;
        landmarks[node.uniqueName] = {mark.p.x, mark.p.y, mark.p.z};
        const auto role = boneRoles.find(node.uniqueName);
        const auto moved = (mark.p - templ[node.uniqueName]).length();
        joints.push_back({{"name", node.uniqueName},
                          {"role", role == boneRoles.end() ? "" : role->second},
                          {"confidence", mark.confidence},
                          {"source", mark.source},
                          {"moved_from_template", moved},
                          {"fit_editable", !derived.count(node.uniqueName)}});
        total += mark.confidence;
        if (mark.confidence < kLowConfidence && !derived.count(node.uniqueName))
            low.push_back(node.uniqueName);
    }
    output = {{"character", character},
              {"mesh", mesh},
              {"rig_revision", model->rigRevision},
              {"method", humanoid ? "humanoid_sections_v1" : "generic_sections_v1"},
              {"forward_check", forwardCheck},
              {"landmarks", std::move(landmarks)},
              {"joints", std::move(joints)},
              {"low_confidence", std::move(low)},
              {"low_confidence_threshold", kLowConfidence},
              {"mean_confidence", marks.empty() ? 0.f : total / float(marks.size())},
              {"warnings", std::move(warnings)},
              {"axes", "+Y up, +Z forward, +X character left; same as get_fit_setup"}};
    return true;
}
} // namespace RigAuthoring
