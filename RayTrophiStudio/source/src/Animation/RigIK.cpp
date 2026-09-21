#include "Animation/RigIK.h"
#include "Animation/RigAimIK.h"
#include "Animation/RigChainIK.h"
#include "Animation/RigSplineIK.h"
#include "Animation/RigPosePreview.h"
#include "Animation/RigBindMath.h"
#include "Animation/AnimationKeys.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>
namespace RigAuthoring {
namespace {
bool finite(const Vec3& p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
Vec3 origin(const Matrix4x4& m) {
    return Vec3(m.m[0][3], m.m[1][3], m.m[2][3]);
}
size_t index(const RayTrophi::NodeHierarchy& h, const std::string& key) {
    return static_cast<size_t>(h.find(key) - h.nodes.data());
}
Vec3 perpendicular(const Vec3& axis) {
    const auto v = std::fabs(axis.x) < .7f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    return (v - axis * Vec3::dot(axis, v)).normalize();
}
bool rigid(const Matrix4x4& m) {
    Vec3 p, s;
    Quaternion q;
    RayTrophi::decomposeTRS(m, p, q, s);
    q.normalize();
    const auto rebuilt = Matrix4x4::translation(p) * q.toMatrix();
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            if (!std::isfinite(m.m[r][c]) || !std::isfinite(rebuilt.m[r][c]) ||
                std::fabs(m.m[r][c] - rebuilt.m[r][c]) > 1e-4f)
                return false;
    return true;
}
Matrix4x4 rotationLocal(const Matrix4x4& old, const Quaternion& rotation) {
    auto q = rotation;
    q.normalize();
    auto m = q.toMatrix();
    for (int r = 0; r < 3; ++r)
        m.m[r][3] = old.m[r][3];
    return m;
}
Quaternion rotation(const Matrix4x4& m) {
    Vec3 t, s;
    Quaternion q;
    RayTrophi::decomposeTRS(m, t, q, s);
    q.normalize();
    return q;
}
bool validPose(const IKPose& p) {
    const auto& q = p.orientationWorld;
    const float norm = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    return validateSplineIKPose(p) && std::isfinite(norm) && std::fabs(norm - 1.f) < 1e-3f &&
           finite(p.target) && finite(p.pole) && std::isfinite(p.target.length_squared()) &&
           std::isfinite(p.pole.length_squared()) && std::isfinite(p.blend) && p.blend >= 0 &&
           p.blend <= 1;
}
}
bool hasEnabledIK(const IKPoses& poses) {
    for (const auto& p : poses)
        if (p.second.enabled && p.second.blend > 0)
            return true;
    return false;
}
std::vector<std::string> ikControlBones(const IKControl& control) {
    if (control.solver == "aim") {
        return {control.root};
    }
    return control.chain.empty() ? std::vector<std::string>{control.root, control.mid, control.tip}
                                 : control.chain;
}
bool ikDrivesBone(const std::vector<IKControl>& controls, const IKPoses& poses,
                  const std::string& bone) {
    for (const auto& control : controls) {
        const auto found = poses.find(control.name);
        if (found == poses.end() || !found->second.enabled || found->second.blend <= 0) {
            continue;
        }
        const auto bones = ikControlBones(control);
        if (std::find(bones.begin(), bones.end(), bone) != bones.end()) {
            return true;
        }
    }
    return false;
}
bool sameIKPoses(const IKPoses& a, const IKPoses& b) {
    if (a.size() != b.size())
        return false;
    for (const auto& p : a) {
        const auto f = b.find(p.first);
        if (f == b.end())
            return false;
        const auto& x = p.second;
        const auto& y = f->second;
        if (x.splineEnabled != y.splineEnabled || x.splineWorld.size() != y.splineWorld.size()) {
            return false;
        }
        for (size_t i = 0; i < x.splineWorld.size(); ++i) {
            const auto& a = x.splineWorld[i];
            const auto& b = y.splineWorld[i];
            if (a.x != b.x || a.y != b.y || a.z != b.z) {
                return false;
            }
        }
        if (x.orientationEnabled != y.orientationEnabled ||
            x.orientationWorld.w != y.orientationWorld.w ||
            x.orientationWorld.x != y.orientationWorld.x ||
            x.orientationWorld.y != y.orientationWorld.y ||
            x.orientationWorld.z != y.orientationWorld.z || x.enabled != y.enabled ||
            x.contact != y.contact || x.blend != y.blend || x.target.x != y.target.x ||
            x.target.y != y.target.y || x.target.z != y.target.z || x.pole.x != y.pole.x ||
            x.pole.y != y.pole.y || x.pole.z != y.pole.z)
            return false;
    }
    return true;
}
bool validateIKControls(const std::vector<IKControl>& controls, const RayTrophi::NodeHierarchy& h,
                        std::string& error) {
    if (controls.empty()) {
        error.clear();
        return true;
    }
    error.clear();
    if (controls.size() > 256 || h.size() > 4096) {
        error = "rig_ik_limit";
        return false;
    }
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(h, nullptr, 0, joints, error))
        return false;
    std::unordered_set<std::string> names, driven;
    for (const auto& c : controls) {
        if (c.name.empty() || c.name.size() > 128) {
            error = "rig_ik_invalid_name";
            return false;
        }
        for (unsigned char v : c.name)
            if (!((v >= 'a' && v <= 'z') || (v >= 'A' && v <= 'Z') || (v >= '0' && v <= '9') ||
                  v == '_' || v == '-' || v == '.')) {
                error = "rig_ik_invalid_name";
                return false;
            }
        if (!names.insert(c.name).second) {
            error = "rig_ik_duplicate_control";
            return false;
        }
        if (c.solver != "two_bone" && c.solver != "aim") {
            error = "rig_ik_invalid_solver";
            return false;
        }
        if (c.solver == "aim") {
            const float aimLength = c.aimAxis.length_squared();
            const float upLength = c.upAxis.length_squared();
            if (!c.chain.empty() || c.root != c.mid || c.root != c.tip || !finite(c.aimAxis) ||
                !finite(c.upAxis) || std::fabs(aimLength - 1.f) > 1e-3f ||
                std::fabs(upLength - 1.f) > 1e-3f ||
                std::fabs(Vec3::dot(c.aimAxis, c.upAxis)) > 1e-3f) {
                error = "rig_ik_invalid_aim_axes";
                return false;
            }
            const auto* node = h.find(c.root);
            if (!node) {
                error = "unknown_bone";
                return false;
            }
            if (!rigid(node->localBind)) {
                error = "rig_ik_requires_rigid_pose";
                return false;
            }
            if (!driven.insert(c.root).second) {
                error = "rig_ik_overlapping_controls";
                return false;
            }
            continue;
        }
        const auto bones = ikControlBones(c);
        if ((!c.chain.empty() && (bones.size() < 4 || bones.size() > 64)) ||
            bones.front() != c.root || bones[1] != c.mid || bones.back() != c.tip) {
            error = "rig_ik_invalid_chain";
            return false;
        }
        std::unordered_set<std::string> chainBones;
        for (size_t i = 0; i < bones.size(); ++i) {
            const auto* node = h.find(bones[i]);
            if (!node) {
                error = "unknown_bone";
                return false;
            }
            if (!chainBones.insert(bones[i]).second ||
                (i > 0 && node->parent != static_cast<int>(index(h, bones[i - 1])))) {
                error = "rig_ik_invalid_chain";
                return false;
            }
            if (!rigid(node->localBind)) {
                error = "rig_ik_requires_rigid_pose";
                return false;
            }
            if (i > 0 && origin(node->localBind).length_squared() < 1e-12f) {
                error = "rig_ik_zero_length";
                return false;
            }
            if (!driven.insert(bones[i]).second) {
                error = "rig_ik_overlapping_controls";
                return false;
            }
        }
    }
    return true;
}
nlohmann::json serializeIKControls(const std::vector<IKControl>& controls) {
    auto rows = nlohmann::json::array();
    for (const auto& c : controls) {
        nlohmann::json row = {{"name", c.name}, {"root", c.root}, {"mid", c.mid}, {"tip", c.tip}};
        if (c.solver == "aim") {
            row["solver"] = "aim";
            row["aim_axis"] = {c.aimAxis.x, c.aimAxis.y, c.aimAxis.z};
            row["up_axis"] = {c.upAxis.x, c.upAxis.y, c.upAxis.z};
        }
        if (!c.chain.empty()) {
            row["chain"] = c.chain;
        }
        rows.push_back(std::move(row));
    }
    return rows;
}
bool deserializeIKControls(const nlohmann::json& value, const RayTrophi::NodeHierarchy& h,
                           std::vector<IKControl>& output, std::string& error) {
    error.clear();
    try {
        if (!value.is_array()) {
            error = "rig_ik_invalid_controls";
            return false;
        }
        if (value.size() > 256) {
            error = "rig_ik_limit";
            return false;
        }
        std::vector<IKControl> staged;
        for (const auto& row : value) {
            const bool aim = row.is_object() && row.value("solver", std::string()) == "aim";
            if (!row.is_object() ||
                (!aim && row.size() != 4 && !(row.size() == 5 && row.contains("chain"))) ||
                (aim && row.size() != 7)) {
                error = "rig_ik_invalid_controls";
                return false;
            }
            for (const auto* key : {"name", "root", "mid", "tip"})
                if (!row.contains(key) || !row[key].is_string()) {
                    error = "rig_ik_invalid_controls";
                    return false;
                }
            staged.push_back({row["name"].get<std::string>(), row["root"].get<std::string>(),
                              row["mid"].get<std::string>(), row["tip"].get<std::string>()});
            if (aim) {
                if (!row.contains("aim_axis") || !row.contains("up_axis") ||
                    !row["aim_axis"].is_array() || row["aim_axis"].size() != 3 ||
                    !row["up_axis"].is_array() || row["up_axis"].size() != 3) {
                    error = "rig_ik_invalid_controls";
                    return false;
                }
                for (const auto* key : {"aim_axis", "up_axis"}) {
                    for (const auto& component : row[key]) {
                        if (!component.is_number()) {
                            error = "rig_ik_invalid_controls";
                            return false;
                        }
                    }
                }
                staged.back().solver = "aim";
                staged.back().aimAxis =
                    Vec3(row["aim_axis"][0].get<float>(), row["aim_axis"][1].get<float>(),
                         row["aim_axis"][2].get<float>());
                staged.back().upAxis =
                    Vec3(row["up_axis"][0].get<float>(), row["up_axis"][1].get<float>(),
                         row["up_axis"][2].get<float>());
            }
            if (row.contains("chain")) {
                if (!row["chain"].is_array() || row["chain"].size() < 4 ||
                    row["chain"].size() > 64) {
                    error = "rig_ik_invalid_chain";
                    return false;
                }
                for (const auto& bone : row["chain"]) {
                    if (!bone.is_string()) {
                        error = "rig_ik_invalid_chain";
                        return false;
                    }
                    staged.back().chain.push_back(bone.get<std::string>());
                }
            }
        }
        if (!validateIKControls(staged, h, error))
            return false;
        output = std::move(staged);
        return true;
    } catch (const nlohmann::json::exception&) {
        error = "rig_ik_invalid_controls";
        return false;
    }
}
bool matchIKPose(const RayTrophi::NodeHierarchy& h, const IKControl& c, const Matrix4x4& placement,
                 IKPose& result, std::string& error) {
    if (!validateIKControls({c}, h, error))
        return false;
    if (c.solver == "aim") {
        return matchAimIKPose(h, c, placement, result, error);
    }
    Matrix4x4 inverse;
    if (!bindAffineInverse(placement, inverse)) {
        error = "rig_ik_invalid_placement";
        return false;
    }
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(h, nullptr, 0, joints, error))
        return false;
    const auto a = origin(joints[index(h, c.root)].world),
               b = origin(joints[index(h, c.mid)].world), t = origin(joints[index(h, c.tip)].world);
    auto axis = t - a;
    if (axis.length_squared() < 1e-12f)
        axis = b - a;
    axis = axis.normalize();
    auto bend = b - a - axis * Vec3::dot(b - a, axis);
    float scale = 0;
    const auto bones = ikControlBones(c);
    for (size_t i = 1; i < bones.size(); ++i) {
        scale += (origin(joints[index(h, bones[i])].world) -
                  origin(joints[index(h, bones[i - 1])].world))
                     .length();
    }
    if (bend.length_squared() < scale * scale * 1e-10f)
        bend = perpendicular(axis) * scale;
    auto staged = result;
    staged.orientationWorld = rotation(placement) * rotation(joints[index(h, c.tip)].world);
    staged.orientationWorld.normalize();
    staged.target = placement.transform_point(t);
    if (!c.chain.empty() && !staged.splineEnabled) {
        const auto first = (b - a).normalize();
        const auto last = (t - origin(joints[index(h, bones[bones.size() - 2])].world)).normalize();
        staged.splineWorld = {placement.transform_point(a + first * (scale / 3)),
                              placement.transform_point(t - last * (scale / 3))};
    }
    staged.pole = placement.transform_point(b + bend.normalize() * scale);
    if (!validPose(staged)) {
        error = "rig_ik_invalid_target";
        return false;
    }
    result = staged;
    return true;
}
bool solveIKPose(const RayTrophi::NodeHierarchy& input, const std::vector<IKControl>& controls,
                 const IKPoses& poses, const Matrix4x4& placement, RayTrophi::NodeHierarchy& output,
                 std::string& error) {
    if (!validateIKControls(controls, input, error))
        return false;
    for (const auto& p : poses) {
        if (!validPose(p.second)) {
            error = "rig_ik_invalid_target";
            return false;
        }
        if (std::none_of(controls.begin(), controls.end(),
                         [&](const auto& c) { return c.name == p.first; })) {
            error = "rig_ik_unknown_control";
            return false;
        }
    }
    Matrix4x4 inverse;
    if (!bindAffineInverse(placement, inverse)) {
        error = "rig_ik_invalid_placement";
        return false;
    }
    auto ordered = controls;
    auto depth = [&](const IKControl& c) {
        int n = 0, p = static_cast<int>(index(input, c.root));
        while (p >= 0) {
            ++n;
            p = input.nodes[static_cast<size_t>(p)].parent;
        }
        return n;
    };
    std::stable_sort(ordered.begin(), ordered.end(),
                     [&](const auto& a, const auto& b) { return depth(a) < depth(b); });
    auto staged = input;
    std::vector<PreviewJoint> joints;
    for (const auto& c : ordered) {
        const auto found = poses.find(c.name);
        if (found == poses.end() || !found->second.enabled || found->second.blend == 0)
            continue;
        const auto& state = found->second;
        if (c.solver == "aim") {
            if (state.contact || state.splineEnabled || state.orientationEnabled) {
                error = "rig_ik_aim_incompatible_pose";
                return false;
            }
            RayTrophi::NodeHierarchy solved;
            if (!solveAimIKPose(staged, c, state, placement, solved, error)) {
                return false;
            }
            staged = std::move(solved);
            continue;
        }
        if (state.splineEnabled && c.chain.empty()) {
            error = "rig_ik_spline_requires_chain";
            return false;
        }
        if (!c.chain.empty()) {
            RayTrophi::NodeHierarchy solved;
            if (!solveChainIK(staged, c, state, placement, solved, error)) {
                return false;
            }
            staged = std::move(solved);
            continue;
        }
        if (!sampleRigPose(staged, nullptr, 0, joints, error))
            return false;
        const auto ri = index(staged, c.root), mi = index(staged, c.mid), ti = index(staged, c.tip);
        const auto oldRoot = staged.nodes[ri].localBind, oldMid = staged.nodes[mi].localBind;
        const auto a = origin(joints[ri].world), b = origin(joints[mi].world),
                   t = origin(joints[ti].world);
        const double l1 = (b - a).length(), l2 = (t - b).length(), scale = l1 + l2;
        if (!std::isfinite(scale) || l1 < 1e-6 || l2 < 1e-6) {
            error = "rig_ik_zero_length";
            return false;
        }
        const auto target = inverse.transform_point(state.target),
                   pole = inverse.transform_point(state.pole);
        if (!finite(target) || !finite(pole)) {
            error = "rig_ik_invalid_target";
            return false;
        }
        auto direction = target - a;
        double distance = direction.length();
        if (!std::isfinite(distance) || !std::isfinite((pole - a).length_squared())) {
            error = "rig_ik_invalid_target";
            return false;
        }
        if (distance < scale * 1e-7) {
            direction = t - a;
            if (direction.length_squared() < scale * scale * 1e-12)
                direction = b - a;
        }
        direction = direction.normalize();
        const double inner = (std::min)(scale, std::fabs(l1 - l2) + scale * 1e-6);
        const double d = (std::max)(inner, (std::min)(scale, distance));
        auto bend = pole - a - direction * Vec3::dot(pole - a, direction);
        if (bend.length_squared() < scale * scale * 1e-12)
            bend = b - a - direction * Vec3::dot(b - a, direction);
        if (bend.length_squared() < scale * scale * 1e-12)
            bend = perpendicular(direction);
        else
            bend = bend.normalize();
        const double x = (l1 * l1 - l2 * l2 + d * d) / (2 * d),
                     y = std::sqrt((std::max)(0.0, l1 * l1 - x * x));
        const auto desiredMid =
            a + direction * static_cast<float>(x) + bend * static_cast<float>(y);
        const auto desiredTip = a + direction * static_cast<float>(d);
        auto rootWorld = Quaternion::rotationBetween(b - a, desiredMid - a) *
                         Quaternion::fromMatrix(joints[ri].world);
        rootWorld.normalize();
        Quaternion parentRotation;
        const int parent = staged.nodes[ri].parent;
        if (parent >= 0)
            parentRotation = Quaternion::fromMatrix(joints[static_cast<size_t>(parent)].world);
        parentRotation.normalize();
        staged.nodes[ri].localBind = rotationLocal(oldRoot, parentRotation.conjugate() * rootWorld);
        if (!sampleRigPose(staged, nullptr, 0, joints, error))
            return false;
        const auto midPosition = origin(joints[mi].world), tipPosition = origin(joints[ti].world);
        auto midWorld =
            Quaternion::rotationBetween(tipPosition - midPosition, desiredTip - midPosition) *
            Quaternion::fromMatrix(joints[mi].world);
        midWorld.normalize();
        staged.nodes[mi].localBind =
            rotationLocal(oldMid, Quaternion::fromMatrix(joints[ri].world).conjugate() * midWorld);
        // Blend once in local quaternion space; translation and lengths never change.
        staged.nodes[ri].localBind = rotationLocal(
            oldRoot,
            Quaternion::slerp(Quaternion::fromMatrix(oldRoot),
                              Quaternion::fromMatrix(staged.nodes[ri].localBind), state.blend));
        staged.nodes[mi].localBind = rotationLocal(
            oldMid,
            Quaternion::slerp(Quaternion::fromMatrix(oldMid),
                              Quaternion::fromMatrix(staged.nodes[mi].localBind), state.blend));
        if (state.orientationEnabled) {
            if (!sampleRigPose(staged, nullptr, 0, joints, error))
                return false;
            const auto local = rotation(joints[mi].world).conjugate() *
                               rotation(placement).conjugate() * state.orientationWorld;
            const auto oldTip = staged.nodes[ti].localBind;
            staged.nodes[ti].localBind =
                rotationLocal(oldTip, Quaternion::slerp(rotation(oldTip), local, state.blend));
        }
    }
    if (!sampleRigPose(staged, nullptr, 0, joints, error))
        return false;
    output = std::move(staged);
    return true;
}
nlohmann::json inspectIKPose(const RayTrophi::NodeHierarchy& h,
                             const std::vector<IKControl>& controls, const IKPoses& poses,
                             const Matrix4x4& placement) {
    auto rows = nlohmann::json::array();
    std::vector<PreviewJoint> joints;
    std::string error;
    if (!sampleRigPose(h, nullptr, 0, joints, error))
        return rows;
    for (const auto& c : controls) {
        IKPose state;
        const auto f = poses.find(c.name);
        if (f != poses.end())
            state = f->second;
        else if (!matchIKPose(h, c, placement, state, error))
            continue;
        if (!c.chain.empty() && state.splineWorld.empty()) {
            auto matched = state;
            if (!matchIKPose(h, c, placement, matched, error)) {
                continue;
            }
            state.splineWorld = std::move(matched.splineWorld);
        }
        const auto tip = placement.transform_point(origin(joints[index(h, c.tip)].world));
        const auto actual = rotation(placement) * rotation(joints[index(h, c.tip)].world);
        const auto& q = state.orientationWorld;
        const float dot =
            std::fabs(actual.w * q.w + actual.x * q.x + actual.y * q.y + actual.z * q.z);
        float length = 0;
        float worldLength = 0;
        const auto bones = ikControlBones(c);
        for (size_t i = 1; i < bones.size(); ++i) {
            const Vec3 actorPoint = origin(joints[index(h, bones[i])].world);
            const Vec3 actorPrevious = origin(joints[index(h, bones[i - 1])].world);
            length += (actorPoint - actorPrevious).length();
            worldLength += (placement.transform_point(actorPoint) -
                            placement.transform_point(actorPrevious))
                               .length();
        }
        auto spline = nlohmann::json::array();
        for (const auto& point : state.splineWorld) {
            spline.push_back({point.x, point.y, point.z});
        }
        auto guide = nlohmann::json::array();
        if (state.splineEnabled) {
            const auto anchor = placement.transform_point(origin(joints[index(h, c.root)].world));
            for (const auto& point : splineIKWorldCurve(anchor, state, 33)) {
                guide.push_back({point.x, point.y, point.z});
            }
        }
        auto fkHandles = nlohmann::json::array();
        if (c.solver == "two_bone" && c.chain.empty()) {
            for (const auto& bone : {c.root, c.mid}) {
                const auto point = placement.transform_point(origin(joints[index(h, bone)].world));
                fkHandles.push_back({{"bone", bone}, {"world", {point.x, point.y, point.z}}});
            }
        }
        rows.push_back({{"spline_enabled", state.splineEnabled},
                        {"spline_world", spline},
                        {"spline_guide_world", guide},
                        {"fk_handles", fkHandles},
                        {"solver", c.solver == "aim"     ? "aim"
                                   : c.chain.empty()     ? "two_bone"
                                   : state.splineEnabled ? "spline_fabrik"
                                                         : "fabrik"},
                        {"aim_axis", {c.aimAxis.x, c.aimAxis.y, c.aimAxis.z}},
                        {"up_axis", {c.upAxis.x, c.upAxis.y, c.upAxis.z}},
                        {"bones", bones},
                        {"tip_orientation_world", {actual.w, actual.x, actual.y, actual.z}},
                        {"orientation_enabled", state.orientationEnabled},
                        {"orientation_world", {q.w, q.x, q.y, q.z}},
                        {"orientation_error_degrees",
                         2.f * std::acos((std::min)(1.f, dot)) * 180.f / 3.14159265359f},
                        {"name", c.name},
                        {"root", c.root},
                        {"mid", c.mid},
                        {"tip", c.tip},
                        {"enabled", state.enabled},
                        {"blend", state.blend},
                        {"contact", state.contact},
                        {"target_world", {state.target.x, state.target.y, state.target.z}},
                        {"pole_world", {state.pole.x, state.pole.y, state.pole.z}},
                        {"tip_world", {tip.x, tip.y, tip.z}},
                        {"target_error_world", (tip - state.target).length()},
                        {"aim_error_degrees",
                         c.solver == "aim" ? aimIKErrorDegrees(h, c, state, placement) : 0.f},
                        {"length_actor", length},
                        {"length_world", worldLength}});
    }
    return rows;
}
}
