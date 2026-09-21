#include "Animation/Retarget.h"
#include <algorithm>
#include <cmath>
#include <functional>

namespace RigAuthoring {
namespace {
struct RestFrame { Vec3 position, scale; Quaternion local, global; float globalScale = 1; };
bool finite(const Vec3& v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
bool unit(Quaternion& q) {
    const double n = double(q.w)*q.w + double(q.x)*q.x + double(q.y)*q.y + double(q.z)*q.z;
    if (!std::isfinite(n) || n < 1e-12) return false;
    const float inv = static_cast<float>(1 / std::sqrt(n));
    q.w *= inv; q.x *= inv; q.y *= inv; q.z *= inv; return true;
}
bool uniform(const Vec3& s) {
    if (!finite(s) || s.x < 1e-6f || s.y < 1e-6f || s.z < 1e-6f) return false;
    return std::fabs(s.x-s.y) <= 1e-4f*s.x && std::fabs(s.x-s.z) <= 1e-4f*s.x;
}
bool frames(const RayTrophi::NodeHierarchy& h, std::vector<RestFrame>& out, std::string& error) {
    out.resize(h.size()); std::vector<int> state(h.size(), 0);
    std::function<bool(size_t)> visit = [&](size_t i) {
        if (state[i] == 2) return true;
        if (state[i] == 1) { error = "invalid_retarget_hierarchy"; return false; }
        state[i] = 1;
        const auto& node = h.nodes[i]; const auto& m = node.localBind; auto& r = out[i];
        for (int a=0; a<4; ++a) for (int b=0; b<4; ++b)
            if (!std::isfinite(m.m[a][b])) { error = "invalid_retarget_rest"; return false; }
        RayTrophi::decomposeTRS(m, r.position, r.local, r.scale);
        if (!uniform(r.scale) || !unit(r.local)) { error = "unsupported_retarget_rest"; return false; }
        // Decomposition alone silently loses reflection/shear. Require TRS reconstruction.
        const auto reconstructed = Matrix4x4::translation(r.position) * r.local.toMatrix() * Matrix4x4::scaling(r.scale);
        for (int a=0; a<4; ++a) for (int b=0; b<4; ++b)
            if (std::fabs(m.m[a][b]-reconstructed.m[a][b]) > 1e-4f * std::max(1.f, std::fabs(m.m[a][b]))) {
                error = "unsupported_retarget_rest"; return false;
            }
        r.global = r.local; r.globalScale = r.scale.x;
        if (node.parent >= 0) {
            const auto parent = static_cast<size_t>(node.parent);
            if (parent >= h.size()) { error = "invalid_retarget_hierarchy"; return false; }
            if (!visit(parent)) return false;
            r.global = out[parent].global * r.local; r.globalScale *= out[parent].globalScale;
        } else if (node.parent != -1) { error = "invalid_retarget_hierarchy"; return false; }
        if (!unit(r.global) || !std::isfinite(r.globalScale) || r.globalScale < 1e-8f) {
            error = "invalid_retarget_rest"; return false;
        }
        state[i] = 2; return true;
    };
    for (size_t i=0; i<h.size(); ++i) if (!visit(i)) return false;
    return true;
}
template<class Keys, class Validate> bool validKeys(const Keys& keys, double duration, Validate validate) {
    double previous = -1;
    for (const auto& key : keys) {
        if (!std::isfinite(key.time) || key.time < 0 || key.time > duration || key.time <= previous || !validate(key.value)) return false;
        previous = key.time;
    }
    return true;
}
}
bool applyRestBasisRetarget(const AnimationData& clip, const RayTrophi::NodeHierarchy& source,
                           const RayTrophi::NodeHierarchy& target, const ClipBindingReport& report,
                           float translationScale, AnimationData& output, std::string& error) {
    if (!std::isfinite(translationScale) || translationScale <= 0 || translationScale > 10000) {
        error = "invalid_translation_scale"; return false;
    }
    std::vector<RestFrame> from, to;
    if (!frames(source, from, error) || !frames(target, to, error)) return false;
    for (const auto& match : report.matches) {
        const auto* sn = source.find(match.source); const auto* tn = target.find(match.target);
        if (!sn || !tn) { error = "invalid_retarget_mapping"; return false; }
        const auto& s = from[static_cast<size_t>(sn-source.nodes.data())];
        const auto& t = to[static_cast<size_t>(tn-target.nodes.data())];
        const Quaternion basis = s.global.conjugate() * t.global;
        const auto positions = clip.positionKeys.find(match.source);
        if (positions != clip.positionKeys.end() && !positions->second.empty()) {
            if (!validKeys(positions->second, clip.duration, [](const Vec3& v) { return finite(v); })) {
                error = "invalid_retarget_keys"; return false;
            }
            const Quaternion sp = sn->parent < 0 ? Quaternion() : from[sn->parent].global;
            const Quaternion tp = tn->parent < 0 ? Quaternion() : to[tn->parent].global;
            const float ss = sn->parent < 0 ? 1.f : from[sn->parent].globalScale;
            const float ts = tn->parent < 0 ? 1.f : to[tn->parent].globalScale;
            auto& keys = output.positionKeys[match.target]; keys = positions->second;
            for (auto& key : keys) {
                key.value = t.position + (tp.conjugate()*sp).rotate(key.value-s.position) * (translationScale*ss/ts);
                if (!finite(key.value)) { error = "invalid_retarget_keys"; return false; }
            }
        }
        const auto rotations = clip.rotationKeys.find(match.source);
        if (rotations != clip.rotationKeys.end() && !rotations->second.empty()) {
            if (!validKeys(rotations->second, clip.duration, [](Quaternion q) { return unit(q); })) {
                error = "invalid_retarget_keys"; return false;
            }
            auto& keys = output.rotationKeys[match.target]; keys = rotations->second;
            for (auto& key : keys) {
                unit(key.value);
                key.value = t.local * basis.conjugate() * s.local.conjugate() * key.value * basis;
                if (!unit(key.value)) { error = "invalid_retarget_keys"; return false; }
            }
        }
        const auto scales = clip.scalingKeys.find(match.source);
        if (scales != clip.scalingKeys.end() && !scales->second.empty()) {
            if (!validKeys(scales->second, clip.duration, [](const Vec3& v) { return uniform(v); })) {
                error = "unsupported_retarget_scale_keys"; return false;
            }
            auto& keys = output.scalingKeys[match.target]; keys = scales->second;
            for (auto& key : keys) {
                key.value = t.scale * (key.value.x/s.scale.x);
                if (!uniform(key.value)) { error = "unsupported_retarget_scale_keys"; return false; }
            }
        }
    }
    return true;
}
}
