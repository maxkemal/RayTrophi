#include "Animation/RigPosePreview.h"
#include <cmath>
#include <functional>
#include <limits>
#include <set>
namespace RigAuthoring {
bool sampleRigPose(const RayTrophi::NodeHierarchy& h, const AnimationData* clip,
                   double timeSeconds, std::vector<PreviewJoint>& output, std::string& error) {
    output.clear(); error.clear();
    if (!std::isfinite(timeSeconds) || timeSeconds < 0) { error="invalid_preview_time"; return false; }
    if (clip && (!std::isfinite(clip->duration) || !std::isfinite(clip->ticksPerSecond) || clip->duration<=0 || clip->ticksPerSecond<=0)) {
        error="invalid_clip_timing"; return false;
    }
    // Existing sampler loops in seconds. Reduce before sampling to avoid tick overflow.
    const double period = clip ? clip->duration/clip->ticksPerSecond : 1;
    if (!std::isfinite(period) || period<=0 || period>std::numeric_limits<float>::max()) {
        error="invalid_clip_timing"; return false;
    }
    const double time = clip ? std::fmod(timeSeconds, clip->duration/clip->ticksPerSecond) : 0;
    std::set<std::string> names;
    for (const auto& node : h.nodes) if (node.uniqueName.empty() || !names.insert(node.uniqueName).second) {
        error="invalid_preview_hierarchy"; return false;
    }
    std::vector<PreviewJoint> staged(h.size()); std::vector<int> state(h.size(),0);
    std::function<bool(size_t)> visit = [&](size_t i) {
        if (state[i]==2) return true;
        if (state[i]==1) { error="invalid_preview_hierarchy"; return false; }
        state[i]=1; const auto& node=h.nodes[i]; auto& joint=staged[i]; joint.name=node.uniqueName;
        joint.world=clip ? clip->calculateAnimationTransform(*clip,static_cast<float>(time),node.uniqueName,node.localBind) : node.localBind;
        if (node.parent>=0) {
            const auto p=static_cast<size_t>(node.parent);
            if (p>=h.size()) { error="invalid_preview_hierarchy"; return false; }
            if (!visit(p)) return false;
            joint.parent=h.nodes[p].uniqueName; joint.world=staged[p].world*joint.world;
        } else if (node.parent!=-1) { error="invalid_preview_hierarchy"; return false; }
        for (int r=0;r<4;++r) for (int c=0;c<4;++c)
            if (!std::isfinite(joint.world.m[r][c])) { error="invalid_preview_pose"; return false; }
        state[i]=2; return true;
    };
    for (size_t i=0;i<h.size();++i) if (!visit(i)) return false;
    output=std::move(staged); return true;
}
}
