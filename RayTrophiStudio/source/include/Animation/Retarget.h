#pragma once
#include "Animation/ClipBinding.h"

namespace RigAuthoring {
// Rest-frame delta transfer for matching parent chains. No IK/contact solving.
bool applyRestBasisRetarget(const AnimationData& sourceClip,
                           const RayTrophi::NodeHierarchy& source,
                           const RayTrophi::NodeHierarchy& target,
                           const ClipBindingReport&, float translationScale,
                           AnimationData& output, std::string& error);
}
