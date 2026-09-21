#pragma once
#include <string>
#include <map>
namespace RigAuthoring { struct ClipBindingReport; struct ClipPosePreview; }
namespace rtapi {
struct Result;
Result previewClipBinding(const std::string& sourceCharacter, const std::string& sourceClip,
                          const std::string& targetCharacter, RigAuthoring::ClipBindingReport&,
                          const std::map<std::string, std::string>& nodeMap = {},
                          const std::string& mode = "same_rig", float translationScale = 1.f);
Result sampleClipBinding(const std::string& sourceCharacter, const std::string& sourceClip,
                         const std::string& targetCharacter, double timeSeconds,
                         RigAuthoring::ClipPosePreview&,
                         const std::map<std::string, std::string>& nodeMap = {},
                         const std::string& mode = "same_rig", float translationScale = 1.f,
                         const std::string& sourcePoseView = "animated", const std::string& targetPoseView = "animated");
Result bindAnimationClip(const std::string& sourceCharacter, const std::string& sourceClip,
                         const std::string& targetCharacter, const std::string& outputName,
                         RigAuthoring::ClipBindingReport&,
                          const std::map<std::string, std::string>& nodeMap = {},
                          const std::string& mode = "same_rig", float translationScale = 1.f);
}
