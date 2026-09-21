#include "RtApiInternal.h"
#include "Api/RtApiClipBinding.h"
#include "Animation/ClipBinding.h"
#include "Animation/RigPosePreview.h"
#include <cmath>
#include "OzzRuntime.h"
#include "ProjectManager.h"
#include <algorithm>
#include <exception>
#include <utility>

namespace {
SceneData::ImportedModelContext* findModel(SceneData& scene, const std::string& name) {
    for (auto& ctx : scene.importedModelContexts) if (ctx.importName == name) return &ctx;
    return nullptr;
}
class BindClipCommand final : public SceneCommand {
    std::shared_ptr<AnimationData> clip;
    std::shared_ptr<AnimationController> beforeAnimator, afterAnimator;
    std::shared_ptr<OzzRuntime::AnimationSet> beforeOzz, afterOzz;
    bool beforeHasAnimation = false, beforeRestPose = false;
public:
    BindClipCommand(UIContext& ctx, std::shared_ptr<AnimationData> value) : clip(std::move(value)) {
        auto* model = findModel(ctx.scene, clip->modelName);
        beforeAnimator = model->animator; beforeOzz = model->ozzAnimationSet;
        beforeHasAnimation = model->hasAnimation; beforeRestPose = model->restPoseApplied;
        // Stage fallible runtime work before touching scene-owned data.
        afterAnimator = beforeAnimator ? std::make_shared<AnimationController>(*beforeAnimator) : std::make_shared<AnimationController>();
        std::vector<std::shared_ptr<AnimationData>> clips;
        for (const auto& c : ctx.scene.animationDataList) if (c && c->modelName == clip->modelName) clips.push_back(c);
        clips.push_back(clip); afterAnimator->registerClips(clips);
        afterOzz = OzzRuntime::buildStubAnimationSet(clip->modelName, ctx.scene.boneData, clips);
    }
    void execute(UIContext& ctx) override {
        auto* model = findModel(ctx.scene, clip->modelName); if (!model) return;
        ctx.scene.animationDataList.push_back(clip);
        model->animator = afterAnimator; model->ozzAnimationSet = afterOzz;
        model->hasAnimation = true; model->restPoseApplied = false; model->rigJointGlobals.clear();
        ProjectManager::getInstance().markModified();
    }
    void undo(UIContext& ctx) override {
        auto& clips = ctx.scene.animationDataList;
        clips.erase(std::remove(clips.begin(), clips.end(), clip), clips.end());
        auto* model = findModel(ctx.scene, clip->modelName);
        if (model) { model->animator = beforeAnimator; model->ozzAnimationSet = beforeOzz; model->hasAnimation = beforeHasAnimation; model->restPoseApplied = beforeRestPose; model->rigJointGlobals.clear(); }
        ProjectManager::getInstance().markModified();
    }
    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override { return "Bind animation clip: " + clip->name; }
};
rtapi::Result prepare(const std::string& source, const std::string& clipName, const std::string& target,
                      const std::string& requestedName, RigAuthoring::ClipBindingReport& report,
                      std::shared_ptr<AnimationData>& output, const std::map<std::string, std::string>& nodeMap, const std::string& mode, float translationScale) {
    report = {}; output.reset();
    if (!rtapi::g_ctx) return rtapi::Result::fail("api_not_bound");
    if (rtapi::renderJobActive()) return rtapi::Result::fail("scene_locked");
    auto& scene = rtapi::g_ctx->scene;
    const auto* from = findModel(scene, source); const auto* to = findModel(scene, target);
    if (!from || !to) return rtapi::Result::fail("unknown_character");
    if (source == target) return rtapi::Result::fail("source_equals_target");
    if (!to->hasSkeletonRepresentation) return rtapi::Result::fail("target_has_no_skeleton");
    std::shared_ptr<AnimationData> clip;
    for (const auto& candidate : scene.animationDataList) if (candidate && candidate->name == clipName && candidate->modelName == source) {
        if (clip) return rtapi::Result::fail("ambiguous_source_clip"); clip = candidate;
    }
    if (!clip) return rtapi::Result::fail("unknown_source_clip");
    auto exists = [&](const std::string& name) { for (const auto& c : scene.animationDataList) if (c && c->name == name) return true; return false; };
    std::string base = clipName;
    if (base.find(source + "_") == 0) base.erase(0, source.size() + 1);
    std::string name = requestedName.empty() ? target + "_" + base : requestedName;
    if (!requestedName.empty() && exists(name)) return rtapi::Result::fail("clip_name_conflict");
    const std::string initial = name;
    for (int suffix = 2; exists(name); ++suffix) name = initial + "_" + std::to_string(suffix);
    std::string error;
    return RigAuthoring::buildSameRigClip(*clip, from->nodeHierarchy, to->nodeHierarchy, target, name, report, output, error, nodeMap, mode, translationScale)
        ? rtapi::Result::success() : rtapi::Result::fail(error);
}
}
namespace rtapi {
Result previewClipBinding(const std::string& source, const std::string& clip, const std::string& target,
                          RigAuthoring::ClipBindingReport& report, const std::map<std::string, std::string>& nodeMap, const std::string& mode, float translationScale) {
    try { std::shared_ptr<AnimationData> staged; return prepare(source, clip, target, {}, report, staged, nodeMap, mode, translationScale); }
    catch (const std::exception&) { return Result::fail("clip_binding_failed"); }
}
Result sampleClipBinding(const std::string& source, const std::string& clipName, const std::string& target,
                         double timeSeconds, RigAuthoring::ClipPosePreview& preview,
                         const std::map<std::string, std::string>& nodeMap, const std::string& mode, float translationScale,
                         const std::string& sourcePoseView, const std::string& targetPoseView) {
    preview = {};
    if((sourcePoseView!="rest" && sourcePoseView!="animated") || (targetPoseView!="rest" && targetPoseView!="animated"))return Result::fail("invalid_rig_pose_view");
    if (!std::isfinite(timeSeconds) || timeSeconds < 0) return Result::fail("invalid_preview_time");
    try {
        std::shared_ptr<AnimationData> staged;
        auto result = prepare(source, clipName, target, {}, preview.binding, staged, nodeMap, mode, translationScale);
        if (!result.ok) return result;
        auto* from = findModel(g_ctx->scene, source); auto* to = findModel(g_ctx->scene, target);
        const AnimationData* clip = nullptr;
        for (const auto& c : g_ctx->scene.animationDataList) if (c && c->modelName==source && c->name==clipName) clip=c.get();
        if (!clip || !from || !to) return Result::fail("unknown_source_clip");
        preview.duration_seconds = clip->duration/clip->ticksPerSecond;
        preview.time_seconds = timeSeconds;
        preview.source_pose_source=sourcePoseView=="rest"?"rest":"source_clip";
        preview.target_pose_source = targetPoseView=="rest"?"rest":(staged ? "bound_clip" : "bind");
        std::string error;
        if (!RigAuthoring::sampleRigPose(from->nodeHierarchy, sourcePoseView=="rest"?nullptr:clip, timeSeconds, preview.source, error) ||
            !RigAuthoring::sampleRigPose(to->nodeHierarchy, targetPoseView=="rest"?nullptr:staged.get(), timeSeconds, preview.target, error)) {
            preview.source.clear(); preview.target.clear(); return Result::fail(error);
        }
        return Result::success();
    } catch (const std::exception&) { preview.source.clear(); preview.target.clear(); preview.binding.ready=false; return Result::fail("clip_preview_failed"); }
}
Result bindAnimationClip(const std::string& source, const std::string& clip, const std::string& target,
                         const std::string& name, RigAuthoring::ClipBindingReport& report, const std::map<std::string, std::string>& nodeMap, const std::string& mode, float translationScale) {
    try {
    std::shared_ptr<AnimationData> staged;
    auto result = prepare(source, clip, target, name, report, staged, nodeMap, mode, translationScale);
    if (!result.ok) return result;
    if (!report.ready || !staged) return Result::fail("incompatible_clip_binding");
    if (!g_history) return Result::fail("history_not_bound");
    auto command = std::make_unique<BindClipCommand>(*g_ctx, std::move(staged));
    command->execute(*g_ctx); g_history->record(std::move(command));
    return Result::success();
    } catch (const std::exception&) { return Result::fail("clip_binding_failed"); }
}
}
