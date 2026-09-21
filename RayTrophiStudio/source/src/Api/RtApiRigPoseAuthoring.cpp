#include "RtApiInternal.h"
#include "Api/RtApiRigPoseAuthoring.h"
#include "Animation/RigPoseAuthoring.h"
#include "Animation/RigIKTimeline.h"
#include "Animation/RigSplineIK.h"
#include "Animation/RigBatchRest.h"
#include "Animation/RigMirror.h"
#include "Animation/RigView.h"
#include "Animation/RigWeights.h"
#include "Animation/RigBindingScope.h"
#include "Animation/RigClipRuntimeSync.h"
#include "Animation/RigBoneCurves.h"
#include "Animation/RigDrivenControls.h"
#include "TriangleMesh.h"
#include "ProjectManager.h"
#include "Backend/IBackend.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
namespace rtapi {
namespace {
using State = RigAuthoring::PoseAuthoringState;
void wake() {
    if (g_ctx) {
        g_ctx->scene.rigView.pose.invalidateEvaluation();
        g_ctx->start_render = true;
        g_ctx->renderer.resetCPUAccumulation();
        if (g_ctx->backend_ptr)
            g_ctx->backend_ptr->resetAccumulation();
    }
}
Result guard(const std::string& character) {
    if (!g_ctx)
        return Result::fail("api_not_bound");
    if (renderJobActive())
        return Result::fail("scene_locked");
    const auto& state = g_ctx->scene.rigView.pose;
    if (!state.active || state.character != character)
        return Result::fail("rig_pose_mode_required");
    std::string error;
    if (!RigAuthoring::canAuthorPose(g_ctx->scene, character, error))
        return Result::fail(error);
    return Result::success();
}
class PoseCommand final : public SceneCommand {
    State pending;
    std::vector<std::shared_ptr<AnimationData>> clips;
    bool persistent = false;
    void exchange(UIContext& ctx) {
        const bool active = ctx.scene.rigView.pose.active;
        std::string character;
        character.swap(ctx.scene.rigView.pose.character);
        std::swap(ctx.scene.rigView.pose, pending);
        ctx.scene.rigView.pose.active = active;
        ctx.scene.rigView.pose.character.swap(character);
        ctx.scene.rigView.pose.preview.clear();
        ctx.scene.rigView.pose.hasPreview = false;
        ctx.scene.rigView.pose.limitHits.clear();
        ctx.scene.rigView.pose.previewIK.clear();
        ctx.scene.animationDataList.swap(clips);
        RigAuthoring::synchronizeClipRuntime(
            ctx.scene, ctx.scene.rigView.pose.character);
        wake();
        ctx.renderer.animation_groups_dirty = true;
        if (persistent)
            ProjectManager::getInstance().markModified();
    }

  public:
    PoseCommand(State state, std::vector<std::shared_ptr<AnimationData>> data, bool persist)
        : pending(std::move(state)), clips(std::move(data)), persistent(persist) {
    }
    void execute(UIContext& ctx) override {
        exchange(ctx);
    }
    void undo(UIContext& ctx) override {
        exchange(ctx);
    }
    Type getType() const override {
        return Type::Generic;
    }
    bool isHeavyGeometry() const override {
        return false;
    }
    std::string getDescription() const override {
        return "Rig pose / bone keys";
    }
};
Result publish(State state, std::vector<std::shared_ptr<AnimationData>> clips) {
    if (!g_history)
        return Result::fail("history_not_bound");
    const bool persistent = clips != g_ctx->scene.animationDataList;
    ++state.serial;
    auto command = std::make_unique<PoseCommand>(std::move(state), std::move(clips), persistent);
    auto* raw = command.get();
    g_history->record(std::move(command));
    raw->execute(*g_ctx);
    return Result::success();
}
Result change(const std::string& character, const std::function<Result()>& body) {
    auto r = guard(character);
    if (!r.ok)
        return r;
    try {
        return body();
    } catch (const std::exception&) {
        return Result::fail("rig_pose_failed");
    }
}
size_t clipIndex(const State& state, const std::string& character) {
    const auto selected = state.clips.find(character);
    if (selected == state.clips.end())
        return g_ctx->scene.animationDataList.size();
    const auto& clips = g_ctx->scene.animationDataList;
    for (size_t i = 0; i < clips.size(); ++i)
        if (clips[i] && clips[i]->rigAuthoring && clips[i]->name == selected->second &&
            clips[i]->modelName == character)
            return i;
    return clips.size();
}
Result writeKeys(State state, const RayTrophi::NodeHierarchy& hierarchy,
                 const std::vector<std::string>& bones) {
    auto clips = g_ctx->scene.animationDataList;
    const auto index = clipIndex(state, state.character);
    if (index == clips.size())
        return Result::fail("rig_pose_clip_required");
    auto clip = std::make_shared<AnimationData>(*clips[index]);
    std::string error;
    if (!RigAuthoring::insertPoseKeys(*clip, hierarchy, bones,
                                      double(g_ctx->scene.timeline.current_frame) / state.fps,
                                      error))
        return Result::fail(error);
    clips[index] = std::move(clip);
    return publish(std::move(state), std::move(clips));
}
Result previewHierarchy(const std::string& character, const RayTrophi::NodeHierarchy& hierarchy,
                        uint64_t revision, const RigAuthoring::IKPoses* controls = nullptr) {
    const auto& scene = g_ctx->scene;
    State state = scene.rigView.pose;
    for (const auto& model : scene.importedModelContexts)
        if (model.importName == character && model.rigRevision != revision)
            return Result::fail("rig_edit_stale_revision");
    AnimationData validate;
    validate.rigAuthoring = true;
    validate.duration = 1;
    validate.ticksPerSecond = 24;
    std::vector<std::string> names;
    for (const auto& n : hierarchy.nodes)
        names.push_back(n.uniqueName);
    std::string error;
    if (!RigAuthoring::insertPoseKeys(validate, hierarchy, names, 0, error))
        return Result::fail(error);
    state.previewIK = controls ? *controls : state.ik;
    RigAuthoring::IKPoses evaluated;
    if (!RigAuthoring::effectiveIKControls(scene, character, false, evaluated, error))
        return Result::fail(error);
    for (const auto& p : state.previewIK)
        evaluated[p.first] = p.second;
    RayTrophi::NodeHierarchy constrained, solved = hierarchy;
    std::vector<std::string> hits;
    for (const auto& m : scene.importedModelContexts)
        if (m.importName == character) {
            Matrix4x4 placement;
            if (!RigAuthoring::rigScenePlacement(scene, character, placement, error))
                return Result::fail(error);
            if (RigAuthoring::hasEnabledIK(evaluated) &&
                !RigAuthoring::solveIKPose(hierarchy, m.rigAnatomy.controls, evaluated, placement,
                                           solved, error))
                return Result::fail(error);
            if (!RigAuthoring::constrainJointPose(m.nodeHierarchy, m.rigAnatomy.joints, solved,
                                                  constrained, hits, error))
                return Result::fail(error);
        }
    // Store FK input, not the IK result. Evaluation solves/blends exactly once.
    state.limitHits = std::move(hits);
    state.preview.clear();
    for (const auto& n : hierarchy.nodes)
        state.preview[n.uniqueName] = n.localBind;
    if (scene.rigView.pose.hasPreview && scene.rigView.pose.frame == scene.timeline.current_frame &&
        scene.rigView.pose.revision == revision) {
        bool same = state.preview.size() == scene.rigView.pose.preview.size() &&
                    state.limitHits == scene.rigView.pose.limitHits &&
                    RigAuthoring::sameIKPoses(state.previewIK, scene.rigView.pose.previewIK);
        for (const auto& p : state.preview) {
            const auto old = scene.rigView.pose.preview.find(p.first);
            if (old == scene.rigView.pose.preview.end()) {
                same = false;
                break;
            }
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    same = same && p.second.m[r][c] == old->second.m[r][c];
        }
        if (same)
            return Result::success(); // Stationary drag does not invalidate the viewport.
    }
    state.hasPreview = true;
    state.frame = scene.timeline.current_frame;
    state.revision = revision;
    if (state.frame != scene.rigView.pose.frame)
        state.locals.clear();
    g_ctx->scene.rigView.pose = std::move(state);
    ui.timeline.pausePlayback();
    wake();
    return Result::success();
}
Result fkSelection(const std::string& character, const std::vector<std::string>& bones) {
    const auto& scene = g_ctx->scene;
    RigAuthoring::IKPoses controls;
    std::string error;
    if (!RigAuthoring::effectiveIKControls(scene, character, true, controls, error))
        return Result::fail(error);
    for (const auto& model : scene.importedModelContexts)
        if (model.importName == character)
            for (const auto& bone : bones)
                if (RigAuthoring::ikDrivesBone(model.rigAnatomy.controls, controls, bone))
                    return Result::fail("rig_ik_bone_driven");
    return Result::success();
}
using IKEdit = std::function<Result(RigAuthoring::IKPose&, const RayTrophi::NodeHierarchy&,
                                    const RigAuthoring::IKControl&, const Matrix4x4&)>;
Result previewControl(const std::string& character, const std::string& control, uint64_t revision,
                      const IKEdit& edit) {
    return change(character, [&]() {
        const auto& scene = g_ctx->scene;
        const auto& session = scene.rigView.pose;
        for (const auto& model : scene.importedModelContexts)
            if (model.importName == character) {
                if (model.rigRevision != revision)
                    return Result::fail("rig_edit_stale_revision");
                const auto definition =
                    std::find_if(model.rigAnatomy.controls.begin(), model.rigAnatomy.controls.end(),
                                 [&](const auto& c) { return c.name == control; });
                if (definition == model.rigAnatomy.controls.end())
                    return Result::fail("rig_ik_unknown_control");
                RayTrophi::NodeHierarchy base, solved;
                std::string error;
                Matrix4x4 placement;
                if (!RigAuthoring::rigScenePlacement(scene, character, placement, error))
                    return Result::fail(error);
                if (!RigAuthoring::currentPoseHierarchy(scene, character, base, error, true,
                                                        nullptr, false) ||
                    !RigAuthoring::currentPoseHierarchy(scene, character, solved, error, true))
                    return Result::fail(error);
                auto controls = session.hasPreview ? session.previewIK : session.ik;
                if (!controls.count(control)) {
                    RigAuthoring::IKPoses evaluated;
                    if (!RigAuthoring::effectiveIKControls(scene, character, true, evaluated,
                                                           error))
                        return Result::fail(error);
                    const auto found = evaluated.find(control);
                    if (found != evaluated.end()) {
                        auto matched = found->second;
                        matched.contact = false;
                        controls[control] = matched;
                    }
                }
                auto& value = controls[control];
                if (!value.enabled &&
                    !RigAuthoring::matchIKPose(solved, *definition, placement, value, error))
                    return Result::fail(error);
                auto r = edit(value, solved, *definition, placement);
                if (!r.ok)
                    return r;
                if (!value.enabled) {
                    // Switching back to FK bakes the achieved rotations into the FK input.
                    for (const auto& key : RigAuthoring::ikControlBones(*definition))
                        for (auto& node : base.nodes)
                            if (node.uniqueName == key)
                                node.localBind = solved.find(key)->localBind;
                }
                return previewHierarchy(character, base, revision, &controls);
            }
        return Result::fail("unknown_character");
    });
}
}
Result setRigIKTarget(const std::string& character, const std::string& control, const Vec3& target,
                      const Vec3& pole, uint64_t revision) {
    return previewControl(character, control, revision,
                          [&](RigAuthoring::IKPose& state, const auto&, const auto&, const auto&) {
                              state.enabled = true;
                              if (state.blend == 0)
                                  state.blend = 1;
                              state.target = target;
                              state.pole = pole;
                              return Result::success();
                          });
}
Result setRigIKOrientation(const std::string& character, const std::string& control,
                           const Quaternion& orientationWorld, bool enabled, uint64_t revision) {
    const auto& q = orientationWorld;
    const float norm = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (!std::isfinite(norm) || std::fabs(norm - 1.f) > 1e-3f)
        return Result::fail("rig_ik_invalid_orientation");
    return previewControl(
        character, control, revision,
        [&](RigAuthoring::IKPose& state, const auto&, const auto& definition, const auto&) {
            if (definition.solver == "aim")
                return Result::fail("rig_ik_orientation_unsupported");
            state.orientationWorld = q;
            state.orientationWorld.normalize();
            state.orientationEnabled = enabled;
            if (enabled) {
                state.enabled = true;
                if (state.blend == 0)
                    state.blend = 1;
            }
            return Result::success();
        });
}
Result setRigIKSpline(const std::string& character, const std::string& control,
                      const std::vector<Vec3>& pointsWorld, bool enabled, uint64_t revision) {
    RigAuthoring::IKPose candidate;
    candidate.splineEnabled = enabled;
    candidate.splineWorld = pointsWorld;
    if (!RigAuthoring::validateSplineIKPose(candidate)) {
        return Result::fail("rig_ik_invalid_spline");
    }
    return previewControl(
        character, control, revision,
        [&](RigAuthoring::IKPose& state, const auto&, const auto& definition, const auto&) {
            if (definition.chain.empty()) {
                return Result::fail("rig_ik_spline_requires_chain");
            }
            state.splineEnabled = enabled;
            state.splineWorld = pointsWorld;
            if (enabled) {
                state.enabled = true;
                if (state.blend == 0) {
                    state.blend = 1;
                }
            }
            return Result::success();
        });
}
Result setRigIKFK(const std::string& character, const std::string& control, float blend,
                  uint64_t revision) {
    if (!std::isfinite(blend) || blend < 0 || blend > 1)
        return Result::fail("rig_ik_invalid_blend");
    return previewControl(character, control, revision,
                          [&](RigAuthoring::IKPose& state, const auto&, const auto&, const auto&) {
                              state.blend = blend;
                              state.enabled = blend > 0;
                              if (!state.enabled)
                                  state.contact = false;
                              return Result::success();
                          });
}
Result matchRigIKToFK(const std::string& character, const std::string& control,
                      uint64_t revision) {
    return previewControl(
        character, control, revision,
        [&](RigAuthoring::IKPose& state, const auto& solved, const auto& definition,
            const auto& placement) {
            if (definition.solver != "two_bone" || !definition.chain.empty()) {
                return Result::fail("rig_ik_match_requires_two_bone");
            }
            std::string error;
            auto matched = state;
            if (!RigAuthoring::matchIKPose(solved, definition, placement, matched, error)) {
                return Result::fail(error);
            }
            matched.enabled = true;
            matched.blend = 1.f;
            matched.contact = false;
            matched.orientationEnabled = true;
            state = std::move(matched);
            return Result::success();
        });
}
Result matchRigFKToIK(const std::string& character, const std::string& control,
                      uint64_t revision) {
    return previewControl(
        character, control, revision,
        [&](RigAuthoring::IKPose& state, const auto&, const auto& definition, const auto&) {
            if (definition.solver != "two_bone" || !definition.chain.empty()) {
                return Result::fail("rig_ik_match_requires_two_bone");
            }
            state.blend = 0.f;
            state.enabled = false;
            state.contact = false;
            return Result::success();
        });
}
Result setRigIKContact(const std::string& character, const std::string& control, bool enabled,
                       uint64_t revision) {
    return previewControl(
        character, control, revision,
        [&](RigAuthoring::IKPose& state, const auto& solved, const auto& definition,
            const auto& placement) {
            if (definition.solver == "aim")
                return Result::fail("rig_ik_contact_unsupported");
            if (enabled) {
                std::string error;
                if (!RigAuthoring::matchIKPose(solved, definition, placement, state, error))
                    return Result::fail(error);
                state.enabled = true;
                state.blend = 1;
            }
            state.contact = enabled;
            return Result::success();
        });
}
namespace {
using ChannelEdit =
    std::function<Result(AnimationData&, State&, const SceneData::ImportedModelContext&)>;
Result editIKChannels(const std::string& character, uint64_t revision, const ChannelEdit& edit) {
    return change(character, [&]() {
        auto state = g_ctx->scene.rigView.pose;
        if (state.hasPreview)
            return Result::fail("rig_pose_preview_active");
        auto clips = g_ctx->scene.animationDataList;
        const auto index = clipIndex(state, character);
        if (index == clips.size())
            return Result::fail("rig_pose_clip_required");
        for (const auto& model : g_ctx->scene.importedModelContexts)
            if (model.importName == character) {
                if (model.rigRevision != revision)
                    return Result::fail("rig_edit_stale_revision");
                auto clip = std::make_shared<AnimationData>(*clips[index]);
                if (!std::isfinite(clip->duration) || clip->duration <= 0 ||
                    !std::isfinite(clip->ticksPerSecond) || clip->ticksPerSecond <= 0 ||
                    !std::isfinite(state.fps) || state.fps <= 0)
                    return Result::fail("invalid_clip_timing");
                auto result = edit(*clip, state, model);
                if (!result.ok)
                    return result;
                std::string error;
                if (!RigAuthoring::validateIKChannelControls(clip->ikChannels,
                                                             model.rigAnatomy.controls, error))
                    return Result::fail(error);
                clips[index] = std::move(clip);
                return publish(std::move(state), std::move(clips));
            }
        return Result::fail("unknown_character");
    });
}
Result capturedIK(const std::string& character, const std::string& control,
                  const SceneData::ImportedModelContext& model, RigAuthoring::IKPose& value) {
    const auto definition =
        std::find_if(model.rigAnatomy.controls.begin(), model.rigAnatomy.controls.end(),
                     [&](const auto& c) { return c.name == control; });
    if (definition == model.rigAnatomy.controls.end())
        return Result::fail("rig_ik_unknown_control");
    std::string error;
    RigAuthoring::IKPoses effective;
    if (!RigAuthoring::effectiveIKControls(g_ctx->scene, character, false, effective, error))
        return Result::fail(error);
    const auto found = effective.find(control);
    if (found != effective.end())
        value = found->second;
    if (found == effective.end()) {
        RayTrophi::NodeHierarchy pose;
        Matrix4x4 placement;
        if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, pose, error) ||
            !RigAuthoring::rigScenePlacement(g_ctx->scene, character, placement, error) ||
            !RigAuthoring::matchIKPose(pose, *definition, placement, value, error))
            return Result::fail(error);
    }
    value.contact = false;
    return Result::success();
}
void extendIKClip(AnimationData& clip, double seconds) {
    clip.duration = (std::max)(clip.duration, seconds * clip.ticksPerSecond + 1);
    clip.endFrame =
        (std::max)(clip.endFrame, static_cast<int>(std::ceil(seconds * clip.ticksPerSecond)));
}
}
Result getRigIKChannels(const std::string& character, nlohmann::json& output) {
    output = nullptr;
    auto result = guard(character);
    if (!result.ok)
        return result;
    try {
        const auto* clip = RigAuthoring::authoredIKClip(g_ctx->scene, character);
        if (!clip)
            return Result::fail("rig_pose_clip_required");
        std::string error;
        for (const auto& model : g_ctx->scene.importedModelContexts)
            if (model.importName == character &&
                !RigAuthoring::validateIKChannelControls(clip->ikChannels,
                                                         model.rigAnatomy.controls, error))
                return Result::fail(error);
        output = RigAuthoring::serializeIKChannels(clip->ikChannels);
        output["clip"] = clip->name;
        output["fps"] = g_ctx->scene.rigView.pose.fps;
        output["coordinate_space"] = "world";
        return Result::success();
    } catch (const std::exception&) {
        return Result::fail("rig_ik_invalid_channel");
    }
}
Result insertRigIKKey(const std::string& character, const std::string& control, uint64_t revision) {
    return editIKChannels(
        character, revision, [&](AnimationData& clip, State& state, const auto& model) {
            RigAuthoring::IKPose value;
            auto result = capturedIK(character, control, model, value);
            if (!result.ok)
                return result;
            const double seconds = double(g_ctx->scene.timeline.current_frame) / state.fps;
            if (!std::isfinite(seconds) || seconds < 0 || seconds * clip.ticksPerSecond > 1000000)
                return Result::fail("rig_pose_time_limit");
            std::string error;
            if (!RigAuthoring::insertIKChannelKey(clip.ikChannels, control, seconds, value, error))
                return Result::fail(error);
            extendIKClip(clip, seconds);
            state.ik.erase(control);
            return Result::success();
        });
}
Result removeRigIKKey(const std::string& character, const std::string& control,
                      uint64_t revision) {
    return editIKChannels(
        character, revision, [&](AnimationData& clip, State& state, const auto&) {
            const double seconds = double(g_ctx->scene.timeline.current_frame) / state.fps;
            std::string error;
            if (!RigAuthoring::removeIKChannelKey(clip.ikChannels, control, seconds, error))
                return Result::fail(error);
            state.ik.erase(control);
            return Result::success();
        });
}
Result setRigIKContactInterval(const std::string& character, const std::string& control,
                               int startFrame, int endFrame, uint64_t revision) {
    return editIKChannels(
        character, revision, [&](AnimationData& clip, State& state, const auto& model) {
            if (startFrame < 0 || endFrame <= startFrame || endFrame > 1000000 ||
                !std::isfinite(state.fps) || state.fps <= 0)
                return Result::fail("rig_ik_invalid_contact_interval");
            const double start = double(startFrame) / state.fps, end = double(endFrame) / state.fps;
            if (end * clip.ticksPerSecond > 1000000)
                return Result::fail("rig_pose_time_limit");
            RigAuthoring::IKPose value;
            auto result = capturedIK(character, control, model, value);
            if (!result.ok)
                return result;
            const auto definition =
                std::find_if(model.rigAnatomy.controls.begin(), model.rigAnatomy.controls.end(),
                             [&](const auto& c) { return c.name == control; });
            if (definition != model.rigAnatomy.controls.end() && definition->solver == "aim")
                return Result::fail("rig_ik_contact_unsupported");
            RayTrophi::NodeHierarchy pose;
            Matrix4x4 placement;
            std::string error;
            if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, pose, error) ||
                !RigAuthoring::rigScenePlacement(g_ctx->scene, character, placement, error) ||
                !RigAuthoring::matchIKPose(pose, *definition, placement, value, error))
                return Result::fail(error);
            value.enabled = true;
            value.blend = 1;
            auto& contacts = clip.ikChannels[control].contacts;
            for (auto& k : contacts)
                if (std::fabs(k.start - start) < 1e-8 && std::fabs(k.end - end) < 1e-8) {
                    k.pose = value;
                    state.ik.erase(control);
                    return Result::success();
                }
            for (const auto& k : contacts)
                if (start < k.end && end > k.start)
                    return Result::fail("rig_ik_contact_overlap");
            contacts.push_back({start, end, value});
            std::sort(contacts.begin(), contacts.end(),
                      [](const auto& a, const auto& b) { return a.start < b.start; });
            extendIKClip(clip, end);
            state.ik.erase(control);
            return Result::success();
        });
}
Result clearRigIKChannels(const std::string& character, const std::string& control,
                          uint64_t revision) {
    return editIKChannels(character, revision, [&](AnimationData& clip, State& state, const auto&) {
        if (!clip.ikChannels.erase(control))
            return Result::fail("rig_edit_no_change");
        state.ik.erase(control);
        return Result::success();
    });
}
Result bakeRigIKChannels(const std::string& character, const std::string& name, int startFrame,
                         int endFrame, uint64_t revision) {
    return change(character, [&]() {
        auto state = g_ctx->scene.rigView.pose;
        if (state.hasPreview)
            return Result::fail("rig_pose_preview_active");
        if (name.empty() || name.size() > 128)
            return Result::fail("invalid_clip_name");
        auto clips = g_ctx->scene.animationDataList;
        const auto index = clipIndex(state, character);
        if (index == clips.size())
            return Result::fail("rig_pose_clip_required");
        if (clips[index]->ikChannels.empty())
            return Result::fail("rig_ik_channels_empty");
        for (const auto& c : clips)
            if (c && c->modelName == character && c->name == name)
                return Result::fail("rig_pose_clip_name_conflict");
        for (const auto& model : g_ctx->scene.importedModelContexts)
            if (model.importName == character) {
                if (model.rigRevision != revision)
                    return Result::fail("rig_edit_stale_revision");
                auto baked = std::make_shared<AnimationData>();
                std::string error;
                if (!RigAuthoring::bakeIKChannels(g_ctx->scene, model, *clips[index], startFrame,
                                                  endFrame, *baked, error))
                    return Result::fail(error);
                baked->name = name;
                clips.push_back(baked);
                state.clips[character] = name;
                state.locals.clear();
                state.ik.clear();
                state.control.clear();
                return publish(std::move(state), std::move(clips));
            }
        return Result::fail("unknown_character");
    });
}
Result createRigPoseClip(const std::string& character, const std::string& name, float fps) {
    return change(character, [&]() {
        if (name.empty() || name.size() > 128)
            return Result::fail("invalid_clip_name");
        if (!std::isfinite(fps) || fps < 1 || fps > 240)
            return Result::fail("invalid_clip_timing");
        for (const auto& c : g_ctx->scene.animationDataList)
            if (c && c->modelName == character && c->name == name)
                return Result::fail("rig_pose_clip_name_conflict");
        State state = g_ctx->scene.rigView.pose;
        state.clips[character] = name;
        auto clip = std::make_shared<AnimationData>();
        clip->rigAuthoring = true;
        clip->name = name;
        clip->modelName = character;
        clip->duration = 1;
        clip->ticksPerSecond = fps;
        std::string error;
        for (const auto& m : g_ctx->scene.importedModelContexts)
            if (m.importName == character) {
                std::vector<std::string> names;
                for (const auto& n : m.nodeHierarchy.nodes)
                    names.push_back(n.uniqueName);
                if (!RigAuthoring::insertPoseKeys(*clip, m.nodeHierarchy, names, 0, error))
                    return Result::fail(error);
            }
        state.locals.clear();
        state.preview.clear();
        state.ik.clear();
        state.previewIK.clear();
        state.hasPreview = false;
        state.limitHits.clear();
        auto clips = g_ctx->scene.animationDataList;
        clips.push_back(std::move(clip));
        return publish(std::move(state), std::move(clips));
    });
}
Result selectRigPoseClip(const std::string& character, const std::string& clip) {
    return change(character, [&]() {
        for (const auto& c : g_ctx->scene.animationDataList)
            if (c && c->rigAuthoring && c->modelName == character && c->name == clip) {
                auto state = g_ctx->scene.rigView.pose;
                ++state.serial;
                state.clips[character] = clip;
                state.locals.clear();
                state.preview.clear();
                state.ik.clear();
                state.previewIK.clear();
                state.hasPreview = false;
                state.limitHits.clear();
                g_ctx->scene.rigView.pose = std::move(state);
                wake();
                return Result::success();
            }
        return Result::fail("rig_pose_clip_not_editable");
    });
}
Result setRigPoseAutoKey(bool enabled) {
    if (!g_ctx)
        return Result::fail("api_not_bound");
    return change(g_ctx->scene.rigView.pose.character, [&]() {
        if (enabled && clipIndex(g_ctx->scene.rigView.pose, g_ctx->scene.rigView.pose.character) ==
                           g_ctx->scene.animationDataList.size())
            return Result::fail("rig_pose_clip_required");
        g_ctx->scene.rigView.pose.autoKey = enabled;
        return Result::success();
    });
}
Result setRigPoseFrame(int frame) {
    if (!g_ctx)
        return Result::fail("api_not_bound");
    return change(g_ctx->scene.rigView.pose.character, [&]() {
        if (frame < 0 || frame > 1000000)
            return Result::fail("rig_pose_time_limit");
        ui.timeline.pausePlayback();
        ui.timeline.setCurrentFrame(frame);
        g_ctx->scene.timeline.current_frame = frame;
        g_ctx->render_settings.animation_current_frame = frame;
        g_ctx->render_settings.animation_playback_frame = frame;
        RigAuthoring::synchronizePoseFrame(g_ctx->scene);
        wake();
        return Result::success();
    });
}
Result previewRigPoseTransform(const std::string& character, const std::vector<std::string>& bones,
                               const Matrix4x4& delta, uint64_t revision) {
    return change(character, [&]() {
        auto r = fkSelection(character, bones);
        if (!r.ok)
            return r;
        RayTrophi::NodeHierarchy before, after;
        std::string error;
        if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, before, error, false,
                                                nullptr, false))
            return Result::fail(error);
        Matrix4x4 placement;
        std::vector<RigAuthoring::BoneView> views;
        if (!RigAuthoring::listBones(g_ctx->scene, character, views, error) || views.empty())
            return Result::fail(error.empty() ? "character_has_no_skeleton" : error);
        placement = views.front().scene_transform;
        if (!RigAuthoring::transformRestHierarchy(before, placement, bones, delta, after, error))
            return Result::fail(error);
        return previewHierarchy(character, after, revision);
    });
}
Result previewRigPoseLocals(const std::string& character, const nlohmann::json& locals,
                            uint64_t revision) {
    return change(character, [&]() {
        std::vector<std::string> bones;
        if (locals.is_object())
            for (const auto& p : locals.items())
                bones.push_back(p.key());
        auto r = fkSelection(character, bones);
        if (!r.ok)
            return r;
        if (!locals.is_object() || locals.empty() || locals.size() > 4096)
            return Result::fail("invalid_pose_transforms");
        RayTrophi::NodeHierarchy before, after;
        std::string error;
        if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, before, error, false,
                                                nullptr, false))
            return Result::fail(error);
        RigAuthoring::JointGlobals values;
        for (const auto& p : locals.items()) {
            if (!p.value().is_array() || p.value().size() != 16)
                return Result::fail("invalid_pose_transforms");
            Matrix4x4 matrix;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) {
                    if (!p.value()[r * 4 + c].is_number())
                        return Result::fail("invalid_pose_transforms");
                    matrix.m[r][c] = p.value()[r * 4 + c].get<float>();
                }
            values[p.key()] = matrix;
        }
        if (!RigAuthoring::poseHierarchy(before, nullptr, 0, values, after, error))
            return Result::fail(error);
        return previewHierarchy(character, after, revision);
    });
}
Result getRigDrivenControls(const std::string& character, nlohmann::json& output) {
    if (!g_ctx)
        return Result::fail("api_not_bound");
    for (const auto& model : g_ctx->scene.importedModelContexts) {
        if (model.importName == character) {
            output = {{"character", character},
                      {"rig_revision", model.rigRevision},
                      {"controls", RigAuthoring::serializeRigDrivenControls(
                                       model.rigAnatomy.drivenControls)}};
            return Result::success();
        }
    }
    return Result::fail("unknown_character");
}
Result previewRigControlValues(const std::string& character, const nlohmann::json& values,
                               uint64_t revision) {
    return change(character, [&]() {
        if (!values.is_object() || values.empty() || values.size() > 4096)
            return Result::fail("rig_control_values_invalid");
        std::map<std::string, float> requested;
        for (const auto& value : values.items()) {
            if (!value.value().is_number())
                return Result::fail("rig_control_values_invalid");
            requested[value.key()] = value.value().get<float>();
        }
        RayTrophi::NodeHierarchy before, after;
        std::string error;
        if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, before, error, false,
                                                nullptr, false)) {
            return Result::fail(error);
        }
        for (const auto& model : g_ctx->scene.importedModelContexts) {
            if (model.importName != character)
                continue;
            if (model.rigRevision != revision)
                return Result::fail("rig_edit_stale_revision");
            std::vector<std::string> affected;
            if (!RigAuthoring::evaluateRigDrivenControls(
                    before, model.rigAnatomy.drivenControls, requested, after, affected, error)) {
                return Result::fail(error);
            }
            auto selection = fkSelection(character, affected);
            if (!selection.ok)
                return selection;
            return previewHierarchy(character, after, revision);
        }
        return Result::fail("unknown_character");
    });
}
Result cancelRigPosePreview(const std::string& character) {
    return change(character, [&]() {
        auto& s = g_ctx->scene.rigView.pose;
        if (!s.hasPreview)
            return Result::success();
        s.preview.clear();
        s.previewIK.clear();
        s.hasPreview = false;
        s.limitHits.clear();
        wake();
        return Result::success();
    });
}
Result mirrorRigPose(const std::string& character, const std::vector<std::string>& bones,
                     uint64_t revision, const std::string& direction, const std::string& axis) {
    return change(character, [&]() {
        if (g_ctx->scene.rigView.pose.hasPreview)
            return Result::fail("rig_pose_preview_active");
        RigAuthoring::IKPoses controls;
        std::string controlError;
        if (!RigAuthoring::effectiveIKControls(g_ctx->scene, character, false, controls,
                                               controlError))
            return Result::fail(controlError);
        if (RigAuthoring::hasEnabledIK(controls))
            return Result::fail("rig_ik_switch_to_fk");
        RayTrophi::NodeHierarchy before, after;
        std::string error;
        if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, before, error))
            return Result::fail(error);
        for (const auto& model : g_ctx->scene.importedModelContexts)
            if (model.importName == character) {
                if (model.rigRevision != revision)
                    return Result::fail("rig_edit_stale_revision");
                if (!RigAuthoring::mirrorPose(model.nodeHierarchy, before, model.rigAnatomy, bones,
                                              direction, axis, after, error))
                    return Result::fail(error);
                return previewHierarchy(character, after, revision);
            }
        return Result::fail("unknown_character");
    });
}
Result applyRigPosePreview(const std::string& character) {
    return change(character, [&]() {
        State state = g_ctx->scene.rigView.pose;
        if (!state.hasPreview)
            return Result::fail("rig_pose_preview_required");
        if (state.frame != g_ctx->scene.timeline.current_frame)
            return Result::fail("rig_pose_stale_preview");
        for (const auto& m : g_ctx->scene.importedModelContexts)
            if (m.importName == character && m.rigRevision != state.revision)
                return Result::fail("rig_pose_stale_preview");
        RayTrophi::NodeHierarchy before, after;
        std::string error;
        if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, before, error) ||
            !RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, after, error, true))
            return Result::fail(error);
        std::vector<std::string> changed;
        for (size_t i = 0; i < before.size(); ++i) {
            bool differs = false;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    differs = differs || std::fabs(before.nodes[i].localBind.m[r][c] -
                                                   after.nodes[i].localBind.m[r][c]) > 1e-6f;
            if (differs)
                changed.push_back(before.nodes[i].uniqueName);
        }
        const bool controlsChanged = !RigAuthoring::sameIKPoses(state.ik, state.previewIK);
        if (changed.empty() && !controlsChanged) {
            if (!state.limitHits.empty()) {
                auto& s = g_ctx->scene.rigView.pose;
                s.preview.clear();
                s.previewIK.clear();
                s.hasPreview = false;
                s.limitHits.clear();
                wake();
                return Result::success();
            }
            return Result::fail("rig_edit_no_change");
        }
        state.locals = state.preview;
        state.ik = state.previewIK;
        state.preview.clear();
        state.previewIK.clear();
        state.hasPreview = false;
        state.limitHits.clear();
        const auto index = clipIndex(state, character);
        const bool timed = index < g_ctx->scene.animationDataList.size() &&
                           !g_ctx->scene.animationDataList[index]->ikChannels.empty();
        if (state.autoKey && (controlsChanged || timed)) {
            auto clips = g_ctx->scene.animationDataList;
            if (index == clips.size())
                return Result::fail("rig_pose_clip_required");
            auto clip = std::make_shared<AnimationData>(*clips[index]);
            const double seconds = double(g_ctx->scene.timeline.current_frame) / state.fps;
            if (!std::isfinite(seconds) || seconds < 0 || seconds * clip->ticksPerSecond > 1000000)
                return Result::fail("rig_pose_time_limit");
            if (controlsChanged)
                for (const auto& p : state.ik)
                    if (!RigAuthoring::insertIKChannelKey(clip->ikChannels, p.first, seconds,
                                                          p.second, error))
                        return Result::fail(error);
            RigAuthoring::IKPoses evaluated;
            if (!RigAuthoring::effectiveIKControls(g_ctx->scene, character, true, evaluated, error))
                return Result::fail(error);
            std::vector<std::string> fkBones;
            for (const auto& model : g_ctx->scene.importedModelContexts)
                if (model.importName == character) {
                    if (!RigAuthoring::validateIKChannelControls(clip->ikChannels,
                                                                 model.rigAnatomy.controls, error))
                        return Result::fail(error);
                    for (const auto& bone : changed)
                        if (!RigAuthoring::ikDrivesBone(model.rigAnatomy.controls, evaluated, bone))
                            fkBones.push_back(bone);
                }
            if (!fkBones.empty()) {
                RayTrophi::NodeHierarchy fk;
                if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, fk, error, true,
                                                        nullptr, false) ||
                    !RigAuthoring::insertPoseKeys(*clip, fk, fkBones, seconds, error))
                    return Result::fail(error);
            }
            extendIKClip(*clip, seconds);
            if (controlsChanged)
                for (auto p = state.ik.begin(); p != state.ik.end();)
                    if (!p->second.contact)
                        p = state.ik.erase(p);
                    else
                        ++p;
            clips[index] = std::move(clip);
            return publish(std::move(state), std::move(clips));
        }
        return state.autoKey && !changed.empty()
                   ? writeKeys(std::move(state), after, changed)
                   : publish(std::move(state), g_ctx->scene.animationDataList);
    });
}
Result insertRigPoseKeys(const std::string& character, const std::vector<std::string>& bones) {
    return change(character, [&]() {
        auto state = g_ctx->scene.rigView.pose;
        if (state.hasPreview)
            return Result::fail("rig_pose_preview_active");
        RayTrophi::NodeHierarchy pose;
        std::string error;
        if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, pose, error))
            return Result::fail(error);
        const auto* clip = RigAuthoring::authoredIKClip(g_ctx->scene, character);
        if (clip && !clip->ikChannels.empty()) {
            RigAuthoring::IKPoses controls;
            if (!RigAuthoring::effectiveIKControls(g_ctx->scene, character, false, controls, error))
                return Result::fail(error);
            for (const auto& model : g_ctx->scene.importedModelContexts)
                if (model.importName == character)
                    for (const auto& bone : bones)
                        if (RigAuthoring::ikDrivesBone(model.rigAnatomy.controls, controls, bone))
                            return Result::fail("rig_ik_use_control_keys_or_bake");
        }
        return writeKeys(std::move(state), pose, bones);
    });
}
Result removeRigPoseKeys(const std::string& character, const std::vector<std::string>& bones) {
    return change(character, [&]() {
        auto state = g_ctx->scene.rigView.pose;
        if (state.hasPreview)
            return Result::fail("rig_pose_preview_active");
        RayTrophi::NodeHierarchy pose;
        std::string error;
        if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, pose, error))
            return Result::fail(error);
        auto clips = g_ctx->scene.animationDataList;
        const auto index = clipIndex(state, character);
        if (index == clips.size())
            return Result::fail("rig_pose_clip_required");
        auto clip = std::make_shared<AnimationData>(*clips[index]);
        if (!RigAuthoring::removePoseKeys(
                *clip, pose, bones,
                double(g_ctx->scene.timeline.current_frame) / state.fps, error))
            return Result::fail(error);
        clips[index] = std::move(clip);
        return publish(std::move(state), std::move(clips));
    });
}
Result editRigPoseKey(const std::string& character, const std::string& bone,
                      const std::string& channel, int sourceFrame, int targetFrame,
                      const nlohmann::json& value) {
    return change(character, [&]() {
        auto state = g_ctx->scene.rigView.pose;
        if (state.hasPreview)
            return Result::fail("rig_pose_preview_active");
        if (sourceFrame < 0 || targetFrame < 0 || sourceFrame > 1000000 ||
            targetFrame > 1000000)
            return Result::fail("invalid_frame");
        if (!std::isfinite(state.fps) || state.fps <= 0.0f)
            return Result::fail("invalid_clip_timing");
        bool knownBone = false;
        for (const auto& model : g_ctx->scene.importedModelContexts)
            if (model.importName == character)
                knownBone = model.nodeHierarchy.find(bone) != nullptr;
        if (!knownBone)
            return Result::fail("unknown_bone");

        RigAuthoring::BoneCurveChannel kind;
        Vec3 position;
        Quaternion rotation;
        const Vec3* positionValue = nullptr;
        const Quaternion* rotationValue = nullptr;
        if (channel == "position") {
            kind = RigAuthoring::BoneCurveChannel::Position;
            if (!value.is_null()) {
                if (!value.is_array() || value.size() != 3)
                    return Result::fail("rig_curve_invalid_value");
                for (const auto& component : value)
                    if (!component.is_number())
                        return Result::fail("rig_curve_invalid_value");
                position = Vec3(value[0].get<float>(), value[1].get<float>(),
                                value[2].get<float>());
                positionValue = &position;
            }
        } else if (channel == "rotation") {
            kind = RigAuthoring::BoneCurveChannel::Rotation;
            if (!value.is_null()) {
                if (!value.is_array() || value.size() != 4)
                    return Result::fail("rig_curve_invalid_value");
                for (const auto& component : value)
                    if (!component.is_number())
                        return Result::fail("rig_curve_invalid_value");
                rotation = Quaternion(value[0].get<float>(), value[1].get<float>(),
                                      value[2].get<float>(), value[3].get<float>());
                rotationValue = &rotation;
            }
        } else {
            return Result::fail("rig_curve_invalid_channel");
        }

        auto clips = g_ctx->scene.animationDataList;
        const auto index = clipIndex(state, character);
        if (index == clips.size())
            return Result::fail("rig_pose_clip_required");
        const auto& selectedClip = clips[index];
        if (!std::isfinite(selectedClip->duration) || selectedClip->duration <= 0.0 ||
            !std::isfinite(selectedClip->ticksPerSecond) ||
            selectedClip->ticksPerSecond <= 0.0)
            return Result::fail("invalid_clip_timing");
        if (!selectedClip->ikChannels.empty()) {
            const double durationSeconds =
                selectedClip->duration / selectedClip->ticksPerSecond;
            const auto drivenAt = [&](const auto& model, int frame) {
                const double seconds = std::fmod(
                    static_cast<double>(frame) / state.fps, durationSeconds);
                const auto controls = RigAuthoring::sampleIKChannels(
                    selectedClip->ikChannels, seconds);
                return RigAuthoring::ikDrivesBone(
                    model.rigAnatomy.controls, controls, bone);
            };
            for (const auto& model : g_ctx->scene.importedModelContexts) {
                if (model.importName == character &&
                    (drivenAt(model, sourceFrame) || drivenAt(model, targetFrame)))
                    return Result::fail("rig_ik_use_control_keys_or_bake");
            }
        }
        auto clip = std::make_shared<AnimationData>(*clips[index]);
        std::string error;
        if (!RigAuthoring::editBoneCurveKey(
                *clip, bone, kind,
                static_cast<double>(sourceFrame) / state.fps,
                static_cast<double>(targetFrame) / state.fps,
                positionValue, rotationValue, error))
            return Result::fail(error);
        clips[index] = std::move(clip);
        return publish(std::move(state), std::move(clips));
    });
}
Result getRigPoseCoverage(const std::string& character, nlohmann::json& output) {
    output = nullptr;
    if (!g_ctx)
        return Result::fail("api_not_bound");
    if (renderJobActive())
        return Result::fail("scene_locked");
    try {
        std::string error;
        if (!RigAuthoring::canAuthorPose(g_ctx->scene, character, error))
            return Result::fail(error);
        size_t count = 0, empty = 0;
        auto parts = nlohmann::json::array();
        for (const auto& o : g_ctx->scene.world.objects) {
            auto mesh = std::dynamic_pointer_cast<TriangleMesh>(o);
            if (!mesh || !mesh->hasSkinWeights() ||
                !RigAuthoring::meshBelongsToRig(g_ctx->scene, character, *mesh))
                continue;
            nlohmann::json stats;
            if (!RigAuthoring::weightStats(g_ctx->scene, mesh->nodeName, stats, error))
                return Result::fail(error);
            count += stats["vertex_count"].get<size_t>();
            empty += stats["unweighted_vertices"].get<size_t>();
            parts.push_back(stats);
        }
        output = {{"character", character},
                  {"skeleton_authoring", true},
                  {"deformation_preview", !parts.empty()},
                  {"vertex_count", count},
                  {"unweighted_vertices", empty},
                  {"parts", parts},
                  {"unweighted_behavior", "bind_position"}};
        return Result::success();
    } catch (const std::exception&) {
        return Result::fail("rig_pose_failed");
    }
}
Result getRigPoseState(const std::string& character, nlohmann::json& output) {
    output = nullptr;
    if (!g_ctx)
        return Result::fail("api_not_bound");
    if (renderJobActive())
        return Result::fail("scene_locked");
    try {
        std::string error;
        if (!RigAuthoring::canAuthorPose(g_ctx->scene, character, error))
            return Result::fail(error);
        RayTrophi::NodeHierarchy pose;
        std::vector<std::string> limitHits;
        if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, pose, error, true,
                                                &limitHits))
            return Result::fail(error);
        nlohmann::json locals = nlohmann::json::object(), clips = nlohmann::json::array();
        for (const auto& n : pose.nodes) {
            auto matrix = nlohmann::json::array();
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    matrix.push_back(n.localBind.m[r][c]);
            locals[n.uniqueName] = matrix;
        }
        for (const auto& clip : g_ctx->scene.animationDataList)
            if (clip && clip->rigAuthoring && clip->modelName == character) {
                size_t keys = 0;
                for (const auto& p : clip->rotationKeys)
                    keys += p.second.size();
                clips.push_back({{"name", clip->name},
                                 {"rotation_keys", keys},
                                 {"duration", clip->duration},
                                 {"ticks_per_second", clip->ticksPerSecond}});
            }
        const auto& state = g_ctx->scene.rigView.pose;
        const auto selected = state.clips.find(character);
        auto keyedBones = nlohmann::json::array();
        if (const auto* clip = RigAuthoring::authoredIKClip(g_ctx->scene, character)) {
            const double time = double(g_ctx->scene.timeline.current_frame) / state.fps *
                                clip->ticksPerSecond;
            for (const auto& entry : clip->rotationKeys) {
                const auto key = std::lower_bound(
                    entry.second.begin(), entry.second.end(), time,
                    [](const auto& value, double requested) { return value.time < requested; });
                if (key != entry.second.end() && std::fabs(key->time - time) < 1e-8)
                    keyedBones.push_back(entry.first);
            }
        }
        output = {{"character", character},
                  {"active", state.active && state.character == character},
                  {"auto_key", state.autoKey},
                  {"preview", state.hasPreview && state.character == character &&
                                  state.frame == g_ctx->scene.timeline.current_frame},
                  {"frame", g_ctx->scene.timeline.current_frame},
                  {"fps", state.fps},
                  {"clip", selected == state.clips.end() ? "" : selected->second},
                  {"clips", clips},
                  {"keyed_bones_at_frame", std::move(keyedBones)},
                  {"local_transforms", locals},
                  {"limit_hits", state.active && state.character == character && state.hasPreview
                                     ? state.limitHits
                                     : limitHits}};
        for (const auto& m : g_ctx->scene.importedModelContexts)
            if (m.importName == character)
                output["rig_revision"] = m.rigRevision;
        return Result::success();
    } catch (const std::exception&) {
        return Result::fail("rig_pose_failed");
    }
}
}
