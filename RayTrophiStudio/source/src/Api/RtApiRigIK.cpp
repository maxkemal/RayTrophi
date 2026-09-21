#include "RtApiInternal.h"
#include "Api/RtApiRigIK.h"
#include "Animation/RigPoseAuthoring.h"
#include "Animation/RigIKTimeline.h"
#include "Animation/RigControlDisplay.h"
#include "ProjectManager.h"
#include "Backend/IBackend.h"
#include <limits>
#include <algorithm>
#include <cmath>
namespace rtapi {
namespace {
Vec3 origin(const Matrix4x4& matrix) {
    return Vec3(matrix.m[0][3], matrix.m[1][3], matrix.m[2][3]);
}

Vec3 aimUpAxis(const Vec3& aim) {
    const auto reference = std::fabs(aim.z) < .9f ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
    return (reference - aim * Vec3::dot(aim, reference)).normalize();
}

class ControlsCommand final : public SceneCommand {
    std::string character;
    std::vector<RigAuthoring::IKControl> pending;
    uint64_t revision;
    RigAuthoring::PoseAuthoringState pendingPose;
    void exchange(UIContext& ctx) {
        for (auto& m : ctx.scene.importedModelContexts)
            if (m.importName == character) {
                m.rigAnatomy.controls.swap(pending);
                std::swap(m.rigRevision, revision);
                auto& p = ctx.scene.rigView.pose;
                const bool active = p.active;
                const auto currentCharacter = p.character;
                if (!active || currentCharacter == character) {
                    std::swap(p, pendingPose);
                    p.active = active;
                    p.character = currentCharacter;
                    p.preview.clear();
                    p.previewIK.clear();
                    p.hasPreview = false;
                    p.limitHits.clear();
                    p.invalidateEvaluation();
                }
                ctx.start_render = true;
                ctx.renderer.resetCPUAccumulation();
                if (ctx.backend_ptr)
                    ctx.backend_ptr->resetAccumulation();
                ProjectManager::getInstance().markModified();
                break;
            }
    }

  public:
    ControlsCommand(std::string c, std::vector<RigAuthoring::IKControl> controls, uint64_t r,
                    const RigAuthoring::PoseAuthoringState& pose)
        : character(std::move(c)), pending(std::move(controls)), revision(r), pendingPose(pose) {
        pendingPose.ik.clear();
        pendingPose.previewIK.clear();
        pendingPose.control.clear();
        ++pendingPose.serial;
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
        return "Rig IK controls";
    }
};
}
Result getRigControls(const std::string& character, nlohmann::json& output) {
    output = nullptr;
    if (!g_ctx)
        return Result::fail("api_not_bound");
    if (renderJobActive())
        return Result::fail("scene_locked");
    try {
        std::string error;
        if (!RigAuthoring::canAuthorPose(g_ctx->scene, character, error))
            return Result::fail(error);
        for (const auto& model : g_ctx->scene.importedModelContexts)
            if (model.importName == character) {
                RayTrophi::NodeHierarchy pose;
                if (!RigAuthoring::currentPoseHierarchy(g_ctx->scene, character, pose, error, true))
                    return Result::fail(error);
                Matrix4x4 placement;
                if (!RigAuthoring::rigScenePlacement(g_ctx->scene, character, placement, error))
                    return Result::fail(error);
                const auto& state = g_ctx->scene.rigView.pose;
                RigAuthoring::IKPoses runtime;
                if (!RigAuthoring::effectiveIKControls(g_ctx->scene, character, true, runtime,
                                                       error))
                    return Result::fail(error);
                if (!RigAuthoring::validateIKControls(model.rigAnatomy.controls, pose, error))
                    return Result::fail(error);
                for (const auto& control : model.rigAnatomy.controls)
                    if (!runtime.count(control.name)) {
                        RigAuthoring::IKPose matched;
                        if (!RigAuthoring::matchIKPose(pose, control, placement, matched, error))
                            return Result::fail(error);
                        runtime[control.name] = matched;
                    }
                auto controls = RigAuthoring::inspectIKPose(pose, model.rigAnatomy.controls,
                                                            runtime, placement);
                for (size_t index = 0; index < controls.size(); ++index) {
                    auto& row = controls[index];
                    const auto display = RigAuthoring::deriveRigControlDisplay(
                        model.rigAnatomy, model.rigAnatomy.controls[index]);
                    row["display"] = RigAuthoring::serializeRigControlDisplay(display);
                }
                output = {{"character", character},
                          {"rig_revision", model.rigRevision},
                          {"controls", std::move(controls)},
                          {"display_contract", RigAuthoring::rigControlDisplayContract()},
                          {"selected_control",
                           state.active && state.character == character ? state.control : ""},
                          {"handle", state.controlHandle},
                          {"coordinate_space", "world"},
                          {"contact_kind", "position_only"},
                          {"target_keys", true}};
                return Result::success();
            }
        return Result::fail("unknown_character");
    } catch (const std::exception&) {
        return Result::fail("rig_ik_failed");
    }
}
Result selectRigControl(const std::string& character, const std::string& control,
                        const std::string& handle) {
    if (!g_ctx)
        return Result::fail("api_not_bound");
    if (renderJobActive())
        return Result::fail("scene_locked");
    if (handle != "target" && handle != "pole" && handle != "orientation" && handle != "spline_0" &&
        handle != "spline_1")
        return Result::fail("rig_ik_invalid_handle");
    auto& state = g_ctx->scene.rigView.pose;
    if (!state.active || state.character != character)
        return Result::fail("rig_pose_mode_required");
    if (state.hasPreview)
        return Result::fail("rig_pose_preview_active");
    for (const auto& m : g_ctx->scene.importedModelContexts)
        if (m.importName == character) {
            if (!control.empty() &&
                std::none_of(m.rigAnatomy.controls.begin(), m.rigAnatomy.controls.end(),
                             [&](const auto& c) { return c.name == control; }))
                return Result::fail("rig_ik_unknown_control");
            if (handle == "spline_0" || handle == "spline_1") {
                const auto found =
                    std::find_if(m.rigAnatomy.controls.begin(), m.rigAnatomy.controls.end(),
                                 [&](const auto& c) { return c.name == control; });
                if (found == m.rigAnatomy.controls.end() || found->chain.empty()) {
                    return Result::fail("rig_ik_spline_requires_chain");
                }
            }
            state.control = control;
            state.controlHandle = handle;
            ++state.serial;
            g_ctx->start_render = true;
            return Result::success();
        }
    return Result::fail("unknown_character");
}
Result createRigChainControl(const std::string& character, const std::string& chain,
                             uint64_t revision) {
    if (!g_ctx) {
        return Result::fail("api_not_bound");
    }
    if (renderJobActive()) {
        return Result::fail("scene_locked");
    }
    for (const auto& model : g_ctx->scene.importedModelContexts) {
        if (model.importName != character) {
            continue;
        }
        const auto found =
            std::find_if(model.rigAnatomy.chains.begin(), model.rigAnatomy.chains.end(),
                         [&](const auto& value) { return value.name == chain; });
        if (found == model.rigAnatomy.chains.end()) {
            return Result::fail("rig_ik_unknown_chain");
        }
        if (found->bones.size() < 4 || found->bones.size() > 64) {
            return Result::fail("rig_ik_invalid_chain");
        }
        auto controls = model.rigAnatomy.controls;
        controls.push_back(
            {chain, found->bones.front(), found->bones[1], found->bones.back(), found->bones});
        return createRigControls(character, RigAuthoring::serializeIKControls(controls), revision);
    }
    return Result::fail("unknown_character");
}
Result createRigAimControl(const std::string& character, const std::string& role,
                           uint64_t revision) {
    if (!g_ctx) {
        return Result::fail("api_not_bound");
    }
    if (renderJobActive()) {
        return Result::fail("scene_locked");
    }
    for (const auto& model : g_ctx->scene.importedModelContexts) {
        if (model.importName != character) {
            continue;
        }
        const auto found =
            std::find_if(model.rigAnatomy.roles.begin(), model.rigAnatomy.roles.end(),
                         [&](const auto& value) { return value.role == role; });
        if (found == model.rigAnatomy.roles.end()) {
            return Result::fail("rig_ik_unknown_role");
        }
        const auto* bone = model.nodeHierarchy.find(found->bone);
        if (!bone) {
            return Result::fail("unknown_bone");
        }
        const size_t index = static_cast<size_t>(bone - model.nodeHierarchy.nodes.data());
        Vec3 axis(0, 1, 0);
        for (const auto& child : model.nodeHierarchy.nodes) {
            if (child.parent != static_cast<int>(index)) {
                continue;
            }
            const auto candidate = origin(child.localBind);
            if (candidate.length_squared() > 1e-12f) {
                axis = candidate.normalize();
                break;
            }
        }
        RigAuthoring::IKControl control;
        control.name = role + ".aim";
        control.root = found->bone;
        control.mid = found->bone;
        control.tip = found->bone;
        control.solver = "aim";
        control.aimAxis = axis;
        control.upAxis = aimUpAxis(axis);
        auto controls = model.rigAnatomy.controls;
        controls.push_back(std::move(control));
        return createRigControls(character, RigAuthoring::serializeIKControls(controls), revision);
    }
    return Result::fail("unknown_character");
}
Result createRigControls(const std::string& character, const nlohmann::json& controls,
                         uint64_t revision) {
    if (!g_ctx)
        return Result::fail("api_not_bound");
    if (renderJobActive())
        return Result::fail("scene_locked");
    if (!g_history)
        return Result::fail("history_not_bound");
    try {
        std::string error;
        if (!RigAuthoring::canAuthorPose(g_ctx->scene, character, error))
            return Result::fail(error);
        if (g_ctx->scene.rigView.pose.hasPreview)
            return Result::fail("rig_pose_preview_active");
        if (g_ctx->scene.rigView.pose.active && g_ctx->scene.rigView.pose.character != character)
            return Result::fail("rig_pose_character_mismatch");
        for (const auto& m : g_ctx->scene.importedModelContexts)
            if (m.importName == character) {
                if (m.rigRevision != revision)
                    return Result::fail("rig_edit_stale_revision");
                if (revision == std::numeric_limits<uint64_t>::max())
                    return Result::fail("rig_revision_overflow");
                std::vector<RigAuthoring::IKControl> staged;
                if (controls.is_null()) {
                    if (!RigAuthoring::buildLimbIKControls(m.rigAnatomy, m.nodeHierarchy, staged,
                                                           error))
                        return Result::fail(error);
                } else if (!RigAuthoring::deserializeIKControls(controls, m.nodeHierarchy, staged,
                                                                error))
                    return Result::fail(error);
                if (RigAuthoring::serializeIKControls(staged) ==
                    RigAuthoring::serializeIKControls(m.rigAnatomy.controls))
                    return Result::fail("rig_edit_no_change");
                for (const auto& clip : g_ctx->scene.animationDataList)
                    if (clip && clip->modelName == character &&
                        !RigAuthoring::validateIKChannelControls(clip->ikChannels, staged, error))
                        return Result::fail(error);
                auto command = std::make_unique<ControlsCommand>(
                    character, std::move(staged), revision + 1, g_ctx->scene.rigView.pose);
                auto* raw = command.get();
                g_history->record(std::move(command));
                raw->execute(*g_ctx);
                return Result::success();
            }
        return Result::fail("unknown_character");
    } catch (const std::exception&) {
        return Result::fail("rig_ik_failed");
    }
}
}
