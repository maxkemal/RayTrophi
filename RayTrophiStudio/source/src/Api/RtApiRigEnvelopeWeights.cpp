#include "Api/RtApiRigEnvelopeWeights.h"
#include "ProjectManager.h"
#include "RtApiInternal.h"
#include "TriangleMesh.h"
#include <exception>
#include <utility>

namespace {
class EnvelopeWeightCommand final : public SceneCommand {
    RigAuthoring::EnvelopeWeightState pending;
    size_t triangles = 0;

    void exchange(UIContext &context) {
        for (auto &model : context.scene.importedModelContexts) {
            if (model.importName == pending.character) {
                std::swap(model.rigRevision, pending.revision);
                std::swap(model.rigWeightAlgorithm, pending.algorithm);
                std::swap(model.rigEnvelopeTorsoRadius, pending.settings.torsoRadius);
                std::swap(model.rigEnvelopeLimbRadius, pending.settings.limbRadius);
                std::swap(model.rigEnvelopeExtremityRadius, pending.settings.extremityRadius);
                std::swap(model.rigEnvelopeFalloff, pending.settings.falloff);
                std::swap(model.rigEnvelopeProfiles, pending.profiles);
                break;
            }
        }
        for (auto &part : pending.parts) {
            part.mesh->geometry.swap(part.geometry);
            part.mesh->local_bvh.reset();
            part.mesh->pointiness.clear();
            part.mesh->material_attribs.clear();
            part.mesh->geometry->last_skinned_pose_hash = 0;
        }
        context.renderer.invalidateAnimationGeometry();
        context.renderer.animation_groups_dirty = true;
        g_geometry_dirty = true;
        g_bvh_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        g_vulkan_rebuild_pending = true;
        g_viewport_raster_rebuild_pending = true;
        ui.mesh_cache_valid = false;
        ProjectManager::getInstance().markModified();
        try {
            scheduleSceneMutationRebuilds(context, true);
        } catch (...) {
            g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
        }
    }

  public:
    explicit EnvelopeWeightCommand(RigAuthoring::EnvelopeWeightState state)
        : pending(std::move(state)) {
        for (const auto &part : pending.parts) {
            triangles += part.mesh->num_triangles();
        }
    }

    void execute(UIContext &context) override {
        exchange(context);
    }

    void undo(UIContext &context) override {
        exchange(context);
    }

    Type getType() const override {
        return Type::Heavy;
    }

    bool isHeavyGeometry() const override {
        return true;
    }

    size_t getTriangleCount() const override {
        return triangles;
    }

    std::string getDescription() const override {
        return "Rebuild anatomical envelope weights: " + pending.character;
    }
};
} // namespace

namespace rtapi {
Result previewRigEnvelopeWeights(const std::string &character,
                                 const RigAuthoring::EnvelopeWeightSettings &settings,
                                 nlohmann::json &output) {
    output = nullptr;
    if (!g_ctx) {
        return Result::fail("api_not_bound");
    }
    if (renderJobActive()) {
        return Result::fail("scene_locked");
    }
    try {
        std::string error;
        return RigAuthoring::previewEnvelopeWeights(g_ctx->scene, character, settings, output,
                                                    error)
                   ? Result::success()
                   : Result::fail(error);
    } catch (const std::exception &) {
        return Result::fail("rig_envelope_failed");
    }
}

Result applyRigEnvelopeWeights(const std::string &character,
                               const RigAuthoring::EnvelopeWeightSettings &settings,
                               uint64_t revision) {
    if (!g_ctx) {
        return Result::fail("api_not_bound");
    }
    if (renderJobActive()) {
        return Result::fail("scene_locked");
    }
    if (!g_history) {
        return Result::fail("history_not_bound");
    }
    try {
        RigAuthoring::EnvelopeWeightState state;
        nlohmann::json report;
        std::string error;
        if (!RigAuthoring::stageEnvelopeWeights(g_ctx->scene, character, settings, revision, state,
                                                report, error)) {
            return Result::fail(error);
        }
        auto command = std::make_unique<EnvelopeWeightCommand>(std::move(state));
        auto *recorded = command.get();
        g_history->record(std::move(command));
        recorded->execute(*g_ctx);
        return Result::success();
    } catch (const std::exception &) {
        return Result::fail("rig_envelope_failed");
    }
}

Result getRigEnvelopeOverlay(nlohmann::json &output) {
    output = nullptr;
    if (!g_ctx)
        return Result::fail("api_not_bound");
    const auto &view = g_ctx->scene.rigView;
    output = {{"visible", view.envelope_overlay_visible},
              {"character", view.envelope_overlay_character},
              {"torso_radius", view.envelope_overlay_torso_radius},
              {"limb_radius", view.envelope_overlay_limb_radius},
              {"extremity_radius", view.envelope_overlay_extremity_radius},
              {"falloff", view.envelope_overlay_falloff}};
    return Result::success();
}

Result setRigEnvelopeOverlay(const std::string &character,
                             const RigAuthoring::EnvelopeWeightSettings &settings, bool visible) {
    if (!g_ctx)
        return Result::fail("api_not_bound");
    if (renderJobActive())
        return Result::fail("scene_locked");
    if (visible) {
        std::vector<RigAuthoring::EnvelopeSegmentView> segments;
        std::string error;
        if (!RigAuthoring::envelopeSegmentViews(g_ctx->scene, character, settings, segments,
                                                error)) {
            return Result::fail(error);
        }
    } else if (!RigAuthoring::validEnvelopeWeightSettings(settings)) {
        return Result::fail("rig_envelope_invalid_settings");
    }
    auto &view = g_ctx->scene.rigView;
    view.envelope_overlay_visible = visible;
    view.envelope_overlay_character = character;
    view.envelope_overlay_torso_radius = settings.torsoRadius;
    view.envelope_overlay_limb_radius = settings.limbRadius;
    view.envelope_overlay_extremity_radius = settings.extremityRadius;
    view.envelope_overlay_falloff = settings.falloff;
    g_ctx->start_render = true;
    return Result::success();
}

Result getRigBoneEnvelope(const std::string &character, const std::string &bone,
                          nlohmann::json &output) {
    output = nullptr;
    if (!g_ctx)
        return Result::fail("api_not_bound");
    try {
        RigAuthoring::EnvelopeBoneProfile profile;
        uint64_t revision = 0;
        bool overridden = false;
        std::string error;
        if (!RigAuthoring::getEnvelopeBoneProfile(g_ctx->scene, character, bone, profile, revision,
                                                  overridden, error)) {
            return Result::fail(error);
        }
        output = {{"character", character},
                  {"bone", profile.bone},
                  {"rig_revision", revision},
                  {"overridden", overridden},
                  {"start_radius", profile.startRadius},
                  {"end_radius", profile.endRadius},
                  {"start_extension", profile.startExtension},
                  {"end_extension", profile.endExtension},
                  {"falloff", profile.falloff}};
        return Result::success();
    } catch (const std::exception &) {
        return Result::fail("rig_envelope_failed");
    }
}

Result applyRigBoneEnvelope(const std::string &character,
                            const RigAuthoring::EnvelopeBoneProfile &profile,
                            uint64_t revision) {
    if (!g_ctx)
        return Result::fail("api_not_bound");
    if (renderJobActive())
        return Result::fail("scene_locked");
    if (!g_history)
        return Result::fail("history_not_bound");
    try {
        RigAuthoring::EnvelopeWeightState state;
        nlohmann::json report;
        std::string error;
        if (!RigAuthoring::stageEnvelopeBoneProfile(g_ctx->scene, character, profile, revision,
                                                    state, report, error)) {
            return Result::fail(error);
        }
        auto command = std::make_unique<EnvelopeWeightCommand>(std::move(state));
        auto *recorded = command.get();
        g_history->record(std::move(command));
        recorded->execute(*g_ctx);
        return Result::success();
    } catch (const std::exception &) {
        return Result::fail("rig_envelope_failed");
    }
}
} // namespace rtapi
