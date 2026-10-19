#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "Audio/AudioRequestValidation.h"
#include "Audio/AudioDomainBridge.h"
#include "Audio/AcousticScene.h"
#include "AssetRegistry.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "SceneSelection.h"
#include "SceneHistory.h"
#include "ProjectManager.h"
#include "Api/RtApi.h"
#include "Camera.h"
#include <cmath>
#include <stdexcept>

namespace audio {
namespace {
Matrix4x4 parentMatrix(const nlohmann::json& emitter) {
    Matrix4x4 matrix;
    const auto parent = emitter.value("parent", std::string());
    if (!parent.empty()) {
        // getObjectWorldTransform resolves the canonical flat mesh/spline transform and
        // composes the existing rigid-body world delta; no Triangle facades are involved.
        const auto result = rtapi::getObjectWorldTransform(parent, matrix);
        if (!result.ok) {
            throw std::invalid_argument("Audio parent unavailable: " + parent);
        }
        const float determinant =
            matrix.m[0][0] * (matrix.m[1][1] * matrix.m[2][2] - matrix.m[1][2] * matrix.m[2][1]) -
            matrix.m[0][1] * (matrix.m[1][0] * matrix.m[2][2] - matrix.m[1][2] * matrix.m[2][0]) +
            matrix.m[0][2] * (matrix.m[1][0] * matrix.m[2][1] - matrix.m[1][1] * matrix.m[2][0]);
        if (!std::isfinite(determinant) || std::abs(determinant) < 1e-9f) {
            throw std::invalid_argument("Audio parent has a singular transform: " + parent);
        }
    }
    return matrix;
}
nlohmann::json transform(const Matrix4x4& matrix, const nlohmann::json& value, float w) {
    const auto v = matrix.multiplyVector(Vec4(value[0], value[1], value[2], w));
    if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
        throw std::invalid_argument("Audio world transform must be finite");
    }
    return nlohmann::json::array({v.x, v.y, v.z});
}
}

nlohmann::json worldPose(UIContext&, const nlohmann::json& emitter) {
    const auto matrix = parentMatrix(emitter);
    return {{"position", transform(matrix, emitter.at("position"), 1)},
            {"direction", transform(matrix, emitter.at("direction"), 0)}};
}

nlohmann::json localPosition(UIContext&, const nlohmann::json& emitter,
                             const nlohmann::json& worldPosition) {
    return transform(parentMatrix(emitter).inverse(), worldPosition, 1);
}

void syncScene(UIContext& ctx) {
    syncDomainAudioTelemetry(ctx);
    const auto data = ctx.scene.audio->snapshot();
    bool selectionExists = false;
    for (const auto& emitter : data.at("emitters")) {
        selectionExists = selectionExists || emitter.at("id") == ctx.selection.selected.audio_source_id;
        try {
            const auto pose = worldPose(ctx, emitter);
            ctx.scene.audio->resolvedPose(emitter.at("id"), pose.at("position"),
                                          pose.at("direction"), true);
        } catch (const std::exception&) {
            ctx.scene.audio->resolvedPose(emitter.at("id"), emitter.at("position"),
                                          emitter.at("direction"), false);
        }
    }
    if (ctx.selection.selected.type == SelectableType::AudioSource && !selectionExists) {
        ctx.selection.clearSelection();
    }
}

namespace {
class AudioCommand final : public SceneCommand {
public:
    AudioCommand(nlohmann::json before, nlohmann::json after, std::string description)
        : before_(std::move(before)), after_(std::move(after)), description_(std::move(description)) {}

    void execute(UIContext& ctx) override {
        ctx.scene.audio->restore(after_);
        syncScene(ctx);
        ProjectManager::getInstance().markModified();
    }
    void undo(UIContext& ctx) override {
        ctx.scene.audio->restore(before_);
        syncScene(ctx);
        ProjectManager::getInstance().markModified();
    }
    Type getType() const override {
        return Type::Generic;
    }
    bool handlesUiCacheSync() const override {
        return true;
    }
    std::string getDescription() const override {
        return description_;
    }
private:
    nlohmann::json before_;
    nlohmann::json after_;
    std::string description_;
};
}

void recordEdit(UIContext& ctx, SceneHistory* history, const nlohmann::json& before,
                const std::string& description) {
    const auto after = ctx.scene.audio->snapshot();
    if (before == after) {
        return;
    }
    if (history) {
        history->record(std::make_unique<AudioCommand>(before, after, description));
    }
    ProjectManager::getInstance().markModified();
}

nlohmann::json author(UIContext& ctx, SceneHistory* history, const std::string& operation,
                      const nlohmann::json& params) {
    if (!params.is_object()) {
        throw std::invalid_argument("audio params must be an object");
    }
    if (g_scene_loading_in_progress.load()) {
        throw std::runtime_error("Scene audio is unavailable while a project is loading");
    }
    if (AudioService::isAuthoring(operation) && rtapi::renderOutputPending()) {
        throw std::runtime_error("Scene audio authoring is locked by the render job");
    }
    if (operation == "analysis_get" || operation == "physical_catalogue") {
        return ctx.scene.audio->invoke(operation, params);
    }
    if (operation == "acoustics_boundary") {
        return ctx.scene.audio->invoke(operation, params);
    }
    if (operation == "acoustics_debug") {
        return queryAcousticDebug(ctx, params);
    }
    if (operation == "acoustics_trace") {
        return querySceneAcoustics(ctx, params);
    }
    if (operation == "domains") {
        if (!params.is_object() || !params.empty()) {
            throw std::invalid_argument("audio.domains takes no parameters");
        }
        return domainAudioInputs(ctx);
    }
    auto input = params;
    if ((operation == "design_destination" || operation == "design_save") &&
        !input.contains("folder") && !input.contains("path")) {
        const auto project = ProjectManager::getInstance().getCurrentFilePath();
        auto root = ctx.scene_ui_ptr ? ctx.scene_ui_ptr->getActiveAssetLibraryRoot() :
                                      std::filesystem::path();
        if (!project.empty()) {
            root = std::filesystem::path(std::u8string(project.begin(), project.end())).parent_path() /
                   "assets";
        } else if (root.empty()) {
            root = AssetRegistry::resolveDefaultRoot();
        }
        if (root.empty()) {
            throw std::invalid_argument("No project or asset-library folder; choose a sound folder");
        }
        input["folder"] = pathUtf8(std::filesystem::absolute(root / "audio" / "generated"));
    }
    if ((operation == "arrange_preview" || operation == "design_preview" || operation == "design_render" ||
         operation == "design_undo" || operation == "design_redo" || operation == "design_reuse") &&
        rtapi::renderOutputPending()) {
        throw std::runtime_error("Sound design preview is locked by the render job");
    }
    if (operation == "design_destination") {
        return ctx.scene.audio->invoke(operation, input);
    }
    if (operation == "thunder_event") {
        if (!ctx.scene.camera) {
            throw std::invalid_argument("Thunder needs an active listener camera");
        }
        const auto& camera = *ctx.scene.camera;
        const Vec3 direction = camera.lookat - camera.lookfrom;
        ctx.scene.audio->listener(camera.lookfrom.x, camera.lookfrom.y, camera.lookfrom.z,
            direction.x, direction.y, direction.z, camera.vup.x, camera.vup.y, camera.vup.z);
    }
    if ((operation == "environment_setup" || operation == "thunder_setup") &&
        !input.contains("assets_root")) {
        input["assets_root"] = pathUtf8(AssetRegistry::resolveDefaultRoot());
    }
    validateAudioRequest(operation, input);
    if (input.contains("reaction") && input.at("reaction").value("driver", std::string()) ==
        "domain_contact") {
        const auto& reaction = input.at("reaction");
        bool found = false;
        for (const auto& domain : domainAudioInputs(ctx)) {
            found = found || (domain.at("system_id") == reaction.value("system_id", 0) &&
                domain.at("domain_name") == reaction.value("domain_name", std::string()) &&
                domain.at("domain_channel") == reaction.value("domain_channel", "mixed_contact"));
        }
        if (!found) {
            throw std::invalid_argument("Audio domain binding not found; inspect audio.domains");
        }
    }
    if (params.contains("parent") && !params.at("parent").get<std::string>().empty()) {
        parentMatrix(params);
    }
    syncScene(ctx);
    if (operation == "play") {
        const auto sceneData = ctx.scene.audio->snapshot();
        for (const auto& e : sceneData.at("emitters")) {
            if (e.at("id") == params.at("id")) {
                parentMatrix(e);
            }
        }
    }
    const bool mutation = AudioService::isAuthoring(operation);
    const auto before = mutation ? ctx.scene.audio->snapshot() : nlohmann::json();
    auto result = ctx.scene.audio->invoke(operation, input);
    if (operation == "workspace_update" && input.contains("selected_clip") &&
        !input.at("selected_clip").get<std::string>().empty()) {
        ctx.selection.clearSelection();
        if (ctx.scene_ui_ptr) {
            ctx.scene_ui_ptr->tab_to_focus = "Audio";
        }
    }
    if (operation == "select") {
        ctx.selection.clearSelection();
        auto& item = ctx.selection.selected;
        item.type = SelectableType::AudioSource;
        item.audio_source_id = result.at("id");
        item.name = result.at("name");
        item.position = Vec3(result.at("position")[0], result.at("position")[1], result.at("position")[2]);
        try {
            const auto pose = worldPose(ctx, result);
            item.position = Vec3(pose.at("position")[0], pose.at("position")[1], pose.at("position")[2]);
        } catch (const std::exception&) {
            // Missing parents remain selectable for repair.
        }
        ctx.selection.multi_selection.push_back(item);
        ctx.scene.audio->invoke("workspace_update", {{"selected_clip", ""}});
    }
    syncScene(ctx);
    if (mutation) {
        recordEdit(ctx, history, before, "Audio " + operation);
    }
    return result;
}
}
