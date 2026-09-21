#include "UI/RigFingerFKUI.h"
#include "Api/RtApi.h"
#include "imgui.h"
#include "scene_ui.h"
#include <cmath>

namespace RigUI {
namespace {

int requestedSide = -1;

} // namespace

void selectRigFingerFKSide(const std::string& side) {
    if (side == "left")
        requestedSide = 0;
    else if (side == "right")
        requestedSide = 1;
    else if (side == "both")
        requestedSide = 2;
}

void drawRigFingerFKControls(UIContext& context, const std::string& character,
                             uint64_t revision) {
    if (requestedSide >= 0)
        ImGui::SetNextItemOpen(true);
    if (!ImGui::CollapsingHeader("Grouped Finger FK"))
        return;

    static int sideIndex = 2;
    static float curl = 0.f;
    static float spread = 0.f;
    static float thumb = 0.f;
    static bool ownsPreview = false;
    static std::string activeCharacter;
    static std::string message;
    const char* sides[] = {"Left", "Right", "Both"};
    if (requestedSide >= 0) {
        sideIndex = requestedSide;
        requestedSide = -1;
    }

    auto report = [&](const rtapi::Result& result) {
        message = result.ok ? "" : result.error;
        return result.ok;
    };
    if (activeCharacter != character) {
        activeCharacter = character;
        curl = 0.f;
        spread = 0.f;
        thumb = 0.f;
        ownsPreview = false;
        message.clear();
    }
    if (!context.scene.rigView.pose.hasPreview)
        ownsPreview = false;
    bool hasLeft = false;
    bool hasRight = false;
    for (const auto& model : context.scene.importedModelContexts) {
        if (model.importName == character) {
            int leftChannels = 0;
            int rightChannels = 0;
            for (const auto& control : model.rigAnatomy.drivenControls) {
                if (control.id == "left_hand.curl" || control.id == "left_hand.spread" ||
                    control.id == "left_hand.thumb")
                    ++leftChannels;
                if (control.id == "right_hand.curl" || control.id == "right_hand.spread" ||
                    control.id == "right_hand.thumb")
                    ++rightChannels;
            }
            hasLeft = leftChannels == 3;
            hasRight = rightChannels == 3;
            break;
        }
    }
    if (!hasLeft && !hasRight) {
        ImGui::TextDisabled("Add hand scalar controls to this rig's control definition.");
        return;
    }
    if (sideIndex == 0 && !hasLeft)
        sideIndex = 1;
    if (sideIndex == 1 && !hasRight)
        sideIndex = 0;
    if (sideIndex == 2 && (!hasLeft || !hasRight))
        sideIndex = hasLeft ? 0 : 1;
    const bool externalPreview = context.scene.rigView.pose.hasPreview && !ownsPreview;
    ImGui::BeginDisabled(externalPreview);
    ImGui::SetNextItemWidth(130.f);
    ImGui::Combo("Hand", &sideIndex, sides, 3);
    ImGui::TextDisabled("Relative gesture; release commits through pose undo / Auto Key.");

    const bool changedCurl = ImGui::SliderFloat("Curl delta", &curl, -1.f, 1.f, "%.2f");
    const bool finishedCurl = ImGui::IsItemDeactivatedAfterEdit();
    const bool changedSpread =
        ImGui::SliderFloat("Spread delta", &spread, -1.f, 1.f, "%.2f");
    const bool finishedSpread = ImGui::IsItemDeactivatedAfterEdit();
    const bool changedThumb =
        ImGui::SliderFloat("Thumb delta", &thumb, -1.f, 1.f, "%.2f");
    const bool finishedThumb = ImGui::IsItemDeactivatedAfterEdit();

    const bool changed = changedCurl || changedSpread || changedThumb;
    if (changed) {
        const bool zero = std::fabs(curl) < 1e-6f && std::fabs(spread) < 1e-6f &&
                          std::fabs(thumb) < 1e-6f;
        if (zero) {
            if (ownsPreview && context.scene.rigView.pose.hasPreview)
                report(rtapi::cancelRigPosePreview(character));
            ownsPreview = false;
        } else {
            const bool wasOwned = ownsPreview;
            nlohmann::json values = nlohmann::json::object();
            const auto add = [&](const std::string& prefix) {
                values[prefix + ".curl"] = curl;
                values[prefix + ".spread"] = spread;
                values[prefix + ".thumb"] = thumb;
            };
            if (hasLeft && (sideIndex == 0 || sideIndex == 2))
                add("left_hand");
            if (hasRight && (sideIndex == 1 || sideIndex == 2))
                add("right_hand");
            ownsPreview = report(
                rtapi::previewRigControlValues(character, values, revision));
            if (!ownsPreview && wasOwned && context.scene.rigView.pose.hasPreview)
                rtapi::cancelRigPosePreview(character);
        }
    }
    if ((finishedCurl || finishedSpread || finishedThumb) &&
        context.scene.rigView.pose.hasPreview && ownsPreview) {
        if (report(rtapi::applyRigPosePreview(character))) {
            curl = 0.f;
            spread = 0.f;
            thumb = 0.f;
            ownsPreview = false;
        }
    }
    ImGui::EndDisabled();
    if (ImGui::Button("Reset gesture")) {
        if (context.scene.rigView.pose.hasPreview && ownsPreview)
            report(rtapi::cancelRigPosePreview(character));
        curl = 0.f;
        spread = 0.f;
        thumb = 0.f;
        ownsPreview = false;
    }
    if (!message.empty())
        ImGui::TextWrapped("%s", message.c_str());
}

} // namespace RigUI
