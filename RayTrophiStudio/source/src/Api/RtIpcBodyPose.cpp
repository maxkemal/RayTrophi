#include "RtIpcBodyPose.h"
#include "Api/RtApi.h"
#include "Api/RtApiBodyPose.h"
#include "Api/RtApiExternalBody.h"
#include "Api/RtApiCharacter.h"

using json = nlohmann::json;

bool dispatchBodyPoseIpc(const std::string &method, const json &params,
                         const RtIpcPaintWetEnqueue &enqueue, json &output) {
    if (method == "anim.delete_character") {
        if (!params.contains("character") || !params["character"].is_string()) {
            output = {{"__error", "character must be a string"}};
            return true;
        }
        const auto character = params.at("character").get<std::string>();
        output = enqueue([character](UIContext&) {
            const auto result = rtapi::deleteCharacter(character);
            return result.ok ? json{{"ok", true}} : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "anim.set_external_pose") {
        const json request = {{"character", params.at("character")},
                              {"person_id", params.at("person_id")},
                              {"config", params.value("config", json::object())}};
        output = enqueue([request](UIContext&) {
            const auto result = rtapi::setExternalBodyPose(request);
            return result.ok ? json{{"ok", true}} : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "anim.external_poses") {
        output = enqueue([](UIContext&) {
            json value;
            const auto result = rtapi::getExternalBodyPoses(value);
            return result.ok ? value : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "anim.clear_external_pose") {
        const json request = {{"character", params.at("character")}};
        output = enqueue([request](UIContext&) {
            const auto result = rtapi::clearExternalBodyPose(request);
            return result.ok ? json{{"ok", true}} : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "input.set_body_poses") {
        const json packet = {{"t", params.at("t")}, {"persons", params.at("persons")}};
        output = enqueue([packet](UIContext &) {
            const auto result = rtapi::setBodyPoses(packet);
            return result.ok ? json{{"ok", true}} : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "input.body_poses") {
        output = enqueue([](UIContext &) {
            json value;
            const auto result = rtapi::getBodyPoses(value);
            return result.ok ? value : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "input.clear_body_poses") {
        output = enqueue([](UIContext &) {
            const auto result = rtapi::clearBodyPoses();
            return result.ok ? json{{"ok", true}} : json{{"__error", result.error}};
        });
        return true;
    }
    return false;
}
