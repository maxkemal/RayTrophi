#include "Api/RtApiExternalBody.h"
#include "Input/ExternalBodyRig.h"
#include "RtApiInternal.h"
#include <limits>

namespace rtapi {
Result setExternalBodyPose(const nlohmann::json &request) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    try {
        if (!request.is_object() || !request.contains("character") ||
            !request["character"].is_string() || request["character"].get<std::string>().empty() ||
            !request.contains("person_id") || !request["person_id"].is_number_integer() ||
            request["person_id"].get<double>() < 0 ||
            request["person_id"].get<double>() > std::numeric_limits<int>::max()) {
            return Result::fail("external_pose_requires_character_and_int32_person_id");
        }
        const auto config = request.value("config", nlohmann::json::object());
        std::string error;
        return Input::bindExternalBodyRig(g_ctx->scene, request["character"].get<std::string>(),
                                          request["person_id"].get<int>(), config, error)
                   ? Result::success()
                   : Result::fail(error);
    } catch (const std::exception &exception) {
        return Result::fail(std::string("invalid_external_pose_request: ") + exception.what());
    }
}
Result getExternalBodyPoses(nlohmann::json &output) {
    if (!g_ctx) {
        return notBound();
    }
    output = Input::externalBodyRigs(g_ctx->scene);
    return Result::success();
}
Result clearExternalBodyPose(const nlohmann::json &request) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    if (!request.is_object() || !request.contains("character") ||
        !request["character"].is_string()) {
        return Result::fail("external_pose_requires_character");
    }
    std::string error;
    return Input::clearExternalBodyRig(g_ctx->scene, request["character"].get<std::string>(), error)
               ? Result::success()
               : Result::fail(error);
}
} // namespace rtapi
