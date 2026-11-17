#include "Api/RtApiBodyPose.h"
#include "Input/BodyPoseRuntime.h"
#include "Input/ExternalBodyRig.h"
#include "RtApiInternal.h"

namespace rtapi {

Result setBodyPoses(const nlohmann::json &packet) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    std::string error;
    try {
        if (!Input::bodyPoses().set(packet, error)) {
            return Result::fail(error);
        }
        Input::tickExternalBodyRigs(g_ctx->scene);
        return Result::success();
    } catch (const std::exception &exception) {
        return Result::fail(std::string("invalid_body_packet: ") + exception.what());
    }
}

Result getBodyPoses(nlohmann::json &output) {
    if (!g_ctx) {
        return notBound();
    }
    output = Input::bodyPoses().snapshot();
    return Result::success();
}

Result clearBodyPoses() {
    if (!g_ctx) {
        return notBound();
    }
    Input::bodyPoses().clear();
    Input::tickExternalBodyRigs(g_ctx->scene);
    return Result::success();
}

} // namespace rtapi
