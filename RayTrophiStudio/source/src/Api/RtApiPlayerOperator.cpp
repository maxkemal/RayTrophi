#include "Api/RtApiPlayerOperator.h"
#include "Core/PlayerOperator.h"
#include "Core/RuntimeProjectHost.h"
#include "RtApiInternal.h"

namespace rtapi {
Result playerOperator(const std::string &operation, const nlohmann::json &config,
                      nlohmann::json &output) {
    if (!g_ctx) {
        return notBound();
    }
    nlohmann::json display;
    const auto available = AppRuntime::runtimeDisplayStatus(display);
    if (!available.ok) {
        return Result::fail("operator console requires the Player display host");
    }
    if (renderJobActive() && operation != "status") {
        return Result::fail("scene is locked by the final render job");
    }
    return AppRuntime::playerOperatorCommand(*g_ctx, operation, config, output);
}
} // namespace rtapi
