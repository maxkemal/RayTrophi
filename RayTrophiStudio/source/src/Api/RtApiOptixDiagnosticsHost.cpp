#include "RtApiInternal.h"

namespace rtapi {
Result optixAccumStatus(OptixAccumStatus& out) {
    out = OptixAccumStatus{};
    if (!g_ctx) {
        return notBound();
    }
    return Result::success();
}
} // namespace rtapi
