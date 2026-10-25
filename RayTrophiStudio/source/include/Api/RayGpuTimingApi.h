#pragma once

#include "Api/RtApi.h"
#include "json.hpp"

namespace rtapi {
Result setRayGpuTimingCapture(bool enabled);
nlohmann::json rayGpuTimingCaptureStats();
bool rayGpuTimingCaptureEnabled();
}

// Existing Volume Performance panel integration; no automatic result readback.
void drawRayGpuTimingCapture();
