#pragma once
#include "Api/RtApiRasterDiagnostics.h"

template<class Dict> Dict rasterDepthPrepassDictionary() {
    const auto status = rtapi::rasterDepthPrepassStatus();
    Dict result;
    result["enabled"] = status.enabled;
    result["forced_by_rt_shadow"] = status.forced_by_rt_shadow;
    result["effective"] = status.effective;
    result["observed"] = status.observed;
    result["effective_scope"] = "last_recorded_raster_frame";
    return result;
}
