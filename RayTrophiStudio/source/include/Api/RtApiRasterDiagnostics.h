#pragma once

namespace rtapi {
struct RasterDepthPrepassInfo {
    bool enabled = false; // current request
    bool forced_by_rt_shadow = false; // last recorded raster frame
    bool effective = false; // last recorded raster frame, not a prediction
    bool observed = false; // false after reset, before a new timing sample
};
RasterDepthPrepassInfo rasterDepthPrepassStatus();
}
