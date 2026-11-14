#include "RtApiInternal.h"
#include "Backend/OptixBackend.h"
#include "OptixWrapper.h"
#include "globals.h"

int rtOptixFirstCudaError();
std::string rtOptixFirstCudaErrorSite();

namespace rtapi {
Result optixAccumStatus(OptixAccumStatus& out) {
    out = OptixAccumStatus{};
    if (!g_ctx)
        return notBound();
    auto* ob = dynamic_cast<Backend::OptixBackend*>(g_ctx->backend_ptr);
    if (!ob)
        return Result::success(); // available=false; "OptiX yok" bir HATA degil
    OptixWrapper* w = ob->getOptixWrapper();
    if (!w)
        return Result::success();
    out.available = true;
    out.accumulated_samples = w->getAccumulatedSamples();
    out.wipe_count = w->getAccumWipeCount();
    out.wipe_resolution = w->getAccumWipeResolutionCount();
    out.wipe_camera = w->getAccumWipeCameraCount();
    w->getAccumLastWipeSizes(out.last_wipe_from[0], out.last_wipe_from[1], out.last_wipe_to[0],
                             out.last_wipe_to[1]);
    out.buffer_w_read =
        w->sampleAccumulationW(out.buffer_w_mean, out.buffer_w_max, out.buffer_w_center);
    out.adaptive_sampling = w->getUseAdaptiveSampling();
    out.min_samples = w->getMinSamplesParam();
    out.variance_threshold = w->getVarianceThresholdParam();
    out.samples_per_pixel = w->getSamplesPerPixelParam();
    out.prev_zero_pixels = w->getAccumPrevZeroLastLaunch();
    w->getAccumSampleGeometry(out.read_w, out.read_h, out.image_w, out.image_h);
    out.reset_buffers_calls = w->getResetBuffersCalls();
    out.set_render_params_calls = g_set_render_params_calls.load(std::memory_order_acquire);
    out.set_render_params_reason = g_set_render_params_reason;
    out.first_cuda_error = rtOptixFirstCudaError();
    out.first_cuda_error_site = rtOptixFirstCudaErrorSite();
    return Result::success();
}
} // namespace rtapi
