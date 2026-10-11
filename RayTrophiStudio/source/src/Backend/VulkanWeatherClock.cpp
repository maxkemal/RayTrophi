#include "Backend/WeatherClock.h"
#include "Backend/VulkanBackend.h"

#include <cmath>

namespace Backend {

void VulkanBackendAdapter::setWeatherTime(float time) {
    if (!std::isfinite(time) || m_weatherTime == time) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_weatherTime = time;
    const auto& weather = m_cachedWorld.weather;
    if (!weatherNeedsRendering(weather) || weather.visual_mode == WEATHER_VISUAL_SURFACE_ONLY ||
        weather.intensity <= 0.0f || weather.density <= 0.0f) {
        return;
    }
    if (m_device && m_device->isInitialized() && m_cloudStateSet) {
        CloudParamsGPU params = cloudParamsGPU();
        params.misc[3] += m_cloudRtRendering ? 1.0f : 0.0f;
        m_device->updateCloudRtParams(params);
    }
    resetAccumulation();
}

void syncWeatherClock(IBackend* render, IBackend* viewport, float timelineTime,
                      float previewTime, bool realtimePreview) {
    const float time = realtimePreview ? previewTime : timelineTime;
    if (auto* vulkan = dynamic_cast<VulkanBackendAdapter*>(render)) {
        vulkan->setWeatherTime(time);
    }
    if (viewport != render) {
        if (auto* vulkan = dynamic_cast<VulkanBackendAdapter*>(viewport)) {
            vulkan->setWeatherTime(time);
        }
    }
}

} // namespace Backend
