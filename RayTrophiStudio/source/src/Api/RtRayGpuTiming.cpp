#include "RtApiInternal.h"
#include "Api/RayGpuTimingApi.h"
#include "RtRayGpuTimingBindings.h"
#include "Backend/VulkanBackend.h"
#include "imgui.h"
#include <pybind11/pybind11.h>
#include <algorithm>
#include <stdexcept>

namespace {
Backend::VulkanBackendAdapter* timingBackend() {
    if (!rtapi::g_ctx) {
        return nullptr;
    }
    auto* backend = dynamic_cast<Backend::VulkanBackendAdapter*>(rtapi::g_ctx->backend_ptr);
    return backend && backend->getVulkanDevice() && backend->getVulkanDevice()->isInitialized()
        ? backend : nullptr;
}
}

namespace rtapi {
Result setRayGpuTimingCapture(bool enabled) {
    auto* backend = timingBackend();
    if (!backend) {
        return Result::fail("GPU RT timing requires an initialized Vulkan render backend");
    }
    std::string error;
    return backend->setRayGpuTimingCapture(enabled, error) ? Result::success() : Result::fail(error);
}

bool rayGpuTimingCaptureEnabled() {
    auto* backend = timingBackend();
    return backend && backend->rayGpuTimingCaptureEnabled();
}

nlohmann::json rayGpuTimingCaptureStats() {
    auto* backend = timingBackend();
    if (!backend) {
        return {{"available", false}, {"reason", "Vulkan render device is unavailable"}};
    }
    const auto stats = backend->rayGpuTimingCaptureStats();
    nlohmann::json rows = nlohmann::json::array();
    double sum = 0.0;
    double maximum = 0.0;
    for (const auto& sample : stats.samples) {
        rows.push_back({{"capture_index", sample.captureIndex},
            {"sample_index", sample.sampleIndex}, {"width", sample.width},
            {"height", sample.height}, {"gpu_ms", sample.gpuMs}});
        sum += sample.gpuMs;
        maximum = std::max(maximum, sample.gpuMs);
    }
    return {{"available", true}, {"enabled", stats.enabled},
        {"capturing", stats.capturing}, {"capacity", VulkanRT::VulkanRayGpuTiming::capacity},
        {"scope", "camera_rt_dispatch"}, {"unit", "milliseconds"},
        {"support", stats.supported ? "supported" : "not_initialized"},
        {"recorded", stats.recorded}, {"pending", stats.pending},
        {"failed", stats.failed}, {"error", stats.error}, {"samples", rows},
        {"measured", !rows.empty()}, {"gpu_sum_ms", sum},
        {"gpu_mean_ms", rows.empty() ? nlohmann::json(nullptr) :
            nlohmann::json(sum / static_cast<double>(rows.size()))},
        {"gpu_max_ms", rows.empty() ? nlohmann::json(nullptr) : nlohmann::json(maximum)},
        {"sync_submit_host_waits", stats.hostWaits},
        {"sync_submit_host_wait_measured", stats.hostWaits > 0},
        {"sync_submit_host_wait_ms", stats.hostWaitMs},
        {"sync_submit_host_wait_max_ms", stats.maxHostWaitMs},
        {"async_slot_host_wait_measured", stats.asyncSlotWaits > 0},
        {"async_slot_host_waits", stats.asyncSlotWaits},
        {"async_slot_host_wait_ms", stats.asyncSlotWaitMs},
        {"async_slot_host_wait_max_ms", stats.maxAsyncSlotWaitMs}};
}
}

void bindRayGpuTimingPython(pybind11::module_& render) {
    namespace py = pybind11;
    render.def("set_gpu_timing", [](bool enabled) {
        const auto result = rtapi::setRayGpuTimingCapture(enabled);
        if (!result.ok) {
            throw std::runtime_error(result.error);
        }
    }, py::arg("enabled"));
    render.def("gpu_timing_stats", [] {
        return py::module_::import("json").attr("loads")(
            rtapi::rayGpuTimingCaptureStats().dump());
    });
}

void drawRayGpuTimingCapture() {
    static nlohmann::json cached;
    static std::string error;
    bool enabled = rtapi::rayGpuTimingCaptureEnabled();
    ImGui::Separator();
    ImGui::TextUnformatted("RT GPU timestamp capture");
    if (ImGui::Checkbox("Capture next 64 RT dispatches", &enabled)) {
        const auto result = rtapi::setRayGpuTimingCapture(enabled);
        error = result.ok ? "" : result.error;
        if (result.ok) {
            cached = nlohmann::json();
        }
    }
    ImGui::TextDisabled("Off: no timing commands or automatic readback. Setup waits once.");
    ImGui::TextDisabled("Arm, reset accumulation, then refresh. Keep counters off for timings.");
    if (ImGui::Button("Refresh RT GPU timings")) {
        cached = rtapi::rayGpuTimingCaptureStats();
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    if (!cached.is_object() || !cached.value("available", false)) {
        return;
    }
    ImGui::Text("Recorded %u / pending %u / failed %u",
        cached.value("recorded", 0u), cached.value("pending", 0u), cached.value("failed", 0u));
    const std::string captureError = cached.value("error", std::string());
    if (!captureError.empty()) {
        ImGui::TextWrapped("%s", captureError.c_str());
    }
    if (cached.value("measured", false)) {
        ImGui::Text("RT GPU mean %.3f ms / max %.3f ms",
            cached["gpu_mean_ms"].get<double>(), cached["gpu_max_ms"].get<double>());
        ImGui::Text("Synchronous submit host wait: %.3f ms",
            cached["sync_submit_host_wait_ms"].get<double>());
        ImGui::Text("Async frame-slot host wait: %.3f ms",
            cached["async_slot_host_wait_ms"].get<double>());
        float timings[VulkanRT::VulkanRayGpuTiming::capacity]{};
        int count = 0;
        for (const auto& row : cached["samples"]) {
            timings[count++] = row["gpu_ms"].get<float>();
        }
        ImGui::PlotLines("GPU ms / dispatch", timings, count, 0, nullptr, 0.0f,
            static_cast<float>(cached["gpu_max_ms"].get<double>()), ImVec2(0, 80));
    }
    if (ImGui::Button("Copy RT GPU timing report")) {
        ImGui::SetClipboardText(cached.dump(2).c_str());
    }
    ImGui::TextDisabled("Total camera dispatch, not separate solid/shadow shader times.");
}
