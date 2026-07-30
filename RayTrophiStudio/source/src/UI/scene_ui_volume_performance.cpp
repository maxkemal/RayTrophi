#include "UI/scene_ui_volume_performance.hpp"

#include "scene_ui.h"
#include "Backend/VulkanBackend.h"
#include "ui_modern.h"
#include "imgui.h"

#include <cstdio>
#include <string>

namespace {

VulkanRT::VolumePerformanceStats g_cachedStats{};
bool g_hasCachedStats = false;
bool g_instrumentationEnabled = false;

float safeRatio(uint32_t numerator, uint32_t denominator) {
    return denominator ? float(numerator) / float(denominator) : 0.0f;
}

Backend::VulkanBackendAdapter* activeVulkanBackend(UIContext& ctx) {
    return dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr);
}

std::string buildMetricsReport(const VulkanRT::VolumePerformanceStats& stats) {
    const uint32_t traversalWork =
        stats.densitySamples + stats.emptySegmentsSkipped;
    const uint32_t temporalTotal =
        stats.temporalAccepted + stats.temporalRejected;
    char report[1400];
    std::snprintf(
        report, sizeof(report),
        "RayTrophi Vulkan Volume Metrics v6\n"
        "volume_rays: %u\n"
        "density_samples: %u\n"
        "shadow_density_samples: %u\n"
        "empty_segments_skipped: %u\n"
        "topology_segments_skipped: %u\n"
        "majorant_segments_skipped: %u\n"
        "extinction_terminations: %u\n"
        "step_budget_exhausted: %u\n"
        "completed_intervals: %u\n"
        "temporal_accepted: %u\n"
        "temporal_rejected: %u\n"
        "majorant_queries: %u\n"
        "majorant_available_queries: %u\n"
        "density_samples_per_ray: %.4f\n"
        "shadow_samples_per_ray: %.4f\n"
        "empty_space_skip_ratio_percent: %.3f\n"
        "topology_share_of_skips_percent: %.3f\n"
        "majorant_share_of_skips_percent: %.3f\n"
        "majorant_availability_percent: %.3f\n"
        "extinction_termination_ratio_percent: %.3f\n"
        "step_budget_exhausted_ratio_percent: %.3f\n"
        "temporal_acceptance_percent: %.3f\n",
        stats.volumeRays,
        stats.densitySamples,
        stats.shadowDensitySamples,
        stats.emptySegmentsSkipped,
        stats.topologySegmentsSkipped,
        stats.majorantSegmentsSkipped,
        stats.extinctionTerminations,
        stats.stepBudgetExhausted,
        stats.completedIntervals,
        stats.temporalAccepted,
        stats.temporalRejected,
        stats.majorantQueries,
        stats.majorantAvailableQueries,
        safeRatio(stats.densitySamples, stats.volumeRays),
        safeRatio(stats.shadowDensitySamples, stats.volumeRays),
        100.0f * safeRatio(stats.emptySegmentsSkipped, traversalWork),
        100.0f * safeRatio(stats.topologySegmentsSkipped, stats.emptySegmentsSkipped),
        100.0f * safeRatio(stats.majorantSegmentsSkipped, stats.emptySegmentsSkipped),
        100.0f * safeRatio(stats.majorantAvailableQueries, stats.majorantQueries),
        100.0f * safeRatio(stats.extinctionTerminations, stats.volumeRays),
        100.0f * safeRatio(stats.stepBudgetExhausted, stats.volumeRays),
        100.0f * safeRatio(stats.temporalAccepted, temporalTotal));
    return report;
}

} // namespace

bool GetCachedVolumePerformanceStats(VulkanRT::VolumePerformanceStats& stats) {
    if (!g_hasCachedStats) {
        return false;
    }
    stats = g_cachedStats;
    return true;
}

void DrawVolumePerformancePanel(UIContext& ctx) {
    if (!UIWidgets::BeginSection(
            "Performance", ImVec4(0.40f, 0.82f, 0.96f, 1.0f))) {
        return;
    }

    UIWidgets::ColoredHeader(
        "Volumetrics (Vulkan RT)", ImVec4(0.52f, 0.84f, 1.0f, 1.0f));

    auto* backend = activeVulkanBackend(ctx);
    if (!backend) {
        UIWidgets::StatusIndicator(
            "Available when the Vulkan RT backend is active",
            UIWidgets::StatusType::Warning);
        UIWidgets::EndSection();
        return;
    }

    if (ImGui::Checkbox("Enable GPU Counters", &g_instrumentationEnabled)) {
        backend->resetVolumePerformanceStats(g_instrumentationEnabled);
        g_cachedStats = {};
        g_hasCachedStats = false;
    }
    UIWidgets::HelpMarker(
        "Enable while collecting a metrics report. Disable for uncontaminated "
        "frame-time/FPS measurements because GPU atomics have a small cost.");

    if (!g_instrumentationEnabled) {
        ImGui::TextDisabled("GPU counters disabled; cached overlay remains available.");
    }

    if (!g_instrumentationEnabled) ImGui::BeginDisabled();
    if (ImGui::Button("Refresh Metrics")) {
        g_cachedStats = backend->getVolumePerformanceStats(true);
        g_hasCachedStats = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Counters")) {
        backend->resetVolumePerformanceStats(true);
        g_cachedStats = {};
        g_hasCachedStats = true;
    }
    if (!g_instrumentationEnabled) ImGui::EndDisabled();
    if (g_hasCachedStats) {
        if (ImGui::Button("Copy Metrics Report")) {
            const std::string report = buildMetricsReport(g_cachedStats);
            ImGui::SetClipboardText(report.c_str());
        }
    }
    UIWidgets::HelpMarker(
        "Refresh takes a synchronized GPU snapshot. Counters accumulate until "
        "Reset Counters is pressed; normal rendering has no CPU readback stall. "
        "Copy Metrics Report creates a paste-ready text report.");

    if (!g_hasCachedStats) {
        ImGui::TextDisabled("Press Refresh Metrics to capture the first snapshot.");
        UIWidgets::EndSection();
        return;
    }

    const uint32_t traversalWork =
        g_cachedStats.densitySamples + g_cachedStats.emptySegmentsSkipped;
    const uint32_t temporalTotal =
        g_cachedStats.temporalAccepted + g_cachedStats.temporalRejected;

    ImGui::Separator();
    ImGui::Text("Volume rays: %u", g_cachedStats.volumeRays);
    ImGui::Text("Density samples: %u", g_cachedStats.densitySamples);
    ImGui::Text("Shadow samples: %u", g_cachedStats.shadowDensitySamples);
    ImGui::Text("Skipped empty segments: %u", g_cachedStats.emptySegmentsSkipped);
    ImGui::Text("  Topology skips: %u", g_cachedStats.topologySegmentsSkipped);
    ImGui::Text("  Majorant skips: %u", g_cachedStats.majorantSegmentsSkipped);
    ImGui::Text("  Majorant queries: %u", g_cachedStats.majorantQueries);
    ImGui::Text("  Majorant available: %u", g_cachedStats.majorantAvailableQueries);
    ImGui::Text("Extinction terminations: %u", g_cachedStats.extinctionTerminations);
    ImGui::Text("Step budget exhausted: %u", g_cachedStats.stepBudgetExhausted);
    ImGui::Text("Completed intervals: %u", g_cachedStats.completedIntervals);

    ImGui::Separator();
    ImGui::Text("Density samples / ray: %.2f",
                safeRatio(g_cachedStats.densitySamples, g_cachedStats.volumeRays));
    ImGui::Text("Shadow samples / ray: %.2f",
                safeRatio(g_cachedStats.shadowDensitySamples, g_cachedStats.volumeRays));
    ImGui::Text("Empty-space skip ratio: %.1f%%",
                100.0f * safeRatio(g_cachedStats.emptySegmentsSkipped, traversalWork));
    ImGui::Text("Majorant share of skips: %.1f%%",
                100.0f * safeRatio(
                    g_cachedStats.majorantSegmentsSkipped,
                    g_cachedStats.emptySegmentsSkipped));
    ImGui::Text("Majorant availability: %.1f%%",
                100.0f * safeRatio(
                    g_cachedStats.majorantAvailableQueries,
                    g_cachedStats.majorantQueries));
    ImGui::Text("Extinction termination ratio: %.1f%%",
                100.0f * safeRatio(g_cachedStats.extinctionTerminations, g_cachedStats.volumeRays));
    ImGui::Text("Step budget exhausted ratio: %.1f%%",
                100.0f * safeRatio(g_cachedStats.stepBudgetExhausted, g_cachedStats.volumeRays));
    ImGui::Text("Temporal acceptance: %.1f%%",
                100.0f * safeRatio(g_cachedStats.temporalAccepted, temporalTotal));

    UIWidgets::EndSection();
}
