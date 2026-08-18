#include "UI/scene_ui_volume_performance.hpp"

#include "scene_ui.h"
#include "Backend/VulkanBackend.h"
#include "Backend/IViewportBackend.h"
#include "GpuFoliageScatter.h"
#include "InstanceManager.h"
#include "InstanceGroup.h"
#include "Triangle.h"
#include "ui_modern.h"
#include "imgui.h"

#include <cstdio>
#include <memory>
#include <string>
#include <unordered_set>

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

namespace {

VulkanRT::VolumePerformanceStats g_cachedStats{};
bool g_hasCachedStats = false;
bool g_instrumentationEnabled = false;

float safeRatio(uint32_t numerator, uint32_t denominator) {
    return denominator ? float(numerator) / float(denominator) : 0.0f;
}

Backend::VulkanBackendAdapter* activeVulkanBackend(UIContext& ctx) {
    if (auto* render = dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr)) {
        return render;
    }
    // Solid/Material Preview is owned by the Vulkan viewport even when the
    // dormant Rendered backend is CPU or OptiX.
    return dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get());
}

std::string buildMetricsReport(const VulkanRT::VolumePerformanceStats& stats) {
    const uint32_t traversalWork =
        stats.densitySamples + stats.emptySegmentsSkipped;
    const uint32_t temporalTotal =
        stats.temporalAccepted + stats.temporalRejected;
    char report[1800];
    std::snprintf(
        report, sizeof(report),
        "RayTrophi Vulkan Volume Metrics v8\n"
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
        "solid_probe_runs: %u\n"
        "solid_probe_hits: %u\n"
        "gas_handoffs: %u\n"
        "layered_handoffs: %u\n"
        "arbiter_gate_open: %u\n"
        "arbiter_candidates: %u\n"
        "arbiter_rejects: %u\n"
        "arbiter_no_box: %u\n"
        "arbiter_empty_range: %u\n"
        "arbiter_no_crossing: %u\n"
        "teleports: %u\n"
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
        // Embedded-solid probe counters (shader-side volumeRecordSolidProbe).
        // Runs==0 means the perf gate suppressed the probe; runs>0 with hits==0
        // means it ran and missed. They were named reserved[0]/reserved[1] and
        // reached only this report — the panel itself never showed them.
        stats.solidProbeRuns,
        stats.solidProbeHits,
        stats.gasHandoffs,
        stats.layeredHandoffs,
        stats.arbiterGateOpen,
        stats.arbiterCandidates,
        stats.arbiterRejects,
        stats.arbiterNoBox,
        stats.arbiterEmptyRange,
        stats.arbiterNoCrossing,
        stats.teleports,
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
        "GPU Performance", ImVec4(0.52f, 0.84f, 1.0f, 1.0f));

    auto* backend = activeVulkanBackend(ctx);
    if (!backend) {
        UIWidgets::StatusIndicator(
            "Available when the Vulkan RT backend is active",
            UIWidgets::StatusType::Warning);
        UIWidgets::EndSection();
        return;
    }

    const auto instanceStats=backend->getInstancePreparationStats();
    UIWidgets::ColoredHeader("RT Instance Preparation",ImVec4(0.58f,0.95f,0.68f,1.0f));
    const bool solidRasterPath = instanceStats.lastPath == 3;
    if (solidRasterPath) {
        UIWidgets::StatusIndicator("Solid raster instance refresh active",
            UIWidgets::StatusType::Success);
    } else {
        UIWidgets::StatusIndicator(
            instanceStats.gpuPathAvailable?"GPU compute pipeline available":"CPU fallback (instance_prepare.spv unavailable)",
            instanceStats.gpuPathAvailable?UIWidgets::StatusType::Success:UIWidgets::StatusType::Warning);
    }
    const char* instancePath=instanceStats.lastPath==1?"GPU TLAS BUILD":instanceStats.lastPath==2?"GPU TLAS UPDATE":instanceStats.lastPath==3?"GPU RASTER":"CPU / unchanged";
    ImGui::Text("Last path: %s",instancePath);
    ImGui::Text("Instances: %u",instanceStats.instanceCount);
    ImGui::Text("CPU prepare: %.3f ms",instanceStats.cpuPrepareMs);
    if (ImGui::CollapsingHeader("Instance Diagnostics")) {
        ImGui::Text("Dirty groups: %u",instanceStats.dirtyGroupCount);
        ImGui::Text("Source upload: %.2f MB",double(instanceStats.uploadBytes)/(1024.0*1024.0));
        ImGui::Text("GPU dispatches: %llu",static_cast<unsigned long long>(instanceStats.gpuDispatches));
        ImGui::Text("CPU fallbacks: %llu",static_cast<unsigned long long>(instanceStats.cpuFallbacks));
        if(ImGui::Button("Copy Instance Metrics")){
            char report[700];
            std::snprintf(report,sizeof(report),
                "RayTrophi Vulkan Instance Metrics v2\nGPU RT prepare available: %s\nLast path: %s\nInstances: %u\nDirty groups: %u\nUpload bytes: %llu\nCPU prepare ms: %.4f\nGPU dispatches: %llu\nCPU fallbacks: %llu\n",
                instanceStats.gpuPathAvailable?"yes":"no",instancePath,
                instanceStats.instanceCount,instanceStats.dirtyGroupCount,
                static_cast<unsigned long long>(instanceStats.uploadBytes),instanceStats.cpuPrepareMs,
                static_cast<unsigned long long>(instanceStats.gpuDispatches),static_cast<unsigned long long>(instanceStats.cpuFallbacks));
            ImGui::SetClipboardText(report);
        }
        UIWidgets::HelpMarker("GPU instance preparation supports scatter topology BUILD and transform UPDATE. Unsupported edits retain the conservative CPU fallback.");
    }

    uint64_t foliageLogical = 0;
    uint64_t foliageRecords = 0;
    for (const auto& group : InstanceManager::getInstance().getGroups()) {
        if (group.point_sphere_mode) continue;
        foliageLogical += group.instances.size();
        std::vector<uint32_t> recordsPerSource(group.sources.size(), 0u);
        for (size_t si = 0; si < group.sources.size(); ++si) {
            const auto& source = group.sources[si];
            const auto* centered = source.centered_triangles_ptr ? source.centered_triangles_ptr.get() : nullptr;
            const auto* tris = (centered && !centered->empty()) ? centered : &source.triangles;
            std::unordered_set<std::string> parts;
            if (tris) for (const auto& tri : *tris) if (tri) {
                parts.insert(tri->getNodeName() + "#" + std::to_string(tri->getMaterialID()));
            }
            recordsPerSource[si] = static_cast<uint32_t>((std::max)(size_t(1), parts.size()));
        }
        for (const auto& inst : group.instances) {
            int si = inst.source_index;
            if (si < 0 || si >= static_cast<int>(recordsPerSource.size())) si = 0;
            if (si < static_cast<int>(recordsPerSource.size())) foliageRecords += recordsPerSource[si];
        }
    }
    constexpr uint64_t kRtInstanceBytes = 64;
    constexpr uint64_t kRtMetadataBytes = sizeof(VulkanRT::VkInstanceData);
    constexpr uint64_t kPrepareSourceBytes = sizeof(VulkanRT::GpuTLASInstanceSource);
    constexpr uint64_t kRasterMatrixBytes = 64;
    const uint64_t rtBytes = foliageRecords * (kRtInstanceBytes + kRtMetadataBytes + kPrepareSourceBytes);
    const uint64_t rasterBytes = foliageRecords * kRasterMatrixBytes;
    UIWidgets::ColoredHeader("Foliage VRAM Estimate", ImVec4(0.60f,0.88f,1.0f,1.0f));
    ImGui::Text("Logical instances: %llu", static_cast<unsigned long long>(foliageLogical));
    ImGui::Text("Combined active estimate: %.2f MiB", double(rtBytes + rasterBytes) / (1024.0 * 1024.0));
    if (ImGui::CollapsingHeader("Foliage Memory Details")) {
        ImGui::Text("GPU records (material parts): %llu", static_cast<unsigned long long>(foliageRecords));
        ImGui::Text("Vulkan RT records: %.2f MiB", double(rtBytes) / (1024.0 * 1024.0));
        ImGui::Text("Solid matrix records: %.2f MiB", double(rasterBytes) / (1024.0 * 1024.0));
        UIWidgets::HelpMarker("Incremental foliage cost only: 160 bytes per Vulkan RT record and 64 bytes per Solid record. Shared source BLAS geometry and driver-dependent TLAS storage are excluded. Terrain height/mask candidate buffers are transient and released after scatter.");
        if (ImGui::Button("Copy Foliage VRAM")) {
        char report[600];
        std::snprintf(report, sizeof(report),
            "RayTrophi Foliage VRAM Estimate v1\nLogical instances: %llu\nGPU records: %llu\nVulkan RT bytes: %llu\nSolid bytes: %llu\nCombined bytes: %llu\nPer RT record: 160\nPer Solid record: 64\nShared BLAS: excluded/reused\nScatter candidate buffers: transient\n",
            static_cast<unsigned long long>(foliageLogical), static_cast<unsigned long long>(foliageRecords),
            static_cast<unsigned long long>(rtBytes), static_cast<unsigned long long>(rasterBytes),
            static_cast<unsigned long long>(rtBytes + rasterBytes));
            ImGui::SetClipboardText(report);
        }
    }

    const auto& terrainScatter = FoliageGPU::getLastTerrainScatterStats();
    if (terrainScatter.candidateCount > 0)
        ImGui::Text("Last scatter: %s / %.2f ms", terrainScatter.gpuPathUsed ? "GPU" : "CPU",
            terrainScatter.gpuMs + terrainScatter.cpuCompactMs);

    if (ImGui::CollapsingHeader("Developer Tests")) {
    UIWidgets::ColoredHeader("GPU Foliage Scatter Parity",ImVec4(0.83f,0.72f,1.0f,1.0f));
    if(ImGui::Button("Run 1M Scatter Parity Test")) {
        FoliageGPU::runParityTest(1000000u);
    }
    const auto& parity = FoliageGPU::getLastParityStats();
    if(parity.candidateCount > 0) {
        const bool passed = parity.completed && parity.rngMismatches == 0 &&
            parity.acceptanceMismatches == 0 && parity.rejectionMaskMismatches == 0;
        UIWidgets::StatusIndicator(
            passed ? "PASS: CPU/GPU scatter contract matches" :
            (parity.gpuAvailable ? "FAIL: parity mismatch or dispatch failure" : "GPU compute unavailable"),
            passed ? UIWidgets::StatusType::Success : UIWidgets::StatusType::Warning);
        ImGui::Text("Candidates: %u", parity.candidateCount);
        ImGui::Text("RNG mismatches: %u", parity.rngMismatches);
        ImGui::Text("Acceptance mismatches: %u", parity.acceptanceMismatches);
        ImGui::Text("Rejection-mask mismatches: %u", parity.rejectionMaskMismatches);
        ImGui::Text("CPU reference: %.3f ms", parity.cpuMs);
        ImGui::Text("GPU upload+dispatch+readback: %.3f ms", parity.gpuDispatchReadbackMs);
        if(ImGui::Button("Copy Scatter Parity")) {
            char report[600];
            std::snprintf(report,sizeof(report),
                "RayTrophi GPU Foliage Scatter Parity v1\nGPU available: %s\nCompleted: %s\nCandidates: %u\nRNG mismatches: %u\nAcceptance mismatches: %u\nRejection-mask mismatches: %u\nCPU ms: %.4f\nGPU upload+dispatch+readback ms: %.4f\n",
                parity.gpuAvailable?"yes":"no",parity.completed?"yes":"no",parity.candidateCount,
                parity.rngMismatches,parity.acceptanceMismatches,parity.rejectionMaskMismatches,
                parity.cpuMs,parity.gpuDispatchReadbackMs);
            ImGui::SetClipboardText(report);
        }
    }
    if (terrainScatter.candidateCount > 0 || terrainScatter.gpuAvailable) {
        ImGui::Spacing();
        ImGui::TextUnformatted("Last Terrain Fill");
        ImGui::Text("Path: %s", terrainScatter.gpuPathUsed ? "gpu" : "cpu fallback");
        ImGui::Text("Candidates: %u", terrainScatter.candidateCount);
        ImGui::Text("Accepted before spacing: %u", terrainScatter.acceptedBeforeSpacing);
        ImGui::Text("Spawned: %u", terrainScatter.spawned);
        ImGui::Text("GPU upload+dispatch+readback: %.3f ms", terrainScatter.gpuMs);
        ImGui::Text("CPU spacing+transform compact: %.3f ms", terrainScatter.cpuCompactMs);
        ImGui::Text("Upload: %.2f MiB", static_cast<double>(terrainScatter.uploadBytes) / (1024.0 * 1024.0));
        if (ImGui::Button("Copy Terrain Scatter Metrics")) {
            char report[700];
            std::snprintf(report, sizeof(report),
                "RayTrophi GPU Terrain Scatter Metrics v1\nGPU available: %s\nPath: %s\nCandidates: %u\nAccepted before spacing: %u\nSpawned: %u\nUpload bytes: %llu\nGPU upload+dispatch+readback ms: %.4f\nCPU spacing+transform compact ms: %.4f\n",
                terrainScatter.gpuAvailable ? "yes" : "no",
                terrainScatter.gpuPathUsed ? "gpu" : "cpu fallback",
                terrainScatter.candidateCount, terrainScatter.acceptedBeforeSpacing,
                terrainScatter.spawned,
                static_cast<unsigned long long>(terrainScatter.uploadBytes),
                terrainScatter.gpuMs, terrainScatter.cpuCompactMs);
            ImGui::SetClipboardText(report);
        }
    }
    }
    ImGui::Separator();

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
    ImGui::Text("Solid probes run: %u", g_cachedStats.solidProbeRuns);
    ImGui::Text("Solid probes hit: %u", g_cachedStats.solidProbeHits);
    ImGui::Text("Gas handoffs: %u", g_cachedStats.gasHandoffs);
    ImGui::Text("Layered handoffs: %u", g_cachedStats.layeredHandoffs);
    ImGui::Text("Arbiter gate opened: %u", g_cachedStats.arbiterGateOpen);
    ImGui::Text("Arbiter candidates seen: %u", g_cachedStats.arbiterCandidates);
    if (g_cachedStats.arbiterGateOpen == 0u && g_cachedStats.volumeRays > 0u) {
        ImGui::TextDisabled("  (gate NEVER opened - this volume reads itself as\n                             liquid/cloud: customIndex vs SSBO order mismatch)");
    }
    ImGui::Text("Arbiter rejects: %u", g_cachedStats.arbiterRejects);
    ImGui::Text("Teleports past box: %u", g_cachedStats.teleports);
    if (g_cachedStats.teleports > g_cachedStats.volumeRays / 4u &&
        g_cachedStats.volumeRays > 0u) {
        ImGui::TextDisabled("  (a quarter of gas rays jump past everything "
                            "inside the box — the black-band signature)");
    }
    if (g_cachedStats.solidProbeRuns == 0 && g_cachedStats.volumeRays > 0) {
        ImGui::TextDisabled("  (gate suppressed every probe — surfaces inside a "
                            "volume cannot be found)");
    }

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
