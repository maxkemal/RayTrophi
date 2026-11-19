#pragma once
// Process-wide counters for the raster preview's sky/IBL bake and the RayFusion
// probe field's invalidations. They live in their own tiny header on purpose:
// VulkanBackend.h fans out to ~96 translation units, and a counter must not
// cost a full rebuild.
//
// WHY: "the bake restarts every frame" and "the probe field is rebuilt every
// frame" look identical from the outside (a flickering shadow) and neither can
// be told apart by looking at the image. A rate that climbs with the frame
// counter is the proof; a rate that stays flat clears the hypothesis.
#include <atomic>
#include <cstdint>

namespace Backend {
struct PreviewBakeCounters {
    // Compute dispatches of world_sky_capture (the 256x128 sky bake).
    std::atomic<uint64_t> sky_captures{0};
    // Full irradiance + prefilter chain regenerations.
    std::atomic<uint64_t> ibl_regenerations{0};
    // Times the probe producer's source image really changed.
    std::atomic<uint64_t> probe_source_changes{0};
    // Whole-field invalidations (every probe slot dropped and re-measured).
    std::atomic<uint64_t> probe_invalidations{0};
    // Shape changes (grid counts/spacing) that rebuilt the slot array.
    std::atomic<uint64_t> probe_reconfigures{0};
    // Soft refreshes (instance placement only): data kept, re-measured in place.
    std::atomic<uint64_t> probe_soft_refreshes{0};
    // Batches the traced producer refused and released for retry (was: filled
    // with unoccluded sky and marked valid).
    std::atomic<uint64_t> trace_deferred{0};
    std::atomic<uint32_t> deferred_streak{0};   // consecutive refusals; reset on a traced batch
    // What differed at each invalidation, accumulated per cause. A single cause
    // that tracks probe_invalidations one-to-one is the culprit.
    std::atomic<uint64_t> cause_sky{0}, cause_geometry{0}, cause_instances{0},
        cause_bounce{0}, cause_producer_switch{0};
    // Last component hashes seen by serviceMaterialPreviewProbeField.
    std::atomic<uint64_t> last_sky{0}, last_geometry{0}, last_instances{0},
        last_bounce{0};
    std::atomic<uint32_t> last_traced{0};
};

inline PreviewBakeCounters& previewBakeCounters() {
    static PreviewBakeCounters counters;
    return counters;
}
} // namespace Backend
