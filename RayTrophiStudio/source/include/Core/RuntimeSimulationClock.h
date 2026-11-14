#pragma once

#include <cstdint>
#include <string>

struct SceneData;
namespace Backend {
class IBackend;
}

namespace AppRuntime {

// The one simulation clock both executables drive (docs/dev/PLAYER_GAME_PAINT_ROOM.md
// step 1). It used to live inside SceneUI's draw pass, read ImGui's DeltaTime and
// widget state, and therefore never ran in the Player. Callers now pass every UI
// fact explicitly; the clock owns only simulation work.
struct SimulationClockInput {
    float dt_seconds = 0.0f;       // wall time since the previous tick; clamped to 1/30 s
    int timeline_frame = 0;        // parked playhead in live mode
    int timeline_start = 0;
    int timeline_end = 0;
    bool playing = false;
    bool live_mode = false;        // free run (games, Studio live preview) vs timeline bake/scrub
    bool ui_editing = false;       // Studio: a widget is being dragged, defer cache invalidation
    bool render_owns_sim = false;  // a sequence-render worker is the sole stepper
    // Vulkan RT backend that may still read simulation device buffers. Null for a
    // raster-only viewport, which never samples them.
    Backend::IBackend* render_backend = nullptr;
};

struct SimulationClockResult {
    bool drove = false;            // the simulation timeline was driven this tick
    bool disk_bake_slice = false;  // a cooperative disk bake owned the tick
    bool render_updated = false;   // present a new frame / reset accumulation
    bool rewind_to_start = false;  // caller moves its playhead to timeline_start
    bool pause_requested = false;  // a step error stopped playback
    std::string step_error;
};

struct SimulationClockStats {
    std::uint64_t ticks = 0;          // tick() calls that found a simulation runtime
    std::uint64_t driven_ticks = 0;   // ticks that drove the simulation timeline
    std::uint64_t render_updates = 0;
    std::uint64_t rewinds = 0;
    double driven_seconds = 0.0;      // sum of clamped dt over driven ticks
    float last_dt_seconds = 0.0f;
    bool last_live_mode = false;
    bool last_playing = false;
    std::string last_step_error;
};

class SimulationClock {
public:
    SimulationClockResult tick(SceneData& scene, const SimulationClockInput& input);
    const SimulationClockStats& stats() const {
        return stats_;
    }

private:
    bool previous_playing_ = false;
    bool range_known_ = false;
    int previous_start_ = 0;
    int previous_end_ = 0;
    SimulationClockStats stats_;
};

// Process-wide clock. Studio's SceneUI and the Player loop both tick this one, so
// the IPC runtime summary reports the same counters in either executable.
SimulationClock& simulationClock();

} // namespace AppRuntime
