// Standalone C++17 regression. No renderer, GPU or Vec3 linkage is needed.
#include "VolumeRampPacking.h"

#include <array>
#include <cassert>
#include <vector>

struct Color {
    float x, y, z;
};

struct Stop {
    float position;
    Color color;
};

struct Ramp {
    bool enabled = true;
    std::vector<Stop> stops;
};

struct Payload {
    int color_ramp_enabled = 0;
    int ramp_stop_count = 0;
    float ramp_positions[8] = {};
    float ramp_colors_r[8] = {};
    float ramp_colors_g[8] = {};
    float ramp_colors_b[8] = {};
};

int main() {
    // All supported-size ramps retain exact authored values, including HDR.
    for (int count = 0; count <= 8; ++count) {
        Ramp ramp;
        for (int i = 0; i < count; ++i) {
            ramp.stops.push_back({i / 8.0f, {i * 2.0f, i * 0.1f, i * 0.2f}});
        }
        Payload packed;
        volume_material::packColorRamp(ramp, packed);
        assert(packed.color_ramp_enabled == 1);
        assert(packed.ramp_stop_count == count);
        for (int i = 0; i < count; ++i) {
            assert(packed.ramp_positions[i] == ramp.stops[i].position);
            assert(packed.ramp_colors_r[i] == ramp.stops[i].color.x);
            assert(packed.ramp_colors_g[i] == ramp.stops[i].color.y);
            assert(packed.ramp_colors_b[i] == ramp.stops[i].color.z);
        }
    }

    // Previously the spike at stop 9 and hot endpoint were discarded entirely.
    Ramp ramp;
    for (int i = 0; i <= 10; ++i) {
        ramp.stops.push_back({i / 10.0f, {0.0f, 0.0f, 0.0f}});
    }
    ramp.stops[8].color = {4.0f, 0.0f, 0.0f};
    ramp.stops.back().color = {1.0f, 1.0f, 0.8f};
    const auto authored = ramp.stops;
    Payload packed;
    volume_material::packColorRamp(ramp, packed);
    assert(packed.ramp_stop_count == 8);
    assert(packed.ramp_positions[0] == 0.0f);
    assert(packed.ramp_positions[7] == 1.0f);
    assert(packed.ramp_colors_g[7] == 1.0f);
    bool retainedSpike = false;
    for (int i = 1; i < 8; ++i) {
        assert(packed.ramp_positions[i] > packed.ramp_positions[i - 1]);
        retainedSpike |= packed.ramp_positions[i] == 0.8f &&
                         packed.ramp_colors_r[i] == 4.0f;
    }
    assert(retainedSpike);
    assert(ramp.stops.size() == authored.size());
    for (std::size_t i = 0; i < authored.size(); ++i) {
        assert(ramp.stops[i].position == authored[i].position);
        assert(ramp.stops[i].color.x == authored[i].color.x);
    }

    // Unsorted oversized input is packed deterministically without editing it.
    std::reverse(ramp.stops.begin(), ramp.stops.end());
    Payload reversed;
    volume_material::packColorRamp(ramp, reversed);
    for (int i = 0; i < 8; ++i) {
        assert(reversed.ramp_positions[i] == packed.ramp_positions[i]);
        assert(reversed.ramp_colors_r[i] == packed.ramp_colors_r[i]);
    }
    ramp.enabled = false;
    volume_material::packColorRamp(ramp, packed);
    assert(packed.color_ramp_enabled == 0);
    return 0;
}
