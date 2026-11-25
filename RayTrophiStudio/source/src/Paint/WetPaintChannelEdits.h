#pragma once

#include "Paint/WetPaintGpu.h"

namespace Paint::WetPaintChannelEdits {

// Bits match the plane indices in the output atlas and the channel shader.
constexpr uint32_t Color = 1u << 0;
constexpr uint32_t Roughness = 1u << 1;
constexpr uint32_t Height = 1u << 2;
constexpr uint32_t Normal = 1u << 3;
constexpr uint32_t DispatchRange = 1u << 5;

inline uint32_t changedPlanes(const WetPaintSurfaceSettings& previous,
                              const WetPaintSurfaceSettings& next) {
    uint32_t planes = 0;
    if (next.roughness_channel &&
        (previous.wet_roughness != next.wet_roughness ||
         previous.dry_roughness != next.dry_roughness)) {
        planes |= Roughness;
    }
    if (next.height_normal_channel) {
        if (previous.height_contribution != next.height_contribution) {
            planes |= Height | Normal;
        }
        if (previous.normal_strength != next.normal_strength) {
            planes |= Normal;
        }
    }
    if (!next.reveal_picture.empty() &&
        (previous.picture_tolerance != next.picture_tolerance ||
         previous.picture_hint != next.picture_hint ||
         previous.picture_outline != next.picture_outline ||
         (next.reveal_picture == "@base" &&
          (previous.picture_blank[0] != next.picture_blank[0] ||
           previous.picture_blank[1] != next.picture_blank[1] ||
           previous.picture_blank[2] != next.picture_blank[2])))) {
        planes |= Color;
    }
    return planes;
}

template <typename Surfaces, typename Constants, typename Dispatch>
void dispatchRanges(const Surfaces& surfaces, Constants& constants, Dispatch dispatch) {
    const auto saved = constants;
    constants._p1 |= DispatchRange;
    for (const auto& surface : surfaces) {
        if (surface->edited_planes == 0 || surface->tiles.empty()) {
            continue;
        }
        constants.source_count = surface->tile_offset;
        constants.source_kind = surface->edited_planes;
        dispatch(static_cast<uint32_t>(surface->tiles.size()),
                 (surface->edited_planes & Height) != 0);
    }
    constants = saved;
}

template <typename Surfaces>
void queuePublication(const Surfaces& surfaces, bool full_pass) {
    for (auto& surface : surfaces) {
        auto& s = *surface;
        if (s.edited_planes == 0 || s.tiles.empty()) {
            continue;
        }
        if (!full_pass) {
            auto full = s.pending[0];
            full = {true, 0, 0, static_cast<int>(s.width) - 1, static_cast<int>(s.height) - 1};
            for (int backend = 0; backend < 2; ++backend) {
                if ((s.edited_planes & Color) != 0) {
                    s.pending[backend] = full;
                }
                for (int channel = 0; channel < 4; ++channel) {
                    if ((s.edited_planes & (1u << (channel + 1))) != 0) {
                        s.channel_pending[backend][channel] = full;
                    }
                }
            }
        }
        s.edited_planes = 0;
    }
}

} // namespace Paint::WetPaintChannelEdits
