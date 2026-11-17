#pragma once

#include <algorithm>
#include <utility>

namespace AppRuntime {

// Resolution belongs to the render target, independently of window/fullscreen
// presentation. F2 and player.set_display render_scale use this same policy.
class PlayerRenderExtent {
public:
    PlayerRenderExtent(int width, int height)
        : width_((std::max)(16, width)), height_((std::max)(16, height)) {
    }

    std::pair<int, int> base() const {
        return {width_, height_};
    }

    std::pair<int, int> scaled(float scale) const {
        return {(std::max)(16, static_cast<int>(width_ * scale + 0.5f)),
                (std::max)(16, static_cast<int>(height_ * scale + 0.5f))};
    }

private:
    int width_;
    int height_;
};

} // namespace AppRuntime
