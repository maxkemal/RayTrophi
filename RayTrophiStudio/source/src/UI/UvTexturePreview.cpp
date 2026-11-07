// SPDX-License-Identifier: MIT
#include "UI/UvTexturePreview.h"
#include "Texture.h"
#include <SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern SDL_Renderer* renderer;

namespace UvTexturePreview {
namespace {
SDL_Texture* image = nullptr;
SDL_Renderer* owner = nullptr;
std::weak_ptr<Texture> source;
int width = 0;
int height = 0;
bool hdr_format = false;
}

void release() {
    // SDL owns textures after renderer destruction. Do not destroy them twice.
    if (image && owner == renderer) {
        SDL_DestroyTexture(image);
    }
    image = nullptr;
    owner = nullptr;
    source.reset();
    width = height = 0;
}

bool draw(const std::shared_ptr<Texture>& texture, ImVec2 top_left, ImVec2 bottom_right,
          float opacity, bool refresh, std::string& error) {
    if (!texture || !renderer || texture->width <= 0 || texture->height <= 0) {
        error = "Texture preview unavailable: no pixels or UI renderer.";
        return false;
    }
    const size_t count = size_t(texture->width) * texture->height;
    const bool hdr = texture->is_hdr;
    if ((hdr && texture->float_pixels.size() < count) ||
        (!hdr && texture->pixels.size() < count) ||
        texture->width > (std::numeric_limits<int>::max)() / 4) {
        error = "Texture preview unavailable: incomplete pixel buffer.";
        return false;
    }
    const bool changed = source.lock() != texture || width != texture->width ||
        height != texture->height || owner != renderer || hdr_format != hdr;
    if (changed || refresh || !image) {
        release();
        SDL_RendererInfo info{};
        if (SDL_GetRendererInfo(renderer, &info) != 0 ||
            (info.max_texture_width > 0 && texture->width > info.max_texture_width) ||
            (info.max_texture_height > 0 && texture->height > info.max_texture_height)) {
            error = "Texture exceeds the UI renderer's native resolution limit.";
            return false;
        }
        image = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
            SDL_TEXTUREACCESS_STATIC, texture->width, texture->height);
        owner = renderer;
        if (!image) {
            error = std::string("Texture preview allocation failed: ") + SDL_GetError();
            return false;
        }
        std::vector<uint8_t> rgba(count * 4);
        auto tone = [](float value) {
            value = std::isfinite(value) ? (std::max)(0.0f, value) : 0;
            return uint8_t(std::clamp(std::pow(value / (1 + value), 1.0f / 2.2f),
                                      0.0f, 1.0f) * 255 + 0.5f);
        };
        for (size_t i = 0; i < count; ++i) {
            if (hdr) {
                const auto& p = texture->float_pixels[i];
                rgba[i * 4] = tone(p.x);
                rgba[i * 4 + 1] = tone(p.y);
                rgba[i * 4 + 2] = tone(p.z);
            } else {
                const auto& p = texture->pixels[i];
                rgba[i * 4] = p.r;
                rgba[i * 4 + 1] = p.g;
                rgba[i * 4 + 2] = p.b;
            }
            // RGB inspection: opacity is the editor overlay control, not source alpha.
            rgba[i * 4 + 3] = 255;
        }
        if (SDL_UpdateTexture(image, nullptr, rgba.data(), texture->width * 4) != 0) {
            error = std::string("Texture preview upload failed: ") + SDL_GetError();
            release();
            return false;
        }
        SDL_SetTextureBlendMode(image, SDL_BLENDMODE_BLEND);
        source = texture;
        width = texture->width;
        height = texture->height;
        hdr_format = hdr;
    }
    ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)image, top_left,
        bottom_right, ImVec2(0, 0), ImVec2(1, 1),
        ImGui::GetColorU32(ImVec4(1, 1, 1, std::clamp(opacity, 0.0f, 1.0f))));
    error.clear();
    return true;
}
}
