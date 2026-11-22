#pragma once

#include "Core/PlayerBodyOverlay.h"
#include "Paint/HiddenGameObjects.h"

#include <SDL.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdint>
#include <cstring>

namespace AppRuntime {

// Mirrors tools/perception_bridge/video_card.py. Frames travel only in this
// session-local mapping. The pose stream and display controls still use IPC.
#ifdef _WIN32
class PlayerVideoCard {
public:
    ~PlayerVideoCard() {
        closeMapping(); // SDL owns any remaining texture when the renderer shuts down.
    }

    bool poll(bool enabled, SDL_Renderer* renderer) {
        if (!enabled || !renderer) {
            closeMapping();
            return hide();
        }
        const auto now = GetTickCount64();
        if (!pixels_ && now >= retry_at_) {
            retry_at_ = now + 200;
            openMapping();
        }
        if (pixels_) {
            const DWORD lock_result = WaitForSingleObject(lock_, 0);
            if (lock_result == WAIT_OBJECT_0 || lock_result == WAIT_ABANDONED) {
                Frame frame{};
                if (lock_result == WAIT_OBJECT_0) {
                    std::memcpy(&frame, pixels_, sizeof(frame));
                }
                const bool changed = accept(frame, GetTickCount64(), renderer);
                ReleaseMutex(lock_);
                return changed;
            }
        }
        return texture_ && now - shown_.received > kExpiryMs && hide();
    }

    void draw(const PlayerOverlayProjection& projection) const {
        if (!texture_) {
            return;
        }
        const Vec3 anchor = Paint::constrainHiddenPlayerAnchor(
            "video_card", Vec3(shown_.x, 0.0f, shown_.z), 0.25f, shown_.height);
        const float width = shown_.height * shown_.width_px / shown_.height_px;
        const Vec3 half_width = projection.right * (width * 0.5f);
        ImVec2 top_left, bottom_right;
        if (projection.project(anchor + projection.up * shown_.height - half_width, top_left) &&
            projection.project(anchor + half_width, bottom_right)) {
            ImGui::GetBackgroundDrawList()->AddImage(
                (ImTextureID)(std::intptr_t)texture_, top_left, bottom_right,
                ImVec2(0.5f / kMaxSide, 0.5f / kMaxSide),
                ImVec2((shown_.width_px - 0.5f) / kMaxSide,
                       (shown_.height_px - 0.5f) / kMaxSide));
        }
    }

private:
    struct Frame {
        std::uint64_t serial;
        std::uint64_t received;
        float x, z, height;
        std::uint32_t width_px, height_px, magic;
    };
    static_assert(sizeof(Frame) == 40);
    static constexpr std::uint32_t kMagic = 0x31565452;
    static constexpr std::uint32_t kMaxSide = 640;
    static constexpr std::size_t kMappingBytes = sizeof(Frame) + kMaxSide * kMaxSide * 4;
    static constexpr std::uint64_t kExpiryMs = 300;

    void openMapping() {
        lock_ = OpenMutexW(SYNCHRONIZE | MUTEX_MODIFY_STATE, FALSE,
                           L"Local\\RayTrophiVideoCard.v1.lock");
        if (!lock_) {
            return;
        }
        mapping_ = OpenFileMappingW(FILE_MAP_READ, FALSE, L"Local\\RayTrophiVideoCard.v1");
        if (mapping_) {
            pixels_ = static_cast<const unsigned char*>(
                MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, kMappingBytes));
        }
        if (!pixels_) {
            closeMapping();
        }
    }

    void closeMapping() {
        if (pixels_) {
            UnmapViewOfFile(pixels_);
            pixels_ = nullptr;
        }
        if (mapping_) {
            CloseHandle(mapping_);
            mapping_ = nullptr;
        }
        if (lock_) {
            CloseHandle(lock_);
            lock_ = nullptr;
        }
    }

    bool hide() {
        if (!texture_) {
            return false;
        }
        SDL_DestroyTexture(texture_);
        texture_ = nullptr;
        shown_ = {};
        return true;
    }

    bool accept(const Frame& frame, std::uint64_t now, SDL_Renderer* renderer) {
        if (frame.magic != kMagic || frame.received > now || now - frame.received > kExpiryMs ||
            frame.width_px == 0 || frame.height_px == 0 ||
            frame.width_px > kMaxSide || frame.height_px > kMaxSide ||
            !std::isfinite(frame.x) || std::abs(frame.x) > 10000.0f ||
            !std::isfinite(frame.z) || std::abs(frame.z) > 10000.0f ||
            !(frame.height >= 0.3f && frame.height <= 3.2f)) {
            return hide();
        }
        if (texture_ && shown_.serial == frame.serial) {
            return false;
        }
        if (!texture_) {
            texture_ = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
                                         SDL_TEXTUREACCESS_STREAMING,
                                         kMaxSide, kMaxSide);
            if (!texture_) {
                return false;
            }
            if (SDL_SetTextureBlendMode(texture_, SDL_BLENDMODE_BLEND) != 0) {
                hide();
                return true;
            }
        }
        const SDL_Rect rect{0, 0, static_cast<int>(frame.width_px),
                            static_cast<int>(frame.height_px)};
        if (SDL_UpdateTexture(texture_, &rect, pixels_ + sizeof(Frame),
                              static_cast<int>(frame.width_px * 4)) != 0) {
            return hide();
        }
        shown_ = frame;
        return true;
    }

    HANDLE mapping_ = nullptr;
    HANDLE lock_ = nullptr;
    const unsigned char* pixels_ = nullptr;
    std::uint64_t retry_at_ = 0;
    SDL_Texture* texture_ = nullptr;
    Frame shown_{};
};
#else
// The perception video card is a Windows named-mapping feed; no source exists on other platforms.
class PlayerVideoCard {
public:
    bool poll(bool, SDL_Renderer*) { return false; }
    void draw(const PlayerOverlayProjection&) const {}
};
#endif

} // namespace AppRuntime
