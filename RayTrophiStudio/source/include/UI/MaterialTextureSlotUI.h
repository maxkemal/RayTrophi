#pragma once

#include "Texture.h"
#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

namespace MaterialTextureSlotUI {

enum class Action { None, Load, Remove, Copy, Paste };

// Bounded, live CPU sampling: no file reloads, renderer-specific handles or
// persistent GPU resources. Paint changes are reflected on the next UI frame.
inline bool preview(const Texture &texture, ImVec2 origin, ImVec2 size, int channel, bool opacity,
                    int samples) {
    if (texture.width <= 0 || texture.height <= 0) {
        return false;
    }
    const size_t count = static_cast<size_t>(texture.width) * texture.height;
    const bool hdr = texture.is_hdr && texture.float_pixels.size() >= count;
    if (!hdr && texture.pixels.size() < count) {
        return false;
    }
    ImDrawList *list = ImGui::GetWindowDrawList();
    for (int y = 0; y < samples; ++y) {
        const int sy = (std::min)(texture.height - 1, y * texture.height / samples);
        for (int x = 0; x < samples; ++x) {
            const int sx = (std::min)(texture.width - 1, x * texture.width / samples);
            const size_t index = static_cast<size_t>(sy) * texture.width + sx;
            float rgba[4];
            if (hdr) {
                const auto &p = texture.float_pixels[index];
                rgba[0] = p.x;
                rgba[1] = p.y;
                rgba[2] = p.z;
                rgba[3] = p.w;
            } else {
                const auto &p = texture.pixels[index];
                rgba[0] = p.r / 255.0f;
                rgba[1] = p.g / 255.0f;
                rgba[2] = p.b / 255.0f;
                rgba[3] = p.a / 255.0f;
            }
            if (channel >= 0) {
                rgba[0] = rgba[1] = rgba[2] = rgba[channel];
            }
            if (opacity) {
                const ImVec4 checker = ImGui::GetStyleColorVec4(
                    ((x / 4 + y / 4) % 2) ? ImGuiCol_FrameBg : ImGuiCol_Button);
                const float alpha = std::clamp(rgba[3], 0.0f, 1.0f);
                rgba[0] = rgba[0] * alpha + checker.x * (1.0f - alpha);
                rgba[1] = rgba[1] * alpha + checker.y * (1.0f - alpha);
                rgba[2] = rgba[2] * alpha + checker.z * (1.0f - alpha);
            }
            for (int c = 0; c < 3; ++c) {
                if (!std::isfinite(rgba[c])) {
                    rgba[c] = 0.0f;
                }
                if (hdr) {
                    const float v = (std::max)(0.0f, rgba[c]);
                    rgba[c] = std::pow(v / (1.0f + v), 1.0f / 2.2f);
                }
            }
            const ImVec2 a(origin.x + x * size.x / samples, origin.y + y * size.y / samples);
            const ImVec2 b(origin.x + (x + 1) * size.x / samples,
                           origin.y + (y + 1) * size.y / samples);
            list->AddRectFilled(
                a, b, ImGui::ColorConvertFloat4ToU32(ImVec4(rgba[0], rgba[1], rgba[2], 1.0f)));
        }
    }
    return true;
}

inline Action draw(const char *label, const std::shared_ptr<Texture> &texture, bool canPaste,
                   bool paint, const std::string &source, const std::string &cache,
                   int channel = -1, bool opacity = false) {
    Action action = Action::None;
    const bool loaded = texture && texture->is_loaded();
    const float edge = ImGui::GetFontSize() * 3.5f;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::Button(loaded ? "##preview" : "+\nEmpty", ImVec2(edge, edge));
    if (loaded) {
        const float ratio =
            static_cast<float>((std::max)(1, texture->width)) / (std::max)(1, texture->height);
        ImVec2 fit(edge - 4.0f, edge - 4.0f);
        if (ratio > 1.0f) {
            fit.y /= ratio;
        } else {
            fit.x *= ratio;
        }
        preview(*texture,
                ImVec2(origin.x + (edge - fit.x) * 0.5f, origin.y + (edge - fit.y) * 0.5f), fit,
                channel, opacity, 24);
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(label);
        if (loaded) {
            const float limit = (std::min)(256.0f, ImGui::GetIO().DisplaySize.x * 0.65f);
            ImVec2 size(limit, limit);
            if (texture->width > texture->height) {
                size.y *= static_cast<float>(texture->height) / texture->width;
            } else if (texture->height > 0) {
                size.x *= static_cast<float>(texture->width) / texture->height;
            }
            const ImVec2 pos = ImGui::GetCursorScreenPos();
            if (!preview(*texture, pos, size, channel, opacity, 80)) {
                ImGui::TextDisabled("Preview unavailable");
            } else {
                ImGui::Dummy(size);
            }
            ImGui::Text("%d x %d", texture->width, texture->height);
            if (channel >= 0) {
                ImGui::Text("Channel: %c", "RGB"[channel]);
            }
            if (texture->is_hdr) {
                ImGui::TextDisabled("HDR / tone-mapped preview");
            }
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
            ImGui::TextUnformatted(texture->name.c_str());
            if (!source.empty()) {
                ImGui::TextWrapped("Paint source: %s", source.c_str());
            }
            ImGui::PopTextWrapPos();
        } else {
            ImGui::TextUnformatted("Click to add a texture");
        }
        ImGui::EndTooltip();
    }
    if (clicked) {
        if (loaded) {
            ImGui::OpenPopup("actions");
        } else {
            action = Action::Load;
        }
    }
    if (ImGui::BeginPopupContextItem("actions")) {
        if (ImGui::MenuItem(loaded ? "Replace..." : "Load...")) {
            action = Action::Load;
        }
        if (ImGui::MenuItem("Copy", nullptr, false, loaded)) {
            action = Action::Copy;
        }
        if (ImGui::MenuItem("Paste", nullptr, false, canPaste)) {
            action = Action::Paste;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Remove from material", nullptr, false, loaded)) {
            action = Action::Remove;
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(label);
    if (loaded) {
        const std::string &path = texture->name;
        std::string name = path.substr(path.find_last_of("/\\") + 1);
        if (name.size() > 20) {
            name = name.substr(0, 17) + "...";
        }
        if (ImGui::GetContentRegionAvail().x < ImGui::GetFontSize() * 12.0f) {
            ImGui::TextDisabled("%d x %d", texture->width, texture->height);
        } else {
            ImGui::TextDisabled("%s", name.c_str());
        }
        if (paint) {
            ImGui::TextDisabled("Paint override");
        } else if (!cache.empty()) {
            ImGui::TextDisabled("%s", cache.c_str());
        }
    } else {
        ImGui::TextDisabled("No texture");
    }
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
    ImGui::Spacing();
    return action;
}

} // namespace MaterialTextureSlotUI
