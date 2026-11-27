#pragma once

#include "Api/RtApiPaintWet.h"
#include "Paint/BalloonMilestoneRewards.h"
#include "imgui.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>

namespace Paint {

inline void drawStarPolygon(ImDrawList* dl, ImVec2 center, float outer_r, float inner_r,
                            float angle_offset, ImU32 color) {
    ImVec2 pts[10];
    constexpr float kPi = 3.14159265358979323846f;
    for (int i = 0; i < 10; ++i) {
        const float r = (i % 2 == 0) ? outer_r : inner_r;
        const float a = angle_offset + (static_cast<float>(i) * kPi / 5.0f) - (kPi * 0.5f);
        pts[i] = ImVec2(center.x + std::cos(a) * r, center.y + std::sin(a) * r);
    }
    dl->AddConvexPolyFilled(pts, 10, color);
}

inline void drawSparklingStar(ImDrawList* dl, ImVec2 center, float radius, float angle,
                              const ImVec4& rgb, float alpha, float twinkle) {
    if (alpha <= 0.01f || radius <= 0.5f) {
        return;
    }
    const float glow_r = radius * (1.75f + 0.35f * twinkle);
    dl->AddCircleFilled(center, glow_r,
                        ImGui::ColorConvertFloat4ToU32(
                            ImVec4(rgb.x, rgb.y, rgb.z, alpha * (0.16f + 0.10f * twinkle))),
                        20);
    dl->AddCircleFilled(center, radius * 1.15f,
                        ImGui::ColorConvertFloat4ToU32(
                            ImVec4(1.0f, 0.96f, 0.72f, alpha * (0.28f + 0.16f * twinkle))),
                        16);

    const ImU32 star_col = ImGui::ColorConvertFloat4ToU32(ImVec4(rgb.x, rgb.y, rgb.z, alpha));
    drawStarPolygon(dl, center, radius, radius * 0.44f, angle, star_col);

    // Bright warm-white inner star + 4-point sparkle flare.
    const ImU32 core_col =
        ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 0.99f, 0.90f, alpha * (0.78f + 0.22f * twinkle)));
    drawStarPolygon(dl, center, radius * 0.55f, radius * 0.22f, angle, core_col);

    const float flare = radius * (1.25f + 0.45f * twinkle);
    const ImU32 flare_col =
        ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, alpha * (0.65f * twinkle)));
    dl->AddLine(ImVec2(center.x - flare, center.y), ImVec2(center.x + flare, center.y), flare_col,
                1.6f);
    dl->AddLine(ImVec2(center.x, center.y - flare), ImVec2(center.x, center.y + flare), flare_col,
                1.6f);
}

inline void drawRainbowText(ImDrawList* dl, ImFont* font, float font_size, ImVec2 pos,
                            const char* text, float time_sec, float alpha) {
    if (!dl || !font || !text || !*text || font_size <= 0.0f || alpha <= 0.001f) {
        return;
    }

    const float outline = std::clamp(font_size * 0.058f, 1.35f, 2.6f);
    const float diag = outline * 0.70710678f;
    const float shadow = std::clamp(font_size * 0.095f, 2.5f, 4.0f);
    const ImU32 outline_col =
        ImGui::ColorConvertFloat4ToU32(ImVec4(0.05f, 0.03f, 0.10f, alpha * 0.94f));
    const ImU32 shadow_col =
        ImGui::ColorConvertFloat4ToU32(ImVec4(0.0f, 0.0f, 0.0f, alpha * 0.78f));
    const ImVec2 outline_offsets[8] = {
        {-outline, 0.0f}, {outline, 0.0f}, {0.0f, -outline}, {0.0f, outline},
        {-diag, -diag},   {diag, -diag},   {-diag, diag},    {diag, diag},
    };

    float cursor_x = pos.x;
    float cursor_y = pos.y;
    int visible_idx = 0;

    const char* p = text;
    while (*p != '\0') {
        const unsigned char lead = static_cast<unsigned char>(*p);
        int seq_len = 1;
        if ((lead & 0x80u) == 0) {
            seq_len = 1;
        } else if ((lead & 0xE0u) == 0xC0u && p[1] != '\0') {
            seq_len = 2;
        } else if ((lead & 0xF0u) == 0xE0u && p[1] != '\0' && p[2] != '\0') {
            seq_len = 3;
        } else if ((lead & 0xF8u) == 0xF0u && p[1] != '\0' && p[2] != '\0' && p[3] != '\0') {
            seq_len = 4;
        }

        const char* next = p + seq_len;
        if (seq_len == 1 && *p == '\n') {
            cursor_x = pos.x;
            cursor_y += font_size;
            p = next;
            continue;
        }

        const float adv = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, p, next).x;
        const bool is_space = (seq_len == 1 && (*p == ' ' || *p == '\t' || *p == '\r'));
        if (!is_space) {
            const float raw_hue =
                std::fmod(static_cast<float>(visible_idx) * 0.11f - time_sec * 0.55f, 1.0f);
            const float hue = (raw_hue < 0.0f) ? (raw_hue + 1.0f) : raw_hue;
            float r = 1.0f;
            float g = 1.0f;
            float b = 1.0f;
            ImGui::ColorConvertHSVtoRGB(hue, 0.85f, 1.0f, r, g, b);
            const ImU32 glyph_col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, alpha));

            dl->AddText(font, font_size, ImVec2(cursor_x + shadow, cursor_y + shadow), shadow_col,
                        p, next);
            for (const ImVec2& off : outline_offsets) {
                dl->AddText(font, font_size, ImVec2(cursor_x + off.x, cursor_y + off.y),
                            outline_col, p, next);
            }
            dl->AddText(font, font_size, ImVec2(cursor_x, cursor_y), glyph_col, p, next);
            ++visible_idx;
        }

        cursor_x += adv;
        p = next;
    }
}

inline void drawRotating3DGiftBox(ImDrawList* dl, ImVec2 center, float half_extent, float yaw,
                                  float pitch, float roll) {
    struct V3 {
        float x;
        float y;
        float z;
    };
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float cr = std::cos(roll), sr = std::sin(roll);

    auto rot = [&](V3 v) -> V3 {
        // Yaw (Y) -> Pitch (X) -> Roll (Z)
        const float x1 = v.x * cy + v.z * sy;
        const float z1 = -v.x * sy + v.z * cy;
        const float y2 = v.y * cp - z1 * sp;
        const float z2 = v.y * sp + z1 * cp;
        const float x3 = x1 * cr - y2 * sr;
        const float y3 = x1 * sr + y2 * cr;
        return {x3, y3, z2};
    };

    const float cam_dist = 4.6f;
    auto proj = [&](V3 p) -> ImVec2 {
        const float s = cam_dist / (cam_dist - p.z);
        return ImVec2(center.x + p.x * half_extent * s, center.y - p.y * half_extent * s);
    };

    auto shade = [](ImVec4 base, float ndl) -> ImU32 {
        const float lit = 0.58f + 0.48f * std::clamp(ndl, 0.0f, 1.0f);
        return ImGui::ColorConvertFloat4ToU32(
            ImVec4((std::min)(1.0f, base.x * lit), (std::min)(1.0f, base.y * lit),
                   (std::min)(1.0f, base.z * lit), base.w));
    };

    struct Face {
        V3 n;
        V3 u;
        V3 v;
        bool is_top;
    };
    const Face faces[6] = {
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}, false},
        {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, false},
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}, false},
        {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, false},
        {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}, true},
        {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}, false},
    };

    const V3 light_dir = {0.42f, 0.72f, 0.55f};
    const ImVec4 body_col(0.95f, 0.25f, 0.54f, 0.98f);
    const ImVec4 lid_col(1.00f, 0.36f, 0.64f, 0.99f);
    const ImVec4 rib_col(1.00f, 0.86f, 0.28f, 1.00f);

    // Draw visible faces (back-face culled by rotated normal z > 0).
    for (int f = 0; f < 6; ++f) {
        const V3 rn = rot(faces[f].n);
        if (rn.z <= 0.02f) {
            continue;
        }
        const V3 ru = rot(faces[f].u);
        const V3 rv = rot(faces[f].v);
        auto pt = [&](float su, float sv, float lift) -> ImVec2 {
            return proj({rn.x * lift + ru.x * su + rv.x * sv,
                         rn.y * lift + ru.y * su + rv.y * sv,
                         rn.z * lift + ru.z * su + rv.z * sv});
        };

        const float ndl = rn.x * light_dir.x + rn.y * light_dir.y + rn.z * light_dir.z;
        const ImVec2 quad[4] = {pt(-1.0f, -1.0f, 1.0f), pt(1.0f, -1.0f, 1.0f),
                                pt(1.0f, 1.0f, 1.0f), pt(-1.0f, 1.0f, 1.0f)};
        dl->AddConvexPolyFilled(quad, 4, shade(faces[f].is_top ? lid_col : body_col, ndl));

        // 3D golden ribbon cross wrapped on every visible face.
        const float rw = 0.20f;
        const ImU32 rc = shade(rib_col, ndl + 0.12f);
        const ImVec2 rib_v[4] = {pt(-rw, -1.0f, 1.015f), pt(rw, -1.0f, 1.015f),
                                 pt(rw, 1.0f, 1.015f), pt(-rw, 1.0f, 1.015f)};
        const ImVec2 rib_h[4] = {pt(-1.0f, -rw, 1.015f), pt(1.0f, -rw, 1.015f),
                                 pt(1.0f, rw, 1.015f), pt(-1.0f, rw, 1.015f)};
        dl->AddConvexPolyFilled(rib_v, 4, rc);
        dl->AddConvexPolyFilled(rib_h, 4, rc);

        dl->AddPolyline(quad, 4, IM_COL32(255, 255, 255, 165), ImDrawFlags_Closed, 1.6f);
    }

    // 3D bow loops perched on top center of the cube.
    const V3 top_c = rot({0.0f, 1.18f, 0.0f});
    const V3 bow_l = rot({-0.36f, 1.32f, 0.0f});
    const V3 bow_r = rot({0.36f, 1.32f, 0.0f});
    const ImVec2 p_top = proj(top_c);
    const ImVec2 p_bl = proj(bow_l);
    const ImVec2 p_br = proj(bow_r);
    dl->AddCircle(p_bl, half_extent * 0.26f, IM_COL32(255, 228, 96, 255), 16, 3.2f);
    dl->AddCircle(p_br, half_extent * 0.26f, IM_COL32(255, 228, 96, 255), 16, 3.2f);
    dl->AddCircleFilled(p_top, half_extent * 0.14f, IM_COL32(255, 240, 140, 255), 12);
}

inline void drawBalloonMilestoneHud() {
    auto& ms = balloonMilestones();
    if (!ms.active) {
        return;
    }

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    if (!vp || vp->WorkSize.x < 320.0f || vp->WorkSize.y < 240.0f) {
        return;
    }

    const ImVec2 origin = vp->WorkPos;
    const ImVec2 size = vp->WorkSize;
    const float ui_scale = std::clamp((std::min)(size.x / 1280.0f, size.y / 720.0f), 0.78f, 1.45f);
    const float t = static_cast<float>(ms.anim_clock);
    ImDrawList* fg = ImGui::GetForegroundDrawList();

    // 1. Top-center transparent progress bar with elastic balloon-blob bounce on milestones.
    {
        float blob_sx = 1.0f;
        float blob_sy = 1.0f;
        float blob_glow = 0.0f;
        if (ms.bar_bounce_timer > 0.0) {
            const float u = static_cast<float>(
                (std::max)(0.0, 1.0 - ms.bar_bounce_timer / BalloonMilestoneState::kBarBounceSeconds));
            const float env = std::exp(-4.8f * u);
            const float wave = std::sin(u * 4.5f * 3.14159265f);
            // Balloon-like elastic squash & stretch: swells outward then bounces back cleanly.
            blob_sx = 1.0f + 0.14f * env * wave;
            blob_sy = 1.0f + 0.36f * env * wave;
            blob_glow = env;
        }

        const float base_w = std::clamp(size.x * 0.42f, 300.0f, 560.0f);
        const float base_h = 22.0f * ui_scale;
        const float bar_w = base_w * blob_sx;
        const float bar_h = base_h * blob_sy;
        const float bar_cx = origin.x + size.x * 0.5f;
        const float bar_cy = origin.y + 30.0f * ui_scale;
        const float bar_x = bar_cx - bar_w * 0.5f;
        const float bar_y = bar_cy - bar_h * 0.5f;

        // Soft outer elastic glow pulse when a milestone hits.
        if (blob_glow > 0.02f) {
            const float pad = (6.0f + 10.0f * blob_glow) * ui_scale;
            fg->AddRectFilled(ImVec2(bar_x - pad, bar_y - pad),
                              ImVec2(bar_x + bar_w + pad, bar_y + bar_h + pad),
                              IM_COL32(255, 218, 92, static_cast<int>(105.0f * blob_glow)),
                              (bar_h + pad * 2.0f) * 0.5f);
        }

        // Fully transparent backdrop — only a subtle glassy outline for the track.
        const ImVec2 t0(bar_x, bar_y);
        const ImVec2 t1(bar_x + bar_w, bar_y + bar_h);
        fg->AddRectFilled(t0, t1, IM_COL32(255, 255, 255, 40), bar_h * 0.5f);
        fg->AddRect(t0, t1, IM_COL32(255, 255, 255, 145), bar_h * 0.5f, 0, 1.8f * ui_scale);

        // Smooth colorful progress fill.
        const float fill_w = static_cast<float>(ms.room_progress) * bar_w;
        if (fill_w > 2.0f) {
            const ImU32 col_l = ms.partyModeActive() ? IM_COL32(255, 184, 42, 242)
                                                     : IM_COL32(64, 214, 248, 242);
            const ImU32 col_r = ms.partyModeActive() ? IM_COL32(255, 92, 178, 242)
                                                     : IM_COL32(110, 242, 134, 242);
            fg->AddRectFilledMultiColor(t0, ImVec2(bar_x + fill_w, bar_y + bar_h),
                                        col_l, col_r, col_r, col_l);
            fg->AddRect(t0, ImVec2(bar_x + fill_w, bar_y + bar_h),
                        IM_COL32(255, 255, 255, 165), bar_h * 0.5f, 0, 1.2f * ui_scale);
        }

        // 5 milestone star pips (20%, 40%, 60%, 80%, 100%).
        for (int i = 1; i <= BalloonMilestoneState::kTotalLevels; ++i) {
            const float frac = static_cast<float>(i) * 0.20f;
            const float cx = bar_x + frac * bar_w;
            const float cy = bar_cy;
            const bool unlocked = ms.level >= i;
            const bool is_gift = (i == 2 || i == 4);
            const float pip_bounce = (unlocked && i == ms.level) ? (1.0f + 0.35f * blob_glow) : 1.0f;
            if (unlocked) {
                const float tw = 0.5f + 0.5f * std::sin(t * 5.2f + static_cast<float>(i) * 1.3f);
                const float r =
                    (is_gift ? 15.0f : 12.5f) * ui_scale * (1.0f + 0.08f * tw) * pip_bounce;
                const ImVec4 rgb = is_gift ? ImVec4(1.0f, 0.45f, 0.76f, 1.0f)
                                           : ImVec4(1.0f, 0.84f, 0.22f, 1.0f);
                drawSparklingStar(fg, ImVec2(cx, cy), r, t * 0.4f + static_cast<float>(i), rgb,
                                  1.0f, tw);
            } else {
                const float r = (is_gift ? 10.0f : 8.5f) * ui_scale;
                drawStarPolygon(fg, ImVec2(cx, cy), r, r * 0.44f, 0.0f,
                                IM_COL32(255, 255, 255, 105));
            }
        }

        // Percentage text with soft shadow, scaled cleanly with the bar.
        ImFont* pct_font = ImGui::GetFont();
        const float pct_size = ImGui::GetFontSize() * 1.18f * ui_scale * (1.0f + 0.22f * blob_glow);
        char pct_buf[16];
        std::snprintf(pct_buf, sizeof(pct_buf), "%%%d",
                      static_cast<int>(std::lround(ms.room_progress * 100.0)));
        const ImVec2 txt_sz = pct_font->CalcTextSizeA(pct_size, FLT_MAX, 0.0f, pct_buf);
        const ImVec2 txt_pos(bar_x + bar_w + 18.0f * ui_scale, bar_cy - txt_sz.y * 0.5f);
        fg->AddText(pct_font, pct_size, ImVec2(txt_pos.x + 2.0f, txt_pos.y + 2.0f),
                    IM_COL32(0, 0, 0, 185), pct_buf);
        fg->AddText(pct_font, pct_size, txt_pos, IM_COL32(255, 250, 224, 255), pct_buf);
    }

    // 2. Frameless sparkling Milestone / Gift-Opened celebration (no black box, only stars & text).
    if (ms.banner_timer > 0.0 && ms.banner_level > 0) {
        const double total = ms.banner_is_gift_claim ? BalloonMilestoneState::kGiftClaimSeconds
                                                     : BalloonMilestoneState::kBannerSeconds;
        const float elapsed = static_cast<float>((std::max)(0.0, total - ms.banner_timer));
        const float remain = static_cast<float>(ms.banner_timer);

        // Smooth elastic pop-in (first 0.35s) and gentle fade-out (last 0.45s).
        float pop = 1.0f;
        if (elapsed < 0.35f) {
            const float u = elapsed / 0.35f;
            pop = 1.0f - std::pow(1.0f - u, 3.0f) + 0.14f * std::sin(u * 3.14159265f);
        }
        const float alpha = std::clamp(remain / 0.45f, 0.0f, 1.0f);
        const auto copy = BalloonMilestoneState::copyFor(ms.banner_level, ms.banner_is_gift_claim);
        const ImVec4 rgb(copy.accent[0], copy.accent[1], copy.accent[2], 1.0f);

        ImFont* font = ImGui::GetFont();
        const float base_font_size = ImGui::GetFontSize();
        const float header_font_size = base_font_size * 1.55f * ui_scale * pop;
        const float sub_font_size = base_font_size * 1.16f * ui_scale * pop;

        const std::string header = std::string(copy.badge) + "  " + copy.title;
        const ImVec2 hsz =
            font ? font->CalcTextSizeA(header_font_size, FLT_MAX, 0.0f, header.c_str(),
                                       header.c_str() + header.size())
                 : ImGui::CalcTextSize(header.c_str());
        const ImVec2 ssz = font ? font->CalcTextSizeA(sub_font_size, FLT_MAX, 0.0f, copy.subtitle)
                                : ImGui::CalcTextSize(copy.subtitle);

        const float span_w = (std::max)(std::clamp(size.x * 0.42f, 320.0f, 560.0f) * pop,
                                        (std::max)(hsz.x, ssz.x) + 72.0f * ui_scale);
        const float span_h = 98.0f * ui_scale * pop;
        const float cx = origin.x + size.x * 0.5f;
        const float cy = origin.y + 106.0f * ui_scale + std::sin(t * 4.0f) * 3.5f * ui_scale;

        // Orbiting bright sparkling stars floating freely over the 3D scene.
        for (int s = 0; s < 10; ++s) {
            const float phase = static_cast<float>(s) * 0.6283185f;
            const float ang = t * (s % 2 == 0 ? 1.15f : -0.85f) + phase;
            const float rx = span_w * (s % 2 == 0 ? 0.50f : 0.36f);
            const float ry = span_h * (s % 2 == 0 ? 0.56f : 0.72f);
            const float sx = cx + std::cos(ang) * rx;
            const float sy = cy + std::sin(ang) * ry;
            const float tw = 0.5f + 0.5f * std::sin(t * 7.0f + static_cast<float>(s) * 1.7f);
            const float sr = ((s % 2 == 0) ? 12.0f : 7.5f) * ui_scale * (0.85f + 0.30f * tw);
            const ImVec4 star_rgb = (s % 3 == 0) ? ImVec4(1.0f, 0.90f, 0.32f, 1.0f) : rgb;
            drawSparklingStar(fg, ImVec2(sx, sy), sr, ang * 1.5f, star_rgb, alpha, tw);
        }

        // Completely frameless rainbow header + rainbow subtitle directly on fg.
        const float gap_y = 6.0f * ui_scale * pop;
        const float total_text_h = hsz.y + gap_y + ssz.y;
        const float hx = cx - hsz.x * 0.5f;
        const float hy = cy - total_text_h * 0.5f;
        drawRainbowText(fg, font, header_font_size, ImVec2(hx, hy), header.c_str(), t, alpha);

        const float sx = cx - ssz.x * 0.5f;
        const float sy = hy + hsz.y + gap_y;
        drawRainbowText(fg, font, sub_font_size, ImVec2(sx, sy), copy.subtitle, t + 1.8f, alpha);
    }

    // 3. Interactive Floating Gift Box (clickable/tappable on PC & Android).
    if (ms.gift_box_active) {
        const float box_sz = 92.0f * ui_scale;
        const float bob = std::sin(t * 3.6f) * 9.0f * ui_scale;
        const float gx = origin.x + size.x * 0.86f;
        const float gy = origin.y + size.y * 0.24f + bob;
        const ImVec2 b0(gx - box_sz * 0.5f, gy - box_sz * 0.5f);
        const ImVec2 b1(gx + box_sz * 0.5f, gy + box_sz * 0.5f);

        // Pulsing halo ring + sparkling corner stars.
        const float halo_r = box_sz * (0.68f + 0.08f * std::sin(t * 5.0f));
        fg->AddCircleFilled(ImVec2(gx, gy), halo_r, IM_COL32(255, 206, 68, 55), 32);
        fg->AddCircle(ImVec2(gx, gy), halo_r, IM_COL32(255, 224, 110, 185), 32, 2.5f * ui_scale);
        for (int s = 0; s < 4; ++s) {
            const float ang = t * 1.6f + static_cast<float>(s) * 1.5708f;
            const float tw = 0.5f + 0.5f * std::sin(t * 6.5f + static_cast<float>(s));
            drawSparklingStar(fg, ImVec2(gx + std::cos(ang) * halo_r, gy + std::sin(ang) * halo_r),
                              9.0f * ui_scale, ang, ImVec4(1.0f, 0.90f, 0.30f, 1.0f), 1.0f, tw);
        }

        // Rotating 3D gift cube (smooth yaw + gentle tilt).
        drawRotating3DGiftBox(fg, ImVec2(gx, gy), box_sz * 0.44f, t * 1.45f,
                              0.32f + 0.12f * std::sin(t * 2.1f), 0.08f * std::sin(t * 1.7f));

        const char* cta = BalloonMilestoneState::isTurkishLocale()
                              ? "HED\xC4\xB0YEY\xC4\xB0 A\xC3\x87!"
                              : "OPEN GIFT!";
        const ImVec2 cta_sz = ImGui::CalcTextSize(cta);
        const ImVec2 pill0(gx - cta_sz.x * 0.5f - 10.0f * ui_scale, b1.y + 8.0f * ui_scale);
        const ImVec2 pill1(gx + cta_sz.x * 0.5f + 10.0f * ui_scale,
                           b1.y + cta_sz.y + 16.0f * ui_scale);
        fg->AddRectFilled(pill0, pill1, IM_COL32(255, 204, 52, 245), 10.0f * ui_scale);
        fg->AddText(ImVec2(gx - cta_sz.x * 0.5f, pill0.y + 4.0f * ui_scale),
                    IM_COL32(34, 22, 12, 255), cta);

        // Update normalized hit rect with generous kid-friendly touch padding for Android & PC.
        const float pad = 28.0f * ui_scale;
        ms.gift_hit_x0 = std::clamp((b0.x - origin.x - pad) / size.x, 0.0f, 1.0f);
        ms.gift_hit_y0 = std::clamp((b0.y - origin.y - pad) / size.y, 0.0f, 1.0f);
        ms.gift_hit_x1 = std::clamp((b1.x - origin.x + pad) / size.x, 0.0f, 1.0f);
        ms.gift_hit_y1 = std::clamp((pill1.y - origin.y + pad) / size.y, 0.0f, 1.0f);

        // Interactive mouse hit target (Android SDL_FINGERDOWN is also handled directly in PlayerMain).
        ImGui::SetNextWindowPos(ImVec2(b0.x - 16.0f, b0.y - 24.0f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(box_sz + 32.0f, box_sz + 60.0f), ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        const ImGuiWindowFlags gflags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav;
        if (ImGui::Begin("##balloon_gift_box_btn", nullptr, gflags)) {
            if (ImGui::InvisibleButton("##claim_gift_hit", ImGui::GetContentRegionAvail())) {
                std::string output;
                (void)rtapi::wetPaintDemoCommand("claim_gift", "{}", output);
            }
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }

    // 4. Local explosion / confetti & sparkling star burst right where the gift box opened.
    if (ms.gift_burst_timer > 0.0) {
        const float u = static_cast<float>(
            (std::max)(0.0, 1.0 - ms.gift_burst_timer / BalloonMilestoneState::kGiftBurstSeconds));
        const float fade = std::clamp(1.0f - u, 0.0f, 1.0f);
        const float gx = origin.x + size.x * 0.86f;
        const float gy = origin.y + size.y * 0.24f;

        // Soft warm-gold center glow + 2 expanding shockwave rings.
        const float glow_r = (35.0f + 135.0f * std::sqrt(u)) * ui_scale;
        fg->AddCircleFilled(ImVec2(gx, gy), glow_r,
                            IM_COL32(255, 228, 110, static_cast<int>(135.0f * fade * fade)), 32);
        const float ring1 = (24.0f + 190.0f * u) * ui_scale;
        const float ring2 = (14.0f + 130.0f * u) * ui_scale;
        fg->AddCircle(ImVec2(gx, gy), ring1,
                      IM_COL32(255, 214, 72, static_cast<int>(230.0f * fade)), 36,
                      (4.5f * fade + 1.0f) * ui_scale);
        fg->AddCircle(ImVec2(gx, gy), ring2,
                      IM_COL32(255, 96, 178, static_cast<int>(210.0f * fade)), 32,
                      (3.2f * fade + 0.8f) * ui_scale);

        // 28 radial rainbow confetti ribbons and sparkling stars bursting from (gx, gy).
        constexpr int kBurstPieces = 28;
        constexpr float kTwoPi = 6.283185307f;
        for (int i = 0; i < kBurstPieces; ++i) {
            const float fi = static_cast<float>(i);
            const float ang = (fi / static_cast<float>(kBurstPieces)) * kTwoPi +
                              0.27f * std::sin(fi * 3.1f);
            const float speed = (140.0f + 170.0f * (0.5f + 0.5f * std::sin(fi * 5.7f))) * ui_scale;
            const float drag_u = 1.0f - std::pow(1.0f - u, 2.2f);
            const float px = gx + std::cos(ang) * speed * drag_u;
            const float py = gy + std::sin(ang) * speed * drag_u + 120.0f * ui_scale * u * u;

            float hue = std::fmod(fi * 0.13f + u * 0.4f, 1.0f);
            float cr = 1.0f, cg = 1.0f, cb = 1.0f;
            ImGui::ColorConvertHSVtoRGB(hue, 0.88f, 1.0f, cr, cg, cb);

            if (i % 3 == 0) {
                const float sr = (11.0f + 4.0f * (1.0f - u)) * ui_scale * (0.4f + 0.6f * fade);
                drawSparklingStar(fg, ImVec2(px, py), sr, ang + u * 5.0f, ImVec4(cr, cg, cb, 1.0f),
                                  fade, 0.6f + 0.4f * std::sin(t * 14.0f + fi));
            } else {
                const float spin = ang + u * (6.0f + (i % 4) * 1.8f);
                const float rw = 9.0f * ui_scale * (0.5f + 0.5f * fade);
                const float rh = 4.5f * ui_scale * (0.5f + 0.5f * fade);
                const float cs = std::cos(spin), sn = std::sin(spin);
                const ImVec2 q[4] = {
                    ImVec2(px - cs * rw + sn * rh, py - sn * rw - cs * rh),
                    ImVec2(px + cs * rw + sn * rh, py + sn * rw - cs * rh),
                    ImVec2(px + cs * rw - sn * rh, py + sn * rw + cs * rh),
                    ImVec2(px - cs * rw - sn * rh, py - sn * rw + cs * rh),
                };
                fg->AddConvexPolyFilled(
                    q, 4, ImGui::ColorConvertFloat4ToU32(ImVec4(cr, cg, cb, fade)));
            }
        }
    }
}

} // namespace Paint
