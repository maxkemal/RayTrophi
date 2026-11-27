#pragma once

#include "Core/Localization.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace Paint {

struct BalloonMilestoneCopy {
    const char* badge;
    const char* title;
    const char* subtitle;
    float accent[3];
};

struct BalloonMilestoneState {
    static constexpr int kTotalLevels = 5;
    static constexpr double kStepFraction = 0.20;
    static constexpr double kBannerSeconds = 3.2;
    static constexpr double kGiftClaimSeconds = 2.4;
    static constexpr double kPartyModeSeconds = 12.0;
    static constexpr double kBarBounceSeconds = 0.90;
    static constexpr double kGiftBurstSeconds = 1.45;

    bool active = false;
    double room_progress = 0.0;
    int level = 0;

    int banner_level = 0;
    double banner_timer = 0.0;
    bool banner_is_gift_claim = false;

    bool gift_box_active = false;
    int gift_box_level = 0;
    double gift_box_age = 0.0;
    int gifts_claimed = 0;

    double bar_bounce_timer = 0.0;
    double gift_burst_timer = 0.0;
    double party_mode_remaining = 0.0;
    double anim_clock = 0.0;

    // Normalized [0..1] screen hit rect updated by BalloonMilestoneHud for direct touch/mouse hits.
    float gift_hit_x0 = 0.78f;
    float gift_hit_y0 = 0.14f;
    float gift_hit_x1 = 0.95f;
    float gift_hit_y1 = 0.38f;

    bool hitTestGiftBoxNormalized(float nx, float ny) const {
        return active && gift_box_active &&
               nx >= gift_hit_x0 && nx <= gift_hit_x1 &&
               ny >= gift_hit_y0 && ny <= gift_hit_y1;
    }

    void reset() {
        room_progress = 0.0;
        level = 0;
        banner_level = 0;
        banner_timer = 0.0;
        banner_is_gift_claim = false;
        gift_box_active = false;
        gift_box_level = 0;
        gift_box_age = 0.0;
        gifts_claimed = 0;
        bar_bounce_timer = 0.0;
        gift_burst_timer = 0.0;
        party_mode_remaining = 0.0;
    }

    void tick(double dt, bool demo_active, double progress) {
        active = demo_active;
        room_progress = std::clamp(progress, 0.0, 1.0);
        if (!active) {
            return;
        }
        const double step = std::clamp(dt, 0.0, 0.25);
        anim_clock += step;
        if (banner_timer > 0.0) {
            banner_timer = (std::max)(0.0, banner_timer - step);
        }
        if (bar_bounce_timer > 0.0) {
            bar_bounce_timer = (std::max)(0.0, bar_bounce_timer - step);
        }
        if (gift_burst_timer > 0.0) {
            gift_burst_timer = (std::max)(0.0, gift_burst_timer - step);
        }
        if (gift_box_active) {
            gift_box_age += step;
        }
        if (party_mode_remaining > 0.0) {
            party_mode_remaining = (std::max)(0.0, party_mode_remaining - step);
        }
    }

    void triggerMilestone(int new_level) {
        const int clamped = std::clamp(new_level, 1, kTotalLevels);
        level = (std::max)(level, clamped);
        banner_level = clamped;
        banner_is_gift_claim = false;
        banner_timer = kBannerSeconds;
        bar_bounce_timer = kBarBounceSeconds;

        // Levels 2 (40%) and 4 (80%) drop an interactive floating gift package.
        if (clamped == 2 || clamped == 4) {
            gift_box_active = true;
            gift_box_level = clamped;
            gift_box_age = 0.0;
        }
        // Level 4 (80%) starts a short Golden Party boost before the finale.
        if (clamped == 4) {
            party_mode_remaining = kPartyModeSeconds;
        }
        // Finale (100%) clears any pending gift prompt so the trophy banner owns the stage.
        if (clamped >= kTotalLevels) {
            gift_box_active = false;
            gift_box_age = 0.0;
        }
    }

    bool claimGift() {
        if (!gift_box_active) {
            return false;
        }
        gift_box_active = false;
        gift_box_age = 0.0;
        ++gifts_claimed;
        banner_level = std::clamp(gift_box_level, 1, kTotalLevels);
        banner_is_gift_claim = true;
        banner_timer = kGiftClaimSeconds;
        bar_bounce_timer = kBarBounceSeconds;
        gift_burst_timer = kGiftBurstSeconds;
        return true;
    }

    bool partyModeActive() const {
        return party_mode_remaining > 0.0;
    }

    static bool isTurkishLocale() {
        const std::string& loc = Localization::locale();
        return loc.size() >= 2 && (loc[0] == 't' || loc[0] == 'T') &&
               (loc[1] == 'r' || loc[1] == 'R');
    }

    static BalloonMilestoneCopy copyFor(int lvl, bool gift_claimed) {
        const bool tr = isTurkishLocale();
        if (gift_claimed) {
            if (tr) {
                return {"HED\xC4\xB0YE A\xC3\x87ILDI!",
                        "G\xC3\xB6kku\xC5\x9F" "a\xC4\x9F\xC4\xB1 S\xC3\xBCrprizi!",
                        "S\xC3\xBCrpriz balonlar ve y\xC4\xB1ld\xC4\xB1z ya\xC4\x9Fmuru!",
                        {1.0f, 0.78f, 0.18f}};
            }
            return {"GIFT OPENED!",
                    "Rainbow Surprise!",
                    "Bonus surprise balloons and star shower!",
                    {1.0f, 0.78f, 0.18f}};
        }
        switch (lvl) {
            case 1:
                if (tr) {
                    return {"%20",
                            "Harika Ba\xC5\x9Flang\xC4\xB1\xC3\xA7!",
                            "S\xC3\xBCrpriz balonlar sahneye geldi!",
                            {0.25f, 0.85f, 0.96f}};
                }
                return {"20%",
                        "Awesome Start!",
                        "Surprise balloons joined the park!",
                        {0.25f, 0.85f, 0.96f}};
            case 2:
                if (tr) {
                    return {"%40",
                            "Renk Avc\xC4\xB1s\xC4\xB1!",
                            "S\xC3\xBCrpriz hediye paketi seni bekliyor!",
                            {0.98f, 0.42f, 0.72f}};
                }
                return {"40%",
                        "Color Explorer!",
                        "A surprise gift box is ready to open!",
                        {0.98f, 0.42f, 0.72f}};
            case 3:
                if (tr) {
                    return {"%60",
                            "S\xC3\xBCper Boyac\xC4\xB1!",
                            "Dev Mega Balonlar haz\xC4\xB1r!",
                            {0.34f, 0.92f, 0.48f}};
                }
                return {"60%",
                        "Super Painter!",
                        "Mega paint balloons unlocked!",
                        {0.34f, 0.92f, 0.48f}};
            case 4:
                if (tr) {
                    return {"%80",
                            "Alt\xC4\xB1n Parti Zaman\xC4\xB1!",
                            "H\xC4\xB1zl\xC4\xB1 balonlar + yeni hediye paketi!",
                            {1.0f, 0.76f, 0.18f}};
                }
                return {"80%",
                        "Golden Party Time!",
                        "Fast balloons + bonus gift box!",
                        {1.0f, 0.76f, 0.18f}};
            default:
                if (tr) {
                    return {"%100",
                            "Park Canland\xC4\xB1!",
                            "Tebrikler! T\xC3\xBCm gizli resimler a\xC3\xA7\xC4\xB1ld\xC4\xB1!",
                            {1.0f, 0.84f, 0.22f}};
                }
                return {"100%",
                        "Park Complete!",
                        "Amazing! Every hidden picture is alive!",
                        {1.0f, 0.84f, 0.22f}};
        }
    }

    nlohmann::json statusJson() const {
        const auto copy = copyFor(banner_level > 0 ? banner_level : 1, banner_is_gift_claim);
        return {{"active", active},
                {"level", level},
                {"total_levels", kTotalLevels},
                {"progress_percent", static_cast<int>(std::lround(room_progress * 100.0))},
                {"banner_active", banner_timer > 0.0},
                {"banner_level", banner_level},
                {"banner_title", banner_timer > 0.0 ? copy.title : ""},
                {"gift_box_active", gift_box_active},
                {"gift_box_level", gift_box_level},
                {"gifts_claimed", gifts_claimed},
                {"party_mode_remaining", party_mode_remaining}};
    }
};

inline BalloonMilestoneState& balloonMilestones() {
    static BalloonMilestoneState instance;
    return instance;
}

} // namespace Paint
