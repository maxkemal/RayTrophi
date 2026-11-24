// Android kid launcher (see UI/PlayerKidLauncher.h). Pure ImGui draw-list drawing: no widgets,
// no windows, no state of its own beyond view state (scroll, animation, finger). The game list,
// load progress and audio mute all belong to the operator service and are re-read from it.
#ifdef __ANDROID__
#include "UI/PlayerKidLauncher.h"
#include "PlayerOperatorWidgets.h"
#include "UI/PlayerOperatorUI.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

namespace AppRuntime {
namespace {
using json = nlohmann::json;

enum class Screen { Playing, Library, Loading, Pause };
enum class Phase { Opening, Waiting, Failed };

struct Game {
    std::string name, path, key, thumbnail;
};

struct Rect {
    ImVec2 a, b;
    float w() const { return b.x - a.x; }
    float h() const { return b.y - a.y; }
    ImVec2 center() const { return ImVec2((a.x + b.x) * .5f, (a.y + b.y) * .5f); }
    bool contains(float x, float y) const { return x >= a.x && x <= b.x && y >= a.y && y <= b.y; }
};

struct Layout {
    float W = 0, H = 0, card_w = 0, card_h = 0, gap = 0, pitch = 0, card_y = 0;
};

struct Button {
    int id = 0; // 0 games, 1 resume, 2 sound, 3 picture quality, 4 exit
    ImVec2 center;
    float size = 0;
};

struct State {
    Screen screen = Screen::Playing;
    double fade_t0 = -1.0; // dark overlay fading out after the loading view hands over to the game
    double last_draw = 0.0;
    float dt = 1.0f / 60.0f;
    float raw_dt = 1.0f / 60.0f; // not clamped to 50 ms: load checkpoints draw only every ~second

    // Mirrors of the operator service (refreshed, never authoritative).
    std::vector<Game> games;
    std::string project;
    bool muted = false;
    float render_scale = 0.5f; // mirror of display status; snapped to the nearest quality level
    bool games_seen = false;
    double next_refresh = 0.0;
    double next_scan = 0.0;

    // Carousel.
    float scroll = 0.0f, scroll_v = 0.0f, target = 0.0f;
    bool index_pending = false; // target holds a card index until the layout pitch is known
    std::vector<float> press, press_v;

    // Finger.
    bool finger_down = false;
    SDL_FingerID finger = 0;
    float down_x = 0, down_y = 0, last_x = 0, last_y = 0, start_scroll = 0;
    double down_time = 0, last_move_time = 0;
    bool moved = false, dragging = false;
    int press_index = -1;
    int press_button = -1;

    // Loading.
    std::string auto_open_key;
    Phase phase = Phase::Opening;
    int load_index = 0;
    double load_t0 = 0.0, fail_t0 = 0.0;
    Rect from{};
    bool open_called = false;
    std::uint64_t attempt0 = 0;
    float shown_progress = 0.0f;
    std::string error;

    // Pause.
    double pause_t0 = 0.0;
    float button_press[5] = {1, 1, 1, 1, 1}, button_press_v[5] = {0, 0, 0, 0, 0};
};
State s;
float g_alpha = 1.0f;

double nowSeconds() {
    return static_cast<double>(SDL_GetTicks64()) * 0.001;
}
float density() {
    static float value = 0.0f;
    if (value <= 0.0f) {
        float dpi = 0.0f;
        if (SDL_GetDisplayDPI(0, &dpi, nullptr, nullptr) != 0 || dpi < 80.0f) {
            dpi = 320.0f;
        }
        value = std::clamp(dpi / 160.0f, 1.0f, 4.5f);
    }
    return value;
}
// One "dp" is a density-independent unit, but never larger than the screen allows: a landscape
// phone is ~360 dp tall; a window much shorter than that (emulator override sizes) shrinks the
// unit instead of overflowing. Set by makeLayout from the real window height.
float g_unit = 0.0f;
float dp(float v) {
    return v * (g_unit > 0.0f ? g_unit : density());
}
float clamp01(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}
float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}
float easeOut(float t) {
    const float u = 1.0f - clamp01(t);
    return 1.0f - u * u * u;
}
ImU32 C(int r, int g, int b, float a = 1.0f) {
    a = clamp01(a * g_alpha);
    return IM_COL32(r, g, b, static_cast<int>(a * 255.0f + 0.5f));
}
ImU32 mix(const int (&p)[3], const int (&q)[3], float t, float a = 1.0f) {
    return C(static_cast<int>(lerpf(static_cast<float>(p[0]), static_cast<float>(q[0]), t)),
             static_cast<int>(lerpf(static_cast<float>(p[1]), static_cast<float>(q[1]), t)),
             static_cast<int>(lerpf(static_cast<float>(p[2]), static_cast<float>(q[2]), t)), a);
}
ImVec2 V(float x, float y) {
    return ImVec2(x, y);
}

// ---- text -----------------------------------------------------------------------------------
ImFont *boldFont() {
    return PlayerConsole::ui.heading ? PlayerConsole::ui.heading : ImGui::GetFont();
}
ImVec2 textSize(float size, const std::string &text) {
    return boldFont()->CalcTextSizeA(size, FLT_MAX, 0.0f, text.c_str());
}
// Shrinks toward min_size, then ellipsizes, so a long title never leaves its card.
std::string fitText(float &size, float min_size, float max_width, std::string text) {
    float width = textSize(size, text).x;
    if (width > max_width && width > 0.0f) {
        size = std::max(min_size, size * max_width / width);
    }
    while (text.size() > 1 && textSize(size, text).x > max_width) {
        do {
            text.pop_back();
        } while (!text.empty() && (static_cast<unsigned char>(text.back()) & 0xC0) == 0x80);
        if (!text.empty()) {
            text.pop_back();
        }
        text += "...";
        if (textSize(size, text).x <= max_width) {
            break;
        }
        text.resize(text.size() - 3);
    }
    return text;
}
// Many Android images ship only a regular-weight (variable) Roboto, so bold is faked with two
// extra passes a hair to each side. A real bold face just comes out slightly heavier.
void drawText(ImDrawList *d, float size, ImVec2 p, ImU32 col, const std::string &text) {
    const float o = size * 0.022f;
    d->AddText(boldFont(), size, V(p.x - o, p.y), col, text.c_str());
    d->AddText(boldFont(), size, V(p.x + o, p.y), col, text.c_str());
    d->AddText(boldFont(), size, p, col, text.c_str());
}
void drawTextCentered(ImDrawList *d, float size, float cx, float cy, ImU32 col,
                      const std::string &text) {
    const auto extent = textSize(size, text);
    drawText(d, size, V(cx - extent.x * .5f, cy - extent.y * .5f), col, text);
}

// ---- operator service access ----------------------------------------------------------------
bool op(const char *name, const json &config = json::object(), json *out = nullptr,
        std::string *error = nullptr) {
    json output;
    const auto result = rtapi::playerOperator(name, config, output);
    if (!result.ok && error) {
        *error = result.error;
    }
    if (out) {
        *out = std::move(output);
    }
    return result.ok;
}

void refreshStatus(double now, bool force) {
    if (!force && now < s.next_refresh) {
        return;
    }
    if (playerSceneLoad().presenting()) {
        return; // inside a load checkpoint: the service is mid-operation
    }
    s.next_refresh = now + 0.5;
    json status;
    if (!op("status", json::object(), &status)) {
        return;
    }
    s.project = status.value("project", "");
    s.muted = status.value("preferences", json::object())
                  .value("audio", json::object())
                  .value("muted", false);
    s.render_scale = status.value("display", json::object()).value("render_scale", s.render_scale);
    std::vector<Game> games;
    for (const auto &entry : status.value("games", json::array())) {
        games.push_back({entry.value("name", ""), entry.value("path", ""), entry.value("key", ""),
                         entry.value("thumbnail", "")});
    }
    const bool first = !s.games_seen && !games.empty();
    s.games = std::move(games);
    s.press.resize(s.games.size(), 1.0f);
    s.press_v.resize(s.games.size(), 0.0f);
    if (first) {
        s.games_seen = true;
        int index = 0;
        for (size_t i = 0; i < s.games.size(); ++i) {
            if (!s.project.empty() && s.games[i].key == s.project) {
                index = static_cast<int>(i);
            }
        }
        s.target = static_cast<float>(index); // converted to pixels by normalizeScroll
        s.scroll_v = 0.0f;
        s.index_pending = true;
        s.press_index = -1;
    }
    if (s.games.empty() && !status.value("scanning", false) && now >= s.next_scan) {
        s.next_scan = now + 3.0;
        op("library.refresh");
    }
}

// ---- layout ---------------------------------------------------------------------------------
Layout makeLayout(float W, float H) {
    Layout l;
    l.W = W;
    l.H = H;
    g_unit = std::clamp(std::min(density(), H / 340.0f), 1.0f, 4.5f);
    l.card_h = H * 0.60f;
    l.gap = dp(24);
    l.card_w = std::min(l.card_h * 1.55f, W * 0.44f);
    l.pitch = l.card_w + l.gap;
    l.card_y = H * 0.19f;
    return l;
}
// scroll/target are kept in PIXELS once the layout is known; before that they hold an index.
void normalizeScroll(const Layout &l) {
    if (s.index_pending) {
        s.scroll = s.target = s.target * l.pitch;
        s.index_pending = false;
    }
}
float maxScroll(const Layout &l) {
    return s.games.empty() ? 0.0f : static_cast<float>(s.games.size() - 1) * l.pitch;
}
int nearestIndex(const Layout &l) {
    if (s.games.empty()) {
        return 0;
    }
    return std::clamp(static_cast<int>(std::lround(s.scroll / l.pitch)), 0,
                      static_cast<int>(s.games.size()) - 1);
}
Rect cardRect(const Layout &l, int i) {
    const float dist = std::fabs(static_cast<float>(i) * l.pitch - s.scroll) / l.pitch;
    float scale = 1.0f - 0.08f * std::min(1.0f, dist);
    if (i >= 0 && i < static_cast<int>(s.press.size())) {
        scale *= s.press[i];
    }
    const float cx = l.W * .5f + static_cast<float>(i) * l.pitch - s.scroll;
    const float cy = l.card_y + l.card_h * .5f;
    const float hw = l.card_w * .5f * scale, hh = l.card_h * .5f * scale;
    return {V(cx - hw, cy - hh), V(cx + hw, cy + hh)};
}
int cardAt(const Layout &l, float x, float y) {
    int best = -1;
    float best_dist = FLT_MAX;
    for (int i = 0; i < static_cast<int>(s.games.size()); ++i) {
        const auto rect = cardRect(l, i);
        const float dist = std::fabs(rect.center().x - x);
        if (rect.contains(x, y) && dist < best_dist) {
            best = i;
            best_dist = dist;
        }
    }
    return best;
}
int pauseButtons(const Layout &l, Button (&out)[5]) {
    const bool games = s.games.size() >= 2;
    const float big = l.H * .44f, small = l.H * .26f;
    float gap = l.H * .035f;
    const int ids[5] = {0, 1, 2, 3, 4};
    int count = 0;
    float total = 0.0f;
    for (int id : ids) {
        if (id == 0 && !games) {
            continue;
        }
        out[count].id = id;
        out[count].size = id == 1 ? big : small;
        total += out[count].size;
        ++count;
    }
    total += gap * static_cast<float>(count - 1);
    // Narrow windows: shrink everything uniformly instead of overflowing the screen.
    float fit = 1.0f;
    if (total > l.W * .94f) {
        fit = l.W * .94f / total;
        for (int i = 0; i < count; ++i) {
            out[i].size *= fit;
        }
        gap *= fit;
        total *= fit;
    }
    float x = l.W * .5f - total * .5f;
    for (int i = 0; i < count; ++i) {
        out[i].center = V(x + out[i].size * .5f, l.H * .5f);
        x += out[i].size + gap;
    }
    return count;
}
int buttonAt(const Layout &l, float x, float y) {
    Button buttons[5];
    const int count = pauseButtons(l, buttons);
    const float slack = dp(8);
    for (int i = 0; i < count; ++i) {
        const float half = buttons[i].size * .5f + slack;
        if (std::fabs(x - buttons[i].center.x) <= half && std::fabs(y - buttons[i].center.y) <= half) {
            return buttons[i].id;
        }
    }
    return -1;
}
Layout currentLayout() {
    const auto &io = ImGui::GetIO();
    const float W = io.DisplaySize.x > 8 ? io.DisplaySize.x : 1280.0f;
    const float H = io.DisplaySize.y > 8 ? io.DisplaySize.y : 720.0f;
    return makeLayout(W, H);
}

// ---- screen transitions ---------------------------------------------------------------------
void showPause(double now) {
    s.screen = Screen::Pause;
    s.pause_t0 = now;
    s.press_button = -1;
    s.finger_down = false;
    s.next_refresh = 0.0;
}
void goPlaying(double now, bool fade) {
    s.screen = Screen::Playing;
    s.fade_t0 = fade ? now : -1.0;
    s.finger_down = false;
}
void showLibrary() {
    s.screen = Screen::Library;
    s.finger_down = false;
    s.dragging = false;
    s.next_refresh = 0.0;
    const auto l = currentLayout();
    normalizeScroll(l);
    for (size_t i = 0; i < s.games.size(); ++i) {
        if (!s.project.empty() && s.games[i].key == s.project) {
            s.target = static_cast<float>(i) * l.pitch;
        }
    }
}
void startOpen(int index) {
    if (index < 0 || index >= static_cast<int>(s.games.size())) {
        return;
    }
    s.auto_open_key.clear();
    const auto now = nowSeconds();
    const auto &game = s.games[index];
    if (!game.key.empty() && game.key == s.project) {
        // Already the loaded game: nothing to load, just play.
        if (op("game.start")) {
            goPlaying(now, true);
        }
        return;
    }
    const auto l = currentLayout();
    s.scroll = s.target = static_cast<float>(index) * l.pitch;
    s.scroll_v = 0.0f;
    s.from = cardRect(l, index);
    s.screen = Screen::Loading;
    s.phase = Phase::Opening;
    s.load_index = index;
    s.load_t0 = now;
    s.open_called = false;
    s.attempt0 = playerSceneLoad().status().value("attempt", 0ull);
    s.shown_progress = 0.0f;
    s.error.clear();
    s.finger_down = false;
}
void failLoad(const std::string &message) {
    s.phase = Phase::Failed;
    s.fail_t0 = nowSeconds();
    s.error = message;
}

// ---- per-frame update -----------------------------------------------------------------------
void updateSpring(float &x, float &v, float target, float k, float c, float dt) {
    // Semi-implicit Euler in fixed substeps: stable for any k*dt^2 this file uses.
    int steps = std::max(1, static_cast<int>(std::ceil(dt / (1.0f / 240.0f))));
    const float h = dt / static_cast<float>(steps);
    for (int i = 0; i < steps; ++i) {
        v += (-k * (x - target) - c * v) * h;
        x += v * h;
    }
}
void updateLoading(double now) {
    const auto &game = s.games[std::clamp(s.load_index, 0, static_cast<int>(s.games.size()) - 1)];
    if (s.phase == Phase::Opening) {
        if (now - s.load_t0 >= 0.45 && !s.open_called) {
            s.open_called = true;
            std::string error;
            if (op("library.open", {{"path", game.path}}, nullptr, &error)) {
                s.phase = Phase::Waiting;
            } else {
                failLoad(error);
            }
        }
        return;
    }
    if (s.phase == Phase::Waiting) {
        const auto load = playerSceneLoad().status();
        if (load.value("attempt", 0ull) <= s.attempt0) {
            return; // the service has not picked the request up yet
        }
        const auto state = load.value("state", "");
        if (state == "loading") {
            s.shown_progress = std::max(s.shown_progress,
                                        static_cast<float>(load.value("progress", 0)) / 100.0f);
        } else if (state == "failed") {
            failLoad(load.value("error", "load failed"));
        } else if (state == "ready" && !playerSceneLoad().presenting()) {
            s.shown_progress = 1.0f;
            std::string error;
            if (op("game.start", json::object(), nullptr, &error)) {
                refreshStatus(now, true);
                goPlaying(now, true);
            } else {
                failLoad(error);
            }
        }
        return;
    }
    if (now - s.fail_t0 > 4.0) {
        showLibrary();
    }
}
void tick(double now) {
    const float raw = s.last_draw > 0.0 ? std::min(0.6f, static_cast<float>(now - s.last_draw)) : 1.0f / 60.0f;
    const float dt = s.last_draw > 0.0 ? clamp01(static_cast<float>(now - s.last_draw) / .05f) * .05f
                                         : 1.0f / 60.0f;
    s.last_draw = now;
    s.dt = dt;
    s.raw_dt = raw;
    const auto l = currentLayout();
    normalizeScroll(l);
    if (s.screen != Screen::Playing) {
        refreshStatus(now, false);
    }
    if (s.screen == Screen::Library && !s.auto_open_key.empty()) {
        const auto game = std::find_if(s.games.begin(), s.games.end(), [](const Game &entry) {
            return entry.key == s.auto_open_key;
        });
        if (game != s.games.end()) {
            startOpen(static_cast<int>(game - s.games.begin()));
            return;
        }
    }
    if (s.screen == Screen::Library) {
        if (!s.dragging) {
            updateSpring(s.scroll, s.scroll_v, s.target, 170.0f, 26.0f, dt);
        }
        for (size_t i = 0; i < s.press.size(); ++i) {
            const float goal = (s.finger_down && !s.moved && s.press_index == static_cast<int>(i))
                                   ? 0.96f
                                   : 1.0f;
            updateSpring(s.press[i], s.press_v[i], goal, 520.0f, 24.0f, dt);
        }
    } else if (s.screen == Screen::Pause) {
        for (int i = 0; i < 5; ++i) {
            const float goal = (s.finger_down && s.press_button == i) ? 0.94f : 1.0f;
            updateSpring(s.button_press[i], s.button_press_v[i], goal, 520.0f, 24.0f, dt);
        }
    } else if (s.screen == Screen::Loading && !s.games.empty()) {
        updateLoading(now);
    }
}

// ---- drawing --------------------------------------------------------------------------------
void drawBackground(ImDrawList *d, const Layout &l, double now) {
    const float t = 0.5f + 0.5f * std::sin(static_cast<float>(now) * 0.18f);
    const int tl0[3] = {255, 240, 214}, tl1[3] = {255, 228, 222};
    const int tr0[3] = {255, 221, 204}, tr1[3] = {250, 226, 236};
    const int br0[3] = {245, 190, 188}, br1[3] = {236, 196, 214};
    const int bl0[3] = {252, 222, 186}, bl1[3] = {248, 206, 196};
    d->AddRectFilledMultiColor(V(0, 0), V(l.W, l.H), mix(tl0, tl1, t), mix(tr0, tr1, t),
                               mix(br0, br1, t), mix(bl0, bl1, t));
    for (int k = 0; k < 6; ++k) {
        const float phase = static_cast<float>(now) * (0.02f + 0.007f * k) + k * 1.7f;
        const float x = l.W * (0.5f + 0.48f * std::sin(phase));
        const float y = l.H * (0.5f + 0.42f * std::cos(phase * 0.8f + k));
        d->AddCircleFilled(V(x, y), dp(70.0f + 36.0f * (k % 3)), C(255, 255, 255, 0.11f), 48);
    }
}
void drawShadow(ImDrawList *d, const Rect &r, float rounding, float strength, float dy) {
    for (int k = 0; k < 12; ++k) {
        const float e = dp(2.2f) * static_cast<float>(k);
        d->AddRectFilled(V(r.a.x - e, r.a.y - e + dy), V(r.b.x + e, r.b.y + e + dy),
                         C(96, 52, 48, strength * 0.022f), rounding + e);
    }
}
// Crop-to-fill UVs: the cover covers the rect, extra pixels are cut on the longer side.
void coverUv(PlayerConsole::ArtworkTexture *texture, const Rect &r, ImVec2 &uv0, ImVec2 &uv1) {
    const int tw = texture ? texture->Width : 0;
    const int th = texture ? texture->Height : 0;
    uv0 = V(0, 0);
    uv1 = V(1, 1);
    if (tw <= 0 || th <= 0 || r.w() <= 0 || r.h() <= 0) {
        return;
    }
    const float texture_aspect = static_cast<float>(tw) / static_cast<float>(th);
    const float rect_aspect = r.w() / r.h();
    if (texture_aspect > rect_aspect) {
        const float keep = rect_aspect / texture_aspect;
        uv0.x = (1.0f - keep) * .5f;
        uv1.x = uv0.x + keep;
    } else {
        const float keep = texture_aspect / rect_aspect;
        uv0.y = (1.0f - keep) * .5f;
        uv1.y = uv0.y + keep;
    }
}
PlayerConsole::ArtworkTexture *coverTexture(SDL_Renderer *renderer, const Game &game) {
    return PlayerConsole::artwork(renderer, game.thumbnail, 1024);
}
void drawImageCover(ImDrawList *d, PlayerConsole::ArtworkTexture *texture, const Rect &r, float rounding,
                    ImU32 tint = IM_COL32_WHITE) {
    ImVec2 uv0, uv1;
    coverUv(texture, r, uv0, uv1);
    const ImU32 col = (tint & 0x00FFFFFFu) | (static_cast<ImU32>(clamp01(g_alpha) * 255.0f) << 24);
    d->AddImageRounded(PlayerConsole::artworkRef(texture), r.a, r.b,
                       uv0, uv1, col, rounding);
}
void drawPlaceholder(ImDrawList *d, const Rect &r, float rounding, int index,
                     const std::string &name) {
    static const int palette[4][3] = {{255, 183, 139}, {170, 205, 255}, {190, 226, 168}, {236, 178, 220}};
    const auto &p = palette[index % 4];
    d->AddRectFilled(r.a, r.b, C(p[0], p[1], p[2]), rounding);
    const auto c = r.center();
    d->AddCircleFilled(V(c.x, c.y), std::min(r.w(), r.h()) * .30f, C(255, 255, 255, 0.35f), 48);
    d->AddCircleFilled(V(r.a.x + r.w() * .22f, r.a.y + r.h() * .30f), r.h() * .05f,
                       C(255, 255, 255, 0.5f));
    d->AddCircleFilled(V(r.a.x + r.w() * .80f, r.a.y + r.h() * .70f), r.h() * .07f,
                       C(255, 255, 255, 0.5f));
    std::string initial = name.empty() ? "?" : name.substr(0, 1);
    drawTextCentered(d, r.h() * .30f, c.x, c.y, C(120, 70, 70), initial);
}

void iconPlay(ImDrawList *d, ImVec2 c, float u, ImU32 col) {
    const ImVec2 pts[3] = {V(c.x - .30f * u, c.y - .46f * u), V(c.x - .30f * u, c.y + .46f * u),
                           V(c.x + .52f * u, c.y)};
    d->AddConvexPolyFilled(pts, 3, col);
    d->AddPolyline(pts, 3, col, ImDrawFlags_Closed, u * .18f);
}
void iconGrid(ImDrawList *d, ImVec2 c, float u, ImU32 col) {
    const float cell = .46f * u, gap = .14f * u;
    for (int iy = 0; iy < 2; ++iy) {
        for (int ix = 0; ix < 2; ++ix) {
            const float x = c.x - cell - gap * .5f + ix * (cell + gap);
            const float y = c.y - cell - gap * .5f + iy * (cell + gap);
            d->AddRectFilled(V(x, y), V(x + cell, y + cell), col, .13f * u);
        }
    }
}
void iconSpeaker(ImDrawList *d, ImVec2 c, float u, ImU32 col, bool muted) {
    const ImVec2 body[6] = {V(c.x - .64f * u, c.y - .24f * u), V(c.x - .32f * u, c.y - .24f * u),
                            V(c.x + .02f * u, c.y - .56f * u), V(c.x + .02f * u, c.y + .56f * u),
                            V(c.x - .32f * u, c.y + .24f * u), V(c.x - .64f * u, c.y + .24f * u)};
    d->AddConvexPolyFilled(body, 6, col);
    d->AddPolyline(body, 6, col, ImDrawFlags_Closed, u * .12f);
    if (!muted) {
        d->PathArcTo(V(c.x + .06f * u, c.y), .40f * u, -0.75f, 0.75f, 20);
        d->PathStroke(col, 0, u * .12f);
        d->PathArcTo(V(c.x + .06f * u, c.y), .74f * u, -0.70f, 0.70f, 24);
        d->PathStroke(col, 0, u * .12f);
    } else {
        d->AddLine(V(c.x + .30f * u, c.y - .30f * u), V(c.x + .78f * u, c.y + .30f * u), col, u * .13f);
        d->AddLine(V(c.x + .78f * u, c.y - .30f * u), V(c.x + .30f * u, c.y + .30f * u), col, u * .13f);
    }
}

// Picture quality levels: internal render scale 0.5 / 0.75 / 1.0 (Fast / Balanced / Sharp).
constexpr float kQualityScale[3] = {0.5f, 0.75f, 1.0f};
int qualityLevel() {
    int best = 0;
    for (int i = 1; i < 3; ++i) {
        if (std::fabs(kQualityScale[i] - s.render_scale) < std::fabs(kQualityScale[best] - s.render_scale)) {
            best = i;
        }
    }
    return best;
}
// Three bars, the first `level + 1` filled: reads as signal strength / detail.
void iconQuality(ImDrawList *d, ImVec2 c, float u, ImU32 col, int level) {
    for (int i = 0; i < 3; ++i) {
        const float x = c.x + (static_cast<float>(i) - 1.0f) * .50f * u;
        const float h = (.50f + .38f * static_cast<float>(i)) * u;
        const ImVec2 a = V(x - .17f * u, c.y + .56f * u - h), b = V(x + .17f * u, c.y + .56f * u);
        if (i <= level) {
            d->AddRectFilled(a, b, col, .08f * u);
        } else {
            d->AddRect(a, b, col, .08f * u, 0, u * .09f);
        }
    }
}
// Power symbol: open ring plus a stroke through the gap.
void iconExit(ImDrawList *d, ImVec2 c, float u, ImU32 col) {
    d->PathArcTo(c, .56f * u, -1.5708f + 0.62f, -1.5708f + 6.2832f - 0.62f, 28);
    d->PathStroke(col, 0, u * .15f);
    d->AddLine(V(c.x, c.y - .66f * u), V(c.x, c.y - .06f * u), col, u * .16f);
}

void drawCard(ImDrawList *d, SDL_Renderer *renderer, const Layout &l, int i) {
    const auto &game = s.games[i];
    const auto rect = cardRect(l, i);
    const float rounding = dp(28);
    const float dist = std::min(1.0f, std::fabs(static_cast<float>(i) * l.pitch - s.scroll) / l.pitch);
    drawShadow(d, rect, rounding, 1.0f - 0.35f * dist, dp(12));
    if (auto *texture = coverTexture(renderer, game)) {
        drawImageCover(d, texture, rect, rounding);
    } else {
        drawPlaceholder(d, rect, rounding, i, game.name);
    }
    if (dist > 0.01f) {
        d->AddRectFilled(rect.a, rect.b, C(255, 244, 232, 0.30f * dist), rounding);
    }
    // Title under the card, centered; bold, never smaller than 28 dp.
    float size = dp(32);
    const auto title = fitText(size, dp(28), rect.w() - dp(16), game.name);
    drawTextCentered(d, size, rect.center().x, rect.b.y + dp(26), C(92, 54, 62, 1.0f - 0.45f * dist), title);
}

void drawLibrary(ImDrawList *d, SDL_Renderer *renderer, const Layout &l) {
    // Top bar: logo + app name.
    const float logo = dp(40), pad = dp(24), top = dp(14);
    if (auto *texture = PlayerConsole::brand(renderer)) {
        d->AddImageRounded(PlayerConsole::artworkRef(texture),
                           V(pad, top), V(pad + logo, top + logo), V(0, 0), V(1, 1),
                           IM_COL32_WHITE, dp(14));
    } else {
        d->AddRectFilled(V(pad, top), V(pad + logo, top + logo), C(255, 138, 76), dp(14));
    }
    drawText(d, dp(24), V(pad + logo + dp(12), top + (logo - textSize(dp(24), "A").y) * .5f),
             C(92, 54, 62), PlayerConsole::tr("player.raytrophi_player", "RayTrophi Player"));
    if (s.games.empty()) {
        drawTextCentered(d, dp(32), l.W * .5f, l.H * .5f, C(92, 54, 62),
                         PlayerConsole::tr("player.finding_your_games", "Finding your games..."));
        return;
    }
    // Far cards first so the centered one overlaps its neighbours.
    std::vector<int> order(s.games.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<int>(i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return std::fabs(a * l.pitch - s.scroll) > std::fabs(b * l.pitch - s.scroll);
    });
    for (int i : order) {
        const auto rect = cardRect(l, i);
        if (rect.b.x < -dp(40) || rect.a.x > l.W + dp(40)) {
            continue;
        }
        drawCard(d, renderer, l, i);
    }
    // Page dots.
    const int n = static_cast<int>(s.games.size());
    if (n > 1 && n <= 14) {
        const float spacing = dp(26);
        const float x0 = l.W * .5f - spacing * (n - 1) * .5f;
        const float y = l.H - dp(12);
        for (int i = 0; i < n; ++i) {
            const float near = clamp01(1.0f - std::fabs(i * l.pitch - s.scroll) / l.pitch);
            d->AddCircleFilled(V(x0 + spacing * i, y), dp(5.0f + 4.0f * near),
                               C(255, 120, 84, 0.35f + 0.65f * near), 24);
        }
    }
}

void drawBalloons(ImDrawList *d, float cx, float cy, double now) {
    static const int colors[5][3] = {{255, 99, 99}, {255, 200, 80}, {110, 205, 130}, {96, 160, 255}, {236, 130, 210}};
    const float r = dp(17), spacing = dp(50);
    for (int i = 0; i < 5; ++i) {
        const float bounce = std::fabs(std::sin(static_cast<float>(now) * 4.0f + i * 0.7f));
        const float x = cx + (i - 2) * spacing;
        const float y = cy - bounce * dp(26);
        const auto &c = colors[i];
        d->AddLine(V(x, y + r), V(x, y + r + dp(14)), C(255, 255, 255, 0.35f), dp(1.2f));
        d->AddCircleFilled(V(x, y), r, C(c[0], c[1], c[2]), 32);
        d->AddCircleFilled(V(x - r * .35f, y - r * .35f), r * .28f, C(255, 255, 255, 0.55f), 16);
    }
}

void drawLoading(ImDrawList *d, SDL_Renderer *renderer, const Layout &l, double now) {
    const auto &game = s.games[std::clamp(s.load_index, 0, static_cast<int>(s.games.size()) - 1)];
    const float t = easeOut(static_cast<float>((now - s.load_t0) / 0.45));
    const Rect full{V(0, 0), V(l.W, l.H)};
    const Rect r{V(lerpf(s.from.a.x, 0, t), lerpf(s.from.a.y, 0, t)),
                 V(lerpf(s.from.b.x, l.W, t), lerpf(s.from.b.y, l.H, t))};
    const float rounding = dp(28) * (1.0f - t);
    drawShadow(d, r, rounding, 1.0f - t, dp(12));
    auto *texture = coverTexture(renderer, game);
    if (texture) {
        drawImageCover(d, texture, r, rounding);
        // Blur: the same cover as a ~40 px texture stretched with linear filtering, faded in with
        // the zoom (a real blur pass is not available to draw-list drawing).
        if (auto *blurred = PlayerConsole::artwork(renderer, game.thumbnail, 24)) {
            const float saved = g_alpha;
            g_alpha = saved * clamp01(t * 1.5f);
            drawImageCover(d, blurred, r, rounding);
            g_alpha = saved;
        }
    } else {
        drawPlaceholder(d, r, rounding, s.load_index, game.name);
    }
    d->AddRectFilled(r.a, r.b, C(38, 20, 36, 0.50f * t), rounding);
    const float content = clamp01((t - 0.5f) * 2.0f);
    if (content <= 0.0f) {
        return;
    }
    const float saved = g_alpha;
    g_alpha = saved * content;
    if (s.phase == Phase::Failed) {
        const float cx = l.W * .5f, cy = l.H * .36f, rr = dp(54);
        d->AddCircleFilled(V(cx, cy), rr, C(255, 214, 120), 48);
        d->AddCircleFilled(V(cx - rr * .36f, cy - rr * .2f), rr * .11f, C(92, 54, 62), 16);
        d->AddCircleFilled(V(cx + rr * .36f, cy - rr * .2f), rr * .11f, C(92, 54, 62), 16);
        d->PathArcTo(V(cx, cy + rr * .62f), rr * .36f, 3.7f, 5.7f, 20);
        d->PathStroke(C(92, 54, 62), 0, rr * .10f);
        drawTextCentered(d, dp(32), cx, l.H * .58f, C(255, 255, 255),
                         PlayerConsole::tr("kid.load_failed", "Oops! This game did not open."));
        drawTextCentered(d, dp(26), cx, l.H * .58f + dp(46), C(255, 235, 225, 0.9f),
                         PlayerConsole::tr("kid.load_failed_hint", "Tap to pick another one."));
    } else {
        float size = dp(44);
        const auto title = fitText(size, dp(30), l.W * .8f, game.name);
        drawTextCentered(d, size, l.W * .5f, l.H * .30f, C(255, 255, 255), title);
        drawTextCentered(d, dp(26), l.W * .5f, l.H * .30f + dp(52), C(255, 240, 228, 0.92f),
                         PlayerConsole::tr("kid.getting_ready", "Getting ready..."));
        drawBalloons(d, l.W * .5f, l.H * .625f, now);
        // Progress pill: smoothed toward the service's checkpoint progress.
        s.shown_progress = std::min(1.0f, s.shown_progress);
        static float display = 0.0f;
        if (s.phase == Phase::Opening) {
            display = 0.0f;
        }
        display += (s.shown_progress - display) * (1.0f - std::exp(-s.raw_dt * 6.0f));
        const float bw = l.W * .5f, bh = dp(22), bx = l.W * .25f, by = l.H * .765f;
        d->AddRectFilled(V(bx, by), V(bx + bw, by + bh), C(255, 255, 255, 0.25f), bh * .5f);
        const float fill = std::max(bh, bw * display);
        d->AddRectFilled(V(bx, by), V(bx + fill, by + bh), C(255, 176, 96), bh * .5f);
        d->AddRectFilled(V(bx + dp(4), by + dp(3)), V(bx + fill - dp(4), by + bh * .45f),
                         C(255, 255, 255, 0.35f), bh * .25f);
    }
    g_alpha = saved;
}

void drawPause(ImDrawList *d, const Layout &l, double now) {
    const float a = easeOut(static_cast<float>((now - s.pause_t0) / 0.28));
    const float saved = g_alpha;
    g_alpha = saved * a;
    d->AddRectFilledMultiColor(V(0, 0), V(l.W, l.H), C(70, 38, 58, 0.78f), C(70, 38, 58, 0.78f),
                               C(34, 20, 40, 0.88f), C(34, 20, 40, 0.88f));
    Button buttons[5];
    const int count = pauseButtons(l, buttons);
    for (int i = 0; i < count; ++i) {
        const auto &b = buttons[i];
        const float stagger = easeOut(static_cast<float>((now - s.pause_t0) / 0.28) - 0.08f * i);
        const float scale = lerpf(0.85f, 1.0f, stagger) * s.button_press[b.id];
        const float half = b.size * .5f * scale;
        const Rect r{V(b.center.x - half, b.center.y - half), V(b.center.x + half, b.center.y + half)};
        const float rounding = b.size * .22f * scale;
        const bool primary = b.id == 1;
        const bool muted = b.id == 2 && s.muted;
        drawShadow(d, r, rounding, 1.4f, dp(10));
        if (primary) {
            d->AddRectFilled(r.a, r.b, C(255, 126, 82), rounding);
            d->AddRectFilled(V(r.a.x + dp(6), r.a.y + dp(6)), V(r.b.x - dp(6), r.a.y + r.h() * .30f),
                             C(255, 255, 255, 0.10f), rounding * .6f);
        } else {
            d->AddRectFilled(r.a, r.b, muted ? C(226, 216, 222) : C(255, 246, 236), rounding);
        }
        const ImU32 ink_color = primary ? C(255, 255, 255) : C(100, 62, 70);
        const ImVec2 icon_center = V(r.center().x, r.a.y + r.h() * .41f);
        const float u = r.h() * .24f;
        if (b.id == 1) {
            iconPlay(d, icon_center, u * 1.15f, ink_color);
        } else if (b.id == 0) {
            iconGrid(d, icon_center, u, ink_color);
        } else if (b.id == 3) {
            iconQuality(d, icon_center, u, ink_color, qualityLevel());
        } else if (b.id == 4) {
            iconExit(d, icon_center, u, ink_color);
        } else {
            iconSpeaker(d, icon_center, u, ink_color, s.muted);
        }
        static const char *const quality_keys[3] = {"kid.quality_fast", "kid.quality_balanced",
                                                    "kid.quality_sharp"};
        static const char *const quality_en[3] = {"Fast", "Balanced", "Sharp"};
        const int quality = qualityLevel();
        const char *text = b.id == 1   ? PlayerConsole::tr("kid.resume", "Continue")
                           : b.id == 0 ? PlayerConsole::tr("kid.games", "Games")
                           : b.id == 3 ? PlayerConsole::tr(quality_keys[quality], quality_en[quality])
                           : b.id == 4 ? PlayerConsole::tr("kid.exit", "Exit")
                           : s.muted   ? PlayerConsole::tr("kid.sound_off", "Muted")
                                       : PlayerConsole::tr("kid.sound", "Sound");
        float size = std::max(dp(28), r.h() * .115f);
        const auto label = fitText(size, dp(24), r.w() - dp(18), text);
        drawTextCentered(d, size, r.center().x, r.b.y - r.h() * .17f, ink_color, label);
    }
    g_alpha = saved;
}

// Fade from the loading view into the running game.
void drawFade(ImDrawList *d, const Layout &l, double now) {
    const float t = static_cast<float>((now - s.fade_t0) / 0.45);
    if (t >= 1.0f) {
        s.fade_t0 = -1.0;
        return;
    }
    d->AddRectFilled(V(0, 0), V(l.W, l.H), C(38, 20, 36, 1.0f - easeOut(t)));
}
} // namespace

// ---- public ---------------------------------------------------------------------------------
bool kidLauncherActive() {
    return s.screen != Screen::Playing || s.fade_t0 >= 0.0;
}
bool kidLauncherPaused() {
    return s.screen == Screen::Pause || s.screen == Screen::Library;
}

void kidLauncherDraw(SDL_Renderer *renderer) {
    if (!kidLauncherActive()) {
        return;
    }
    const double now = nowSeconds();
    tick(now);
    const auto l = currentLayout();
    auto *d = ImGui::GetForegroundDrawList();
    g_alpha = 1.0f;
    switch (s.screen) {
    case Screen::Library:
        drawBackground(d, l, now);
        drawLibrary(d, renderer, l);
        break;
    case Screen::Loading:
        drawBackground(d, l, now);
        if (!s.games.empty()) {
            drawLoading(d, renderer, l, now);
        }
        break;
    case Screen::Pause:
        drawPause(d, l, now);
        break;
    case Screen::Playing:
        if (s.fade_t0 >= 0.0) {
            drawFade(d, l, now);
        }
        break;
    }
}

bool kidLauncherTouch(const SDL_Event &event, SDL_Window *window) {
    if (!kidLauncherActive()) {
        return false;
    }
    if (event.type != SDL_FINGERDOWN && event.type != SDL_FINGERMOTION &&
        event.type != SDL_FINGERUP) {
        return false;
    }
    int width = 0, height = 0;
    SDL_GetWindowSize(window, &width, &height);
    const float x = event.tfinger.x * static_cast<float>(width);
    const float y = event.tfinger.y * static_cast<float>(height);
    const auto l = makeLayout(static_cast<float>(width), static_cast<float>(height));
    normalizeScroll(l);
    const double now = nowSeconds();
    const bool mine = s.finger_down && event.tfinger.fingerId == s.finger;

    if (event.type == SDL_FINGERDOWN && !s.finger_down) {
        s.finger_down = true;
        s.finger = event.tfinger.fingerId;
        s.down_x = s.last_x = x;
        s.down_y = s.last_y = y;
        s.down_time = s.last_move_time = now;
        s.moved = s.dragging = false;
        s.press_index = -1;
        s.press_button = -1;
        if (s.screen == Screen::Library) {
            s.start_scroll = s.scroll;
            s.press_index = cardAt(l, x, y);
        } else if (s.screen == Screen::Pause) {
            s.press_button = buttonAt(l, x, y);
        }
        return true;
    }
    if (!mine) {
        return true; // extra fingers are swallowed while a launcher screen is up
    }
    if (event.type == SDL_FINGERMOTION) {
        if (s.screen == Screen::Library) {
            const float slop = dp(10);
            if (!s.moved && std::hypot(x - s.down_x, y - s.down_y) > slop) {
                s.moved = true;
                s.dragging = true;
            }
            if (s.dragging) {
                float raw = s.start_scroll - (x - s.down_x);
                const float max_scroll = maxScroll(l);
                if (raw < 0.0f) {
                    raw *= 0.35f;
                } else if (raw > max_scroll) {
                    raw = max_scroll + (raw - max_scroll) * 0.35f;
                }
                const float elapsed = std::max(0.004f, static_cast<float>(now - s.last_move_time));
                const float instant = -(x - s.last_x) / elapsed;
                s.scroll_v = 0.55f * s.scroll_v + 0.45f * instant;
                s.scroll = raw;
                s.last_move_time = now;
            }
        }
        s.last_x = x;
        s.last_y = y;
        return true;
    }
    // FINGERUP
    s.finger_down = false;
    if (s.screen == Screen::Library) {
        if (s.dragging) {
            s.dragging = false;
            if (now - s.last_move_time > 0.1) {
                s.scroll_v = 0.0f; // finger rested before lifting: no fling
            }
            const float projected = s.scroll + s.scroll_v * 0.18f;
            const int index = std::clamp(static_cast<int>(std::lround(projected / l.pitch)), 0,
                                         std::max(0, static_cast<int>(s.games.size()) - 1));
            s.target = static_cast<float>(index) * l.pitch;
        } else if (!s.moved && s.press_index >= 0 && s.press_index == cardAt(l, x, y)) {
            const int index = s.press_index;
            s.press_index = -1;
            startOpen(index);
        }
        s.press_index = -1;
    } else if (s.screen == Screen::Pause) {
        const int button = buttonAt(l, x, y);
        const int pressed = s.press_button;
        s.press_button = -1;
        if (button >= 0 && button == pressed) {
            if (button == 1) {
                goPlaying(now, false);
            } else if (button == 0) {
                showLibrary();
            } else if (button == 3) {
                const float next = kQualityScale[(qualityLevel() + 1) % 3];
                if (op("display.configure", {{"render_scale", next}})) {
                    s.render_scale = next;
                }
                s.next_refresh = 0.0;
            } else if (button == 4) {
                kidLauncherExit();
            } else {
                op("audio.configure", {{"muted", !s.muted}});
                s.muted = !s.muted;
                s.next_refresh = 0.0;
            }
        }
    } else if (s.screen == Screen::Loading && s.phase == Phase::Failed) {
        showLibrary();
    }
    return true;
}

void kidLauncherEvent(const SDL_Event &event) {
    const bool left = event.type == SDL_APP_WILLENTERBACKGROUND ||
                      (event.type == SDL_WINDOWEVENT &&
                       event.window.event == SDL_WINDOWEVENT_FOCUS_LOST);
    if (left && s.screen == Screen::Playing && s.fade_t0 < 0.0 && !s.project.empty()) {
        showPause(nowSeconds());
    }
}

void kidLauncherBack() {
    if (s.screen == Screen::Playing && s.fade_t0 < 0.0) {
        showPause(nowSeconds());
    } else if (s.screen == Screen::Pause) {
        goPlaying(nowSeconds(), false);
    }
}

void kidLauncherAutoOpen(const std::string &scene_key) {
    showLibrary();
    s.auto_open_key = scene_key;
}

bool kidLauncherShow(const std::string &screen, std::string &error) {
    const double now = nowSeconds();
    if (s.screen == Screen::Loading) {
        error = "A game is loading";
        return false;
    }
    refreshStatus(now, true);
    if (screen == "library") {
        showLibrary();
    } else if (screen == "pause") {
        if (s.project.empty()) {
            error = "No game is loaded";
            return false;
        }
        showPause(now);
    } else if (screen == "playing") {
        if (s.project.empty()) {
            error = "No game is loaded";
            return false;
        }
        goPlaying(now, false);
    } else {
        error = "Unknown launcher screen (library|pause|playing)";
        return false;
    }
    return true;
}

bool kidLauncherSelect(int index, std::string &error) {
    if (s.screen != Screen::Library) {
        error = "Open the library first";
        return false;
    }
    if (index < 0 || index >= static_cast<int>(s.games.size())) {
        error = "Game index out of range";
        return false;
    }
    startOpen(index);
    return true;
}

void kidLauncherExit() {
    SDL_Event quit{};
    quit.type = SDL_QUIT;
    SDL_PushEvent(&quit);
}

json kidLauncherStatus() {
    refreshStatus(nowSeconds(), true); // the cache is lazy while playing; IPC reads fresh values
    const char *names[] = {"playing", "library", "loading", "pause"};
    const auto l = currentLayout();
    json out = {{"screen", names[static_cast<int>(s.screen)]},
                {"games", s.games.size()},
                {"selected_index", nearestIndex(l)},
                {"project", s.project},
                {"muted", s.muted},
                {"render_scale", s.render_scale},
                {"quality_level", qualityLevel()},
                {"fading", s.fade_t0 >= 0.0}};
    if (s.screen == Screen::Loading) {
        out["phase"] = s.phase == Phase::Opening ? "opening"
                       : s.phase == Phase::Waiting ? "waiting" : "failed";
        out["load_index"] = s.load_index;
        out["progress"] = s.shown_progress;
        out["error"] = s.error;
    }
    return out;
}
} // namespace AppRuntime
#endif
