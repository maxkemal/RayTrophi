#include "PlayerOperatorWidgets.h"
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
#include "Viewport/PlayerSwapchainPresenter.h"
#endif
#include "stb_image.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shobjidl.h>
#include <wrl/client.h>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#else
#include <SDL_image.h>
#endif

namespace PlayerConsole {
namespace {
ArtworkTexture *logo = nullptr;
#ifndef _WIN32
bool logo_failed = false;
#endif
std::map<std::string, ArtworkTexture *> thumbnails;

#ifdef _WIN32
std::string browseFolder() {
    using Microsoft::WRL::ComPtr;
    const auto initialized = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    ComPtr<IFileOpenDialog> dialog;
    std::string result;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                   IID_PPV_ARGS(&dialog)))) {
        DWORD options = 0;
        dialog->GetOptions(&options);
        dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
        const auto *title = tr("native.add_game_folder", "Add a RayTrophi game folder");
        const int count = MultiByteToWideChar(CP_UTF8, 0, title, -1, nullptr, 0);
        std::wstring wide_title(static_cast<size_t>(count), L'\0');
        if (count > 0) {
            MultiByteToWideChar(CP_UTF8, 0, title, -1, wide_title.data(), count);
            dialog->SetTitle(wide_title.c_str());
        }
        if (SUCCEEDED(dialog->Show(nullptr))) {
            ComPtr<IShellItem> item;
            PWSTR path = nullptr;
            if (SUCCEEDED(dialog->GetResult(&item)) &&
                SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                result = AppRuntime::projectPathToUtf8(std::filesystem::path(path));
                CoTaskMemFree(path);
            }
        }
    }
    dialog.Reset();
    if (SUCCEEDED(initialized)) {
        CoUninitialize();
    }
    return result;
}
#endif
ArtworkTexture *thumbnail(SDL_Renderer *renderer, const std::string &path, int max_dim = 512) {
    if (path.empty()) {
        return nullptr;
    }
    const std::string cache_key = path + "#" + std::to_string(max_dim);
    const auto found = thumbnails.find(cache_key);
    if (found != thumbnails.end()) {
        return found->second;
    }
    if (thumbnails.size() >= 64) {
        return nullptr;
    }
    // Cache failed loads too, so a broken artwork file is not decoded every frame.
    thumbnails.emplace(cache_key, nullptr);
    std::ifstream stream(AppRuntime::projectPathFromUtf8(path), std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() <= 0 || stream.tellg() > 16 * 1024 * 1024) {
        return nullptr;
    }
    const size_t bytes = static_cast<size_t>(stream.tellg());
    std::vector<unsigned char> data(bytes);
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes));
    int width = 0, height = 0, channels = 0;
    if (!stbi_info_from_memory(data.data(), static_cast<int>(bytes), &width, &height, &channels) ||
        width > 4096 || height > 4096) {
        return nullptr;
    }
    auto *pixels =
        stbi_load_from_memory(data.data(), static_cast<int>(bytes), &width, &height, &channels, 4);
    if (!pixels) {
        return nullptr;
    }
    auto *source = SDL_CreateRGBSurfaceWithFormatFrom(pixels, width, height, 32, width * 4,
                                                      SDL_PIXELFORMAT_RGBA32);
    const float scale = std::min(1.0f, static_cast<float>(max_dim) / static_cast<float>(std::max(width, height)));
    auto *reduced = SDL_CreateRGBSurfaceWithFormat(0, std::max(1, static_cast<int>(width * scale)),
                                                   std::max(1, static_cast<int>(height * scale)),
                                                   32, SDL_PIXELFORMAT_RGBA32);
    ArtworkTexture *texture = nullptr;
    if (source) {
        SDL_SetSurfaceBlendMode(source, SDL_BLENDMODE_NONE);
    }
    if (source && reduced && SDL_BlitScaled(source, nullptr, reduced, nullptr) == 0) {
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
        (void)renderer;
        auto& presenter = AppRuntime::PlayerSwapchainPresenter::instance();
        texture = presenter.createUiTexture(reduced->w, reduced->h);
        presenter.updateUiTexture(*texture, reduced->pixels, reduced->w, reduced->h,
                                  reduced->pitch);
#else
        texture = SDL_CreateTextureFromSurface(renderer, reduced);
        if (texture) {
            SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND);
            SDL_SetTextureScaleMode(texture, SDL_ScaleModeLinear);
        }
#endif
    }
    if (source) {
        SDL_FreeSurface(source);
    }
    if (reduced) {
        SDL_FreeSurface(reduced);
    }
    stbi_image_free(pixels);

    thumbnails[cache_key] = texture;
    return texture;
}
std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}
} // namespace

ArtworkTexture *artwork(SDL_Renderer *renderer, const std::string &path, int max_dim) {
    return thumbnail(renderer, path, max_dim);
}
ArtworkTexture *brand(SDL_Renderer *renderer) {
#ifdef _WIN32
    if (logo) {
        return logo;
    }
    const auto source = static_cast<HICON>(LoadImageW(
        GetModuleHandleW(nullptr), MAKEINTRESOURCEW(101), IMAGE_ICON, 64, 64, LR_DEFAULTCOLOR));
    if (!source) {
        return nullptr;
    }
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = 64;
    info.bmiHeader.biHeight = -64;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void *pixels = nullptr;
    const auto bitmap = CreateDIBSection(nullptr, &info, DIB_RGB_COLORS, &pixels, nullptr, 0);
    const auto dc = CreateCompatibleDC(nullptr);
    if (bitmap && dc && pixels) {
        const auto previous = SelectObject(dc, bitmap);
        std::memset(pixels, 0, 64 * 64 * 4);
        DrawIconEx(dc, 0, 0, source, 64, 64, 0, nullptr, DI_NORMAL);
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
        (void)renderer;
        auto* iconSurface = SDL_CreateRGBSurfaceWithFormatFrom(
            pixels, 64, 64, 32, 64 * 4, SDL_PIXELFORMAT_BGRA32);
        if (iconSurface) {
            auto* rgba = SDL_ConvertSurfaceFormat(iconSurface, SDL_PIXELFORMAT_RGBA32, 0);
            if (rgba) {
                auto& presenter = AppRuntime::PlayerSwapchainPresenter::instance();
                logo = presenter.createUiTexture(64, 64);
                presenter.updateUiTexture(*logo, rgba->pixels, 64, 64, rgba->pitch);
                SDL_FreeSurface(rgba);
            }
            SDL_FreeSurface(iconSurface);
        }
#else
        logo =
            SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, 64, 64);
        if (logo) {
            SDL_UpdateTexture(logo, nullptr, pixels, 64 * 4);
            SDL_SetTextureBlendMode(logo, SDL_BLENDMODE_BLEND);
        }
#endif
        SelectObject(dc, previous);
    }
    if (dc) {
        DeleteDC(dc);
    }
    if (bitmap) {
        DeleteObject(bitmap);
    }
    DestroyIcon(source);
#else
    // No Win32 resource icon: the packaged APK ships the same artwork as assets/player_icon.png
    // (largest .ico frame, converted by scripts/stage_player_android_assets.py).
    if (logo || logo_failed) {
        return logo;
    }
    logo_failed = true;
    (void)renderer;
    auto *surface = IMG_Load("assets/player_icon.png");
    if (surface) {
        auto *rgba = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGBA32, 0);
        if (rgba) {
            auto& presenter = AppRuntime::PlayerSwapchainPresenter::instance();
            logo = presenter.createUiTexture(rgba->w, rgba->h);
            if (logo) {
                presenter.updateUiTexture(*logo, rgba->pixels, rgba->w, rgba->h, rgba->pitch);
            }
            SDL_FreeSurface(rgba);
        }
        SDL_FreeSurface(surface);
    }
#endif
    return logo;
}
void releaseArtwork() {
    for (const auto &entry : thumbnails) {
        if (entry.second) {
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
            AppRuntime::PlayerSwapchainPresenter::instance().destroyUiTexture(entry.second);
#else
            SDL_DestroyTexture(entry.second);
#endif
        }
    }
    thumbnails.clear();
    if (logo) {
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
        AppRuntime::PlayerSwapchainPresenter::instance().destroyUiTexture(logo);
#else
        SDL_DestroyTexture(logo);
#endif
        logo = nullptr;
    }
#ifndef _WIN32
    logo_failed = false;
#endif
}
void games(SDL_Renderer *renderer) {
    heading(tr("player.game_library", "Game library"),
            tr("player.pick_an_experience_for_the_room_or_drop_an_rtp_project_into_player",
               "Pick an experience for the room, or drop an .rtp project into Player."));
#ifdef _WIN32
    ImGui::SetNextItemWidth(std::max(100.0f, ImGui::GetContentRegionAvail().x - 220));
#else
    // No native folder picker on Android: library roots come from the bundled scenes folder.
    ImGui::SetNextItemWidth(std::max(100.0f, ImGui::GetContentRegionAvail().x - S(110)));
#endif
    ImGui::InputTextWithHint("##search", tr("player.search_your_games", "Search your games"),
                             ui.search, sizeof(ui.search));
    ImGui::SameLine();
#ifdef _WIN32
    if (action(trId("player.add_folder", "Add folder"), ImVec2(106, 32))) {
        const auto folder = browseFolder();
        if (!folder.empty()) {
            call("library.add", {{"path", folder}});
        }
    }
    ImGui::SameLine();
#endif
    ImGui::BeginDisabled(ui.status.value("scanning", false));
    if (action(trId("player.refresh", "Refresh"), ImVec2(88, 32))) {
        call("library.refresh");
    }
    ImGui::EndDisabled();
    ImGui::Spacing();
    const auto load = AppRuntime::playerSceneLoad().status();
    const auto load_state = load.value("state", "");
    const auto load_key = load.value("key", "");
    const bool loading = load_state == "loading";
    const auto current = load_state == "ready" ? load_key : ui.status.value("project", "");
    const auto entries = ui.status.value("games", json::array());
    if (!current.empty()) {
        const auto active = std::find_if(entries.begin(), entries.end(), [&](const json& game) {
            return game.value("key", "") == current;
        });
        const auto title = active == entries.end()
            ? AppRuntime::projectPathToUtf8(AppRuntime::projectPathFromUtf8(current).stem())
            : active->value("name", "");
        label((std::string(tr("player.current_scene", "Current scene")) + "  ·  " + title).c_str(),
              mint);
    }
    const auto selected = std::find_if(entries.begin(), entries.end(), [](const json& game) {
        return game.value("key", "") == ui.selected_scene;
    });
#ifndef __ANDROID__
    ImGui::BeginDisabled(loading || selected == entries.end() || ui.selected_scene == current);
    if (action(trId("player.load_selected_scene", "Load selected scene"), ImVec2(0, 34), true)) {
        call("library.open", {{"path", (*selected)["path"]}});
    }
    ImGui::EndDisabled();
    if (selected != entries.end()) {
        ImGui::SameLine();
        const auto name = selected->value("name", "");
        label(name.c_str(), muted);
    }
#endif
    const float available = ImGui::GetContentRegionAvail().x;
    const int columns = std::max(1, std::min(3, static_cast<int>((available + S(14)) / S(230))));
    const float width = (available - (columns - 1) * S(14)) / columns;
    int shown = 0;
    const auto search = lower(ui.search);
    for (const auto &game : entries) {
        const auto title = game.value("name", tr("player.untitled", "Untitled"));
        const auto key = game.value("key", "");
        const bool card_loading = loading && key == load_key;
        const bool failed = load_state == "failed" && key == load_key;
        const bool active = !key.empty() && key == current;
        const bool chosen = !key.empty() && key == ui.selected_scene;
        if (!card_loading && !search.empty() && lower(title).find(search) == std::string::npos) {
            continue;
        }
        if (shown % columns) {
            ImGui::SameLine(0, S(14));
        }
        ImGui::PushID(game["path"].get<std::string>().c_str());
        ImGui::BeginGroup();
        const auto p = ImGui::GetCursorScreenPos();
        const bool clicked = ImGui::InvisibleButton("card", ImVec2(width, S(190))) && !loading;
        const bool hovered = ImGui::IsItemHovered();
        auto *draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(p, add(p, width, S(190)),
                            (chosen || hovered) ? IM_COL32(37, 49, 71, 255) : panel, 13);
        draw->AddRect(p, add(p, width, S(190)),
                      (active || card_loading) ? mint : (chosen ? violet : border), 13, 0, 1);
        if (auto *image = thumbnail(renderer, game.value("thumbnail", ""))) {
            draw->AddImageRounded(artworkRef(image),
                                  add(p, S(8), S(8)), add(p, width - S(8), S(120)), ImVec2(0, 0), ImVec2(1, 1),
                                  IM_COL32_WHITE, 9);
        } else {
            draw->PushClipRect(add(p, 1, 1), add(p, width - 1, S(120)), true);
            draw->AddRectFilled(add(p, S(8), S(8)), add(p, width - S(8), S(120)), IM_COL32(36, 43, 69, 255), 9);
            const auto category = lower(title);
            if (category.find("paint") != std::string::npos ||
                category.find("balloon") != std::string::npos ||
                category.find("pastel") != std::string::npos) {
                const ImU32 colors[] = {violet, mint, IM_COL32(246, 184, 135, 255)};
                for (int i = 0; i < 3; ++i) {
                    const auto center = add(p, width * (.25f + i * .25f), S(50 + (i % 2) * 18.0f));
                    draw->AddCircleFilled(center, S(22), colors[i]);
                    draw->AddCircleFilled(add(center, S(-6), S(-7)), S(5), IM_COL32(255, 255, 255, 85));
                    draw->AddBezierCubic(add(center, 0, S(22)), add(center, S(-8), S(36)),
                                         add(center, S(8), S(44)), add(center, 0, S(58)), colors[i], 1.3f);
                }
            } else {
                draw->AddCircleFilled(add(p, width * .5f, S(63)), S(45), IM_COL32(43, 64, 79, 255));
                icon(draw, add(p, width * .5f, S(63)), "games", mint, S(26));
                draw->AddCircleFilled(add(p, width * .22f, S(34)), S(6), violet);
                draw->AddCircleFilled(add(p, width * .78f, S(93)), S(4), violet);
            }
            draw->PopClipRect();
        }
        draw->PushClipRect(add(p, S(14), S(130)), add(p, width - S(10), S(180)), true);
        draw->AddText(add(p, S(14), S(132)), ink, title.c_str());
        if (card_loading) {
            const auto progress = load.value("progress", 0);
            const auto text = tr("player.scene_loading", "Loading scene");
            const auto stage = load.value("stage", "");
            const auto phase = Localization::text("player.load_stage." + stage, text);
            draw->AddText(add(p, S(14), S(155)), mint, phase);
            const float track = std::max(0.0f, width - S(28));
            draw->AddRectFilled(add(p, S(14), S(174)), add(p, S(14) + track, S(178)), border, 2);
            draw->AddRectFilled(add(p, S(14), S(174)), add(p, S(14) + track * progress / 100.0f, S(178)),
                                mint, 2);
        } else if (failed) {
            draw->AddText(add(p, S(14), S(160)), IM_COL32(255, 173, 133, 255),
                          tr("player.scene_load_failed", "Load failed"));
        } else if (active) {
            draw->AddText(add(p, S(14), S(160)), mint, tr("player.current_scene", "Current scene"));
        } else {
            draw->AddText(add(p, S(14), S(160)), chosen ? violet : muted,
                          chosen ? tr("player.selected_scene", "Selected") :
                          (game.value("packaged", false)
                              ? tr("player.raytrophi_experience", "RAYTROPHI EXPERIENCE")
                              : tr("player.local_project", "LOCAL PROJECT")));
        }
        draw->PopClipRect();
        if (hovered) {
            const auto detail = failed ? load.value("error", "") :
                                (card_loading ? load.value("detail", "") : game.value("path", ""));
            ImGui::SetTooltip("%s\n%s", title.c_str(), detail.c_str());
        }
        if (clicked) {
            ui.selected_scene = key;
#ifdef __ANDROID__
            // IPC-only console on Android (the kid launcher is the visible library): a tap loads.
            if (!active) {
                call("library.open", {{"path", game["path"]}});
            }
#endif
        }
        ImGui::EndGroup();
        ImGui::PopID();
        ++shown;
    }
    if (!shown) {
        label(ui.status.value("scanning", false)
                  ? tr("player.finding_your_games", "Finding your games...")
                  : tr("player.your_next_game_starts_here", "Your next game starts here."),
              mint);
        wrapped(
            tr("player.add_a_folder_containing_rtp_projects_scanning_never_opens_a_project",
               "Add a folder containing .rtp projects. Scanning never opens a project."));
    }
    if (ImGui::CollapsingHeader(trId("player.library_folders", "Library folders"))) {
        ImGui::SetNextItemWidth(-110);
        ImGui::InputTextWithHint("##folder", tr("player.folder_path", "Folder path"), ui.folder,
                                 sizeof(ui.folder));
        ImGui::SameLine();
        if (action(trId("player.add_path", "Add path"), ImVec2(98, 30))) {
            call("library.add", {{"path", ui.folder}});
        }
        for (const auto &folder : ui.status["preferences"]["roots"]) {
            ImGui::PushID(folder.get<std::string>().c_str());
            if (smallAction(trId("player.remove", "Remove"))) {
                call("library.remove", {{"path", folder}});
            }
            ImGui::SameLine();
            ImGui::TextWrapped("%s", folder.get<std::string>().c_str());
            ImGui::PopID();
        }
    }
    for (const auto &warning : ui.status.value("scan_warnings", json::array())) {
        ImGui::TextWrapped("%s", translatedMessage(warning.get<std::string>()).c_str());
    }
}
} // namespace PlayerConsole
