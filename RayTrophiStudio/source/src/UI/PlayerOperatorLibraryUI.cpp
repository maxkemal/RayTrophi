#include "PlayerOperatorWidgets.h"
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
#endif

namespace PlayerConsole {
namespace {
SDL_Texture *logo = nullptr;
std::map<std::string, SDL_Texture *> thumbnails;

std::string browseFolder() {
#ifdef _WIN32
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
#else
    return {};
#endif
}
SDL_Texture *thumbnail(SDL_Renderer *renderer, const std::string &path) {
    if (path.empty()) {
        return nullptr;
    }
    const auto found = thumbnails.find(path);
    if (found != thumbnails.end()) {
        return found->second;
    }
    if (thumbnails.size() >= 64) {
        return nullptr;
    }
    // Cache failed loads too, so a broken artwork file is not decoded every frame.
    thumbnails.emplace(path, nullptr);
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
    const float scale = std::min(1.0f, 512.0f / static_cast<float>(std::max(width, height)));
    auto *reduced = SDL_CreateRGBSurfaceWithFormat(0, std::max(1, static_cast<int>(width * scale)),
                                                   std::max(1, static_cast<int>(height * scale)),
                                                   32, SDL_PIXELFORMAT_RGBA32);
    SDL_Texture *texture = nullptr;
    if (source) {
        SDL_SetSurfaceBlendMode(source, SDL_BLENDMODE_NONE);
    }
    if (source && reduced && SDL_BlitScaled(source, nullptr, reduced, nullptr) == 0) {
        texture = SDL_CreateTextureFromSurface(renderer, reduced);
        if (texture) {
            SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND);
        }
    }
    if (source) {
        SDL_FreeSurface(source);
    }
    if (reduced) {
        SDL_FreeSurface(reduced);
    }
    stbi_image_free(pixels);

    thumbnails[path] = texture;
    return texture;
}
std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}
} // namespace

SDL_Texture *brand(SDL_Renderer *renderer) {
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
        logo =
            SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, 64, 64);
        if (logo) {
            SDL_UpdateTexture(logo, nullptr, pixels, 64 * 4);
            SDL_SetTextureBlendMode(logo, SDL_BLENDMODE_BLEND);
        }
        SelectObject(dc, previous);
    }
    if (dc) {
        DeleteDC(dc);
    }
    if (bitmap) {
        DeleteObject(bitmap);
    }
    DestroyIcon(source);
#endif
    return logo;
}
void releaseArtwork() {
    for (const auto &entry : thumbnails) {
        if (entry.second) {
            SDL_DestroyTexture(entry.second);
        }
    }
    thumbnails.clear();
    if (logo) {
        SDL_DestroyTexture(logo);
        logo = nullptr;
    }
}
void games(SDL_Renderer *renderer) {
    heading(tr("player.game_library", "Game library"),
            tr("player.pick_an_experience_for_the_room_or_drop_an_rtp_project_into_player",
               "Pick an experience for the room, or drop an .rtp project into Player."));
    ImGui::SetNextItemWidth(std::max(100.0f, ImGui::GetContentRegionAvail().x - 220));
    ImGui::InputTextWithHint("##search", tr("player.search_your_games", "Search your games"),
                             ui.search, sizeof(ui.search));
    ImGui::SameLine();
    if (action(trId("player.add_folder", "Add folder"), ImVec2(106, 32))) {
        const auto folder = browseFolder();
        if (!folder.empty()) {
            call("library.add", {{"path", folder}});
        }
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(ui.status.value("scanning", false));
    if (action(trId("player.refresh", "Refresh"), ImVec2(88, 32))) {
        call("library.refresh");
    }
    ImGui::EndDisabled();
    ImGui::Spacing();
    const float available = ImGui::GetContentRegionAvail().x;
    const int columns = std::max(1, std::min(3, static_cast<int>((available + 14) / 230)));
    const float width = (available - (columns - 1) * 14) / columns;
    int shown = 0;
    const auto search = lower(ui.search);
    for (const auto &game : ui.status.value("games", json::array())) {
        const auto title = game.value("name", tr("player.untitled", "Untitled"));
        if (!search.empty() && lower(title).find(search) == std::string::npos) {
            continue;
        }
        if (shown % columns) {
            ImGui::SameLine(0, 14);
        }
        ImGui::PushID(game["path"].get<std::string>().c_str());
        ImGui::BeginGroup();
        const auto p = ImGui::GetCursorScreenPos();
        const bool clicked = ImGui::InvisibleButton("card", ImVec2(width, 190));
        const bool hovered = ImGui::IsItemHovered();
        auto *draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(p, add(p, width, 190), hovered ? IM_COL32(37, 49, 71, 255) : panel, 13);
        draw->AddRect(p, add(p, width, 190), hovered ? mint : border, 13, 0, 1);
        if (auto *image = thumbnail(renderer, game.value("thumbnail", ""))) {
            draw->AddImageRounded(static_cast<ImTextureID>(reinterpret_cast<intptr_t>(image)),
                                  add(p, 8, 8), add(p, width - 8, 120), ImVec2(0, 0), ImVec2(1, 1),
                                  IM_COL32_WHITE, 9);
        } else {
            draw->PushClipRect(add(p, 1, 1), add(p, width - 1, 120), true);
            draw->AddRectFilled(add(p, 8, 8), add(p, width - 8, 120), IM_COL32(36, 43, 69, 255), 9);
            const auto category = lower(title);
            if (category.find("paint") != std::string::npos ||
                category.find("balloon") != std::string::npos ||
                category.find("pastel") != std::string::npos) {
                const ImU32 colors[] = {violet, mint, IM_COL32(246, 184, 135, 255)};
                for (int i = 0; i < 3; ++i) {
                    const auto center = add(p, width * (.25f + i * .25f), 50 + (i % 2) * 18.0f);
                    draw->AddCircleFilled(center, 22, colors[i]);
                    draw->AddCircleFilled(add(center, -6, -7), 5, IM_COL32(255, 255, 255, 85));
                    draw->AddBezierCubic(add(center, 0, 22), add(center, -8, 36),
                                         add(center, 8, 44), add(center, 0, 58), colors[i], 1.3f);
                }
            } else {
                draw->AddCircleFilled(add(p, width * .5f, 63), 45, IM_COL32(43, 64, 79, 255));
                icon(draw, add(p, width * .5f, 63), "games", mint, 26);
                draw->AddCircleFilled(add(p, width * .22f, 34), 6, violet);
                draw->AddCircleFilled(add(p, width * .78f, 93), 4, violet);
            }
            draw->PopClipRect();
        }
        draw->PushClipRect(add(p, 14, 130), add(p, width - 10, 180), true);
        draw->AddText(add(p, 14, 132), ink, title.c_str());
        draw->AddText(add(p, 14, 160), muted,
                      game.value("packaged", false)
                          ? tr("player.raytrophi_experience", "RAYTROPHI EXPERIENCE")
                          : tr("player.local_project", "LOCAL PROJECT"));
        draw->PopClipRect();
        if (hovered) {
            ImGui::SetTooltip(tr("player.open_s_s", "Open %s\n%s"), title.c_str(),
                              game["path"].get<std::string>().c_str());
        }
        if (clicked) {
            call("library.open", {{"path", game["path"]}});
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
        ImGui::TextWrapped(
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
            if (ImGui::SmallButton(trId("player.remove", "Remove"))) {
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
