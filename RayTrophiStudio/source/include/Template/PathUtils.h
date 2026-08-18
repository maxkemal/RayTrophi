#pragma once
#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace raytrophi::pathutils {

inline std::filesystem::path pathFromUtf8(const std::string& utf8_path) {
    if (utf8_path.empty()) return {};
#ifdef _WIN32
    auto toWide = [](const std::string& src, UINT codepage, DWORD flags) -> std::wstring {
        const int size = MultiByteToWideChar(codepage, flags, src.c_str(), -1, nullptr, 0);
        if (size <= 0) return {};
        std::wstring out(static_cast<size_t>(size - 1), L'\0');
        if (MultiByteToWideChar(codepage, flags, src.c_str(), -1, out.data(), size) <= 0) return {};
        return out;
    };

    std::wstring wide = toWide(utf8_path, CP_UTF8, MB_ERR_INVALID_CHARS);
    if (wide.empty()) {
        wide = toWide(utf8_path, CP_ACP, 0);
    }
    if (!wide.empty()) {
        return std::filesystem::path(wide);
    }
#endif
    return std::filesystem::path(utf8_path);
}

inline std::string pathToUtf8(const std::filesystem::path& p) {
    if (p.empty()) return {};
#ifdef _WIN32
    const std::wstring wide = p.wstring();
    if (wide.empty()) return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) return {};
    std::string out(static_cast<size_t>(size - 1), '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, out.data(), size, nullptr, nullptr) <= 0) return {};
    return out;
#else
    return p.string();
#endif
}

} // namespace raytrophi::pathutils
