#pragma once
// Thin OS layer for the Android port (docs/dev/PLAYER_ANDROID_PORT.md, batch A1).
//
// Contract (owned by Claude, implemented in source/src/Platform/PlatformOs.cpp; not "Platform.cpp": ozz-animation platform.cc would share platform.obj (LNK4042, one object silently dropped)):
//  - Windows behaviour MUST stay byte-identical to the code each call replaces.
//  - No function fakes success: a failure returns an empty path / 0 / nullptr and
//    the caller decides; nothing here logs on the hot path.
//  - Header stays free of <windows.h> and <android/...> so it can be included anywhere
//    (windows.h leaks min/max and near/far macros into the codebase).
//  - Paths are UTF-8 std::filesystem::path; on Windows they are built from the wide API.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace Platform {

// Folder of the running executable (Windows: GetModuleFileNameW parent).
// Android: there is no executable folder; returns appDataRoot() so existing
// "next to the exe" lookups (shaders/, scenes/, assets/) resolve to the extracted APK assets.
std::filesystem::path executableDir();

// Full path of the running executable (Windows only; Android returns empty).
// Kept separate from executableDir() because a few callers hash/verify the binary itself.
std::filesystem::path executablePath();

// Writable per-app data root. Windows: executableDir() (portable Player layout).
// Android: SDL_AndroidGetInternalStoragePath(). Asset extraction (A4) writes here.
std::filesystem::path appDataRoot();

// Resident set size of this process in bytes, 0 if unknown.
// Windows: PROCESS_MEMORY_COUNTERS::WorkingSetSize. POSIX: /proc/self/statm resident * page size.
std::uint64_t processResidentBytes();

// Physical RAM of the machine, in bytes; 0 if unknown (callers keep their own fallback).
// Windows: GlobalMemoryStatusEx ullTotalPhys / ullAvailPhys. POSIX: sysconf pages * page size
// (Android "available" = MemAvailable-like free pages; coarse, but the same use: a budget hint).
std::uint64_t systemTotalMemoryBytes();
std::uint64_t systemAvailableMemoryBytes();

// Numeric id of this process (Windows: GetCurrentProcessId, POSIX: getpid).
std::uint32_t processId();

// Dynamic library handle helpers (Windows: LoadLibraryW/GetProcAddress/FreeLibrary,
// POSIX: dlopen(RTLD_NOW|RTLD_LOCAL)/dlsym/dlclose). Name is passed as-is (caller picks
// "vulkan-1.dll" vs "libvulkan.so"). Returns nullptr on failure.
void* loadLibrary(const char* name);
void* findSymbol(void* library, const char* symbol);
void  unloadLibrary(void* library);

// Fill `out` with `size` bytes from the OS cryptographic RNG. False on failure (out is then
// unspecified; callers must not use it). Windows: BCryptGenRandom(system preferred RNG).
// POSIX: getrandom(2), retried on EINTR/partial reads.
bool secureRandomBytes(void* out, std::size_t size);

// Packaged files that are not plain files on disk (Android: inside the APK) are copied into
// appDataRoot() so the rest of the code reads them with std::filesystem as on Windows.
// Android: reads "rt_asset_manifest.txt" from the APK assets (line 1 "stamp <hex>", then one
// relative path per line, written by scripts/stage_player_android_assets.py). Skips the copy
// when the stored stamp matches; deletes files the previous manifest listed and the new one
// does not; writes the stamp LAST so an interrupted copy is redone on the next launch.
// Files the app creates itself (settings, logs) are never touched.
// Windows/other: nothing is bundled, returns true with an empty report.
// `report` gets one human-readable line for the log either way (empty when nothing happened).
bool extractBundledAssets(std::string& report);

// True when child processes may be spawned (Windows). Android: false — callers must
// return an explicit "not supported on this platform" error, never a fake success.
bool supportsChildProcesses();

} // namespace Platform
