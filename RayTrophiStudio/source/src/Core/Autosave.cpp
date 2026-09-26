#include "Autosave.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "Template/StartupPreferences.h"
#include "globals.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <thread>

namespace raytrophi::autosave {
namespace {

using Clock = std::chrono::steady_clock;

std::mutex g_mutex;
Status     g_status;
bool       g_haveLastWrite = false;
Clock::time_point g_lastWrite;
Clock::time_point g_lastTick;
bool       g_haveLastTick = false;

// Background writer. Only one at a time; `g_finished` lets the UI see an
// edge ("a write just ended") without taking g_mutex every frame.
std::atomic<bool>     g_writing{false};
std::atomic<uint64_t> g_finished{0};

std::filesystem::path autosavePath() {
    return raytrophi::templates::StartupPreferencesManager::instance().getAutosavePath();
}

void refreshFileFacts(Status& s, const std::filesystem::path& p) {
    std::error_code ec;
    s.path = p.string();
    s.file_exists = std::filesystem::exists(p, ec) && !ec;
    s.file_bytes = s.file_exists
        ? static_cast<uint64_t>(std::filesystem::file_size(p, ec))
        : 0u;
    if (ec) s.file_bytes = 0u;
}

}  // namespace

bool writeNow(SceneData& scene, RenderSettings& settings, Renderer& renderer,
              const std::string& reason) {
    const std::filesystem::path path = autosavePath();

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    // ★★★★ saveProjectCopy, not saveProject. saveProject retargets the
    //   project's identity (current path, name, recent list, modified flag)
    //   and clears texture save-dirty flags. The autosave used to swap the
    //   identity back afterwards, which only worked while it ran on the main
    //   thread: from a background thread a Ctrl+S issued mid-write would read
    //   the autosave path and save the user's work into the recovery file.
    const auto t0 = Clock::now();
    bool ok = false;
    std::string err;
    try {
        ok = ProjectManager::getInstance().saveProjectCopy(
            path.string(), scene, settings, renderer);
    } catch (const std::exception& e) {
        err = e.what();
    } catch (...) {
        err = "unknown exception";
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_status.last_ok = ok;
        g_status.last_error = ok ? std::string() : (err.empty() ? "saveProjectCopy returned false" : err);
        g_status.last_reason = reason;
        g_status.last_write_ms = ms;
        if (ok) {
            ++g_status.write_count;
            g_lastWrite = Clock::now();
            g_haveLastWrite = true;
        }
        refreshFileFacts(g_status, path);
    }

    if (ok) {
        SCENE_LOG_INFO("[Autosave] wrote " + path.string() + " (" + reason + ", " +
                       std::to_string(static_cast<int>(ms)) + " ms)");
    } else {
        SCENE_LOG_ERROR("[Autosave] FAILED (" + reason + "): " +
                        (err.empty() ? "saveProjectCopy returned false" : err));
    }
    return ok;
}

void tick(SceneData& scene, RenderSettings& settings, Renderer& renderer, bool busy) {
    auto& prefs = raytrophi::templates::StartupPreferencesManager::instance();
    const bool enabled = prefs.isAutoSaveEnabled();
    const int  interval = prefs.getAutoSaveIntervalSec();

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_status.enabled = enabled;
        g_status.interval_sec = interval;
        g_status.scene_is_modified = g_project.is_modified;
    }

    // A running write (ours or a user save) is "busy" too: never queue a
    // second copy behind it.
    if (!enabled || interval <= 0 || busy || g_writing.load(std::memory_order_acquire) ||
        ProjectManager::getInstance().isSaveInProgress()) {
        return;
    }

    const auto now = Clock::now();
    if (!g_haveLastTick) {
        // Ilk kare referans noktasidir; acilisin hemen ardindan yazmayiz.
        g_haveLastTick = true;
        g_lastTick = now;
        g_lastWrite = now;
        return;
    }
    g_lastTick = now;

    const double since = std::chrono::duration<double>(now - g_lastWrite).count();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_status.seconds_since_last_write = g_haveLastWrite
            ? std::chrono::duration<double>(now - g_lastWrite).count()
            : -1.0;
        g_status.seconds_until_next = static_cast<double>(interval) - since;
    }
    if (since < static_cast<double>(interval)) return;

    // ★★ Degismemis sahneyi tekrar yazmak buyuk projelerde bedava degil. Ama
    //   `is_modified` yanlissa bu kosul autosave'i SESSIZCE olduran sey olur,
    //   o yuzden atlama AYRI sayilir ve status'te gorunur.
    if (!g_project.is_modified) {
        std::lock_guard<std::mutex> lock(g_mutex);
        ++g_status.skipped_unmodified;
        g_lastWrite = now;  // sayaci ilerlet, her karede tekrar denemesin
        return;
    }

    // ★★★ Off the main thread, like the menu's Ctrl+S. Measured 2026-09-25:
    //   on the main thread one interval write stalled the app for 122.8 s
    //   (texture re-encode) and read as "the fluid sim freezes sometimes".
    //   The same trade-off as the menu save applies: the scene is read while
    //   the loop keeps running, which is why `busy` includes playback (the sim
    //   would be mutating what is being written).
    // Restart the interval now so a slow write is not immediately re-armed.
    g_lastWrite = now;
    g_writing.store(true, std::memory_order_release);
    std::thread([&scene, &settings, &renderer]() {
        writeNow(scene, settings, renderer, "interval");
        g_writing.store(false, std::memory_order_release);
        g_finished.fetch_add(1, std::memory_order_acq_rel);
    }).detach();
}

Progress progress() {
    Progress p;
    p.writing = g_writing.load(std::memory_order_acquire);
    p.finished = g_finished.load(std::memory_order_acquire);
    std::lock_guard<std::mutex> lock(g_mutex);
    p.last_ok = g_status.last_ok;
    p.last_write_ms = g_status.last_write_ms;
    p.last_error = g_status.last_error;
    return p;
}

void setEnabled(bool enabled) {
    raytrophi::templates::StartupPreferencesManager::instance().setAutoSaveEnabled(enabled);
}

bool setIntervalSec(int seconds) {
    if (seconds < kMinIntervalSec || seconds > kMaxIntervalSec) return false;
    raytrophi::templates::StartupPreferencesManager::instance().setAutoSaveIntervalSec(seconds);
    return true;
}

Status status() {
    std::lock_guard<std::mutex> lock(g_mutex);
    Status s = g_status;
    refreshFileFacts(s, autosavePath());
    s.scene_is_modified = g_project.is_modified;
    s.writing = g_writing.load(std::memory_order_acquire);
    // tick() refreshes these; report the preference itself so a change made
    // over IPC is visible before the next frame.
    auto& prefs = raytrophi::templates::StartupPreferencesManager::instance();
    s.enabled = prefs.isAutoSaveEnabled();
    s.interval_sec = prefs.getAutoSaveIntervalSec();
    return s;
}

}  // namespace raytrophi::autosave
