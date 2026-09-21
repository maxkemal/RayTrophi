#include "Autosave.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "Template/StartupPreferences.h"
#include "UI/TemplateHubUI.h"
#include "globals.h"

#include <chrono>
#include <filesystem>
#include <mutex>

namespace raytrophi::autosave {
namespace {

using Clock = std::chrono::steady_clock;

std::mutex g_mutex;
Status     g_status;
bool       g_haveLastWrite = false;
Clock::time_point g_lastWrite;
Clock::time_point g_lastTick;
bool       g_haveLastTick = false;

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

    // ★★★★ Kimligi kaydin ETRAFINDA koru. `saveProject` `current_file_path`i
    //   kendi yoluna cevirir, "Untitled" ise proje adini dosya adindan uretir
    //   ve yolu son projeler listesine ekler. Bunlar bir KULLANICI kaydi icin
    //   dogru, bir autosave icin felakettir: kullanicinin bir sonraki Ctrl+S'i
    //   kendi projesine degil autosave.rtp'ye giderdi.
    const std::string prevPath = g_project.current_file_path;
    const std::string prevName = g_project.project_name;
    const bool        prevModified = g_project.is_modified;

    const auto t0 = Clock::now();
    bool ok = false;
    std::string err;
    try {
        ok = ProjectManager::getInstance().saveProject(
            path.string(), scene, settings, renderer);
    } catch (const std::exception& e) {
        err = e.what();
    } catch (...) {
        err = "unknown exception";
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    g_project.current_file_path = prevPath;
    g_project.project_name = prevName;
    // ★★ "kaydedildi" bayragi da geri yuklenir: autosave kullanicinin
    //   kaydedilmemis degisikligini KAYDEDILMIS gostermemeli, yoksa cikista
    //   "kaydetmek ister misiniz" sorusu sorulmaz ve is sessizce kaybolur.
    g_project.is_modified = prevModified;
    raytrophi::templates::TemplateHubUI::instance().removeRecentProject(path);

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_status.last_ok = ok;
        g_status.last_error = ok ? std::string() : (err.empty() ? "saveProject returned false" : err);
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
        SCENE_LOG_INFO("[Autosave] wrote " + path.string() + " (" + reason + ")");
    } else {
        SCENE_LOG_ERROR("[Autosave] FAILED (" + reason + "): " +
                        (err.empty() ? "saveProject returned false" : err));
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

    if (!enabled || interval <= 0 || busy) return;

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

    writeNow(scene, settings, renderer, "interval");
}

Status status() {
    std::lock_guard<std::mutex> lock(g_mutex);
    Status s = g_status;
    refreshFileFacts(s, autosavePath());
    s.scene_is_modified = g_project.is_modified;
    return s;
}

}  // namespace raytrophi::autosave
