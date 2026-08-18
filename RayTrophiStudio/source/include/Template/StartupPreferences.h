#pragma once

#include <string>
#include <filesystem>
#include <json.hpp>

namespace raytrophi::templates {

enum class StartupMode {
    ShowHub = 0,     // Always Show Template Hub (Default)
    OpenLastProject, // Automatically open the last project from recent list
    StartEmpty,      // Start with empty production canvas
    RestoreAutosave  // Restore last autosaved session
};

class StartupPreferencesManager {
public:
    static StartupPreferencesManager& instance();

    void load();
    void save();

    StartupMode getStartupMode() const { return startup_mode_; }
    void setStartupMode(StartupMode mode);

    std::string getStartupModeString() const;
    void setStartupModeFromString(const std::string& mode_str);

    std::filesystem::path getAutosavePath() const { return autosave_path_; }
    bool hasValidAutosave() const;

    bool isAutoSaveEnabled() const { return auto_save_enabled_; }
    void setAutoSaveEnabled(bool enabled) { auto_save_enabled_ = enabled; save(); }

    int getAutoSaveIntervalSec() const { return auto_save_interval_sec_; }
    void setAutoSaveIntervalSec(int sec) { auto_save_interval_sec_ = sec; save(); }

private:
    StartupPreferencesManager();
    ~StartupPreferencesManager() = default;

    StartupMode startup_mode_ = StartupMode::ShowHub;
    bool auto_save_enabled_ = true;
    int auto_save_interval_sec_ = 300; // 5 minutes default

    std::filesystem::path config_file_path_;
    std::filesystem::path autosave_path_;
};

} // namespace raytrophi::templates
