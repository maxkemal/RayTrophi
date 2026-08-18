#include "Template/StartupPreferences.h"
#include "Template/PathUtils.h"
#include <fstream>
#include <iostream>

namespace raytrophi::templates {
using pathutils::pathFromUtf8;
using pathutils::pathToUtf8;

StartupPreferencesManager& StartupPreferencesManager::instance() {
    static StartupPreferencesManager mgr;
    return mgr;
}

StartupPreferencesManager::StartupPreferencesManager() {
    config_file_path_ = pathFromUtf8("assets/config/startup_preferences.json");
    autosave_path_ = pathFromUtf8("assets/config/autosave.rtp");
    load();
}

void StartupPreferencesManager::load() {
    try {
        if (std::filesystem::exists(config_file_path_)) {
            std::ifstream file(config_file_path_);
            if (file) {
                nlohmann::json j;
                file >> j;
                std::string mode = j.value("startup_mode", "hub");
                setStartupModeFromString(mode);
                auto_save_enabled_ = j.value("auto_save_enabled", true);
                auto_save_interval_sec_ = j.value("auto_save_interval_sec", 300);
            }
        }
    } catch (...) {
        startup_mode_ = StartupMode::ShowHub;
    }
}

void StartupPreferencesManager::save() {
    try {
        std::filesystem::create_directories(config_file_path_.parent_path());
        nlohmann::json j;
        j["startup_mode"] = getStartupModeString();
        j["auto_save_enabled"] = auto_save_enabled_;
        j["auto_save_interval_sec"] = auto_save_interval_sec_;

        std::ofstream file(config_file_path_);
        if (file) {
            file << j.dump(2);
        }
    } catch (...) {}
}

void StartupPreferencesManager::setStartupMode(StartupMode mode) {
    startup_mode_ = mode;
    save();
}

std::string StartupPreferencesManager::getStartupModeString() const {
    switch (startup_mode_) {
        case StartupMode::OpenLastProject: return "last";
        case StartupMode::StartEmpty:      return "empty";
        case StartupMode::RestoreAutosave: return "restore";
        case StartupMode::ShowHub:
        default:                            return "hub";
    }
}

void StartupPreferencesManager::setStartupModeFromString(const std::string& mode_str) {
    if (mode_str == "last") {
        startup_mode_ = StartupMode::OpenLastProject;
    } else if (mode_str == "empty") {
        startup_mode_ = StartupMode::StartEmpty;
    } else if (mode_str == "restore") {
        startup_mode_ = StartupMode::RestoreAutosave;
    } else {
        startup_mode_ = StartupMode::ShowHub;
    }
}

bool StartupPreferencesManager::hasValidAutosave() const {
    return std::filesystem::exists(autosave_path_) && std::filesystem::file_size(autosave_path_) > 0;
}

} // namespace raytrophi::templates
