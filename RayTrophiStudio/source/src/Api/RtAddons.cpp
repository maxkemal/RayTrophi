/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtAddons.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* Addon loader (Faz 4a). An addon is a folder under scripts/addons/<name>/ with
* an __init__.py exposing register()/unregister() (Blender-style contract).
* State lives in two in-memory sets mirrored to addon_state.json; all functions
* run on the main thread with the GIL held (same as rtpython::execute()).
*
* Split out of RtPython.cpp, which had grown past the 2000-line working limit.
* This file owns discovery/enable/disable/reload only — the interpreter itself,
* the console and the `rt` module stay in RtPython.cpp, and the addon UI surface
* is in RtUi.cpp.
*/

#include "Api/RtPython.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#include <pybind11/embed.h>

#include "json.hpp"

namespace py = pybind11;

namespace rtpython {
namespace {

std::set<std::string> g_enabled_addons;   // persisted enable set
std::set<std::string> g_loaded_addons;    // register() called this session

// Console helper. RtPython.cpp owns the buffer; appendConsoleText is its public
// entry point, so this file needs no access to its internals.
void logLine(const std::string& text) {
    appendConsoleText(text + "\n");
}

std::filesystem::path addonsDir() {
    return std::filesystem::current_path() / "scripts" / "addons";
}
std::filesystem::path addonStatePath() {
    return std::filesystem::current_path() / "addon_state.json";
}

void loadAddonState() {
    g_enabled_addons.clear();
    std::error_code ec;
    const auto path = addonStatePath();
    if (!std::filesystem::is_regular_file(path, ec)) return;
    try {
        std::ifstream in(path);
        nlohmann::json j;
        in >> j;
        if (j.contains("enabled") && j["enabled"].is_array()) {
            for (const auto& e : j["enabled"]) g_enabled_addons.insert(e.get<std::string>());
        }
    } catch (...) {
        // Corrupt state file: start from an empty enabled set rather than crash.
    }
}

void saveAddonState() {
    try {
        nlohmann::json j;
        j["enabled"] = nlohmann::json::array();
        for (const auto& n : g_enabled_addons) j["enabled"].push_back(n);
        std::ofstream out(addonStatePath());
        out << j.dump(2);
    } catch (...) {
    }
}

bool isAddonFolder(const std::filesystem::path& dir) {
    std::error_code ec;
    return std::filesystem::is_directory(dir, ec) &&
           std::filesystem::is_regular_file(dir / "__init__.py", ec);
}

void ensureAddonsOnPath() {
    py::module_ sys = py::module_::import("sys");
    py::list path = sys.attr("path");
    const std::string dir = addonsDir().string();
    for (auto item : path) {
        if (py::cast<std::string>(item) == dir) return;
    }
    path.insert(0, dir);
}

// import + register(); does not touch persisted state (callers own that).
bool registerAddon(const std::string& name, std::string& error) {
    try {
        ensureAddonsOnPath();
        py::module_ mod = py::module_::import(name.c_str());
        if (py::hasattr(mod, "register")) mod.attr("register")();
        g_loaded_addons.insert(name);
        return true;
    } catch (const py::error_already_set& e) {
        error = e.what();
        return false;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool unregisterAddon(const std::string& name, std::string& error) {
    try {
        if (g_loaded_addons.count(name)) {
            py::module_ mod = py::module_::import(name.c_str());  // sys.modules cache hit
            if (py::hasattr(mod, "unregister")) mod.attr("unregister")();
            g_loaded_addons.erase(name);
        }
        return true;
    } catch (const py::error_already_set& e) {
        error = e.what();
        return false;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

void readBlInfo(py::module_& mod, AddonInfo& info) {
    if (!py::hasattr(mod, "bl_info")) return;
    try {
        py::dict bl = mod.attr("bl_info");
        if (bl.contains("name"))        info.display_name = py::cast<std::string>(bl["name"]);
        if (bl.contains("description")) info.description  = py::cast<std::string>(bl["description"]);
        if (bl.contains("version"))     info.version      = py::cast<std::string>(py::str(bl["version"]));
    } catch (...) {
    }
}

} // namespace

std::vector<AddonInfo> listAddons() {
    std::vector<AddonInfo> result;
    if (!isInitialized()) return result;
    std::error_code ec;
    const auto dir = addonsDir();
    if (!std::filesystem::is_directory(dir, ec)) return result;

    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!isAddonFolder(entry.path())) continue;
        const std::string name = entry.path().filename().string();
        AddonInfo info;
        info.module_name = name;
        info.display_name = name;
        info.enabled = g_enabled_addons.count(name) != 0;
        info.loaded  = g_loaded_addons.count(name) != 0;
        // Only read bl_info from already-loaded modules: importing an unloaded one
        // would run its top-level code as a side effect. Its display name shows as
        // the folder name until enabled.
        if (info.loaded) {
            try {
                py::module_ mod = py::module_::import(name.c_str());
                readBlInfo(mod, info);
            } catch (...) {
            }
        }
        result.push_back(std::move(info));
    }
    std::sort(result.begin(), result.end(),
              [](const AddonInfo& a, const AddonInfo& b) { return a.module_name < b.module_name; });
    return result;
}

bool enableAddon(const std::string& module_name, std::string& error) {
    if (!isInitialized()) { error = "Python runtime is not initialized"; return false; }
    if (!registerAddon(module_name, error)) return false;
    g_enabled_addons.insert(module_name);
    saveAddonState();
    logLine("Addon enabled: " + module_name);
    return true;
}

bool disableAddon(const std::string& module_name, std::string& error) {
    if (!isInitialized()) { error = "Python runtime is not initialized"; return false; }
    if (!unregisterAddon(module_name, error)) return false;
    g_enabled_addons.erase(module_name);
    saveAddonState();
    logLine("Addon disabled: " + module_name);
    return true;
}

bool reloadAddon(const std::string& module_name, std::string& error) {
    if (!isInitialized()) { error = "Python runtime is not initialized"; return false; }
    try {
        std::string ignore;
        unregisterAddon(module_name, ignore);  // best-effort; a fresh addon may not be loaded yet
        ensureAddonsOnPath();
        py::module_ importlib = py::module_::import("importlib");
        py::module_ mod = py::module_::import(module_name.c_str());
        importlib.attr("reload")(mod);
        if (py::hasattr(mod, "register")) mod.attr("register")();
        g_loaded_addons.insert(module_name);
        logLine("Addon reloaded: " + module_name);
        return true;
    } catch (const py::error_already_set& e) {
        error = e.what();
        return false;
    }
}

void loadEnabledAddons() {
    if (!isInitialized()) return;
    loadAddonState();
    if (g_enabled_addons.empty()) return;
    ensureAddonsOnPath();
    // Iterate a copy: registerAddon mutates g_loaded_addons (not g_enabled_addons,
    // but copy anyway for clarity/safety against future changes).
    const std::vector<std::string> names(g_enabled_addons.begin(), g_enabled_addons.end());
    for (const auto& name : names) {
        std::string err;
        if (registerAddon(name, err)) {
            logLine("Addon loaded: " + name);
        } else {
            logLine("Addon '" + name + "' failed to load: " + err);
        }
    }
}

void unloadAllAddons() noexcept {
    if (!isInitialized()) return;
    try {
        const std::vector<std::string> names(g_loaded_addons.begin(), g_loaded_addons.end());
        for (const auto& name : names) {
            std::string err;
            unregisterAddon(name, err);
        }
    } catch (...) {
        // Teardown must never throw into renderer shutdown.
    }
}

} // namespace rtpython
