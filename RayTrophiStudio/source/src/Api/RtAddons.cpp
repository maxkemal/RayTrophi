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
#include <cctype>
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

bool validModuleName(const std::string& name) {
    if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) || name[0] == '_')) return false;
    return std::all_of(name.begin() + 1, name.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '_';
    });
}

std::string normalizedModuleName(std::string name) {
    for (char& c : name) if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    if (name.empty() || std::isdigit(static_cast<unsigned char>(name[0]))) name.insert(name.begin(), '_');
    return name;
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
        if (bl.contains("author"))      info.author       = py::cast<std::string>(py::str(bl["author"]));
        if (bl.contains("category"))    info.category     = py::cast<std::string>(py::str(bl["category"]));
        if (bl.contains("location"))    info.location     = py::cast<std::string>(py::str(bl["location"]));
        if (bl.contains("warning"))     info.warning      = py::cast<std::string>(py::str(bl["warning"]));
        if (py::hasattr(mod, "addon_settings")) {
            py::dict settings = mod.attr("addon_settings");
            for (auto item : settings)
                info.settings.emplace_back(py::cast<std::string>(py::str(item.first)),
                                           py::cast<std::string>(py::str(item.second)));
        }
    } catch (...) {
    }
}

void readBlInfoWithoutImport(const std::filesystem::path& init, AddonInfo& info) {
    try {
        py::module_ ast = py::module_::import("ast");
        std::ifstream in(init, std::ios::binary);
        std::string source((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        py::object tree = ast.attr("parse")(source, init.string());
        py::list body = tree.attr("body");
        for (py::handle borrowedNode : body) {
            // Keep owned references for every value obtained through an
            // accessor. Storing node.attr("targets")[0] in a py::handle leaves
            // the handle pointing at the temporary accessor's released object.
            py::object node = py::reinterpret_borrow<py::object>(borrowedNode);
            if (!py::hasattr(node, "targets")) continue;
            py::list targets = node.attr("targets");
            if (py::len(targets) != 1) continue;
            py::object target = targets[0];
            if (!py::hasattr(target, "id") || py::cast<std::string>(target.attr("id")) != "bl_info") continue;
            py::object value = node.attr("value");
            py::dict bl = ast.attr("literal_eval")(value);
            if (bl.contains("name")) info.display_name = py::cast<std::string>(py::str(bl["name"]));
            if (bl.contains("description")) info.description = py::cast<std::string>(py::str(bl["description"]));
            if (bl.contains("version")) info.version = py::cast<std::string>(py::str(bl["version"]));
            if (bl.contains("author")) info.author = py::cast<std::string>(py::str(bl["author"]));
            if (bl.contains("category")) info.category = py::cast<std::string>(py::str(bl["category"]));
            if (bl.contains("location")) info.location = py::cast<std::string>(py::str(bl["location"]));
            if (bl.contains("warning")) info.warning = py::cast<std::string>(py::str(bl["warning"]));
            break;
        }
    } catch (...) {}
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
        if (info.loaded) {
            try {
                py::module_ mod = py::module_::import(name.c_str());
                readBlInfo(mod, info);
            } catch (...) {
            }
        } else readBlInfoWithoutImport(entry.path() / "__init__.py", info);
        result.push_back(std::move(info));
    }
    std::sort(result.begin(), result.end(),
              [](const AddonInfo& a, const AddonInfo& b) { return a.module_name < b.module_name; });
    return result;
}

bool installAddon(const std::string& source_path, std::string& module_name, std::string& error) {
    if (!isInitialized()) { error = "Python runtime is not initialized"; return false; }
    try {
        const std::filesystem::path source(source_path);
        if (!std::filesystem::is_regular_file(source)) { error = "Selected addon file does not exist"; return false; }
        std::filesystem::create_directories(addonsDir());
        std::string ext = source.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".py") {
            module_name = normalizedModuleName(source.stem().string());
            const auto target = addonsDir() / module_name;
            if (std::filesystem::exists(target)) { error = "Addon already installed: " + module_name; return false; }
            std::filesystem::create_directories(target);
            std::ifstream in(source, std::ios::binary);
            std::ofstream out(target / "__init__.py", std::ios::binary);
            out << "bl_info = {\"name\": \"" << module_name << "\", \"description\": \"Installed script addon\", \"version\": (1, 0, 0)}\naddon_settings = {}\n"
                   "import runpy\nfrom pathlib import Path\n_instance = None\ndef register():\n    global _instance, bl_info, addon_settings\n    _instance = runpy.run_path(str(Path(__file__).with_name(\"script.py\")), run_name=__name__ + \".script\")\n    bl_info = _instance.get(\"bl_info\", bl_info)\n    addon_settings = _instance.get(\"addon_settings\", addon_settings)\n    fn = _instance.get(\"register\")\n    if callable(fn): fn()\n"
                   "def unregister():\n    global _instance\n    fn = (_instance or {}).get(\"unregister\")\n    if callable(fn): fn()\n    _instance = None\n";
            std::filesystem::copy_file(source, target / "script.py");
        } else if (ext == ".zip") {
            py::module_ zipfile = py::module_::import("zipfile");
            py::object zip = zipfile.attr("ZipFile")(source.string(), "r");
            py::list infos = zip.attr("infolist")();
            std::vector<std::string> files;
            for (py::handle item : infos) {
                const std::string name = py::cast<std::string>(item.attr("filename"));
                std::filesystem::path rel = std::filesystem::path(name).lexically_normal();
                if (rel.is_absolute() || name.find(':') != std::string::npos || (!rel.empty() && *rel.begin() == ".."))
                    throw std::runtime_error("Unsafe path in ZIP: " + name);
                if (!py::cast<bool>(item.attr("is_dir")())) files.push_back(name);
            }
            std::string prefix;
            bool rootInit = std::find(files.begin(), files.end(), "__init__.py") != files.end();
            if (!rootInit) {
                for (const auto& f : files) if (std::filesystem::path(f).filename() == "__init__.py") {
                    auto parent = std::filesystem::path(f).parent_path();
                    if (std::distance(parent.begin(), parent.end()) == 1) { prefix = parent.generic_string() + "/"; break; }
                }
            }
            if (!rootInit && prefix.empty()) { error = "ZIP must contain __init__.py at its root or in one top-level folder"; return false; }
            module_name = normalizedModuleName(prefix.empty() ? source.stem().string() : std::filesystem::path(prefix).parent_path().filename().string());
            if (!validModuleName(module_name)) { error = "Invalid addon module name"; return false; }
            const auto target = addonsDir() / module_name;
            if (std::filesystem::exists(target)) { error = "Addon already installed: " + module_name; return false; }
            std::filesystem::create_directories(target);
            try {
                for (const auto& f : files) {
                    if (!prefix.empty() && f.rfind(prefix, 0) != 0) continue;
                    const std::string stripped = prefix.empty() ? f : f.substr(prefix.size());
                    if (stripped.empty()) continue;
                    const auto dest = target / std::filesystem::path(stripped);
                    std::filesystem::create_directories(dest.parent_path());
                    py::bytes data = zip.attr("read")(f);
                    std::string bytes = data;
                    std::ofstream out(dest, std::ios::binary); out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                }
            } catch (...) { std::filesystem::remove_all(target); throw; }
            zip.attr("close")();
        } else { error = "Choose a .py or .zip addon"; return false; }
        logLine("Addon installed: " + module_name);
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool removeAddon(const std::string& module_name, std::string& error) {
    if (!isInitialized()) { error = "Python runtime is not initialized"; return false; }
    if (!validModuleName(module_name)) { error = "Invalid addon module name"; return false; }
    if (!disableAddon(module_name, error)) return false;
    try {
        py::module_::import("sys").attr("modules").attr("pop")(module_name, py::none());
        const auto target = addonsDir() / module_name;
        if (!isAddonFolder(target)) { error = "Addon folder not found"; return false; }
        std::filesystem::remove_all(target);
        logLine("Addon removed: " + module_name);
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
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
