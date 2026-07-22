/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          Api/RtPython.h
* Date:          July 2026
* License:       MIT
* =========================================================================
*/
#pragma once

#include <string>
#include <vector>
namespace rtpython {

struct ExecutionResult {
    bool ok = false;
    std::string error;
};

// Main-thread lifecycle. initialize() installs the embedded `rt` module and
// redirects Python stdout/stderr to the in-app console.
bool initialize(std::string& error);
void shutdown() noexcept;
bool isInitialized();

ExecutionResult execute(const std::string& source, const std::string& filename = "<console>");
ExecutionResult executeFile(const std::string& filepath);

// Console output is append-only until clearConsoleOutput(). The snapshot API
// avoids exposing Python or ImGui types through this lightweight header.
std::string consoleOutputSnapshot();
void appendConsoleText(const std::string& text);
void clearConsoleOutput();

// ImGui console window. `open` follows the usual ImGui window ownership rule.
void drawConsole(bool* open);
bool wantsInputCapture();

// Dynamic reflection helpers for API Explorer & Auto-complete
std::vector<std::string> getModuleAttributes(const std::string& module_path);
std::string getAttributeDocstring(const std::string& module_path, const std::string& attr);
std::string getAttributeType(const std::string& module_path, const std::string& attr);

// "rt" plus every submodule currently registered on it ("rt.scene", "rt.mesh",
// "rt.anim", ...), discovered by reflection so the Explorer/autocomplete pick up
// new submodules automatically instead of from a hand-maintained list.
std::vector<std::string> getSubmodulePaths();

// ---------------------------------------------------------------------------
// Addons (Faz 4a). An addon is a folder under scripts/addons/<name>/ with an
// __init__.py exposing register()/unregister() (Blender-style contract).
// Enable = import + register(); disable = unregister(). The enabled set is
// persisted to addon_state.json next to the executable and re-applied on
// startup. Optional bl_info dict {"name","description","version"} supplies
// display metadata. All calls run on the main thread with the GIL held.
// ---------------------------------------------------------------------------
struct AddonInfo {
    std::string module_name;    ///< folder name == import name
    std::string display_name;   ///< bl_info["name"], else module_name
    std::string description;
    std::string version;
    bool enabled = false;       ///< persisted enable state
    bool loaded = false;        ///< register() has run this session
    std::string error;          ///< last import/register error, if any
};

std::vector<AddonInfo> listAddons();
bool enableAddon(const std::string& module_name, std::string& error);
bool disableAddon(const std::string& module_name, std::string& error);
bool reloadAddon(const std::string& module_name, std::string& error);
void loadEnabledAddons();       ///< startup: register every persisted-enabled addon
void unloadAllAddons() noexcept; ///< shutdown: unregister loaded addons before teardown

// Faz 4b: draw every panel an addon registered via rt.ui.register_panel().
// Call once per UI frame from the host (between ImGui::NewFrame and Render).
// Each panel's Python draw callback runs here on the main thread; the rt.ui.*
// widget calls are only valid inside that callback.
void drawAddonPanels();

} // namespace rtpython
