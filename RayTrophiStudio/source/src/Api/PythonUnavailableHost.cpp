// Embedded Python runtime: Player (no-Python) link alternative.
//
// PLAYER BUILD ONLY. Studio links the real runtime (RtPython.cpp); the two must
// never be linked into the same target (Studio lists this file as None).
// Only the entry points shared code reaches are defined here: rtapi reports
// "not initialized" and script execution fails with an explicit error, so IPC
// callers of script.run_file get a clear refusal instead of a silent no-op.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md (Faz 3b).

#include "Api/RtPython.h"

namespace rtpython {

bool isInitialized() {
    return false;
}

ExecutionResult executeFile(const std::string& filepath) {
    ExecutionResult result;
    result.ok = false;
    result.error = "Python scripting is not available in RayTrophi Player: " + filepath;
    return result;
}

// Addons are Python modules: none exist without the runtime.
std::vector<AddonInfo> listAddons() {
    return {};
}

static bool addonsUnavailable(const std::string& module_name, std::string& error) {
    error = "Python addons are not available in RayTrophi Player: " + module_name;
    return false;
}

bool enableAddon(const std::string& module_name, std::string& error) {
    return addonsUnavailable(module_name, error);
}

bool disableAddon(const std::string& module_name, std::string& error) {
    return addonsUnavailable(module_name, error);
}

bool reloadAddon(const std::string& module_name, std::string& error) {
    return addonsUnavailable(module_name, error);
}

} // namespace rtpython
