// Studio-only IPC namespaces: Player link alternative.
//
// PLAYER BUILD ONLY. RtIpc.cpp walks one central chain of per-namespace
// dispatchers. A namespace whose implementation is bound to Studio UI state
// (template hub/session) is not compiled into the Player; its dispatcher is
// defined here and reports "not mine", so the request falls through to the
// existing unknown-method error. Fail-closed, same contract as an unlisted
// method. Studio links the real dispatchers and lists this file as None.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md (Faz 3c).

#include "RtIpcTemplates.h"

// template.* : TemplateSession / TemplateHubUI are Studio UI services.
bool dispatchTemplateIpc(const std::string& /*method*/,
                         const nlohmann::json& /*params*/,
                         const RtIpcTemplateEnqueue& /*enqueue*/,
                         nlohmann::json& /*out_result*/) {
    return false;
}
