#include "Import/ImportSettingsUI.h"
#include "Api/RtApi.h"
#include "imgui.h"
namespace rtimport {
void drawImportSettingsMenu() {
    if (!ImGui::BeginMenu("FBX Reader")) return;
    const auto current = rtapi::getFbxReader();
    if (ImGui::MenuItem("Assimp", nullptr, current == "assimp")) rtapi::setFbxReader("assimp");
    if (ImGui::MenuItem("ufbx (static models)", nullptr, current == "ufbx")) rtapi::setFbxReader("ufbx");
    ImGui::Separator();
    ImGui::TextDisabled("Applies to subsequent FBX imports this session.");
    ImGui::TextDisabled("ufbx: skinning and animation are not available yet.");
    ImGui::EndMenu();
}
}
