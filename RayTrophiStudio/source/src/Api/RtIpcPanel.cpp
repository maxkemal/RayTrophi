#include "Api/RtIpcPanel.h"
#include "Api/RtIpc.h"

#include "imgui.h"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace rtipc_panel {
namespace {

std::string g_one_time_token;
char g_token_name[96] = "Remote operator";
char g_cidrs[256] = "127.0.0.1/32";
int g_capabilities = 1; // read
int64_t g_expires_at = 0;
std::string g_edit_token_id;

std::vector<std::string> splitCidrs() {
    std::vector<std::string> result;
    std::istringstream input(g_cidrs);
    std::string value;
    while (std::getline(input, value, ',')) {
        value.erase(0, value.find_first_not_of(" \t"));
        const size_t end = value.find_last_not_of(" \t");
        if (end != std::string::npos) value.erase(end + 1);
        if (!value.empty()) result.push_back(value);
    }
    return result;
}

void drawSessions() {
    const auto sessions = rtipc::listSessions(true);
    if (ImGui::Button("Disconnect all")) rtipc::disconnectAllSessions();
    ImGui::SameLine(); ImGui::TextDisabled("%zu retained session(s)", sessions.size());
    if (ImGui::BeginTable("ipc_sessions", 8,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
        for (const char* heading : {"State", "Transport", "Peer", "TLS", "Token", "Requests", "Errors", "Action"})
            ImGui::TableSetupColumn(heading);
        ImGui::TableHeadersRow();
        for (const auto& session : sessions) {
            ImGui::PushID(session.connection_id.c_str()); ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(session.active ? "active" : "closed");
            ImGui::TableNextColumn(); ImGui::TextUnformatted(session.transport.c_str());
            ImGui::TableNextColumn(); ImGui::Text("%s:%u", session.peer_address.c_str(), session.peer_port);
            ImGui::TableNextColumn(); ImGui::Text("%s %s", session.tls_version.c_str(), session.tls_cipher.c_str());
            ImGui::TableNextColumn(); ImGui::TextUnformatted(session.token_id.c_str());
            ImGui::TableNextColumn(); ImGui::Text("%llu", static_cast<unsigned long long>(session.request_count));
            ImGui::TableNextColumn(); ImGui::Text("%llu", static_cast<unsigned long long>(session.error_count));
            ImGui::TableNextColumn();
            if (session.active && ImGui::SmallButton("Disconnect"))
                rtipc::disconnectSession(session.connection_id);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void capability(const char* label, int bit) {
    bool enabled = (g_capabilities & bit) != 0;
    if (ImGui::Checkbox(label, &enabled))
        g_capabilities = enabled ? (g_capabilities | bit) : (g_capabilities & ~bit);
}

void drawTokens() {
    bool open_secret_popup = false;
    ImGui::InputText("Display name", g_token_name, sizeof(g_token_name));
    ImGui::InputText("Allowed IPv4 CIDRs", g_cidrs, sizeof(g_cidrs));
    ImGui::InputScalar("Expires at (Unix, 0=never)", ImGuiDataType_S64, &g_expires_at);
    capability("Read", 1 << 0); ImGui::SameLine(); capability("Scene write", 1 << 1);
    capability("Render", 1 << 2); ImGui::SameLine(); capability("Files read", 1 << 3);
    capability("Files write", 1 << 4); ImGui::SameLine(); capability("Scripts", 1 << 5);
    capability("Addons", 1 << 6); ImGui::SameLine(); capability("Admin", 1 << 7);
    if (ImGui::Button("Create token")) {
        rtipc::TokenInfo info; std::string error;
        if (!rtipc::createToken(g_token_name, static_cast<uint32_t>(g_capabilities),
                                splitCidrs(), g_expires_at, info, g_one_time_token, error))
            g_one_time_token = "ERROR: " + error;
        open_secret_popup = true;
    }
    ImGui::SeparatorText("Token vault");
    for (const auto& token : rtipc::listTokens()) {
        ImGui::PushID(token.id.c_str());
        ImGui::Text("%s [%s] scopes=0x%02x%s", token.display_name.c_str(), token.id.c_str(),
                    token.capabilities, token.revoked ? " REVOKED" : "");
        ImGui::SameLine();
        if (!token.revoked && ImGui::SmallButton("Revoke")) {
            std::string error; rtipc::revokeToken(token.id, error);
        }
        ImGui::SameLine();
        if (!token.revoked && ImGui::SmallButton("Rotate")) {
            rtipc::TokenInfo replacement; std::string error;
            if (!rtipc::rotateToken(token.id, replacement, g_one_time_token, error))
                g_one_time_token = "ERROR: " + error;
            open_secret_popup = true;
        }
        ImGui::SameLine();
        if (!token.revoked && ImGui::SmallButton("Edit")) {
            g_edit_token_id = token.id; g_capabilities = static_cast<int>(token.capabilities);
            g_expires_at = token.expires_at;
            std::string joined;
            for (const auto& cidr : token.allowed_cidrs) {
                if (!joined.empty()) joined += ",";
                joined += cidr;
            }
            std::snprintf(g_cidrs, sizeof(g_cidrs), "%s", joined.c_str());
        }
        ImGui::PopID();
    }
    if (!g_edit_token_id.empty()) {
        ImGui::SeparatorText("Edit selected token policy");
        ImGui::TextUnformatted(g_edit_token_id.c_str());
        if (ImGui::Button("Save policy")) {
            std::string error;
            if (rtipc::updateToken(g_edit_token_id, static_cast<uint32_t>(g_capabilities),
                                   splitCidrs(), g_expires_at, error)) g_edit_token_id.clear();
            else { g_one_time_token = "ERROR: " + error; open_secret_popup = true; }
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel edit")) g_edit_token_id.clear();
    }
    if (open_secret_popup) ImGui::OpenPopup("One-time token secret");
    if (ImGui::BeginPopupModal("One-time token secret", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%s", g_one_time_token.c_str());
        if (ImGui::Button("Copy secret")) ImGui::SetClipboardText(g_one_time_token.c_str());
        ImGui::SameLine();
        if (ImGui::Button("Dismiss")) { g_one_time_token.clear(); ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
}

void drawAudit() {
    static char filter[128]{};
    static bool denied_only = false;
    ImGui::InputText("Filter method/peer/token", filter, sizeof(filter));
    ImGui::SameLine(); ImGui::Checkbox("Denied only", &denied_only);
    if (ImGui::Button("Clear audit")) rtipc::clearAuditEvents();
    const auto events = rtipc::recentAuditEvents(512);
    ImGui::SameLine(); ImGui::TextDisabled("%zu event(s)", events.size());
    if (ImGui::BeginTable("ipc_audit", 7,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
        for (const char* heading : {"Seq", "Peer", "Token", "Method", "Decision", "Outcome", "us"})
            ImGui::TableSetupColumn(heading);
        ImGui::TableHeadersRow();
        for (auto it = events.rbegin(); it != events.rend(); ++it) {
            if (denied_only && it->allowed) continue;
            if (*filter && it->method.find(filter) == std::string::npos &&
                it->peer_address.find(filter) == std::string::npos &&
                it->token_id.find(filter) == std::string::npos) continue;
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%llu", static_cast<unsigned long long>(it->sequence));
            ImGui::TableNextColumn(); ImGui::TextUnformatted(it->peer_address.c_str());
            ImGui::TableNextColumn(); ImGui::TextUnformatted(it->token_id.c_str());
            ImGui::TableNextColumn(); ImGui::TextUnformatted(it->method.c_str());
            ImGui::TableNextColumn(); ImGui::TextUnformatted(it->allowed ? "allow" : "deny");
            ImGui::TableNextColumn(); ImGui::TextUnformatted(it->outcome.c_str());
            ImGui::TableNextColumn(); ImGui::Text("%llu", static_cast<unsigned long long>(it->duration_us));
        }
        ImGui::EndTable();
    }
}

} // namespace

void draw(bool* open) {
    if (!open || !*open) return;
    if (!ImGui::Begin("Remote IPC Control", open)) { ImGui::End(); return; }
    const auto remote = rtipc::remoteStatus();
    ImGui::Text("Local IPC: %s", rtipc::isRunning() ? "running" : "stopped");
    ImGui::SameLine(); ImGui::Text("TLS: %s", remote.running ? "running" : "stopped");
    if (!remote.bind_address.empty()) {
        ImGui::Text("Listener: %s:%u", remote.bind_address.c_str(), remote.port);
        ImGui::TextWrapped("Certificate: %s", remote.certificate_path.c_str());
        ImGui::TextWrapped("Expires: %s", remote.certificate_not_after.c_str());
        ImGui::TextWrapped("SHA-256: %s", remote.certificate_sha256.c_str());
        for (const auto& san : remote.subject_alt_names) ImGui::BulletText("%s", san.c_str());
    }
    if (rtipc::isRemoteRunning()) {
        ImGui::SameLine();
        if (ImGui::Button("Disable Remote Access")) rtipc::disableRemoteAccess();
    }
    if (ImGui::BeginTabBar("ipc_control_tabs")) {
        if (ImGui::BeginTabItem("Sessions")) { drawSessions(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Tokens")) { drawTokens(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Audit")) { drawAudit(); ImGui::EndTabItem(); }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

} // namespace rtipc_panel
