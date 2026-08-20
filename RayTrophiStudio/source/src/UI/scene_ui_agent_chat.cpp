/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          UI/scene_ui_agent_chat.cpp
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 */

#include "scene_ui_agent_chat.hpp"
#include "../Api/RtIpcAudit.h"
#include "../Api/RtAgentLifecycleManager.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <vector>

namespace rtui {

namespace {
// One entry until the transport and the runtime actually support several
// agents. A dropdown listing five roles that all resolve to the same process
// is a panel telling the user something that is not true.
const char* kTargetName = "Agent";

// Calls the agent makes constantly to stay connected. Mirroring them would
// bury the calls that change something.
bool isHeartbeatMethod(const std::string& method) {
    return method == "agent.chat_poll" || method == "agent.chat_send";
}
}  // namespace

AgentChatPanel::AgentChatPanel()
    : show_activity_log(true)
    , scroll_to_bottom(false)
    , reclaim_focus_next_frame(false)
{
    memset(input_buffer, 0, sizeof(input_buffer));
    last_poll_time = std::chrono::steady_clock::now() - std::chrono::hours(1);
    pushMessage(AgentMessageType::SystemEvent, "System", kTargetName,
                "Agent chat ready. Start the agent runtime to connect.");
}

AgentChatPanel::~AgentChatPanel() {
    // ★ Without this the Python runtime outlives Studio: it keeps its pipe
    // client alive, reconnects, and shows up attached to the next session.
    stopAgentProcess();
}

void AgentChatPanel::markPoll() {
    std::lock_guard<std::mutex> lock(msg_mutex);
    last_poll_time = std::chrono::steady_clock::now();
    ever_polled = true;
}

std::string AgentChatPanel::getCurrentTimeStr() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm bt{};
#if defined(_MSC_VER)
    localtime_s(&bt, &time_t_now);
#else
    localtime_r(&time_t_now, &bt);
#endif
    std::ostringstream oss;
    oss << std::put_time(&bt, "%H:%M:%S");
    return oss.str();
}

// ---------------------------------------------------------------------------
// Agent process lifetime
// ---------------------------------------------------------------------------

void AgentChatPanel::startAgentProcess() {
    if (isAgentProcessAlive()) return;
    
    if (AgentLifecycleManager::instance().startAgent(active_agent_id)) {
        pushMessage(AgentMessageType::SystemEvent, "System", kTargetName,
                    "Agent process started via Lifecycle Manager. It reports connected once it polls.");
    } else {
        pushMessage(AgentMessageType::SystemEvent, "System", kTargetName,
                    "Failed to start the agent process. Check if RayTrophiAgent directory exists.");
    }
}

bool AgentChatPanel::isAgentProcessAlive() {
    return AgentLifecycleManager::instance().isAgentRunning(active_agent_id);
}

void AgentChatPanel::stopAgentProcess() {
    AgentLifecycleManager::instance().stopAgent(active_agent_id);
    pushMessage(AgentMessageType::SystemEvent, "System", kTargetName,
                "Agent process stopped.");
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

void AgentChatPanel::pushMessage(AgentMessageType type, const std::string& sender,
                                 const std::string& target, const std::string& content) {
    std::lock_guard<std::mutex> lock(msg_mutex);
    AgentMessage msg;
    msg.id = next_message_id++;
    msg.type = type;
    msg.sender = sender;
    msg.target = target;
    msg.content = content;
    msg.timestamp = getCurrentTimeStr();
    messages.push_back(std::move(msg));
    if (messages.size() > kMaxMessages) {
        const size_t excess = messages.size() - kMaxMessages;
        messages.erase(messages.begin(), messages.begin() + excess);
        dropped_messages += excess;
    }
    scroll_to_bottom = true;
}

void AgentChatPanel::pumpCoreActivity() {
    const std::vector<rtipc_audit::Event> events = rtipc_audit::recent(128);
    if (events.empty()) return;

    // First pass after startup only records where the log is, so opening the
    // panel does not replay everything that happened before it was opened.
    if (!activity_cursor_primed) {
        for (const auto& event : events)
            activity_cursor = (std::max)(activity_cursor, event.sequence);
        activity_cursor_primed = true;
        return;
    }

    uint64_t highest = activity_cursor;
    for (const auto& event : events) {
        if (event.sequence <= activity_cursor) continue;
        highest = (std::max)(highest, event.sequence);
        if (isHeartbeatMethod(event.method)) continue;

        std::string line = event.method.empty() ? std::string("<unparsed request>")
                                                : event.method;
        line += "  ->  " + (event.outcome.empty() ? std::string("ok") : event.outcome);
        if (!event.allowed) line += "  (refused)";
        if (event.duration_us >= 1000)
            line += "  " + std::to_string(event.duration_us / 1000) + " ms";
        pushMessage(AgentMessageType::AgentActivity, "Core", kTargetName, line);
    }
    activity_cursor = highest;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

void AgentChatPanel::draw(bool* p_open) {
    if (!ImGui::Begin("Agent Chat", p_open)) {
        ImGui::End();
        return;
    }

    // Pull the core's audit log even when the feed is hidden, so the cursor
    // stays current and enabling the checkbox does not dump a backlog.
    pumpCoreActivity();

    ImGui::Checkbox("Show Core Activity (IPC)", &show_activity_log);
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Every IPC call the core served, from its audit log - "
                          "not the agent's own account of itself.");

    ImGui::SameLine();
    ImGui::SameLine();

    std::vector<AgentConfig> configs = AgentLifecycleManager::instance().getConfigs();
    AgentConfig current_config;
    AgentLifecycleManager::instance().getConfig(active_agent_id, current_config);
    
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::BeginCombo("##AgentSelector", current_config.name.c_str())) {
        for (const auto& cfg : configs) {
            bool is_selected = (active_agent_id == cfg.id);
            if (ImGui::Selectable(cfg.name.c_str(), is_selected)) {
                if (active_agent_id != cfg.id) {
                    active_agent_id = cfg.id;
                }
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    
    ImGui::SameLine();
    ImGui::SameLine();
    if (isAgentProcessAlive()) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        if (ImGui::Button("Stop Selected Agent")) stopAgentProcess();
        ImGui::PopStyleColor();
    } else {
        if (ImGui::Button("Start Selected Agent")) startAgentProcess();
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Settings")) {
        show_settings_window = true;
    }

    drawStatusLine();
    ImGui::Separator();
    drawMessageList();
    ImGui::Separator();
    drawInputArea();

    ImGui::End();

    if (show_settings_window) {
        drawAgentSettings();
    }
}

void AgentChatPanel::drawStatusLine() {
    bool polling_recently = false;
    bool polled_before = false;
    {
        std::lock_guard<std::mutex> lock(msg_mutex);
        polled_before = ever_polled;
        polling_recently =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - last_poll_time).count() < 5;
    }
    const bool process_alive = isAgentProcessAlive();

    if (last_frame_process_alive && !process_alive) {
        pushMessage(AgentMessageType::SystemEvent, "SYSTEM", "all",
                    "Agent process terminated unexpectedly! Please check RayTrophiAgent/agent.log for errors (e.g. missing python packages like anthropic, or an invalid API key).");
    }
    last_frame_process_alive = process_alive;

    // ★ Three states, not two. "Busy" is not "gone": an agent working through a
    // model turn stops polling for as long as the turn takes.
    if (polling_recently) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Connected - agent is polling.");
    } else if (process_alive && polled_before) {
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.4f, 1.0f),
                           "Busy - agent process is running but has not polled recently.");
    } else if (process_alive) {
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.4f, 1.0f),
                           "Starting - agent process is running, waiting for its first poll.");
    } else if (polled_before) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                           "Disconnected - the agent stopped polling.");
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                           "No agent running. Messages you send are queued until one connects.");
    }
}

void AgentChatPanel::drawMessageList() {
    const float footer_height_to_reserve =
        ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing() + 35.0f;
    ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footer_height_to_reserve), false,
                      ImGuiWindowFlags_HorizontalScrollbar);

    std::lock_guard<std::mutex> lock(msg_mutex);

    if (dropped_messages > 0) {
        ImGui::TextDisabled("... %zu older messages dropped", dropped_messages);
    }

    for (const auto& msg : messages) {
        if (msg.type == AgentMessageType::AgentActivity && !show_activity_log) continue;

        ImGui::PushID(static_cast<int>(msg.id));
        ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x - 10.0f);

        switch (msg.type) {
            case AgentMessageType::UserPrompt:
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "[%s] User:",
                                   msg.timestamp.c_str());
                break;
            case AgentMessageType::AgentReply:
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "[%s] %s:",
                                   msg.timestamp.c_str(), msg.sender.c_str());
                break;
            case AgentMessageType::SystemEvent:
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.4f, 1.0f), "[%s] SYSTEM:",
                                   msg.timestamp.c_str());
                break;
            case AgentMessageType::AgentActivity:
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "[%s] core:",
                                   msg.timestamp.c_str());
                break;
            case AgentMessageType::AgentThought:
                ImGui::TextColored(ImVec4(0.7f, 0.6f, 0.9f, 1.0f), "[%s] %s thinking:",
                                   msg.timestamp.c_str(), msg.sender.c_str());
                break;
        }

        ImGui::TextUnformatted(msg.content.c_str());
        ImGui::PopTextWrapPos();

        if (msg.type == AgentMessageType::AgentReply ||
            msg.type == AgentMessageType::SystemEvent) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
            if (ImGui::Button("Copy")) ImGui::SetClipboardText(msg.content.c_str());
            ImGui::PopStyleColor();
        }

        ImGui::Spacing();
        ImGui::PopID();
    }

    if (scroll_to_bottom || ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
        scroll_to_bottom = false;
    }

    ImGui::EndChild();
}

void AgentChatPanel::queuePrompt(const std::string& sender,
                                 const std::string& target,
                                 const std::string& content) {
    if (content.empty()) return;
    pushMessage(AgentMessageType::UserPrompt, sender, target, content);
    std::lock_guard<std::mutex> lock(msg_mutex);
    user_prompts_queue.push_back({target, content});
}

void AgentChatPanel::submitPrompt() {
    std::string content(input_buffer);
    if (content.empty()) return;

    std::string target = "all"; // Default broadcast
    std::string text_to_send = content;

    // Check for @agent_id tag
    if (content.rfind("@", 0) == 0) { // starts with @
        size_t space_pos = content.find(' ');
        if (space_pos != std::string::npos) {
            target = content.substr(1, space_pos - 1);
            text_to_send = content.substr(space_pos + 1);
        }
    } else {
        // Alternatively, use active_agent_id if it's explicitly set? No, let's keep broadcast default unless @ is used.
    }

    // ★ The queued text is the message WITHOUT the @tag: the tag is routing, and
    // leaving it in makes the receiving model read its own address as part of
    // the task. The panel still shows what the user typed.
    pushMessage(AgentMessageType::UserPrompt, "User", target, content);
    {
        std::lock_guard<std::mutex> lock(msg_mutex);
        user_prompts_queue.push_back({target, text_to_send});
    }
    memset(input_buffer, 0, sizeof(input_buffer));
    reclaim_focus_next_frame = true;
}

void AgentChatPanel::drawInputArea() {
    // ★ The input stays available in every state. Hiding it while the agent was
    // mid-turn is what pushed users at the Start button and produced a second
    // agent process; a queued prompt is picked up on the next poll.
    ImGuiInputTextFlags input_flags =
        ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CtrlEnterForNewLine;

    ImGui::SetNextItemWidth(-60.0f);
    if (reclaim_focus_next_frame) {
        ImGui::SetKeyboardFocusHere();
        reclaim_focus_next_frame = false;
    }
    if (ImGui::InputText("##AgentInput", input_buffer, sizeof(input_buffer), input_flags))
        submitPrompt();

    ImGui::SameLine();
    if (ImGui::Button("Send")) submitPrompt();

    size_t queued = 0;
    {
        std::lock_guard<std::mutex> lock(msg_mutex);
        queued = user_prompts_queue.size();
    }
    if (queued > 0)
        ImGui::TextDisabled("%zu prompt(s) waiting for the agent to collect", queued);
}

bool AgentChatPanel::popUserPrompt(const std::string& polling_agent_id, QueuedPrompt& out_prompt) {
    std::lock_guard<std::mutex> lock(msg_mutex);
    for (auto it = user_prompts_queue.begin(); it != user_prompts_queue.end(); ++it) {
        // If target is empty/all, or matches the polling agent specifically, deliver it.
        if (it->target.empty() || it->target == "all" || it->target == polling_agent_id) {
            out_prompt = *it;
            user_prompts_queue.erase(it);
            return true;
        }
    }
    return false;
}

} // namespace rtui

void rtui::AgentChatPanel::drawAgentSettings() {
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Agent Configurations", &show_settings_window)) {
        ImGui::End();
        return;
    }

    auto& manager = AgentLifecycleManager::instance();
    auto configs = manager.getConfigs();

    static char input_id[64] = "";
    static char input_name[128] = "";
    static int provider_idx = 0;
    static char input_model[128] = "";
    static char input_apikey[256] = "";
    static char input_baseurl[256] = "";

    const char* providers[] = { "local", "gemini", "openai", "anthropic" };

    if (ImGui::BeginTabBar("ConfigTabs")) {
        if (ImGui::BeginTabItem("Existing Profiles")) {
            if (ImGui::BeginListBox("##ProfilesList", ImVec2(-FLT_MIN, 150))) {
                for (const auto& cfg : configs) {
                    ImGui::PushID(cfg.id.c_str());
                    if (ImGui::Selectable(cfg.name.c_str())) {
                        strncpy_s(input_id, cfg.id.c_str(), sizeof(input_id));
                        strncpy_s(input_name, cfg.name.c_str(), sizeof(input_name));
                        strncpy_s(input_model, cfg.model.c_str(), sizeof(input_model));
                        strncpy_s(input_apikey, cfg.api_key.c_str(), sizeof(input_apikey));
                        strncpy_s(input_baseurl, cfg.base_url.c_str(), sizeof(input_baseurl));
                        for (int i = 0; i < 4; ++i) {
                            if (cfg.provider == providers[i]) provider_idx = i;
                        }
                    }
                    ImGui::PopID();
                }
                ImGui::EndListBox();
            }

            if (ImGui::Button("Delete Selected")) {
                if (strlen(input_id) > 0) {
                    manager.removeConfig(input_id);
                    memset(input_id, 0, sizeof(input_id));
                    memset(input_name, 0, sizeof(input_name));
                    memset(input_model, 0, sizeof(input_model));
                    memset(input_apikey, 0, sizeof(input_apikey));
                    memset(input_baseurl, 0, sizeof(input_baseurl));
                }
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::Separator();
    ImGui::Text("Edit / Create Profile");

    ImGui::InputText("ID (no spaces)", input_id, sizeof(input_id));
    ImGui::InputText("Display Name", input_name, sizeof(input_name));
    ImGui::Combo("Provider", &provider_idx, providers, 4);
    ImGui::InputText("Model Name", input_model, sizeof(input_model));
    ImGui::InputText("API Key", input_apikey, sizeof(input_apikey), ImGuiInputTextFlags_Password);

    if (provider_idx == 0) { // local
        ImGui::InputText("Base URL", input_baseurl, sizeof(input_baseurl));
    }

    if (ImGui::Button("Save Configuration")) {
        if (strlen(input_id) > 0 && strlen(input_name) > 0) {
            AgentConfig cfg;
            cfg.id = input_id;
            cfg.name = input_name;
            cfg.provider = providers[provider_idx];
            cfg.model = input_model;
            cfg.api_key = input_apikey;
            cfg.base_url = input_baseurl;
            manager.addConfig(cfg);
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Clear Form")) {
        memset(input_id, 0, sizeof(input_id));
        memset(input_name, 0, sizeof(input_name));
        memset(input_model, 0, sizeof(input_model));
        memset(input_apikey, 0, sizeof(input_apikey));
        memset(input_baseurl, 0, sizeof(input_baseurl));
    }

    ImGui::End();
}
