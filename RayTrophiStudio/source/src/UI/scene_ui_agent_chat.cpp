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
#ifdef _WIN32
    if (isAgentProcessAlive()) return;

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    ZeroMemory(&pi, sizeof(pi));

    std::string work_dir = "RayTrophiAgent";
    if (!std::filesystem::exists(work_dir)) {
        if (std::filesystem::exists("../RayTrophiAgent")) {
            work_dir = "../RayTrophiAgent";
        } else if (std::filesystem::exists("../../RayTrophiAgent")) {
            work_dir = "../../RayTrophiAgent";
        } else if (std::filesystem::exists("../../../RayTrophiAgent")) {
            work_dir = "../../../RayTrophiAgent";
        } else {
            pushMessage(AgentMessageType::SystemEvent, "System", kTargetName,
                        "Cannot find the RayTrophiAgent directory next to the "
                        "executable or up to three levels above it.");
            return;
        }
    }
    const std::string abs_work_dir = std::filesystem::absolute(work_dir).string();

    // Run python directly. Not through cmd.exe: TerminateProcess would then
    // kill the shell and leave python running as an orphan.
    // -E makes python ignore PYTHON* environment variables, which is what made
    // this launch differ from a working terminal launch.
    std::string cmd = "python -E main.py";
    std::vector<char> cmdBuffer(cmd.begin(), cmd.end());
    cmdBuffer.push_back('\0');

    if (CreateProcessA(NULL, cmdBuffer.data(), NULL, NULL, FALSE, CREATE_NO_WINDOW,
                       NULL, abs_work_dir.c_str(), &si, &pi)) {
        agent_process_handle = pi.hProcess;
        agent_process_id = pi.dwProcessId;
        CloseHandle(pi.hThread);
        pushMessage(AgentMessageType::SystemEvent, "System", kTargetName,
                    "Agent process started (pid " + std::to_string(pi.dwProcessId) +
                    "). It reports connected once it polls.");
    } else {
        pushMessage(AgentMessageType::SystemEvent, "System", kTargetName,
                    "Failed to start the agent process (error " +
                    std::to_string(GetLastError()) +
                    "). Is python on PATH?");
    }
#else
    pushMessage(AgentMessageType::SystemEvent, "System", kTargetName,
                "Starting the agent process is only implemented on Windows.");
#endif
}

bool AgentChatPanel::isAgentProcessAlive() {
#ifdef _WIN32
    if (!agent_process_handle) return false;
    // ★ Holding a handle is not the same as the process running. A crashed
    // runtime kept the Stop button showing and the Start button hidden.
    const DWORD state = WaitForSingleObject((HANDLE)agent_process_handle, 0);
    if (state == WAIT_TIMEOUT) return true;
    CloseHandle((HANDLE)agent_process_handle);
    agent_process_handle = nullptr;
    agent_process_id = 0;
    return false;
#else
    return false;
#endif
}

void AgentChatPanel::stopAgentProcess() {
#ifdef _WIN32
    if (!agent_process_handle) return;
    TerminateProcess((HANDLE)agent_process_handle, 0);
    CloseHandle((HANDLE)agent_process_handle);
    agent_process_handle = nullptr;
    agent_process_id = 0;
    pushMessage(AgentMessageType::SystemEvent, "System", kTargetName,
                "Agent process stopped.");
#endif
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
#ifdef _WIN32
    if (isAgentProcessAlive()) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        if (ImGui::Button("Stop Agent")) stopAgentProcess();
        ImGui::PopStyleColor();
    } else {
        if (ImGui::Button("Start AI Agent")) startAgentProcess();
    }
#endif

    drawStatusLine();
    ImGui::Separator();
    drawMessageList();
    ImGui::Separator();
    drawInputArea();

    ImGui::End();
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

void AgentChatPanel::submitPrompt() {
    std::string content(input_buffer);
    if (content.empty()) return;
    pushMessage(AgentMessageType::UserPrompt, "User", kTargetName, content);
    {
        std::lock_guard<std::mutex> lock(msg_mutex);
        user_prompts_queue.push_back({kTargetName, content});
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

bool AgentChatPanel::popUserPrompt(QueuedPrompt& out_prompt) {
    std::lock_guard<std::mutex> lock(msg_mutex);
    if (user_prompts_queue.empty()) return false;
    out_prompt = user_prompts_queue.front();
    user_prompts_queue.erase(user_prompts_queue.begin());
    return true;
}

} // namespace rtui
