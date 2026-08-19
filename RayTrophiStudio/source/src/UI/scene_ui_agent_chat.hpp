/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          UI/scene_ui_agent_chat.hpp
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 *
 * ImGui chat panel for the external agent runtime (RayTrophiAgent/).
 *
 * The panel owns three things: the message list, the queue of user prompts the
 * agent picks up over agent.chat_poll, and the lifetime of the agent process.
 *
 * ★ The panel does not infer the agent's state from silence. "Not polling" and
 * "not running" are different states and are shown differently: an agent busy
 * in a long model turn used to read as offline, which put a Start button in
 * front of the user and spawned a second agent against a single-client pipe.
 * =========================================================================
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rtui {

// Types of chat messages
enum class AgentMessageType {
    UserPrompt,      // Message sent by the user
    AgentReply,      // Reply from an agent
    SystemEvent,     // System event, or an error the agent reported
    AgentActivity,   // One IPC call the core actually served
    AgentThought     // The agent's own reasoning, when it chooses to share it
};

struct AgentMessage {
    uint64_t id = 0;         // stable, monotonic; used for widget identity
    AgentMessageType type = AgentMessageType::SystemEvent;
    std::string sender;      // e.g. "RayTrophi Agent", "System"
    std::string target;      // e.g. "Agent"
    std::string content;
    std::string timestamp;   // HH:MM:SS
};

class AgentChatPanel {
public:
    AgentChatPanel();
    ~AgentChatPanel();

    // Main ImGui render function, called every frame while the panel is open.
    void draw(bool* p_open);

    // Thread-safe: called from the IPC worker thread as well as the UI thread.
    void pushMessage(AgentMessageType type, const std::string& sender,
                     const std::string& target, const std::string& content);

    // Called by agent.chat_poll. Doubles as the agent's heartbeat.
    void markPoll();

    // Prompts waiting for the agent to collect.
    struct QueuedPrompt { std::string target; std::string content; };
    bool popUserPrompt(QueuedPrompt& out_prompt);

    // Stop the agent process if this panel started one. Called from the
    // destructor too, so closing Studio does not leave an orphan behind.
    void stopAgentProcess();

private:
    // Oldest messages are dropped past this; a long agent session would
    // otherwise grow the list without bound and take the frame time with it.
    static constexpr size_t kMaxMessages = 600;

    std::vector<AgentMessage> messages;
    std::vector<QueuedPrompt> user_prompts_queue;
    std::mutex msg_mutex;
    uint64_t next_message_id = 1;
    size_t dropped_messages = 0;

    char input_buffer[2048];
    bool show_activity_log;

    std::chrono::steady_clock::time_point last_poll_time;
    bool ever_polled = false;

    // Highest audit sequence already mirrored into the activity feed.
    uint64_t activity_cursor = 0;
    bool activity_cursor_primed = false;

    bool scroll_to_bottom;
    bool reclaim_focus_next_frame;

#ifdef _WIN32
    void* agent_process_handle = nullptr;  // HANDLE
    unsigned long agent_process_id = 0;
#endif
    void startAgentProcess();
    bool isAgentProcessAlive();

    // Mirror the core's own audit log into the panel, so "Core Activity" shows
    // what the engine actually served rather than what the agent chose to
    // report about itself.
    void pumpCoreActivity();

    void submitPrompt();
    void drawMessageList();
    void drawInputArea();
    void drawStatusLine();
    std::string getCurrentTimeStr();
};

} // namespace rtui
