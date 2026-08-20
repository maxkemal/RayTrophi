/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtAgentLifecycleManager.h
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <mutex>

struct AgentConfig {
    std::string id;
    std::string name;
    std::string provider; // "openai", "gemini", "anthropic", "local"
    std::string model;
    std::string api_key;
    std::string base_url;
};

class AgentLifecycleManager {
public:
    static AgentLifecycleManager& instance();

    bool startAgent(const std::string& id);
    void stopAgent(const std::string& id);
    bool isAgentRunning(const std::string& id);
    void stopAll();

    // Config management
    void addConfig(const AgentConfig& config);
    void removeConfig(const std::string& id);
    std::vector<AgentConfig> getConfigs() const;
    bool getConfig(const std::string& id, AgentConfig& out_config) const;

    void saveConfigs();
    void loadConfigs();

private:
    AgentLifecycleManager();
    ~AgentLifecycleManager();

    std::map<std::string, AgentConfig> configs_;
    std::map<std::string, void*> active_handles_; // HANDLE
    std::map<std::string, unsigned long> active_pids_; // DWORD
    mutable std::recursive_mutex mutex_;
};
