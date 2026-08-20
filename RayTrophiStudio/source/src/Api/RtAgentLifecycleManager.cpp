/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtAgentLifecycleManager.cpp
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 */

#include "RtAgentLifecycleManager.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <filesystem>
#include <iostream>
#include <fstream>
#include <json.hpp>

AgentLifecycleManager& AgentLifecycleManager::instance() {
    static AgentLifecycleManager inst;
    return inst;
}

AgentLifecycleManager::AgentLifecycleManager() {
    loadConfigs();
    if (configs_.empty()) {
        // Add some default configurations if nothing was loaded
        addConfig({"local_qwen", "Local Qwen3 (Ollama)", "local", "qwen3:8b", "", "http://localhost:11434/v1"});
        addConfig({"cloud_gemini", "Gemini 2.5 Flash", "gemini", "gemini-2.5-flash", "", ""});
        addConfig({"cloud_openai", "OpenAI GPT-4o", "openai", "gpt-4o-mini", "", ""});
        addConfig({"cloud_claude", "Claude 3.5 Sonnet", "anthropic", "claude-3-5-sonnet-20240620", "", ""});
    }
}

AgentLifecycleManager::~AgentLifecycleManager() {
    stopAll();
}

void AgentLifecycleManager::addConfig(const AgentConfig& config) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    configs_[config.id] = config;
    saveConfigs();
}

void AgentLifecycleManager::removeConfig(const std::string& id) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    configs_.erase(id);
    stopAgent(id);
    saveConfigs();
}

std::vector<AgentConfig> AgentLifecycleManager::getConfigs() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<AgentConfig> res;
    for (const auto& pair : configs_) {
        res.push_back(pair.second);
    }
    return res;
}

bool AgentLifecycleManager::getConfig(const std::string& id, AgentConfig& out_config) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = configs_.find(id);
    if (it != configs_.end()) {
        out_config = it->second;
        return true;
    }
    return false;
}

bool AgentLifecycleManager::startAgent(const std::string& id) {
#ifdef _WIN32
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    auto it = configs_.find(id);
    if (it == configs_.end()) return false;
    AgentConfig& config = it->second;

    if (active_handles_.find(id) != active_handles_.end()) {
        // Already running, check if it's actually alive
        DWORD state = WaitForSingleObject((HANDLE)active_handles_[id], 0);
        if (state == WAIT_TIMEOUT) {
            return true; // Still running
        } else {
            // Process crashed or stopped
            CloseHandle((HANDLE)active_handles_[id]);
            active_handles_.erase(id);
            active_pids_.erase(id);
        }
    }

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    ZeroMemory(&pi, sizeof(pi));

    std::string work_dir = "RayTrophiAgent";
    if (!std::filesystem::exists(work_dir)) {
        if (std::filesystem::exists("../RayTrophiAgent")) work_dir = "../RayTrophiAgent";
        else if (std::filesystem::exists("../../RayTrophiAgent")) work_dir = "../../RayTrophiAgent";
        else if (std::filesystem::exists("../../../RayTrophiAgent")) work_dir = "../../../RayTrophiAgent";
        else return false;
    }
    const std::string abs_work_dir = std::filesystem::absolute(work_dir).string();

    std::string cmd = "python -E main.py";
    std::vector<char> cmdBuffer(cmd.begin(), cmd.end());
    cmdBuffer.push_back('\0');

    // Set Environment Variables temporarily for this spawn
    SetEnvironmentVariableA("LLM_PROVIDER", config.provider.c_str());
    if (config.provider == "openai") {
        SetEnvironmentVariableA("OPENAI_API_KEY", config.api_key.c_str());
        SetEnvironmentVariableA("OPENAI_MODEL", config.model.c_str());
    } else if (config.provider == "gemini") {
        SetEnvironmentVariableA("GEMINI_API_KEY", config.api_key.c_str());
        SetEnvironmentVariableA("GEMINI_MODEL", config.model.c_str());
    } else if (config.provider == "anthropic") {
        SetEnvironmentVariableA("ANTHROPIC_API_KEY", config.api_key.c_str());
        SetEnvironmentVariableA("ANTHROPIC_MODEL", config.model.c_str());
    } else if (config.provider == "local") {
        SetEnvironmentVariableA("LOCAL_LLM_URL", config.base_url.c_str());
        SetEnvironmentVariableA("LOCAL_LLM_MODEL", config.model.c_str());
    }

    bool success = CreateProcessA(NULL, cmdBuffer.data(), NULL, NULL, FALSE, CREATE_NO_WINDOW,
                                  NULL, abs_work_dir.c_str(), &si, &pi);

    // Unset Environment Variables so they don't leak to other processes
    SetEnvironmentVariableA("LLM_PROVIDER", NULL);
    SetEnvironmentVariableA("OPENAI_API_KEY", NULL);
    SetEnvironmentVariableA("OPENAI_MODEL", NULL);
    SetEnvironmentVariableA("GEMINI_API_KEY", NULL);
    SetEnvironmentVariableA("GEMINI_MODEL", NULL);
    SetEnvironmentVariableA("ANTHROPIC_API_KEY", NULL);
    SetEnvironmentVariableA("ANTHROPIC_MODEL", NULL);
    SetEnvironmentVariableA("LOCAL_LLM_URL", NULL);
    SetEnvironmentVariableA("LOCAL_LLM_MODEL", NULL);

    if (success) {
        active_handles_[id] = pi.hProcess;
        active_pids_[id] = pi.dwProcessId;
        CloseHandle(pi.hThread);
        return true;
    }
#endif
    return false;
}

void AgentLifecycleManager::stopAgent(const std::string& id) {
#ifdef _WIN32
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = active_handles_.find(id);
    if (it != active_handles_.end()) {
        TerminateProcess((HANDLE)it->second, 0);
        CloseHandle((HANDLE)it->second);
        active_handles_.erase(it);
        active_pids_.erase(id);
    }
#endif
}

bool AgentLifecycleManager::isAgentRunning(const std::string& id) {
#ifdef _WIN32
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = active_handles_.find(id);
    if (it == active_handles_.end()) return false;

    DWORD state = WaitForSingleObject((HANDLE)it->second, 0);
    if (state == WAIT_TIMEOUT) {
        return true;
    }
    // Process terminated
    CloseHandle((HANDLE)it->second);
    active_handles_.erase(it);
    active_pids_.erase(id);
    return false;
#else
    return false;
#endif
}

void AgentLifecycleManager::stopAll() {
#ifdef _WIN32
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (auto& pair : active_handles_) {
        TerminateProcess((HANDLE)pair.second, 0);
        CloseHandle((HANDLE)pair.second);
    }
    active_handles_.clear();
    active_pids_.clear();
#endif
}

void AgentLifecycleManager::saveConfigs() {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& pair : configs_) {
        nlohmann::json j_config;
        j_config["id"] = pair.second.id;
        j_config["name"] = pair.second.name;
        j_config["provider"] = pair.second.provider;
        j_config["model"] = pair.second.model;
        j_config["api_key"] = pair.second.api_key;
        j_config["base_url"] = pair.second.base_url;
        j.push_back(j_config);
    }
    std::ofstream o("agents_config.json");
    if (o.is_open()) {
        o << std::setw(4) << j << std::endl;
    }
}

void AgentLifecycleManager::loadConfigs() {
    std::ifstream i("agents_config.json");
    if (i.is_open()) {
        try {
            nlohmann::json j;
            i >> j;
            for (const auto& j_config : j) {
                AgentConfig cfg;
                cfg.id = j_config.value("id", "");
                cfg.name = j_config.value("name", "");
                cfg.provider = j_config.value("provider", "");
                cfg.model = j_config.value("model", "");
                cfg.api_key = j_config.value("api_key", "");
                cfg.base_url = j_config.value("base_url", "");
                if (!cfg.id.empty()) {
                    configs_[cfg.id] = cfg;
                }
            }
        } catch (const nlohmann::json::exception& e) {
            std::cerr << "Failed to parse agents_config.json: " << e.what() << std::endl;
        }
    }
}
