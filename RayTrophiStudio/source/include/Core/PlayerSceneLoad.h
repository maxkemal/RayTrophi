#pragma once

#include "Core/RuntimeProjectHost.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string_view>
#include <utility>

namespace AppRuntime {

inline std::string playerScenePathKey(const std::filesystem::path& path) {
    std::error_code error;
    const auto resolved = std::filesystem::weakly_canonical(path, error);
    auto key = projectPathToUtf8(error ? path.lexically_normal() : resolved);
#ifdef _WIN32
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
    return key;
}

// Main-thread load telemetry. Progress represents loader checkpoints.
class PlayerSceneLoad {
public:
    bool busy() const { return state_ == "loading"; }
    bool presenting() const { return presenting_; }
    std::string_view stage() const { return stage_; }
    void setPresenter(std::function<void()> presenter) { presenter_ = std::move(presenter); }

    void begin(const std::string& path) {
        ++attempt_;
        path_ = path;
        key_ = playerScenePathKey(projectPathFromUtf8(path));
        progress_ = 0;
        state_ = "loading";
        stage_.clear();
        detail_.clear();
        error_.clear();
        stages_ = nlohmann::json::array();
        started_ = checkpoint_ = Clock::now();
    }

    void checkpoint(int progress, std::string stage, std::string detail = {}) {
        recordStage();
        progress_ = std::max(progress_, std::clamp(progress, 0, 99));
        stage_ = std::move(stage);
        detail_ = std::move(detail);
        if (presenter_) {
            presenting_ = true;
            try {
                presenter_();
            } catch (...) {
                presenting_ = false;
                throw;
            }
            presenting_ = false;
        }
        checkpoint_ = Clock::now();
    }

    void finish(std::string error = {}) {
        if (!busy()) {
            return;
        }
        recordStage();
        finished_ = Clock::now();
        error_ = std::move(error);
        state_ = error_.empty() ? "ready" : "failed";
        if (error_.empty()) {
            progress_ = 100;
        }
    }

    nlohmann::json status() const {
        const double elapsed = state_ == "idle" ? 0.0 :
            milliseconds(started_, busy() ? Clock::now() : finished_);
        return {{"attempt", attempt_}, {"state", state_}, {"path", path_}, {"key", key_},
                {"progress", progress_},
                {"stage", stage_}, {"detail", detail_}, {"elapsed_ms", elapsed},
                {"stages", stages_}, {"error", error_}};
    }

private:
    using Clock = std::chrono::steady_clock;

    static double milliseconds(Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    }

    void recordStage() {
        if (!stage_.empty()) {
            stages_.push_back({{"stage", stage_}, {"detail", detail_},
                               {"ms", milliseconds(checkpoint_, Clock::now())}});
        }
    }

    std::string path_;
    std::string key_;
    std::string state_ = "idle";
    std::string stage_;
    std::string detail_;
    std::string error_;
    nlohmann::json stages_ = nlohmann::json::array();
    int progress_ = 0;
    std::uint64_t attempt_ = 0;
    bool presenting_ = false;
    Clock::time_point started_;
    Clock::time_point checkpoint_;
    Clock::time_point finished_;
    std::function<void()> presenter_;
};

inline PlayerSceneLoad& playerSceneLoad() {
    static PlayerSceneLoad load;
    return load;
}

} // namespace AppRuntime
