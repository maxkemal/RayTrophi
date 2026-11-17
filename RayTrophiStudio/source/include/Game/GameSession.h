#pragma once
#include "Game/GameRules.h"
#include "Input/BodyPoseRuntime.h"
#include <set>

namespace Game {
class Session {
  public:
    bool command(const std::string &operation, const nlohmann::json &config, nlohmann::json &output,
                 std::string &error);
    void tick(double dt, const std::map<int, Input::BodyPerson> &persons, double coverage,
              std::vector<RuleAction> &actions);
    nlohmann::json snapshot() const;
    void actionResult(bool ok, const std::string &error);
    void clear();
    bool configured() const {
        return configured_;
    }

  private:
    nlohmann::json config_;
    Rules rules_;
    bool configured_ = false;
    std::string state_ = "attract";
    double time_ = 0.0;
    double state_time_ = 0.0;
    double coverage_ = 0.0;
    std::uint64_t sessions_ = 0;
    std::uint64_t action_ok_ = 0;
    std::uint64_t action_failed_ = 0;
    std::size_t person_count_ = 0;
    std::map<std::string, std::uint64_t> events_;
    std::map<std::string, std::set<int>> zones_;
    std::string last_error_;
    std::vector<Event> local_events_;
    void transition(const std::string &state);
};
Session &session();
} // namespace Game
