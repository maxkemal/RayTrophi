#pragma once
#include "Game/GameEvents.h"
#include "json.hpp"
#include <map>
#include <string>
#include <vector>

namespace Game {
struct RuleAction {
    nlohmann::json action;
    Event event;
};

class Rules {
  public:
    bool configure(const nlohmann::json &rules, std::string &error);
    void evaluate(const Event &event, const std::string &state, double time,
                  std::vector<RuleAction> &output);
    nlohmann::json snapshot() const;
    void resetHistory();

  private:
    struct Rule {
        nlohmann::json config;
        std::map<std::pair<std::string, int>, double> last_fire;
        std::map<std::pair<std::string, int>, double> people;
        std::uint64_t fires = 0;
        std::uint64_t dropped = 0;
    };
    std::vector<Rule> rules_;
};
} // namespace Game
