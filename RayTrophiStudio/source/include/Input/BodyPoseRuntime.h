#pragma once

#include "json.hpp"
#include "Game/GameEvents.h"

#include <array>
#include <chrono>
#include <map>
#include <string>

namespace Input {

inline constexpr std::array<const char *, 15> kBodyJoints = {
    "head",    "neck",       "pelvis",  "l_shoulder", "l_elbow", "l_wrist", "l_hip",  "l_knee",
    "l_ankle", "r_shoulder", "r_elbow", "r_wrist",    "r_hip",   "r_knee",  "r_ankle"};

struct BodyPerson {
    int id = 0;
    double confidence = 0.0;
    std::map<std::string, std::array<double, 4>> joints;
};

// Main-thread transient input. Both scripting and IPC marshal onto that thread.
// Sender time drives velocity; receiver steady time drives loss, never sender time.
class BodyPoseRuntime {
  public:
    using Clock = std::chrono::steady_clock;
    bool set(const nlohmann::json &packet, std::string &error,
             Clock::time_point received = Clock::now());
    void expire(Clock::time_point now = Clock::now());
    void clear();
    nlohmann::json snapshot(Clock::time_point now = Clock::now());
    const std::map<int, BodyPerson> persons() const;
    double seconds() const {
        return seconds_;
    }
    void tick(double dt);
    std::uint64_t packetCount() const {
        return packets_;
    }
    // Main-thread borrowed view; expire first when sampling outside the live tick.
    const BodyPerson* findPerson(int id) const;

  private:
    struct Track {
        BodyPerson person;
        Clock::time_point received;
        double sender_t = 0.0;
        bool arms_up = false;
        bool reach = false;
        double last_stomp = -1.0;
    };
    std::map<int, Track> tracks_;
    double last_sender_t_ = -1.0;
    double seconds_ = 0.0;
    std::uint64_t packets_ = 0;
    std::uint64_t gestures_ = 0;
    std::uint64_t expired_ = 0;
    void gestures(Track &current, const Track *previous);
};

BodyPoseRuntime &bodyPoses();

} // namespace Input
