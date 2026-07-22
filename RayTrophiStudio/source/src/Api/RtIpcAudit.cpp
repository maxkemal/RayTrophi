#include "RtIpcAudit.h"

#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>

using json = nlohmann::json;

namespace rtipc_audit {
namespace {

constexpr size_t kRingCapacity = 2048;
constexpr uintmax_t kMaxJsonlBytes = 8u * 1024u * 1024u;
constexpr int kRequestsPerSecond = 240;
std::mutex g_mutex;
std::deque<Event> g_events;
std::filesystem::path g_jsonl_path;
uint64_t g_sequence = 0;

struct AuthState { int failures = 0; int64_t blocked_until_ms = 0; };
struct RateState { int64_t window_ms = 0; int requests = 0; int expensive = 0; };
std::unordered_map<std::string, AuthState> g_auth;
std::unordered_map<std::string, RateState> g_rates;

int64_t unixSeconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
int64_t steadyMilliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

json eventJson(const Event& event) {
    return json{{"sequence", event.sequence}, {"timestamp", event.timestamp},
                {"connection_id", event.connection_id}, {"token_id", event.token_id},
                {"peer_address", event.peer_address}, {"method", event.method},
                {"allowed", event.allowed}, {"outcome", event.outcome},
                {"duration_us", event.duration_us},
                {"bytes_received", event.bytes_received}, {"bytes_sent", event.bytes_sent}};
}

void rotateLocked() {
    if (g_jsonl_path.empty()) return;
    std::error_code ec;
    if (!std::filesystem::exists(g_jsonl_path, ec) ||
        std::filesystem::file_size(g_jsonl_path, ec) < kMaxJsonlBytes) return;
    auto rotated = g_jsonl_path; rotated += L".1";
    std::filesystem::remove(rotated, ec); ec.clear();
    std::filesystem::rename(g_jsonl_path, rotated, ec);
}

bool expensiveMethod(const std::string& method) {
    return method == "render.start" || method == "render.start_sequence" ||
           method.find("evaluate") != std::string::npos ||
           method.find("bake") != std::string::npos ||
           method.find("simulation") != std::string::npos;
}

} // namespace

bool initialize(const std::string& optional_jsonl_path, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_jsonl_path = optional_jsonl_path.empty() ? std::filesystem::path{}
                                               : std::filesystem::path(optional_jsonl_path);
    if (!g_jsonl_path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(g_jsonl_path.parent_path(), ec);
        std::ofstream probe(g_jsonl_path, std::ios::app);
        if (!probe) { error = "cannot open IPC audit JSONL"; g_jsonl_path.clear(); return false; }
    }
    return true;
}

void shutdown() noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_events.clear(); g_auth.clear(); g_rates.clear(); g_jsonl_path.clear();
}

void record(Event event) {
    std::lock_guard<std::mutex> lock(g_mutex);
    event.sequence = ++g_sequence; event.timestamp = unixSeconds();
    g_events.push_back(event);
    while (g_events.size() > kRingCapacity) g_events.pop_front();
    if (!g_jsonl_path.empty()) {
        rotateLocked();
        std::ofstream output(g_jsonl_path, std::ios::app | std::ios::binary);
        if (output) output << eventJson(event).dump() << '\n';
    }
}

std::vector<Event> recent(size_t maximum) {
    std::lock_guard<std::mutex> lock(g_mutex);
    maximum = (std::min)(maximum, g_events.size());
    return std::vector<Event>(g_events.end() - static_cast<std::ptrdiff_t>(maximum), g_events.end());
}

void clear() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_events.clear();
}

bool exportJsonl(const std::string& path, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::ofstream output(std::filesystem::path(path), std::ios::binary | std::ios::trunc);
    if (!output) { error = "cannot create audit export"; return false; }
    for (const auto& event : g_events) output << eventJson(event).dump() << '\n';
    if (!output) { error = "cannot write audit export"; return false; }
    return true;
}

bool allowAuthenticationAttempt(const std::string& peer, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_auth.find(peer);
    if (found != g_auth.end() && found->second.blocked_until_ms > steadyMilliseconds()) {
        error = "authentication temporarily throttled"; return false;
    }
    return true;
}

void recordAuthentication(const std::string& peer, bool success) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& state = g_auth[peer];
    if (success) { state = {}; return; }
    state.failures = (std::min)(state.failures + 1, 10);
    if (state.failures >= 3) {
        const int delay_seconds = (std::min)(1 << (state.failures - 3), 60);
        state.blocked_until_ms = steadyMilliseconds() + delay_seconds * 1000;
    }
}

bool allowRequest(const std::string& token_id, const std::string& method,
                  std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const int64_t now = steadyMilliseconds();
    auto& state = g_rates[token_id];
    if (now - state.window_ms >= 1000) {
        state.window_ms = now; state.requests = 0; state.expensive = 0;
    }
    if (++state.requests > kRequestsPerSecond) {
        error = "token request rate exceeded"; return false;
    }
    if (expensiveMethod(method) && ++state.expensive > 4) {
        error = "expensive operation budget exceeded"; return false;
    }
    return true;
}

} // namespace rtipc_audit
