#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace rtipc_audit {

struct Event {
    uint64_t sequence = 0;
    int64_t timestamp = 0;
    std::string connection_id;
    std::string token_id;
    std::string peer_address;
    std::string method;
    std::string outcome;
    uint64_t duration_us = 0;
    uint64_t bytes_received = 0;
    uint64_t bytes_sent = 0;
    bool allowed = false;
};

bool initialize(const std::string& optional_jsonl_path, std::string& error);
void shutdown() noexcept;
void record(Event event);
std::vector<Event> recent(size_t maximum = 256);
void clear();
bool exportJsonl(const std::string& path, std::string& error);

bool allowAuthenticationAttempt(const std::string& peer_address,
                                std::string& error);
void recordAuthentication(const std::string& peer_address, bool success);
bool allowRequest(const std::string& token_id, const std::string& method,
                  std::string& error);

} // namespace rtipc_audit
