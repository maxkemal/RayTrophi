#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rtipc_security {

enum Capability : uint32_t {
    Read       = 1u << 0u,
    SceneWrite = 1u << 1u,
    Render     = 1u << 2u,
    FilesRead  = 1u << 3u,
    FilesWrite = 1u << 4u,
    Scripts    = 1u << 5u,
    Addons     = 1u << 6u,
    Admin      = 1u << 7u
};

struct TokenInfo {
    std::string id;
    std::string display_name;
    uint32_t capabilities = 0;
    int64_t created_at = 0;
    int64_t expires_at = 0;
    int64_t last_used_at = 0;
    bool revoked = false;
    std::vector<std::string> allowed_cidrs;
};

struct Authentication {
    bool ok = false;
    std::string token_id;
    uint32_t capabilities = 0;
    std::string error;
};

// Loads the protected token database. The environment bootstrap token is
// converted to a SHA-256 digest record; its raw value is never persisted.
bool initialize(const std::string& bootstrap_token, uint32_t bootstrap_capabilities,
                const std::vector<std::string>& bootstrap_allowed_cidrs,
                const std::string& optional_store_path, std::string& error);
void shutdown() noexcept;

Authentication authenticate(const std::string& raw_token,
                            const std::string& remote_address = {});
bool authorize(uint32_t capabilities, const std::string& method, std::string& error);
bool authorizePath(const std::string& method, const std::string& path,
                   std::string& error);

bool createToken(const std::string& display_name, uint32_t capabilities,
                 const std::vector<std::string>& allowed_cidrs,
                 int64_t expires_at, TokenInfo& out_info,
                 std::string& out_raw_token, std::string& error);
bool revokeToken(const std::string& token_id, std::string& error);
bool updateToken(const std::string& token_id, uint32_t capabilities,
                 const std::vector<std::string>& allowed_cidrs,
                 int64_t expires_at, std::string& error);
bool rotateToken(const std::string& token_id, TokenInfo& out_info,
                 std::string& out_raw_token, std::string& error);
std::vector<TokenInfo> listTokens();

} // namespace rtipc_security
