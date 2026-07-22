#include "RtIpcSecurity.h"

#include "json.hpp"
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <aclapi.h>
#include <sddl.h>
#endif

using json = nlohmann::json;

namespace rtipc_security {
namespace {

struct StoredToken : TokenInfo {
    std::string digest;
};

std::mutex g_mutex;
std::vector<StoredToken> g_tokens;
std::filesystem::path g_store_path;
bool g_initialized = false;

int64_t unixNow() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string hex(const unsigned char* bytes, size_t count) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(count * 2, '0');
    for (size_t i = 0; i < count; ++i) {
        result[i * 2] = digits[bytes[i] >> 4u];
        result[i * 2 + 1] = digits[bytes[i] & 0x0fu];
    }
    return result;
}

bool randomHex(size_t bytes, std::string& out) {
    std::vector<unsigned char> buffer(bytes);
    if (RAND_bytes(buffer.data(), static_cast<int>(buffer.size())) != 1) return false;
    out = hex(buffer.data(), buffer.size());
    return true;
}

bool sha256(const std::string& value, std::string& out) {
    unsigned char digest[EVP_MAX_MD_SIZE]{};
    unsigned int digest_size = 0;
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) return false;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(context, value.data(), value.size()) == 1 &&
                    EVP_DigestFinal_ex(context, digest, &digest_size) == 1;
    EVP_MD_CTX_free(context);
    if (ok) out = hex(digest, digest_size);
    return ok;
}

bool constantTimeEqual(const std::string& lhs, const std::string& rhs) {
    const size_t size = (std::max)(lhs.size(), rhs.size());
    unsigned char difference = static_cast<unsigned char>(lhs.size() ^ rhs.size());
    for (size_t i = 0; i < size; ++i) {
        const unsigned char a = i < lhs.size() ? static_cast<unsigned char>(lhs[i]) : 0;
        const unsigned char b = i < rhs.size() ? static_cast<unsigned char>(rhs[i]) : 0;
        difference |= static_cast<unsigned char>(a ^ b);
    }
    return difference == 0;
}

std::filesystem::path defaultStorePath() {
#ifdef _WIN32
    std::wstring executable(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, executable.data(),
                                            static_cast<DWORD>(executable.size()));
    if (length > 0 && length < executable.size()) {
        executable.resize(length);
        return std::filesystem::path(executable).parent_path() / L"ipc_tokens.json";
    }
#endif
    return std::filesystem::current_path() / "ipc_tokens.json";
}

#ifdef _WIN32
bool protectFileForCurrentUser(const std::filesystem::path& path, std::string& error) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
        error = "cannot query current Windows token"; return false;
    }
    DWORD bytes = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &bytes);
    std::vector<unsigned char> storage(bytes);
    if (bytes == 0 || !GetTokenInformation(token, TokenUser, storage.data(), bytes, &bytes)) {
        CloseHandle(token); error = "cannot read current Windows SID"; return false;
    }
    CloseHandle(token);
    LPWSTR sid = nullptr;
    if (!ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER*>(storage.data())->User.Sid, &sid)) {
        error = "cannot format current Windows SID"; return false;
    }
    const std::wstring sddl = L"D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;" +
                              std::wstring(sid) + L")";
    LocalFree(sid);
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.c_str(), SDDL_REVISION_1, &descriptor, nullptr)) {
        error = "cannot build token-store ACL"; return false;
    }
    BOOL present = FALSE, defaulted = FALSE;
    PACL dacl = nullptr;
    const BOOL got_dacl = GetSecurityDescriptorDacl(descriptor, &present, &dacl, &defaulted);
    const DWORD status = got_dacl && present
        ? SetNamedSecurityInfoW(const_cast<LPWSTR>(path.c_str()), SE_FILE_OBJECT,
                                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                                nullptr, nullptr, dacl, nullptr)
        : ERROR_INVALID_SECURITY_DESCR;
    LocalFree(descriptor);
    if (status != ERROR_SUCCESS) {
        error = "cannot protect token store ACL (error " + std::to_string(status) + ")";
        return false;
    }
    return true;
}
#endif

bool validCidr(const std::string& cidr);

json tokenJson(const StoredToken& token) {
    return json{{"id", token.id}, {"display_name", token.display_name},
                {"digest_sha256", token.digest}, {"capabilities", token.capabilities},
                {"created_at", token.created_at}, {"expires_at", token.expires_at},
                {"last_used_at", token.last_used_at}, {"revoked", token.revoked},
                {"allowed_cidrs", token.allowed_cidrs}};
}

bool saveLocked(std::string& error) {
    json root{{"version", 1}, {"tokens", json::array()}};
    for (const auto& token : g_tokens) root["tokens"].push_back(tokenJson(token));
    std::error_code ec;
    std::filesystem::create_directories(g_store_path.parent_path(), ec);
    std::filesystem::path temporary = g_store_path;
    temporary += L".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { error = "cannot write IPC token store temporary file"; return false; }
        output << root.dump(2);
        output.flush();
        if (!output) { error = "cannot flush IPC token store"; return false; }
    }
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), g_store_path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        DeleteFileW(temporary.c_str());
        error = "cannot atomically replace IPC token store (error " +
                std::to_string(GetLastError()) + ")";
        return false;
    }
    return protectFileForCurrentUser(g_store_path, error);
#else
    std::filesystem::rename(temporary, g_store_path, ec);
    if (ec) { error = "cannot replace IPC token store: " + ec.message(); return false; }
    return true;
#endif
}

bool loadLocked(std::string& error) {
    g_tokens.clear();
    if (!std::filesystem::exists(g_store_path)) return true;
    try {
        std::ifstream input(g_store_path, std::ios::binary);
        if (!input) { error = "cannot open IPC token store"; return false; }
        json root; input >> root;
        if (root.value("version", 0) != 1 || !root.contains("tokens") || !root["tokens"].is_array())
            throw std::runtime_error("unsupported token-store format");
        for (const auto& value : root["tokens"]) {
            StoredToken token;
            token.id = value.at("id").get<std::string>();
            token.display_name = value.value("display_name", token.id);
            token.digest = value.at("digest_sha256").get<std::string>();
            token.capabilities = value.value("capabilities", 0u);
            token.created_at = value.value("created_at", int64_t{0});
            token.expires_at = value.value("expires_at", int64_t{0});
            token.last_used_at = value.value("last_used_at", int64_t{0});
            token.revoked = value.value("revoked", false);
            token.allowed_cidrs = value.value("allowed_cidrs", std::vector<std::string>{});
            if (token.id.empty() || token.digest.size() != 64)
                throw std::runtime_error("invalid token record");
            if (!std::all_of(token.allowed_cidrs.begin(), token.allowed_cidrs.end(), validCidr))
                throw std::runtime_error("invalid token CIDR record");
            g_tokens.push_back(std::move(token));
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("invalid IPC token store: ") + e.what();
        g_tokens.clear(); return false;
    }
}

uint32_t requiredCapabilities(const std::string& method) {
    if (method == "ipc.admin.audit.export") return Admin | FilesWrite;
    if (method.rfind("ipc.admin.", 0) == 0) return Admin;
    if (method == "script.run_file") return Scripts | FilesRead;
    if (method == "addons.enable" || method == "addons.disable" || method == "addons.reload")
        return Addons;
    if (method == "project.open" || method == "scene.import_model" ||
        method == "terrain.import_heightmap" || method == "paint.import_channel")
        return FilesRead | SceneWrite;
    if (method == "project.save" || method == "terrain.export_heightmap" ||
        method == "paint.export_channel") return FilesWrite;
    if (method.rfind("render.", 0) == 0) {
        if (method == "render.start" || method == "render.start_sequence")
            return Render | FilesWrite;
        return Render;
    }
    if (method == "request_render" || method == "reset_accumulation") return Render;
    const bool read_method = method == "version" || method == "project.path" ||
        method == "undo_description" || method == "redo_description" ||
        method.find(".get") != std::string::npos || method.find(".list") != std::string::npos ||
        method.find(".status") != std::string::npos || method.find(".types") != std::string::npos ||
        method.find(".object_exists") != std::string::npos ||
        method.find(".sample_height") != std::string::npos;
    if (read_method) return Read;
    static const char* namespaces[] = {
        "scene.", "object.", "material.", "light.", "timeline.", "camera.",
        "world.", "post.", "sequence.", "keyframe.", "node.", "modifier.",
        "scatter.", "physics.", "fluid.", "gas.", "terrain.", "river.",
        "hair.", "paint.", "sculpt.", "project.", "undo", "redo"
    };
    for (const char* prefix : namespaces)
        if (method.rfind(prefix, 0) == 0) return SceneWrite;
    return 0;
}

bool parseIpv4(const std::string& text, uint32_t& output) {
    std::istringstream stream(text);
    int a = -1, b = -1, c = -1, d = -1;
    char dot1 = 0, dot2 = 0, dot3 = 0, extra = 0;
    if (!(stream >> a >> dot1 >> b >> dot2 >> c >> dot3 >> d) ||
        (stream >> extra) || dot1 != '.' || dot2 != '.' || dot3 != '.' ||
        a < 0 || a > 255 || b < 0 || b > 255 ||
        c < 0 || c > 255 || d < 0 || d > 255) return false;
    output = (static_cast<uint32_t>(a) << 24u) |
             (static_cast<uint32_t>(b) << 16u) |
             (static_cast<uint32_t>(c) << 8u) | static_cast<uint32_t>(d);
    return true;
}

bool addressAllowed(const std::string& address, const std::vector<std::string>& cidrs) {
    if (cidrs.empty()) return true;
    uint32_t candidate = 0;
    if (!parseIpv4(address, candidate)) return false;
    for (const auto& rule : cidrs) {
        const size_t slash = rule.find('/');
        const std::string network_text = rule.substr(0, slash);
        int bits = 32;
        try { if (slash != std::string::npos) bits = std::stoi(rule.substr(slash + 1)); }
        catch (...) { continue; }
        uint32_t network = 0;
        if (bits < 0 || bits > 32 || !parseIpv4(network_text, network)) continue;
        const uint32_t mask = bits == 0 ? 0u : 0xffffffffu << (32 - bits);
        if ((candidate & mask) == (network & mask)) return true;
    }
    return false;
}

bool validCidr(const std::string& cidr) {
    const size_t slash = cidr.find('/');
    uint32_t ignored = 0;
    int bits = 32;
    try { if (slash != std::string::npos) bits = std::stoi(cidr.substr(slash + 1)); }
    catch (...) { return false; }
    return bits >= 0 && bits <= 32 && parseIpv4(cidr.substr(0, slash), ignored);
}

bool pathWithin(const std::filesystem::path& candidate,
                const std::filesystem::path& root) {
    auto candidate_it = candidate.begin();
    for (auto root_it = root.begin(); root_it != root.end(); ++root_it, ++candidate_it) {
        if (candidate_it == candidate.end()) return false;
#ifdef _WIN32
        std::wstring lhs = candidate_it->wstring(), rhs = root_it->wstring();
        std::transform(lhs.begin(), lhs.end(), lhs.begin(), ::towlower);
        std::transform(rhs.begin(), rhs.end(), rhs.begin(), ::towlower);
        if (lhs != rhs) return false;
#else
        if (*candidate_it != *root_it) return false;
#endif
    }
    return true;
}

} // namespace

bool initialize(const std::string& bootstrap_token, uint32_t bootstrap_capabilities,
                const std::vector<std::string>& bootstrap_allowed_cidrs,
                const std::string& optional_store_path, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!std::all_of(bootstrap_allowed_cidrs.begin(), bootstrap_allowed_cidrs.end(), validCidr)) {
        error = "invalid bootstrap IPv4 CIDR rule"; return false;
    }
    g_store_path = optional_store_path.empty() ? defaultStorePath()
                                               : std::filesystem::path(optional_store_path);
    if (!loadLocked(error)) return false;
    if (!bootstrap_token.empty()) {
        std::string digest;
        if (!sha256(bootstrap_token, digest)) { error = "cannot hash bootstrap IPC token"; return false; }
        auto existing = std::find_if(g_tokens.begin(), g_tokens.end(), [&](const StoredToken& value) {
            return constantTimeEqual(value.digest, digest);
        });
        if (existing == g_tokens.end()) {
            StoredToken token;
            if (!randomHex(8, token.id)) { error = "cannot create IPC token id"; return false; }
            token.display_name = "Environment bootstrap";
            token.digest = std::move(digest);
            token.capabilities = bootstrap_capabilities;
            token.allowed_cidrs = bootstrap_allowed_cidrs;
            token.created_at = unixNow();
            g_tokens.push_back(std::move(token));
        } else {
            existing->capabilities = bootstrap_capabilities;
            existing->allowed_cidrs = bootstrap_allowed_cidrs;
            existing->revoked = false;
        }
        if (!saveLocked(error)) return false;
    }
    g_initialized = true;
    return true;
}

void shutdown() noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_initialized) { std::string ignored; saveLocked(ignored); }
    g_tokens.clear(); g_store_path.clear(); g_initialized = false;
}

Authentication authenticate(const std::string& raw_token,
                            const std::string& remote_address) {
    Authentication result;
    std::string digest;
    if (!sha256(raw_token, digest)) { result.error = "token hashing failed"; return result; }
    const int64_t now = unixNow();
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto& token : g_tokens) {
        if (!constantTimeEqual(token.digest, digest)) continue;
        if (token.revoked) { result.error = "token revoked"; return result; }
        if (token.expires_at > 0 && token.expires_at <= now) {
            result.error = "token expired"; return result;
        }
        if (!addressAllowed(remote_address, token.allowed_cidrs)) {
            result.error = "source address denied"; return result;
        }
        token.last_used_at = now;
        result.ok = true; result.token_id = token.id; result.capabilities = token.capabilities;
        return result;
    }
    result.error = "authentication failed";
    return result;
}

bool authorize(uint32_t capabilities, const std::string& method, std::string& error) {
    const uint32_t required = requiredCapabilities(method);
    if (required == 0) {
        error = "method is not enabled for remote IPC: " + method; return false;
    }
    if ((capabilities & required) != required) {
        error = "token lacks capability for method: " + method; return false;
    }
    return true;
}

bool authorizePath(const std::string& method, const std::string& path,
                   std::string& error) {
    const bool output = method == "project.save" || method == "terrain.export_heightmap" ||
                        method == "paint.export_channel" || method == "render.start" ||
                        method == "render.start_sequence" || method == "ipc.admin.audit.export";
    const char* root_text = std::getenv(output ? "RAYTROPHI_REMOTE_IPC_EXPORT_ROOT"
                                               : "RAYTROPHI_REMOTE_IPC_WORKSPACE_ROOT");
    if (!root_text || !*root_text) return true;
    std::error_code ec;
    const auto root = std::filesystem::weakly_canonical(std::filesystem::path(root_text), ec);
    if (ec) { error = "configured IPC path root is invalid"; return false; }
    std::filesystem::path requested(path);
    if (requested.is_relative()) requested = root / requested;
    const auto canonical = std::filesystem::weakly_canonical(requested, ec);
    if (ec || !pathWithin(canonical, root)) {
        error = "path is outside the configured IPC root"; return false;
    }
    return true;
}

bool createToken(const std::string& display_name, uint32_t capabilities,
                 const std::vector<std::string>& allowed_cidrs,
                 int64_t expires_at, TokenInfo& out_info,
                 std::string& out_raw_token, std::string& error) {
    constexpr uint32_t all_capabilities = Read | SceneWrite | Render | FilesRead |
                                          FilesWrite | Scripts | Addons | Admin;
    if (display_name.empty() || capabilities == 0 || (capabilities & ~all_capabilities) != 0) {
        error = "token name and valid capabilities are required"; return false;
    }
    if (expires_at > 0 && expires_at <= unixNow()) {
        error = "token expiry must be in the future"; return false;
    }
    if (!std::all_of(allowed_cidrs.begin(), allowed_cidrs.end(), validCidr)) {
        error = "invalid IPv4 CIDR rule"; return false;
    }
    StoredToken token;
    std::string secret;
    if (!randomHex(8, token.id) || !randomHex(32, secret)) { error = "secure token generation failed"; return false; }
    out_raw_token = "rt_" + secret;
    if (!sha256(out_raw_token, token.digest)) { out_raw_token.clear(); error = "token hashing failed"; return false; }
    token.display_name = display_name; token.capabilities = capabilities;
    token.allowed_cidrs = allowed_cidrs;
    token.created_at = unixNow(); token.expires_at = expires_at;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_tokens.push_back(token);
    if (!saveLocked(error)) { g_tokens.pop_back(); out_raw_token.clear(); return false; }
    out_info = token; return true;
}

bool revokeToken(const std::string& token_id, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto token = std::find_if(g_tokens.begin(), g_tokens.end(), [&](const StoredToken& value) { return value.id == token_id; });
    if (token == g_tokens.end()) { error = "token not found"; return false; }
    token->revoked = true;
    return saveLocked(error);
}

bool updateToken(const std::string& token_id, uint32_t capabilities,
                 const std::vector<std::string>& allowed_cidrs,
                 int64_t expires_at, std::string& error) {
    constexpr uint32_t all = Read | SceneWrite | Render | FilesRead | FilesWrite |
                             Scripts | Addons | Admin;
    if (capabilities == 0 || (capabilities & ~all) != 0 ||
        (expires_at > 0 && expires_at <= unixNow()) ||
        !std::all_of(allowed_cidrs.begin(), allowed_cidrs.end(), validCidr)) {
        error = "invalid token policy"; return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = std::find_if(g_tokens.begin(), g_tokens.end(),
        [&](const StoredToken& token) { return token.id == token_id; });
    if (found == g_tokens.end() || found->revoked) {
        error = "active token not found"; return false;
    }
    const auto previous = *found;
    found->capabilities = capabilities; found->allowed_cidrs = allowed_cidrs;
    found->expires_at = expires_at;
    if (!saveLocked(error)) { *found = previous; return false; }
    return true;
}

bool rotateToken(const std::string& token_id, TokenInfo& out_info,
                 std::string& out_raw_token, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto old = std::find_if(g_tokens.begin(), g_tokens.end(), [&](const StoredToken& value) { return value.id == token_id; });
    if (old == g_tokens.end()) { error = "token not found"; return false; }
    StoredToken replacement;
    std::string secret;
    if (!randomHex(8, replacement.id) || !randomHex(32, secret)) {
        error = "secure token generation failed"; return false;
    }
    out_raw_token = "rt_" + secret;
    if (!sha256(out_raw_token, replacement.digest)) {
        out_raw_token.clear(); error = "token hashing failed"; return false;
    }
    replacement.display_name = old->display_name;
    replacement.capabilities = old->capabilities;
    replacement.allowed_cidrs = old->allowed_cidrs;
    replacement.created_at = unixNow();
    replacement.expires_at = old->expires_at;
    const size_t old_index = static_cast<size_t>(std::distance(g_tokens.begin(), old));
    const bool previous_revoked = old->revoked;
    g_tokens[old_index].revoked = true;
    g_tokens.push_back(replacement);
    if (!saveLocked(error)) {
        g_tokens.pop_back(); g_tokens[old_index].revoked = previous_revoked;
        out_raw_token.clear(); return false;
    }
    out_info = replacement; return true;
}

std::vector<TokenInfo> listTokens() {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::vector<TokenInfo> result;
    result.reserve(g_tokens.size());
    for (const auto& token : g_tokens) result.push_back(token);
    return result;
}

} // namespace rtipc_security
