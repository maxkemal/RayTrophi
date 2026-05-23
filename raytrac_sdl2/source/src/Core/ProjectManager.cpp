#include "ProjectManager.h"
#include "globals.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "Backend/IBackend.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"
#include "AssimpLoader.h"
#include "Triangle.h"
#include "OptixAccelManager.h"
#include "InstanceManager.h"
#include "PrincipledBSDF.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "WaterSystem.h"
#include "MeshModifiers.h"
#include "Paint/PaintLayerStack.h"
#include "json.hpp"
#include "simdjson.h"
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <limits>
#include <thread>
#include <chrono>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

#include <chrono>
#include "TerrainManager.h"
#include "RiverSpline.h"

#include "stb_image_write.h"
#include "stb_image.h"
#include "ColorProcessingParams.h"

// Helper for stbi_write_to_func
namespace {
    void write_to_vector_func(void* context, void* data, int size) {
        auto* vec = static_cast<std::vector<char>*>(context);
        const char* d = static_cast<const char*>(data);
        vec->insert(vec->end(), d, d + size);
    }

    // JPG quality used for embedded color textures. q=92 is visually transparent
    // for albedo/emission while shrinking 5-10x vs PNG.
    constexpr int kEmbeddedJpegQuality = 92;

    bool isLossyEncodableUsage(const std::string& usage) {
        // Only pure color channels tolerate JPG. Normal/roughness/metallic/height/
        // opacity/transmission carry numeric data — JPG artifacts corrupt shading
        // or masks, so they must stay PNG.
        return usage == "albedo" || usage == "emission";
    }

    // Role-aware encoder: JPG for color textures with fully-opaque alpha,
    // PNG everywhere else (including any albedo/emission that uses alpha).
    bool encodeTextureRoleAware(const uint8_t* rgba, int width, int height,
                                const std::string& usage,
                                std::vector<char>& out_bytes,
                                std::string& out_ext) {
        out_bytes.clear();
        out_ext.clear();

        const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);

        bool prefer_jpg = isLossyEncodableUsage(usage);
        if (prefer_jpg) {
            for (size_t i = 0; i < pixel_count; ++i) {
                if (rgba[i * 4 + 3] != 255) { prefer_jpg = false; break; }
            }
        }

        if (prefer_jpg) {
            std::vector<uint8_t> rgb(pixel_count * 3);
            for (size_t i = 0; i < pixel_count; ++i) {
                rgb[i * 3 + 0] = rgba[i * 4 + 0];
                rgb[i * 3 + 1] = rgba[i * 4 + 1];
                rgb[i * 3 + 2] = rgba[i * 4 + 2];
            }
            if (stbi_write_jpg_to_func(write_to_vector_func, &out_bytes,
                                       width, height, 3, rgb.data(),
                                       kEmbeddedJpegQuality) && !out_bytes.empty()) {
                out_ext = ".jpg";
                return true;
            }
            out_bytes.clear();
        }

        if (stbi_write_png_to_func(write_to_vector_func, &out_bytes,
                                   width, height, 4, rgba, width * 4) &&
            !out_bytes.empty()) {
            out_ext = ".png";
            return true;
        }
        return false;
    }

    // Result of attempting PNG → JPG recompression on a passthrough blob.
    // If recompressed is true, `bytes` owns the new encoded data; otherwise
    // callers should write the original bytes through unchanged.
    struct ColorRecompressionResult {
        bool recompressed = false;
        std::vector<char> bytes;
        std::string ext;
    };

    // For albedo/emission blobs that are PNG with fully-opaque alpha, decode
    // and re-encode as JPG q92. Anything else (other roles, non-PNG inputs,
    // any alpha < 255, or recompression that didn't shrink the file) returns
    // recompressed=false so the caller falls back to byte-for-byte passthrough.
    ColorRecompressionResult pickColorRecompression(const char* in_data, size_t in_size,
                                                     const std::string& usage) {
        ColorRecompressionResult result;
        if (!isLossyEncodableUsage(usage) || in_data == nullptr || in_size < 8) {
            return result;
        }

        const unsigned char* magic = reinterpret_cast<const unsigned char*>(in_data);
        const bool is_png = (magic[0] == 0x89 && magic[1] == 0x50 && magic[2] == 0x4E && magic[3] == 0x47 &&
                             magic[4] == 0x0D && magic[5] == 0x0A && magic[6] == 0x1A && magic[7] == 0x0A);
        if (!is_png) {
            return result;
        }

        int w = 0, h = 0, ch = 0;
        unsigned char* decoded = stbi_load_from_memory(
            reinterpret_cast<const stbi_uc*>(in_data),
            static_cast<int>(in_size),
            &w, &h, &ch, 4);
        if (!decoded || w <= 0 || h <= 0) {
            if (decoded) stbi_image_free(decoded);
            return result;
        }

        const size_t pixel_count = static_cast<size_t>(w) * static_cast<size_t>(h);
        bool fully_opaque = true;
        for (size_t i = 0; i < pixel_count; ++i) {
            if (decoded[i * 4 + 3] != 255) { fully_opaque = false; break; }
        }
        if (!fully_opaque) {
            stbi_image_free(decoded);
            return result;
        }

        std::vector<char> recoded;
        std::string recoded_ext;
        const bool ok = encodeTextureRoleAware(decoded, w, h, usage, recoded, recoded_ext);
        stbi_image_free(decoded);

        if (!ok || recoded_ext != ".jpg" || recoded.empty() || recoded.size() >= in_size) {
            return result;
        }

        result.recompressed = true;
        result.bytes = std::move(recoded);
        result.ext = std::move(recoded_ext);
        return result;
    }

    class ScopedPerfTimer {
    public:
        explicit ScopedPerfTimer(std::string label)
            : label_(std::move(label)), start_(std::chrono::steady_clock::now()) {}

        ~ScopedPerfTimer() {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_).count();
            SCENE_LOG_INFO("[Perf] " + label_ + ": " + std::to_string(elapsed) + " ms");
        }

    private:
        std::string label_;
        std::chrono::steady_clock::time_point start_;
    };

    size_t streamFileToOutput(std::ifstream& in, std::ofstream& out, std::vector<char>& buffer) {
        size_t total_written = 0;
        while (in) {
            in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize read_count = in.gcount();
            if (read_count <= 0) {
                break;
            }

            out.write(buffer.data(), read_count);
            total_written += static_cast<size_t>(read_count);
        }
        return total_written;
    }

    std::string sanitizeFilenameComponent(const std::string& input) {
        std::string out;
        out.reserve(input.size());
        for (unsigned char c : input) {
            if ((c >= 'a' && c <= 'z') ||
                (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') ||
                c == '.' || c == '_' || c == '-') {
                out.push_back(static_cast<char>(c));
            } else {
                out.push_back('_');
            }
        }
        if (out.empty()) out = "texture";
        return out;
    }

    std::string sniffTextureExtensionFromBytes(const std::vector<char>& data) {
        if (data.size() >= 8) {
            const unsigned char* b = reinterpret_cast<const unsigned char*>(data.data());
            if (b[0] == 0x89 && b[1] == 0x50 && b[2] == 0x4E && b[3] == 0x47 &&
                b[4] == 0x0D && b[5] == 0x0A && b[6] == 0x1A && b[7] == 0x0A) {
                return ".png";
            }
        }
        if (data.size() >= 3) {
            const unsigned char* b = reinterpret_cast<const unsigned char*>(data.data());
            if (b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF) {
                return ".jpg";
            }
        }
        if (data.size() >= 4) {
            const char* b = data.data();
            if (std::memcmp(b, "DDS ", 4) == 0) return ".dds";
            if (std::memcmp(b, "qoif", 4) == 0) return ".qoi";
            if (std::memcmp(b, "RIFF", 4) == 0 && data.size() >= 12 && std::memcmp(b + 8, "WEBP", 4) == 0) return ".webp";
            if (data.size() >= 10 && std::memcmp(b, "#?RADIANCE", 10) == 0) return ".hdr";
            if (data.size() >= 6 && std::memcmp(b, "#?RGBE", 6) == 0) return ".hdr";
            if (std::memcmp(b, "v/1\x01", 4) == 0 || std::memcmp(b, "v/1\x02", 4) == 0) return ".exr";
        }
        if (data.size() >= 18) {
            const unsigned char* b = reinterpret_cast<const unsigned char*>(data.data());
            const uint8_t image_type = b[2];
            const uint8_t pixel_depth = b[16];
            if ((image_type == 2 || image_type == 3 || image_type == 10 || image_type == 11) &&
                (pixel_depth == 8 || pixel_depth == 16 || pixel_depth == 24 || pixel_depth == 32)) {
                return ".tga";
            }
        }
        return ".bin";
    }

    std::string chooseMaterializedTextureExtension(const std::filesystem::path& source_path, const std::vector<char>* cached_bytes = nullptr) {
        std::string ext = source_path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (!ext.empty() && ext != ".bin") {
            return ext;
        }
        if (cached_bytes && !cached_bytes->empty()) {
            return sniffTextureExtensionFromBytes(*cached_bytes);
        }
        return ext.empty() ? std::string(".png") : ext;
    }

    struct PreviousEmbeddedTextureEntry {
        std::filesystem::path bin_path;
        int64_t offset = -1;
        int64_t size = 0;
        std::string format;
    };

    std::unordered_map<std::string, PreviousEmbeddedTextureEntry> g_previous_embedded_texture_entries;

    std::string previousEmbeddedTextureKey(const std::string& original_name,
                                           const std::string& usage,
                                           int width,
                                           int height) {
        return original_name + "\n" + usage + "\n" +
               std::to_string(width) + "x" + std::to_string(height);
    }

    void loadPreviousEmbeddedTextureEntries(const std::filesystem::path& json_path,
                                            const std::filesystem::path& bin_path) {
        g_previous_embedded_texture_entries.clear();
        if (!std::filesystem::exists(json_path) || !std::filesystem::exists(bin_path)) {
            return;
        }

        try {
            std::ifstream in(json_path);
            if (!in.is_open()) {
                return;
            }

            nlohmann::json root;
            in >> root;
            if (!root.contains("textures") || !root["textures"].is_array()) {
                return;
            }

            for (const auto& tex : root["textures"]) {
                if (tex.value("mode", std::string{}) != "embed") {
                    continue;
                }

                const std::string original_name = tex.value("original_name", std::string{});
                const std::string usage = tex.value("usage", std::string{});
                const int width = tex.value("width", 0);
                const int height = tex.value("height", 0);
                const int64_t offset = tex.value("offset", int64_t(-1));
                const int64_t size = tex.value("size", int64_t(0));
                if (original_name.empty() || usage.empty() || width <= 0 || height <= 0 || offset < 0 || size <= 0) {
                    continue;
                }

                PreviousEmbeddedTextureEntry entry;
                entry.bin_path = bin_path;
                entry.offset = offset;
                entry.size = size;
                entry.format = tex.value("format", std::string{});
                g_previous_embedded_texture_entries[previousEmbeddedTextureKey(original_name, usage, width, height)] = std::move(entry);
            }

            SCENE_LOG_INFO("[ProjectManager] Loaded " +
                           std::to_string(g_previous_embedded_texture_entries.size()) +
                           " previous embedded texture entries for save reuse.");
        } catch (const std::exception& e) {
            g_previous_embedded_texture_entries.clear();
            SCENE_LOG_WARN("[ProjectManager] Failed to read previous embedded texture manifest: " + std::string(e.what()));
        }
    }

    bool readPreviousEmbeddedTextureBlob(const PreviousEmbeddedTextureEntry& entry,
                                         std::vector<char>& out) {
        out.clear();
        if (entry.offset < 0 || entry.size <= 0 || !std::filesystem::exists(entry.bin_path)) {
            return false;
        }

        std::error_code ec;
        const auto source_size = std::filesystem::file_size(entry.bin_path, ec);
        if (ec || source_size < static_cast<uint64_t>(entry.offset + entry.size)) {
            return false;
        }

        std::ifstream in(entry.bin_path, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }

        in.seekg(entry.offset, std::ios::beg);
        out.resize(static_cast<size_t>(entry.size));
        in.read(out.data(), static_cast<std::streamsize>(out.size()));
        if (in.gcount() != static_cast<std::streamsize>(out.size())) {
            out.clear();
            return false;
        }
        return true;
    }

    bool copyPreviousEmbeddedTextureBlob(const PreviousEmbeddedTextureEntry& entry,
                                         std::ofstream& out,
                                         std::vector<char>& buffer) {
        if (entry.offset < 0 || entry.size <= 0 || !std::filesystem::exists(entry.bin_path)) {
            return false;
        }

        std::error_code ec;
        const auto source_size = std::filesystem::file_size(entry.bin_path, ec);
        if (ec || source_size < static_cast<uint64_t>(entry.offset + entry.size)) {
            return false;
        }

        std::ifstream in(entry.bin_path, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }

        in.seekg(entry.offset, std::ios::beg);
        int64_t remaining = entry.size;
        while (remaining > 0 && in) {
            const std::streamsize chunk = static_cast<std::streamsize>(
                std::min<int64_t>(remaining, static_cast<int64_t>(buffer.size())));
            in.read(buffer.data(), chunk);
            const std::streamsize read_count = in.gcount();
            if (read_count <= 0) {
                return false;
            }
            out.write(buffer.data(), read_count);
            remaining -= static_cast<int64_t>(read_count);
        }

        return remaining == 0;
    }

}

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
enum class ProjectBackendMode {
    CPU,
    OPTIX,
    VULKAN
};

static ProjectBackendMode modeFromSettings(const RenderSettings& settings) {
    if (settings.use_vulkan) return ProjectBackendMode::VULKAN;
    if (settings.use_optix) return ProjectBackendMode::OPTIX;
    return ProjectBackendMode::CPU;
}

static ProjectBackendMode modeFromBackendPtr(Backend::IBackend* backend) {
    if (!backend) return ProjectBackendMode::CPU;

    Backend::BackendType type = Backend::BackendType::CPU_EMBREE;
    try {
        type = backend->getInfo().type;
    } catch (...) {
        return ProjectBackendMode::CPU;
    }

    if (type == Backend::BackendType::VULKAN_RT || type == Backend::BackendType::VULKAN_COMPUTE) {
        return ProjectBackendMode::VULKAN;
    }
    if (type == Backend::BackendType::OPTIX) {
        return ProjectBackendMode::OPTIX;
    }
    return ProjectBackendMode::CPU;
}

static const char* modeToString(ProjectBackendMode mode) {
    switch (mode) {
        case ProjectBackendMode::VULKAN: return "Vulkan";
        case ProjectBackendMode::OPTIX: return "OptiX";
        default: return "CPU";
    }
}

static fs::path pathFromUtf8(const std::string& utf8_path) {
#ifdef _WIN32
    auto toWide = [](const std::string& src, UINT codepage, DWORD flags) -> std::wstring {
        const int size = MultiByteToWideChar(codepage, flags, src.c_str(), -1, nullptr, 0);
        if (size <= 0) return {};
        std::wstring out(static_cast<size_t>(size - 1), L'\0');
        if (MultiByteToWideChar(codepage, flags, src.c_str(), -1, out.data(), size) <= 0) return {};
        return out;
    };

    std::wstring wide = toWide(utf8_path, CP_UTF8, MB_ERR_INVALID_CHARS);
    if (wide.empty()) {
        wide = toWide(utf8_path, CP_ACP, 0);
    }
    if (!wide.empty()) {
        return fs::path(wide);
    }
#endif
    return fs::path(utf8_path);
}

static simdjson::error_code loadJsonRootFromFile(simdjson::dom::parser& parser,
                                                 const std::string& filepath,
                                                 simdjson::dom::element& root,
                                                 std::string* read_error = nullptr) {
    auto readFileToString = [](const fs::path& path, std::string& out_text, std::string& out_err) -> bool {
        std::ifstream in_json(path, std::ios::binary);
        if (!in_json.is_open()) {
            out_err = "open failed";
            return false;
        }

        out_text.assign(std::istreambuf_iterator<char>(in_json), std::istreambuf_iterator<char>());
        if (!in_json.good() && !in_json.eof()) {
            out_err = "stream read failed";
            return false;
        }

        return true;
    };

    std::string json_text;
    std::string last_error;

#ifdef _WIN32
    auto readFileToStringWin32 = [](const std::string& utf8_path, std::string& out_text, std::string& out_err) -> bool {
        auto toWide = [](const std::string& src, UINT codepage, DWORD flags) -> std::wstring {
            int size = MultiByteToWideChar(codepage, flags, src.c_str(), -1, nullptr, 0);
            if (size <= 0) return {};
            std::wstring out(static_cast<size_t>(size), L'\0');
            if (MultiByteToWideChar(codepage, flags, src.c_str(), -1, out.data(), size) <= 0) return {};
            out.resize(static_cast<size_t>(size - 1));
            return out;
        };

        std::wstring path_w = toWide(utf8_path, CP_UTF8, MB_ERR_INVALID_CHARS);
        if (path_w.empty()) {
            path_w = toWide(utf8_path, CP_ACP, 0);
        }
        if (path_w.empty()) {
            out_err = "utf8/acp to wide conversion failed";
            return false;
        }

        HANDLE file = CreateFileW(path_w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            out_err = "CreateFileW failed (err=" + std::to_string(GetLastError()) + ")";
            return false;
        }

        LARGE_INTEGER file_size{};
        if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0) {
            out_err = "GetFileSizeEx failed (err=" + std::to_string(GetLastError()) + ")";
            CloseHandle(file);
            return false;
        }

        if (file_size.QuadPart > static_cast<LONGLONG>(std::numeric_limits<size_t>::max())) {
            out_err = "file too large";
            CloseHandle(file);
            return false;
        }

        out_text.resize(static_cast<size_t>(file_size.QuadPart));
        DWORD read_total = 0;
        while (read_total < static_cast<DWORD>(out_text.size())) {
            DWORD chunk_read = 0;
            const DWORD remaining = static_cast<DWORD>(out_text.size()) - read_total;
            if (!ReadFile(file, out_text.data() + read_total, remaining, &chunk_read, nullptr)) {
                out_err = "ReadFile failed (err=" + std::to_string(GetLastError()) + ")";
                CloseHandle(file);
                return false;
            }
            if (chunk_read == 0) break;
            read_total += chunk_read;
        }
        out_text.resize(read_total);
        CloseHandle(file);
        return true;
    };
#endif

    const fs::path utf8_path = pathFromUtf8(filepath);

    if (!readFileToString(utf8_path, json_text, last_error)) {
        fs::path native_path(filepath);
        std::string native_error;
        if (!readFileToString(native_path, json_text, native_error)) {
#ifdef _WIN32
            std::string winapi_error;
            if (readFileToStringWin32(filepath, json_text, winapi_error)) {
                last_error = "u8path=" + last_error + " | native=" + native_error + " | winapi=ok";
            } else {
#endif
            std::error_code ec_u8_exists;
            std::error_code ec_native_exists;
            const bool u8_exists = fs::exists(utf8_path, ec_u8_exists);
            const bool native_exists = fs::exists(native_path, ec_native_exists);
            if (read_error) {
                *read_error = "u8path=" + last_error +
                              " (exists=" + std::string(u8_exists ? "true" : "false") + ")" +
                              " | native=" + native_error +
                              " (exists=" + std::string(native_exists ? "true" : "false") + ")" +
#ifdef _WIN32
                              " | winapi=" + winapi_error +
#endif
                              " | cwd=" + fs::current_path().string();
            }
            return simdjson::IO_ERROR;
#ifdef _WIN32
            }
#endif
        }
    }

    simdjson::padded_string padded_json(json_text);
    return parser.parse(padded_json).get(root);
}
}

// Global project data instance
ProjectData g_project;

// Extern access to UI State (referenced for serialization)
#include "../UI/scene_ui_animgraph.hpp"

// ============================================================================
// Helper Functions
// ============================================================================

static json vec3ToJson(const Vec3& v) {
    return { v.x, v.y, v.z };
}

static Vec3 jsonToVec3(const json& j) {
    if (j.is_array() && j.size() >= 3)
        return Vec3(j[0], j[1], j[2]);
    return Vec3(0, 0, 0);
}

static json mat4ToJson(const Matrix4x4& m) {
    json j = json::array();
    for(int i = 0; i < 4; ++i)
        for(int k = 0; k < 4; ++k)
            j.push_back(m.m[i][k]);
    return j;
}

static Matrix4x4 jsonToMat4(const json& j) {
    Matrix4x4 m;
    for(int i=0; i<4; ++i) for(int k=0; k<4; ++k) m.m[i][k] = (i==k ? 1.0f : 0.0f);

    if (j.is_array() && j.size() == 16) {
        int idx = 0;
        for(int i = 0; i < 4; ++i)
            for(int k = 0; k < 4; ++k)
                m.m[i][k] = j[idx++];
    }
    return m;
}

// Helper to check for identity matrix with epsilon
static bool isIdentity(const Matrix4x4& m) {
    const float epsilon = 0.0001f;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            if (std::abs(m.m[r][c] - expected) > epsilon) return false;
        }
    }
    return true;
}

// Binary IO Helpers
static void writeStringBinary(std::ofstream& out, const std::string& str) {
    uint16_t len = static_cast<uint16_t>(str.length());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len > 0) out.write(str.data(), len);
}

// simdjson helpers
static json sjsonToNlohmann(simdjson::dom::element el) {
    return json::parse(std::string(simdjson::minify(el)));
}
static json sjsonToNlohmann(simdjson::simdjson_result<simdjson::dom::element> res) {
    simdjson::dom::element el;
    if (res.get(el)) return json();
    return sjsonToNlohmann(el);
}

static Vec3 sjsonToVec3(simdjson::dom::element el) {
    simdjson::dom::array arr;
    if (el.get_array().get(arr)) return Vec3(0, 0, 0);
    float x = 0, y = 0, z = 0;
    size_t i = 0;
    for (simdjson::dom::element val : arr) {
        double d = 0;
        val.get(d);
        if (i == 0) x = (float)d;
        else if (i == 1) y = (float)d;
        else if (i == 2) z = (float)d;
        i++;
    }
    return Vec3(x, y, z);
}
static Vec3 sjsonToVec3(simdjson::simdjson_result<simdjson::dom::element> res) {
    simdjson::dom::element el;
    if (res.get(el)) return Vec3(0, 0, 0);
    return sjsonToVec3(el);
}

static Matrix4x4 sjsonToMat4(simdjson::dom::element el) {
    Matrix4x4 m;
    simdjson::dom::array arr;
    if (el.get_array().get(arr) || arr.size() != 16) return m;
    int idx = 0;
    for (simdjson::dom::element val : arr) {
        int r = idx / 4;
        int c = idx % 4;
        double d = 0;
        val.get(d);
        m.m[r][c] = (float)d;
        idx++;
    }
    return m;
}
static Matrix4x4 sjsonToMat4(simdjson::simdjson_result<simdjson::dom::element> res) {
    simdjson::dom::element el;
    if (res.get(el)) return Matrix4x4();
    return sjsonToMat4(el);
}

static std::string readStringBinary(std::ifstream& in) {
    uint16_t len = 0;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!in || len == 0) return "";
    std::string str(len, '\0');
    in.read(&str[0], len);
    return str;
}

// Generate internal package path like "models/0001/filename.glb"
std::string ProjectManager::generatePackagePath(const std::string& original_path, 
                                                 const std::string& folder, 
                                                 uint32_t id) {
    fs::path orig(original_path);
    std::ostringstream oss;
    // Use subfolders to prevent name collisions and preserve relative paths (e.g. for .bin files)
    oss << folder << "/" << std::setfill('0') << std::setw(4) << id << "/" << orig.filename().string();
    return oss.str();
}

// ============================================================================
// Serialization / Sync Logic
// ============================================================================

void ProjectManager::syncProjectToScene(SceneData& scene) {
    // 1. Build a Quick Lookup Map for Scene Objects
    std::unordered_map<std::string, std::shared_ptr<Triangle>> scene_obj_map;
    // Utilize cache if available, otherwise build it
    for (const auto& obj : scene.world.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->nodeName.empty()) {
                scene_obj_map[tri->nodeName] = tri;
            }
        }
    }

    // 2. Iterate Imported Models and Sync
    for (auto& model : g_project.imported_models) {
        // Clear old deleted list - we will rebuild it
        // model.deleted_objects.clear(); // Keep manual ones? No, state should track scene.
        // Actually, if we just check existing model.objects, we might miss objects that were ALREADY deleted and removed from model.objects?
        // Wait, imported_models.objects is the LIST of ALL imported nodes.
        // We assume model.objects contains everything from the source file.
        // If it was partial, we are in trouble.
        // Assimp load populates it fully in importModel phase.
        
        // Re-evaluate deleted objects
        std::vector<std::string> current_deleted;
        
        for (auto& inst : model.objects) {
            auto it = scene_obj_map.find(inst.node_name);
            if (it != scene_obj_map.end()) {
                // Object Exists in Scene - Update Status
                auto tri = it->second;
                
                // Update Transform
                if (auto th = tri->getTransformPtr()) {
                    inst.transform = th->base;
                } else {
                    // Fallback if no handle (shouldn't happen for valid objects)
                }
                
                // Update Material
                inst.material_id = tri->getMaterialID();
                
                // Update Visibility (approximation, as Triangle doesn't strictly have 'visible' flag exposed easily usually, 
                // but let's assume if it's in the list, it's visible. 
                // If you implement a specific hide flag in Triangle, use it here.)
                inst.visible = true; 
            } else {
                // Object MISSING from Scene -> It is DELETED
                current_deleted.push_back(inst.node_name);
                inst.visible = false; // Mark implicit
            }
        }
        
        model.deleted_objects = current_deleted;
    }
    
    // 3. Sync Procedurals
    // Procedurals are usually directly managed in `procedural_objects` list by Add/Remove calls.
    // But we should sync transforms just in case.
    for (auto& proc : g_project.procedural_objects) {
        // Find procedural object in scene by Name (or ID if we stored it in Triangle?)
        // Currently relying on Name matching for procedurals is risky if duplicates allowed.
        // But let's try.
        // Better: Procedural creation assigns 'nodeName' = 'display_name'.
        auto it = scene_obj_map.find(proc.display_name);
        if (it != scene_obj_map.end()) {
             if (auto th = it->second->getTransformPtr()) {
                 proc.transform = th->base;
             }
             proc.material_id = it->second->getMaterialID();
        }
    }
}

// ============================================================================
// Project Lifecycle
// ============================================================================

// ============================================================================
// Project Lifecycle
// ============================================================================

void ProjectManager::newProject(SceneData& scene, Renderer& renderer, bool defer_backend_reset) {
    // 1. Ensure GPU is idle before clearing resources
    if (render_settings.use_optix) {
        cudaDeviceSynchronize();
    }

    g_project.clear();
    m_package_files.clear();
    clearEmbeddedTextureCache();
    
    // 2. Reset Globals
    bool was_optix = render_settings.use_optix;
    bool was_vulkan = render_settings.use_vulkan;
    bool was_denoiser = render_settings.use_denoiser;
    DenoiserMode was_denoiser_mode = render_settings.denoiser_mode;
    
    render_settings = RenderSettings(); // Default constructor resets to defaults
    
    // Restore Device Preference
    render_settings.use_optix = was_optix;
    render_settings.use_vulkan = was_vulkan;
    render_settings.use_denoiser = was_denoiser;
    render_settings.denoiser_mode = was_denoiser_mode;

    renderer.world.reset();             // Reset Atmosphere/Godrays
    // Reset global and scene color processing (tonemap/postfx) to defaults for new project
    extern ColorProcessor color_processor; // global in Main.cpp
    color_processor.params = ColorProcessor::ColorProcessingParams();
    scene.color_processor.params = color_processor.params;
    
    // 3. Clear Central Subsystems
    MaterialManager::getInstance().clear();
    RiverManager::getInstance().clear(&scene);  // Clear rivers BEFORE WaterManager (rivers own WaterSurfaces)
    WaterManager::getInstance().clear();
    TerrainManager::getInstance().removeAllTerrains(scene);
    InstanceManager::getInstance().clearAll();  // Clear foliage/scatter instances 
    VDBVolumeManager::getInstance().unloadAll(); // Clear VDB/Gas GPU handles and reset IDs
    renderer.getHairSystem().clearAll();        // Clear all hair grooms

    // Ensure scene containers are empty before backend geometry rebuild.
    // This guarantees "empty scene" cleanup path and prevents stale instance carryover.
    scene.clear();

    // Do not call uploadHairToGPU() here after clearing the hair system.
    // The empty-scene backend rebuild below is the authoritative GPU reset.
    // Calling the hair upload path with zero grooms can make Vulkan interpret
    // stale pre-reset BLAS entries as hair BLASes and destroy old buffers twice
    // during project open/new-project transitions.

    // 4. CRITICAL: Rebuild backend geometry for the empty scene.
    // This ensures that all GPU BLAS, instances, and TLAS are fully cleared/reset.
    // If we only call uploadHairToGPU, old meshes from previous projects may linger
    // in the AccelManager's internal instance list, causing slowness even if
    // they aren't visible.
    //
    // Vulkan must hit this path too: VulkanBackendAdapter::rebuildAccelerationStructure
    // is the only place that purges m_uploadedImages and the material preview
    // descriptor bindings. Skipping it on Vulkan leaves textures from the
    // previous scene bleeding into the new project's material preview mode.
    //
    // openProject passes defer_backend_reset=true: the post-load finalize already
    // drives a full rebuild (Main.cpp syncActiveRenderBackendScene or the
    // g_vulkan_rebuild_pending block), so running it here on the empty pre-load
    // scene is wasted work for foliage-heavy projects.
    if (!defer_backend_reset &&
        (render_settings.use_optix || render_settings.use_vulkan) && renderer.m_backend) {
        renderer.rebuildBackendGeometry(scene);
    }

    // [VIEWPORT TEXTURE LEAK FIX] The dedicated raster viewport backend keeps its
    // own m_uploadedImages cache that rebuildBackendGeometry never touches (the
    // render-backend pending-block only calls rebuildAccelerationStructure on
    // g_backend). Without purging it here, consecutive project loads leave stale
    // texture IDs in the viewport's material preview descriptor array, causing old
    // textures to bleed into the new project's Material Preview mode.
    extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
    if (g_viewport_backend) {
        try { g_viewport_backend->waitForCompletion(); } catch (...) {}
        // Use resetForProjectReload (not rebuildAccelerationStructure directly):
        // the viewport backend owns a material preview descriptor set whose
        // binding-1 texture array holds VkImageViews from m_uploadedImages. A
        // bare rebuildAccelerationStructure would destroy those images while
        // the descSet still references them, crashing the driver on the next
        // material-preview draw. resetForProjectReload tears the descSet down
        // first so it will be recreated fresh against the new project.
        if (auto* vkViewport = dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get())) {
            vkViewport->resetForProjectReload();
        } else {
            g_viewport_backend->rebuildAccelerationStructure();
        }
        g_viewport_raster_rebuild_pending = true;
    }


    // 4. Clear Animation/Bone Caches (CRITICAL - prevents bone corruption on new project)
    // Clear per-model animator contexts and their bone caches
    for (auto& ctx : scene.importedModelContexts) {
        if (ctx.animator) {
            ctx.animator->clear();
        }
        if (ctx.graph) {
            ctx.graph.reset();
        }
        ctx.members.clear();
    }
    scene.importedModelContexts.clear();
    
    // Clear scene-level animation and bone data
    scene.animationDataList.clear();
    scene.boneData.clear();
    
    // Clear renderer's cached bone matrices
    renderer.finalBoneMatrices.clear();
    
    // Clear Animation Graphs
    g_animGraphUI.graphs.clear();
    g_animGraphUI.activeCharacter = "";
    g_animGraphUI = AnimGraphUIState();
    
    SCENE_LOG_INFO("New project created. Subsystems and animation caches cleared.");
}

// ============================================================================
// NEW SAVE PROJECT (Self-Contained with Geometry)
// ============================================================================

bool ProjectManager::saveProject(SceneData& scene, RenderSettings& settings, Renderer& renderer,
                                  std::function<void(int, const std::string&)> progress_callback) {
    if (g_project.current_file_path.empty()) {
        SCENE_LOG_ERROR("No file path set. Use saveProject(filepath, scene, settings) instead.");
        return false;
    }
    return saveProject(g_project.current_file_path, scene, settings, renderer, progress_callback);
}

bool ProjectManager::saveProject(const std::string& filepath, SceneData& scene, RenderSettings& settings, Renderer& renderer,
                                  std::function<void(int, const std::string&)> progress_callback) {
    ScopedPerfTimer total_timer("ProjectManager::saveProject total");

    if (progress_callback) progress_callback(0, "Preparing to save...");

    // Update Project Name from File Path if it's "Untitled" or empty
    // This ensures assets (like terrains) get the correct prefix
    fs::path p(filepath);
    std::string filename = p.stem().string(); // "my_project" from "C:/.../my_project.rtp"
    if (g_project.project_name == "Untitled" || g_project.project_name.empty()) {
        g_project.project_name = filename;
    }
    // Also store the current path
    g_project.current_file_path = filepath;

    {
        ScopedPerfTimer timer("saveProject compactPendingDeletedObjects");
        const size_t compacted = scene.compactPendingDeletedObjects();
        if (compacted > 0) {
            SCENE_LOG_INFO("[ProjectManager] Compacted " + std::to_string(compacted) + " pending-deleted object(s) before save.");
        }
    }

    // Sync project data with current scene state
    {
        ScopedPerfTimer timer("saveProject syncProjectToScene");
        syncProjectToScene(scene);
    }

    // 1. Prepare Safe Save paths
    fs::path final_json_path(filepath);
    fs::path final_bin_path = final_json_path;
    final_bin_path += ".bin";
    
    fs::path temp_json_path = final_json_path;
    temp_json_path += ".tmp";
    fs::path temp_bin_path = final_bin_path;
    temp_bin_path += ".tmp";

    loadPreviousEmbeddedTextureEntries(final_json_path, final_bin_path);

    // 2. Open Streams
    std::ofstream out_json(temp_json_path);
    std::ofstream out_bin(temp_bin_path, std::ios::binary);

    if (!out_json.is_open() || !out_bin.is_open()) {
        SCENE_LOG_ERROR("Failed to create temporary save files.");
        return false;
    }

    try {
        std::unordered_map<std::string, uint64_t> bin_section_sizes;
        auto currentBinOffset = [&out_bin]() -> uint64_t {
            const std::streampos pos = out_bin.tellp();
            if (pos < 0) return 0;
            return static_cast<uint64_t>(pos);
        };
        auto measureBinarySection = [&](const std::string& name, auto&& fn) {
            const uint64_t begin = currentBinOffset();
            fn();
            const uint64_t end = currentBinOffset();
            bin_section_sizes[name] += (end >= begin) ? (end - begin) : 0;
        };

        if (progress_callback) progress_callback(5, "Writing geometry...");
        
        // Write geometry to binary file FIRST
        if (save_settings.save_geometry) {
            ScopedPerfTimer timer("saveProject writeGeometryBinary");
            measureBinarySection("geometry", [&]() {
                writeGeometryBinary(out_bin, scene);
            });
        }
        
        if (progress_callback) progress_callback(30, "Writing metadata...");

        // 3. Write JSON 
        json root;
        root["format_version"] = "3.0";
        root["project_name"] = g_project.project_name;
        root["author"] = g_project.author;
        root["description"] = g_project.description;
        root["next_model_id"] = g_project.next_model_id;
       root["next_object_id"] = g_project.next_object_id;
        root["next_texture_id"] = g_project.next_texture_id;
        root["has_geometry"] = save_settings.save_geometry;
        root["save_settings"] = {
            {"texture_storage_mode", static_cast<int>(save_settings.texture_storage_mode)},
            {"embed_missing_only", save_settings.embed_missing_only},
            {"save_geometry", save_settings.save_geometry}
        };

        json imported_models_arr = json::array();
        for (const auto& model : g_project.imported_models) {
            json m;
            m["id"] = model.id;
            m["original_path"] = model.original_path;
            m["package_path"] = model.package_path;
            m["display_name"] = model.display_name;
            m["deleted_objects"] = model.deleted_objects;

            json objects_arr = json::array();
            for (const auto& inst : model.objects) {
                json o;
                o["node_name"] = inst.node_name;
                o["transform"] = mat4ToJson(inst.transform);
                o["material_id"] = inst.material_id;
                o["visible"] = inst.visible;
                objects_arr.push_back(o);
            }
            m["objects"] = objects_arr;
            imported_models_arr.push_back(m);
        }
        root["imported_models"] = imported_models_arr;
        
        // Procedural objects (still saved for non-geometry mode)
        json procedurals_arr = json::array();
        for (const auto& proc : g_project.procedural_objects) {
            json p;
            p["id"] = proc.id;
            p["mesh_type"] = static_cast<int>(proc.mesh_type);
            p["display_name"] = proc.display_name;
            p["transform"] = mat4ToJson(proc.transform);
            p["material_id"] = proc.material_id;
            p["visible"] = proc.visible;
            procedurals_arr.push_back(p);
        }
        root["procedural_objects"] = procedurals_arr;
        
        if (progress_callback) progress_callback(50, "Saving materials...");
        
        // Materials
        auto& mat_mgr = MaterialManager::getInstance();
        {
            ScopedPerfTimer timer("saveProject serialize materials");
            root["materials"] = mat_mgr.serialize(fs::path(filepath).parent_path().string());
        }
        
        if (progress_callback) progress_callback(60, "Saving lights...");
        
        // Lights
        root["lights"] = serializeLights(scene.lights);
        
        if (progress_callback) progress_callback(70, "Saving cameras...");
        
        // Cameras
        root["cameras"] = serializeCameras(scene.cameras, scene.active_camera_index);
        
        if (progress_callback) progress_callback(80, "Saving render settings...");
        
        // Render Settings
        root["render_settings"] = serializeRenderSettings(settings);

        // Ensure UI/global color processor state is reflected into the scene before saving
        extern ColorProcessor color_processor; // defined in Main.cpp
        scene.color_processor.params = color_processor.params;

        // Post-Process / Color Grading settings (store from scene's color processor)
        const auto& pp = scene.color_processor.params;
        root["postfx"] = {
            {"exposure", pp.global_exposure},
            {"gamma", pp.global_gamma},
            {"saturation", pp.saturation},
            {"color_temperature", pp.color_temperature},
            {"tone_mapping", static_cast<int>(pp.tone_mapping_type)},
            {"vignette_enabled", pp.enable_vignette},
            {"vignette_strength", pp.vignette_strength}
        };

        // UI Layout (ImGui state)
        if (!g_project.ui_layout_data.empty()) {
            root["ui_layout"] = g_project.ui_layout_data; 
        }
        
        // Timeline/Animation (keyframes)
        if (progress_callback) progress_callback(82, "Saving animation keyframes...");
        json j_timeline;
        scene.timeline.serialize(j_timeline);
        root["timeline"] = j_timeline;
        
        // BoneData (skeleton/skinning information)
        if (progress_callback) progress_callback(83, "Saving bone data...");
        {
            json j_bones;
            
            // Save bone name to index mapping
            json j_bone_map = json::object();
            for (const auto& [name, idx] : scene.boneData.boneNameToIndex) {
                j_bone_map[name] = idx;
            }
            j_bones["boneNameToIndex"] = j_bone_map;
            
            // Save bone offset matrices
            json j_offsets = json::object();
            for (const auto& [name, matrix] : scene.boneData.boneOffsetMatrices) {
                j_offsets[name] = mat4ToJson(matrix);
            }
            j_bones["boneOffsetMatrices"] = j_offsets;
            
            // Save bone default local transforms (bind pose)
            json j_defaults = json::object();
            for (const auto& [name, matrix] : scene.boneData.boneDefaultTransforms) {
                j_defaults[name] = mat4ToJson(matrix);
            }
            j_bones["boneDefaultTransforms"] = j_defaults;

            json j_weighted = json::array();
            for (const auto& name : scene.boneData.weightedBoneNames) {
                j_weighted.push_back(name);
            }
            j_bones["weightedBoneNames"] = j_weighted;
            
            // Save bone hierarchy
            json j_parents = json::object();
            for (const auto& [child, parent] : scene.boneData.boneParents) {
                j_parents[child] = parent;
            }
            j_bones["boneParents"] = j_parents;
            
            // Save per-model inverses
            json j_model_invs = json::object();
            for (const auto& [prefix, inv] : scene.boneData.perModelInverses) {
                j_model_invs[prefix] = mat4ToJson(inv);
            }
            j_bones["perModelInverses"] = j_model_invs;
            
            // Save global inverse transform
            j_bones["globalInverseTransform"] = mat4ToJson(scene.boneData.globalInverseTransform);
            
            root["boneData"] = j_bones;
        }
        
        // AnimationData (bone animation keyframes)
        if (progress_callback) progress_callback(83, "Saving animation data...");
        {
            json j_animations = json::array();
            
            for (const auto& anim : scene.animationDataList) {
                if (!anim) continue;
                json j_anim;
                j_anim["name"] = anim->name;
                j_anim["modelName"] = anim->modelName;
                j_anim["duration"] = anim->duration;
                j_anim["ticksPerSecond"] = anim->ticksPerSecond;
                j_anim["startFrame"] = anim->startFrame;
                j_anim["endFrame"] = anim->endFrame;
                
                // Position keys
                json j_pos_keys = json::object();
                for (const auto& [nodeName, keys] : anim->positionKeys) {
                    json j_keys = json::array();
                    for (const auto& key : keys) {
                        j_keys.push_back({
                            {"time", key.mTime},
                            {"x", key.mValue.x},
                            {"y", key.mValue.y},
                            {"z", key.mValue.z}
                        });
                    }
                    j_pos_keys[nodeName] = j_keys;
                }
                j_anim["positionKeys"] = j_pos_keys;
                
                // Rotation keys
                json j_rot_keys = json::object();
                for (const auto& [nodeName, keys] : anim->rotationKeys) {
                    json j_keys = json::array();
                    for (const auto& key : keys) {
                        j_keys.push_back({
                            {"time", key.mTime},
                            {"w", key.mValue.w},
                            {"x", key.mValue.x},
                            {"y", key.mValue.y},
                            {"z", key.mValue.z}
                        });
                    }
                    j_rot_keys[nodeName] = j_keys;
                }
                j_anim["rotationKeys"] = j_rot_keys;
                
                // Scaling keys
                json j_scale_keys = json::object();
                for (const auto& [nodeName, keys] : anim->scalingKeys) {
                    json j_keys = json::array();
                    for (const auto& key : keys) {
                        j_keys.push_back({
                            {"time", key.mTime},
                            {"x", key.mValue.x},
                            {"y", key.mValue.y},
                            {"z", key.mValue.z}
                        });
                    }
                    j_scale_keys[nodeName] = j_keys;
                }
                j_anim["scalingKeys"] = j_scale_keys;
                
                j_animations.push_back(j_anim);
            }
            
           
            root["animationDataList"] = j_animations;
            SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(scene.animationDataList.size()) + " animation clips.");
        }
        
        // Animation Graphs (Visual Scripting)
        if (progress_callback) progress_callback(83, "Saving animation graphs...");
        {
            json j_graphs = json::object();
            for(const auto& [name, graph] : g_animGraphUI.graphs) {
                if(graph) {
                    json g;
                    graph->saveToJson(g);
                    j_graphs[name] = g;
                }
            }
            root["animationGraphs"] = j_graphs;
            root["activeGraphCharacter"] = g_animGraphUI.activeCharacter;
        }

        // World Settings (Atmosphere, Godrays, etc.)
        if (progress_callback) progress_callback(82, "Saving world settings...");
        json j_world;
        renderer.world.serialize(j_world);
        root["world"] = j_world;
        
        // Water System
        if (progress_callback) progress_callback(83, "Saving water surfaces...");
        root["water"] = WaterManager::getInstance().serialize();
        SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(WaterManager::getInstance().getWaterSurfaces().size()) + " water surfaces.");

        // Terrain System
        if (progress_callback) progress_callback(84, "Saving terrain system...");
        auto abs_path = std::filesystem::absolute(filepath);
        std::string terrainDir = abs_path.parent_path().string();
        SCENE_LOG_INFO("[ProjectManager] Saving terrain system to: " + terrainDir);
        root["terrain_system"] = TerrainManager::getInstance().serialize(terrainDir);
        
        // River System
        if (progress_callback) progress_callback(84, "Saving river system...");
        root["rivers"] = RiverManager::getInstance().serialize();
        SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(RiverManager::getInstance().getRivers().size()) + " rivers");
        
        // Mesh Modifiers System
        if (progress_callback) progress_callback(84, "Saving mesh modifiers...");
        root["mesh_modifiers"] = json::object();
        for (const auto& [nodeName, stack] : scene.mesh_modifiers) {
            if (!stack.modifiers.empty()) {
                json stackJson;
                stack.serialize(stackJson);
                root["mesh_modifiers"][nodeName] = stackJson;
            }
        }
        SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(scene.mesh_modifiers.size()) + " mesh modifiers.");

        // Paint Layer Stacks
        if (!scene.mesh_paint_layer_stacks.empty()) {
            if (progress_callback) progress_callback(84, "Saving paint layers...");
            root["mesh_paint_layers"] = json::object();
            measureBinarySection("paint_layers", [&]() {
                for (const auto& [key, stack] : scene.mesh_paint_layer_stacks) {
                    if (stack.empty()) continue;
                    json j_stack;
                    stack.serialize(j_stack, out_bin);
                    root["mesh_paint_layers"][key] = j_stack;
                }
            });
            SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(scene.mesh_paint_layer_stacks.size()) + " paint layer stacks.");
        }

        // Foliage System
        if (progress_callback) progress_callback(84, "Saving foliage system...");
        measureBinarySection("foliage", [&]() {
            root["instances"] = InstanceManager::getInstance().serialize(&out_bin);
        });
        SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(InstanceManager::getInstance().getGroups().size()) + " foliage groups.");
        
        // VDB Volumes
        if (progress_callback) progress_callback(84, "Saving VDB volumes...");
        root["vdb_volumes"] = serializeVDBVolumes(scene.vdb_volumes);
        SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(scene.vdb_volumes.size()) + " VDB volumes.");

        // Gas Volumes
        if (progress_callback) progress_callback(84, "Saving Gas volumes...");
        root["gas_volumes"] = serializeGasVolumes(scene.gas_volumes);
        SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(scene.gas_volumes.size()) + " Gas volumes.");

        // Force Fields
        if (progress_callback) progress_callback(84, "Saving Force Fields...");
        root["force_fields"] = serializeForceFields(scene.force_field_manager);
        SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(scene.force_field_manager.force_fields.size()) + " Force Fields.");
        
        // Hair System
        if (progress_callback) progress_callback(84, "Saving hair system...");
        // Use binary stream for faster saving
        {
            ScopedPerfTimer timer("saveProject serialize hair");
            measureBinarySection("hair", [&]() {
                root["hair"] = renderer.getHairSystem().serialize(&out_bin);
            });
        }
        root["hair_material"] = renderer.getHairMaterial();
        SCENE_LOG_INFO("[ProjectManager] Saved hair system with " + std::to_string(renderer.getHairSystem().getGroomCount()) + " grooms.");

        
        // Imported Model Context Metadata
        if (progress_callback) progress_callback(84, "Saving model contexts...");
        json j_contexts = json::array();
        for (const auto& ctx : scene.importedModelContexts) {
            json c;
            c["importName"] = ctx.importName;
            c["animGraphAssetKey"] = ctx.animGraphAssetKey;
            c["hasAnimation"] = ctx.hasAnimation;
            c["globalInverseTransform"] = mat4ToJson(ctx.globalInverseTransform);
            c["useRootMotion"] = ctx.useRootMotion;
            c["rootMotionBone"] = ctx.rootMotionBone;
            c["useAnimGraph"] = ctx.useAnimGraph;
            c["animGraphFollowTimeline"] = ctx.animGraphFollowTimeline;
            c["preferOzzRuntime"] = ctx.preferOzzRuntime;
            c["visible"] = ctx.visible;
            j_contexts.push_back(c);
        }
        root["importedModelContexts"] = j_contexts;

        // Textures (with embed option)
        if (progress_callback) progress_callback(85, "Processing textures...");
        {
            ScopedPerfTimer timer("saveProject serialize textures");
            const bool embed_textures = (save_settings.texture_storage_mode == ProjectManager::TextureStorageMode::Embedded);
            measureBinarySection("textures", [&]() {
                root["textures"] = serializeTextures(out_bin, embed_textures);
            });
        }
        
        // Write JSON - Use error handler to replace invalid UTF-8 characters
        // Write JSON - Use minified output (-1) for maximum performance and smallest size
        {
            ScopedPerfTimer timer("saveProject dump json");
            out_json << root.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        }
        
        {
            ScopedPerfTimer timer("saveProject flush streams");
            out_json.flush();
            out_bin.flush();
        }

        const uint64_t total_bin_bytes = currentBinOffset();
        SCENE_LOG_INFO("[ProjectManager] Binary section size report:");
        for (const auto& [name, bytes] : bin_section_sizes) {
            const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
            const double pct = total_bin_bytes > 0
                ? (100.0 * static_cast<double>(bytes) / static_cast<double>(total_bin_bytes))
                : 0.0;
            SCENE_LOG_INFO("  - " + name + ": " + std::to_string(bytes) + " bytes (" +
                           std::to_string(mb) + " MB, " + std::to_string(pct) + "%)");
        }
        SCENE_LOG_INFO("[ProjectManager] Binary total: " + std::to_string(total_bin_bytes) +
                       " bytes (" + std::to_string(static_cast<double>(total_bin_bytes) / (1024.0 * 1024.0)) + " MB)");

        out_json.close();
        out_bin.close();
        
        if (progress_callback) progress_callback(95, "Finalizing...");

        // Atomic Commit
        try {
            ScopedPerfTimer timer("saveProject atomic commit");
            if (fs::exists(final_json_path)) fs::remove(final_json_path);
            if (fs::exists(final_bin_path)) fs::remove(final_bin_path);
            fs::rename(temp_json_path, final_json_path);
            fs::rename(temp_bin_path, final_bin_path);
        } catch (const fs::filesystem_error& e) {
            SCENE_LOG_ERROR("FileSystem Error during rename: " + std::string(e.what()));
            return false;
        }
        
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("Save failed: " + std::string(e.what()));
        if (out_json.is_open()) out_json.close();
        if (out_bin.is_open()) out_bin.close();
        if (fs::exists(temp_json_path)) fs::remove(temp_json_path);
        if (fs::exists(temp_bin_path)) fs::remove(temp_bin_path);
        return false;
    }
    
    g_project.current_file_path = filepath;
    g_project.is_modified = false;
    
    if (progress_callback) progress_callback(100, "Done.");
    SCENE_LOG_INFO("Project saved successfully: " + filepath);
    return true;
}

bool ProjectManager::openProject(const std::string& filepath, SceneData& scene,
                                  RenderSettings& settings, Renderer& renderer, 
                                  Backend::IBackend* backend,
                                  std::function<void(int, const std::string&)> progress_callback) {
    ScopedPerfTimer total_timer("ProjectManager::openProject total");

    const bool prev_use_optix = settings.use_optix;
    const bool prev_use_vulkan = settings.use_vulkan;
    const ProjectBackendMode prev_backend_mode = modeFromBackendPtr(backend);
    
    if (progress_callback) progress_callback(0, "Opening project file with turbo parser...");
    
    // 1. Read JSON Metadata with simdjson
    simdjson::dom::parser parser;
    simdjson::dom::element root;

    std::string read_error;
    simdjson::error_code error;
    {
        ScopedPerfTimer timer("openProject parse json");
        error = loadJsonRootFromFile(parser, filepath, root, &read_error);
    }
    if (error) {
        SCENE_LOG_ERROR("Failed to parse project file with simdjson: " + std::string(simdjson::error_message(error)) +
                        " | path=" + filepath +
                        (read_error.empty() ? "" : " | read_error=" + read_error));
        return false;
    }

    // 2. Prepare Binary Stream
    fs::path bin_path = pathFromUtf8(filepath);
    bin_path += ".bin";
    std::ifstream in_bin(bin_path, std::ios::binary);
    bool has_binary = in_bin.is_open();
    
    // Clear current project and scene
    if (progress_callback) progress_callback(5, "Clearing scene...");
    
    if (backend) {
        backend->waitForCompletion();
    }

    const bool had_runtime_scene_content =
        !scene.world.objects.empty() ||
        !scene.gas_volumes.empty() ||
        !scene.vdb_volumes.empty() ||
        !scene.lights.empty();

    {
        ScopedPerfTimer timer("openProject reset scene");
        // When opening a project over an existing runtime scene, keep the
        // backend reset so old BLAS/textures are freed before the new load.
        // On first startup the scene is already empty; deferring avoids waking
        // the render backend during project parsing while the viewport is Solid.
        newProject(scene, renderer, !had_runtime_scene_content);
    }
    // newProject already called TerrainManager::removeAllTerrains and scene.clear();
    // don't repeat them here.
    
    auto temp_camera = std::make_shared<Camera>(
        Vec3(0, 2, 5), Vec3(0, 0, 0), Vec3(0, 1, 0),
        60.0f, 16.0f / 9.0f, 0.0f, 10.0f, 6);
    temp_camera->nodeName = "Loading...";
    scene.addCamera(temp_camera);
    
    renderer.resetCPUAccumulation();
    if (backend) backend->resetAccumulation();
    
    try {

        // Load metadata
        {
            std::string_view fv_sv = "2.0";
            simdjson::dom::element fv_el;
            if (!root["format_version"].get(fv_el)) {
                if (fv_el.is_number()) {
                    double fv_val = 2.0;
                    fv_el.get(fv_val);
                    g_project.format_version = std::to_string(fv_val).substr(0, 3);
                } else {
                    fv_el.get(fv_sv);
                    g_project.format_version = std::string(fv_sv);
                }
            } else {
                g_project.format_version = "2.0";
            }
        }
        
        {
            std::string_view name_sv = "Untitled", author_sv = "", desc_sv = "";
            root["project_name"].get(name_sv);
            root["author"].get(author_sv);
            root["description"].get(desc_sv);
            g_project.project_name = std::string(name_sv);
            g_project.author = std::string(author_sv);
            g_project.description = std::string(desc_sv);
        }

        if (auto save_settings_el = root["save_settings"]; !save_settings_el.error()) {
            simdjson::dom::object save_settings_obj;
            if (!save_settings_el.get(save_settings_obj)) {
                int64_t texture_storage_mode = static_cast<int>(ProjectManager::TextureStorageMode::Embedded);
                bool embed_missing_only = false;
                bool save_geometry = true;

                if (auto mode_el = save_settings_obj["texture_storage_mode"]; !mode_el.error()) {
                    mode_el.get(texture_storage_mode);
                }
                if (auto embed_missing_el = save_settings_obj["embed_missing_only"]; !embed_missing_el.error()) {
                    embed_missing_el.get(embed_missing_only);
                }
                if (auto save_geometry_el = save_settings_obj["save_geometry"]; !save_geometry_el.error()) {
                    save_geometry_el.get(save_geometry);
                }

                save_settings.texture_storage_mode = static_cast<ProjectManager::TextureStorageMode>(texture_storage_mode);
                save_settings.embed_missing_only = embed_missing_only;
                save_settings.save_geometry = save_geometry;
            }
        }

        int64_t next_model = 1, next_obj = 1, next_tex = 1;
        root["next_model_id"].get(next_model);
        root["next_object_id"].get(next_obj);
        root["next_texture_id"].get(next_tex);
        g_project.next_model_id = (uint32_t)next_model;
        g_project.next_object_id = (uint32_t)next_obj;
        g_project.next_texture_id = (uint32_t)next_tex;

        g_project.imported_models.clear();
        if (auto imported_models_el = root["imported_models"]; !imported_models_el.error()) {
            simdjson::dom::array imported_models_arr;
            if (!imported_models_el.get(imported_models_arr)) {
                for (auto j_model_el : imported_models_arr) {
                    auto j_model = sjsonToNlohmann(j_model_el);
                    ImportedModelData model;
                    model.id = j_model.value("id", 0u);
                    model.original_path = j_model.value("original_path", "");
                    model.package_path = j_model.value("package_path", "");
                    model.display_name = j_model.value("display_name", "");
                    model.deleted_objects = j_model.value("deleted_objects", std::vector<std::string>{});

                    if (j_model.contains("objects") && j_model["objects"].is_array()) {
                        for (const auto& j_obj : j_model["objects"]) {
                            ImportedModelData::ObjectInstance inst;
                            inst.node_name = j_obj.value("node_name", "");
                            if (j_obj.contains("transform")) {
                                inst.transform = jsonToMat4(j_obj["transform"]);
                            }
                            inst.material_id = j_obj.value("material_id", static_cast<uint16_t>(0));
                            inst.visible = j_obj.value("visible", true);
                            model.objects.push_back(inst);
                        }
                    }

                    g_project.imported_models.push_back(std::move(model));
                }
            }
        }
        
        fs::path project_folder = pathFromUtf8(filepath).parent_path();
        
        bool is_v3 = (g_project.format_version == "3.0");
        bool has_geometry = false;
        root["has_geometry"].get(has_geometry);
        
        if (is_v3 && has_geometry && has_binary) {
            if (progress_callback) progress_callback(10, "Loading geometry...");
            {
                ScopedPerfTimer timer("openProject readGeometryBinary");
                if (!readGeometryBinary(in_bin, scene)) {
                    SCENE_LOG_ERROR("Failed to read geometry from binary file.");
                    return false;
                }
            }

            // Textures
            simdjson::dom::element tex_el;
            if (!root["textures"].get(tex_el)) {
                if (progress_callback) progress_callback(35, "Restoring textures...");
                ScopedPerfTimer timer("openProject deserializeTextures");
                deserializeTextures(sjsonToNlohmann(tex_el), in_bin, project_folder.string());
            }

            // Materials
            simdjson::dom::element mat_el;
            if (!root["materials"].get(mat_el)) {
                if (progress_callback) progress_callback(40, "Loading materials...");
                ScopedPerfTimer timer("openProject deserialize materials");
                MaterialManager::getInstance().deserialize(sjsonToNlohmann(mat_el), project_folder.string());
            }

            // Lights
            simdjson::dom::element lights_el;
            if (!root["lights"].get(lights_el)) {
                if (progress_callback) progress_callback(50, "Loading lights...");
                deserializeLights(sjsonToNlohmann(lights_el), scene.lights);
            }

            // Cameras
            simdjson::dom::element cams_el;
            if (!root["cameras"].get(cams_el)) {
                if (progress_callback) progress_callback(60, "Loading cameras...");
                deserializeCameras(sjsonToNlohmann(cams_el), scene);
            } else {
                auto default_cam = std::make_shared<Camera>(
                    Vec3(0, 2, 5), Vec3(0, 0, 0), Vec3(0, 1, 0),
                    60.0f, 16.0f / 9.0f, 0.0f, 10.0f, 10);
                default_cam->nodeName = "Default Camera";
                scene.addCamera(default_cam);
            }

            // Render Settings
            simdjson::dom::element rs_el;
            if (!root["render_settings"].get(rs_el)) {
                if (progress_callback) progress_callback(70, "Loading render settings...");
                deserializeRenderSettings(sjsonToNlohmann(rs_el), settings);
            }
            // Post-Process / Color Grading
            simdjson::dom::element postfx_el;
            if (!root["postfx"].get(postfx_el)) {
                json pj = sjsonToNlohmann(postfx_el);
                auto& pp = scene.color_processor.params;
                pp.global_exposure = pj.value("exposure", pp.global_exposure);
                pp.global_gamma = pj.value("gamma", pp.global_gamma);
                pp.saturation = pj.value("saturation", pp.saturation);
                pp.color_temperature = pj.value("color_temperature", pp.color_temperature);
                pp.tone_mapping_type = static_cast<ToneMappingType>(pj.value("tone_mapping", static_cast<int>(pp.tone_mapping_type)));
                pp.enable_vignette = pj.value("vignette_enabled", pp.enable_vignette);
                pp.vignette_strength = pj.value("vignette_strength", pp.vignette_strength);
                // Also copy into global UI color processor so UI reflects loaded values
                extern ColorProcessor color_processor; // defined in Main.cpp
                color_processor.params = pp;
            }
            
            // UI Layout
            std::string_view ui_layout;
            if (!root["ui_layout"].get(ui_layout)) {
                g_project.ui_layout_data = std::string(ui_layout);
            }

            // World Settings
            simdjson::dom::element world_el;
            if (!root["world"].get(world_el)) {
                ScopedPerfTimer timer("openProject deserialize world");
                renderer.world.deserialize(sjsonToNlohmann(world_el));
            }

            // Water Surfaces
            simdjson::dom::element water_el;
            if (!root["water"].get(water_el)) {
                WaterManager::getInstance().deserialize(sjsonToNlohmann(water_el), scene);
            }

            // Terrain System
            simdjson::dom::element ts_el;
            if (!root["terrain_system"].get(ts_el)) {
                if (progress_callback) progress_callback(75, "Loading terrain system...");
                auto abs_path = std::filesystem::absolute(filepath);
                std::string terrainDir = abs_path.parent_path().string();
                ScopedPerfTimer timer("openProject deserialize terrain");
                TerrainManager::getInstance().deserialize(sjsonToNlohmann(ts_el), terrainDir, scene);
            }

            // River System
            simdjson::dom::element rivers_el;
            if (!root["rivers"].get(rivers_el)) {
                if (progress_callback) progress_callback(76, "Loading river system...");
                RiverManager::getInstance().clear(&scene);
                RiverManager::getInstance().deserialize(sjsonToNlohmann(rivers_el), scene);
            }

            // Foliage System
            simdjson::dom::element inst_el;
            if (!root["instances"].get(inst_el)) {
                if (progress_callback) progress_callback(77, "Loading foliage system...");
                ScopedPerfTimer timer("openProject deserialize foliage");
                InstanceManager::getInstance().deserializeFast(inst_el, scene);
                if (has_binary) {
                    InstanceManager::getInstance().deserializeBinaryInstances(inst_el, in_bin);
                }
            }

            if (InstanceManager::getInstance().getGroups().empty() &&
                TerrainManager::getInstance().hasLegacyFoliage()) {
                int migrated_legacy_groups = TerrainManager::getInstance().migrateLegacyFoliageToInstanceGroups(scene, true);
                if (migrated_legacy_groups > 0 && progress_callback) {
                    progress_callback(77, "Migrating legacy foliage...");
                }
            }

            if (!InstanceManager::getInstance().getGroups().empty()) {
                if (progress_callback) progress_callback(77, "Building foliage BVH & instances...");
                InstanceManager::getInstance().rebuildSceneObjects(scene);
            }

            // Mesh Modifiers System
            scene.mesh_modifiers.clear();
            scene.base_mesh_cache.clear();
            simdjson::dom::element modRoot;
            if (!root["mesh_modifiers"].get(modRoot)) {
                if (progress_callback) progress_callback(78, "Loading mesh modifiers...");
                nlohmann::json nlohmannMods = sjsonToNlohmann(modRoot);
                for (auto it = nlohmannMods.begin(); it != nlohmannMods.end(); ++it) {
                    std::string nodeName = it.key();
                    MeshModifiers::ModifierStack stack;
                    stack.deserialize(it.value());
                    scene.mesh_modifiers[nodeName] = stack;
                }

                // We must lazily build base_mesh_cache for these modifiers, evaluate them and set new objects
                // NOTE: Foliage instances (tail of objects vector) are never mesh-modified,
                // so we can limit the scan to non-foliage objects for efficiency.
                size_t mod_foliage_count = InstanceManager::getInstance().getTotalInstanceCount();
                size_t mod_selectable = (mod_foliage_count <= scene.world.objects.size())
                                            ? (scene.world.objects.size() - mod_foliage_count)
                                            : scene.world.objects.size();

                for(const auto& [nodeName, stack] : scene.mesh_modifiers) {
                    std::vector<std::shared_ptr<Triangle>> baseTriangles;
                    std::vector<std::shared_ptr<Hittable>> remainingObjects;
                    remainingObjects.reserve(scene.world.objects.size());

                    // Scan only non-foliage objects for modifier matches
                    for (size_t mi = 0; mi < mod_selectable; ++mi) {
                        const auto& obj = scene.world.objects[mi];
                        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                            if (tri->getNodeName() == nodeName) {
                                baseTriangles.push_back(tri);
                            } else {
                                remainingObjects.push_back(obj);
                            }
                        } else {
                            remainingObjects.push_back(obj);
                        }
                    }
                    // Append foliage tail untouched
                    for (size_t mi = mod_selectable; mi < scene.world.objects.size(); ++mi) {
                        remainingObjects.push_back(scene.world.objects[mi]);
                    }

                    if (!baseTriangles.empty() && !stack.modifiers.empty()) {
                        scene.base_mesh_cache[nodeName] = baseTriangles;
                        auto newMesh = stack.evaluate(baseTriangles);
                        // Insert new mesh BEFORE foliage tail
                        size_t insert_pos = remainingObjects.size() - mod_foliage_count;
                        for (const auto& tri : newMesh) {
                            remainingObjects.insert(remainingObjects.begin() + insert_pos, tri);
                            ++insert_pos;
                            ++mod_selectable; // Track new non-foliage count
                        }
                        scene.world.objects = remainingObjects;
                    }
                }
            }

            // Paint Layer Stacks
            simdjson::dom::element paint_layers_el;
            if (!root["mesh_paint_layers"].get(paint_layers_el)) {
                if (progress_callback) progress_callback(79, "Loading paint layers...");
                nlohmann::json j_paint = sjsonToNlohmann(paint_layers_el);
                for (auto it = j_paint.begin(); it != j_paint.end(); ++it) {
                    Paint::PaintLayerStack stack;
                    stack.deserialize(it.value(), in_bin);
                    scene.mesh_paint_layer_stacks[it.key()] = std::move(stack);
                }
                SCENE_LOG_INFO("[ProjectManager] Loaded " + std::to_string(scene.mesh_paint_layer_stacks.size()) + " paint layer stacks.");
            }

            // VDB / Gas / Force Fields
            simdjson::dom::element vdb_el, gas_el, ff_el;
            if (!root["vdb_volumes"].get(vdb_el)) deserializeVDBVolumes(sjsonToNlohmann(vdb_el), scene);
            if (!root["gas_volumes"].get(gas_el)) deserializeGasVolumes(sjsonToNlohmann(gas_el), scene);
            if (!root["force_fields"].get(ff_el)) deserializeForceFields(sjsonToNlohmann(ff_el), scene);

            // Hair System
            // newProject already cleared HairSystem + uploaded empty hair to GPU.
            // Skip the redundant second clear/upload here; only load + upload when
            // the project actually has hair data. For projects without hair the
            // newProject clear is sufficient.
            simdjson::dom::element hair_el;
            if (!root["hair"].get(hair_el)) {
                if (progress_callback) progress_callback(78, "Loading hair system...");
                renderer.getHairSystem().clearAll();
                renderer.getHairSystem().deserialize(sjsonToNlohmann(hair_el), &in_bin);

                simdjson::dom::element hm_el;
                if (!root["hair_material"].get(hm_el)) {
                    auto j_hm = sjsonToNlohmann(hm_el);
                    renderer.setHairMaterial(j_hm.get<Hair::HairMaterialParams>());
                }
                renderer.getHairSystem().buildBVH();
                renderer.uploadHairToGPU();
            }

            // Timeline
            simdjson::dom::element timeline_el;
            if (!root["timeline"].get(timeline_el)) {
                scene.timeline.deserialize(sjsonToNlohmann(timeline_el));
            }

            // Animation Graphs
            simdjson::dom::element anim_graphs_el;
            if (!root["animationGraphs"].get(anim_graphs_el)) {
                g_animGraphUI = AnimGraphUIState();
                auto j_graphs = sjsonToNlohmann(anim_graphs_el);
                for (auto& [name, j_graph] : j_graphs.items()) {
                    auto graph = std::make_unique<AnimationGraph::AnimationNodeGraph>();
                    graph->loadFromJson(j_graph);
                    g_animGraphUI.graphs[name] = std::move(graph);
                }
                std::string_view active_graph;
                if (!root["activeGraphCharacter"].get(active_graph)) {
                    g_animGraphUI.activeCharacter = std::string(active_graph);
                }
            }

            // Bone Data
            simdjson::dom::element bone_el;
            if (!root["boneData"].get(bone_el)) {
                scene.boneData.clear();
                auto j_bones = sjsonToNlohmann(bone_el);
                for (auto& [name, idx] : j_bones["boneNameToIndex"].items()) {
                    scene.boneData.boneNameToIndex[name] = idx.get<int>();
                }
                if (j_bones.contains("boneOffsetMatrices")) {
                    for (auto& [name, mat] : j_bones["boneOffsetMatrices"].items()) {
                        scene.boneData.boneOffsetMatrices[name] = jsonToMat4(mat);
                    }
                }
                if (j_bones.contains("boneDefaultTransforms")) {
                    for (auto& [name, mat] : j_bones["boneDefaultTransforms"].items()) {
                        scene.boneData.boneDefaultTransforms[name] = jsonToMat4(mat);
                    }
                }
                if (j_bones.contains("weightedBoneNames")) {
                    for (const auto& name : j_bones["weightedBoneNames"]) {
                        scene.boneData.weightedBoneNames.insert(name.get<std::string>());
                    }
                }
                if (j_bones.contains("boneParents")) {
                    for (auto& [child, parent] : j_bones["boneParents"].items()) {
                        scene.boneData.boneParents[child] = parent.get<std::string>();
                    }
                }
                if (j_bones.contains("perModelInverses")) {
                    for (auto& [prefix, mat] : j_bones["perModelInverses"].items()) {
                        scene.boneData.perModelInverses[prefix] = jsonToMat4(mat);
                    }
                }
                if (j_bones.contains("globalInverseTransform")) {
                    scene.boneData.globalInverseTransform = jsonToMat4(j_bones["globalInverseTransform"]);
                }
                scene.boneData.rebuildReverseLookup();
            }

            // UI Settings
            simdjson::dom::element ui_set_el;
            if (!root["ui_settings"].get(ui_set_el)) {
                scene.ui_settings_json_str = std::string(simdjson::minify(ui_set_el));
                scene.load_counter++;
            }

            // Imported Models Contexts
            simdjson::dom::array imc_arr;
            if (!root["importedModelContexts"].get(imc_arr)) {
                scene.importedModelContexts.clear();
                for (auto j_ctx_el : imc_arr) {
                    auto j_ctx = sjsonToNlohmann(j_ctx_el);
                    SceneData::ImportedModelContext ctx;
                    ctx.importName = j_ctx.value("importName", "");
                    ctx.animGraphAssetKey = j_ctx.value("animGraphAssetKey", ctx.importName);
                    ctx.hasAnimation = j_ctx.value("hasAnimation", false);
                    if (j_ctx.contains("globalInverseTransform")) {
                        ctx.globalInverseTransform = jsonToMat4(j_ctx["globalInverseTransform"]);
                    } else {
                        ctx.globalInverseTransform = Matrix4x4::identity();
                    }
                    ctx.useRootMotion = j_ctx.value("useRootMotion", false);
                    ctx.rootMotionBone = j_ctx.value("rootMotionBone", "");
                    ctx.useAnimGraph = j_ctx.value("useAnimGraph", false);
                    ctx.animGraphFollowTimeline = j_ctx.value("animGraphFollowTimeline", false);
                    ctx.preferOzzRuntime = j_ctx.value("preferOzzRuntime", true);
                    ctx.loggedOzzRuntimeUsage = false;
                    ctx.visible = j_ctx.value("visible", true);
                    if (!ctx.animGraphAssetKey.empty()) {
                        auto itGraph = g_animGraphUI.graphs.find(ctx.animGraphAssetKey);
                        if (itGraph != g_animGraphUI.graphs.end() && itGraph->second) {
                            ctx.runtimeGraph = itGraph->second->clone();
                            ctx.graph = ctx.runtimeGraph;
                        }
                    }
                    scene.importedModelContexts.push_back(ctx);
                }
            }

            // Animation Data List
            simdjson::dom::array adl_arr;
            if (!root["animationDataList"].get(adl_arr)) {
                scene.animationDataList.clear();
                for (auto j_anim_el : adl_arr) {
                    auto j_anim = sjsonToNlohmann(j_anim_el);
                    AnimationData anim;
                    anim.name = j_anim.value("name", "");
                    anim.modelName = j_anim.value("modelName", "");
                    anim.duration = j_anim.value("duration", 0.0);
                    anim.ticksPerSecond = j_anim.value("ticksPerSecond", 24.0);
                    anim.startFrame = j_anim.value("startFrame", 0);
                    anim.endFrame = j_anim.value("endFrame", 0);
                    
                    if (j_anim.contains("positionKeys")) {
                         for (auto& [nodeName, j_keys] : j_anim["positionKeys"].items()) {
                             std::vector<aiVectorKey> keys;
                             for (const auto& k : j_keys) {
                                 aiVectorKey key;
                                 key.mTime = k.value("time", 0.0);
                                 key.mValue.x = k.value("x", 0.0f);
                                 key.mValue.y = k.value("y", 0.0f);
                                 key.mValue.z = k.value("z", 0.0f);
                                 keys.push_back(key);
                             }
                             anim.positionKeys[nodeName] = keys;
                         }
                    }
                    if (j_anim.contains("rotationKeys")) {
                         for (auto& [nodeName, j_keys] : j_anim["rotationKeys"].items()) {
                             std::vector<aiQuatKey> keys;
                             for (const auto& k : j_keys) {
                                 aiQuatKey key;
                                 key.mTime = k.value("time", 0.0);
                                 key.mValue.w = k.value("w", 1.0f);
                                 key.mValue.x = k.value("x", 0.0f);
                                 key.mValue.y = k.value("y", 0.0f);
                                 key.mValue.z = k.value("z", 0.0f);
                                 keys.push_back(key);
                             }
                             anim.rotationKeys[nodeName] = keys;
                         }
                    }
                    if (j_anim.contains("scalingKeys")) {
                         for (auto& [nodeName, j_keys] : j_anim["scalingKeys"].items()) {
                             std::vector<aiVectorKey> keys;
                             for (const auto& k : j_keys) {
                                 aiVectorKey key;
                                 key.mTime = k.value("time", 0.0);
                                 key.mValue.x = k.value("x", 1.0f);
                                 key.mValue.y = k.value("y", 1.0f);
                                 key.mValue.z = k.value("z", 1.0f);
                                 keys.push_back(key);
                             }
                             anim.scalingKeys[nodeName] = keys;
                         }
                    }
                    
                    scene.animationDataList.push_back(std::make_shared<AnimationData>(anim));
                }
            }

            clearEmbeddedTextureCache();
        } else {
            SCENE_LOG_ERROR("Project file is in legacy format. Please re-import your models.");
            return false;
        }

    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("Error during project loading: " + std::string(e.what()));
        return false;
    }
    
    if (in_bin.is_open()) in_bin.close();

    const ProjectBackendMode requested_backend_mode = modeFromSettings(settings);
    const bool backend_mode_changed =
        (settings.use_optix != prev_use_optix) ||
        (settings.use_vulkan != prev_use_vulkan) ||
        (requested_backend_mode != prev_backend_mode);
    if (backend_mode_changed) {
        settings.backend_changed = true;
        SCENE_LOG_INFO(std::string("[ProjectLoad] Backend switch scheduled: ") +
            modeToString(prev_backend_mode) + " -> " + modeToString(requested_backend_mode));
    }
    
    g_project.current_file_path = filepath;
    g_project.is_modified = false;
    
    if (progress_callback) progress_callback(88, "Rebuilding model contexts...");

    scene.initialized = true;
    
    // Foliage instances are always at the tail of scene.world.objects
    // (appended by InstanceManager::rebuildSceneObjects). Skip them to
    // avoid O(models × 2M) RTTI casts + string comparisons.
    size_t foliage_count = InstanceManager::getInstance().getTotalInstanceCount();
    size_t selectable_count = (foliage_count <= scene.world.objects.size())
                                  ? (scene.world.objects.size() - foliage_count)
                                  : scene.world.objects.size();

    // Rebuild members
    for (auto& ctx : scene.importedModelContexts) {
        if (ctx.importName.empty()) continue;
        ctx.members.clear();
        std::string prefix = ctx.importName + "_";
        for (size_t i = 0; i < selectable_count; ++i) {
            auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[i]);
            if (tri && tri->nodeName.find(prefix) == 0) {
                ctx.members.push_back(tri);
            }
        }
        ctx.rebuildSkeletonRepresentation(scene.boneData);
    }

    if (!scene.animationDataList.empty() && !scene.boneData.boneNameToIndex.empty()) {
        renderer.initializeAnimationSystem(scene);
        renderer.updateAnimationWithGraph(scene, 0.0f, true);
    }

    // IMPORTANT: Do not push GPU data from this loader thread.
    // Backend can be switched/destroyed concurrently (e.g. project requests different engine),
    // causing null/dangling backend access. Main thread performs deferred full GPU sync.
    
    g_needs_geometry_rebuild.store(true);
    // IMPORTANT: Must be true so main-thread scene-load finalization performs full backend sync.
    // Otherwise GPU backend may keep stale scene state until a manual backend toggle occurs.
    g_needs_optix_sync.store(true);
    g_camera_dirty = true;
    g_lights_dirty = true;
    g_world_dirty = true;
    g_geometry_dirty = true;
    g_materials_dirty = true;
    g_gas_volumes_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    
    if (scene.camera) {
        scene.camera->update_camera_vectors();
        if (backend) renderer.syncCameraToBackend(*scene.camera);
    }
    
    if (progress_callback) progress_callback(91, "Project data loaded, finalizing...");
    SCENE_LOG_INFO("Project loaded with turbo parser: " + filepath);
    return true;
}

// ============================================================================
// Asset Import
// ============================================================================

bool ProjectManager::importModel(const std::string& filepath, SceneData& scene,
    Renderer& renderer, Backend::IBackend* backend,
    std::function<void(int, const std::string&)> progress_callback,
    bool rebuild) {
    if (!fs::exists(filepath)) {
        SCENE_LOG_ERROR("File not found: " + filepath);
        return false;
    }
    
    if (progress_callback) progress_callback(0, "Preparing import...");

    // Generate package path and unique ID
    uint32_t id = g_project.generateModelId();
    std::string package_path = generatePackagePath(filepath, "models", id);
    std::string import_prefix = std::to_string(id);
    
    ImportedModelData model;
    model.id = id;
    model.original_path = filepath;
    model.package_path = package_path;
    model.display_name = fs::path(filepath).stem().string();
    
    // NOTE: In v3.0+ format, geometry is embedded in the binary file (.bin)
    // so we NO LONGER copy the original model file to the project folder.
    // This saves disk space and avoids confusion.
    // The original_path is kept for reference only (e.g., to show source).

    
    size_t objects_before = scene.world.objects.size();
    
    if (progress_callback) progress_callback(20, "Parsing model geometry...");

    // Load with Assimp - append mode (don't clear existing scene)
    // AssimpLoader usually just takes time, ideally we'd pass progress callback into it too
    try {
        renderer.create_scene(scene, backend, filepath, progress_callback, true, import_prefix);  // append = true, import_prefix = unique id
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("Failed to load model '" + filepath + "': " + std::string(e.what()));
        return false;
    } catch (...) {
        SCENE_LOG_ERROR("Unknown error while loading model: " + filepath);
        return false;
    }
    
    if (progress_callback) progress_callback(80, "Processing objects...");

    // Track new objects
    for (size_t i = objects_before; i < scene.world.objects.size(); ++i) {
        auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[i]);
        if (tri) {
            std::string unique_name = tri->nodeName; // Already prefixed by AssimpLoader now!
            
            ImportedModelData::ObjectInstance inst;
            inst.node_name = unique_name;
            auto th = tri->getTransformPtr();
            if (th) {
                inst.transform = th->base;
            }
            inst.material_id = tri->getMaterialID();
            inst.visible = true;
            model.objects.push_back(inst);
        }
    }
    
    g_project.imported_models.push_back(model);
    g_project.is_modified = true;
    
    // AUTO-ACTIVATE CAMERA: If imported model has cameras, set first new one as active
    if (!scene.cameras.empty()) {
        // Find if we have new cameras (cameras added during this import)
        // Simple approach: if active camera is default and we have imported cameras, activate first imported
        // Better: check if camera count increased
        size_t camera_count = scene.cameras.size();
        if (camera_count > 1 && scene.active_camera_index == 0) {
            // Switch to the newly imported camera (likely the last one)
            scene.setActiveCamera(camera_count - 1);
            SCENE_LOG_INFO("Auto-activated imported camera: Camera #" + std::to_string(camera_count - 1));
        } else if (camera_count == 1 && scene.camera) {
            // Only one camera exists, ensure it's active
            scene.setActiveCamera(0);
        }
    }
    
    if (rebuild) {
        if (progress_callback) progress_callback(90, "Rebuilding BVH...");
        extern RenderSettings render_settings;  // From globals or Main.cpp
        renderer.rebuildBVH(scene, render_settings.UI_use_embree);
        renderer.resetCPUAccumulation();
        
        if (backend) {
            if (progress_callback) progress_callback(95, "Uploading to GPU...");
            renderer.rebuildBackendGeometry(scene);
            backend->setLights(scene.lights);
            if (scene.camera) {
                renderer.syncCameraToBackend(*scene.camera);
            }
            backend->resetAccumulation();
        }
    }
    
    if (progress_callback) progress_callback(100, "Done.");
    SCENE_LOG_INFO("Model imported: " + model.display_name + " (ID: " + std::to_string(id) + ", " + std::to_string(model.objects.size()) + " objects)");
    return true;
}

// ... Rest of the file (procedural, textures, etc.) ...
// Just keeping the rest as is, but ensuring the file is complete.
// Since I'm using write_to_file, I must include EVERYTHING.

bool ProjectManager::importTexture(const std::string& filepath) {
    if (!fs::exists(filepath)) {
        SCENE_LOG_ERROR("Texture file not found: " + filepath);
        return false;
    }
    
    uint32_t id = g_project.generateTextureId();
    std::string package_path = generatePackagePath(filepath, "textures", id);
    
    TextureAssetData tex;
    tex.id = id;
    tex.original_path = filepath;
    tex.package_path = package_path;
    tex.usage = "unknown";
    
    // Copy to project folder if exists
    if (!g_project.current_file_path.empty()) {
        fs::path project_folder = pathFromUtf8(g_project.current_file_path).parent_path();
        fs::path dest_path = project_folder / package_path;
        
        try {
            fs::create_directories(dest_path.parent_path());
            fs::copy_file(filepath, dest_path, fs::copy_options::overwrite_existing);
        } catch (const std::exception& e) {
            SCENE_LOG_WARN("Could not copy texture file: " + std::string(e.what()));
        }
    }
    
    g_project.texture_assets.push_back(tex);
    g_project.is_modified = true;
    
    SCENE_LOG_INFO("Texture imported: " + fs::path(filepath).filename().string());
    return true;
}

uint32_t ProjectManager::addProceduralObject(ProceduralMeshType type, const std::string& name,
                                              const Matrix4x4& transform, SceneData& scene,
                                              Renderer& renderer, Backend::IBackend* backend) {
    uint32_t id = g_project.generateObjectId();
    
    ProceduralObjectData proc;
    proc.id = id;
    proc.mesh_type = type;
    proc.display_name = name;
    proc.transform = transform;
    proc.material_id = 0;
    proc.visible = true;
    
    g_project.procedural_objects.push_back(proc);
    g_project.is_modified = true;
    
    return id;
}

bool ProjectManager::removeProceduralObject(uint32_t id, SceneData& scene) {
    auto it = std::find_if(g_project.procedural_objects.begin(), 
                           g_project.procedural_objects.end(),
                           [id](const ProceduralObjectData& p) { return p.id == id; });
    
    if (it == g_project.procedural_objects.end()) {
        return false;
    }
    
    g_project.procedural_objects.erase(it);
    g_project.is_modified = true;
    
    return true;
}

std::vector<uint8_t> ProjectManager::extractFileFromPackage(const std::string& internal_path) {
    if (!g_project.current_file_path.empty()) {
        fs::path project_folder = pathFromUtf8(g_project.current_file_path).parent_path();
        fs::path file_path = project_folder / internal_path;
        
        if (fs::exists(file_path)) {
            std::ifstream in(file_path, std::ios::binary);
            if (in) {
                return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
            }
        }
    }
    return {};
}

bool ProjectManager::fileExistsInPackage(const std::string& internal_path) {
    if (!g_project.current_file_path.empty()) {
        fs::path project_folder = pathFromUtf8(g_project.current_file_path).parent_path();
        return fs::exists(project_folder / internal_path);
    }
    return false;
}

bool ProjectManager::writeZipPackage(const std::string& filepath) {
    SCENE_LOG_WARN("ZIP packaging not yet implemented. Save using regular project format.");
    return false; // Not implemented
}

bool ProjectManager::readZipPackage(const std::string& filepath) {
    SCENE_LOG_WARN("ZIP packaging not yet implemented.");
    return false;
}

// ============================================================================
// GEOMETRY SERIALIZATION (Self-Contained Project)
// ============================================================================

// Binary format magic number and version
// v7 adds a shared node-name table so triangle records no longer duplicate strings.
static constexpr char RTP_MAGIC[4] = {'R', 'T', 'P', '7'};
static constexpr uint32_t RTP_VERSION = 7;

bool ProjectManager::writeGeometryBinary(std::ofstream& out, const SceneData& scene) {
    // Write header
    out.write(RTP_MAGIC, 4);
    uint32_t version = RTP_VERSION;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Collect all triangles
    std::vector<std::shared_ptr<Triangle>> triangles;
    std::unordered_map<std::shared_ptr<Transform>, uint32_t> transform_map;
    std::vector<std::shared_ptr<Transform>> transforms;
    std::unordered_map<std::string, uint32_t> node_name_map;
    std::vector<std::string> node_names;
    
    // FILTERING: Identify terrain triangles to exclude from binary geometry
    std::unordered_set<std::shared_ptr<Triangle>> terrain_triangles;
    auto& terrains = TerrainManager::getInstance().getTerrains();
    for (auto& t : terrains) {
        for (auto& tri : t.mesh_triangles) {
            terrain_triangles.insert(tri);
        }
    }

    // Track processed nodeNames so we don't save the base mesh multiple times if skipping modified instances
    std::unordered_set<std::string> nodes_processed;

    for (const auto& obj : scene.world.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            // Skip terrain chunks
            if (terrain_triangles.count(tri)) continue;
            
            // Skip foliage instances (they are serialized via InstanceManager)
            if (tri->nodeName.find("_inst_") != std::string::npos) continue;

            const std::string& nodeName = tri->nodeName;

            // Check if we have a base mesh cache for this node
            if (scene.base_mesh_cache.find(nodeName) != scene.base_mesh_cache.end()) {
                if (nodes_processed.find(nodeName) == nodes_processed.end()) {
                    nodes_processed.insert(nodeName);
                    for (const auto& baseTri : scene.base_mesh_cache.at(nodeName)) {
                        triangles.push_back(baseTri);
                        if (node_name_map.find(baseTri->nodeName) == node_name_map.end()) {
                            node_name_map[baseTri->nodeName] = static_cast<uint32_t>(node_names.size());
                            node_names.push_back(baseTri->nodeName);
                        }
                        auto th = baseTri->getTransformHandle();
                        if (th && transform_map.find(th) == transform_map.end()) {
                            transform_map[th] = static_cast<uint32_t>(transforms.size());
                            transforms.push_back(th);
                        }
                    }
                }
            } else {
                triangles.push_back(tri);
                if (node_name_map.find(tri->nodeName) == node_name_map.end()) {
                    node_name_map[tri->nodeName] = static_cast<uint32_t>(node_names.size());
                    node_names.push_back(tri->nodeName);
                }
                
                // Track unique transforms
                auto th = tri->getTransformHandle();
                if (th && transform_map.find(th) == transform_map.end()) {
                    transform_map[th] = static_cast<uint32_t>(transforms.size());
                    transforms.push_back(th);
                }
            }
        }
    }
    
    // Write transform count and data
    uint32_t transform_count = static_cast<uint32_t>(transforms.size());
    out.write(reinterpret_cast<const char*>(&transform_count), sizeof(transform_count));
    
    for (const auto& tr : transforms) {
        out.write(reinterpret_cast<const char*>(&tr->base), sizeof(Matrix4x4));
        // v5: Write components
        out.write(reinterpret_cast<const char*>(&tr->position), sizeof(Vec3));
        out.write(reinterpret_cast<const char*>(&tr->rotation), sizeof(Vec3));
        out.write(reinterpret_cast<const char*>(&tr->scale), sizeof(Vec3));
    }

    // Write shared node-name table (v7+)
    const uint32_t node_name_count = static_cast<uint32_t>(node_names.size());
    out.write(reinterpret_cast<const char*>(&node_name_count), sizeof(node_name_count));
    for (const auto& node_name : node_names) {
        writeStringBinary(out, node_name);
    }
    
    // Write triangle count
    uint32_t tri_count = static_cast<uint32_t>(triangles.size());
    out.write(reinterpret_cast<const char*>(&tri_count), sizeof(tri_count));
    
    // Write each triangle
    for (const auto& tri : triangles) {
        // Vertices (original, not transformed)
        Vec3 v0 = tri->getOriginalVertexPosition(0);
        Vec3 v1 = tri->getOriginalVertexPosition(1);
        Vec3 v2 = tri->getOriginalVertexPosition(2);
        out.write(reinterpret_cast<const char*>(&v0), sizeof(Vec3));
        out.write(reinterpret_cast<const char*>(&v1), sizeof(Vec3));
        out.write(reinterpret_cast<const char*>(&v2), sizeof(Vec3));
        
        // Normals
        Vec3 n0 = tri->getOriginalVertexNormal(0);
        Vec3 n1 = tri->getOriginalVertexNormal(1);
        Vec3 n2 = tri->getOriginalVertexNormal(2);
        out.write(reinterpret_cast<const char*>(&n0), sizeof(Vec3));
        out.write(reinterpret_cast<const char*>(&n1), sizeof(Vec3));
        out.write(reinterpret_cast<const char*>(&n2), sizeof(Vec3));
        
        // Active UVs
        Vec2 uv0 = tri->t0;
        Vec2 uv1 = tri->t1;
        Vec2 uv2 = tri->t2;
        out.write(reinterpret_cast<const char*>(&uv0), sizeof(Vec2));
        out.write(reinterpret_cast<const char*>(&uv1), sizeof(Vec2));
        out.write(reinterpret_cast<const char*>(&uv2), sizeof(Vec2));

        // All UV sets (v6+)
        uint32_t uv_set_count = static_cast<uint32_t>(tri->getUVSetCount());
        out.write(reinterpret_cast<const char*>(&uv_set_count), sizeof(uv_set_count));
        for (uint32_t uv_set = 0; uv_set < uv_set_count; ++uv_set) {
            const auto [set_uv0, set_uv1, set_uv2] = tri->getUVSetCoordinates(uv_set);
            out.write(reinterpret_cast<const char*>(&set_uv0), sizeof(Vec2));
            out.write(reinterpret_cast<const char*>(&set_uv1), sizeof(Vec2));
            out.write(reinterpret_cast<const char*>(&set_uv2), sizeof(Vec2));
        }
        
        // Material ID
        uint16_t mat_id = tri->getMaterialID();
        out.write(reinterpret_cast<const char*>(&mat_id), sizeof(mat_id));
        
        // Transform index
        uint32_t tr_idx = 0xFFFFFFFF; // No transform
        auto th = tri->getTransformHandle();
        if (th) {
            auto it = transform_map.find(th);
            if (it != transform_map.end()) {
                tr_idx = it->second;
            }
        }
        out.write(reinterpret_cast<const char*>(&tr_idx), sizeof(tr_idx));

        // Node name index (v7+)
        uint32_t node_name_idx = 0xFFFFFFFF;
        auto node_it = node_name_map.find(tri->nodeName);
        if (node_it != node_name_map.end()) {
            node_name_idx = node_it->second;
        }
        out.write(reinterpret_cast<const char*>(&node_name_idx), sizeof(node_name_idx));
        
        // Skinning data (v4+)
        uint8_t has_skin = tri->hasSkinData() ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&has_skin), sizeof(has_skin));
        
        if (has_skin) {
            // Write bone weights for each vertex (3 vertices)
            for (int vi = 0; vi < 3; ++vi) {
                const auto& weights = tri->getSkinBoneWeights(vi);
                uint8_t weight_count = static_cast<uint8_t>(std::min(weights.size(), (size_t)255));
                out.write(reinterpret_cast<const char*>(&weight_count), sizeof(weight_count));
                
                for (size_t w = 0; w < weight_count; ++w) {
                    int32_t bone_idx = weights[w].first;
                    float weight = weights[w].second;
                    out.write(reinterpret_cast<const char*>(&bone_idx), sizeof(bone_idx));
                    out.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
                }
            }
        }
    }
    
    SCENE_LOG_INFO("[ProjectManager] Wrote " + std::to_string(tri_count) + " triangles, " + 
                   std::to_string(transform_count) + " transforms, " +
                   std::to_string(node_name_count) + " node names to binary (v7).");
    return true;
}

bool ProjectManager::readGeometryBinary(std::ifstream& in, SceneData& scene) {
    // Read and validate header (accept RTP3, RTP4, RTP5, RTP6, RTP7 formats)
    char magic[4];
    in.read(magic, 4);
    
    // Check for valid magic
    bool is_v3 = (magic[0] == 'R' && magic[1] == 'T' && magic[2] == 'P' && magic[3] == '3');
    bool is_v4 = (magic[0] == 'R' && magic[1] == 'T' && magic[2] == 'P' && magic[3] == '4');
    bool is_v5 = (magic[0] == 'R' && magic[1] == 'T' && magic[2] == 'P' && magic[3] == '5');
    bool is_v6 = (magic[0] == 'R' && magic[1] == 'T' && magic[2] == 'P' && magic[3] == '6');
    bool is_v7 = (magic[0] == 'R' && magic[1] == 'T' && magic[2] == 'P' && magic[3] == '7');
    
    if (!is_v3 && !is_v4 && !is_v5 && !is_v6 && !is_v7) {
        SCENE_LOG_ERROR("[ProjectManager] Invalid geometry file format.");
        return false;
    }
    
    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!in) {
        SCENE_LOG_ERROR("[ProjectManager] Geometry binary: failed to read version.");
        return false;
    }
    if (version > RTP_VERSION) {
        SCENE_LOG_WARN("[ProjectManager] Newer geometry format (v" + std::to_string(version) + "), some features may not load.");
    }
    
    // Read transforms
    uint32_t transform_count;
    in.read(reinterpret_cast<char*>(&transform_count), sizeof(transform_count));
    if (!in) {
        SCENE_LOG_ERROR("[ProjectManager] Invalid transform count in geometry file: " + std::to_string(transform_count));
        return false;
    }
    
    std::vector<std::shared_ptr<Transform>> transforms;
    transforms.reserve(transform_count);
    
    for (uint32_t i = 0; i < transform_count; ++i) {
        auto tr = std::make_shared<Transform>();
        in.read(reinterpret_cast<char*>(&tr->base), sizeof(Matrix4x4));
        
        if (version >= 5) {
            // Read components
            in.read(reinterpret_cast<char*>(&tr->position), sizeof(Vec3));
            in.read(reinterpret_cast<char*>(&tr->rotation), sizeof(Vec3));
            in.read(reinterpret_cast<char*>(&tr->scale), sizeof(Vec3));
        } else {
            // v4 or older (or if components missing): Decompose base matrix to restore components
            // This prevents "Identity Reset" when UI updates the matrix
            tr->base.decompose(tr->position, tr->rotation, tr->scale);
        }
        
        // Ensure dirty or final is updated?
        tr->markDirty(); 
        
        transforms.push_back(tr);
    }

    std::vector<std::string> node_names;
    if (version >= 7) {
        uint32_t node_name_count = 0;
        in.read(reinterpret_cast<char*>(&node_name_count), sizeof(node_name_count));
        if (!in) {
            SCENE_LOG_ERROR("[ProjectManager] Invalid node name table in geometry file.");
            return false;
        }

        node_names.reserve(node_name_count);
        for (uint32_t i = 0; i < node_name_count; ++i) {
            node_names.push_back(readStringBinary(in));
            if (!in) {
                SCENE_LOG_ERROR("[ProjectManager] Failed reading node name table entry.");
                return false;
            }
        }
    }
    
    // Read triangles
    uint32_t tri_count;
    in.read(reinterpret_cast<char*>(&tri_count), sizeof(tri_count));
    if (!in) {
        SCENE_LOG_ERROR("[ProjectManager] Invalid triangle count in geometry file: " + std::to_string(tri_count));
        return false;
    }
    
    scene.world.objects.reserve(scene.world.objects.size() + tri_count);
    
    for (uint32_t i = 0; i < tri_count; ++i) {
        // Vertices
        Vec3 v0, v1, v2;
        in.read(reinterpret_cast<char*>(&v0), sizeof(Vec3));
        in.read(reinterpret_cast<char*>(&v1), sizeof(Vec3));
        in.read(reinterpret_cast<char*>(&v2), sizeof(Vec3));
        
        // Normals
        Vec3 n0, n1, n2;
        in.read(reinterpret_cast<char*>(&n0), sizeof(Vec3));
        in.read(reinterpret_cast<char*>(&n1), sizeof(Vec3));
        in.read(reinterpret_cast<char*>(&n2), sizeof(Vec3));
        
        // UVs
        Vec2 uv0, uv1, uv2;
        in.read(reinterpret_cast<char*>(&uv0), sizeof(Vec2));
        in.read(reinterpret_cast<char*>(&uv1), sizeof(Vec2));
        in.read(reinterpret_cast<char*>(&uv2), sizeof(Vec2));
        
        uint32_t uv_set_count = 0;
        std::vector<std::array<Vec2, 3>> uv_sets;
        if (version >= 6) {
            in.read(reinterpret_cast<char*>(&uv_set_count), sizeof(uv_set_count));
            uv_sets.resize(uv_set_count);
            for (uint32_t uv_set = 0; uv_set < uv_set_count; ++uv_set) {
                in.read(reinterpret_cast<char*>(&uv_sets[uv_set][0]), sizeof(Vec2));
                in.read(reinterpret_cast<char*>(&uv_sets[uv_set][1]), sizeof(Vec2));
                in.read(reinterpret_cast<char*>(&uv_sets[uv_set][2]), sizeof(Vec2));
            }
        }

        // Material ID
        uint16_t mat_id;
        in.read(reinterpret_cast<char*>(&mat_id), sizeof(mat_id));
        
        // Transform index
        uint32_t tr_idx;
        in.read(reinterpret_cast<char*>(&tr_idx), sizeof(tr_idx));

        // Node name
        std::string node_name;
        if (version >= 7) {
            uint32_t node_name_idx = 0xFFFFFFFF;
            in.read(reinterpret_cast<char*>(&node_name_idx), sizeof(node_name_idx));
            if (node_name_idx < node_names.size()) {
                node_name = node_names[node_name_idx];
            }
        } else {
            node_name = readStringBinary(in);
        }
        
        if (!in) {
            SCENE_LOG_ERROR("[ProjectManager] Geometry binary: stream failed at triangle " +
                            std::to_string(i) + "/" + std::to_string(tri_count) + ". File may be truncated.");
            break;
        }
        
        // Create triangle
        auto tri = std::make_shared<Triangle>(v0, v1, v2, n0, n1, n2, uv0, uv1, uv2, mat_id);
        tri->setNodeName(node_name);

        if (version >= 6) {
            for (uint32_t uv_set = 0; uv_set < uv_set_count; ++uv_set) {
                tri->setUVSetCoordinates(uv_set, uv_sets[uv_set][0], uv_sets[uv_set][1], uv_sets[uv_set][2]);
            }
        } else {
            tri->setUVSetCoordinates(0, uv0, uv1, uv2);
        }

        if (Material* mat = MaterialManager::getInstance().getMaterial(mat_id)) {
            if (auto* pbsdf = dynamic_cast<PrincipledBSDF*>(mat)) {
                tri->applyUVSet(static_cast<size_t>(std::max(0, pbsdf->selected_uv_set)));
            }
        }
        
        // Read skinning data (v4+)
        bool has_skin_data = false;
        if (version >= 4) {
            uint8_t has_skin;
            in.read(reinterpret_cast<char*>(&has_skin), sizeof(has_skin));
            has_skin_data = (has_skin != 0);
            
            if (has_skin_data) {
                tri->initializeSkinData();
                
                // Read bone weights for each vertex
                for (int vi = 0; vi < 3; ++vi) {
                    uint8_t weight_count;
                    in.read(reinterpret_cast<char*>(&weight_count), sizeof(weight_count));
                    
                    std::vector<std::pair<int, float>> weights;
                    weights.reserve(weight_count);
                    
                    for (uint8_t w = 0; w < weight_count; ++w) {
                        int32_t bone_idx;
                        float weight;
                        in.read(reinterpret_cast<char*>(&bone_idx), sizeof(bone_idx));
                        in.read(reinterpret_cast<char*>(&weight), sizeof(weight));
                        weights.emplace_back(bone_idx, weight);
                    }
                    
                    tri->setSkinBoneWeights(vi, weights);
                }
            }
        }
        
        // Assign transform
        if (tr_idx < transforms.size()) {
            tri->setTransformHandle(transforms[tr_idx]);
            
            // Static meshes need an initial CPU-space update for BVH/picking.
            // Skinned meshes must stay in bind/local space until animation updates them.
            if (!has_skin_data) {
                tri->updateTransformedVertices();
            }
        }
        
        tri->update_bounding_box();
        scene.world.objects.push_back(tri);
    }
    
    SCENE_LOG_INFO("[ProjectManager] Loaded " + std::to_string(tri_count) + " triangles, " + 
                   std::to_string(transform_count) + " transforms from binary.");
    return true;
}

// ============================================================================
// LIGHT SERIALIZATION
// ============================================================================

json ProjectManager::serializeLights(const std::vector<std::shared_ptr<Light>>& lights) {
    json arr = json::array();
    
    for (size_t i = 0; i < lights.size(); ++i) {
        const auto& light = lights[i];
        json l;
        
        l["id"] = i;
        l["type"] = static_cast<int>(light->type());
        l["position"] = vec3ToJson(light->position);
        l["direction"] = vec3ToJson(light->direction);
        l["color"] = vec3ToJson(light->color);
        l["intensity"] = light->intensity;
        l["radius"] = light->radius;
        
        // Spot light specific
        if (light->type() == LightType::Spot) {
            auto spot = std::dynamic_pointer_cast<SpotLight>(light);
            if (spot) {
                l["angle"] = spot->getAngleDegrees();
                l["falloff"] = spot->getFalloff();
            }
        }
        
        // Area light specific
        if (light->type() == LightType::Area) {
            l["width"] = light->width;
            l["height"] = light->height;
        }
        
        arr.push_back(l);
    }
    
    return arr;
}

void ProjectManager::deserializeLights(const json& j, std::vector<std::shared_ptr<Light>>& lights) {
    lights.clear();
    
    for (const auto& l : j) {
        LightType type = static_cast<LightType>(l.value("type", 0));
        
        Vec3 position = jsonToVec3(l.value("position", json::array({0, 5, 0})));
        Vec3 direction = jsonToVec3(l.value("direction", json::array({0, -1, 0})));
        Vec3 color = jsonToVec3(l.value("color", json::array({1, 1, 1})));
        float intensity = l.value("intensity", 100.0f);
        float radius = l.value("radius", 0.01f);
        
        std::shared_ptr<Light> light;
        
        switch (type) {
            case LightType::Point: {
                auto pl = std::make_shared<PointLight>(position, color * intensity, radius);
                light = pl;
                break;
            }
            case LightType::Directional: {
                auto dl = std::make_shared<DirectionalLight>(direction, color * intensity, radius);
                dl->position = position; // Very distant, but use position for reference
                light = dl;
                break;
            }
            case LightType::Spot: {
                float angle = l.value("angle", 45.0f);
                auto sl = std::make_shared<SpotLight>(position, direction, color * intensity, angle, radius);
                sl->setFalloff(l.value("falloff", 0.1f));
                light = sl;
                break;
            }
            case LightType::Area: {
                float width = l.value("width", 1.0f);
                float height = l.value("height", 1.0f);
                Vec3 u_vec(1, 0, 0);
                Vec3 v_vec(0, 0, 1);
                auto al = std::make_shared<AreaLight>(position, u_vec, v_vec, width, height, color * intensity);
                light = al;
                break;
            }
            default:
                SCENE_LOG_WARN("[ProjectManager] Unknown light type: " + std::to_string(static_cast<int>(type)));
                continue;
        }
        
        if (light) {
            light->nodeName = l.value("name", "Light_" + std::to_string(lights.size()));
            lights.push_back(light);
        }
    }
    
    SCENE_LOG_INFO("[ProjectManager] Loaded " + std::to_string(lights.size()) + " lights.");
}

// ============================================================================
// CAMERA SERIALIZATION
// ============================================================================

json ProjectManager::serializeCameras(const std::vector<std::shared_ptr<Camera>>& cameras, size_t active_index) {
    json arr = json::array();
    
    for (size_t i = 0; i < cameras.size(); ++i) {
        const auto& cam = cameras[i];
        json c;
        
        c["id"] = i;
        c["name"] = cam->nodeName.empty() ? ("Camera_" + std::to_string(i)) : cam->nodeName;
        c["position"] = vec3ToJson(cam->lookfrom);
        c["target"] = vec3ToJson(cam->lookat);
        c["up"] = vec3ToJson(cam->vup);
        c["fov"] = cam->vfov;
        c["aperture"] = cam->aperture;
        c["focus_dist"] = cam->focus_dist;
        c["is_active"] = (i == active_index);
        
        // Extended Camera Parameters
        c["iso"] = cam->iso_preset_index;
        c["shutter"] = cam->shutter_preset_index;
        c["fstop_idx"] = cam->fstop_preset_index;
        c["auto_exposure"] = cam->auto_exposure;
        c["ev_comp"] = cam->ev_compensation;
        c["output_aspect_idx"] = cam->output_aspect_index;
        c["use_physical_lens"] = cam->use_physical_lens;
        c["focal_length_mm"] = cam->focal_length_mm;
        c["sensor_width_mm"] = cam->sensor_width_mm;
        c["enable_motion_blur"] = cam->enable_motion_blur;
        c["rig_mode"] = static_cast<int>(cam->rig_mode);
        c["dolly_pos"] = cam->dolly_position;
        c["lens_radius"] = cam->lens_radius;
        c["blade_count"] = cam->blade_count;
        c["distortion"] = cam->distortion;
        c["body_preset_index"] = cam->body_preset_index;
        c["iso_val"] = cam->iso;
        c["sensor_height_mm"] = cam->sensor_height_mm;
        
        arr.push_back(c);
    }
    
    return arr;
}

void ProjectManager::deserializeCameras(const json& j, SceneData& scene) {
    scene.cameras.clear();
    size_t active_idx = 0;
    
    for (const auto& c : j) {
        Vec3 position = jsonToVec3(c.value("position", json::array({0, 2, 5})));
        Vec3 target = jsonToVec3(c.value("target", json::array({0, 0, 0})));
        Vec3 up = jsonToVec3(c.value("up", json::array({0, 1, 0})));
        float fov = c.value("fov", 60.0f);
        float aperture = c.value("aperture", 0.0f);
        float focus_dist = c.value("focus_dist", 10.0f);
        int blade_count = c.value("blade_count", 6);
        
        // Get aspect ratio from render settings (or default 16:9)
        float aspect = 16.0f / 9.0f;
        
        auto cam = std::make_shared<Camera>(position, target, up, fov, aspect, aperture, focus_dist, blade_count);
        cam->nodeName = c.value("name", "Camera");
        
        // Restore Extended Parameters
        cam->iso_preset_index = c.value("iso", 1);
        cam->shutter_preset_index = c.value("shutter", 1);
        cam->fstop_preset_index = c.value("fstop_idx", 4);
        cam->auto_exposure = c.value("auto_exposure", true);
        cam->ev_compensation = c.value("ev_comp", 0.0f);
        cam->output_aspect_index = c.value("output_aspect_idx", 2);
        cam->use_physical_lens = c.value("use_physical_lens", false);
        cam->focal_length_mm = c.value("focal_length_mm", 50.0f);
        cam->sensor_width_mm = c.value("sensor_width_mm", 36.0f);
        cam->enable_motion_blur = c.value("enable_motion_blur", false);
        cam->rig_mode = static_cast<Camera::RigMode>(c.value("rig_mode", 0));
        cam->dolly_position = c.value("dolly_pos", 0.0f);
        cam->lens_radius = c.value("lens_radius", aperture * 0.5f);
        cam->lens_radius = c.value("lens_radius", aperture * 0.5f);
        cam->distortion = c.value("distortion", 0.0f);
        cam->body_preset_index = c.value("body_preset_index", 1); // Default to Full Frame (index 1)
        cam->iso = c.value("iso_val", 100);
        cam->sensor_height_mm = c.value("sensor_height_mm", 24.0f);
        
        scene.cameras.push_back(cam);
        
        if (c.value("is_active", false)) {
            active_idx = scene.cameras.size() - 1;
        }
    }
    
    if (!scene.cameras.empty()) {
        scene.setActiveCamera(active_idx);
    }
    
    SCENE_LOG_INFO("[ProjectManager] Loaded " + std::to_string(scene.cameras.size()) + " cameras.");
}

// ============================================================================
// RENDER SETTINGS SERIALIZATION
// ============================================================================

json ProjectManager::serializeRenderSettings(const RenderSettings& settings) {
    json j;
    
    j["samples_per_pixel"] = settings.samples_per_pixel;
    j["samples_per_pass"] = settings.samples_per_pass;
    j["mouse_sensitivity"] = settings.mouse_sensitivity;
    j["max_bounces"] = settings.max_bounces;
    j["diffuse_bounces"] = settings.diffuse_bounces;
    j["transmission_bounces"] = settings.transmission_bounces;
    j["final_render_width"] = settings.final_render_width;
    j["final_render_height"] = settings.final_render_height;
    j["use_embree"] = settings.UI_use_embree;
    // use_denoiser (viewport denoiser) intentionally not serialized:
    // restoring it as true on project load can trigger a data race during
    // the OptiX path-trace GPU transition before the denoiser is ready.
    j["render_use_denoiser"] = settings.render_use_denoiser;
    j["denoiser_mode"] = static_cast<int>(settings.denoiser_mode);
    j["denoiser_quality"] = static_cast<int>(settings.denoiser_quality);
    j["max_samples"] = settings.max_samples;
    j["min_samples"] = settings.min_samples;
    j["use_adaptive_sampling"] = settings.use_adaptive_sampling;
    j["variance_threshold"] = settings.variance_threshold;
    // Additional UI / post-process fields
    j["denoiser_blend_factor"] = settings.denoiser_blend_factor;
    j["quality_preset"] = static_cast<int>(settings.quality_preset);
    j["raster_viewport_quality_preset"] = static_cast<int>(settings.raster_viewport_quality_preset);
    j["material_preview_lighting_preset"] = static_cast<int>(settings.material_preview_lighting_preset);
    j["resolution_source"] = static_cast<int>(settings.resolution_source);
    j["aspect_base_height"] = settings.aspect_base_height;
    j["aspect_ratio_index"] = settings.aspect_ratio_index;
    j["show_background"] = settings.show_background;
    j["persistent_tonemap"] = settings.persistent_tonemap;
    
    return j;
}

void ProjectManager::deserializeRenderSettings(const json& j, RenderSettings& settings) {
    settings.samples_per_pixel = j.value("samples_per_pixel", 1);
    settings.samples_per_pass = j.value("samples_per_pass", 1);
    settings.mouse_sensitivity = j.value("mouse_sensitivity", 0.4f);
    settings.max_bounces = j.value("max_bounces", 10);
    settings.max_bounces = std::max(1, settings.max_bounces);
    settings.diffuse_bounces = std::clamp(j.value("diffuse_bounces", 4), 1, settings.max_bounces);
    settings.transmission_bounces = std::clamp(j.value("transmission_bounces", 8), 1, settings.max_bounces);
    settings.final_render_width = j.value("final_render_width", 1280);
    settings.final_render_height = j.value("final_render_height", 720);
    // Backend/device selection is no longer serialized with project files.
    // Keep current runtime backend choice and ignore legacy fields if present.
    settings.UI_use_embree = j.value("use_embree", true);
    // use_denoiser not restored from disk — always starts false to avoid
    // GPU transition races before the denoiser pipeline is initialized.
    settings.render_use_denoiser = j.value("render_use_denoiser", true);
    settings.denoiser_mode = static_cast<DenoiserMode>(j.value("denoiser_mode", static_cast<int>(DenoiserMode::Quality)));
    settings.denoiser_quality = static_cast<DenoiserQuality>(j.value("denoiser_quality", static_cast<int>(DenoiserQuality::Fast)));
    settings.max_samples = j.value("max_samples", 32);
    settings.min_samples = j.value("min_samples", 1);
    settings.use_adaptive_sampling = j.value("use_adaptive_sampling", true);
    settings.variance_threshold = j.value("variance_threshold", 0.1f);
    // Load additional UI / post-process fields (safe defaults used when missing)
    settings.denoiser_blend_factor = j.value("denoiser_blend_factor", 1.0f);
    settings.quality_preset = static_cast<QualityPreset>(j.value("quality_preset", static_cast<int>(QualityPreset::Preview)));
    settings.raster_viewport_quality_preset = static_cast<RasterViewportQualityPreset>(
        j.value("raster_viewport_quality_preset", static_cast<int>(RasterViewportQualityPreset::Auto)));
    settings.material_preview_lighting_preset = static_cast<MaterialPreviewLightingPreset>(
        j.value("material_preview_lighting_preset", static_cast<int>(MaterialPreviewLightingPreset::Studio)));
    settings.resolution_source = static_cast<ResolutionSource>(j.value("resolution_source", static_cast<int>(ResolutionSource::Custom)));
    settings.aspect_base_height = j.value("aspect_base_height", settings.aspect_base_height);
    settings.aspect_ratio_index = j.value("aspect_ratio_index", settings.aspect_ratio_index);
    settings.show_background = j.value("show_background", settings.show_background);
    settings.persistent_tonemap = j.value("persistent_tonemap", settings.persistent_tonemap);
    
    SCENE_LOG_INFO("[ProjectManager] Loaded render settings.");
}

// ============================================================================
// TEXTURE SERIALIZATION (Path or Embed)
// ============================================================================

json ProjectManager::serializeTextures(std::ofstream& bin_out, bool embed_textures) {
    ScopedPerfTimer total_timer("serializeTextures total");
    json arr = json::array();
    
    // Get all textures from MaterialManager
    auto& mgr = MaterialManager::getInstance();
    std::unordered_map<std::string, uint32_t> texture_map; // path -> id
    texture_map.reserve(mgr.getMaterialCount() * 4);
    uint32_t texture_id = 0;
    size_t embedded_count = 0;
    size_t referenced_count = 0;
    size_t png_reencoded_count = 0;
    size_t file_copied_count = 0;
    size_t cache_passthrough_count = 0;
    size_t previous_embed_reused_count = 0;
    uint64_t texture_bin_bytes_written = 0;
    uint64_t project_local_bytes_written = 0;
    std::vector<char> file_copy_buffer(1024 * 1024);
    const fs::path current_project_path = g_project.current_file_path.empty() ? fs::path() : pathFromUtf8(g_project.current_file_path);
    const fs::path project_dir = current_project_path.empty() ? fs::path() : current_project_path.parent_path();
    const fs::path project_local_texture_dir = current_project_path.empty()
        ? fs::path()
        : (project_dir / (current_project_path.stem().string() + "_textures"));
    const fs::path generated_texture_dir = current_project_path.empty()
        ? fs::path()
        : (project_dir / (current_project_path.stem().string() + "_generated_textures"));
    
    SCENE_LOG_INFO("Starting texture serialization for " + std::to_string(mgr.getMaterialCount()) + " materials.");

    for (size_t i = 0; i < mgr.getMaterialCount(); ++i) {
        auto mat = mgr.getMaterial(static_cast<uint16_t>(i));
        if (!mat) continue;

        // CRITICAL FIX: Must cast to PrincipledBSDF to access texture properties correctly!
        // Material base class may have different/shadowed albedoProperty member
        if (mat->type() != MaterialType::PrincipledBSDF) continue;
        
        PrincipledBSDF* pbsdf = dynamic_cast<PrincipledBSDF*>(mat);
        if (!pbsdf) continue;
        
        // Check all texture properties - use PrincipledBSDF pointer for correct member access
        auto checkTexture = [&](MaterialProperty& prop, const std::string& usage) {
            if (prop.texture && prop.texture->is_loaded()) {
                std::string path = prop.texture->name;
                
                // Fallback for unnamed textures (embedded or generated)
                if (path.empty()) {
                    path = "texture_" + std::to_string(texture_id) + ".png"; // Default name
                    SCENE_LOG_WARN("Texture has no name, using fallback: " + path);
                }

                // Skip if already processed
                if (texture_map.find(path) != texture_map.end()) return;
                
                json tex_entry;
                tex_entry["id"] = texture_id;
                tex_entry["usage"] = usage;
                tex_entry["width"] = prop.texture->width;
                tex_entry["height"] = prop.texture->height;
                
                bool should_embed = embed_textures;
                if (save_settings.embed_missing_only && !fs::exists(path)) {
                    should_embed = true;
                }
                if (save_settings.texture_storage_mode == ProjectManager::TextureStorageMode::ProjectLocal ||
                    save_settings.texture_storage_mode == ProjectManager::TextureStorageMode::KeepOriginalPaths) {
                    should_embed = false;
                }
                
                // Unnamed/embedded cache textures should only force embedding in explicit Embedded mode.
                if (prop.texture->name.empty() || prop.texture->name.find("embedded_") == 0) {
                     should_embed = (save_settings.texture_storage_mode == ProjectManager::TextureStorageMode::Embedded);
                }

                tex_entry["original_name"] = path; // Save Original Name for restoration

                // Detect generated/painted texture names so we prefer in-memory pixels
                bool is_generated_name = (path.find("generated/") != std::string::npos) ||
                                         (path.find("generated_tex_") != std::string::npos) ||
                                         (path.find("_generated_textures") != std::string::npos);

                if (should_embed) {
                    ++embedded_count;
                    // Embed texture data
                    tex_entry["mode"] = "embed";
                    tex_entry["offset"] = static_cast<long long>(bin_out.tellp());

                    const std::string previous_key = previousEmbeddedTextureKey(
                        path, usage, prop.texture->width, prop.texture->height);
                    auto previous_it = g_previous_embedded_texture_entries.find(previous_key);
                    if (!prop.texture->isSaveDirty() &&
                        previous_it != g_previous_embedded_texture_entries.end()) {
                        std::vector<char> prev_blob;
                        if (readPreviousEmbeddedTextureBlob(previous_it->second, prev_blob)) {
                            auto recomp = pickColorRecompression(prev_blob.data(), prev_blob.size(), usage);
                            const char* write_data = recomp.recompressed ? recomp.bytes.data() : prev_blob.data();
                            const size_t write_size = recomp.recompressed ? recomp.bytes.size() : prev_blob.size();
                            const std::string write_ext = recomp.recompressed ? recomp.ext : previous_it->second.format;

                            bin_out.write(write_data, static_cast<std::streamsize>(write_size));
                            tex_entry["size"] = write_size;
                            tex_entry["format"] = write_ext;
                            if (recomp.recompressed) ++png_reencoded_count;
                            else ++previous_embed_reused_count;
                            texture_bin_bytes_written += static_cast<uint64_t>(write_size);
                            prop.texture->clearSaveDirty();
                            texture_map[path] = texture_id++;
                            arr.push_back(tex_entry);
                            return;
                        }
                    }

                    // If we have in-memory pixels (e.g. painted/generated texture) prefer encoding them
                    // even if a file exists with the same name. This ensures painted overrides are serialized.
                    if (fs::exists(path) && prop.texture->pixels.empty() && !is_generated_name) {
                        // Read the source into memory so we can attempt PNG → JPG
                        // recompression for color textures; if it doesn't pay off,
                        // we fall back to byte-for-byte passthrough.
                        std::ifstream tex_file(path, std::ios::binary | std::ios::ate);
                        if (tex_file) {
                            const std::streamoff total_size = tex_file.tellg();
                            tex_file.seekg(0, std::ios::beg);
                            std::vector<char> blob;
                            if (total_size > 0) {
                                blob.resize(static_cast<size_t>(total_size));
                                tex_file.read(blob.data(), static_cast<std::streamsize>(blob.size()));
                                if (tex_file.gcount() != static_cast<std::streamsize>(blob.size())) {
                                    blob.clear();
                                }
                            }

                            auto recomp = pickColorRecompression(blob.data(), blob.size(), usage);
                            const char* write_data = recomp.recompressed ? recomp.bytes.data() : blob.data();
                            const size_t write_size = recomp.recompressed ? recomp.bytes.size() : blob.size();
                            const std::string write_ext = recomp.recompressed
                                ? recomp.ext
                                : fs::path(path).extension().string();

                            bin_out.write(write_data, static_cast<std::streamsize>(write_size));
                            if (recomp.recompressed) ++png_reencoded_count;
                            else ++file_copied_count;
                            texture_bin_bytes_written += static_cast<uint64_t>(write_size);

                            tex_entry["size"] = write_size;
                            tex_entry["format"] = write_ext;
                        }
                    } else if (!prop.texture->pixels.empty()) {
                         // Encode from in-memory pixels (painted/generated textures).
                         std::vector<uint8_t> raw_pixels(prop.texture->pixels.size() * 4);

                         auto linear_to_srgb_byte = [](float lin) -> uint8_t {
                             lin = std::clamp(lin, 0.0f, 1.0f);
                             float sr = (lin <= 0.0031308f) ? (lin * 12.92f) : (1.055f * powf(lin, 1.0f / 2.4f) - 0.055f);
                             return static_cast<uint8_t>(std::roundf(sr * 255.0f));
                         };

                         bool convert_to_srgb = (usage == "albedo") && (!prop.texture->is_srgb);

                         for (size_t pi = 0; pi < prop.texture->pixels.size(); ++pi) {
                             const auto& px = prop.texture->pixels[pi];
                             if (convert_to_srgb) {
                                 // pixels are stored as linear bytes -> convert to sRGB bytes
                                 float r_lin = px.r / 255.0f;
                                 float g_lin = px.g / 255.0f;
                                 float b_lin = px.b / 255.0f;
                                 raw_pixels[pi * 4 + 0] = linear_to_srgb_byte(r_lin);
                                 raw_pixels[pi * 4 + 1] = linear_to_srgb_byte(g_lin);
                                 raw_pixels[pi * 4 + 2] = linear_to_srgb_byte(b_lin);
                                 raw_pixels[pi * 4 + 3] = px.a;
                             } else {
                                 raw_pixels[pi * 4 + 0] = px.r;
                                 raw_pixels[pi * 4 + 1] = px.g;
                                 raw_pixels[pi * 4 + 2] = px.b;
                                 raw_pixels[pi * 4 + 3] = px.a;
                             }
                         }

                         std::vector<char> encoded;
                         std::string encoded_ext;
                         if (encodeTextureRoleAware(raw_pixels.data(),
                                                    prop.texture->width, prop.texture->height,
                                                    usage, encoded, encoded_ext)) {
                             ++png_reencoded_count;
                             bin_out.write(encoded.data(), encoded.size());
                             tex_entry["size"] = encoded.size();
                             tex_entry["format"] = encoded_ext;
                             texture_bin_bytes_written += static_cast<uint64_t>(encoded.size());
                         } else {
                             SCENE_LOG_ERROR("Failed to encode memory texture: " + path);
                         }

                    } else if (const auto* cached = getEmbeddedTexture(path)) {
                        auto recomp = pickColorRecompression(cached->data.data(), cached->data.size(), usage);
                        const char* write_data = recomp.recompressed ? recomp.bytes.data() : cached->data.data();
                        const size_t write_size = recomp.recompressed ? recomp.bytes.size() : cached->data.size();
                        const std::string write_ext = recomp.recompressed
                            ? recomp.ext
                            : fs::path(path).extension().string();

                        bin_out.write(write_data, static_cast<std::streamsize>(write_size));
                        tex_entry["size"] = write_size;
                        tex_entry["format"] = write_ext;
                        if (recomp.recompressed) ++png_reencoded_count;
                        else ++cache_passthrough_count;
                        texture_bin_bytes_written += static_cast<uint64_t>(write_size);
                        prop.texture->clearSaveDirty();
                        texture_map[path] = texture_id++;
                        arr.push_back(tex_entry);
                        return;
                    } else if (!prop.texture->float_pixels.empty()) {
                        // Handle Float textures (EXR/HDR) - TODO: Need stbi_write_hdr or similar
                        // For now, skip or implement later.
                        SCENE_LOG_WARN("Float texture embedding not fully implemented (skipping encode): " + path);
                    } 
                } else {
                    bool wrote_project_local = false;
                    // Role-aware encoder: takes a target path WITHOUT extension and
                    // returns the actual final path (with .jpg or .png suffix), or
                    // empty fs::path on failure.
                    auto write_generated_image_from_pixels = [&](const fs::path& target_stem) -> fs::path {
                        if (prop.texture->pixels.empty()) {
                            return {};
                        }

                        auto linear_to_srgb_byte = [](float lin) -> uint8_t {
                            lin = std::clamp(lin, 0.0f, 1.0f);
                            float sr = (lin <= 0.0031308f) ? (lin * 12.92f) : (1.055f * powf(lin, 1.0f / 2.4f) - 0.055f);
                            return static_cast<uint8_t>(std::roundf(sr * 255.0f));
                        };

                        bool convert_to_srgb = (usage == "albedo") && (!prop.texture->is_srgb);

                        std::vector<uint8_t> raw_pixels(prop.texture->pixels.size() * 4);
                        for (size_t pi = 0; pi < prop.texture->pixels.size(); ++pi) {
                            const auto& px = prop.texture->pixels[pi];
                            if (convert_to_srgb) {
                                float r_lin = px.r / 255.0f;
                                float g_lin = px.g / 255.0f;
                                float b_lin = px.b / 255.0f;
                                raw_pixels[pi * 4 + 0] = linear_to_srgb_byte(r_lin);
                                raw_pixels[pi * 4 + 1] = linear_to_srgb_byte(g_lin);
                                raw_pixels[pi * 4 + 2] = linear_to_srgb_byte(b_lin);
                                raw_pixels[pi * 4 + 3] = px.a;
                            } else {
                                raw_pixels[pi * 4 + 0] = px.r;
                                raw_pixels[pi * 4 + 1] = px.g;
                                raw_pixels[pi * 4 + 2] = px.b;
                                raw_pixels[pi * 4 + 3] = px.a;
                            }
                        }

                        std::vector<char> encoded;
                        std::string encoded_ext;
                        if (!encodeTextureRoleAware(raw_pixels.data(),
                                                    prop.texture->width, prop.texture->height,
                                                    usage, encoded, encoded_ext)) {
                            return {};
                        }

                        fs::path local_target = target_stem;
                        local_target += encoded_ext;

                        std::ofstream local_out(local_target, std::ios::binary);
                        if (!local_out.is_open()) {
                            return {};
                        }
                        local_out.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
                        ++png_reencoded_count;
                        project_local_bytes_written += static_cast<uint64_t>(encoded.size());
                        return local_target;
                    };
                    if (save_settings.texture_storage_mode == ProjectManager::TextureStorageMode::ProjectLocal &&
                        !project_local_texture_dir.empty()) {
                        try {
                            fs::create_directories(project_local_texture_dir);
                            const std::string base_name = sanitizeFilenameComponent(fs::path(path).stem().string());

                            auto finalize_project_local_path = [&](const fs::path& local_target) {
                                tex_entry["mode"] = "path";
                                tex_entry["path"] = fs::relative(local_target, project_dir).generic_string();
                                wrote_project_local = true;
                            };

                            auto target_matches_source_file = [&](const fs::path& local_target, const fs::path& source_path) -> bool {
                                std::error_code ec;
                                if (!fs::exists(local_target, ec) || !fs::exists(source_path, ec)) return false;
                                const auto target_size = fs::file_size(local_target, ec);
                                if (ec) return false;
                                const auto source_size = fs::file_size(source_path, ec);
                                if (ec || target_size != source_size) return false;
                                const auto target_time = fs::last_write_time(local_target, ec);
                                if (ec) return false;
                                const auto source_time = fs::last_write_time(source_path, ec);
                                if (ec) return false;
                                return target_time >= source_time;
                            };

                            auto target_matches_cached_bytes = [&](const fs::path& local_target, size_t expected_size) -> bool {
                                std::error_code ec;
                                if (!fs::exists(local_target, ec)) return false;
                                return fs::file_size(local_target, ec) == expected_size;
                            };

                            if (is_generated_name && !prop.texture->pixels.empty()) {
                                const fs::path local_target_stem = project_local_texture_dir /
                                    (std::to_string(texture_id) + "_" + base_name);
                                fs::path local_target = write_generated_image_from_pixels(local_target_stem);
                                if (!local_target.empty() && fs::exists(local_target)) {
                                    finalize_project_local_path(local_target);
                                }
                            } else if (fs::exists(path)) {
                                const fs::path source_path(path);
                                const std::string ext_orig = !source_path.extension().string().empty()
                                    ? source_path.extension().string()
                                    : std::string(".bin");

                                bool wrote_recompressed = false;
                                if (isLossyEncodableUsage(usage)) {
                                    std::ifstream src(source_path, std::ios::binary | std::ios::ate);
                                    if (src) {
                                        const std::streamoff sz = src.tellg();
                                        src.seekg(0, std::ios::beg);
                                        std::vector<char> blob;
                                        if (sz > 0) {
                                            blob.resize(static_cast<size_t>(sz));
                                            src.read(blob.data(), static_cast<std::streamsize>(blob.size()));
                                            if (src.gcount() != static_cast<std::streamsize>(blob.size())) {
                                                blob.clear();
                                            }
                                        }
                                        auto recomp = pickColorRecompression(blob.data(), blob.size(), usage);
                                        if (recomp.recompressed) {
                                            const fs::path local_target = project_local_texture_dir /
                                                (std::to_string(texture_id) + "_" + base_name + recomp.ext);
                                            std::ofstream out(local_target, std::ios::binary);
                                            if (out.is_open()) {
                                                out.write(recomp.bytes.data(), static_cast<std::streamsize>(recomp.bytes.size()));
                                                ++png_reencoded_count;
                                                project_local_bytes_written += static_cast<uint64_t>(recomp.bytes.size());
                                                finalize_project_local_path(local_target);
                                                wrote_recompressed = true;
                                            }
                                        }
                                    }
                                }

                                if (!wrote_recompressed) {
                                    const fs::path local_target = project_local_texture_dir /
                                        (std::to_string(texture_id) + "_" + base_name + ext_orig);
                                    if (!target_matches_source_file(local_target, source_path)) {
                                        fs::copy_file(source_path, local_target, fs::copy_options::overwrite_existing);
                                        ++file_copied_count;
                                        std::error_code size_ec;
                                        project_local_bytes_written += static_cast<uint64_t>(fs::file_size(local_target, size_ec));
                                    }
                                    finalize_project_local_path(local_target);
                                }
                            } else if (const auto* cached = getEmbeddedTexture(path)) {
                                auto recomp = pickColorRecompression(cached->data.data(), cached->data.size(), usage);
                                const std::string ext = recomp.recompressed
                                    ? recomp.ext
                                    : chooseMaterializedTextureExtension(fs::path(path), &cached->data);
                                const fs::path local_target = project_local_texture_dir /
                                    (std::to_string(texture_id) + "_" + base_name + ext);
                                const char* write_data = recomp.recompressed ? recomp.bytes.data() : cached->data.data();
                                const size_t write_size = recomp.recompressed ? recomp.bytes.size() : cached->data.size();
                                const bool already_present = !recomp.recompressed &&
                                    target_matches_cached_bytes(local_target, write_size);
                                if (!already_present) {
                                    std::ofstream local_out(local_target, std::ios::binary);
                                    if (local_out.is_open()) {
                                        local_out.write(write_data, static_cast<std::streamsize>(write_size));
                                        if (recomp.recompressed) ++png_reencoded_count;
                                        else ++cache_passthrough_count;
                                        project_local_bytes_written += static_cast<uint64_t>(write_size);
                                    }
                                }
                                if (fs::exists(local_target)) {
                                    finalize_project_local_path(local_target);
                                }
                            } else if (!prop.texture->pixels.empty()) {
                                const fs::path local_target_stem = project_local_texture_dir /
                                    (std::to_string(texture_id) + "_" + base_name);
                                fs::path local_target = write_generated_image_from_pixels(local_target_stem);
                                if (!local_target.empty() && fs::exists(local_target)) {
                                    finalize_project_local_path(local_target);
                                }
                            }
                        } catch (const std::exception& e) {
                            SCENE_LOG_WARN("[ProjectManager] Failed to create project-local texture copy for " + path +
                                           " | " + e.what());
                        }
                    }

                    ++referenced_count;
                    tex_entry["mode"] = "path";
                    if (!wrote_project_local) {
                        bool wrote_generated_keep_path = false;
                        if (save_settings.texture_storage_mode == ProjectManager::TextureStorageMode::KeepOriginalPaths &&
                            (is_generated_name || !fs::exists(path)) &&
                            !generated_texture_dir.empty()) {
                            try {
                                fs::create_directories(generated_texture_dir);
                                const std::string base_name = sanitizeFilenameComponent(fs::path(path).stem().string().empty()
                                    ? ("generated_tex_" + std::to_string(texture_id))
                                    : fs::path(path).stem().string());
                                const fs::path generated_target_stem = generated_texture_dir /
                                    (std::to_string(texture_id) + "_" + base_name);
                                fs::path generated_target = write_generated_image_from_pixels(generated_target_stem);
                                wrote_generated_keep_path = !generated_target.empty();
                                if (wrote_generated_keep_path) {
                                    tex_entry["path"] = fs::relative(generated_target, project_dir).generic_string();
                                }
                            } catch (const std::exception& e) {
                                SCENE_LOG_WARN("[ProjectManager] Failed to materialize generated texture for KeepOriginalPaths: " +
                                               path + " | " + e.what());
                            }

                            if (!wrote_generated_keep_path) {
                                if (const auto* cached = getEmbeddedTexture(path)) {
                                    try {
                                        const std::string base_name = sanitizeFilenameComponent(fs::path(path).stem().string().empty()
                                            ? ("generated_tex_" + std::to_string(texture_id))
                                            : fs::path(path).stem().string());
                                        auto recomp = pickColorRecompression(cached->data.data(), cached->data.size(), usage);
                                        const std::string ext = recomp.recompressed
                                            ? recomp.ext
                                            : chooseMaterializedTextureExtension(fs::path(path), &cached->data);
                                        const fs::path generated_target = generated_texture_dir /
                                            (std::to_string(texture_id) + "_" + base_name + ext);
                                        const char* write_data = recomp.recompressed ? recomp.bytes.data() : cached->data.data();
                                        const size_t write_size = recomp.recompressed ? recomp.bytes.size() : cached->data.size();
                                        std::ofstream generated_out(generated_target, std::ios::binary);
                                        if (generated_out.is_open()) {
                                            generated_out.write(write_data, static_cast<std::streamsize>(write_size));
                                            project_local_bytes_written += static_cast<uint64_t>(write_size);
                                            if (recomp.recompressed) ++png_reencoded_count;
                                            else ++cache_passthrough_count;
                                        }
                                        if (fs::exists(generated_target)) {
                                            wrote_generated_keep_path = true;
                                            tex_entry["path"] = fs::relative(generated_target, project_dir).generic_string();
                                        }
                                    } catch (const std::exception& e) {
                                        SCENE_LOG_WARN("[ProjectManager] Failed to materialize cached embedded texture for KeepOriginalPaths: " +
                                                       path + " | " + e.what());
                                    }
                                }
                            }
                        }

                        if (!wrote_generated_keep_path) {
                            tex_entry["path"] = path;
                        }
                    }
                }
                
                prop.texture->clearSaveDirty();
                texture_map[path] = texture_id++;
                arr.push_back(tex_entry);
            }
        };
        
        // Use PrincipledBSDF pointer for correct texture property access
        checkTexture(pbsdf->albedoProperty, "albedo");
        checkTexture(pbsdf->normalProperty, "normal");
        checkTexture(pbsdf->roughnessProperty, "roughness");
        checkTexture(pbsdf->metallicProperty, "metallic");
        checkTexture(pbsdf->emissionProperty, "emission");
        checkTexture(pbsdf->heightProperty, "height");
        checkTexture(pbsdf->opacityProperty, "opacity");
        checkTexture(pbsdf->transmissionProperty, "transmission");
    }
    
    SCENE_LOG_INFO("[ProjectManager] Serialized " + std::to_string(arr.size()) + " textures.");
    SCENE_LOG_INFO("[ProjectManager] Texture modes - embedded: " + std::to_string(embedded_count) +
                   ", path refs: " + std::to_string(referenced_count) +
                   ", png re-encoded: " + std::to_string(png_reencoded_count) +
                   ", file-copied-bytesources: " + std::to_string(file_copied_count) +
                   ", cache-passthrough: " + std::to_string(cache_passthrough_count) +
                   ", previous-embed-reused: " + std::to_string(previous_embed_reused_count) +
                   ", bin-bytes: " + std::to_string(texture_bin_bytes_written) +
                   ", project-local-bytes: " + std::to_string(project_local_bytes_written));
    return arr;
}

void ProjectManager::deserializeTextures(const json& j, std::ifstream& bin_in, const std::string& project_dir) {
    // Clear previous caches
    m_embedded_texture_cache.clear();
    m_texture_path_remap.clear();

    for (const auto& tex : j) {
        std::string mode = tex.value("mode", "path");

        if (mode == "embed") {
            // Store embedded texture in memory cache (NO DISK WRITE!)
            // MaterialManager::deserializeProperty will load from this cache
            long long offset = tex.value("offset", 0LL);
            size_t size = tex.value("size", 0);
            std::string original_name = tex.value("original_name", "");
            std::string usage = tex.value("usage", "albedo");

            if (size > 0 && bin_in.is_open() && !original_name.empty()) {
                bin_in.seekg(offset);
                std::vector<char> data(size);
                bin_in.read(data.data(), size);

                // Determine TextureType from usage string
                TextureType texType = TextureType::Albedo;
                if (usage == "normal") texType = TextureType::Normal;
                else if (usage == "roughness") texType = TextureType::Roughness;
                else if (usage == "metallic") texType = TextureType::Metallic;
                else if (usage == "specular") texType = TextureType::Specular;
                else if (usage == "emission") texType = TextureType::Emission;
                else if (usage == "height") texType = TextureType::Unknown;
                else if (usage == "opacity") texType = TextureType::Opacity;
                else if (usage == "transmission") texType = TextureType::Transmission;

                // Store in memory cache - keyed by original_name
                m_embedded_texture_cache[original_name] = {std::move(data), texType};
            }
        } else {
            // Path mode: build a remap from original_name -> resolved disk path.
            // This is needed when a previously-embedded texture (e.g. "embedded_0") was
            // saved as a project-local copy. The material JSON still stores the original
            // embedded name, so without this remap deserializeProperty can't find the file.
            std::string path = tex.value("path", "");
            std::string original_name = tex.value("original_name", "");
            if (!original_name.empty() && !path.empty() && original_name != path) {
                // Resolve against project_dir if not absolute
                std::string full_path = path;
                if (!fs::path(full_path).is_absolute() && !project_dir.empty()) {
                    full_path = (fs::path(project_dir) / path).generic_string();
                }
                if (fs::exists(full_path)) {
                    m_texture_path_remap[original_name] = full_path;
                }
            }
        }
    }
    SCENE_LOG_INFO("[ProjectManager] Cached " + std::to_string(m_embedded_texture_cache.size()) +
                   " embedded textures in memory, " + std::to_string(m_texture_path_remap.size()) +
                   " path remaps (no temp files).");
}

// ============================================================================
// VDB SERIALIZATION
// ============================================================================

json ProjectManager::serializeVDBVolumes(const std::vector<std::shared_ptr<VDBVolume>>& vdb_volumes) {
    json arr = json::array();
    
    for (size_t i = 0; i < vdb_volumes.size(); ++i) {
        const auto& vdb = vdb_volumes[i];
        json j;
        
        j["id"] = i;
        j["name"] = vdb->name;
        j["filepath"] = vdb->getFilePath();
        j["transform"] = mat4ToJson(vdb->getPivotMatrix());
        j["density_scale"] = vdb->density_scale;
        j["pivot_offset"] = {vdb->getPivotOffset().x, vdb->getPivotOffset().y, vdb->getPivotOffset().z};
        
        // Animation
        j["is_sequence"] = vdb->isAnimated();
        j["current_frame"] = vdb->getCurrentFrame();
        j["timeline_linked"] = vdb->isLinkedToTimeline();
        j["frame_offset"] = vdb->getFrameOffset();
        
        // Shader
        if (auto shader = vdb->volume_shader) {
             json s;
             s["name"] = shader->name;
             s["density"] = shader->density.toJson();
             s["scattering"] = shader->scattering.toJson();
             s["absorption"] = shader->absorption.toJson();
             s["emission"] = shader->emission.toJson();
             s["quality"] = shader->quality.toJson();
             
             // Motion blur manual
             s["motion_blur"] = {
                 {"enabled", shader->motion_blur.enabled},
                 {"velocity_channel", shader->motion_blur.velocity_channel},
                 {"scale", shader->motion_blur.scale}
             };
             
             j["shader"] = s;
        }
        
        arr.push_back(j);
    }
    
    return arr;
}

void ProjectManager::deserializeVDBVolumes(const json& j_arr, SceneData& scene) {
    // VDB volumes are stored in scene.vdb_volumes AND scene.world.objects
    // We must ensure they are added to both!
    
    for (const auto& j : j_arr) {
      try {
        std::string filepath = j.value("filepath", "");
        if (filepath.empty()) continue;
        
        auto vdb = std::make_shared<VDBVolume>();
        
        bool is_sequence = j.value("is_sequence", false);
        
        if (is_sequence) {
            if (vdb->loadVDBSequence(filepath)) {
                // Success
                if (!vdb->uploadToGPU()) SCENE_LOG_WARN("Failed to upload VDB sequence to GPU: " + filepath);
            } else {
                 SCENE_LOG_WARN("Failed to load VDB sequence: " + filepath);
                 continue;
            }
        } else {
             if (vdb->loadVDB(filepath)) {
                 // Success
                 if (!vdb->uploadToGPU()) SCENE_LOG_WARN("Failed to upload VDB to GPU: " + filepath);
             } else {
                 SCENE_LOG_WARN("Failed to load VDB file: " + filepath);
                 continue;
             }
        }
        
        vdb->name = j.value("name", "VDB Volume");
        if (j.contains("pivot_offset")) {
            auto po = j["pivot_offset"];
            vdb->setPivotOffset(Vec3(po[0], po[1], po[2]), false);
        }
        if (j.contains("transform")) vdb->setPivotMatrix(jsonToMat4(j["transform"]));
        vdb->density_scale = j.value("density_scale", 1.0f);
        
        vdb->setCurrentFrame(j.value("current_frame", 0));
        vdb->setLinkedToTimeline(j.value("timeline_linked", true));
        vdb->setFrameOffset(j.value("frame_offset", 0));
        
        // Shader
        if (j.contains("shader")) {
            auto j_shader = j["shader"];
            auto shader = std::make_shared<VolumeShader>();
            shader->name = j_shader.value("name", "Volume Shader");
            
            if (j_shader.contains("density")) shader->density.fromJson(j_shader["density"]);
            if (j_shader.contains("scattering")) shader->scattering.fromJson(j_shader["scattering"]);
            if (j_shader.contains("absorption")) shader->absorption.fromJson(j_shader["absorption"]);
            if (j_shader.contains("emission")) shader->emission.fromJson(j_shader["emission"]);
            if (j_shader.contains("quality")) shader->quality.fromJson(j_shader["quality"]);
            
             if (j_shader.contains("motion_blur")) {
                 shader->motion_blur.enabled = j_shader["motion_blur"].value("enabled", false);
                 shader->motion_blur.velocity_channel = j_shader["motion_blur"].value("velocity_channel", "vel");
                 shader->motion_blur.scale = j_shader["motion_blur"].value("scale", 1.0f);
             }
            
            vdb->setShader(shader);
        }
        
        scene.addVDBVolume(vdb);
        scene.world.objects.push_back(vdb);
      } catch (const std::exception& e) {
          SCENE_LOG_ERROR("[ProjectManager] Skipping VDB volume — deserialize failed: " + std::string(e.what()));
      } catch (...) {
          SCENE_LOG_ERROR("[ProjectManager] Skipping VDB volume — unknown error during deserialize");
      }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GAS VOLUME SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

json ProjectManager::serializeGasVolumes(const std::vector<std::shared_ptr<GasVolume>>& gas_volumes) {
    json arr = json::array();
    for (const auto& gas : gas_volumes) {
        if (gas) arr.push_back(gas->toJson());
    }
    return arr;
}

void ProjectManager::deserializeGasVolumes(const json& j_arr, SceneData& scene) {
    for (const auto& j : j_arr) {
        try {
            auto gas = std::make_shared<GasVolume>();
            // NOTE: fromJson() calls initialize() internally.
            // Do NOT call initialize() again afterwards — that causes a double CUDA
            // alloc / driver deadlock, and is especially dangerous when the original
            // baked / live VDB data is missing on the current machine.
            gas->fromJson(j);
            scene.addGasVolume(gas);
            scene.world.objects.push_back(gas);
        } catch (const std::exception& e) {
            SCENE_LOG_ERROR("[ProjectManager] Skipping GAS volume — deserialize failed: " + std::string(e.what()));
        } catch (...) {
            SCENE_LOG_ERROR("[ProjectManager] Skipping GAS volume — unknown error during deserialize");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FORCE FIELD SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

json ProjectManager::serializeForceFields(const Physics::ForceFieldManager& ffm) {
    return ffm.toJson();
}

void ProjectManager::deserializeForceFields(const json& j, SceneData& scene) {
    scene.force_field_manager.fromJson(j);
}
