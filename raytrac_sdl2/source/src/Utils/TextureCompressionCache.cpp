#include "TextureCompressionCache.h"
#include "ProjectManager.h"

#include <DirectXTex.h>

#include <fstream>
#include <sstream>

namespace {
    namespace fs = std::filesystem;

    const char* targetSuffix(TextureCompressionTarget target) {
        switch (target) {
            case TextureCompressionTarget::BC4: return "bc4";
            case TextureCompressionTarget::BC5: return "bc5";
            case TextureCompressionTarget::BC7: return "bc7";
            default: return "raw";
        }
    }

    uint64_t hashTextureCacheIdentity(
        const Texture& tex,
        TextureType type,
        bool srgb,
        TextureCompressionTarget target,
        uint8_t sourceChannel = 0)
    {
        std::ostringstream oss;
        oss << tex.name << '|'
            << tex.width << 'x' << tex.height << '|'
            << static_cast<int>(type) << '|'
            << static_cast<int>(target) << '|'
            << static_cast<int>(sourceChannel) << '|'
            << (srgb ? 1 : 0) << '|'
            << (tex.has_alpha ? 1 : 0) << '|'
            << (tex.is_gray_scale ? 1 : 0) << '|'
            << (tex.is_hdr ? 1 : 0);

        if (!tex.name.empty()) {
            try {
                const auto mod = fs::last_write_time(tex.name).time_since_epoch().count();
                oss << '|' << mod;
            } catch (...) {
            }
        }

        return std::hash<std::string>{}(oss.str());
    }

    std::optional<fs::path> getManagedProjectTextureCacheDirectory() {
        const std::string currentProjectPath = ProjectManager::getInstance().getCurrentFilePath();
        if (currentProjectPath.empty()) return std::nullopt;

        fs::path projectPath(currentProjectPath);
        fs::path projectDir = projectPath.has_parent_path() ? projectPath.parent_path() : fs::current_path();
        const std::string projectKey = projectPath.lexically_normal().string();
        const uint64_t projectHash = std::hash<std::string>{}(projectKey);

        std::ostringstream projectFolder;
        projectFolder << projectPath.stem().string() << '_' << std::hex << projectHash;

        return projectDir / ".raytrophi_cache" / projectFolder.str() / "textures";
    }

    std::optional<TextureCompressedCacheCandidate> makeManagedCacheCandidate(
        const Texture& tex,
        TextureType type,
        bool srgb,
        TextureCompressionTarget target,
        uint8_t sourceChannel = 0)
    {
        if (tex.name.empty() || target == TextureCompressionTarget::None) return std::nullopt;

        const auto cacheDir = getManagedProjectTextureCacheDirectory();
        if (!cacheDir) return std::nullopt;

        const uint64_t identity = hashTextureCacheIdentity(tex, type, srgb, target, sourceChannel);
        std::ostringstream filename;
        filename << fs::path(tex.name).stem().string()
                 << '.'
                 << targetSuffix(target)
                 << '.'
                 << (srgb ? "srgb" : "linear")
                 << '.'
                 << std::hex << identity
                 << ".dds";

        TextureCompressedCacheCandidate candidate;
        candidate.ddsPath = *cacheDir / filename.str();
        candidate.target = target;
        candidate.srgb = srgb;
        return candidate;
    }

    // Returns the DXGI format written in a DDS file header, or DXGI_FORMAT_UNKNOWN on failure.
    // Handles both legacy FourCC (ATI1/ATI2/BC4U/BC5U) and DX10-extended headers (required for BC7).
    DXGI_FORMAT peekDDSFormat(const fs::path& ddsPath) {
        std::ifstream f(ddsPath, std::ios::binary);
        if (!f) return DXGI_FORMAT_UNKNOWN;

        uint32_t magic = 0;
        f.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != 0x20534444u) return DXGI_FORMAT_UNKNOWN; // 'DDS '

        // DDS_HEADER.ddpfPixelFormat.dwFourCC is at byte offset:
        //   4 (magic) + 72 (bytes into DDS_HEADER to reach ddpfPixelFormat) + 8 (dwSize+dwFlags) = 84
        f.seekg(84);
        uint32_t fourCC = 0;
        f.read(reinterpret_cast<char*>(&fourCC), 4);
        if (!f) return DXGI_FORMAT_UNKNOWN;

        constexpr uint32_t kDX10 = 0x30315844u; // 'DX10'
        constexpr uint32_t kATI1 = 0x31495441u; // 'ATI1' (BC4)
        constexpr uint32_t kBC4U = 0x55344342u; // 'BC4U'
        constexpr uint32_t kATI2 = 0x32495441u; // 'ATI2' (BC5)
        constexpr uint32_t kBC5U = 0x55354342u; // 'BC5U'

        if (fourCC == kDX10) {
            // DDS_HEADER_DXT10 starts at byte 4 (magic) + 124 (header) = 128; first field is dxgiFormat.
            f.seekg(128);
            uint32_t dxgi = 0;
            f.read(reinterpret_cast<char*>(&dxgi), 4);
            if (!f) return DXGI_FORMAT_UNKNOWN;
            return static_cast<DXGI_FORMAT>(dxgi);
        }
        if (fourCC == kATI1 || fourCC == kBC4U) return DXGI_FORMAT_BC4_UNORM;
        if (fourCC == kATI2 || fourCC == kBC5U) return DXGI_FORMAT_BC5_UNORM;
        return DXGI_FORMAT_UNKNOWN;
    }

    DXGI_FORMAT dxgiFormatForTarget(TextureCompressionTarget target, bool srgb) {
        switch (target) {
            case TextureCompressionTarget::BC4:
                return DXGI_FORMAT_BC4_UNORM;
            case TextureCompressionTarget::BC5:
                return DXGI_FORMAT_BC5_UNORM;
            case TextureCompressionTarget::BC7:
                return srgb ? DXGI_FORMAT_BC7_UNORM_SRGB : DXGI_FORMAT_BC7_UNORM;
            default:
                return DXGI_FORMAT_UNKNOWN;
        }
    }

    std::optional<TextureCompressedCacheCandidate> findAdjacentDDSCandidate(
        const Texture& tex,
        TextureCompressionTarget target,
        bool srgb)
    {
        if (tex.name.empty() || target == TextureCompressionTarget::None) return std::nullopt;

        fs::path sourcePath(tex.name);
        fs::path ddsPath = sourcePath;
        ddsPath.replace_extension(".dds");
        if (!fs::exists(ddsPath)) return std::nullopt;

        // Validate that the adjacent DDS was actually compressed with the expected format.
        // Without this, a BC4 roughness.dds sitting next to roughness.png would be consumed
        // as a BC7 roughness texture, making the shader read .g/.b = 0.
        const DXGI_FORMAT actualFmt   = peekDDSFormat(ddsPath);
        const DXGI_FORMAT expectedFmt = dxgiFormatForTarget(target, srgb);
        // BC7 has sRGB and linear variants — accept either since the app controls gamma at upload.
        const DXGI_FORMAT altFmt = (target == TextureCompressionTarget::BC7)
            ? dxgiFormatForTarget(target, !srgb)
            : DXGI_FORMAT_UNKNOWN;
        if (actualFmt == DXGI_FORMAT_UNKNOWN ||
            (actualFmt != expectedFmt && actualFmt != altFmt)) {
            return std::nullopt;
        }

        TextureCompressedCacheCandidate candidate;
        candidate.ddsPath = ddsPath;
        candidate.target = target;
        candidate.srgb = srgb;
        return candidate;
    }

    // channel: 0=R, 1=G, 2=B, 3=A — must match pbr_texture_policy.glsl
    bool buildScalarSourceImage(const Texture& tex, std::vector<uint8_t>& bytes, DirectX::Image& image, uint8_t channel = 0) {
        if (tex.width <= 0 || tex.height <= 0 || tex.pixels.empty()) return false;
        bytes.resize(static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height));
        for (size_t i = 0; i < tex.pixels.size(); ++i) {
            switch (channel) {
                case 1:  bytes[i] = tex.pixels[i].g; break;
                case 2:  bytes[i] = tex.pixels[i].b; break;
                case 3:  bytes[i] = tex.pixels[i].a; break;
                default: bytes[i] = tex.pixels[i].r; break;
            }
        }

        image.width = static_cast<size_t>(tex.width);
        image.height = static_cast<size_t>(tex.height);
        image.format = DXGI_FORMAT_R8_UNORM;
        image.rowPitch = static_cast<size_t>(tex.width);
        image.slicePitch = bytes.size();
        image.pixels = bytes.data();
        return true;
    }

    // BC5: extract only RG channels (normal map XY — Z reconstructed in shader)
    bool buildNormalSourceImage(const Texture& tex, std::vector<uint8_t>& bytes, DirectX::Image& image) {
        if (tex.width <= 0 || tex.height <= 0 || tex.pixels.empty()) return false;
        bytes.resize(static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height) * 2ull);
        for (size_t i = 0; i < tex.pixels.size(); ++i) {
            bytes[i * 2 + 0] = tex.pixels[i].r;
            bytes[i * 2 + 1] = tex.pixels[i].g;
        }
        image.width      = static_cast<size_t>(tex.width);
        image.height     = static_cast<size_t>(tex.height);
        image.format     = DXGI_FORMAT_R8G8_UNORM;
        image.rowPitch   = static_cast<size_t>(tex.width) * 2ull;
        image.slicePitch = bytes.size();
        image.pixels     = bytes.data();
        return true;
    }

    // srgb=true: tells DirectXTex the source data is sRGB-encoded, keeping it
    // consistent with BC7_UNORM_SRGB on the Vulkan side (prevents double-gamma shift).
    bool buildColorSourceImage(const Texture& tex, std::vector<uint8_t>& bytes, DirectX::Image& image, bool srgb) {
        if (tex.width <= 0 || tex.height <= 0 || tex.pixels.empty()) return false;
        bytes.resize(static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height) * 4ull);
        for (size_t i = 0; i < tex.pixels.size(); ++i) {
            bytes[i * 4 + 0] = tex.pixels[i].r;
            bytes[i * 4 + 1] = tex.pixels[i].g;
            bytes[i * 4 + 2] = tex.pixels[i].b;
            bytes[i * 4 + 3] = tex.pixels[i].a;
        }

        image.width      = static_cast<size_t>(tex.width);
        image.height     = static_cast<size_t>(tex.height);
        image.format     = srgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;
        image.rowPitch   = static_cast<size_t>(tex.width) * 4ull;
        image.slicePitch = bytes.size();
        image.pixels     = bytes.data();
        return true;
    }

    bool buildCompressedCacheFile(
        const Texture& tex,
        TextureCompressionTarget target,
        bool srgb,
        uint8_t sourceChannel,
        const fs::path& outputPath,
        std::string* outReason)
    {
        const DXGI_FORMAT targetFormat = dxgiFormatForTarget(target, srgb);
        if (targetFormat == DXGI_FORMAT_UNKNOWN) {
            if (outReason) *outReason = "Unsupported compression target.";
            return false;
        }

        std::vector<uint8_t> sourceBytes;
        DirectX::Image sourceImage{};
        bool builtSourceImage = false;
        if (target == TextureCompressionTarget::BC4) {
            builtSourceImage = buildScalarSourceImage(tex, sourceBytes, sourceImage, sourceChannel);
        } else if (target == TextureCompressionTarget::BC5) {
            builtSourceImage = buildNormalSourceImage(tex, sourceBytes, sourceImage);
        } else {
            builtSourceImage = buildColorSourceImage(tex, sourceBytes, sourceImage, srgb);
        }
        if (!builtSourceImage) {
            if (outReason) *outReason = "Texture pixels are not available for cache generation.";
            return false;
        }

        DirectX::TexMetadata metadata{};
        metadata.width = sourceImage.width;
        metadata.height = sourceImage.height;
        metadata.depth = 1;
        metadata.arraySize = 1;
        metadata.mipLevels = 1;
        metadata.miscFlags = 0;
        metadata.miscFlags2 = 0;
        metadata.dimension = DirectX::TEX_DIMENSION_TEXTURE2D;
        metadata.format = sourceImage.format;

        // BC7_QUICK: ~10x faster encoder. PARALLEL: uses all cores within one texture.
        // Bake thread limits to 2 concurrent calls so total threads stay reasonable.
        const DirectX::TEX_COMPRESS_FLAGS compressFlags =
            (target == TextureCompressionTarget::BC7)
                ? static_cast<DirectX::TEX_COMPRESS_FLAGS>(DirectX::TEX_COMPRESS_BC7_QUICK | DirectX::TEX_COMPRESS_PARALLEL)
                : DirectX::TEX_COMPRESS_PARALLEL;

        DirectX::ScratchImage compressedImage;
        const HRESULT hr = DirectX::Compress(
            &sourceImage,
            1,
            metadata,
            targetFormat,
            compressFlags,
            1.0f,
            compressedImage);
        if (FAILED(hr)) {
            if (outReason) {
                *outReason = "DirectXTex compression failed (HRESULT=" + std::to_string(static_cast<unsigned long>(hr)) + ").";
            }
            return false;
        }

        try {
            fs::create_directories(outputPath.parent_path());
        } catch (const std::exception& e) {
            if (outReason) *outReason = "Failed to create cache directory: " + std::string(e.what());
            return false;
        }

        const HRESULT saveHr = DirectX::SaveToDDSFile(
            compressedImage.GetImages(),
            compressedImage.GetImageCount(),
            compressedImage.GetMetadata(),
            DirectX::DDS_FLAGS_NONE,
            outputPath.wstring().c_str());
        if (FAILED(saveHr)) {
            if (outReason) {
                *outReason = "Failed to save DDS cache (HRESULT=" + std::to_string(static_cast<unsigned long>(saveHr)) + ").";
            }
            return false;
        }

        return true;
    }
}

std::optional<std::filesystem::path> getProjectTextureCacheDirectory() {
    return getManagedProjectTextureCacheDirectory();
}

bool clearProjectTextureCache(std::string* outReason) {
    const auto cacheDir = getManagedProjectTextureCacheDirectory();
    if (!cacheDir) {
        if (outReason) *outReason = "No active saved project path is available.";
        return false;
    }

    try {
        if (!fs::exists(*cacheDir)) {
            if (outReason) *outReason = "Project cache folder does not exist yet.";
            return true;
        }
        fs::remove_all(cacheDir->parent_path());
        return true;
    } catch (const std::exception& e) {
        if (outReason) *outReason = std::string("Failed to clear project cache: ") + e.what();
        return false;
    }
}

std::optional<TextureCompressedCacheCandidate> findCompressedTextureCacheCandidate(
    const Texture& tex,
    TextureType type,
    bool srgb)
{
    const TextureCompressionPlan plan = buildTextureCompressionPlan(&tex, type);
    if (plan.preferredTarget == TextureCompressionTarget::None) return std::nullopt;

    if (auto adjacent = findAdjacentDDSCandidate(tex, plan.preferredTarget, srgb)) {
        return adjacent;
    }

    if (auto managed = makeManagedCacheCandidate(tex, type, srgb, plan.preferredTarget, plan.sourceChannel)) {
        if (fs::exists(managed->ddsPath)) {
            return managed;
        }
    }

    return std::nullopt;
}

bool tryBuildCompressedTextureCache(
    const Texture& tex,
    TextureType type,
    bool srgb,
    TextureCompressedCacheCandidate& outCandidate,
    std::string* outReason)
{
    const TextureCompressionPlan plan = buildTextureCompressionPlan(&tex, type);
    if (plan.preferredTarget == TextureCompressionTarget::None) {
        if (outReason) *outReason = "Texture type is not eligible for compression cache.";
        return false;
    }

    auto managed = makeManagedCacheCandidate(tex, type, srgb, plan.preferredTarget, plan.sourceChannel);
    if (!managed) {
        if (outReason) *outReason = "Managed project cache path is unavailable for this texture.";
        return false;
    }

    if (fs::exists(managed->ddsPath)) {
        outCandidate = *managed;
        if (outReason) *outReason = "Compressed cache already exists.";
        return true;
    }

    if (!buildCompressedCacheFile(tex, plan.preferredTarget, srgb, plan.sourceChannel, managed->ddsPath, outReason)) {
        return false;
    }

    outCandidate = *managed;
    return true;
}

std::string queryManagedCacheTag(const Texture& tex, TextureType type) {
    const TextureCompressionPlan plan = buildTextureCompressionPlan(&tex, type);
    if (plan.preferredTarget == TextureCompressionTarget::None) return {};

    for (const bool srgb : {false, true}) {
        const auto c = makeManagedCacheCandidate(tex, type, srgb, plan.preferredTarget, plan.sourceChannel);
        if (c && fs::exists(c->ddsPath)) {
            switch (plan.preferredTarget) {
                case TextureCompressionTarget::BC7: return "BC7";
                case TextureCompressionTarget::BC5: return "BC5";
                case TextureCompressionTarget::BC4: return "BC4";
                default: return {};
            }
        }
    }
    return {};
}
