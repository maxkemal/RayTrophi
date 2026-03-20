#include "TextureCompressionCache.h"
#include "ProjectManager.h"

#include <DirectXTex.h>

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
        TextureCompressionTarget target)
    {
        std::ostringstream oss;
        oss << tex.name << '|'
            << tex.width << 'x' << tex.height << '|'
            << static_cast<int>(type) << '|'
            << static_cast<int>(target) << '|'
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
        TextureCompressionTarget target)
    {
        if (tex.name.empty() || target == TextureCompressionTarget::None) return std::nullopt;

        const auto cacheDir = getManagedProjectTextureCacheDirectory();
        if (!cacheDir) return std::nullopt;

        const uint64_t identity = hashTextureCacheIdentity(tex, type, srgb, target);
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

        TextureCompressedCacheCandidate candidate;
        candidate.ddsPath = ddsPath;
        candidate.target = target;
        candidate.srgb = srgb;
        return candidate;
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

    bool buildScalarSourceImage(const Texture& tex, std::vector<uint8_t>& bytes, DirectX::Image& image) {
        if (tex.width <= 0 || tex.height <= 0 || tex.pixels.empty()) return false;
        bytes.resize(static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height));
        for (size_t i = 0; i < tex.pixels.size(); ++i) {
            bytes[i] = tex.pixels[i].r;
        }

        image.width = static_cast<size_t>(tex.width);
        image.height = static_cast<size_t>(tex.height);
        image.format = DXGI_FORMAT_R8_UNORM;
        image.rowPitch = static_cast<size_t>(tex.width);
        image.slicePitch = bytes.size();
        image.pixels = bytes.data();
        return true;
    }

    bool buildColorSourceImage(const Texture& tex, std::vector<uint8_t>& bytes, DirectX::Image& image) {
        if (tex.width <= 0 || tex.height <= 0 || tex.pixels.empty()) return false;
        bytes.resize(static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height) * 4ull);
        for (size_t i = 0; i < tex.pixels.size(); ++i) {
            bytes[i * 4 + 0] = tex.pixels[i].r;
            bytes[i * 4 + 1] = tex.pixels[i].g;
            bytes[i * 4 + 2] = tex.pixels[i].b;
            bytes[i * 4 + 3] = tex.pixels[i].a;
        }

        image.width = static_cast<size_t>(tex.width);
        image.height = static_cast<size_t>(tex.height);
        image.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        image.rowPitch = static_cast<size_t>(tex.width) * 4ull;
        image.slicePitch = bytes.size();
        image.pixels = bytes.data();
        return true;
    }

    bool buildCompressedCacheFile(
        const Texture& tex,
        TextureCompressionTarget target,
        bool srgb,
        const fs::path& outputPath,
        std::string* outReason)
    {
        const DXGI_FORMAT targetFormat = dxgiFormatForTarget(target, srgb);
        if (targetFormat == DXGI_FORMAT_UNKNOWN) {
            if (outReason) *outReason = "Unsupported compression target.";
            return false;
        }

        if (target == TextureCompressionTarget::BC5) {
            if (outReason) *outReason = "BC5 normal cache generation is not enabled yet.";
            return false;
        }

        std::vector<uint8_t> sourceBytes;
        DirectX::Image sourceImage{};
        const bool builtSourceImage =
            (target == TextureCompressionTarget::BC4)
                ? buildScalarSourceImage(tex, sourceBytes, sourceImage)
                : buildColorSourceImage(tex, sourceBytes, sourceImage);
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

        DirectX::ScratchImage compressedImage;
        const HRESULT hr = DirectX::Compress(
            &sourceImage,
            1,
            metadata,
            targetFormat,
            DirectX::TEX_COMPRESS_PARALLEL,
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

    if (auto managed = makeManagedCacheCandidate(tex, type, srgb, plan.preferredTarget)) {
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

    auto managed = makeManagedCacheCandidate(tex, type, srgb, plan.preferredTarget);
    if (!managed) {
        if (outReason) *outReason = "Managed project cache path is unavailable for this texture.";
        return false;
    }

    if (fs::exists(managed->ddsPath)) {
        outCandidate = *managed;
        if (outReason) *outReason = "Compressed cache already exists.";
        return true;
    }

    if (!buildCompressedCacheFile(tex, plan.preferredTarget, srgb, managed->ddsPath, outReason)) {
        return false;
    }

    outCandidate = *managed;
    return true;
}
