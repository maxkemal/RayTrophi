#include "AssetRegistry.h"
#include "globals.h"
#include "VDBVolume.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <set>
#include <unordered_set>
#include <windows.h>

namespace {

bool isSupportedModelExtension(const std::filesystem::path& path) {
    const std::string ext = AssetRegistry::toLowerCopy(path.extension().string());
    return ext == ".glb" || ext == ".gltf" || ext == ".fbx" || ext == ".obj";
}

bool isSupportedVolumeExtension(const std::filesystem::path& path) {
    const std::string ext = AssetRegistry::toLowerCopy(path.extension().string());
    return ext == ".vdb" || ext == ".nvdb";
}

bool isSupportedAssetExtension(const std::filesystem::path& path) {
    return isSupportedModelExtension(path) || isSupportedVolumeExtension(path);
}

bool isPreviewExtension(const std::filesystem::path& path) {
    const std::string ext = AssetRegistry::toLowerCopy(path.extension().string());
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp";
}

bool readBinaryFileBytes(const std::filesystem::path& path, std::vector<unsigned char>& out_bytes) {
    out_bytes.clear();

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return false;
    }

    out_bytes.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    return file.read(reinterpret_cast<char*>(out_bytes.data()), size).good();
}

std::string pathToUtf8(const std::filesystem::path& path) {
#ifdef _WIN32
    const std::wstring wide = path.native();
    if (wide.empty()) {
        return {};
    }
    const int size_needed = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (size_needed <= 0) {
        return path.string();
    }
    std::string utf8(static_cast<std::size_t>(size_needed), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, utf8.data(), size_needed, nullptr, nullptr);
    utf8.resize(static_cast<std::size_t>(size_needed - 1));
    return utf8;
#else
    return path.string();
#endif
}

bool pathExists(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

bool isDirectoryPath(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

bool isRegularFilePath(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

std::filesystem::path getExecutableDirectory() {
    std::wstring buffer(MAX_PATH, L'\0');
    DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    while (length >= buffer.size() - 1) {
        buffer.resize(buffer.size() * 2);
        length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    }

    if (length == 0) {
        return {};
    }

    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path();
}

std::uint64_t countMaterialTextureRefs(const aiMaterial* material) {
    if (!material) {
        return 0;
    }

    static const aiTextureType texture_types[] = {
        aiTextureType_DIFFUSE,
        aiTextureType_SPECULAR,
        aiTextureType_AMBIENT,
        aiTextureType_EMISSIVE,
        aiTextureType_HEIGHT,
        aiTextureType_NORMALS,
        aiTextureType_SHININESS,
        aiTextureType_OPACITY,
        aiTextureType_DISPLACEMENT,
        aiTextureType_LIGHTMAP,
        aiTextureType_REFLECTION,
        aiTextureType_BASE_COLOR,
        aiTextureType_NORMAL_CAMERA,
        aiTextureType_EMISSION_COLOR,
        aiTextureType_METALNESS,
        aiTextureType_DIFFUSE_ROUGHNESS,
        aiTextureType_AMBIENT_OCCLUSION
    };

    std::uint64_t count = 0;
    for (aiTextureType type : texture_types) {
        count += material->GetTextureCount(type);
    }
    return count;
}

std::string safeReadString(const nlohmann::json& json, const char* key, const std::string& fallback = "") {
    if (!json.contains(key) || !json[key].is_string()) {
        return fallback;
    }
    return json[key].get<std::string>();
}

std::vector<std::string> safeReadStringArray(const nlohmann::json& json, const char* key) {
    std::vector<std::string> result;
    if (!json.contains(key) || !json[key].is_array()) {
        return result;
    }
    for (const auto& value : json[key]) {
        if (value.is_string()) {
            result.push_back(value.get<std::string>());
        }
    }
    return result;
}

struct SequenceScanInfo {
    bool is_sequence = false;
    std::string pattern;
    int start_frame = 0;
    int end_frame = 0;
    int digit_count = 0;
    std::filesystem::path representative_path;
};

bool parseTrailingFrameNumber(const std::filesystem::path& path,
                              std::string& out_prefix,
                              std::string& out_suffix,
                              int& out_frame,
                              int& out_digit_count) {
    const std::string stem = path.stem().string();
    std::size_t last_digit = std::string::npos;
    std::size_t first_digit = std::string::npos;

    for (std::size_t i = stem.size(); i > 0; --i) {
        const char ch = stem[i - 1];
        if (std::isdigit(static_cast<unsigned char>(ch))) {
            if (last_digit == std::string::npos) {
                last_digit = i - 1;
            }
            first_digit = i - 1;
        } else if (last_digit != std::string::npos) {
            break;
        }
    }

    if (last_digit == std::string::npos || first_digit == std::string::npos) {
        return false;
    }

    const std::string frame_text = stem.substr(first_digit, last_digit - first_digit + 1);
    if (frame_text.empty()) {
        return false;
    }

    out_prefix = stem.substr(0, first_digit);
    out_suffix = stem.substr(last_digit + 1);
    out_digit_count = static_cast<int>(frame_text.size());
    out_frame = std::atoi(frame_text.c_str());
    return true;
}

SequenceScanInfo scanVolumeSequence(const std::filesystem::path& entry_path) {
    SequenceScanInfo info;
    if (!isSupportedVolumeExtension(entry_path)) {
        return info;
    }

    std::string prefix;
    std::string suffix;
    int frame = 0;
    int digit_count = 0;
    if (!parseTrailingFrameNumber(entry_path, prefix, suffix, frame, digit_count)) {
        return info;
    }

    int min_frame = frame;
    int max_frame = frame;
    int match_count = 0;
    const std::filesystem::path directory = entry_path.parent_path();
    const std::string ext = AssetRegistry::toLowerCopy(entry_path.extension().string());

    std::error_code ec;
    for (std::filesystem::directory_iterator it(directory, ec), end; it != end; it.increment(ec)) {
        if (ec || !it->is_regular_file()) {
            continue;
        }

        const std::filesystem::path candidate = it->path();
        if (AssetRegistry::toLowerCopy(candidate.extension().string()) != ext) {
            continue;
        }

        std::string candidate_prefix;
        std::string candidate_suffix;
        int candidate_frame = 0;
        int candidate_digits = 0;
        if (!parseTrailingFrameNumber(candidate, candidate_prefix, candidate_suffix, candidate_frame, candidate_digits)) {
            continue;
        }

        if (candidate_prefix != prefix || candidate_suffix != suffix) {
            continue;
        }

        ++match_count;
        min_frame = (std::min)(min_frame, candidate_frame);
        max_frame = (std::max)(max_frame, candidate_frame);
        digit_count = (std::max)(digit_count, candidate_digits);
    }

    if (match_count <= 1) {
        return info;
    }

    info.is_sequence = true;
    info.digit_count = digit_count;
    info.start_frame = min_frame;
    info.end_frame = max_frame;
    info.pattern = (directory / (prefix + std::string(static_cast<std::size_t>(digit_count), '#') + suffix + ext)).string();
    info.representative_path = entry_path;
    return info;
}

} // namespace

bool AssetRegistry::refresh(const std::filesystem::path& root_path) {
    m_root_path = root_path;
    m_assets.clear();

    std::error_code root_ec;
    if (m_root_path.empty() ||
        !std::filesystem::exists(m_root_path, root_ec) ||
        !std::filesystem::is_directory(m_root_path, root_ec)) {
        if (!m_root_path.empty()) {
            SCENE_LOG_WARN("[AssetRegistry] Invalid or inaccessible asset root: " + pathToUtf8(m_root_path));
        }
        return false;
    }

    std::error_code ec;
    const auto options = std::filesystem::directory_options::skip_permission_denied;
    for (std::filesystem::recursive_directory_iterator it(m_root_path, options, ec), end; it != end; it.increment(ec)) {
        if (ec) {
            SCENE_LOG_WARN("[AssetRegistry] Skipping folder due to access/path issue: " + pathToUtf8(it->path()) + " | error=" + ec.message());
            ec.clear();
            continue;
        }
        if (!it->is_directory()) {
            continue;
        }

        const std::filesystem::path directory_path = it->path();
        const std::vector<std::filesystem::path> entry_files = findEntryFiles(directory_path);
        if (entry_files.empty() && !pathExists(directory_path / "asset.json")) {
            continue;
        }

        for (const auto& entry_path : entry_files) {
            AssetRecord record = buildRecordFromFile(m_root_path, directory_path, entry_path);
            if (!record.entry_path.empty()) {
                m_assets.push_back(std::move(record));
            }
        }
    }

    std::sort(m_assets.begin(), m_assets.end(), [](const AssetRecord& a, const AssetRecord& b) {
        if (a.category != b.category) return a.category < b.category;
        if (a.subcategory != b.subcategory) return a.subcategory < b.subcategory;
        return a.name < b.name;
    });

    return true;
}

bool AssetRegistry::refresh() {
    return refresh(m_root_path.empty() ? resolveDefaultRoot() : m_root_path);
}

std::vector<std::string> AssetRegistry::getCategories() const {
    std::set<std::string> categories;
    for (const auto& asset : m_assets) {
        if (!asset.category.empty()) {
            categories.insert(asset.category);
        }
    }
    return std::vector<std::string>(categories.begin(), categories.end());
}

const AssetRecord* AssetRegistry::findByRelativeDirectory(const std::string& relative_directory) const {
    for (const auto& asset : m_assets) {
        if (asset.relative_entry_path.generic_string() == relative_directory) {
            return &asset;
        }
    }
    return nullptr;
}

bool AssetRegistry::writeMetadataStub(const AssetRecord& asset) const {
    if (asset.directory_path.empty() || asset.metadata_path.empty()) {
        return false;
    }

    std::ofstream file(asset.metadata_path);
    if (!file.is_open()) {
        SCENE_LOG_WARN("[AssetRegistry] Failed to write metadata stub: " + pathToUtf8(asset.metadata_path));
        return false;
    }

    file << buildMetadataJson(asset).dump(2);
    return true;
}

std::filesystem::path AssetRegistry::resolveDefaultRoot() {
    const std::filesystem::path exe_dir = getExecutableDirectory();
    const std::vector<std::filesystem::path> candidates = {
        exe_dir / "assets",
        exe_dir / ".." / "assets",
        exe_dir / ".." / ".." / "raytrac_sdl2" / "assets",
        "assets",
        "../assets",
        "raytrac_sdl2/assets",
        "../../raytrac_sdl2/assets"
    };

    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && std::filesystem::is_directory(candidate, ec)) {
            return std::filesystem::absolute(candidate, ec);
        }
    }
    return {};
}

AssetRecord AssetRegistry::buildRecordFromFile(const std::filesystem::path& root_path, const std::filesystem::path& directory_path, const std::filesystem::path& entry_path) {
    AssetRecord record;
    record.directory_path = directory_path;
    record.relative_directory = std::filesystem::relative(directory_path, root_path);
    record.entry_path = entry_path;
    record.relative_entry_path = std::filesystem::relative(entry_path, root_path);
    record.metadata_path = findMetadataFile(directory_path, entry_path);
    record.has_metadata = !record.metadata_path.empty() && pathExists(record.metadata_path);
    record.preview_path = findPreviewFile(directory_path, entry_path);
    record.has_preview = !record.preview_path.empty();

    if (!record.entry_path.empty()) {
        std::error_code ec;
        record.file_size_bytes = std::filesystem::file_size(record.entry_path, ec);
        record.format = toLowerCopy(record.entry_path.extension().string());
        if (!record.format.empty() && record.format[0] == '.') {
            record.format.erase(record.format.begin());
        }
    }

    const auto parts = record.relative_directory;
    auto it = parts.begin();
    if (it != parts.end()) {
        record.category = it->string();
        ++it;
    }
    if (it != parts.end()) {
        record.subcategory = it->string();
    }

    const std::string stem_name = !record.entry_path.empty() ? record.entry_path.stem().string() : directory_path.filename().string();
    record.id = stem_name;
    record.name = deriveDisplayName(record.id);
    record.type = (record.category == "scenes") ? "scene" : "asset";
    record.asset_kind = isSupportedVolumeExtension(record.entry_path) ? "vdb" : "model";
    const std::string lowered_stem = AssetRegistry::toLowerCopy(stem_name);
    const bool looks_like_anim_clip_name =
        lowered_stem == "anim" ||
        lowered_stem == "animation" ||
        lowered_stem.find("anim_") == 0 ||
        lowered_stem.find("_anim") != std::string::npos ||
        lowered_stem.find("animation") != std::string::npos ||
        lowered_stem.find("_clip") != std::string::npos ||
        lowered_stem.find("clip_") == 0;
    if (looks_like_anim_clip_name) {
        record.asset_kind = "anim_clip";
    }
    record.tags = deriveTags(record.relative_directory, stem_name);

    const SequenceScanInfo detected_sequence = scanVolumeSequence(record.entry_path);
    if (detected_sequence.is_sequence) {
        record.asset_kind = "vdb_sequence";
        record.is_sequence = true;
        record.sequence_pattern = detected_sequence.pattern;
        record.sequence_start_frame = detected_sequence.start_frame;
        record.sequence_end_frame = detected_sequence.end_frame;
        record.sequence_fps = 24.0f;
    }

    if (record.has_metadata) {
        try {
            std::ifstream file(record.metadata_path);
            nlohmann::json j;
            file >> j;

            record.id = safeReadString(j, "id", record.id);
            record.name = safeReadString(j, "name", record.name);
            record.type = safeReadString(j, "type", record.type);
            record.asset_kind = safeReadString(j, "assetKind", record.asset_kind);
            record.category = safeReadString(j, "category", record.category);
            record.subcategory = safeReadString(j, "subcategory", record.subcategory);
            record.format = safeReadString(j, "format", record.format);
            record.description = safeReadString(j, "description", record.description);
            record.license = safeReadString(j, "license", record.license);
            record.source = safeReadString(j, "source", record.source);
            record.author = safeReadString(j, "author", record.author);
            record.version = j.value("version", record.version);
            record.favorite = j.value("favorite", record.favorite);
            record.clip_mode = safeReadString(j, "clipMode", record.clip_mode);
            record.animation_binding = safeReadString(j, "animationBinding", record.animation_binding);
            record.sequence_fps = j.value("fps", record.sequence_fps);

            const std::string entry = safeReadString(j, "entry");
            if (!entry.empty()) {
                const std::filesystem::path candidate = directory_path / entry;
                if (pathExists(candidate)) {
                    record.entry_path = candidate;
                }
            }

            const std::string preview = safeReadString(j, "preview");
            if (!preview.empty()) {
                const std::filesystem::path candidate = directory_path / preview;
                if (pathExists(candidate)) {
                    record.preview_path = candidate;
                    record.has_preview = true;
                }
            }

            if (j.contains("sequence") && j["sequence"].is_object()) {
                const auto& seq = j["sequence"];
                record.is_sequence = seq.value("enabled", record.is_sequence);
                record.sequence_pattern = safeReadString(seq, "pattern", record.sequence_pattern);
                record.sequence_start_frame = seq.value("startFrame", record.sequence_start_frame);
                record.sequence_end_frame = seq.value("endFrame", record.sequence_end_frame);
                record.sequence_fps = seq.value("fps", record.sequence_fps);
                if (record.is_sequence && record.asset_kind == "vdb") {
                    record.asset_kind = "vdb_sequence";
                }
            }

            if (record.type == "vdb_sequence") {
                record.type = "asset";
                record.is_sequence = true;
                if (record.asset_kind == "vdb") {
                    record.asset_kind = "vdb_sequence";
                }
            }

            if (j.contains("tags") && j["tags"].is_array()) {
                record.tags.clear();
                for (const auto& tag : j["tags"]) {
                    if (tag.is_string()) {
                        record.tags.push_back(tag.get<std::string>());
                    }
                }
            }

            if (j.contains("metrics") && j["metrics"].is_object()) {
                const auto& metrics = j["metrics"];
                record.animation_clip_count = metrics.value("animationClipCount", record.animation_clip_count);
            }

            if (j.contains("grids") && j["grids"].is_object()) {
                const auto& grids = j["grids"];
                record.vdb_grids = safeReadStringArray(grids, "available");
                record.vdb_primary_grid = safeReadString(grids, "primary", record.vdb_primary_grid);
            }

            if (j.contains("bounds") && j["bounds"].is_object()) {
                const auto& bounds = j["bounds"];
                record.vdb_voxel_size = bounds.value("voxelSize", record.vdb_voxel_size);
                if (bounds.contains("min") && bounds["min"].is_array() && bounds["min"].size() >= 3 &&
                    bounds.contains("max") && bounds["max"].is_array() && bounds["max"].size() >= 3) {
                    record.vdb_has_bounds = true;
                    record.vdb_bounds_min_x = bounds["min"][0].get<double>();
                    record.vdb_bounds_min_y = bounds["min"][1].get<double>();
                    record.vdb_bounds_min_z = bounds["min"][2].get<double>();
                    record.vdb_bounds_max_x = bounds["max"][0].get<double>();
                    record.vdb_bounds_max_y = bounds["max"][1].get<double>();
                    record.vdb_bounds_max_z = bounds["max"][2].get<double>();
                }
            }

            if (j.contains("renderHints") && j["renderHints"].is_object()) {
                const auto& hints = j["renderHints"];
                record.vdb_is_fire = hints.value("isFire", record.vdb_is_fire);
                record.vdb_is_smoke = hints.value("isSmoke", record.vdb_is_smoke);
                record.vdb_has_velocity = hints.value("hasVelocity", record.vdb_has_velocity);
            }

            if (j.contains("shaderDefaults") && j["shaderDefaults"].is_object()) {
                const auto& shader = j["shaderDefaults"];
                record.vdb_shader_preset = safeReadString(shader, "preset", record.vdb_shader_preset);
                record.vdb_density_multiplier = shader.value("densityMultiplier", record.vdb_density_multiplier);
                record.vdb_temperature_scale = shader.value("temperatureScale", record.vdb_temperature_scale);
                record.vdb_emission_intensity = shader.value("emissionIntensity", record.vdb_emission_intensity);
            }
        } catch (const std::exception& e) {
            SCENE_LOG_WARN("[AssetRegistry] Metadata parse failed: " + pathToUtf8(record.metadata_path) + " | " + e.what());
            record.has_metadata = false;
        } catch (...) {
            SCENE_LOG_WARN("[AssetRegistry] Metadata parse failed: " + pathToUtf8(record.metadata_path));
            record.has_metadata = false;
        }
    }

    if (record.description.empty()) {
        record.description = record.has_metadata ? "" : "Auto-discovered from asset folder.";
    }

    if (record.animation_clip_count == 0 && record.asset_kind == "anim_clip") {
        record.animation_clip_count = 1;
    }

    if ((record.asset_kind == "vdb" || record.asset_kind == "vdb_sequence") && record.vdb_grids.empty()) {
        const AssetAnalysisInfo analysis = analyzeAssetFile(record.entry_path);
        record.vdb_grids = analysis.vdb_grids;
        record.vdb_primary_grid = analysis.vdb_primary_grid;
        record.vdb_voxel_size = analysis.vdb_voxel_size;
        record.vdb_has_bounds = analysis.vdb_has_bounds;
        record.vdb_bounds_min_x = analysis.vdb_bounds_min_x;
        record.vdb_bounds_min_y = analysis.vdb_bounds_min_y;
        record.vdb_bounds_min_z = analysis.vdb_bounds_min_z;
        record.vdb_bounds_max_x = analysis.vdb_bounds_max_x;
        record.vdb_bounds_max_y = analysis.vdb_bounds_max_y;
        record.vdb_bounds_max_z = analysis.vdb_bounds_max_z;
        record.vdb_is_fire = analysis.vdb_is_fire;
        record.vdb_is_smoke = analysis.vdb_is_smoke;
        record.vdb_has_velocity = analysis.vdb_has_velocity;

        if (record.vdb_shader_preset == "auto") {
            if (record.vdb_is_fire) {
                record.vdb_shader_preset = "fire";
                record.vdb_emission_intensity = 15.0f;
            } else {
                record.vdb_shader_preset = "smoke";
            }
        }
    }

    return record;
}

std::vector<std::filesystem::path> AssetRegistry::findEntryFiles(const std::filesystem::path& directory_path) {
    const std::vector<std::string> preferred = { "scene.glb", "model.glb", "scene.gltf", "model.gltf" };
    std::vector<std::filesystem::path> candidates;
    for (const auto& filename : preferred) {
        const std::filesystem::path candidate = directory_path / filename;
        if (pathExists(candidate)) {
            candidates.push_back(candidate);
        }
    }

    std::error_code ec;
    std::unordered_set<std::string> seen_volume_sequences;
    const auto options = std::filesystem::directory_options::skip_permission_denied;
    for (std::filesystem::directory_iterator it(directory_path, options, ec), end; it != end; it.increment(ec)) {
        if (ec) {
            SCENE_LOG_WARN("[AssetRegistry] Skipping file scan in folder: " + pathToUtf8(directory_path) + " | error=" + ec.message());
            ec.clear();
            continue;
        }
        if (!it->is_regular_file()) {
            continue;
        }
        if (!isSupportedAssetExtension(it->path())) {
            continue;
        }

        if (isSupportedVolumeExtension(it->path())) {
            const SequenceScanInfo sequence = scanVolumeSequence(it->path());
            if (sequence.is_sequence) {
                const std::string sequence_key =
                    sequence.representative_path.parent_path().string() + "|" +
                    sequence.representative_path.extension().string() + "|" +
                    sequence.pattern;
                if (!seen_volume_sequences.insert(sequence_key).second) {
                    continue;
                }
            }
        }

        if (std::find(candidates.begin(), candidates.end(), it->path()) == candidates.end()) {
            candidates.push_back(it->path());
        }
    }

    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

std::filesystem::path AssetRegistry::findPreviewFile(const std::filesystem::path& directory_path, const std::filesystem::path& entry_path) {
    const std::string stem = entry_path.stem().string();
    const std::vector<std::string> per_file_preferred = {
        stem + ".png",
        stem + ".jpg",
        stem + ".jpeg",
        stem + ".webp",
        stem + ".preview.png",
        stem + ".preview.jpg",
        stem + ".preview.jpeg",
        stem + ".preview.webp",
        stem + ".thumbnail.png",
        stem + ".thumbnail.jpg",
        stem + ".thumbnail.jpeg",
        stem + ".thumbnail.webp"
    };
    for (const auto& filename : per_file_preferred) {
        const std::filesystem::path candidate = directory_path / filename;
        if (pathExists(candidate)) {
            return candidate;
        }
    }

    const std::vector<std::filesystem::path> entry_files = findEntryFiles(directory_path);
    if (entry_files.size() > 1) {
        return {};
    }

    const std::vector<std::string> shared_preferred = {
        "preview.png",
        "preview.jpg",
        "preview.jpeg",
        "preview.webp",
        "thumbnail.png",
        "thumbnail.jpg",
        "thumbnail.jpeg",
        "thumbnail.webp"
    };
    for (const auto& filename : shared_preferred) {
        const std::filesystem::path candidate = directory_path / filename;
        if (pathExists(candidate)) {
            return candidate;
        }
    }

    return {};
}

std::filesystem::path AssetRegistry::findMetadataFile(const std::filesystem::path& directory_path, const std::filesystem::path& entry_path) {
    const std::filesystem::path per_file_metadata = directory_path / (entry_path.stem().string() + ".asset.json");
    if (pathExists(per_file_metadata)) {
        return per_file_metadata;
    }

    const std::vector<std::filesystem::path> entry_files = findEntryFiles(directory_path);
    if (entry_files.size() <= 1) {
        const std::filesystem::path legacy_metadata = directory_path / "asset.json";
        if (pathExists(legacy_metadata)) {
            return legacy_metadata;
        }
    }

    return per_file_metadata;
}

std::string AssetRegistry::deriveDisplayName(const std::string& raw_name) {
    std::string out;
    out.reserve(raw_name.size());
    bool capitalize = true;
    for (char ch : raw_name) {
        if (ch == '_' || ch == '-') {
            out.push_back(' ');
            capitalize = true;
            continue;
        }

        if (capitalize) {
            out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
            capitalize = false;
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::vector<std::string> AssetRegistry::deriveTags(const std::filesystem::path& relative_directory, const std::string& stem_name) {
    std::vector<std::string> tags;
    std::unordered_set<std::string> seen;

    for (const auto& part : relative_directory) {
        const std::string value = toLowerCopy(part.string());
        if (!value.empty() && seen.insert(value).second) {
            tags.push_back(value);
        }
    }

    std::string token;
    for (char ch : stem_name) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            token.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        } else if (!token.empty()) {
            if (seen.insert(token).second) {
                tags.push_back(token);
            }
            token.clear();
        }
    }
    if (!token.empty() && seen.insert(token).second) {
        tags.push_back(token);
    }

    return tags;
}

std::string AssetRegistry::toLowerCopy(const std::string& value) {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

bool AssetRegistry::hasSupportedAssetFile(const std::filesystem::path& directory_path) {
    std::error_code ec;
    const auto options = std::filesystem::directory_options::skip_permission_denied;
    for (std::filesystem::directory_iterator it(directory_path, options, ec), end; it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        if (!it->is_regular_file()) {
            continue;
        }
        if (isSupportedAssetExtension(it->path())) {
            return true;
        }
    }
    return false;
}

nlohmann::json AssetRegistry::buildMetadataJson(const AssetRecord& asset) {
    nlohmann::json j;
    const AssetAnalysisInfo analysis = analyzeAssetFile(asset.entry_path);

    j["id"] = asset.id;
    j["name"] = asset.name;
    j["type"] = asset.type;
    j["assetKind"] = asset.asset_kind.empty() ? "asset" : asset.asset_kind;
    j["category"] = asset.category;
    if (!asset.subcategory.empty()) {
        j["subcategory"] = asset.subcategory;
    }
    j["format"] = asset.format;
    j["entry"] = asset.entry_path.filename().string();
    if (asset.has_preview) {
        j["preview"] = asset.preview_path.filename().string();
    }
    j["description"] = asset.description;
    j["license"] = asset.license.empty() ? "unknown" : asset.license;
    j["source"] = asset.source.empty() ? "manual" : asset.source;
    if (!asset.author.empty()) {
        j["author"] = asset.author;
    }
    j["tags"] = asset.tags;
    j["favorite"] = asset.favorite;
    j["version"] = asset.version;
    j["fileSizeBytes"] = asset.file_size_bytes;

    if (asset.is_sequence) {
        j["sequence"] = {
            { "enabled", true },
            { "pattern", asset.sequence_pattern },
            { "startFrame", asset.sequence_start_frame },
            { "endFrame", asset.sequence_end_frame },
            { "fps", asset.sequence_fps }
        };
    }

    if (asset.asset_kind == "anim_clip") {
        j["clipMode"] = asset.clip_mode.empty() ? "skeletal" : asset.clip_mode;
        j["animationBinding"] = asset.animation_binding.empty() ? "selected-model" : asset.animation_binding;
    }

    if (asset.asset_kind == "vdb" || asset.asset_kind == "vdb_sequence") {
        j["grids"] = {
            { "available", asset.vdb_grids },
            { "primary", asset.vdb_primary_grid.empty() ? "density" : asset.vdb_primary_grid }
        };
        j["bounds"] = {
            { "voxelSize", asset.vdb_voxel_size },
            { "min", { asset.vdb_bounds_min_x, asset.vdb_bounds_min_y, asset.vdb_bounds_min_z } },
            { "max", { asset.vdb_bounds_max_x, asset.vdb_bounds_max_y, asset.vdb_bounds_max_z } }
        };
        j["renderHints"] = {
            { "isFire", asset.vdb_is_fire },
            { "isSmoke", asset.vdb_is_smoke },
            { "hasVelocity", asset.vdb_has_velocity }
        };
        j["shaderDefaults"] = {
            { "preset", asset.vdb_shader_preset.empty() ? "auto" : asset.vdb_shader_preset },
            { "densityMultiplier", asset.vdb_density_multiplier },
            { "temperatureScale", asset.vdb_temperature_scale },
            { "emissionIntensity", asset.vdb_emission_intensity }
        };
        if (!asset.is_sequence) {
            j["fps"] = asset.sequence_fps;
        }
    }

    nlohmann::json metrics;
    metrics["triangleCount"] = analysis.triangle_count;
    metrics["meshCount"] = analysis.mesh_count;
    metrics["materialCount"] = analysis.material_count;
    metrics["textureReferenceCount"] = analysis.texture_reference_count;
    metrics["animationClipCount"] = analysis.animation_clip_count;
    j["metrics"] = metrics;

    if (analysis.has_dimensions) {
        j["dimensions"] = {
            { "width", analysis.width },
            { "height", analysis.height },
            { "depth", analysis.depth }
        };
        j["placementPivot"] = {
            { "x", analysis.pivot_x },
            { "y", analysis.pivot_y },
            { "z", analysis.pivot_z },
            { "mode", "bottom-center" }
        };
    }

    if (asset.source.empty()) {
        j["sourceTool"] = "unknown";
    }

    return j;
}

AssetAnalysisInfo AssetRegistry::analyzeAssetFile(const std::filesystem::path& entry_path) {
    AssetAnalysisInfo info;
    if (entry_path.empty() || !pathExists(entry_path)) {
        return info;
    }

    if (isSupportedVolumeExtension(entry_path)) {
        VDBVolume vdb;
        const bool loaded = vdb.loadVDBSequence(entry_path.string());
        if (!loaded) {
            return info;
        }

        info.vdb_grids = vdb.getAvailableGrids();
        info.vdb_primary_grid = info.vdb_grids.empty() ? "density" : info.vdb_grids.front();
        info.vdb_voxel_size = vdb.getVoxelSize();
        const Vec3 bmin = vdb.getLocalBoundsMin();
        const Vec3 bmax = vdb.getLocalBoundsMax();
        info.vdb_has_bounds = true;
        info.vdb_bounds_min_x = bmin.x;
        info.vdb_bounds_min_y = bmin.y;
        info.vdb_bounds_min_z = bmin.z;
        info.vdb_bounds_max_x = bmax.x;
        info.vdb_bounds_max_y = bmax.y;
        info.vdb_bounds_max_z = bmax.z;
        info.vdb_has_velocity = vdb.hasGrid("vel");
        info.vdb_is_fire = vdb.hasGrid("temperature");
        info.vdb_is_smoke = vdb.hasGrid("density");
        vdb.unload();
        return info;
    }

    Assimp::Importer importer;
    unsigned int import_flags =
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_ImproveCacheLocality;

    const std::string ext = toLowerCopy(entry_path.extension().string());
    if (ext == ".fbx") {
        import_flags |= aiProcess_GlobalScale;
    }

    std::vector<unsigned char> file_bytes;
    const aiScene* scene = nullptr;
    if (readBinaryFileBytes(entry_path, file_bytes)) {
        scene = importer.ReadFileFromMemory(file_bytes.data(), file_bytes.size(), import_flags, ext.c_str());
    }
    if (!scene) {
        scene = importer.ReadFile(entry_path.string(), import_flags);
    }
    if (!scene) {
        return info;
    }

    info.mesh_count = scene->mNumMeshes;
    info.material_count = scene->mNumMaterials;
    info.animation_clip_count = scene->mNumAnimations;

    aiVector3D bb_min(1e30f, 1e30f, 1e30f);
    aiVector3D bb_max(-1e30f, -1e30f, -1e30f);
    bool found_vertex = false;

    for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
        const aiMesh* mesh = scene->mMeshes[i];
        if (!mesh) {
            continue;
        }

        info.triangle_count += mesh->mNumFaces;
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            const aiVector3D& p = mesh->mVertices[v];
            bb_min.x = (std::min)(bb_min.x, p.x);
            bb_min.y = (std::min)(bb_min.y, p.y);
            bb_min.z = (std::min)(bb_min.z, p.z);
            bb_max.x = (std::max)(bb_max.x, p.x);
            bb_max.y = (std::max)(bb_max.y, p.y);
            bb_max.z = (std::max)(bb_max.z, p.z);
            found_vertex = true;
        }
    }

    for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
        info.texture_reference_count += countMaterialTextureRefs(scene->mMaterials[i]);
    }

    if (found_vertex) {
        info.has_dimensions = true;
        info.width = static_cast<double>(bb_max.x - bb_min.x);
        info.height = static_cast<double>(bb_max.y - bb_min.y);
        info.depth = static_cast<double>(bb_max.z - bb_min.z);
        info.pivot_x = static_cast<double>((bb_min.x + bb_max.x) * 0.5f);
        info.pivot_y = static_cast<double>(bb_min.y);
        info.pivot_z = static_cast<double>((bb_min.z + bb_max.z) * 0.5f);
    }

    return info;
}
