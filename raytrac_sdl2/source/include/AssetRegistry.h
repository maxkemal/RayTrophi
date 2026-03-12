#pragma once

#include "json.hpp"
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct AssetRecord {
    std::string id;
    std::string name;
    std::string type;
    std::string asset_kind;
    std::string category;
    std::string subcategory;
    std::string format;
    std::string description;
    std::string license;
    std::string source;
    std::string author;
    int version = 1;

    std::filesystem::path directory_path;
    std::filesystem::path entry_path;
    std::filesystem::path preview_path;
    std::filesystem::path metadata_path;
    std::filesystem::path relative_directory;
    std::filesystem::path relative_entry_path;
    std::vector<std::string> tags;
    bool favorite = false;

    std::uintmax_t file_size_bytes = 0;
    bool has_metadata = false;
    bool has_preview = false;
    bool is_sequence = false;
    std::string sequence_pattern;
    int sequence_start_frame = 0;
    int sequence_end_frame = 0;
    float sequence_fps = 24.0f;
    std::uint64_t animation_clip_count = 0;
    std::string clip_mode = "skeletal";
    std::string animation_binding = "selected-model";
    std::vector<std::string> vdb_grids;
    std::string vdb_primary_grid = "density";
    float vdb_voxel_size = 0.0f;
    bool vdb_has_bounds = false;
    double vdb_bounds_min_x = 0.0;
    double vdb_bounds_min_y = 0.0;
    double vdb_bounds_min_z = 0.0;
    double vdb_bounds_max_x = 0.0;
    double vdb_bounds_max_y = 0.0;
    double vdb_bounds_max_z = 0.0;
    bool vdb_is_fire = false;
    bool vdb_is_smoke = false;
    bool vdb_has_velocity = false;
    std::string vdb_shader_preset = "auto";
    float vdb_density_multiplier = 1.0f;
    float vdb_temperature_scale = 1.0f;
    float vdb_emission_intensity = 0.0f;
};

struct AssetAnalysisInfo {
    std::uint64_t triangle_count = 0;
    std::uint64_t mesh_count = 0;
    std::uint64_t material_count = 0;
    std::uint64_t texture_reference_count = 0;
    std::uint64_t animation_clip_count = 0;
    std::vector<std::string> vdb_grids;
    std::string vdb_primary_grid = "density";
    float vdb_voxel_size = 0.0f;
    bool vdb_has_bounds = false;
    double vdb_bounds_min_x = 0.0;
    double vdb_bounds_min_y = 0.0;
    double vdb_bounds_min_z = 0.0;
    double vdb_bounds_max_x = 0.0;
    double vdb_bounds_max_y = 0.0;
    double vdb_bounds_max_z = 0.0;
    bool vdb_is_fire = false;
    bool vdb_is_smoke = false;
    bool vdb_has_velocity = false;
    bool has_dimensions = false;
    double width = 0.0;
    double height = 0.0;
    double depth = 0.0;
    double pivot_x = 0.0;
    double pivot_y = 0.0;
    double pivot_z = 0.0;
};

class AssetRegistry {
public:
    bool refresh(const std::filesystem::path& root_path);
    bool refresh();

    const std::vector<AssetRecord>& getAssets() const { return m_assets; }
    const std::filesystem::path& getRootPath() const { return m_root_path; }

    std::vector<std::string> getCategories() const;
    const AssetRecord* findByRelativeDirectory(const std::string& relative_directory) const;
    bool writeMetadataStub(const AssetRecord& asset) const;

    static std::filesystem::path resolveDefaultRoot();

private:
    std::filesystem::path m_root_path;
    std::vector<AssetRecord> m_assets;

    static AssetRecord buildRecordFromFile(const std::filesystem::path& root_path, const std::filesystem::path& directory_path, const std::filesystem::path& entry_path);
    static std::vector<std::filesystem::path> findEntryFiles(const std::filesystem::path& directory_path);
    static std::filesystem::path findPreviewFile(const std::filesystem::path& directory_path, const std::filesystem::path& entry_path);
    static std::filesystem::path findMetadataFile(const std::filesystem::path& directory_path, const std::filesystem::path& entry_path);
    static std::string deriveDisplayName(const std::string& raw_name);
    static std::vector<std::string> deriveTags(const std::filesystem::path& relative_directory, const std::string& stem_name);
    static bool hasSupportedAssetFile(const std::filesystem::path& directory_path);
    static nlohmann::json buildMetadataJson(const AssetRecord& asset);
    static AssetAnalysisInfo analyzeAssetFile(const std::filesystem::path& entry_path);

public:
    static std::string toLowerCopy(const std::string& value);
};
