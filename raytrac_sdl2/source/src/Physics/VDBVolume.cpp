#include "VDBVolume.h"
#include "VDBVolumeManager.h"
#include "globals.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include "RayPacket.h"
#include "HitRecordPacket.h"

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════════

VDBVolume::VDBVolume() {
    transform_handle = std::make_shared<Transform>();
    world_transform = Matrix4x4::identity();
    world_transform_inv = Matrix4x4::identity();
    volume_shader = std::make_shared<VolumeShader>();
}

VDBVolume::VDBVolume(const std::string& vdb_path) : VDBVolume() {
    loadVDB(vdb_path);
}

bool VDBVolume::loadVDB(const std::string& path) {
    auto& mgr = VDBVolumeManager::getInstance();
    
    int id = mgr.loadVDB(path);
    if (id < 0) {
        SCENE_LOG_ERROR("VDBVolume: Failed to load " + path);
        return false;
    }
    
    vdb_volume_id = id;
    vdb_sequence_id = -1;
    is_sequence = false;
    filepath = path;
    
    // Extract name from filename
    name = std::filesystem::path(path).stem().string();
    
    // Update bounds from VDB data
    updateBoundsFromVDB();
    
    SCENE_LOG_INFO("VDBVolume: Loaded " + path + " (ID: " + std::to_string(id) + ")");
    
    // Auto-Rotate logic removed as per user request to avoid incorrect orientation for Y-up files.
    // if (name.find("Gas Volume") == std::string::npos && name.find("frame_") == std::string::npos) {
    //    setRotation(Vec3(-90.0f, 0.0f, 0.0f));
    // }
    
    return true;
}



void VDBVolume::updateFromTimeline(int timeline_frame, void* stream) {
    if (!is_sequence || !timeline_linked) return;
    
    // Calculate effective frame
    int relative_frame = timeline_frame + frame_offset;
    
    // Clamp to range
    if (relative_frame < sequence_start_frame) relative_frame = sequence_start_frame;
    if (relative_frame > sequence_end_frame) relative_frame = sequence_end_frame;
    
    if (relative_frame == current_frame) return; // No change
    
    // Replace placeholder with frame number
    std::string placeholder(sequence_digits, '#');
    std::string new_path = sequence_pattern;
    size_t hash_pos = new_path.find(placeholder);
    
    if (hash_pos != std::string::npos) {
        char buf[32];
        std::string fmt = "%0" + std::to_string(sequence_digits) + "d";
        snprintf(buf, 32, fmt.c_str(), relative_frame);
        new_path.replace(hash_pos, sequence_digits, buf);
    } else {
        SCENE_LOG_WARN("VDB pattern match failed: " + sequence_pattern);
        return; 
    }
    
    if (!std::filesystem::exists(new_path)) {
         SCENE_LOG_WARN("Missing VDB frame: " + new_path);
         return;
    }
    
    bool loaded = VDBVolumeManager::getInstance().updateVolume(vdb_volume_id, new_path, stream);
    if (loaded) {
        current_frame = relative_frame;
        filepath = new_path;
        updateBoundsFromVDB(); 
    } else {
        SCENE_LOG_ERROR("Failed to update VDB: " + new_path);
    }
}

bool VDBVolume::loadVDBSequence(const std::string& pattern_or_file) {
    // 1. If input is a specific file (e.g. explosion_0045.vdb), try to detect pattern
    std::filesystem::path path(pattern_or_file);
    std::string filename = path.filename().string();
    std::string directory = path.parent_path().string();
    if (directory.empty()) directory = ".";
    
    std::string stem = path.stem().string();
    std::string ext = path.extension().string();
    
    // Find last number in stem
    size_t last_digit = std::string::npos;
    size_t first_digit = std::string::npos;
    
    for (size_t i = stem.length(); i > 0; --i) {
        if (isdigit(stem[i-1])) {
            if (last_digit == std::string::npos) last_digit = i-1;
            first_digit = i-1;
        } else if (last_digit != std::string::npos) {
            break; 
        }
    }
    
    if (last_digit == std::string::npos) {
        // No number found, not a sequence
        return loadVDB(pattern_or_file);
    }
    
    // Extract base name and number length
    int num_len = (int)(last_digit - first_digit + 1);
    std::string prefix = stem.substr(0, first_digit);
    std::string suffix = stem.substr(last_digit + 1);
    
    // Store digit count
    sequence_digits = num_len;
    
    // Construct pattern for storage
    // Use #### convention, but adapted length
    std::string placeholder(sequence_digits, '#');
    std::string pattern_file = prefix + placeholder + suffix + ext;
    
    sequence_pattern = (std::filesystem::path(directory) / pattern_file).string();
    
    // 2. Scan directory for range
    int min_frame = 999999;
    int max_frame = -999999;
    
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string fname = entry.path().filename().string();
            // Check if matches prefix/suffix/ext
            if (fname.size() >= prefix.size() + suffix.size() + ext.size() + 1 && // +1 for at least 1 digit
                fname.substr(0, prefix.size()) == prefix &&
                fname.substr(fname.size() - ext.size()) == ext) {
                    
                // Extract number
                // ... Simplistic check
                // Better: Check extension
                if (entry.path().extension() == ext) {
                     // Check if it looks like the sequence
                     // Extract the number part
                      // ...
                      // This naive scan is tricky.
                      // Let's assume files are "explosion_0001.vdb"
                }
            }
        }
    }
    
    // SIMPLER: Assume we trust the user provided a file that is part of a sequence
    // We just try to find min/max by checking neighbor files?
    // Or just scan ALL files in dir and regex match?
    
    // Let's implement robust scan
    min_frame = INT_MAX;
    max_frame = INT_MIN;
    
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        
        // Must end with extension
        if (name.length() < ext.length() || name.substr(name.length() - ext.length()) != ext) continue;
        
        // Must start with prefix
        if (name.length() < prefix.length() || name.substr(0, prefix.length()) != prefix) continue;
        
        // Must have suffix?
        
        // Extract number part
        std::string num_part = name.substr(prefix.length(), name.length() - prefix.length() - ext.length() - suffix.length());
        
        if (num_part.empty()) continue;
        
        // Check if all digits
        bool all_digits = true;
        for (char c : num_part) if (!isdigit(c)) all_digits = false;
        if (!all_digits) continue;
        
        int frame = std::stoi(num_part);
        if (frame < min_frame) min_frame = frame;
        if (frame > max_frame) max_frame = frame;
    }
    
    if (min_frame > max_frame) {
        // Failed to find sequence
        return loadVDB(pattern_or_file);
    }
    
    // Found sequence - store these values BEFORE loading
    sequence_start_frame = min_frame;
    sequence_end_frame = max_frame;
    
    // Load the initial file (or the one passed in)
    // NOTE: loadVDB() resets is_sequence, so we set it AFTER
    if (pattern_or_file.find("#") != std::string::npos) {
         // It was a raw pattern, construct path for start frame
         char buf[32];
         std::string fmt = "%0" + std::to_string(sequence_digits) + "d";
         snprintf(buf, 32, fmt.c_str(), min_frame);
         
         std::string path = sequence_pattern;
         std::string placeholder(sequence_digits, '#');
         size_t pos = path.find(placeholder);
         if (pos != std::string::npos) {
             path.replace(pos, sequence_digits, buf);
         }
         
         if (!loadVDB(path)) return false;
         current_frame = min_frame;
    } else {
         // It was a specific file, load it
         if (!loadVDB(pattern_or_file)) return false;
         
         // Extract its frame number
         std::string num = stem.substr(first_digit, num_len);
         current_frame = std::stoi(num);
    }
    
    // CRITICAL: Set sequence flags AFTER loadVDB (which resets them)
    is_sequence = true;
    timeline_linked = true;
    
    SCENE_LOG_INFO("Loaded VDB Sequence: " + std::to_string(sequence_start_frame) + " - " + std::to_string(sequence_end_frame));
    
    return true;
}

void VDBVolume::unload() {
    if (vdb_volume_id >= 0) {
        VDBVolumeManager::getInstance().unloadVDB(vdb_volume_id);
        vdb_volume_id = -1;
    }
    
    vdb_sequence_id = -1;
    is_sequence = false;
    filepath.clear();
    
    // Reset bounds
    local_bbox_min = Vec3(0);
    local_bbox_max = Vec3(1);
    invalidateWorldBounds();
}

// ═══════════════════════════════════════════════════════════════════════════════
// HITTABLE INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

bool VDBVolume::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes) const {
    if (!visible || ignore_volumes || !isLoaded()) return false;
    
    float t_enter, t_exit;
    if (!intersectTransformedAABB(r, t_min, t_max, t_enter, t_exit)) {
        return false;
    }
    
    // We hit the volume bounding box
    // Actual ray marching happens during shading
    rec.t = t_enter;
    rec.point = r.at(t_enter);
    
    // Normal points toward ray origin (outward from box)
    Vec3 entry_point = r.at(t_enter);
    Vec3 local_entry = world_transform_inv.transform_point(entry_point);
    
    // Determine which face was hit
    Vec3 center = (local_bbox_min + local_bbox_max) * 0.5;
    Vec3 local_rel = local_entry - center;
    Vec3 half_size = (local_bbox_max - local_bbox_min) * 0.5;
    
    // Normalize by half size to find dominant axis
    Vec3 normalized = Vec3(
        std::abs(half_size.x) > 0.0001 ? local_rel.x / half_size.x : 0,
        std::abs(half_size.y) > 0.0001 ? local_rel.y / half_size.y : 0,
        std::abs(half_size.z) > 0.0001 ? local_rel.z / half_size.z : 0
    );
    
    Vec3 local_normal(0, 0, 0);
    if (std::abs(normalized.x) > std::abs(normalized.y) && 
        std::abs(normalized.x) > std::abs(normalized.z)) {
        local_normal.x = normalized.x > 0 ? 1.0 : -1.0;
    } else if (std::abs(normalized.y) > std::abs(normalized.z)) {
        local_normal.y = normalized.y > 0 ? 1.0 : -1.0;
    } else {
        local_normal.z = normalized.z > 0 ? 1.0 : -1.0;
    }
    
    // Transform normal to world space
    rec.normal = world_transform.transform_vector(local_normal).normalize();
    rec.set_face_normal(r, rec.normal);
    
    // Mark as VDB volume hit (for special handling in renderer)
    rec.vdb_volume = this;
    
    // For now, we don't set material - the renderer will handle VDB specially
    rec.material = nullptr;
    rec.materialID = 0xFFFF;
    
    return true;
}

bool VDBVolume::occluded(const Ray& r, float t_min, float t_max) const {
    if (!visible || !isLoaded()) return false;
    
    float t_enter, t_exit;
    // Intersect AABB
    if (!intersectTransformedAABB(r, t_min, t_max, t_enter, t_exit)) {
        return false;
    }
    
    // Ray Marching Logic (Same as EmbreeBVH)
    float step_size = 0.5f; 
    if (this->volume_shader) step_size = this->volume_shader->quality.step_size * 2.0f; 
    
    const auto& mgr = VDBVolumeManager::getInstance();
    int vid = this->getVDBVolumeID();
    
    // Density Multiplier
    float density_mult = (this->volume_shader ? this->volume_shader->density.multiplier : 1.0f) * this->density_scale;
    
    // Stochastic Start
    float t = t_enter + ((float)rand() / RAND_MAX) * step_size;
    float transmittance = 1.0f;
    
    while (t < t_exit) {
        Vec3 pos = r.at(t);
        Vec3 local_pos = world_transform_inv.transform_point(pos);
        
        // Sample Density
        float density = mgr.sampleDensityCPU(vid, local_pos.x, local_pos.y, local_pos.z);
        
        if (density > 0.001f) {
            float sigma_t = density * density_mult;
            transmittance *= exp(-sigma_t * step_size);
        }
        
        // Fully Blocked Check
        if (transmittance < 0.01f) return true;
        
        t += step_size;
    }
    
    // Stochastic Shadow Test
    // Returns TRUE (Blocked) if random > transmittance
    // Returns FALSE (Transparent) otherwise -> ParallelBVH continues to check other objects
    return (Vec3::random_float() > transmittance);
}


bool VDBVolume::bounding_box(float time0, float time1, AABB& output_box) const {
    (void)time0;
    (void)time1;
    
    if (!isLoaded()) {
        output_box = AABB(Vec3(0), Vec3(1));
        return false;
    }
    
    output_box = getWorldBounds();
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSFORM
// ═══════════════════════════════════════════════════════════════════════════════

void VDBVolume::setTransform(const Matrix4x4& transform) {
    world_transform = transform;
    world_transform_inv = transform.inverse();
    
    // CRITICAL: Decompose matrix to update internal members.
    // This ensures that 'updateTransformMatrix' (called by UI or frame updates) 
    // uses the values derived from the Gizmo/Input.
    transform.decompose(position, rotation_euler, scale_vec);

    // Update transform handle if present
    if (transform_handle) {
        transform_handle->setBase(transform);
    }
    
    invalidateWorldBounds();
}

void VDBVolume::setPosition(const Vec3& pos) {
    position = pos;
    updateTransformMatrix();
}

void VDBVolume::setRotation(const Vec3& euler_deg) {
    rotation_euler = euler_deg;
    updateTransformMatrix();
}

void VDBVolume::setScale(const Vec3& scale) {
    scale_vec = scale;
    updateTransformMatrix();
}

void VDBVolume::centerPivotToBottomCenter() {
    // Calculate pivot offset: center X/Z, bottom Y
    Vec3 center = (local_bbox_min + local_bbox_max) * 0.5f;
    
    Vec3 offset_needed;
    offset_needed.x = -center.x;
    offset_needed.y = -local_bbox_min.y;
    offset_needed.z = -center.z;
    
    pivot_offset += offset_needed;
    
    // Applying offset to current bounds
    local_bbox_min += offset_needed;
    local_bbox_max += offset_needed;
    
    if (has_first_frame_bounds) {
        first_frame_bbox_min += offset_needed;
        first_frame_bbox_max += offset_needed;
    }
    
    invalidateWorldBounds();
}

void VDBVolume::updateTransformMatrix() {
    // Build TRS matrix using standard Euler order (T * Rz * Ry * Rx * S)
    Matrix4x4 T = Matrix4x4::translation(position);
    Matrix4x4 Rx = Matrix4x4::rotationX(rotation_euler.x * 0.0174532925f);
    Matrix4x4 Ry = Matrix4x4::rotationY(rotation_euler.y * 0.0174532925f);
    Matrix4x4 Rz = Matrix4x4::rotationZ(rotation_euler.z * 0.0174532925f);
    Matrix4x4 S = Matrix4x4::scaling(scale_vec);
    
    world_transform = T * Rz * Ry * Rx * S;
    world_transform_inv = world_transform.inverse();
    
    if (transform_handle) {
        transform_handle->setBase(world_transform);
    }
    
    invalidateWorldBounds();
}

void VDBVolume::updateBoundsFromVDB() {
    auto* vdb_data = VDBVolumeManager::getInstance().getVolume(vdb_volume_id);
    if (!vdb_data) {
        local_bbox_min = Vec3(-0.5);
        local_bbox_max = Vec3(0.5);
        invalidateWorldBounds();
        return;
    }

    voxel_size = vdb_data->voxel_size;
    Vec3 raw_bbox_min = Vec3(vdb_data->bbox_min[0], vdb_data->bbox_min[1], vdb_data->bbox_min[2]);
    Vec3 raw_bbox_max = Vec3(vdb_data->bbox_max[0], vdb_data->bbox_max[1], vdb_data->bbox_max[2]);
    
    // Boundary Validation
    const float MAX_VALID_VAL = 1e9f;
    bool bbox_valid = true;
    if (std::abs(raw_bbox_min.x) > MAX_VALID_VAL || std::abs(raw_bbox_max.x) > MAX_VALID_VAL || 
        std::isnan(raw_bbox_min.x) || std::isnan(raw_bbox_max.x)) {
        raw_bbox_min = Vec3(-0.5);
        raw_bbox_max = Vec3(0.5);
        bbox_valid = false;
    }

    // Applying Pivot
    Vec3 new_bbox_min = raw_bbox_min + pivot_offset;
    Vec3 new_bbox_max = raw_bbox_max + pivot_offset;

    // Auto-Scale Logic: Handled only once on import/initial load (if scale is identity)
    float dx = new_bbox_max.x - new_bbox_min.x;
    float dy = new_bbox_max.y - new_bbox_min.y;
    float dz = new_bbox_max.z - new_bbox_min.z;
    float max_d = std::max({dx, dy, dz});

    if (bbox_valid && max_d > 50.0f && std::abs(scale_vec.x - 1.0f) < 0.0001f) {
        float corrected_scale = 0.01f;
        if (max_d > 500.0f) corrected_scale = 0.001f;
        if (max_d > 5000.0f) corrected_scale = 0.0001f;
        
        scale_vec = Vec3(corrected_scale);
        updateTransformMatrix(); // Apply scale immediately
        SCENE_LOG_INFO("[VDB] Auto-scaled massive bounds (" + std::to_string(max_d) + "m) to " + std::to_string(corrected_scale));
    }

    if (is_sequence) {
        float current_dim = (new_bbox_max - new_bbox_min).length();
        if (!has_first_frame_bounds && bbox_valid && current_dim > 0.001f) {
            first_frame_bbox_min = new_bbox_min;
            first_frame_bbox_max = new_bbox_max;
            has_first_frame_bounds = true;
            local_bbox_min = new_bbox_min;
            local_bbox_max = new_bbox_max;
        } else if (has_first_frame_bounds) {
            if (bbox_valid && current_dim > 0.001f) {
                local_bbox_min.x = std::min(local_bbox_min.x, new_bbox_min.x);
                local_bbox_min.y = std::min(local_bbox_min.y, new_bbox_min.y);
                local_bbox_min.z = std::min(local_bbox_min.z, new_bbox_min.z);
                local_bbox_max.x = std::max(local_bbox_max.x, new_bbox_max.x);
                local_bbox_max.y = std::max(local_bbox_max.y, new_bbox_max.y);
                local_bbox_max.z = std::max(local_bbox_max.z, new_bbox_max.z);
            }
        } else {
            local_bbox_min = Vec3(-0.5);
            local_bbox_max = Vec3(0.5);
        }
    } else {
        local_bbox_min = new_bbox_min;
        local_bbox_max = new_bbox_max;
    }
    
    invalidateWorldBounds();
}

// ═══════════════════════════════════════════════════════════════════════════════
// BOUNDS HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

AABB VDBVolume::getWorldBounds() const {
    if (world_bounds_dirty) {
        AABB local_box(local_bbox_min, local_bbox_max);
        world_bounds_cache = transformAABB(local_box);
        world_bounds_dirty = false;
    }
    return world_bounds_cache;
}

AABB VDBVolume::transformAABB(const AABB& local_box) const {
    // Transform all 8 corners and find new AABB
    Vec3 corners[8] = {
        Vec3(local_box.min.x, local_box.min.y, local_box.min.z),
        Vec3(local_box.max.x, local_box.min.y, local_box.min.z),
        Vec3(local_box.min.x, local_box.max.y, local_box.min.z),
        Vec3(local_box.max.x, local_box.max.y, local_box.min.z),
        Vec3(local_box.min.x, local_box.min.y, local_box.max.z),
        Vec3(local_box.max.x, local_box.min.y, local_box.max.z),
        Vec3(local_box.min.x, local_box.max.y, local_box.max.z),
        Vec3(local_box.max.x, local_box.max.y, local_box.max.z)
    };
    
    Vec3 world_min(1e30, 1e30, 1e30);
    Vec3 world_max(-1e30, -1e30, -1e30);
    
    for (int i = 0; i < 8; ++i) {
        Vec3 world_corner = world_transform.transform_point(corners[i]);
        world_min.x = std::min(world_min.x, world_corner.x);
        world_min.y = std::min(world_min.y, world_corner.y);
        world_min.z = std::min(world_min.z, world_corner.z);
        world_max.x = std::max(world_max.x, world_corner.x);
        world_max.y = std::max(world_max.y, world_corner.y);
        world_max.z = std::max(world_max.z, world_corner.z);
    }
    
    return AABB(world_min, world_max);
}

bool VDBVolume::intersectTransformedAABB(const Ray& r, float t_min, float t_max,
                                          float& out_t_enter, float& out_t_exit) const {
    // FALLBACK: Use World AABB instead of Local OBB
    // This is more robust and matches the Gizmo/Selection logic
    // Volumetrics are tolerant to loose bounds (density is 0 in empty corners)
    
    AABB world_box = getWorldBounds();
    
    float t0 = t_min;
    float t1 = t_max;
    
    for (int i = 0; i < 3; i++) {
        float invD = 1.0f / r.direction[i];
        float t_near = (world_box.min[i] - r.origin[i]) * invD;
        float t_far = (world_box.max[i] - r.origin[i]) * invD;
        
        if (invD < 0.0f) std::swap(t_near, t_far);
        
        t0 = t_near > t0 ? t_near : t0;
        t1 = t_far < t1 ? t_far : t1;
        
        if (t1 <= t0) return false;
    }
    
    out_t_enter = t0;
    out_t_exit = t1;
    
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// VDB DATA ACCESS
// ═══════════════════════════════════════════════════════════════════════════════

std::vector<std::string> VDBVolume::getAvailableGrids() const {
    // TODO: Implement multi-grid support in VDBVolumeManager
    // For now, return standard grid names if VDB is loaded
    
    std::vector<std::string> grids;
    if (isLoaded()) {
        auto* vdb_data = VDBVolumeManager::getInstance().getVolume(vdb_volume_id);
        if (vdb_data && vdb_data->has_density) {
            grids.push_back("density");
        }
        if (vdb_data && vdb_data->has_temperature) {
            grids.push_back("temperature");
        }
        if (vdb_data && vdb_data->has_velocity) {
            grids.push_back("vel");
        }
    }
    
    return grids;
}

bool VDBVolume::hasGrid(const std::string& grid_name) const {
    auto grids = getAvailableGrids();
    return std::find(grids.begin(), grids.end(), grid_name) != grids.end();
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANIMATION
// ═══════════════════════════════════════════════════════════════════════════════

int VDBVolume::getFrameCount() const {
    if (!is_sequence) return 1;
    return std::max(1, sequence_end_frame - sequence_start_frame + 1);
}

void VDBVolume::setCurrentFrame(int frame) {
    if (!is_sequence) return;
    
    current_frame = std::clamp(frame, 0, getFrameCount() - 1);
    
    // TODO: Update vdb_volume_id from sequence
    // VDBVolumeManager::getInstance().setSequenceFrame(vdb_sequence_id, current_frame);
    // vdb_volume_id = VDBVolumeManager::getInstance().getSequenceCurrentVolumeID(vdb_sequence_id);
    
    updateBoundsFromVDB();
}



// ═══════════════════════════════════════════════════════════════════════════════
// GPU DATA
// ═══════════════════════════════════════════════════════════════════════════════

bool VDBVolume::isGPUReady() const {
    if (vdb_volume_id < 0) return false;
    
    auto* vdb_data = VDBVolumeManager::getInstance().getVolume(vdb_volume_id);
    return vdb_data && vdb_data->gpu_uploaded;
}

bool VDBVolume::uploadToGPU() {
    if (vdb_volume_id < 0) return false;
    
    return VDBVolumeManager::getInstance().uploadToGPU(vdb_volume_id);
}

void* VDBVolume::getDensityGridGPU() const {
    return VDBVolumeManager::getInstance().getGPUGrid(vdb_volume_id);
}

void* VDBVolume::getTemperatureGridGPU() const {
    // TODO: Implement multi-grid GPU access
    return nullptr;
}

void* VDBVolume::getVelocityGridGPU() const {
    // TODO: Implement multi-grid GPU access
    return nullptr;
}
