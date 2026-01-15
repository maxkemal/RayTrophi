/**
 * @file VDBVolume.h
 * @brief VDB Volume Scene Object - Standalone volumetric container
 * 
 * This class represents a VDB volume as a scene object, independent of geometry meshes.
 * It follows Blender/Houdini conventions for VDB import and rendering.
 * 
 * Key features:
 * - Uses VDB's native bounding box (not mesh-dependent)
 * - Full transform support (position, rotation, scale)
 * - Shader assignment (VolumeShader)
 * - Animation sequence support
 * 
 * @note This is a Hittable subclass, so it participates in ray tracing like other geometry.
 */

#ifndef VDB_VOLUME_H
#define VDB_VOLUME_H

#include "Hittable.h"
#include "Matrix4x4.h"
#include "VolumeShader.h"
#include "VDBVolumeManager.h"
#include "Transform.h"
#include <string>
#include <memory>
#include <vector>

// Forward declarations
class Triangle;

/**
 * @brief VDB Volume Object - Standalone volumetric object in scene
 * 
 * Key differences from mesh-attached volumetrics:
 * - Has its own transform (not dependent on mesh)
 * - Uses VDB's native bounding box
 * - Supports animation sequences
 * - Shader is a separate, assignable component
 * 
 * @example
 * ```cpp
 * auto vdb = std::make_shared<VDBVolume>();
 * vdb->loadVDB("explosion.vdb");
 * vdb->setPosition(Vec3(0, 2, 0));
 * vdb->setShader(VolumeShader::createFirePreset());
 * scene.world.add(vdb);
 * ```
 */
class VDBVolume : public Hittable {
public:
    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTRUCTION
    // ═══════════════════════════════════════════════════════════════════════════
    VDBVolume();
    explicit VDBVolume(const std::string& vdb_path);
    ~VDBVolume() override = default;
    
    /**
     * @brief Load a single VDB file
     * @param path Path to .vdb file
     * @return true on success
     */
    bool loadVDB(const std::string& path);
    
    /**
     * @brief Load VDB sequence from pattern
     * @param pattern Pattern like "fire_####.vdb" or "smoke.%04d.vdb"
     * @return true on success
     */
    bool loadVDBSequence(const std::string& pattern);
    
    /**
     * @brief Unload and release VDB data
     */
    void unload();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // HITTABLE INTERFACE
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Ray intersection test
     * 
     * For VDB volumes, we test against the transformed AABB.
     * Actual ray marching happens during shading, not here.
     */
    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    
    /**
     * @brief Get world-space bounding box
     */
    bool bounding_box(float time0, float time1, AABB& output_box) const override;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // TRANSFORM (Object → World)
    // ═══════════════════════════════════════════════════════════════════════════
    
    void setTransform(const Matrix4x4& transform);
    Matrix4x4 getTransform() const { return world_transform; }
    Matrix4x4 getInverseTransform() const { return world_transform_inv; }
    
    void setPosition(const Vec3& pos);
    void setRotation(const Vec3& euler_deg);
    void setScale(const Vec3& scale);
    
    Vec3 getPosition() const { return position; }
    Vec3 getRotation() const { return rotation_euler; }
    Vec3 getScale() const { return scale_vec; }
    
    /**
     * @brief Move pivot point to bottom center of bounding box
     * Offsets local bbox so origin is at (centerX, bottomY, centerZ)
     */
    void centerPivotToBottomCenter();
    
    /**
     * @brief Get transform handle for gizmo integration
     */
    std::shared_ptr<Transform> getTransformHandle() { return transform_handle; }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // VDB DATA ACCESS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Get VDBVolumeManager volume ID
     */
    int getVDBVolumeID() const { return vdb_volume_id; }
    
    /**
     * @brief Get sequence ID (if animated)
     */
    int getVDBSequenceID() const { return vdb_sequence_id; }
    
    /**
     * @brief Check if VDB is loaded
     */
    bool isLoaded() const { return vdb_volume_id >= 0 || vdb_sequence_id >= 0; }
    
    /**
     * @brief Get native VDB bounds (before transform)
     */
    Vec3 getLocalBoundsMin() const { return local_bbox_min; }
    Vec3 getLocalBoundsMax() const { return local_bbox_max; }
    
    /**
     * @brief Manually set local bounds (useful for correcting invalid/overflow bounds)
     */
    void setLocalBounds(const Vec3& min, const Vec3& max) { 
        local_bbox_min = min; 
        local_bbox_max = max; 
        
        // Also fix first frame bounds if they were corrupted or uninitialized
        if (is_sequence) {
            first_frame_bbox_min = min;
            first_frame_bbox_max = max;
            has_first_frame_bounds = true;
        }
        
        invalidateWorldBounds();
    }
    
    /**
     * @brief Ray-AABB intersection with transform
     */
    bool intersectTransformedAABB(const Ray& r, float t_min, float t_max, 
                                   float& out_t_enter, float& out_t_exit) const;
    
    /**
     * @brief Get world-space bounds (after transform)
     */
    AABB getWorldBounds() const;
    
    /**
     * @brief Get list of available grid names in the VDB
     */
    std::vector<std::string> getAvailableGrids() const;
    
    /**
     * @brief Check if a specific grid exists
     */
    bool hasGrid(const std::string& grid_name) const;
    
    /**
     * @brief Get VDB file path
     */
    const std::string& getFilePath() const { return filepath; }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SHADER ASSIGNMENT
    // ═══════════════════════════════════════════════════════════════════════════
    
    void setShader(std::shared_ptr<VolumeShader> shader) { volume_shader = shader; }
    std::shared_ptr<VolumeShader> getShader() const { return volume_shader; }
    
    /**
     * @brief Get or create default shader
     */
    std::shared_ptr<VolumeShader> getOrCreateShader() {
        if (!volume_shader) {
            volume_shader = std::make_shared<VolumeShader>();
        }
        return volume_shader;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ANIMATION (VDB Sequences)
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Check if this is an animated sequence
     */
    bool isAnimated() const { return is_sequence; }
    
    /**
     * @brief Get total frame count (1 for single files)
     */
    int getFrameCount() const;
    
    /**
     * @brief Get current frame index
     */
    int getCurrentFrame() const { return current_frame; }
    
    /**
     * @brief Set current frame (updates VDB data reference)
     */
    void setCurrentFrame(int frame);
    
    /**
     * @brief Link/unlink from main timeline
     */
    void setLinkedToTimeline(bool linked) { timeline_linked = linked; }
    bool isLinkedToTimeline() const { return timeline_linked; }
    
    /**
     * @brief Update frame from timeline (called by animation system)
     * @param timeline_frame Current frame from main timeline
     */
    void updateFromTimeline(int timeline_frame);
    
    void setFrameOffset(int offset) { frame_offset = offset; }
    int getFrameOffset() const { return frame_offset; }
    
  
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SCENE HIERARCHY
    // ═══════════════════════════════════════════════════════════════════════════
    
    std::string name = "VDB Volume";
    std::string filepath;  ///< Original file path for reload/export
    int sequence_digits = 4; // Number of digits in sequence (e.g. 4 for 0001, 1 for 0)
    
    /**
     * @brief Get object type identifier
     */
    std::string getObjectType() const { return "VDBVolume"; }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // GPU DATA PREPARATION
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Check if GPU data is ready
     */
    bool isGPUReady() const;
    
    /**
     * @brief Ensure GPU data is uploaded
     */
    bool uploadToGPU();
    
    /**
     * @brief Get GPU grid pointer for density
     */
    void* getDensityGridGPU() const;
    
    /**
     * @brief Get GPU grid pointer for temperature (if available)
     */
    void* getTemperatureGridGPU() const;
    
    /**
     * @brief Get GPU grid pointer for velocity (if available)
     */
    void* getVelocityGridGPU() const;
    // Shader
    std::shared_ptr<VolumeShader> volume_shader;
    
    // Density control
    float density_scale = 1.0f;
private:
    // VDB data reference
    int vdb_volume_id = -1;
    int vdb_sequence_id = -1;
    
    // Transform components
    std::shared_ptr<Transform> transform_handle;
    Matrix4x4 world_transform;
    Matrix4x4 world_transform_inv;   
    Vec3 position = Vec3(0);
    Vec3 pivot_offset = Vec3(0); // Persistent offset for pivot adjustments
    Vec3 rotation_euler = Vec3(0);  // Degrees
    Vec3 scale_vec = Vec3(1);
    
    // Native VDB bounds (local space, from VDB file)
    Vec3 local_bbox_min;
    Vec3 local_bbox_max;
    
    // Cached world bounds
    mutable AABB world_bounds_cache;
    mutable bool world_bounds_dirty = true;
    
  
    
    // Animation
    bool is_sequence = false;
    std::string sequence_pattern; // e.g. "path/to/explosion_####.vdb"
    int sequence_start_frame = 0;
    int sequence_end_frame = 0;
    
    int current_frame = 0;
    bool timeline_linked = true;
    int frame_offset = 0;  // Offset from timeline frame
    
    // Sequence bounds consistency - store first frame's bbox for consistent sizing
    Vec3 first_frame_bbox_min;
    Vec3 first_frame_bbox_max;
    bool has_first_frame_bounds = false;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // INTERNAL HELPERS
    // ═══════════════════════════════════════════════════════════════════════════
    
    void updateTransformMatrix();
    void updateBoundsFromVDB();
    void invalidateWorldBounds() { world_bounds_dirty = true; }
    
    /**
     * @brief Transform AABB to world space
     */
    AABB transformAABB(const AABB& local_box) const;
    

};

#endif // VDB_VOLUME_H
