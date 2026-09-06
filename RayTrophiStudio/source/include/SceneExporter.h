/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          SceneExporter.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <atomic>
#include "Texture.h"
#include "GltfDirectWriter.h"
// Forward Declarations
struct SceneData;
class Material;
struct GpuMaterial;
class Hittable;

struct ExportSettings {
    bool export_geometry = true;
    bool export_materials = true;
    bool export_lights = false;
    bool export_cameras = false;
    bool export_animations = true;
    bool export_skinning = true;  // Bones/Weights
    bool bake_transforms = false; // If true, bakes static transforms (good for static meshes)
    bool binary_mode = true;      // .glb vs .gltf
    bool export_selected_only = false;
    bool embed_textures = true;
    bool bake_terrain_materials = true;
    int terrain_bake_resolution = 1024;
    // Collapse scattered/foliage instance nodes that share one source mesh into a
    // single EXT_mesh_gpu_instancing node (one mesh + a transform-array buffer)
    // instead of writing one node per instance. Works in both .glb and .gltf.
    bool use_gpu_instancing_extension = true;
};

class SceneExporter {
public:
    static SceneExporter& getInstance() {
        static SceneExporter instance;
        return instance;
    }
    
    // UI Logic
    bool show_export_popup = false;
    std::atomic<bool> is_exporting{false}; // Thread-safe status flag
    std::string current_export_status = ""; // For HUD feedback
    
    ExportSettings settings;
    bool drawExportPopup(SceneData& scene); // Returns true if user clicked Export

    // Rough, deliberately conservative pre-export sizing so the popup can warn
    // the user (with a confirmation gate) before a crowded/scattered scene runs
    // away with RAM. Not a substitute for actually running the export.
    struct ExportEstimate {
        size_t object_count = 0;
        size_t triangle_count = 0;          // non-instanced Triangle/TriangleMesh triangles
        size_t legacy_triangle_count = 0;   // of those, legacy Triangle facade objects
                                            // (the only non-instanced geometry the
                                            //  writer has to materialise in RAM)
        // Scatter/foliage placements. ★ Counted from BOTH sources the writer
        // reads: InstanceManager (the canonical Vulkan path - scatter is never
        // expanded into world.objects there) and the HittableInstance facade
        // (CPU-compat projects). Counting only the latter is what made this
        // panel report 0 instances for a scene the exporter happily wrote 1000
        // of - the estimate inherited the exporter's OLD blind spot after the
        // writer was fixed. See GltfWriter::collectScatter.
        size_t instance_count = 0;          // placements (both sources)
        size_t unique_instance_sources = 0; // distinct source meshes among them
        size_t instance_triangle_count = 0; // triangles counted ONCE per unique source
        // Of instance_triangle_count, the part the writer must MATERIALISE in
        // RAM (legacy Triangle-facade sources). Flat SoA scatter sources are
        // streamed from where they already live and cost zero extra heap, so
        // they must not inflate the peak-memory warning.
        size_t materialised_instance_triangles = 0;
        double estimated_peak_mb = 0.0;
        bool computed = false;
    };
    ExportEstimate computeExportEstimate(SceneData& scene, const ExportSettings& settings);

    // Measured result of the most recent exportScene() call. Surfaced over IPC
    // (scene.export_gltf) so export cost is regressable from a script instead of
    // only observable by watching Task Manager during a manual export.
    rtgltf::WriteStats last_stats{};
    std::string last_error;

    /**
     * @brief Exports the entire scene to a GLTF/GLB file.
     * 
     * @param filepath Output path (e.g. "C:/Exports/scene.glb")
     * @param scene Reference to the active SceneData
     * @param settings Export configuration
     * @param selected_objects Optional list of selected objects (if export_selected_only is true)
     * @return true if successful
     */
    bool exportScene(const std::string& filepath, SceneData& scene, const ExportSettings& settings, 
                     const std::vector<std::shared_ptr<Hittable>>& selected_objects = {});

private:
    SceneExporter() = default;
    ~SceneExporter() = default;
    
    // Non-copyable
    SceneExporter(const SceneExporter&) = delete;
    SceneExporter& operator=(const SceneExporter&) = delete;

    // Cached pre-export estimate, recomputed once when the popup transitions
    // from closed to open (not every frame - the scan is O(scene object count)).
    ExportEstimate export_estimate;
    bool popup_open_last_frame = false;
    bool confirm_large_export = false;
};

