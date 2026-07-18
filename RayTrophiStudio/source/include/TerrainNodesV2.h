/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          TerrainNodesV2.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file TerrainNodesV2.h
 * @brief Terrain nodes using the modern NodeSystem V2 types
 * 
 * This implements terrain generation nodes using NodeBase from the new
 * node system, enabling use with NodeEditorUIV2 and the pull-based
 * evaluation system.
 */

#include "NodeSystem/NodeCore.h"
#include "NodeSystem/Node.h"
#include "NodeSystem/Graph.h"
#include "NodeSystem/EvaluationContext.h"
#include "TerrainManager.h"
#include "json.hpp"
#include <unordered_map>
#include <cstring>
#include <future>
#include <memory>
#include <atomic>
#include <chrono>
#include <functional>

namespace TerrainNodesV2 {

    // C++14 compatible clamp helper (std::clamp requires C++17)
    template<typename T>
    inline T clampValue(T val, T lo, T hi) {
        return (val < lo) ? lo : ((val > hi) ? hi : val);
    }

    /**
     * @brief Apply a soft falloff to the edges of a 2D float array
     */
    inline void applyEdgeFalloff(std::vector<float>& data, int w, int h, float width, float targetValue) {
        if (width <= 0.01f) return;
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // Distance to nearest edge in pixels
                float dx = (float)std::min(x, w - 1 - x);
                float dy = (float)std::min(y, h - 1 - y);
                float dist = (std::min)(dx, dy);
                
                if (dist < width) {
                    float t = dist / width;
                    // Quadratic ease-in-out for smoother transition
                    float smoothT = t * t * (3.0f - 2.0f * t);
                    
                    float& val = data[y * w + x];
                    val = targetValue + (val - targetValue) * smoothT;
                }
            }
        }
    }

    // ============================================================================
    // TERRAIN EVALUATION CONTEXT
    // ============================================================================
    
    /**
     * @brief Context data for terrain evaluation
     * 
     * Provides access to the terrain object during node evaluation.
     */
    struct TerrainContext {
        TerrainObject* terrain = nullptr;
        int width = 0;
        int height = 0;
        // Inspector/thumbnail pulls must be observational. Nodes which publish
        // auxiliary data (flowMap, hardnessMap, etc.) gate those writes on this.
        bool publishTerrainState = true;
        
        // Scale values - preserved throughout evaluation
        float scale_xz = 100.0f;
        float scale_y = 10.0f;
        // Main-thread snapshot used by async climate evaluation. Worker nodes
        // never dereference SceneData/World directly.
        bool has_scene_sun = false;
        float scene_sun_x = 0.0f;
        float scene_sun_y = 1.0f;
        float scene_sun_z = 0.0f;
        
        // Default resolution when terrain is not yet initialized
        static constexpr int DEFAULT_RESOLUTION = 256;
        static constexpr float DEFAULT_SCALE_XZ = 100.0f;
        static constexpr float DEFAULT_SCALE_Y = 10.0f;
        
        TerrainContext() = default;
        explicit TerrainContext(TerrainObject* t) : terrain(t) {
            if (t) {
                width = t->heightmap.width;
                height = t->heightmap.height;
                scale_xz = t->heightmap.scale_xz;
                scale_y = t->heightmap.scale_y;
                
                // CRITICAL FIX: If terrain dimensions are invalid (0 or very small),
                // use a sensible default. This prevents issues when modifier nodes
                // are placed in the graph before the first direct Height→Output evaluation.
                if (width < 2) width = DEFAULT_RESOLUTION;
                if (height < 2) height = DEFAULT_RESOLUTION;
                if (scale_xz < 1.0f) scale_xz = DEFAULT_SCALE_XZ;
                if (scale_y < 0.1f) scale_y = DEFAULT_SCALE_Y;
            }
        }
    };

    // ============================================================================
    // NODE TYPES ENUM
    // ============================================================================
    
    enum class NodeType {
        HeightmapInput,
        NoiseGenerator,
        HydraulicErosion,
        ThermalErosion,
        FluvialErosion,
        WindErosion,
        HeightMask,
        SlopeMask,
        CurvatureMask,
        FlowMask,        // Soil/sediment accumulation
        ExposureMask,    // Sun-facing direction
        Add,
        Subtract,
        Multiply,
        Blend,
        Clamp,
        Invert,
        // NEW OPERATORS
        Smooth,
        Normalize,
        Terrace,
        EdgeFalloff,
        MaskCombine,
        Overlay,
        Screen,
        // NEW: Procedural Texture Nodes
        AutoSplat,       // Geospatial auto texture mapping
        MaskPaint,       // Viewport brush painting
        MaskImage,       // Import grayscale mask
        // NEW: Geological Transform Nodes
        Fault,           // Strike-slip fault line
        Mesa,            // Flat-topped plateau
        Shear,           // Diagonal deformation
        Stacks,          // Sea stacks / hoodoos
        Anastomosing,    // Braided channel patterns
        // NEW: Sediment Deposition Nodes
        SedimentDeposition,  // Sediment accumulation in valleys
        AlluvialFan,         // Fan-shaped deposits at mountain bases
        DeltaFormation,      // River delta formation
        // NEW: Erosion Wizard
        ErosionWizard,       // All-in-one erosion with presets
        // Outputs
        HeightOutput,
        SplatOutput,
        HardnessOutput,
        // Inputs
        HardnessInput,
        // Appended for serialized enum stability.
        Resample,
        ChannelExtract,
        SplatCompose,
        Remap,
        MaskMorphology,
        // Geological data and surface synthesis. Appended for enum stability.
        WetnessMap,
        SoilDepth,
        Lithology,
        Strata,
        SurfaceComposer,
        // Climate, snow and glacial processing. Appended for enum stability.
        Climate,
        Snowfall,
        SnowSettle,
        SnowMeltFreeze,
        GlacierFlow,
        SnowClimate,
        MaskAdjust,
        // Shared analysis/field publishing. Appended for serialized enum stability.
        TerrainAnalysis,
        TerrainFieldsOutput,
        BiomeComposer,
        // Hydrology and automatic river authoring. Appended for serialized enum stability.
        WatershedAnalysis,
        RiverNetwork,
        RiverSplineOutput,
        RiverBedCarve,
        LakeBasin,
        LakeSurfaceOutput,
        RiverHydraulics,
        // Foliage authoring recipes. Runtime instances remain owned by InstanceManager.
        // Appended for serialized enum stability.
        FoliageLayer,
        FoliageSet,
        FoliageOutput
    };

    // ============================================================================
    // TERRAIN NODE BASE
    // ============================================================================
    
    /**
     * @brief Base class for terrain nodes
     * 
     * Uses Image2DData for height/mask data transport between nodes.
     */
    class TerrainNodeBase : public NodeSystem::NodeBase {
    public:
        NodeType terrainNodeType;
        // Explicit side-effect/output publication gate. Computational nodes
        // remain pull-driven; terrain/scene sink nodes are skipped when false.
        bool publicationEnabled = true;
        
        TerrainContext* getTerrainContext(NodeSystem::EvaluationContext& ctx) {
            return ctx.getDomainContext<TerrainContext>();
        }
        
        // Raw Image2D access for nodes that intentionally accept multi-channel
        // payloads (Channel Extract and Splat Output).
        NodeSystem::Image2DData getImageInput(int inputIndex, NodeSystem::EvaluationContext& ctx) {
            NodeSystem::PinValue val = getInputValue(inputIndex, ctx);
            if (auto* img = std::get_if<NodeSystem::Image2DData>(&val)) {
                if (img->isValid()) return *img;
            }
            return NodeSystem::Image2DData{};
        }

        // Terrain height/mask operators are single-channel by contract. Keeping
        // this check at the common input boundary prevents RGBA splat/erosion
        // payloads from reaching algorithms that allocate only width*height.
        NodeSystem::Image2DData getHeightInput(int inputIndex, NodeSystem::EvaluationContext& ctx) {
            NodeSystem::Image2DData img = getImageInput(inputIndex, ctx);
            return (img.isValid() && img.channels == 1) ? img : NodeSystem::Image2DData{};
        }
        
        // Helper to create output image
        NodeSystem::Image2DData createHeightOutput(int w, int h) {
            NodeSystem::Image2DData result;
            result.data = std::make_shared<std::vector<float>>(w * h, 0.0f);
            result.width = w;
            result.height = h;
            result.channels = 1;
            result.semantic = NodeSystem::ImageSemantic::Height;
            return result;
        }
        
        NodeSystem::Image2DData createMaskOutput(int w, int h) {
            NodeSystem::Image2DData result;
            result.data = std::make_shared<std::vector<float>>(w * h, 0.0f);
            result.width = w;
            result.height = h;
            result.channels = 1;
            result.semantic = NodeSystem::ImageSemantic::Mask;
            return result;
        }

        // ========================================================================
        // SERIALIZATION
        // ========================================================================
        
        /**
         * @brief Serialize node-specific data to JSON
         * Override in derived classes to save custom parameters
         */
        virtual void serializeToJson(nlohmann::json& j) const {
            // Base: just save position and type
            j["x"] = x;
            j["y"] = y;
            j["nodeType"] = static_cast<int>(terrainNodeType);
            j["typeId"] = getTypeId();
            j["name"] = name;
            j["publicationEnabled"] = publicationEnabled;
        }
        
        /**
         * @brief Deserialize node-specific data from JSON
         * Override in derived classes to load custom parameters
         */
        virtual void deserializeFromJson(const nlohmann::json& j) {
            if (j.contains("x")) x = j["x"].get<float>();
            if (j.contains("y")) y = j["y"].get<float>();
            if (j.contains("name")) name = j["name"].get<std::string>();
            publicationEnabled = j.value("publicationEnabled", true);
        }
    };

    // ============================================================================
    // INPUT NODES
    // ============================================================================
    
    /**
     * @brief Heightmap Input Node - reads from terrain or loads from file
     */
    class HeightmapInputNode : public TerrainNodeBase {
    public:
        enum class SourceMode { Terrain, File };
        
        // UI Interaction flags
        bool browseForHeightmap = false; 
        
        SourceMode sourceMode = SourceMode::Terrain;
        char filePath[256] = "";
        
        // Settings
        float heightScale = 1.0f; // Scale multiplier for loaded heightmap
        bool maintainAspectRatio = false; // Disable padding by default to stretch to terrain
        int maxResolution = 2048; // Limit import resolution
        int smoothIterations = 1; // Smoothing pass count (default 1 = mild blur)
        
        // Edge Falloff Settings
        float edgeFalloffWidth = 0.0f; // Width of the fade area in pixels
        float edgeFalloffValue = 0.0f; // Target height at the absolute edge (0-1)
        
        // Transient loaded data
        std::vector<float> rawHeightData; // Original loaded data (before processing)
        std::vector<float> loadedHeightData; // Processed data (smoothed)
        int loadedWidth = 0;
        int loadedHeight = 0;
        bool fileLoaded = false;
        
        HeightmapInputNode() {
            name = "Heightmap Input";
            terrainNodeType = NodeType::HeightmapInput;
            
            // No inputs - reads from terrain or file
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Heightmap Input";
            metadata.category = "Input";
            metadata.headerColor = IM_COL32(50, 150, 75, 255);
            headerColor = ImVec4(0.2f, 0.6f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            auto* tctx = getTerrainContext(ctx);
            if (!tctx || !tctx->terrain) {
                ctx.addError(id, "No terrain context");
                return NodeSystem::PinValue{};
            }
            
            if (sourceMode == SourceMode::File && fileLoaded && !loadedHeightData.empty()) {
                // Determine output dimensions
                int outW = loadedWidth;
                int outH = loadedHeight;
                
                if (maintainAspectRatio) {
                    int dim = std::max(loadedWidth, loadedHeight);
                    outW = dim;
                    outH = dim;
                }
                
                auto result = createHeightOutput(outW, outH);
                
                if (maintainAspectRatio && (loadedWidth != outW || loadedHeight != outH)) {
                    // Pad with zeros (or edge clamp?) - Using Zeros for now
                    std::fill(result.data->begin(), result.data->end(), 0.0f);
                    
                    // Center the image? Or align top-left? Top-Left is simpler for UV mapping intuition
                    // But centering is often nicer for objects. Let's Center.
                    int offX = (outW - loadedWidth) / 2;
                    int offY = (outH - loadedHeight) / 2;
                    
                    for(int y = 0; y < loadedHeight; y++) {
                        for(int x = 0; x < loadedWidth; x++) {
                            int dstIdx = (y + offY) * outW + (x + offX);
                            int srcIdx = y * loadedWidth + x;
                            (*result.data)[dstIdx] = loadedHeightData[srcIdx] * heightScale;
                        }
                    }
                } else {
                    // Direct copy (Stretch or 1:1)
                    size_t count = loadedHeightData.size();
                    for(size_t i = 0; i < count; i++) {
                        (*result.data)[i] = loadedHeightData[i] * heightScale;
                    }
                }
                
                return result;
            }
            
            // Default: read from terrain. A terrain that has never been evaluated
            // yet has heightmap.width/height == 0 (TerrainSystem.h default) — reading
            // that raw zero here (instead of TerrainContext's already-clamped
            // width/height) produces a 0x0 Image2DData. Every downstream node then
            // resizes terrain->heightmap back to 0x0, a fixed point that never
            // escapes: inserting erosion/modifier nodes before the first direct
            // Height->Output evaluate looked like an infinite loop/hang because of
            // this — each node kept "succeeding" on empty 0-length data forever.
            // NoiseGeneratorNode already reads tctx->width/height (clamped to
            // TerrainContext::DEFAULT_RESOLUTION when uninitialized); mirror that
            // here so a never-evaluated terrain gets a sane starting resolution
            // instead of a degenerate empty one.
            TerrainObject* terrain = tctx->terrain;
            int w = terrain->heightmap.width;
            int h = terrain->heightmap.height;
            if (w < 2 || h < 2) {
                w = tctx->width;
                h = tctx->height;
            }

            auto result = createHeightOutput(w, h);

            const size_t expected = static_cast<size_t>(w) * h;
            // Terrain mode is the authored graph SOURCE, not the previous graph
            // result. Reading heightmap.data here made every Evaluate feed the
            // last Hydraulic/Fluvial/Carve output back into the chain and
            // accumulate erosion indefinitely.
            if (terrain->original_heightmap_data.size() != expected &&
                terrain->heightmap.data.size() == expected) {
                terrain->original_heightmap_data = terrain->heightmap.data;
            }
            if (terrain->original_heightmap_data.size() == expected) {
                *result.data = terrain->original_heightmap_data;
            }
            
            // Apply Edge Falloff if enabled
            if (edgeFalloffWidth > 0.01f) {
                applyEdgeFalloff(*result.data, w, h, edgeFalloffWidth, edgeFalloffValue);
            }
            
            return result;
        }
        
        void drawContent() override {
            const char* modes[] = { "Terrain", "File" };
            int modeIdx = (int)sourceMode;
            if (ImGui::Combo("Source", &modeIdx, modes, 2)) {
                sourceMode = (SourceMode)modeIdx;
                dirty = true;
            }
            
            if (sourceMode == SourceMode::File) {
                // Show current file (truncated if too long)
                if (fileLoaded) {
                    std::string shortPath = filePath;
                    if (shortPath.length() > 30) {
                        shortPath = "..." + shortPath.substr(shortPath.length() - 27);
                    }
                    ImGui::TextDisabled("%s", shortPath.c_str());
                    ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), 
                        "Size: %dx%d", loadedWidth, loadedHeight);
                } else {
                    ImGui::TextDisabled("No file loaded");
                }
                
                if (ImGui::Button("Browse...")) {
                    // This will be called from UI context where openFileDialogW is available
                    browseForHeightmap = true;
                }
                
                // Scale control
                ImGui::Spacing();
                if (ImGui::DragFloat("Intensity", &heightScale, 1.0f, 0.0f, 5000.0f, "%.1f")) {
                    dirty = true;
                }
                if (ImGui::Checkbox("Maintain AR", &maintainAspectRatio)) {
                    dirty = true;
                }
                
                // Resolution limit
                const char* items[] = { "512", "1024", "2048", "4096", "8192" };
                int currentIdx = 2; // Default 2048
                if (maxResolution == 512) currentIdx = 0;
                else if (maxResolution == 1024) currentIdx = 1;
                else if (maxResolution == 2048) currentIdx = 2;
                else if (maxResolution == 4096) currentIdx = 3;
                else if (maxResolution == 8192) currentIdx = 4;
                
                if (ImGui::Combo("Max Resolution", &currentIdx, items, 5)) {
                    if (currentIdx == 0) maxResolution = 512;
                    else if (currentIdx == 1) maxResolution = 1024;
                    else if (currentIdx == 2) maxResolution = 2048;
                    else if (currentIdx == 3) maxResolution = 4096;
                    else if (currentIdx == 4) maxResolution = 8192;
                    
                    // Reload if file exists
                    if (fileLoaded && strlen(filePath) > 0) {
                        loadHeightmapFromFile();
                        dirty = true;
                    }
                }
                
                // Smoothness (Blur Radius)
                if (ImGui::SliderInt("Blur Radius", &smoothIterations, 0, 50)) {
                    if (fileLoaded) applySmoothing();
                    dirty = true;
                }

                ImGui::Separator();
                ImGui::Text("Edge Falloff");
                if (ImGui::DragFloat("Fade Width", &edgeFalloffWidth, 1.0f, 0.0f, 256.0f, "%.0f px")) dirty = true;
                if (ImGui::SliderFloat("Fade Value", &edgeFalloffValue, 0.0f, 1.0f)) dirty = true;
            } else {
                // Terrain mode also has edge falloff
                ImGui::Separator();
                ImGui::Text("Edge Falloff");
                if (ImGui::DragFloat("Fade Width", &edgeFalloffWidth, 1.0f, 0.0f, 256.0f, "%.0f px")) dirty = true;
                if (ImGui::SliderFloat("Fade Value", &edgeFalloffValue, 0.0f, 1.0f)) dirty = true;
            }
        }
        
        void loadHeightmapFromFile();  // Implemented in cpp
        void applySmoothing();         // Implemented in cpp
        
        std::string getTypeId() const override { return "TerrainV2.HeightmapInput"; }
        float getCustomWidth() const override { return 140.0f; }
        
        // Serialization overrides
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["sourceMode"] = static_cast<int>(sourceMode);
            j["filePath"] = std::string(filePath);
            j["heightScale"] = heightScale;
            j["maintainAspectRatio"] = maintainAspectRatio;
            j["maxResolution"] = maxResolution;
            j["smoothIterations"] = smoothIterations;
            j["edgeFalloffWidth"] = edgeFalloffWidth;
            j["edgeFalloffValue"] = edgeFalloffValue;
        }
        
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("sourceMode")) sourceMode = static_cast<SourceMode>(j["sourceMode"].get<int>());
            if (j.contains("filePath")) {
                std::string path = j["filePath"].get<std::string>();
                std::strncpy(filePath, path.c_str(), sizeof(filePath) - 1);
                filePath[sizeof(filePath) - 1] = '\0';
            }
            if (j.contains("heightScale")) heightScale = j["heightScale"].get<float>();
            if (j.contains("maintainAspectRatio")) maintainAspectRatio = j["maintainAspectRatio"].get<bool>();
            if (j.contains("maxResolution")) maxResolution = j["maxResolution"].get<int>();
            if (j.contains("smoothIterations")) smoothIterations = j["smoothIterations"].get<int>();
            if (j.contains("edgeFalloffWidth")) edgeFalloffWidth = j["edgeFalloffWidth"].get<float>();
            if (j.contains("edgeFalloffValue")) edgeFalloffValue = j["edgeFalloffValue"].get<float>();
            
            // Reload file if path exists
            if (sourceMode == SourceMode::File && strlen(filePath) > 0) {
                loadHeightmapFromFile();
            }
        }
    };

    /**
     * @brief Hardness Input Node - reads the current hardness map from terrain
     */
    class HardnessInputNode : public TerrainNodeBase {
    public:
        HardnessInputNode() {
            name = "Hardness Input";
            terrainNodeType = NodeType::HardnessInput;
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Hardness Input";
            metadata.category = "Input";
            metadata.headerColor = IM_COL32(50, 100, 150, 255);
            headerColor = ImVec4(0.2f, 0.4f, 0.6f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        std::string getTypeId() const override { return "TerrainV2.HardnessInput"; }
        
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
        }
    };

    // ============================================================================
    // NOISE GENERATOR
    // ============================================================================
    
    enum class NoiseType { 
        Perlin,      // Gradient FBM noise
        Voronoi,     // Worley cell noise
        Simplex,     // Hash-based FBM
        Ridge,       // Ridge/mountain chains
        Billow,      // Soft rolling hills
        Warped,      // Domain-warped organic
        // FFT-accelerated noise types (uses CUDA if available, CPU fallback otherwise)
        FFT_Ocean,   // Phillips spectrum - ocean-like terrain
        FFT_Ridge,   // FFT-based sharp mountain ridges
        FFT_Billow,  // FFT-based soft hills
        FFT_Turb     // FFT-based turbulence
    };
    
    class NoiseGeneratorNode : public TerrainNodeBase {
    public:
        NoiseType noiseType = NoiseType::Perlin;
        int seed = 1337;
        float scale = 0.1f;            // Terrain-appropriate scale
        float frequency = 0.01f;       // Detail frequency
        float amplitude = 1.0f;       // Reasonable height range
        int octaves = 6;
        float persistance = 0.5f;
        float lacunarity = 2.0f;
        float jitter = 1.0f;           // Voronoi specific
        float warp_strength = 0.3f;    // Domain warp intensity
        float ridge_offset = 1.0f;     // Ridge noise offset
        
        NoiseGeneratorNode() {
            name = "Noise Generator";
            terrainNodeType = NodeType::NoiseGenerator;
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Noise Generator";
            metadata.category = "Input";
            metadata.headerColor = IM_COL32(50, 150, 125, 255);
            headerColor = ImVec4(0.2f, 0.6f, 0.5f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        
        void drawContent() override {
            const char* noiseNames[] = { 
                "Perlin", "Voronoi", "Simplex", "Ridge", "Billow", "Warped",
                // FFT types
                "FFT Ocean", "FFT Ridge", "FFT Billow", "FFT Turb"
            };
            int noiseIdx = (int)noiseType;
            if (ImGui::Combo("Type", &noiseIdx, noiseNames, 10)) {
                noiseType = (NoiseType)noiseIdx;
                dirty = true;
            }
            
            // Show FFT status for FFT types
            if (noiseIdx >= 6) {
                // FFT type selected - show CUDA status
                ImGui::TextDisabled("Uses CUDA if available");
            }
            
            if (ImGui::DragInt("Seed", &seed)) dirty = true;
            if (ImGui::DragFloat("Scale", &scale, 0.1f, 0.1f, 10.0f)) dirty = true;
            if (ImGui::DragFloat("Frequency", &frequency, 0.0001f, 0.0001f, 0.1f, "%.4f")) dirty = true;
            if (ImGui::DragFloat("Amplitude", &amplitude, 1.0f, 1.0f, 1000.0f)) dirty = true;
            if (ImGui::DragInt("Octaves", &octaves, 1, 1, 12)) dirty = true;
            if (ImGui::DragFloat("Persistance", &persistance, 0.01f, 0.0f, 1.0f)) dirty = true;
            if (ImGui::DragFloat("Lacunarity", &lacunarity, 0.1f, 1.0f, 4.0f)) dirty = true;
            
            // Type-specific parameters
            if (noiseType == NoiseType::Voronoi) {
                if (ImGui::DragFloat("Jitter", &jitter, 0.01f, 0.0f, 2.0f)) dirty = true;
            }
            if (noiseType == NoiseType::Ridge || noiseType == NoiseType::FFT_Ridge) {
                if (ImGui::DragFloat("Ridge Offset", &ridge_offset, 0.01f, 0.5f, 2.0f)) dirty = true;
            }
            if (noiseType == NoiseType::Warped) {
                if (ImGui::DragFloat("Warp Strength", &warp_strength, 0.01f, 0.0f, 1.0f)) dirty = true;
            }
        }
        
        std::string getTypeId() const override { return "TerrainV2.NoiseGenerator"; }
        
        // Serialization overrides
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["noiseType"] = static_cast<int>(noiseType);
            j["seed"] = seed;
            j["scale"] = scale;
            j["frequency"] = frequency;
            j["amplitude"] = amplitude;
            j["octaves"] = octaves;
            j["persistance"] = persistance;
            j["lacunarity"] = lacunarity;
            j["jitter"] = jitter;
            j["warp_strength"] = warp_strength;
            j["ridge_offset"] = ridge_offset;
        }
        
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("noiseType")) noiseType = static_cast<NoiseType>(j["noiseType"].get<int>());
            if (j.contains("seed")) seed = j["seed"].get<int>();
            if (j.contains("scale")) scale = j["scale"].get<float>();
            if (j.contains("frequency")) frequency = j["frequency"].get<float>();
            if (j.contains("amplitude")) amplitude = j["amplitude"].get<float>();
            if (j.contains("octaves")) octaves = j["octaves"].get<int>();
            if (j.contains("persistance")) persistance = j["persistance"].get<float>();
            if (j.contains("lacunarity")) lacunarity = j["lacunarity"].get<float>();
            if (j.contains("jitter")) jitter = j["jitter"].get<float>();
            if (j.contains("warp_strength")) warp_strength = j["warp_strength"].get<float>();
            if (j.contains("ridge_offset")) ridge_offset = j["ridge_offset"].get<float>();
        }
    };

    // ============================================================================
    // EROSION NODES
    // ============================================================================
    
    class HydraulicErosionNode : public TerrainNodeBase {
    public:
        HydraulicErosionParams params;
        bool useGPU = true;
        
        // Edge Falloff Settings
        float edgeFalloffWidth = 0.0f;
        float edgeFalloffValue = 0.0f;
        
        HydraulicErosionNode() {
            name = "Hydraulic Erosion";
            terrainNodeType = NodeType::HydraulicErosion;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Erosion Map", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, 4));
            
            metadata.displayName = "Hydraulic Erosion";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(75, 125, 200, 255);
            headerColor = ImVec4(0.3f, 0.5f, 0.8f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.HydraulicErosion"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["useGPU"] = useGPU;
            j["params"] = {
                {"iterations", params.iterations},
                {"dropletLifetime", params.dropletLifetime},
                {"inertia", params.inertia},
                {"sedimentCapacity", params.sedimentCapacity},
                {"minSlope", params.minSlope},
                {"erodeSpeed", params.erodeSpeed},
                {"depositSpeed", params.depositSpeed},
                {"evaporateSpeed", params.evaporateSpeed},
                {"gravity", params.gravity},
                {"erosionRadius", params.erosionRadius}
            };
            j["edgeFalloffWidth"] = edgeFalloffWidth;
            j["edgeFalloffValue"] = edgeFalloffValue;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("useGPU")) useGPU = j["useGPU"].get<bool>();
            if (j.contains("params")) {
                const auto& p = j["params"];
                params.iterations = p.value("iterations", params.iterations);
                params.dropletLifetime = p.value("dropletLifetime", params.dropletLifetime);
                params.inertia = p.value("inertia", params.inertia);
                params.sedimentCapacity = p.value("sedimentCapacity", params.sedimentCapacity);
                params.minSlope = p.value("minSlope", params.minSlope);
                params.erodeSpeed = p.value("erodeSpeed", params.erodeSpeed);
                params.depositSpeed = p.value("depositSpeed", params.depositSpeed);
                params.evaporateSpeed = p.value("evaporateSpeed", params.evaporateSpeed);
                params.gravity = p.value("gravity", params.gravity);
                params.erosionRadius = p.value("erosionRadius", params.erosionRadius);
            }
            if (j.contains("edgeFalloffWidth")) edgeFalloffWidth = j["edgeFalloffWidth"].get<float>();
            if (j.contains("edgeFalloffValue")) edgeFalloffValue = j["edgeFalloffValue"].get<float>();
        }
    };

    class ThermalErosionNode : public TerrainNodeBase {
    public:
        ThermalErosionParams params;
        bool useGPU = true;
        
        // Edge Falloff Settings
        float edgeFalloffWidth = 0.0f;
        float edgeFalloffValue = 0.0f;
        
        ThermalErosionNode() {
            params.iterations = 25; // Even more conservative default
            params.erosionAmount = 0.2f;
            name = "Thermal Erosion";
            terrainNodeType = NodeType::ThermalErosion;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Thermal Erosion";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(75, 125, 200, 255);
            headerColor = ImVec4(0.3f, 0.5f, 0.8f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.ThermalErosion"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["useGPU"] = useGPU;
            j["params"] = {
                {"iterations", params.iterations},
                {"talusAngle", params.talusAngle},
                {"erosionAmount", params.erosionAmount}
            };
            j["edgeFalloffWidth"] = edgeFalloffWidth;
            j["edgeFalloffValue"] = edgeFalloffValue;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("useGPU")) useGPU = j["useGPU"].get<bool>();
            if (j.contains("params")) {
                const auto& p = j["params"];
                params.iterations = p.value("iterations", params.iterations);
                params.talusAngle = p.value("talusAngle", params.talusAngle);
                params.erosionAmount = p.value("erosionAmount", params.erosionAmount);
            }
            if (j.contains("edgeFalloffWidth")) edgeFalloffWidth = j["edgeFalloffWidth"].get<float>();
            if (j.contains("edgeFalloffValue")) edgeFalloffValue = j["edgeFalloffValue"].get<float>();
        }
    };

    class FluvialErosionNode : public TerrainNodeBase {
    public:
        HydraulicErosionParams params;
        bool useGPU = true; // Use GPU by default
        
        // Edge Falloff Settings
        float edgeFalloffWidth = 0.0f;
        float edgeFalloffValue = 0.0f;
        
        FluvialErosionNode() {
            name = "Fluvial Erosion";
            terrainNodeType = NodeType::FluvialErosion;
            
            // Long-lived runoff parcels are required for catchment-scale channels.
            // These defaults describe a persistent river-forming rainfall event,
            // while existing serialized nodes retain their authored values.
            params.iterations = 250000;
            params.dropletLifetime = 384;
            params.inertia = 0.25f;
            params.sedimentCapacity = 1.5f;
            params.erodeSpeed = 0.12f;
            params.depositSpeed = 0.20f;
            params.evaporateSpeed = 0.001f;
            params.erosionRadius = 4;
            params.minSlope = 0.003f;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Guide", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Erosion Map", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, 4));
            
            metadata.displayName = "Fluvial Erosion";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(75, 150, 230, 255);
            headerColor = ImVec4(0.3f, 0.6f, 0.9f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.FluvialErosion"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["useGPU"] = useGPU;
            j["params"] = {
                {"iterations", params.iterations},
                {"dropletLifetime", params.dropletLifetime},
                {"inertia", params.inertia},
                {"sedimentCapacity", params.sedimentCapacity},
                {"minSlope", params.minSlope},
                {"erodeSpeed", params.erodeSpeed},
                {"depositSpeed", params.depositSpeed},
                {"evaporateSpeed", params.evaporateSpeed},
                {"gravity", params.gravity},
                {"erosionRadius", params.erosionRadius}
            };
            j["edgeFalloffWidth"] = edgeFalloffWidth;
            j["edgeFalloffValue"] = edgeFalloffValue;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("useGPU")) useGPU = j["useGPU"].get<bool>();
            if (j.contains("params")) {
                const auto& p = j["params"];
                params.iterations = p.value("iterations", params.iterations);
                params.dropletLifetime = p.value("dropletLifetime", params.dropletLifetime);
                params.inertia = p.value("inertia", params.inertia);
                params.sedimentCapacity = p.value("sedimentCapacity", params.sedimentCapacity);
                params.minSlope = p.value("minSlope", params.minSlope);
                params.erodeSpeed = p.value("erodeSpeed", params.erodeSpeed);
                params.depositSpeed = p.value("depositSpeed", params.depositSpeed);
                params.evaporateSpeed = p.value("evaporateSpeed", params.evaporateSpeed);
                params.gravity = p.value("gravity", params.gravity);
                params.erosionRadius = p.value("erosionRadius", params.erosionRadius);
            }
            if (j.contains("edgeFalloffWidth")) edgeFalloffWidth = j["edgeFalloffWidth"].get<float>();
            if (j.contains("edgeFalloffValue")) edgeFalloffValue = j["edgeFalloffValue"].get<float>();
        }
    };

    class WindErosionNode : public TerrainNodeBase {
    public:
        float strength = 0.2f;   // Reduced default
        float direction = 45.0f;
        int iterations = 10;     // Reduced default
        bool useGPU = true;
        
        // Edge Falloff Settings
        float edgeFalloffWidth = 0.0f;
        float edgeFalloffValue = 0.0f;
        
        WindErosionNode() {
            name = "Wind Erosion";
            terrainNodeType = NodeType::WindErosion;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Wind Erosion";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(180, 150, 100, 255);
            headerColor = ImVec4(0.7f, 0.6f, 0.4f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.WindErosion"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["useGPU"] = useGPU;
            j["strength"] = strength;
            j["direction"] = direction;
            j["iterations"] = iterations;
            j["edgeFalloffWidth"] = edgeFalloffWidth;
            j["edgeFalloffValue"] = edgeFalloffValue;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("useGPU")) useGPU = j["useGPU"].get<bool>();
            if (j.contains("strength")) strength = j["strength"].get<float>();
            if (j.contains("direction")) direction = j["direction"].get<float>();
            if (j.contains("iterations")) iterations = j["iterations"].get<int>();
            if (j.contains("edgeFalloffWidth")) edgeFalloffWidth = j["edgeFalloffWidth"].get<float>();
            if (j.contains("edgeFalloffValue")) edgeFalloffValue = j["edgeFalloffValue"].get<float>();
        }
    };

    // ============================================================================
    // SEDIMENT DEPOSITION NODES
    // ============================================================================
    
    /**
     * @brief Sediment Deposition - Simulates sediment accumulation in low-slope areas
     * 
     * Flow-based sediment transport: erodes high slopes, deposits in valleys.
     */
    class SedimentDepositionNode : public TerrainNodeBase {
    public:
        int iterations = 20;              // Simulation iterations
        float depositionRate = 0.3f;      // Rate of sediment settling
        float transportCapacity = 1.0f;   // Max sediment per flow unit
        float settlingSpeed = 0.5f;       // How fast sediment settles
        bool useGPU = false;              // GPU acceleration (future)
        
        SedimentDepositionNode() {
            name = "Sediment Deposition";
            terrainNodeType = NodeType::SedimentDeposition;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Sediment Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Sediment Deposit";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(150, 120, 80, 255);
            headerColor = ImVec4(0.6f, 0.5f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.SedimentDeposition"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["iterations"] = iterations;
            j["depositionRate"] = depositionRate;
            j["transportCapacity"] = transportCapacity;
            j["settlingSpeed"] = settlingSpeed;
            j["useGPU"] = useGPU;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("iterations")) iterations = j["iterations"].get<int>();
            if (j.contains("depositionRate")) depositionRate = j["depositionRate"].get<float>();
            if (j.contains("transportCapacity")) transportCapacity = j["transportCapacity"].get<float>();
            if (j.contains("settlingSpeed")) settlingSpeed = j["settlingSpeed"].get<float>();
            if (j.contains("useGPU")) useGPU = j["useGPU"].get<bool>();
        }
    };
    
    /**
     * @brief Alluvial Fan - Creates fan-shaped deposits at mountain bases
     * 
     * Detects steep-to-flat transitions and spreads sediment in a fan pattern.
     */
    class AlluvialFanNode : public TerrainNodeBase {
    public:
        float slopeThreshold = 30.0f;     // Degrees: steep-to-flat transition
        float fanSpreadAngle = 60.0f;     // Fan opening angle (degrees)
        float depositionStrength = 0.5f;  // Strength of deposition
        int fanLength = 50;               // Max fan length in pixels
        
        AlluvialFanNode() {
            name = "Alluvial Fan";
            terrainNodeType = NodeType::AlluvialFan;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Slope Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Fan Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Alluvial Fan";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(180, 140, 90, 255);
            headerColor = ImVec4(0.7f, 0.55f, 0.35f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.AlluvialFan"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["slopeThreshold"] = slopeThreshold;
            j["fanSpreadAngle"] = fanSpreadAngle;
            j["depositionStrength"] = depositionStrength;
            j["fanLength"] = fanLength;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("slopeThreshold")) slopeThreshold = j["slopeThreshold"].get<float>();
            if (j.contains("fanSpreadAngle")) fanSpreadAngle = j["fanSpreadAngle"].get<float>();
            if (j.contains("depositionStrength")) depositionStrength = j["depositionStrength"].get<float>();
            if (j.contains("fanLength")) fanLength = j["fanLength"].get<int>();
        }
    };
    
    /**
     * @brief Delta Formation - Creates river delta patterns at low elevations
     * 
     * Uses flow accumulation to identify river mouths and builds branching deltas.
     */
    class DeltaFormationNode : public TerrainNodeBase {
    public:
        float seaLevel = 0.1f;            // Height threshold for delta formation
        float deltaSpread = 45.0f;        // Delta spreading angle
        int branchingFactor = 3;          // Number of delta branches
        float sedimentRatio = 0.4f;       // Sediment deposit height ratio
        
        DeltaFormationNode() {
            name = "Delta Formation";
            terrainNodeType = NodeType::DeltaFormation;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Delta Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Delta Formation";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(100, 140, 180, 255);
            headerColor = ImVec4(0.4f, 0.55f, 0.7f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.DeltaFormation"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["seaLevel"] = seaLevel;
            j["deltaSpread"] = deltaSpread;
            j["branchingFactor"] = branchingFactor;
            j["sedimentRatio"] = sedimentRatio;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("seaLevel")) seaLevel = j["seaLevel"].get<float>();
            if (j.contains("deltaSpread")) deltaSpread = j["deltaSpread"].get<float>();
            if (j.contains("branchingFactor")) branchingFactor = j["branchingFactor"].get<int>();
            if (j.contains("sedimentRatio")) sedimentRatio = j["sedimentRatio"].get<float>();
        }
    };
    
    // ============================================================================
    // EROSION WIZARD NODE
    // ============================================================================
    
    /**
     * @brief Erosion Wizard - All-in-one erosion with geologic/cinematic presets
     * 
     * Combines multiple erosion types with preset configurations for:
     * - Geology education (time-scale erosion simulation)
     * - Film industry (quick dramatic terrain transformations)
     * - Game development (realistic terrain aging)
     */
    
    enum class ErosionPreset {
        Custom = 0,           // User-defined parameters
        YoungMountains,       // 1-10 My: Sharp peaks, V-valleys, active uplift
        MatureMountains,      // 10-50 My: Rounded peaks, wider valleys
        AncientPlateau,       // 100+ My: Peneplain, gentle hills
        TropicalRainforest,   // High rainfall, deep chemical weathering
        AridDesert,           // Wind dominant, mesas, buttes
        GlacialCarving,       // U-valleys, cirques, moraines
        CoastalErosion,       // Sea cliffs, wave-cut platforms
        VolcanicTerrain,      // Lava flows, calderas, tephra
        RiverDelta            // Fluvial deposition, braided channels
    };
    
    class ErosionWizardNode : public TerrainNodeBase {
    public:
        // Preset selection
        ErosionPreset preset = ErosionPreset::MatureMountains;
        
        // Time scale (millions of years simulation)
        float timeScaleMy = 10.0f;  // Millions of years
        
        // Climate modifiers (0-2 range, 1 = normal)
        float rainfallFactor = 0.2f;      // Affects hydraulic erosion
        float temperatureFactor = 0.2f;   // Affects thermal erosion
        float windFactor = 0.2f;          // Affects wind erosion
        
        // Quality/Performance
        int qualityLevel = 2;             // 1=Fast, 2=Medium, 3=High
        bool useGPU = true;
        
        // Output options
        bool outputErosionMask = true;    // Shows where erosion occurred
        
        // Interactive Simulation State
        bool isSimulating = false;
        int currentPass = 0;
        int totalPasses = 0;
        std::vector<float> originalHeight; // For mask calculation
        
        // Erosion parameters for current run
        int hydraulicItersPerPass = 0;
        int thermalItersPerPass = 0;
        int windItersPerPass = 0;
        
        TerrainObject* cachedTerrain = nullptr; // Cached during compute for UI updates
        std::vector<float> cachedMask;          // Cached mask for simulation
        
        ErosionWizardNode() {
            name = "Erosion Wizard";
            terrainNodeType = NodeType::ErosionWizard;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Erosion Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Erosion Wizard";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(255, 180, 50, 255);  // Gold - stands out
            headerColor = ImVec4(1.0f, 0.7f, 0.2f, 1.0f);
        }
        
        // Helper to get preset name for UI
        static const char* getPresetName(ErosionPreset p);
        
        // Apply preset configuration
        void applyPreset(ErosionPreset p);
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.ErosionWizard"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["preset"] = static_cast<int>(preset);
            j["timeScaleMy"] = timeScaleMy;
            j["rainfallFactor"] = rainfallFactor;
            j["temperatureFactor"] = temperatureFactor;
            j["windFactor"] = windFactor;
            j["qualityLevel"] = qualityLevel;
            j["useGPU"] = useGPU;
            j["outputErosionMask"] = outputErosionMask;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("preset")) preset = static_cast<ErosionPreset>(j["preset"].get<int>());
            if (j.contains("timeScaleMy")) timeScaleMy = j["timeScaleMy"].get<float>();
            if (j.contains("rainfallFactor")) rainfallFactor = j["rainfallFactor"].get<float>();
            if (j.contains("temperatureFactor")) temperatureFactor = j["temperatureFactor"].get<float>();
            if (j.contains("windFactor")) windFactor = j["windFactor"].get<float>();
            if (j.contains("qualityLevel")) qualityLevel = j["qualityLevel"].get<int>();
            if (j.contains("useGPU")) useGPU = j["useGPU"].get<bool>();
            if (j.contains("outputErosionMask")) outputErosionMask = j["outputErosionMask"].get<bool>();
        }
    };

    // ============================================================================
    // OUTPUT NODES
    // ============================================================================
    
    class HeightOutputNode : public TerrainNodeBase {
    public:
        HeightOutputNode() {
            name = "Height Output";
            terrainNodeType = NodeType::HeightOutput;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Height Output";
            metadata.category = "Output";
            metadata.headerColor = IM_COL32(200, 100, 75, 255);
            headerColor = ImVec4(0.8f, 0.4f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        std::string getTypeId() const override { return "TerrainV2.HeightOutput"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
        }
    };

    class SplatOutputNode : public TerrainNodeBase {
    public:
        char exportPath[256] = "";
        bool browseForExport = false;
        bool autoApplyToTerrain = true;
        
        SplatOutputNode() {
            name = "Splat Output";
            terrainNodeType = NodeType::SplatOutput;
            
            // Explicit four-channel splat payload. Single-channel masks must be
            // combined through Splat Compose first.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, false, 4));
            
            metadata.displayName = "Splat Output";
            metadata.category = "Output";
            metadata.headerColor = IM_COL32(200, 75, 125, 255);
            headerColor = ImVec4(0.8f, 0.3f, 0.5f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        void exportSplatMap(TerrainObject* terrain);
        std::string getTypeId() const override { return "TerrainV2.SplatOutput"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["exportPath"] = std::string(exportPath);
            j["autoApplyToTerrain"] = autoApplyToTerrain;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("exportPath")) {
                std::string path = j["exportPath"].get<std::string>();
                strncpy(exportPath, path.c_str(), sizeof(exportPath) - 1);
            }
            if (j.contains("autoApplyToTerrain")) autoApplyToTerrain = j["autoApplyToTerrain"].get<bool>();
        }
    };

    /**
     * @brief Hardness Output Node - Drives the physical hardness of the terrain
     * 
     * Values from 0 (Soft/Soil) to 1 (Hard/Bedrock).
     * This data is used by erosion algorithms and physics simulations.
     */
    class HardnessOutputNode : public TerrainNodeBase {
    public:
        HardnessOutputNode() {
            name = "Hardness Output";
            terrainNodeType = NodeType::HardnessOutput;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Hardness Output";
            metadata.category = "Output";
            metadata.headerColor = IM_COL32(120, 130, 140, 255);
            headerColor = ImVec4(0.5f, 0.55f, 0.6f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        std::string getTypeId() const override { return "TerrainV2.HardnessOutput"; }
        
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
        }
    };

    // ============================================================================
    // MATH NODES
    // ============================================================================
    
    enum class MathOp { Add, Subtract, Multiply, Divide, Min, Max };
    
    class MathNode : public TerrainNodeBase {
    public:
        MathOp operation = MathOp::Add;
        float factor = 1.0f;
        
        MathNode() {
            name = "Math";
            terrainNodeType = NodeType::Add;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "A", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "B", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Result", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Math";
            metadata.category = "Math";
            metadata.headerColor = IM_COL32(100, 100, 150, 255);
            headerColor = ImVec4(0.4f, 0.4f, 0.6f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Math"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["operation"] = static_cast<int>(operation);
            j["factor"] = factor;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("operation")) operation = static_cast<MathOp>(j["operation"].get<int>());
            if (j.contains("factor")) factor = j["factor"].get<float>();
        }
    };

    class BlendNode : public TerrainNodeBase {
    public:
        float alpha = 0.5f;
        
        BlendNode() {
            name = "Blend";
            terrainNodeType = NodeType::Blend;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "A", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "B", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Result", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Blend";
            metadata.category = "Math";
            metadata.headerColor = IM_COL32(125, 100, 150, 255);
            headerColor = ImVec4(0.5f, 0.4f, 0.6f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Blend"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["alpha"] = alpha;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("alpha")) alpha = j["alpha"].get<float>();
        }
    };

    class ClampNode : public TerrainNodeBase {
    public:
        float minVal = 0.0f;
        float maxVal = 1.0f;
        
        ClampNode() {
            name = "Clamp";
            terrainNodeType = NodeType::Clamp;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Clamp";
            metadata.category = "Math";
            metadata.headerColor = IM_COL32(100, 100, 100, 255);
            headerColor = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Clamp"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["minVal"] = minVal;
            j["maxVal"] = maxVal;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("minVal")) minVal = j["minVal"].get<float>();
            if (j.contains("maxVal")) maxVal = j["maxVal"].get<float>();
        }
    };

    class InvertNode : public TerrainNodeBase {
    public:
        InvertNode() {
            name = "Invert";
            terrainNodeType = NodeType::Invert;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Invert";
            metadata.category = "Math";
            metadata.headerColor = IM_COL32(100, 100, 100, 255);
            headerColor = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        std::string getTypeId() const override { return "TerrainV2.Invert"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
        }
    };

    // ============================================================================
    // MASK NODES
    // ============================================================================
    
    class SlopeMaskNode : public TerrainNodeBase {
    public:
        // Default to select medium-steep slopes (cliff-like terrain)
        float minSlope = 20.0f;  // 20 degrees minimum
        float maxSlope = 60.0f;  // 60 degrees maximum
        float falloff = 0.2f;
        
        SlopeMaskNode() {
            name = "Slope Mask";
            terrainNodeType = NodeType::SlopeMask;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Slope Mask";
            metadata.category = "Mask";
            metadata.headerColor = IM_COL32(180, 180, 75, 255);
            headerColor = ImVec4(0.7f, 0.7f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.SlopeMask"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["minSlope"] = minSlope;
            j["maxSlope"] = maxSlope;
            j["falloff"] = falloff;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("minSlope")) minSlope = j["minSlope"].get<float>();
            if (j.contains("maxSlope")) maxSlope = j["maxSlope"].get<float>();
            if (j.contains("falloff")) falloff = j["falloff"].get<float>();
        }
    };

    class HeightMaskNode : public TerrainNodeBase {
    public:
        // Default to select mid-range heights (in physical terrain units, scale_y)
        float minHeight = 2.0f;   // Lower threshold
        float maxHeight = 8.0f;   // Upper threshold (typical scale_y=10)
        float falloff = 2.0f;
        
        HeightMaskNode() {
            name = "Height Mask";
            terrainNodeType = NodeType::HeightMask;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Height Mask";
            metadata.category = "Mask";
            metadata.headerColor = IM_COL32(180, 180, 75, 255);
            headerColor = ImVec4(0.7f, 0.7f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.HeightMask"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["minHeight"] = minHeight;
            j["maxHeight"] = maxHeight;
            j["falloff"] = falloff;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("minHeight")) minHeight = j["minHeight"].get<float>();
            if (j.contains("maxHeight")) maxHeight = j["maxHeight"].get<float>();
            if (j.contains("falloff")) falloff = j["falloff"].get<float>();
        }
    };

    class CurvatureMaskNode : public TerrainNodeBase {
    public:
        float minCurve = 0.0f;
        float maxCurve = 1.0f;
        bool selectConvex = true;
        
        CurvatureMaskNode() {
            name = "Curvature Mask";
            terrainNodeType = NodeType::CurvatureMask;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Curvature Mask";
            metadata.category = "Mask";
            metadata.headerColor = IM_COL32(180, 180, 75, 255);
            headerColor = ImVec4(0.7f, 0.7f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.CurvatureMask"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["minCurve"] = minCurve;
            j["maxCurve"] = maxCurve;
            j["selectConvex"] = selectConvex;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("minCurve")) minCurve = j["minCurve"].get<float>();
            if (j.contains("maxCurve")) maxCurve = j["maxCurve"].get<float>();
            if (j.contains("selectConvex")) selectConvex = j["selectConvex"].get<bool>();
        }
    };
    
    /**
     * @brief Flow Mask - Simulates where soil/sediment would accumulate
     * 
     * Uses flow accumulation algorithm to find valleys and depressions.
     */
    class FlowMaskNode : public TerrainNodeBase {
    public:
        int detailLevel = 6;          // 1=main rivers, 8=finest tributaries
        int bankSpread = 1;           // Small channel-width expansion, never global blur
        float strength = 1.0f;        // Flow strength multiplier
        float decay = 0.995f;         // Discharge retained at each downstream step
        float channelSoftness = 0.06f;// Soft transition around the selected detail threshold
        bool normalize = true;        // Normalize output to 0-1
        
        FlowMaskNode() {
            name = "Flow Mask";
            terrainNodeType = NodeType::FlowMask;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Flow / Soil";
            metadata.category = "Mask";
            metadata.headerColor = IM_COL32(100, 150, 200, 255);
            headerColor = ImVec4(0.4f, 0.6f, 0.8f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.FlowMask"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["detailLevel"] = detailLevel;
            j["bankSpread"] = bankSpread;
            j["strength"] = strength;
            j["decay"] = decay;
            j["channelSoftness"] = channelSoftness;
            j["normalize"] = normalize;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("detailLevel")) {
                detailLevel = clampValue(j["detailLevel"].get<int>(), 1, 8);
            } else if (j.contains("iterations")) {
                // Backward compatibility: the old 1..32 blur count becomes a
                // genuine 1..8 tributary-detail selection.
                detailLevel = clampValue((j["iterations"].get<int>() + 3) / 4, 1, 8);
            }
            if (j.contains("bankSpread")) bankSpread = clampValue(j["bankSpread"].get<int>(), 0, 4);
            if (j.contains("strength")) strength = j["strength"].get<float>();
            if (j.contains("decay")) decay = clampValue(j["decay"].get<float>(), 0.95f, 1.0f);
            if (j.contains("channelSoftness")) channelSoftness = clampValue(j["channelSoftness"].get<float>(), 0.01f, 0.20f);
            if (j.contains("normalize")) normalize = j["normalize"].get<bool>();
        }
    };
    
    /**
     * @brief Exposure Mask - Sun-facing direction based mask
     * 
     * Calculates how much each point faces a given direction (e.g., south for snow).
     */
    class ExposureMaskNode : public TerrainNodeBase {
    public:
        float sunAzimuth = 180.0f;    // Sun direction (0=North, 90=East, 180=South)
        float sunElevation = 45.0f;   // Sun elevation angle
        float contrast = 1.0f;        // Output contrast
        bool invert = false;          // Invert for shadow areas
        
        ExposureMaskNode() {
            name = "Exposure Mask";
            terrainNodeType = NodeType::ExposureMask;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Sun Exposure";
            metadata.category = "Mask";
            metadata.headerColor = IM_COL32(220, 180, 80, 255);
            headerColor = ImVec4(0.85f, 0.7f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.ExposureMask"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["sunAzimuth"] = sunAzimuth;
            j["sunElevation"] = sunElevation;
            j["contrast"] = contrast;
            j["invert"] = invert;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("sunAzimuth")) sunAzimuth = j["sunAzimuth"].get<float>();
            if (j.contains("sunElevation")) sunElevation = j["sunElevation"].get<float>();
            if (j.contains("contrast")) contrast = j["contrast"].get<float>();
            if (j.contains("invert")) invert = j["invert"].get<bool>();
        }
    };

    // ============================================================================
    // NEW OPERATOR NODES
    // ============================================================================
    
    /**
     * @brief Smooth Node - Apply blur/smoothing filter to height data
     */
    class SmoothNode : public TerrainNodeBase {
    public:
        int iterations = 3;
        float strength = 0.5f;
        int kernelSize = 3; // 3, 5, or 7
        
        SmoothNode() {
            name = "Smooth";
            terrainNodeType = NodeType::Smooth;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Smooth";
            metadata.category = "Filter";
            metadata.headerColor = IM_COL32(100, 150, 200, 255);
            headerColor = ImVec4(0.4f, 0.6f, 0.8f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Smooth"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["iterations"] = iterations;
            j["strength"] = strength;
            j["kernelSize"] = kernelSize;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("iterations")) iterations = j["iterations"].get<int>();
            if (j.contains("strength")) strength = j["strength"].get<float>();
            if (j.contains("kernelSize")) kernelSize = j["kernelSize"].get<int>();
        }
    };

    /**
     * @brief Normalize Node - Scale height data to specified range
     */
    class NormalizeNode : public TerrainNodeBase {
    public:
        float minOutput = 0.0f;
        float maxOutput = 100.0f;
        bool autoRange = true;
        
        NormalizeNode() {
            name = "Normalize";
            terrainNodeType = NodeType::Normalize;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Normalize";
            metadata.category = "Filter";
            metadata.headerColor = IM_COL32(100, 150, 200, 255);
            headerColor = ImVec4(0.4f, 0.6f, 0.8f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Normalize"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["minOutput"] = minOutput;
            j["maxOutput"] = maxOutput;
            j["autoRange"] = autoRange;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("minOutput")) minOutput = j["minOutput"].get<float>();
            if (j.contains("maxOutput")) maxOutput = j["maxOutput"].get<float>();
            if (j.contains("autoRange")) autoRange = j["autoRange"].get<bool>();
        }
    };

    /**
     * @brief Terrace Node - Create stepped terrain levels
     */
    class TerraceNode : public TerrainNodeBase {
    public:
        int levels = 8;
        float sharpness = 0.5f; // 0 = smooth ramps, 1 = hard steps
        float offset = 0.0f;
        
        TerraceNode() {
            name = "Terrace";
            terrainNodeType = NodeType::Terrace;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Terrace";
            metadata.category = "Filter";
            metadata.headerColor = IM_COL32(150, 120, 180, 255);
            headerColor = ImVec4(0.6f, 0.5f, 0.7f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Terrace"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["levels"] = levels;
            j["sharpness"] = sharpness;
            j["offset"] = offset;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("levels")) levels = j["levels"].get<int>();
            if (j.contains("sharpness")) sharpness = j["sharpness"].get<float>();
            if (j.contains("offset")) offset = j["offset"].get<float>();
        }
    };
    
    enum class FalloffMode { Linear, Smoothstep, Cosine };

    /**
     * @brief Edge Falloff - Smoothly fade terrain edges to a specific value
     */
    class EdgeFalloffNode : public TerrainNodeBase {
    public:
        float fadeWidth = 0.1f;
        float fadeValue = 0.0f;
        FalloffMode mode = FalloffMode::Smoothstep;

        EdgeFalloffNode() {
            name = "Edge Falloff";
            terrainNodeType = NodeType::EdgeFalloff;

            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));

            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));

            metadata.displayName = "Edge Falloff";
            metadata.category = "Filter";
            metadata.headerColor = IM_COL32(180, 100, 100, 255);
            headerColor = ImVec4(0.7f, 0.4f, 0.4f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.EdgeFalloff"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["fadeWidth"] = fadeWidth;
            j["fadeValue"] = fadeValue;
            j["mode"] = static_cast<int>(mode);
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("fadeWidth")) fadeWidth = j["fadeWidth"].get<float>();
            if (j.contains("fadeValue")) fadeValue = j["fadeValue"].get<float>();
            if (j.contains("mode")) mode = static_cast<FalloffMode>(j["mode"].get<int>());
        }
    };

    /**
     * @brief MaskCombine Node - Boolean/blend operations on masks
     */
    enum class MaskCombineOp { 
        AND,       // min(A, B)
        OR,        // max(A, B)
        XOR,       // abs(A - B)
        Multiply,  // A * B
        Add,       // A + B clamped
        Subtract,  // A - B clamped
        Difference // abs(A - B)
    };
    
    class MaskCombineNode : public TerrainNodeBase {
    public:
        MaskCombineOp operation = MaskCombineOp::Multiply;
        
        MaskCombineNode() {
            name = "Mask Combine";
            terrainNodeType = NodeType::MaskCombine;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask A", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask B", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Mask Combine";
            metadata.category = "Mask";
            metadata.headerColor = IM_COL32(180, 180, 75, 255);
            headerColor = ImVec4(0.7f, 0.7f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.MaskCombine"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["operation"] = static_cast<int>(operation);
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("operation")) operation = static_cast<MaskCombineOp>(j["operation"].get<int>());
        }
    };

    /**
     * @brief Overlay Node - Photoshop-style overlay blend mode
     * Result = A < 0.5 ? (2*A*B) : (1 - 2*(1-A)*(1-B))
     */
    class OverlayNode : public TerrainNodeBase {
    public:
        float strength = 1.0f;
        
        OverlayNode() {
            name = "Overlay";
            terrainNodeType = NodeType::Overlay;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Base", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Blend", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Result", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Overlay";
            metadata.category = "Blend";
            metadata.headerColor = IM_COL32(150, 100, 180, 255);
            headerColor = ImVec4(0.6f, 0.4f, 0.7f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Overlay"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["strength"] = strength;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("strength")) strength = j["strength"].get<float>();
        }
    };

    /**
     * @brief Screen Node - Photoshop-style screen blend mode
     * Result = 1 - (1-A)*(1-B)
     */
    class ScreenNode : public TerrainNodeBase {
    public:
        float strength = 1.0f;
        
        ScreenNode() {
            name = "Screen";
            terrainNodeType = NodeType::Screen;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Base", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Blend", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Result", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Screen";
            metadata.category = "Blend";
            metadata.headerColor = IM_COL32(150, 100, 180, 255);
            headerColor = ImVec4(0.6f, 0.4f, 0.7f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Screen"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["strength"] = strength;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("strength")) strength = j["strength"].get<float>();
        }
    };

    // ============================================================================
    // PROCEDURAL TEXTURE NODES
    // ============================================================================
    
    /**
     * @brief AutoSplat Node - Automatic terrain texturing based on height/slope
     * 
     * Generates 4-channel RGBA splat map where each channel represents a texture layer.
     * Rules are defined per-layer based on height range and slope range.
     */
    class AutoSplatNode : public TerrainNodeBase {
    public:
        // Per-layer rules for automatic texture assignment
        struct LayerRule {
            float heightMin = 0.0f;
            float heightMax = 1000.0f;
            float slopeMin = 0.0f;      // degrees (0 = flat)
            float slopeMax = 90.0f;     // degrees (90 = vertical)
            float heightWeight = 0.5f;
            float slopeWeight = 0.5f;
            float falloff = 10.0f;      // Transition smoothness
            float noiseAmount = 0.05f;  // Random variation at boundaries
            bool enabled = true;
        };
        
        LayerRule rules[4];  // R=Layer0, G=Layer1, B=Layer2, A=Layer3
        bool normalizeOutput = true;
        int noiseSeed = 42;
        
        AutoSplatNode() {
            name = "Auto Splat";
            terrainNodeType = NodeType::AutoSplat;
            
            // Default rules: Grass/Rock/Snow/Dirt
            // Layer 0 (R): Grass - flat, low-mid height
            rules[0] = { 0.0f, 50.0f, 0.0f, 25.0f, 0.5f, 0.5f, 10.0f, 0.05f, true };
            // Layer 1 (G): Rock - steep slopes
            rules[1] = { 0.0f, 200.0f, 30.0f, 90.0f, 0.2f, 0.8f, 5.0f, 0.03f, true };
            // Layer 2 (B): Snow - high altitude
            rules[2] = { 80.0f, 200.0f, 0.0f, 45.0f, 0.9f, 0.1f, 15.0f, 0.02f, true };
            // Layer 3 (A): Dirt/Sand - low, flat
            rules[3] = { 0.0f, 20.0f, 0.0f, 15.0f, 0.6f, 0.4f, 8.0f, 0.1f, false };
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            // 4-channel output for splat map
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, 4));
            
            metadata.displayName = "Auto Splat";
            metadata.category = "Texture";
            metadata.headerColor = IM_COL32(200, 150, 50, 255);
            headerColor = ImVec4(0.8f, 0.6f, 0.2f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.AutoSplat"; }
        float getCustomWidth() const override { return 160.0f; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["normalizeOutput"] = normalizeOutput;
            j["noiseSeed"] = noiseSeed;
            nlohmann::json rulesArray = nlohmann::json::array();
            for (int i = 0; i < 4; i++) {
                rulesArray.push_back({
                    {"heightMin", rules[i].heightMin},
                    {"heightMax", rules[i].heightMax},
                    {"slopeMin", rules[i].slopeMin},
                    {"slopeMax", rules[i].slopeMax},
                    {"heightWeight", rules[i].heightWeight},
                    {"slopeWeight", rules[i].slopeWeight},
                    {"falloff", rules[i].falloff},
                    {"noiseAmount", rules[i].noiseAmount},
                    {"enabled", rules[i].enabled}
                });
            }
            j["rules"] = rulesArray;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("normalizeOutput")) normalizeOutput = j["normalizeOutput"].get<bool>();
            if (j.contains("noiseSeed")) noiseSeed = j["noiseSeed"].get<int>();
            if (j.contains("rules") && j["rules"].is_array()) {
                const auto& rulesArray = j["rules"];
                for (int i = 0; i < 4 && i < (int)rulesArray.size(); i++) {
                    const auto& r = rulesArray[i];
                    rules[i].heightMin = r.value("heightMin", rules[i].heightMin);
                    rules[i].heightMax = r.value("heightMax", rules[i].heightMax);
                    rules[i].slopeMin = r.value("slopeMin", rules[i].slopeMin);
                    rules[i].slopeMax = r.value("slopeMax", rules[i].slopeMax);
                    rules[i].heightWeight = r.value("heightWeight", rules[i].heightWeight);
                    rules[i].slopeWeight = r.value("slopeWeight", rules[i].slopeWeight);
                    rules[i].falloff = r.value("falloff", rules[i].falloff);
                    rules[i].noiseAmount = r.value("noiseAmount", rules[i].noiseAmount);
                    rules[i].enabled = r.value("enabled", rules[i].enabled);
                }
            }
        }
    };

    /**
     * @brief MaskPaint Node - Paint masks directly in viewport
     * 
     * Allows users to paint mask values with a brush tool,
     * useful for manual touch-ups after procedural generation.
     */
    class MaskPaintNode : public TerrainNodeBase {
    public:
        std::vector<float> paintBuffer;
        int bufferWidth = 0;
        int bufferHeight = 0;
        
        // Brush settings
        float brushRadius = 20.0f;
        float brushStrength = 0.5f;
        float brushFalloff = 0.5f;  // 0 = hard edge, 1 = soft falloff
        
        // State
        bool isPainting = false;
        bool needsInit = true;
        
        MaskPaintNode() {
            name = "Mask Paint";
            terrainNodeType = NodeType::MaskPaint;
            
            // Optional height input for resolution reference
            inputs.push_back(NodeSystem::Pin::createInput(
                "Reference", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Mask Paint";
            metadata.category = "Mask";
            metadata.headerColor = IM_COL32(220, 100, 150, 255);
            headerColor = ImVec4(0.85f, 0.4f, 0.6f, 1.0f);
        }
        
        // Paint at UV coordinates
        void paint(float u, float v, float strength);
        void clear() { std::fill(paintBuffer.begin(), paintBuffer.end(), 0.0f); }
        void fill(float value) { std::fill(paintBuffer.begin(), paintBuffer.end(), value); }
        void initBuffer(int width, int height);
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.MaskPaint"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["brushRadius"] = brushRadius;
            j["brushStrength"] = brushStrength;
            j["brushFalloff"] = brushFalloff;
            j["bufferWidth"] = bufferWidth;
            j["bufferHeight"] = bufferHeight;
            if (!paintBuffer.empty()) {
                j["paintBuffer"] = paintBuffer;
            }
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("brushRadius")) brushRadius = j["brushRadius"].get<float>();
            if (j.contains("brushStrength")) brushStrength = j["brushStrength"].get<float>();
            if (j.contains("brushFalloff")) brushFalloff = j["brushFalloff"].get<float>();
            if (j.contains("bufferWidth")) bufferWidth = j["bufferWidth"].get<int>();
            if (j.contains("bufferHeight")) bufferHeight = j["bufferHeight"].get<int>();
            if (j.contains("paintBuffer") && j["paintBuffer"].is_array()) {
                paintBuffer = j["paintBuffer"].get<std::vector<float>>();
                constexpr int kMaxPaintDimension = 8192;
                const bool dimensionsValid = bufferWidth > 0 && bufferHeight > 0 &&
                    bufferWidth <= kMaxPaintDimension && bufferHeight <= kMaxPaintDimension;
                const size_t expected = dimensionsValid
                    ? static_cast<size_t>(bufferWidth) * static_cast<size_t>(bufferHeight)
                    : 0u;
                if (paintBuffer.size() == expected && expected > 0) {
                    needsInit = false;
                } else {
                    paintBuffer.clear();
                    bufferWidth = 0;
                    bufferHeight = 0;
                    needsInit = true;
                }
            }
        }
    };

    /**
     * @brief MaskImage Node - Load grayscale image as mask
     * 
     * Imports external PNG/JPG files as mask data.
     */
    class MaskImageNode : public TerrainNodeBase {
    public:
        char filePath[256] = "";
        std::vector<float> loadedMask;
        int loadedWidth = 0;
        int loadedHeight = 0;
        bool fileLoaded = false;
        bool browseForMask = false;
        
        // Adjustments
        float contrast = 1.0f;
        float brightness = 0.0f;
        bool invert = false;
        
        MaskImageNode() {
            name = "Mask Image";
            terrainNodeType = NodeType::MaskImage;
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Mask Image";
            metadata.category = "Mask";
            metadata.headerColor = IM_COL32(150, 200, 100, 255);
            headerColor = ImVec4(0.6f, 0.8f, 0.4f, 1.0f);
        }
        
        void loadMaskFromFile();
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.MaskImage"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["filePath"] = std::string(filePath);
            j["contrast"] = contrast;
            j["brightness"] = brightness;
            j["invert"] = invert;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("filePath")) {
                std::string path = j["filePath"].get<std::string>();
                strncpy(filePath, path.c_str(), sizeof(filePath) - 1);
                filePath[sizeof(filePath) - 1] = '\0';
            }
            if (j.contains("contrast")) contrast = j["contrast"].get<float>();
            if (j.contains("brightness")) brightness = j["brightness"].get<float>();
            if (j.contains("invert")) invert = j["invert"].get<bool>();
            if (strlen(filePath) > 0) loadMaskFromFile();
        }
    };

    // ============================================================================
    // GEOLOGICAL TRANSFORM NODES
    // ============================================================================
    
    /**
     * @brief Fault Node - Strike-slip fault line with lateral offset
     * 
     * Creates a fault line across terrain with configurable offset and direction.
     */
    class FaultNode : public TerrainNodeBase {
    public:
        float direction = 45.0f;      // Fault angle (0-360 degrees)
        float offset = 10.0f;         // Lateral offset (world units)
        float verticalOffset = 0.0f;  // Vertical displacement
        float width = 5.0f;           // Transition width (blur)
        float position = 0.5f;        // Fault position (0-1 normalized)
        
        FaultNode() {
            name = "Fault";
            terrainNodeType = NodeType::Fault;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Fault Line";
            metadata.category = "Geology";
            metadata.headerColor = IM_COL32(180, 100, 80, 255);
            headerColor = ImVec4(0.7f, 0.4f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Fault"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["direction"] = direction;
            j["offset"] = offset;
            j["verticalOffset"] = verticalOffset;
            j["width"] = width;
            j["position"] = position;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("direction")) direction = j["direction"].get<float>();
            if (j.contains("offset")) offset = j["offset"].get<float>();
            if (j.contains("verticalOffset")) verticalOffset = j["verticalOffset"].get<float>();
            if (j.contains("width")) width = j["width"].get<float>();
            if (j.contains("position")) position = j["position"].get<float>();
        }
    };
    
    /**
     * @brief Mesa Node - Flat-topped plateau formation
     * 
     * Creates flat mesas/buttes with steep cliff edges.
     */
    class MesaNode : public TerrainNodeBase {
    public:
        float threshold = 0.5f;       // Height threshold for plateau (0-1)
        float cliffSteepness = 0.9f;  // Cliff edge sharpness (0-1)
        float plateauHeight = 1.0f;   // Plateau height multiplier
        int terraceCount = 1;         // Number of terrace levels
        float noiseAmount = 0.05f;    // Edge noise variation
        
        MesaNode() {
            name = "Mesa";
            terrainNodeType = NodeType::Mesa;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Mesa / Plateau";
            metadata.category = "Geology";
            metadata.headerColor = IM_COL32(160, 120, 80, 255);
            headerColor = ImVec4(0.6f, 0.5f, 0.3f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Mesa"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["threshold"] = threshold;
            j["cliffSteepness"] = cliffSteepness;
            j["plateauHeight"] = plateauHeight;
            j["terraceCount"] = terraceCount;
            j["noiseAmount"] = noiseAmount;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("threshold")) threshold = j["threshold"].get<float>();
            if (j.contains("cliffSteepness")) cliffSteepness = j["cliffSteepness"].get<float>();
            if (j.contains("plateauHeight")) plateauHeight = j["plateauHeight"].get<float>();
            if (j.contains("terraceCount")) terraceCount = j["terraceCount"].get<int>();
            if (j.contains("noiseAmount")) noiseAmount = j["noiseAmount"].get<float>();
        }
    };
    
    /**
     * @brief Shear Node - Diagonal deformation / tectonic stress patterns
     * 
     * Applies shear deformation creating diagonal displacement bands.
     */
    class ShearNode : public TerrainNodeBase {
    public:
        float angle = 30.0f;          // Shear angle (degrees)
        float strength = 0.3f;        // Deformation strength
        int bands = 4;                // Number of shear bands
        float bandWidth = 0.2f;       // Width of each band (0-1)
        bool bidirectional = true;    // Alternate direction per band
        
        ShearNode() {
            name = "Shear";
            terrainNodeType = NodeType::Shear;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Shear Zone";
            metadata.category = "Geology";
            metadata.headerColor = IM_COL32(140, 100, 120, 255);
            headerColor = ImVec4(0.55f, 0.4f, 0.5f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Shear"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["angle"] = angle;
            j["strength"] = strength;
            j["bands"] = bands;
            j["bandWidth"] = bandWidth;
            j["bidirectional"] = bidirectional;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("angle")) angle = j["angle"].get<float>();
            if (j.contains("strength")) strength = j["strength"].get<float>();
            if (j.contains("bands")) bands = j["bands"].get<int>();
            if (j.contains("bandWidth")) bandWidth = j["bandWidth"].get<float>();
            if (j.contains("bidirectional")) bidirectional = j["bidirectional"].get<bool>();
        }
    };

    // ============================================================================
    // IMAGE CONTRACT / AUTHORING UTILITY NODES
    // ============================================================================

    enum class ResampleFilter { Nearest = 0, Bilinear = 1 };
    enum class ResampleSemantic { Height = 0, Mask = 1 };

    class ResampleNode : public TerrainNodeBase {
    public:
        int targetWidth = 512;
        int targetHeight = 512;
        bool matchReference = true;
        ResampleFilter filter = ResampleFilter::Bilinear;
        ResampleSemantic semanticMode = ResampleSemantic::Height;

        ResampleNode() {
            name = "Resample";
            terrainNodeType = NodeType::Resample;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Source", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Reference", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Result", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            metadata.displayName = "Resample";
            metadata.category = "Utility";
            metadata.headerColor = IM_COL32(80, 145, 180, 255);
            headerColor = ImVec4(0.3f, 0.55f, 0.7f, 1.0f);
        }

        void syncSemantic() {
            const auto semantic = semanticMode == ResampleSemantic::Height
                ? NodeSystem::ImageSemantic::Height : NodeSystem::ImageSemantic::Mask;
            inputs[0].imageSemantic = semantic;
            inputs[0].updateVisualCache();
            outputs[0].imageSemantic = semantic;
            outputs[0].updateVisualCache();
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Resample"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["targetWidth"] = targetWidth;
            j["targetHeight"] = targetHeight;
            j["matchReference"] = matchReference;
            j["filter"] = static_cast<int>(filter);
            j["semanticMode"] = static_cast<int>(semanticMode);
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            targetWidth = clampValue(j.value("targetWidth", targetWidth), 2, 8192);
            targetHeight = clampValue(j.value("targetHeight", targetHeight), 2, 8192);
            matchReference = j.value("matchReference", matchReference);
            filter = static_cast<ResampleFilter>(clampValue(j.value("filter", 1), 0, 1));
            semanticMode = static_cast<ResampleSemantic>(clampValue(j.value("semanticMode", 0), 0, 1));
            syncSemantic();
        }
    };

    class ChannelExtractNode : public TerrainNodeBase {
    public:
        int channel = 0;
        bool invert = false;
        ChannelExtractNode() {
            name = "Channel Extract";
            terrainNodeType = NodeType::ChannelExtract;
            inputs.push_back(NodeSystem::Pin::createInput(
                "RGBA", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Generic, false, 0));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Channel Extract";
            metadata.category = "Utility";
            metadata.headerColor = IM_COL32(150, 95, 175, 255);
            headerColor = ImVec4(0.58f, 0.38f, 0.68f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.ChannelExtract"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j); j["channel"] = channel; j["invert"] = invert;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            channel = clampValue(j.value("channel", channel), 0, 3);
            invert = j.value("invert", invert);
        }
    };

    class SplatComposeNode : public TerrainNodeBase {
    public:
        bool normalize = true;
        SplatComposeNode() {
            name = "Splat Compose";
            terrainNodeType = NodeType::SplatCompose;
            for (const char* channelName : {"R", "G", "B", "A"}) {
                inputs.push_back(NodeSystem::Pin::createInput(
                    channelName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            }
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, 4));
            metadata.displayName = "Splat Compose";
            metadata.category = "Texture";
            metadata.headerColor = IM_COL32(195, 135, 55, 255);
            headerColor = ImVec4(0.76f, 0.52f, 0.22f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.SplatCompose"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j); j["normalize"] = normalize;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j); normalize = j.value("normalize", normalize);
        }
    };

    class RemapNode : public TerrainNodeBase {
    public:
        float inputMin = 0.0f, inputMax = 1.0f;
        float outputMin = 0.0f, outputMax = 1.0f;
        float gamma = 1.0f;
        bool clampOutput = true;
        bool maskMode = false;
        RemapNode() {
            name = "Remap";
            terrainNodeType = NodeType::Remap;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            metadata.displayName = "Remap";
            metadata.category = "Filter";
            metadata.headerColor = IM_COL32(90, 150, 195, 255);
            headerColor = ImVec4(0.35f, 0.58f, 0.76f, 1.0f);
        }
        void syncSemantic() {
            const auto semantic = maskMode ? NodeSystem::ImageSemantic::Mask : NodeSystem::ImageSemantic::Height;
            inputs[0].imageSemantic = semantic; inputs[0].updateVisualCache();
            outputs[0].imageSemantic = semantic; outputs[0].updateVisualCache();
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Remap"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["inputMin"] = inputMin; j["inputMax"] = inputMax;
            j["outputMin"] = outputMin; j["outputMax"] = outputMax;
            j["gamma"] = gamma; j["clampOutput"] = clampOutput; j["maskMode"] = maskMode;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            inputMin = j.value("inputMin", inputMin); inputMax = j.value("inputMax", inputMax);
            outputMin = j.value("outputMin", outputMin); outputMax = j.value("outputMax", outputMax);
            gamma = clampValue(j.value("gamma", gamma), 0.01f, 8.0f);
            clampOutput = j.value("clampOutput", clampOutput);
            maskMode = j.value("maskMode", maskMode);
            syncSemantic();
        }
    };

    class MaskAdjustNode : public TerrainNodeBase {
    public:
        float intensity = 1.0f;
        float brightness = 0.0f;
        float contrast = 1.0f;
        float gamma = 1.0f;
        float mix = 1.0f;
        bool invert = false;
        bool clampOutput = true;

        MaskAdjustNode() {
            name = "Mask Adjust";
            terrainNodeType = NodeType::MaskAdjust;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Effect Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Mask Adjust";
            metadata.category = "Filter";
            metadata.headerColor = IM_COL32(130, 105, 190, 255);
            headerColor = ImVec4(0.51f, 0.41f, 0.75f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.MaskAdjust"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["intensity"] = intensity; j["brightness"] = brightness;
            j["contrast"] = contrast; j["gamma"] = gamma; j["mix"] = mix;
            j["invert"] = invert; j["clampOutput"] = clampOutput;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            intensity = clampValue(j.value("intensity", intensity), 0.0f, 8.0f);
            brightness = clampValue(j.value("brightness", brightness), -2.0f, 2.0f);
            contrast = clampValue(j.value("contrast", contrast), 0.0f, 8.0f);
            gamma = clampValue(j.value("gamma", gamma), 0.05f, 8.0f);
            mix = clampValue(j.value("mix", mix), 0.0f, 1.0f);
            invert = j.value("invert", invert);
            clampOutput = j.value("clampOutput", clampOutput);
        }
    };

    enum class MaskMorphologyOp { Dilate = 0, Erode = 1, Blur = 2 };
    class MaskMorphologyNode : public TerrainNodeBase {
    public:
        MaskMorphologyOp operation = MaskMorphologyOp::Blur;
        int radius = 2;
        int iterations = 1;
        MaskMorphologyNode() {
            name = "Mask Morphology";
            terrainNodeType = NodeType::MaskMorphology;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Mask Morphology";
            metadata.category = "Mask";
            metadata.headerColor = IM_COL32(165, 95, 185, 255);
            headerColor = ImVec4(0.64f, 0.37f, 0.72f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.MaskMorphology"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["operation"] = static_cast<int>(operation); j["radius"] = radius; j["iterations"] = iterations;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            operation = static_cast<MaskMorphologyOp>(clampValue(j.value("operation", 2), 0, 2));
            radius = clampValue(j.value("radius", radius), 1, 12);
            iterations = clampValue(j.value("iterations", iterations), 1, 8);
        }
    };

    // Reusable topographic fields for biome, material and foliage branches.
    // Every output is solved together and inserted into EvaluationContext cache.
    class TerrainAnalysisNode : public TerrainNodeBase {
    public:
        float valleyScale = 0.08f;
        float curvatureScale = 1.0f;
        int neighborhoodRadius = 4;

        TerrainAnalysisNode() {
            name = "Terrain Analysis";
            terrainNodeType = NodeType::TerrainAnalysis;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            for (const char* outputName : {"Slope", "Concavity", "Convexity", "Valley", "Wetness"}) {
                outputs.push_back(NodeSystem::Pin::createOutput(
                    outputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            }
            metadata.displayName = "Terrain Analysis";
            metadata.category = "Data Maps";
            metadata.description = "Cached slope, curvature, valley and wetness fields";
            metadata.headerColor = IM_COL32(62, 145, 180, 255);
            headerColor = ImVec4(0.24f, 0.57f, 0.71f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.TerrainAnalysis"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["valleyScale"] = valleyScale;
            j["curvatureScale"] = curvatureScale;
            j["neighborhoodRadius"] = neighborhoodRadius;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            valleyScale = clampValue(j.value("valleyScale", valleyScale), 0.005f, 0.5f);
            curvatureScale = clampValue(j.value("curvatureScale", curvatureScale), 0.05f, 10.0f);
            neighborhoodRadius = clampValue(j.value("neighborhoodRadius", neighborhoodRadius), 1, 64);
        }
    };

    // Priority-flood watershed solve shared by fluvial carving, river extraction,
    // biome wetness and future lake/floodplain nodes. Flow Direction is encoded
    // as 0 for an outlet or (D8 direction + 1) / 9 for downstream cells.
    class WatershedAnalysisNode : public TerrainNodeBase {
    public:
        float rainfall = 1.0f;
        float flatEpsilon = 0.00001f;

        WatershedAnalysisNode() {
            name = "Watershed Analysis";
            terrainNodeType = NodeType::WatershedAnalysis;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Filled Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            for (const char* outputName : {"Accumulation", "Flow Direction", "Drainage Basins"}) {
                outputs.push_back(NodeSystem::Pin::createOutput(
                    outputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            }
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Catchment Area", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            metadata.displayName = "Watershed Analysis";
            metadata.category = "Hydrology";
            metadata.description = "Depression-safe D8 drainage, accumulation and catchments";
            metadata.headerColor = IM_COL32(48, 132, 190, 255);
            headerColor = ImVec4(0.19f, 0.52f, 0.75f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.WatershedAnalysis"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["rainfall"] = rainfall;
            j["flatEpsilon"] = flatEpsilon;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            rainfall = clampValue(j.value("rainfall", rainfall), 0.001f, 100.0f);
            flatEpsilon = clampValue(j.value("flatEpsilon", flatEpsilon), 0.0000001f, 0.01f);
        }
    };

    // Converts the depression fill delta into explicit lake bodies. Raster
    // outputs feed materials/foliage today; pendingWaterBodies is published on
    // the main thread as the stable contract for lake mesh generation next.
    class LakeBasinNode : public TerrainNodeBase {
    public:
        float minimumDepthMeters = 0.10f;
        float minimumAreaSquareMeters = 4.0f;
        int maximumLakes = 64;
        bool includeClosedBasins = true;
        std::vector<WaterBodyData> pendingWaterBodies;

        LakeBasinNode() {
            name = "Lake Basin";
            terrainNodeType = NodeType::LakeBasin;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Original Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Filled Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Direction", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Lake Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Lake Depth", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Water Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            for (const char* outputName : {"Shoreline", "Spill Points", "Lake IDs"}) {
                outputs.push_back(NodeSystem::Pin::createOutput(
                    outputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            }
            metadata.displayName = "Lake Basin";
            metadata.category = "Hydrology";
            metadata.description = "Extracts lake levels, shorelines, storage and spill outlets";
            metadata.headerColor = IM_COL32(35, 145, 190, 255);
            headerColor = ImVec4(0.14f, 0.57f, 0.75f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void publishWaterBodies(TerrainObject* terrain) const;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.LakeBasin"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["minimumDepthMeters"] = minimumDepthMeters;
            j["minimumAreaSquareMeters"] = minimumAreaSquareMeters;
            j["maximumLakes"] = maximumLakes;
            j["includeClosedBasins"] = includeClosedBasins;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            minimumDepthMeters = clampValue(j.value("minimumDepthMeters", minimumDepthMeters), 0.001f, 1000.0f);
            minimumAreaSquareMeters = clampValue(j.value("minimumAreaSquareMeters", minimumAreaSquareMeters), 0.001f, 1000000000.0f);
            maximumLakes = clampValue(j.value("maximumLakes", maximumLakes), 1, 4096);
            includeClosedBasins = j.value("includeClosedBasins", includeClosedBasins);
        }
    };

    // Main-thread sink that converts analytical lake fields into owned water
    // meshes. Marching-squares cell polygons preserve shoreline holes and avoid
    // the block expansion produced by one-quad-per-wet-sample generation.
    class LakeSurfaceOutputNode : public TerrainNodeBase {
    public:
        float surfaceOffsetMeters = 0.02f;
        float uvScaleMeters = 4.0f;
        int maximumGeneratedLakes = 32;
        bool generateWaterMeshes = true;
        int sourceLakeNodeId = -1;
        std::vector<int> generatedWaterSurfaceIds;
        std::array<NodeSystem::Image2DData, 4> pendingFields;

        LakeSurfaceOutputNode() {
            name = "Lake Surface Output";
            terrainNodeType = NodeType::LakeSurfaceOutput;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Depth", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake IDs", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Lake Surface Output";
            metadata.category = "Output";
            metadata.description = "Builds owned WaterSurface meshes from analytical lakes";
            metadata.headerColor = IM_COL32(25, 155, 198, 255);
            headerColor = ImVec4(0.10f, 0.61f, 0.78f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        bool applyGeneratedLakes(struct ::SceneData& scene, TerrainObject* terrain);
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.LakeSurfaceOutput"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["surfaceOffsetMeters"] = surfaceOffsetMeters;
            j["uvScaleMeters"] = uvScaleMeters;
            j["maximumGeneratedLakes"] = maximumGeneratedLakes;
            j["generateWaterMeshes"] = generateWaterMeshes;
            j["generatedWaterSurfaceIds"] = generatedWaterSurfaceIds;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            surfaceOffsetMeters = clampValue(j.value("surfaceOffsetMeters", surfaceOffsetMeters), -10.0f, 10.0f);
            uvScaleMeters = clampValue(j.value("uvScaleMeters", uvScaleMeters), 0.01f, 100000.0f);
            maximumGeneratedLakes = clampValue(j.value("maximumGeneratedLakes", maximumGeneratedLakes), 1, 4096);
            generateWaterMeshes = j.value("generateWaterMeshes", generateWaterMeshes);
            generatedWaterSurfaceIds = j.value("generatedWaterSurfaceIds", std::vector<int>{});
        }
    };

    // Converts the watershed's continuous accumulation/direction fields into a
    // pruned, topologically ordered stream network suitable for spline output.
    class RiverNetworkNode : public TerrainNodeBase {
    public:
        float catchmentThreshold = 0.0015f;
        int minimumBranchLength = 8;

        RiverNetworkNode() {
            name = "River Network";
            terrainNodeType = NodeType::RiverNetwork;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Accumulation", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Direction", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            for (const char* outputName : {"Channels", "Stream Order", "Sources"}) {
                outputs.push_back(NodeSystem::Pin::createOutput(
                    outputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            }
            metadata.displayName = "River Network";
            metadata.category = "Hydrology";
            metadata.description = "Extracts and prunes a connected stream hierarchy";
            metadata.headerColor = IM_COL32(42, 151, 203, 255);
            headerColor = ImVec4(0.16f, 0.59f, 0.80f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.RiverNetwork"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["catchmentThreshold"] = catchmentThreshold;
            j["minimumBranchLength"] = minimumBranchLength;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            catchmentThreshold = clampValue(j.value("catchmentThreshold", catchmentThreshold), 0.00001f, 0.95f);
            minimumBranchLength = clampValue(j.value("minimumBranchLength", minimumBranchLength), 2, 256);
        }
    };

    // Quasi-steady 1D hydraulic solve over the extracted D8 channel graph.
    // Discharge is catchment/rainfall driven; trapezoidal normal depth follows
    // Manning and the reverse graph pass enforces a downstream water profile.
    class RiverHydraulicsNode : public TerrainNodeBase {
    public:
        float rainfallMillimetersPerHour = 25.0f;
        float runoffCoefficient = 0.35f;
        float dischargeScale = 1.0f;
        float manningRoughness = 0.035f;
        float widthCoefficient = 4.5f;
        float widthExponent = 0.50f;
        float minimumWidthMeters = 0.35f;
        float maximumWidthMeters = 80.0f;
        float minimumDepthMeters = 0.05f;
        float maximumDepthMeters = 12.0f;
        float bankSideSlope = 1.5f;
        float minimumBedSlope = 0.0001f;
        float minimumSurfaceSlope = 0.00002f;
        float surfaceOffsetMeters = 0.03f;
        float bankFreeboardRatio = 0.35f;
        float minimumFreeboardMeters = 0.08f;
        float maximumFreeboardMeters = 2.0f;

        RiverHydraulicsNode() {
            name = "River Hydraulics";
            terrainNodeType = NodeType::RiverHydraulics;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Bed Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Catchment Area", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Direction", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Channels", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Reference Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            for (const char* outputName : {"Discharge", "River Width", "Water Depth", "Flow Speed",
                                           "Water Level", "Froude", "Foam Potential"}) {
                outputs.push_back(NodeSystem::Pin::createOutput(
                    outputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            }
            metadata.displayName = "River Hydraulics";
            metadata.category = "Hydrology";
            metadata.description = "Manning discharge, normal depth, velocity and whitewater state";
            metadata.headerColor = IM_COL32(28, 126, 176, 255);
            headerColor = ImVec4(0.11f, 0.49f, 0.69f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.RiverHydraulics"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["rainfallMillimetersPerHour"] = rainfallMillimetersPerHour;
            j["runoffCoefficient"] = runoffCoefficient;
            j["dischargeScale"] = dischargeScale;
            j["manningRoughness"] = manningRoughness;
            j["widthCoefficient"] = widthCoefficient;
            j["widthExponent"] = widthExponent;
            j["minimumWidthMeters"] = minimumWidthMeters;
            j["maximumWidthMeters"] = maximumWidthMeters;
            j["minimumDepthMeters"] = minimumDepthMeters;
            j["maximumDepthMeters"] = maximumDepthMeters;
            j["bankSideSlope"] = bankSideSlope;
            j["minimumBedSlope"] = minimumBedSlope;
            j["minimumSurfaceSlope"] = minimumSurfaceSlope;
            j["surfaceOffsetMeters"] = surfaceOffsetMeters;
            j["bankFreeboardRatio"] = bankFreeboardRatio;
            j["minimumFreeboardMeters"] = minimumFreeboardMeters;
            j["maximumFreeboardMeters"] = maximumFreeboardMeters;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            rainfallMillimetersPerHour = clampValue(j.value("rainfallMillimetersPerHour", rainfallMillimetersPerHour), 0.0f, 2000.0f);
            runoffCoefficient = clampValue(j.value("runoffCoefficient", runoffCoefficient), 0.0f, 1.0f);
            dischargeScale = clampValue(j.value("dischargeScale", dischargeScale), 0.001f, 10000.0f);
            manningRoughness = clampValue(j.value("manningRoughness", manningRoughness), 0.005f, 0.3f);
            widthCoefficient = clampValue(j.value("widthCoefficient", widthCoefficient), 0.01f, 100.0f);
            widthExponent = clampValue(j.value("widthExponent", widthExponent), 0.1f, 1.0f);
            minimumWidthMeters = clampValue(j.value("minimumWidthMeters", minimumWidthMeters), 0.05f, 100.0f);
            maximumWidthMeters = clampValue(j.value("maximumWidthMeters", maximumWidthMeters), minimumWidthMeters, 2000.0f);
            minimumDepthMeters = clampValue(j.value("minimumDepthMeters", minimumDepthMeters), 0.005f, 20.0f);
            maximumDepthMeters = clampValue(j.value("maximumDepthMeters", maximumDepthMeters), minimumDepthMeters, 500.0f);
            bankSideSlope = clampValue(j.value("bankSideSlope", bankSideSlope), 0.0f, 10.0f);
            minimumBedSlope = clampValue(j.value("minimumBedSlope", minimumBedSlope), 0.000001f, 1.0f);
            minimumSurfaceSlope = clampValue(j.value("minimumSurfaceSlope", minimumSurfaceSlope), 0.0f, 0.1f);
            surfaceOffsetMeters = clampValue(j.value("surfaceOffsetMeters", surfaceOffsetMeters), 0.0f, 10.0f);
            bankFreeboardRatio = clampValue(j.value("bankFreeboardRatio", bankFreeboardRatio), 0.0f, 5.0f);
            minimumFreeboardMeters = clampValue(j.value("minimumFreeboardMeters", minimumFreeboardMeters), 0.0f, 20.0f);
            maximumFreeboardMeters = clampValue(
                j.value("maximumFreeboardMeters", maximumFreeboardMeters), minimumFreeboardMeters, 100.0f);
        }
    };

    // Non-destructive height operator driven by the extracted channel field.
    // Width and depth grow with contributing area, while overlapping channel
    // stamps use a max-depth field instead of repeatedly subtracting height.
    class RiverBedCarveNode : public TerrainNodeBase {
    public:
        float minimumWidth = 0.5f;
        float maximumWidth = 4.0f;
        float minimumDepth = 0.08f;
        float maximumDepth = 0.65f;
        float bankSoftness = 0.65f;

        RiverBedCarveNode() {
            name = "River Bed Carve";
            terrainNodeType = NodeType::RiverBedCarve;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Channels", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "River Width", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Water Depth", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Reference Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Water Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Carved Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "River Bed", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "River Bed Carve";
            metadata.category = "Hydrology";
            metadata.description = "Area-scaled, non-destructive channel and bank carving";
            metadata.headerColor = IM_COL32(38, 139, 184, 255);
            headerColor = ImVec4(0.15f, 0.55f, 0.72f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.RiverBedCarve"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["minimumWidth"] = minimumWidth; j["maximumWidth"] = maximumWidth;
            j["minimumDepth"] = minimumDepth; j["maximumDepth"] = maximumDepth;
            j["bankSoftness"] = bankSoftness;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            minimumWidth = clampValue(j.value("minimumWidth", minimumWidth), 0.1f, 100.0f);
            maximumWidth = clampValue(j.value("maximumWidth", maximumWidth), minimumWidth, 500.0f);
            minimumDepth = clampValue(j.value("minimumDepth", minimumDepth), 0.0f, 50.0f);
            maximumDepth = clampValue(j.value("maximumDepth", maximumDepth), minimumDepth, 200.0f);
            bankSoftness = clampValue(j.value("bankSoftness", bankSoftness), 0.05f, 1.0f);
        }
    };

    // Main-thread sink: vectorizes connected channel segments and owns only the
    // RiverManager entries it generated. Manual rivers remain untouched.
    class RiverSplineOutputNode : public TerrainNodeBase {
    public:
        float minimumWidth = 0.35f;
        float maximumWidth = 3.5f;
        float depthScale = 0.22f;
        int minimumSplinePoints = 2;
        int pointSpacing = 3;
        int maximumRivers = 24;
        bool generateWaterMeshes = true;
        std::vector<int> generatedRiverIds;

        struct PendingPoint {
            float x = 0.0f;
            float y = 0.0f;
            float strength = 0.0f;
            float surfaceHeight = 0.0f;
            float widthMeters = 0.0f;
            float depthMeters = 0.0f;
            float flowSpeed = 0.0f;
            float discharge = 0.0f;
            float froude = 0.0f;
            float foamPotential = 0.0f;
            int sourceIndex = -1;
        };
        struct PendingPath {
            std::vector<PendingPoint> points;
            float importance = 0.0f;
            float lengthCells = 0.0f;
        };
        std::vector<PendingPath> pendingPaths;

        RiverSplineOutputNode() {
            name = "River Spline Output";
            terrainNodeType = NodeType::RiverSplineOutput;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Accumulation", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Direction", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Channels", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            for (const char* inputName : {"River Width", "Water Depth", "Flow Speed", "Discharge",
                                          "Froude", "Foam Potential", "River Water Level"}) {
                inputs.push_back(NodeSystem::Pin::createInput(
                    inputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            }
            metadata.displayName = "River Spline Output";
            metadata.category = "Output";
            metadata.description = "Creates owned RiverSpline branches from a river network";
            metadata.headerColor = IM_COL32(35, 167, 214, 255);
            headerColor = ImVec4(0.14f, 0.65f, 0.84f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        bool applyGeneratedRivers(struct ::SceneData& scene, TerrainObject* terrain);
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.RiverSplineOutput"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["minimumWidth"] = minimumWidth;
            j["maximumWidth"] = maximumWidth;
            j["depthScale"] = depthScale;
            j["minimumSplinePoints"] = minimumSplinePoints;
            j["pointSpacing"] = pointSpacing;
            j["maximumRivers"] = maximumRivers;
            j["generateWaterMeshes"] = generateWaterMeshes;
            j["generatedRiverIds"] = generatedRiverIds;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            minimumWidth = clampValue(j.value("minimumWidth", minimumWidth), 0.1f, 100.0f);
            maximumWidth = clampValue(j.value("maximumWidth", maximumWidth), minimumWidth, 500.0f);
            depthScale = clampValue(j.value("depthScale", depthScale), 0.01f, 20.0f);
            minimumSplinePoints = clampValue(j.value("minimumSplinePoints", minimumSplinePoints), 2, 512);
            pointSpacing = clampValue(j.value("pointSpacing", pointSpacing), 1, 64);
            maximumRivers = clampValue(j.value("maximumRivers", maximumRivers), 1, 256);
            generateWaterMeshes = j.value("generateWaterMeshes", generateWaterMeshes);
            generatedRiverIds = j.value("generatedRiverIds", std::vector<int>{});
        }
    };

    // Explicit sink for persistent named fields. Mesh publication is deferred to
    // TerrainManager's main-thread mesh finalize path.
    class TerrainFieldsOutputNode : public TerrainNodeBase {
    public:
        TerrainFieldsOutputNode() {
            name = "Terrain Fields Output";
            terrainNodeType = NodeType::TerrainFieldsOutput;
            for (const char* inputName : {"Slope", "Concavity", "Convexity", "Valley", "Wetness",
                                          "Forest", "Grass", "Rock", "Alpine",
                                          "Flow Accumulation", "Flow Direction", "Drainage Basins",
                                          "River Channels", "Stream Order", "River Sources", "River Bed",
                                          "Lake Mask", "Lake Depth", "Lake Level", "Lake Shoreline",
                                          "Lake Spill Points", "Lake IDs", "Catchment Area",
                                          "River Discharge", "River Width", "River Water Depth", "River Flow Speed",
                                          "River Water Level", "River Froude", "River Foam Potential"}) {
                inputs.push_back(NodeSystem::Pin::createInput(
                    inputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            }
            metadata.displayName = "Terrain Fields Output";
            metadata.category = "Output";
            metadata.description = "Publishes terrain, biome and hydrology fields for downstream systems";
            metadata.headerColor = IM_COL32(55, 170, 135, 255);
            headerColor = ImVec4(0.22f, 0.67f, 0.53f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        std::string getTypeId() const override { return "TerrainV2.TerrainFieldsOutput"; }
    };

    enum class BiomeClimatePreset {
        Custom = 0,
        TemperateMixed,
        LushValleys,
        AlpineTundra,
        AridHighlands,
        BorealMountains
    };

    // Produces a mutually normalized four-biome partition. Every output is
    // solved together and cached, so material and foliage branches reuse the
    // exact same classification without recomputing terrain analysis.
    class BiomeComposerNode : public TerrainNodeBase {
    public:
        BiomeClimatePreset preset = BiomeClimatePreset::TemperateMixed;
        float forestCeiling = 0.72f;
        float alpineLine = 0.68f;
        float forestMoisture = 0.32f;
        float rockSlope = 0.48f;
        float transition = 0.10f;
        float exposureDrying = 0.30f;

        BiomeComposerNode() {
            name = "Biome Composer";
            terrainNodeType = NodeType::BiomeComposer;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            for (const char* inputName : {"Slope", "Valley", "Wetness", "Exposure"}) {
                inputs.push_back(NodeSystem::Pin::createInput(
                    inputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            }
            for (const char* outputName : {"Forest", "Grass", "Rock", "Alpine"}) {
                outputs.push_back(NodeSystem::Pin::createOutput(
                    outputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            }
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Biome Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, 4));
            metadata.displayName = "Biome Composer";
            metadata.category = "Data Maps";
            metadata.headerColor = IM_COL32(74, 154, 92, 255);
            headerColor = ImVec4(0.29f, 0.60f, 0.36f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        static const char* getPresetName(BiomeClimatePreset value);
        void applyPreset(BiomeClimatePreset value);
        std::string getTypeId() const override { return "TerrainV2.BiomeComposer"; }
        float getCustomWidth() const override { return 180.0f; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["biomePreset"] = static_cast<int>(preset);
            j["forestCeiling"] = forestCeiling;
            j["alpineLine"] = alpineLine;
            j["forestMoisture"] = forestMoisture;
            j["rockSlope"] = rockSlope;
            j["transition"] = transition;
            j["exposureDrying"] = exposureDrying;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            // Graphs saved before biome presets already contain explicit values;
            // keep those exact values and classify them as Custom.
            preset = j.contains("biomePreset")
                ? static_cast<BiomeClimatePreset>(clampValue(j.value("biomePreset", 0), 0, 5))
                : BiomeClimatePreset::Custom;
            forestCeiling = clampValue(j.value("forestCeiling", forestCeiling), 0.0f, 1.0f);
            alpineLine = clampValue(j.value("alpineLine", alpineLine), 0.0f, 1.0f);
            forestMoisture = clampValue(j.value("forestMoisture", forestMoisture), 0.0f, 1.0f);
            rockSlope = clampValue(j.value("rockSlope", rockSlope), 0.0f, 1.0f);
            transition = clampValue(j.value("transition", transition), 0.01f, 0.35f);
            exposureDrying = clampValue(j.value("exposureDrying", exposureDrying), 0.0f, 1.0f);
        }
    };

    // Describes one existing InstanceGroup without owning its sources or generated
    // transforms. The Custom output transports a compact JSON recipe through the
    // generic node system; FoliageOutput applies it on the main thread.
    class FoliageLayerNode : public TerrainNodeBase {
    public:
        struct AssetRef {
            std::string id;
            std::string name;
            std::string relativeEntryPath;
            float weight = 1.0f;
            float targetHeight = 0.0f;
            float heightVariation = 0.15f;
            bool alignToNormal = false;
            float normalInfluence = 0.0f;
        };
        std::string instanceGroupName;
        int instanceGroupId = -1;
        bool useAssetLibrary = false;
        std::string assetBiome = "Auto";
        std::string assetSearch;
        std::vector<AssetRef> assetSources;
        // Transient property-panel hook. The node never owns UI textures; the
        // SceneUI thumbnail cache supplies them only while properties are drawn.
        std::function<ImTextureID(const std::string&, int&, int&)> propertyThumbnailProvider;
        bool settingsCaptured = false;
        bool layerEnabled = true;
        float densityMultiplier = 1.0f;
        int targetCount = 1000;
        int seed = 1234;
        float minimumDistance = 0.5f;
        float maximumSlopeDegrees = 45.0f;
        float minimumHeight = -10.0f;
        float maximumHeight = 10.0f;
        std::string densityField;
        std::string exclusionField;
        float exclusionThreshold = 0.5f;
        std::string scaleField;
        float scaleFieldInfluence = 1.0f;

        FoliageLayerNode() {
            name = "Foliage Layer";
            terrainNodeType = NodeType::FoliageLayer;
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Layer", NodeSystem::DataType::Custom));
            metadata.displayName = "Foliage Layer";
            metadata.category = "Foliage";
            metadata.description = "Binds one distribution rule to an existing foliage layer";
            metadata.headerColor = IM_COL32(74, 145, 78, 255);
            headerColor = ImVec4(0.29f, 0.57f, 0.31f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.FoliageLayer"; }
        float getCustomWidth() const override { return 300.0f; }
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
    };

    // Lightweight organizational parent. Each input remains a separate
    // distribution rule; set-level controls are non-destructive multipliers.
    class FoliageSetNode : public TerrainNodeBase {
    public:
        std::string setName = "Biome Foliage";
        bool setEnabled = true;
        float densityMultiplier = 1.0f;
        int seedOffset = 0;

        FoliageSetNode() {
            name = "Foliage Set / Biome";
            terrainNodeType = NodeType::FoliageSet;
            for (int i = 0; i < 8; ++i) {
                inputs.push_back(NodeSystem::Pin::createInput(
                    std::string("Layer ") + std::to_string(i + 1),
                    NodeSystem::DataType::Custom, NodeSystem::ImageSemantic::Generic, true));
            }
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Foliage Set", NodeSystem::DataType::Custom));
            metadata.displayName = "Foliage Set / Biome";
            metadata.category = "Foliage";
            metadata.description = "Groups independent foliage rules for batch control";
            metadata.headerColor = IM_COL32(52, 126, 64, 255);
            headerColor = ImVec4(0.20f, 0.49f, 0.25f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.FoliageSet"; }
        float getCustomWidth() const override { return 190.0f; }
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
    };

    // Main-thread sink. It updates authoring settings and can lazily materialize
    // Asset Library sources without inserting hidden objects into the scene.
    class FoliageOutputNode : public TerrainNodeBase {
    public:
        bool scatterOnApply = true;
        int lastAppliedLayerCount = 0;
        int lastMissingLayerCount = 0;
        int lastMissingAssetCount = 0;
        int lastSpawnedInstanceCount = 0;
        std::vector<int> lastScatteredGroupIds;

        FoliageOutputNode() {
            name = "Foliage Output";
            terrainNodeType = NodeType::FoliageOutput;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Foliage Set", NodeSystem::DataType::Custom));
            metadata.displayName = "Foliage Output";
            metadata.category = "Output";
            metadata.description = "Applies node recipes to existing foliage instance groups";
            metadata.headerColor = IM_COL32(42, 158, 91, 255);
            headerColor = ImVec4(0.16f, 0.62f, 0.36f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.FoliageOutput"; }
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
    };

    // ============================================================================
    // GEOLOGICAL DATA MAPS AND SURFACE SYNTHESIS
    // ============================================================================

    class WetnessMapNode : public TerrainNodeBase {
    public:
        float flowInfluence = 0.55f;
        float concavityInfluence = 0.25f;
        float flatnessInfluence = 0.20f;
        float evaporation = 0.15f;

        WetnessMapNode() {
            name = "Wetness Map";
            terrainNodeType = NodeType::WetnessMap;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Soil", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Wetness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Wetness Map";
            metadata.category = "Data Map";
            metadata.headerColor = IM_COL32(60, 135, 180, 255);
            headerColor = ImVec4(0.24f, 0.53f, 0.71f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.WetnessMap"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["flowInfluence"] = flowInfluence;
            j["concavityInfluence"] = concavityInfluence;
            j["flatnessInfluence"] = flatnessInfluence;
            j["evaporation"] = evaporation;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            flowInfluence = clampValue(j.value("flowInfluence", flowInfluence), 0.0f, 2.0f);
            concavityInfluence = clampValue(j.value("concavityInfluence", concavityInfluence), 0.0f, 2.0f);
            flatnessInfluence = clampValue(j.value("flatnessInfluence", flatnessInfluence), 0.0f, 2.0f);
            evaporation = clampValue(j.value("evaporation", evaporation), 0.0f, 1.0f);
        }
    };

    class SoilDepthNode : public TerrainNodeBase {
    public:
        float production = 0.45f;
        float depositionInfluence = 0.35f;
        float concavityInfluence = 0.30f;
        float slopeLoss = 0.55f;

        SoilDepthNode() {
            name = "Soil Depth";
            terrainNodeType = NodeType::SoilDepth;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Soil Depth", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Soil Depth";
            metadata.category = "Data Map";
            metadata.headerColor = IM_COL32(135, 105, 65, 255);
            headerColor = ImVec4(0.53f, 0.41f, 0.25f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.SoilDepth"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["production"] = production;
            j["depositionInfluence"] = depositionInfluence;
            j["concavityInfluence"] = concavityInfluence;
            j["slopeLoss"] = slopeLoss;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            production = clampValue(j.value("production", production), 0.0f, 2.0f);
            depositionInfluence = clampValue(j.value("depositionInfluence", depositionInfluence), 0.0f, 2.0f);
            concavityInfluence = clampValue(j.value("concavityInfluence", concavityInfluence), 0.0f, 2.0f);
            slopeLoss = clampValue(j.value("slopeLoss", slopeLoss), 0.0f, 2.0f);
        }
    };

    class LithologyNode : public TerrainNodeBase {
    public:
        int layerCount = 8;
        float layerThickness = 4.0f;
        float baseHardness = 0.50f;
        float hardnessContrast = 0.70f;
        float dipDegrees = 8.0f;
        float dipAzimuth = 35.0f;
        float warpStrength = 0.35f;
        int seed = 137;

        LithologyNode() {
            name = "Lithology";
            terrainNodeType = NodeType::Lithology;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Warp", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Geology ID", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Lithology";
            metadata.category = "Geology";
            metadata.headerColor = IM_COL32(150, 100, 72, 255);
            headerColor = ImVec4(0.59f, 0.39f, 0.28f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Lithology"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["layerCount"] = layerCount; j["layerThickness"] = layerThickness;
            j["baseHardness"] = baseHardness; j["hardnessContrast"] = hardnessContrast;
            j["dipDegrees"] = dipDegrees; j["dipAzimuth"] = dipAzimuth;
            j["warpStrength"] = warpStrength; j["seed"] = seed;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            layerCount = clampValue(j.value("layerCount", layerCount), 2, 32);
            layerThickness = clampValue(j.value("layerThickness", layerThickness), 0.05f, 1000.0f);
            baseHardness = clampValue(j.value("baseHardness", baseHardness), 0.0f, 1.0f);
            hardnessContrast = clampValue(j.value("hardnessContrast", hardnessContrast), 0.0f, 1.0f);
            dipDegrees = clampValue(j.value("dipDegrees", dipDegrees), -75.0f, 75.0f);
            dipAzimuth = j.value("dipAzimuth", dipAzimuth);
            warpStrength = clampValue(j.value("warpStrength", warpStrength), 0.0f, 2.0f);
            seed = j.value("seed", seed);
        }
    };

    class StrataNode : public TerrainNodeBase {
    public:
        float layerThickness = 4.0f;
        float dipDegrees = 8.0f;
        float dipAzimuth = 35.0f;
        float reliefStrength = 0.015f;
        float edgeSharpness = 2.0f;

        StrataNode() {
            name = "Strata";
            terrainNodeType = NodeType::Strata;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Strata";
            metadata.category = "Geology";
            metadata.headerColor = IM_COL32(165, 110, 70, 255);
            headerColor = ImVec4(0.65f, 0.43f, 0.27f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Strata"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["layerThickness"] = layerThickness; j["dipDegrees"] = dipDegrees;
            j["dipAzimuth"] = dipAzimuth; j["reliefStrength"] = reliefStrength;
            j["edgeSharpness"] = edgeSharpness;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            layerThickness = clampValue(j.value("layerThickness", layerThickness), 0.05f, 1000.0f);
            dipDegrees = clampValue(j.value("dipDegrees", dipDegrees), -75.0f, 75.0f);
            dipAzimuth = j.value("dipAzimuth", dipAzimuth);
            reliefStrength = clampValue(j.value("reliefStrength", reliefStrength), 0.0f, 0.25f);
            edgeSharpness = clampValue(j.value("edgeSharpness", edgeSharpness), 0.25f, 8.0f);
        }
    };

    class SurfaceComposerNode : public TerrainNodeBase {
    public:
        float textureScale = 12.0f;
        float patchiness = 0.35f;
        float slopeInfluence = 0.65f;
        float soilInfluence = 0.80f;
        float flowInfluence = 0.45f;
        float wetnessInfluence = 0.75f;
        float hardnessInfluence = 0.60f;
        float grassInfluence = 1.0f;
        float rockInfluence = 1.0f;
        float snowInfluence = 1.0f;
        float iceInfluence = 0.85f;
        float contrast = 1.25f;
        int seed = 73;

        SurfaceComposerNode() {
            name = "Surface Composer";
            terrainNodeType = NodeType::SurfaceComposer;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Soil", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Wetness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Snow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Ice", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Appended to preserve serialized pin-index compatibility. Erosion
            // Flow and climate Meltwater are merged internally into splat A.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Meltwater", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Explicit authored layers are appended so old serialized pin indices
            // remain stable. When either pin is unconnected, the legacy
            // height/slope/soil synthesis remains available as a fallback.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Grass / Base", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Rock / Slope", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Surface Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, 4));
            metadata.displayName = "Surface Composer";
            metadata.category = "Texture";
            metadata.headerColor = IM_COL32(190, 135, 52, 255);
            headerColor = ImVec4(0.75f, 0.53f, 0.20f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.SurfaceComposer"; }
        float getCustomWidth() const override { return 175.0f; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["textureScale"] = textureScale; j["patchiness"] = patchiness;
            j["slopeInfluence"] = slopeInfluence; j["soilInfluence"] = soilInfluence;
            j["flowInfluence"] = flowInfluence; j["wetnessInfluence"] = wetnessInfluence;
            j["hardnessInfluence"] = hardnessInfluence; j["snowInfluence"] = snowInfluence;
            j["grassInfluence"] = grassInfluence; j["rockInfluence"] = rockInfluence;
            j["iceInfluence"] = iceInfluence; j["contrast"] = contrast; j["seed"] = seed;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            textureScale = clampValue(j.value("textureScale", textureScale), 1.0f, 256.0f);
            patchiness = clampValue(j.value("patchiness", patchiness), 0.0f, 1.0f);
            slopeInfluence = clampValue(j.value("slopeInfluence", slopeInfluence), 0.0f, 2.0f);
            soilInfluence = clampValue(j.value("soilInfluence", soilInfluence), 0.0f, 2.0f);
            flowInfluence = clampValue(j.value("flowInfluence", flowInfluence), 0.0f, 2.0f);
            wetnessInfluence = clampValue(j.value("wetnessInfluence", wetnessInfluence), 0.0f, 2.0f);
            hardnessInfluence = clampValue(j.value("hardnessInfluence", hardnessInfluence), 0.0f, 2.0f);
            grassInfluence = clampValue(j.value("grassInfluence", grassInfluence), 0.0f, 2.0f);
            rockInfluence = clampValue(j.value("rockInfluence", rockInfluence), 0.0f, 2.0f);
            snowInfluence = clampValue(j.value("snowInfluence", snowInfluence), 0.0f, 2.0f);
            iceInfluence = clampValue(j.value("iceInfluence", iceInfluence), 0.0f, 2.0f);
            contrast = clampValue(j.value("contrast", contrast), 0.1f, 4.0f);
            seed = j.value("seed", seed);
        }
    };

    enum class SnowClimatePreset {
        Custom = 0,
        AlpineBalanced,
        DeepWinter,
        SpringThaw,
        WindblownPeaks,
        GlacierValley
    };

    class SnowClimateNode : public TerrainNodeBase {
    public:
        SnowClimatePreset preset = SnowClimatePreset::AlpineBalanced;
        float snowfallMeters = 0.45f;
        float maxDepthMeters = 1.80f;
        bool affectGeometry = true;
        float geometryAmount = 1.0f;
        float coverageAmount = 1.0f;
        bool relativeSnowLine = true;
        float snowLineFraction = 0.58f;
        float snowLineBlendFraction = 0.12f;
        float snowLine = 6.0f;
        float snowLineTransition = 2.5f;
        float baseTemperature = -3.0f;
        float lapseRate = 6.5f;
        float meltAmount = 0.18f;
        float solarMelt = 0.30f;
        float refreezeRate = 0.35f;
        float valleyCapture = 0.75f;
        float transportRate = 0.32f;
        float slipAngle = 38.0f;
        int settleIterations = 18;
        int waterIterations = 10;
        float windStrength = 0.08f;
        float windAzimuth = 0.0f;
        bool useSceneSun = true;
        float sunAzimuth = 135.0f;
        float sunElevation = 35.0f;

        SnowClimateNode() {
            name = "Snow Layer";
            terrainNodeType = NodeType::SnowClimate;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Base Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Exposure", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Surface Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Snow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Ice", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Meltwater", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Avalanche", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Snow Layer";
            metadata.category = "Snow & Ice";
            metadata.headerColor = IM_COL32(105, 180, 220, 255);
            headerColor = ImVec4(0.41f, 0.71f, 0.86f, 1.0f);
        }

        static const char* getPresetName(SnowClimatePreset value);
        void applyPreset(SnowClimatePreset value);
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.SnowClimate"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["solverVersion"] = 2;
            j["preset"] = static_cast<int>(preset);
            j["snowfallMeters"] = snowfallMeters; j["maxDepthMeters"] = maxDepthMeters;
            j["affectGeometry"] = affectGeometry; j["geometryAmount"] = geometryAmount;
            j["coverageAmount"] = coverageAmount;
            j["relativeSnowLine"] = relativeSnowLine; j["snowLineFraction"] = snowLineFraction;
            j["snowLineBlendFraction"] = snowLineBlendFraction;
            j["snowLine"] = snowLine; j["snowLineTransition"] = snowLineTransition;
            j["baseTemperature"] = baseTemperature; j["lapseRate"] = lapseRate;
            j["meltAmount"] = meltAmount; j["solarMelt"] = solarMelt; j["refreezeRate"] = refreezeRate;
            j["valleyCapture"] = valleyCapture; j["transportRate"] = transportRate;
            j["slipAngle"] = slipAngle; j["settleIterations"] = settleIterations;
            j["waterIterations"] = waterIterations; j["windStrength"] = windStrength;
            j["windAzimuth"] = windAzimuth; j["useSceneSun"] = useSceneSun;
            j["sunAzimuth"] = sunAzimuth; j["sunElevation"] = sunElevation;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (name == "Snow Climate") name = "Snow Layer";
            preset = static_cast<SnowClimatePreset>(clampValue(j.value("preset", static_cast<int>(preset)), 0, 5));
            // Migration from the first Snow Climate version, whose presets
            // stored only an absolute metre snow line. Named presets can safely
            // receive their new relative defaults; Custom values stay untouched.
            if (!j.contains("snowLineFraction") && preset != SnowClimatePreset::Custom) applyPreset(preset);
            snowfallMeters = clampValue(j.value("snowfallMeters", snowfallMeters), 0.0f, 1000.0f);
            maxDepthMeters = clampValue(j.value("maxDepthMeters", maxDepthMeters), 0.01f, 1000.0f);
            affectGeometry = j.value("affectGeometry", affectGeometry);
            geometryAmount = clampValue(j.value("geometryAmount", geometryAmount), 0.0f, 2.0f);
            coverageAmount = clampValue(j.value("coverageAmount", coverageAmount), 0.0f, 2.0f);
            relativeSnowLine = j.value("relativeSnowLine", relativeSnowLine);
            snowLineFraction = clampValue(j.value("snowLineFraction", snowLineFraction), 0.0f, 1.0f);
            snowLineBlendFraction = clampValue(j.value("snowLineBlendFraction", snowLineBlendFraction), 0.001f, 1.0f);
            snowLine = j.value("snowLine", snowLine);
            snowLineTransition = clampValue(j.value("snowLineTransition", snowLineTransition), 0.01f, 1000.0f);
            baseTemperature = clampValue(j.value("baseTemperature", baseTemperature), -80.0f, 60.0f);
            lapseRate = clampValue(j.value("lapseRate", lapseRate), 0.0f, 20.0f);
            meltAmount = clampValue(j.value("meltAmount", meltAmount), 0.0f, 1.0f);
            solarMelt = clampValue(j.value("solarMelt", solarMelt), 0.0f, 1.0f);
            refreezeRate = clampValue(j.value("refreezeRate", refreezeRate), 0.0f, 1.0f);
            valleyCapture = clampValue(j.value("valleyCapture", valleyCapture), 0.0f, 2.0f);
            transportRate = clampValue(j.value("transportRate", transportRate), 0.0f, 1.0f);
            slipAngle = clampValue(j.value("slipAngle", slipAngle), 5.0f, 80.0f);
            settleIterations = clampValue(j.value("settleIterations", settleIterations), 1, 64);
            waterIterations = clampValue(j.value("waterIterations", waterIterations), 1, 64);
            windStrength = clampValue(j.value("windStrength", windStrength), 0.0f, 1.0f);
            windAzimuth = j.value("windAzimuth", windAzimuth);
            useSceneSun = j.value("useSceneSun", useSceneSun);
            sunAzimuth = j.value("sunAzimuth", sunAzimuth);
            sunElevation = clampValue(j.value("sunElevation", sunElevation), -89.0f, 89.0f);
            if (j.value("solverVersion", 1) < 2 && preset != SnowClimatePreset::Custom) {
                // Version 1 presets used depths far outside the default
                // terrain's physical scale. Migrate named presets to the
                // stable solver values while retaining the artist's new
                // geometry/coverage choices when they already exist.
                const bool savedAffectGeometry = affectGeometry;
                const float savedGeometryAmount = geometryAmount;
                const float savedCoverageAmount = coverageAmount;
                applyPreset(preset);
                affectGeometry = savedAffectGeometry;
                geometryAmount = savedGeometryAmount;
                coverageAmount = savedCoverageAmount;
            }
        }
    };

    class ClimateNode : public TerrainNodeBase {
    public:
        float seaLevelTemperature = 4.0f;
        float lapseRate = 6.5f;
        float freezePoint = 0.0f;
        float temperatureTransition = 3.0f;
        float snowLine = 6.0f;
        float snowLineTransition = 2.0f;
        float solarHeating = 4.0f;
        float sunAzimuth = 180.0f;
        float sunElevation = 35.0f;

        ClimateNode() {
            name = "Climate";
            terrainNodeType = NodeType::Climate;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Exposure", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Coldness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Solar Heat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Climate";
            metadata.category = "Snow & Ice";
            metadata.headerColor = IM_COL32(80, 145, 190, 255);
            headerColor = ImVec4(0.31f, 0.57f, 0.75f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Climate"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["seaLevelTemperature"] = seaLevelTemperature; j["lapseRate"] = lapseRate;
            j["freezePoint"] = freezePoint; j["temperatureTransition"] = temperatureTransition;
            j["snowLine"] = snowLine; j["snowLineTransition"] = snowLineTransition;
            j["solarHeating"] = solarHeating; j["sunAzimuth"] = sunAzimuth;
            j["sunElevation"] = sunElevation;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            seaLevelTemperature = clampValue(j.value("seaLevelTemperature", seaLevelTemperature), -50.0f, 50.0f);
            lapseRate = clampValue(j.value("lapseRate", lapseRate), 0.0f, 20.0f);
            freezePoint = clampValue(j.value("freezePoint", freezePoint), -20.0f, 10.0f);
            temperatureTransition = clampValue(j.value("temperatureTransition", temperatureTransition), 0.1f, 20.0f);
            snowLine = j.value("snowLine", snowLine);
            snowLineTransition = clampValue(j.value("snowLineTransition", snowLineTransition), 0.01f, 1000.0f);
            solarHeating = clampValue(j.value("solarHeating", solarHeating), 0.0f, 20.0f);
            sunAzimuth = j.value("sunAzimuth", sunAzimuth);
            sunElevation = clampValue(j.value("sunElevation", sunElevation), 1.0f, 89.0f);
        }
    };

    class SnowfallNode : public TerrainNodeBase {
    public:
        float amount = 0.80f;
        float snowLine = 6.0f;
        float snowLineTransition = 2.0f;
        float slopeAdhesion = 0.45f;
        float sunLoss = 0.35f;
        float windScour = 0.20f;
        float patchScale = 18.0f;
        int seed = 211;

        SnowfallNode() {
            name = "Snowfall";
            terrainNodeType = NodeType::Snowfall;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Coldness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Exposure", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Snow Mass", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Snowfall";
            metadata.category = "Snow & Ice";
            metadata.headerColor = IM_COL32(175, 210, 230, 255);
            headerColor = ImVec4(0.68f, 0.82f, 0.90f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Snowfall"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["amount"] = amount; j["snowLine"] = snowLine; j["snowLineTransition"] = snowLineTransition;
            j["slopeAdhesion"] = slopeAdhesion; j["sunLoss"] = sunLoss; j["windScour"] = windScour;
            j["patchScale"] = patchScale; j["seed"] = seed;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            amount = clampValue(j.value("amount", amount), 0.0f, 4.0f);
            snowLine = j.value("snowLine", snowLine);
            snowLineTransition = clampValue(j.value("snowLineTransition", snowLineTransition), 0.01f, 1000.0f);
            slopeAdhesion = clampValue(j.value("slopeAdhesion", slopeAdhesion), 0.0f, 1.0f);
            sunLoss = clampValue(j.value("sunLoss", sunLoss), 0.0f, 1.0f);
            windScour = clampValue(j.value("windScour", windScour), 0.0f, 1.0f);
            patchScale = clampValue(j.value("patchScale", patchScale), 1.0f, 256.0f);
            seed = j.value("seed", seed);
        }
    };

    class SnowSettleNode : public TerrainNodeBase {
    public:
        int iterations = 8;
        float slipAngle = 38.0f;
        float avalancheRate = 0.35f;
        float compaction = 0.15f;
        float windAzimuth = 45.0f;
        float windStrength = 0.12f;
        float depthScale = 0.025f;

        SnowSettleNode() {
            name = "Snow Settle";
            terrainNodeType = NodeType::SnowSettle;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Snow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Settled Snow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Snow Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            metadata.displayName = "Snow Settle";
            metadata.category = "Snow & Ice";
            metadata.headerColor = IM_COL32(145, 190, 220, 255);
            headerColor = ImVec4(0.57f, 0.75f, 0.86f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.SnowSettle"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["iterations"] = iterations; j["slipAngle"] = slipAngle; j["avalancheRate"] = avalancheRate;
            j["compaction"] = compaction; j["windAzimuth"] = windAzimuth;
            j["windStrength"] = windStrength; j["depthScale"] = depthScale;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            iterations = clampValue(j.value("iterations", iterations), 1, 64);
            slipAngle = clampValue(j.value("slipAngle", slipAngle), 5.0f, 80.0f);
            avalancheRate = clampValue(j.value("avalancheRate", avalancheRate), 0.0f, 1.0f);
            compaction = clampValue(j.value("compaction", compaction), 0.0f, 1.0f);
            windAzimuth = j.value("windAzimuth", windAzimuth);
            windStrength = clampValue(j.value("windStrength", windStrength), 0.0f, 0.5f);
            depthScale = clampValue(j.value("depthScale", depthScale), 0.0f, 0.25f);
        }
    };

    class SnowMeltFreezeNode : public TerrainNodeBase {
    public:
        float meltRate = 0.55f;
        float solarMelt = 0.45f;
        float freezeRate = 0.45f;
        float iceCompaction = 0.15f;

        SnowMeltFreezeNode() {
            name = "Snow Melt / Freeze";
            terrainNodeType = NodeType::SnowMeltFreeze;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Snow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Coldness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Solar Heat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Wetness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Snow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Meltwater", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Ice", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Snow Melt / Freeze";
            metadata.category = "Snow & Ice";
            metadata.headerColor = IM_COL32(90, 170, 210, 255);
            headerColor = ImVec4(0.35f, 0.67f, 0.82f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.SnowMeltFreeze"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["meltRate"] = meltRate; j["solarMelt"] = solarMelt;
            j["freezeRate"] = freezeRate; j["iceCompaction"] = iceCompaction;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            meltRate = clampValue(j.value("meltRate", meltRate), 0.0f, 1.0f);
            solarMelt = clampValue(j.value("solarMelt", solarMelt), 0.0f, 1.0f);
            freezeRate = clampValue(j.value("freezeRate", freezeRate), 0.0f, 1.0f);
            iceCompaction = clampValue(j.value("iceCompaction", iceCompaction), 0.0f, 1.0f);
        }
    };

    class GlacierFlowNode : public TerrainNodeBase {
    public:
        int iterations = 10;
        float flowStrength = 0.25f;
        float iceDepthScale = 0.04f;
        float carvingStrength = 0.004f;
        float depositionStrength = 0.0015f;

        GlacierFlowNode() {
            name = "Glacier Flow";
            terrainNodeType = NodeType::GlacierFlow;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Ice", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Glacial Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Ice", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Glacier Flow";
            metadata.category = "Snow & Ice";
            metadata.headerColor = IM_COL32(70, 155, 200, 255);
            headerColor = ImVec4(0.27f, 0.61f, 0.78f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.GlacierFlow"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["iterations"] = iterations; j["flowStrength"] = flowStrength;
            j["iceDepthScale"] = iceDepthScale; j["carvingStrength"] = carvingStrength;
            j["depositionStrength"] = depositionStrength;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            iterations = clampValue(j.value("iterations", iterations), 1, 64);
            flowStrength = clampValue(j.value("flowStrength", flowStrength), 0.0f, 0.75f);
            iceDepthScale = clampValue(j.value("iceDepthScale", iceDepthScale), 0.0f, 0.25f);
            carvingStrength = clampValue(j.value("carvingStrength", carvingStrength), 0.0f, 0.05f);
            depositionStrength = clampValue(j.value("depositionStrength", depositionStrength), 0.0f, 0.05f);
        }
    };

    // ============================================================================
    // TERRAIN GRAPH WRAPPER
    // ============================================================================
    
    /**
     * @brief Terrain-specialized graph using V2 system
     */
    class TerrainNodeGraphV2 : public NodeSystem::GraphBase {
    public:
        enum class DirtyEvaluationImpact {
            None,
            MaterialOnly,
            FoliageOnly,
            GeometryOrScene
        };

        TerrainNodeGraphV2() = default;
        
        // Factory method for creating nodes by type
        NodeSystem::NodeBase* addTerrainNode(NodeType type, float x = 0, float y = 0);

        // Evaluate with terrain context (synchronous — used by load/deserialize,
        // which run before the UI is interactive and don't need backgrounding)
        void evaluateTerrain(TerrainObject* terrain, struct ::SceneData& scene);

        // ========================================================================
        // ASYNC EVALUATION (interactive "Evaluate" button path)
        // ========================================================================
        // Split of evaluateTerrain() into a phase that's safe to run off the main
        // thread (pure CPU height-data compute) and phases that must stay on the
        // main thread (GPU texture upload for splat/hardness outputs, and mesh/
        // BVH/backend rebuild). See happy-kindling-flame.md plan for rationale.

        // Phase A (safe to call from a worker thread): pulls HeightOutputNode,
        // writes terrain->heightmap.data. No GPU/backend calls. Returns true if
        // heightmap data was updated (mirrors evaluateTerrain's early-returns).
        bool evaluateTerrainHeightData(TerrainObject* terrain, NodeSystem::EvaluationContext& ctx,
                                       uint32_t previewNodeId = 0);

        // Phase B (MAIN THREAD ONLY): pulls Splat/Hardness/field outputs and
        // applies RiverSpline sinks. This may upload textures and mutate scene
        // geometry through RiverManager.
        bool evaluateTerrainAuxOutputs(TerrainObject* terrain, struct ::SceneData& scene,
                                       NodeSystem::EvaluationContext& ctx);

        // Phase C (MAIN THREAD ONLY): resize/topology-mismatch mesh rebuild —
        // touches scene.world.objects / mesh_triangles, shared with render/BVH.
        // See lastFinalizeWasFullRebuild() for deferBackendSignal semantics.
        void finalizeTerrainMesh(struct ::SceneData& scene, TerrainObject* terrain, bool deferBackendSignal = false);

        // Kicks off phase A on a worker thread. No-op (returns immediately) if an
        // evaluation is already in flight for this graph.
        void evaluateTerrainAsync(TerrainObject* terrain, struct ::SceneData& scene);

        // Transient selected-node preview. The selected node becomes a temporary
        // terminal and only its upstream dependency subgraph is pulled. Cached
        // PinValues from the previous evaluation are reused when still valid.
        // Preview never becomes authored terrain state; restoreTerrainPreview()
        // returns to the last committed Height Output result.
        void evaluateTerrainPreviewAsync(uint32_t nodeId, TerrainObject* terrain, struct ::SceneData& scene);
        bool restoreTerrainPreview(TerrainObject* terrain, struct ::SceneData& scene);
        bool isPreviewActive() const { return previewActive_; }
        uint32_t previewNodeId() const { return displayedPreviewNodeId_; }

        // Call once per frame from the node-editor draw path (main thread). If the
        // background phase A has finished, runs phases B+C synchronously and
        // returns true (caller should then fire the usual GPU/backend rebuild
        // flags, same as the old synchronous button handler did). Returns false
        // while still evaluating or when nothing is pending.
        bool pollEvaluateAsync();

        // Classifies the currently dirty downstream contract. A mask edit that
        // reaches only Splat Output can reuse the committed height cache and
        // update material textures without touching terrain geometry/BVH.
        DirtyEvaluationImpact classifyDirtyEvaluationImpact();
        bool evaluateDirtyMaterialOutputs(TerrainObject* terrain, struct ::SceneData& scene);
        bool evaluateDirtyFoliageOutputs(TerrainObject* terrain);
        std::vector<int> getLastScatteredFoliageGroupIds() const;
        bool hasEvaluationCache() const { return cachedEvalContext_ != nullptr; }

        // True if the most recent finalizeTerrainMesh() call had to take the full
        // rebuild branch (terrain resolution/topology changed — new Triangle
        // objects, different triangle count). False means it took the cheap
        // in-place update branch (same Triangle pointers, positions changed only),
        // in which case the caller can use a partial BLAS/raster-mesh refit
        // instead of a full-scene rebuild — see updateTerrainBLASPartial usage in
        // scene_ui_terrain.hpp's brush-stroke commit path for the existing,
        // already-proven pattern this mirrors.
        bool lastFinalizeWasFullRebuild() const { return lastFinalizeWasFullRebuild_.load(); }

        std::atomic<bool> isEvaluating{false};
        std::shared_ptr<NodeSystem::EvaluationContext> activeEvalContext;

        // GraphBase overrides so NodeEditorUIV2::drawNode() can show a per-node
        // "currently active" indicator + overall progress generically.
        // Keep consumers locked until the completed worker result has also gone
        // through pollEvaluateAsync() and its main-thread finalize phase.
        bool isEvaluatingAsync() const override { return isEvaluating.load() || evalFuture_.valid(); }
        uint32_t currentAsyncNodeId() const override {
            return activeEvalContext ? activeEvalContext->getCurrentNodeId() : 0;
        }
        float asyncEvalProgress() const override {
            return activeEvalContext ? activeEvalContext->getProgress() : 0.0f;
        }
        NodeSystem::NodeEvaluationState asyncNodeState(uint32_t nodeId) const override {
            if (activeEvalContext) return activeEvalContext->getNodeState(nodeId);
            if (cachedEvalContext_) return cachedEvalContext_->getNodeState(nodeId);
            return NodeSystem::NodeEvaluationState::Idle;
        }

        // Reuse the most recent terrain-evaluation cache for lightweight UI
        // previews. This avoids pulling an expensive node chain a second time
        // merely to display one of the outputs it already produced.
        bool getCachedImageOutput(uint32_t nodeId, int outputIndex,
                                  NodeSystem::Image2DData& image) const {
            NodeSystem::NodeBase* node = getNode(nodeId);
            if (!node || node->dirty || !cachedEvalContext_ ||
                !cachedEvalContext_->hasCachedValue(nodeId, outputIndex)) {
                return false;
            }
            NodeSystem::PinValue value = cachedEvalContext_->getCachedValue(nodeId, outputIndex);
            const auto* cachedImage = std::get_if<NodeSystem::Image2DData>(&value);
            if (!cachedImage || !cachedImage->isValid()) return false;
            image = *cachedImage;
            return true;
        }

        // Create a default graph with basic nodes
        void createDefaultGraph(TerrainObject* terrain);

        // Non-destructive authoring helper: inserts a Snow Climate node before
        // the active Height Output and wires its material outputs to the active
        // Surface Composer/Splat Output chain. Existing grass/rock/flow inputs
        // on the composer are preserved.
        bool addSnowLayerSetup(float x = 520.0f, float y = 120.0f);

        // Non-destructive biome authoring helper. Reuses an existing analysis,
        // exposure, composer and fields-output chain when present, otherwise
        // creates and wires the missing nodes from the active Height Output source.
        bool addBiomeFieldsSetup(BiomeClimatePreset preset,
                                 float x = 520.0f, float y = 120.0f);

        // Adds four independent biome-driven foliage layers (forest, grass,
        // rock and alpine), grouped under one Foliage Set and output sink.
        bool addBiomeFoliageSetup(float x = 1120.0f, float y = 120.0f);

        // Non-destructive hydrology branch from the active authored height.
        // The Height Output connection is not replaced; the new branch publishes
        // a watershed, vector river network and owned RiverSpline sink.
        bool addRiverNetworkSetup(float x = 520.0f, float y = 520.0f);

        // Destructive example/template graph built from the same public nodes
        // artists use manually. Intended as a learnable starting point.
        void createSnowyMountainValleyGraph(TerrainObject* terrain);
        
        // ========================================================================
        // SERIALIZATION
        // ========================================================================
        
        /**
         * @brief Serialize the entire graph to JSON
         */
        nlohmann::json toJson() const;
        
        /**
         * @brief Deserialize the graph from JSON
         * @param j JSON object containing graph data
         * @param terrain Optional terrain for context during loading
         */
        void fromJson(const nlohmann::json& j, TerrainObject* terrain = nullptr);

    private:
        std::future<void> evalFuture_;
        TerrainObject* pendingFinalizeTerrain_ = nullptr;
        struct ::SceneData* pendingFinalizeScene_ = nullptr;
        std::unique_ptr<TerrainContext> activeTerrainCtx_;
        std::atomic<bool> lastEvaluateResized_{false};
        std::atomic<bool> lastFinalizeWasFullRebuild_{false};
        std::atomic<bool> lastHeightDataUpdated_{false};
        std::shared_ptr<NodeSystem::EvaluationContext> cachedEvalContext_;
        std::unique_ptr<TerrainContext> cachedTerrainCtx_;
        bool pendingEvaluationIsPreview_ = false;
        uint32_t pendingPreviewNodeId_ = 0;
        uint32_t displayedPreviewNodeId_ = 0;
        bool previewActive_ = false;
        int committedPreviewWidth_ = 0;
        int committedPreviewHeight_ = 0;
        float committedPreviewScaleXZ_ = 100.0f;
        float committedPreviewScaleY_ = 10.0f;
        std::vector<float> committedPreviewHeightData_;
        std::vector<float> committedPreviewFlowMap_;
        std::vector<float> committedPreviewHardnessMap_;
        std::vector<float> committedPreviewErosionMapRGBA_;

        void captureCommittedTerrainForPreview(TerrainObject* terrain);
        void restoreCommittedTerrainData(TerrainObject* terrain);
        void clearCommittedPreviewSnapshot();
    };

} // namespace TerrainNodesV2

