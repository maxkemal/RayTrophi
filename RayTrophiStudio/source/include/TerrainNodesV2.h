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
#include "TerrainNodePortPresentation.h"
#include "NodeSystem/EvaluationContext.h"
#include "TerrainManager.h"
#include "TerrainFieldMath.h"
#include "json.hpp"
#include "ui_modern.h"
#include <unordered_map>
#include <unordered_set>
#include <cmath>
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
        // Scene-authored curves are copied on the main thread before graph
        // evaluation. Workers consume only these immutable values.
        std::unordered_map<std::string, NodeSystem::CurveValue> curveSnapshots;
        Matrix4x4 terrainWorldToLocal = Matrix4x4::identity();
        
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
        PublishField,
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
        FoliageOutput,
        // Compact controller for the detailed river/lake hydrology subsystem.
        RiverLakeEasy,
        // Primary macro-landform synthesis. Appended for serialized enum stability.
        MountainRange,
        BasinValley,
        TerrainDetail,
        // Tectonic geology. Appended for serialized enum stability.
        PlateTectonics,
        Fold,
        SatMapColorRamp,
        SatMapOutput,
        // SatMap layer composition and reusable vegetation authoring.
        // Appended for serialized enum stability.
        SatMapBlend,
        GrassMask,
        SurfaceMasks,
        PaintMaskCombine,
        // Volcanic landforms. Appended for serialized enum stability.
        CraterCaldera,
        // XZ-plane placement transform (offset/scale/rotate) for an Image2D
        // field. Appended for serialized enum stability.
        Transform,
        // Terrain-aware detail and erosion resistance. Appended for serialized
        // enum stability; implementations live in TerrainSurfaceNodes.*.
        StructuralHardness,
        SurfaceRelief,
        // Scene spline interoperability. Appended for serialized enum stability.
        CurveInput,
        CurveToMask,
        RoadCarve,
        RoadNetwork,
        // Canonical infrastructure publication. Appended for enum stability.
        RoadFieldsOutput
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

        void onRegisteredToGraph() override {
            configureTerrainNodePorts(*this);
        }
        
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

        /**
         * @brief Read a Mask pin, reconciling an unbounded field into 0-1.
         *
         * Mask pins accept PhysicalScalar sources (discharge, drainage area,
         * water depth) because the hydrology nodes genuinely produce the
         * fields the biome and surface nodes want. Those fields carry SI
         * values, so a consumer that clamps them to 0-1 collapses them into
         * a binary stencil - every channel pixel exactly 1, everything else
         * exactly 0, which is what "the masks look crushed" actually was.
         * Normalizing here, at the single input boundary, keeps every
         * downstream consumer honest without each one re-deriving a range.
         */
        NodeSystem::Image2DData getMaskInput(int inputIndex, NodeSystem::EvaluationContext& ctx) {
            return TerrainFieldMath::asMaskField(getHeightInput(inputIndex, ctx));
        }

        /// Sample metric for a field of @p width samples in the active terrain.
        TerrainFieldMath::FieldMetric getFieldMetric(NodeSystem::EvaluationContext& ctx, int width) {
            if (auto* tctx = getTerrainContext(ctx)) {
                return TerrainFieldMath::makeFieldMetric(tctx->scale_xz, tctx->scale_y, width);
            }
            return TerrainFieldMath::makeFieldMetric(static_cast<float>((std::max)(width - 1, 1)),
                                                     1.0f, width);
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
        // Scale multiplier for loaded heightmap ("Intensity" in the UI).
        // Default 10 pairs with the 1000 m default terrain and scale_y = 10:
        // normalized source values become ~100 m of relief, which is the range
        // the SI-unit hydrology (discharge, channel geometry, fluvial
        // thresholds) is tuned for. Serialized graphs keep their own value.
        float heightScale = 10.0f;
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
            metadata.iconType = (int)UIWidgets::IconType::Terrain;
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
            metadata.iconType = (int)UIWidgets::IconType::ClayTool;
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
        Perlin,      // Fractal gradient FBM
        Voronoi,     // Worley cell walls
        Simplex,     // Turbulence (absolute-folded FBM)
        Ridge,       // Ridge/mountain chains
        Billow,      // Soft rolling hills
        Warped,      // Domain-warped organic
        // Serialized names retained for graph compatibility. These currently
        // use deterministic spectral-style CPU patterns, not a Fourier solver.
        FFT_Ocean,
        FFT_Ridge,
        FFT_Billow,
        FFT_Turb
    };

    enum class TerrainNoiseModel { RawNoise = 0, Continental, Orogenic, Highlands };

    class NoiseGeneratorNode : public TerrainNodeBase {
    public:
        NoiseType noiseType = NoiseType::Perlin;
        int seed = 1337;
        // Height is authored in world metres, always. Feature Size is the
        // wavelength of the largest landform and Relief is the peak-to-trough
        // range the node actually delivers; the generator fits its realised
        // distribution to that range instead of assuming a normalised FBM ever
        // reaches its own theoretical extremes.
        float featureSizeMeters = 600.0f;
        // Auto resolves Feature Size against the terrain the node is evaluated
        // on. A fixed metre value cannot be right for both: 600 m is a broad
        // valley on a 1 km tile and one of seven repeats on a 4 km one, and
        // past that repeat count nothing in the tile is wider than 600 m at any
        // other setting. Old projects carry an authored value and keep it -
        // deserializeFromJson turns Auto OFF when the saved node predates it.
        bool autoFeatureSize = true;
        float baseElevationMeters = 0.0f;
        float reliefMeters = 140.0f;
        int octaves = 10;
        float persistance = 0.5f;
        float jitter = 1.0f;           // Voronoi specific
        float warp_strength = 0.35f;   // Domain warp intensity
        float ridge_offset = 1.0f;     // Ridge crest sharpening exponent
        bool resolutionBandLimit = true;
        float samplesPerDetail = 3.0f;
        // Angle-of-repose ceiling for the generated relief. This is a needle
        // remover, not a terrain shaper: the multifractal base already sits
        // around 20 deg mean slope, so 60 clips only the extreme tail. Setting
        // it near the terrain's own mean slope terraces the result instead.
        // 90 disables the pass.
        float slopeLimitDegrees = 60.0f;
        TerrainNoiseModel terrainModel = TerrainNoiseModel::Orogenic;
        float terrainRoughness = 0.45f;
        // Fraction of the tile that stays gentle country. Authored as an AREA,
        // delivered as an area: the uplift field is thresholded at its own
        // quantile, so the number means the same thing whatever the seed does.
        // Not a look dial - it is the difference between a landscape and a
        // fractal, and 0 does not restore the old behaviour because the old
        // behaviour had no lowland at any setting.
        float lowlandFraction = 0.35f;
        // Diagnostics published by compute() for the panel. The resolution guard
        // silently drops bands the grid cannot carry, so "Detail Bands 12" can
        // mean 7; without this readout that gap is invisible and the control
        // looks broken. Not serialized, not authored.
        int lastEffectiveBands = 0;
        float lastFinestBandMeters = 0.0f;
        float lastFeatureMeters = 0.0f;
        float lastLowlandFraction = 0.0f;
        float terrainDirection = 25.0f;
        float valleyStrength = 0.35f;

        NoiseGeneratorNode() {
            name = "Noise Generator";
            terrainNodeType = NodeType::NoiseGenerator;
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Macro", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Ridge", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Valley", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Terrain Noise Generator";
            metadata.category = "Input";
            metadata.headerColor = IM_COL32(50, 150, 125, 255);
            metadata.iconType = (int)UIWidgets::IconType::Noise;
            headerColor = ImVec4(0.2f, 0.6f, 0.5f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;

        void drawContent() override {
            const char* terrainModels[] = { "Raw Noise", "Continental", "Orogenic Mountains", "Eroded Highlands" };
            int modelIndex = static_cast<int>(terrainModel);
            if (ImGui::Combo("Terrain Model", &modelIndex, terrainModels, IM_ARRAYSIZE(terrainModels))) {
                terrainModel = static_cast<TerrainNoiseModel>(modelIndex);
                dirty = true;
            }
            if (terrainModel == TerrainNoiseModel::RawNoise) {
                const char* noiseNames[] = {
                    "Fractal (fBm)", "Voronoi Cells", "Turbulence", "Ridged", "Billow", "Domain Warped"
                };
                int noiseIdx = (int)noiseType;
                if (ImGui::Combo("Type", &noiseIdx, noiseNames, 6)) {
                    noiseType = (NoiseType)noiseIdx;
                    dirty = true;
                }
            }
            if (ImGui::DragInt("Seed", &seed)) dirty = true;
            if (ImGui::Checkbox("Feature Size from terrain", &autoFeatureSize)) dirty = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "Size the largest landform to the terrain it sits on, about half\n"
                "the tile. Turn this off only to author the wavelength in metres\n"
                "directly, for instance to keep two terrains of different sizes\n"
                "carrying the same landform scale.");
            if (autoFeatureSize) {
                if (lastFeatureMeters > 0.0f)
                    ImGui::TextDisabled("  largest landform %.0f m", lastFeatureMeters);
            } else {
                if (ImGui::DragFloat("Feature Size", &featureSizeMeters, 5.0f, 4.0f, 100000.0f, "%.0f m")) dirty = true;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                    "World-space wavelength of the largest landform.\n"
                    "Keep it near terrain size / 1.5 to / 4. Much below that and\n"
                    "the tile carries no landform wider than this value, which\n"
                    "reads as one texture repeated rather than as country.\n"
                    "terrain.landform_stats measures what you actually got.");
            }
            if (ImGui::DragFloat("Base Elevation", &baseElevationMeters, 1.0f, -10000.0f, 10000.0f, "%.1f m")) dirty = true;
            if (ImGui::DragFloat("Relief", &reliefMeters, 1.0f, 0.0f, 20000.0f, "%.1f m")) dirty = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "Peak-to-trough range of this node in world metres.\n"
                "The generated distribution is fitted to it, so the value is delivered, not approximated.");
            if (ImGui::SliderFloat("Roughness", &terrainRoughness, 0.0f, 1.0f)) dirty = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "Fineness of the multifractal. Low gives smooth, broadly\n"
                "dissected relief; high gives fine, hairy ruggedness.\n"
                "It FEEDS a Hurst exponent rather than being one: the\n"
                "mapped range is 1.05..0.65 but the delivered range\n"
                "measures about a third of that, so both ends are\n"
                "milder than the numbers suggest. terrain.landform_stats\n"
                "reports realised_hurst.");
            if (terrainModel != TerrainNoiseModel::RawNoise) {
                if (ImGui::SliderFloat("Lowland", &lowlandFraction, 0.0f, 0.80f)) dirty = true;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                    "Fraction of the tile that stays gentle country between the\n"
                    "massifs. Raise it for sparse mountains over wide ground;\n"
                    "lower it for a tile that is uplifted edge to edge.\n"
                    "It is an area, not a strength: the uplift field is cut at\n"
                    "its own quantile, so the number holds across seeds.");
                if (lastLowlandFraction > 0.0f)
                    ImGui::TextDisabled("  %.0f%% of the tile at the dissection floor",
                        lastLowlandFraction * 100.0f);
            }
            if (ImGui::SliderFloat("Warp", &warp_strength, 0.0f, 1.0f)) dirty = true;
            if (terrainModel == TerrainNoiseModel::Orogenic) {
                if (ImGui::SliderFloat("Direction", &terrainDirection, 0.0f, 360.0f, "%.0f deg")) dirty = true;
                if (ImGui::SliderFloat("Valley Strength", &valleyStrength, 0.0f, 1.0f)) dirty = true;
            }
            if (ImGui::DragInt("Detail Bands", &octaves, 1, 1, 12)) dirty = true;
            if (lastEffectiveBands > 0) {
                if (lastEffectiveBands < octaves) {
                    ImGui::TextColored(ImVec4(0.95f, 0.78f, 0.35f, 1.0f),
                        "  using %d of %d - grid limit, finest %.0f m",
                        lastEffectiveBands, octaves, lastFinestBandMeters);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                        "The grid cannot represent finer bands at this resolution.\n"
                        "Raise terrain resolution, lower Samples / Detail, or reduce Feature Size.");
                } else {
                    ImGui::TextDisabled("  using %d bands, finest %.0f m",
                        lastEffectiveBands, lastFinestBandMeters);
                }
            }
            if (ImGui::DragFloat("Slope Limit", &slopeLimitDegrees, 0.5f, 25.0f, 90.0f, "%.0f deg")) dirty = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "Angle-of-repose ceiling. Removes sub-cell needles that stall droplet\n"
                "and thermal solvers. Keep it well above the terrain's own mean slope\n"
                "(~20 deg) or it terraces the relief. 90 disables the pass.");
            if (ImGui::Checkbox("Resolution-safe Detail", &resolutionBandLimit)) dirty = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "Drops bands finer than the terrain grid can represent.\n"
                "Each layer is measured against its own finest wavelength, including anisotropic stretch.");
            if (resolutionBandLimit && ImGui::DragFloat("Samples / Detail", &samplesPerDetail, 0.25f, 2.0f, 12.0f, "%.1f cells")) dirty = true;

            if (terrainModel != TerrainNoiseModel::RawNoise) return;
            if (ImGui::DragFloat("Persistence", &persistance, 0.01f, 0.05f, 0.95f)) dirty = true;
            if (noiseType == NoiseType::Voronoi) {
                if (ImGui::DragFloat("Jitter", &jitter, 0.01f, 0.0f, 1.0f)) dirty = true;
            }
            if (noiseType == NoiseType::Ridge) {
                if (ImGui::DragFloat("Crest Sharpness", &ridge_offset, 0.01f, 0.5f, 2.0f)) dirty = true;
            }
        }
        
        std::string getTypeId() const override { return "TerrainV2.NoiseGenerator"; }
        
        // Serialization overrides
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["noiseType"] = static_cast<int>(noiseType);
            j["seed"] = seed;
            j["featureSizeMeters"] = featureSizeMeters;
            j["autoFeatureSize"] = autoFeatureSize;
            j["baseElevationMeters"] = baseElevationMeters;
            j["reliefMeters"] = reliefMeters;
            j["octaves"] = octaves;
            j["persistance"] = persistance;
            j["jitter"] = jitter;
            j["warp_strength"] = warp_strength;
            j["ridge_offset"] = ridge_offset;
            j["resolutionBandLimit"] = resolutionBandLimit;
            j["samplesPerDetail"] = samplesPerDetail;
            j["slopeLimitDegrees"] = slopeLimitDegrees;
            j["terrainModel"] = static_cast<int>(terrainModel);
            j["terrainRoughness"] = terrainRoughness;
            j["lowlandFraction"] = lowlandFraction;
            j["terrainDirection"] = terrainDirection;
            j["valleyStrength"] = valleyStrength;
        }
        
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("noiseType")) {
                int storedType = j["noiseType"].get<int>();
                // The former FFT-labelled modes were CPU procedural variants.
                // Migrate them to their honest base equivalents on load.
                if (storedType == static_cast<int>(NoiseType::FFT_Ocean) ||
                    storedType == static_cast<int>(NoiseType::FFT_Turb)) storedType = static_cast<int>(NoiseType::Perlin);
                else if (storedType == static_cast<int>(NoiseType::FFT_Ridge)) storedType = static_cast<int>(NoiseType::Ridge);
                else if (storedType == static_cast<int>(NoiseType::FFT_Billow)) storedType = static_cast<int>(NoiseType::Billow);
                noiseType = static_cast<NoiseType>(clampValue(storedType, 0, 5));
            }
            if (j.contains("seed")) seed = j["seed"].get<int>();
            featureSizeMeters = (std::max)(j.value("featureSizeMeters", featureSizeMeters), 1.0f);
            // A node saved before Auto existed has an authored wavelength and
            // no flag. Defaulting the flag to its member value would silently
            // discard that wavelength and re-shape every old terrain on load,
            // so absence of the key IS the answer: this node authored metres.
            autoFeatureSize = j.contains("autoFeatureSize")
                ? j["autoFeatureSize"].get<bool>()
                : !j.contains("featureSizeMeters");
            baseElevationMeters = j.value("baseElevationMeters", baseElevationMeters);
            reliefMeters = (std::max)(j.value("reliefMeters", reliefMeters), 0.0f);
            if (j.contains("octaves")) octaves = clampValue(j["octaves"].get<int>(), 1, 12);
            persistance = clampValue(j.value("persistance", persistance), 0.05f, 0.95f);
            jitter = clampValue(j.value("jitter", jitter), 0.0f, 1.0f);
            warp_strength = clampValue(j.value("warp_strength", warp_strength), 0.0f, 1.0f);
            ridge_offset = clampValue(j.value("ridge_offset", ridge_offset), 0.5f, 2.0f);
            resolutionBandLimit = j.value("resolutionBandLimit", resolutionBandLimit);
            samplesPerDetail = clampValue(j.value("samplesPerDetail", samplesPerDetail), 2.0f, 12.0f);
            slopeLimitDegrees = clampValue(j.value("slopeLimitDegrees", slopeLimitDegrees), 25.0f, 90.0f);
            terrainModel = j.contains("terrainModel")
                ? static_cast<TerrainNoiseModel>(clampValue(j["terrainModel"].get<int>(), 0, 3))
                : TerrainNoiseModel::RawNoise;
            terrainRoughness = clampValue(j.value("terrainRoughness", terrainRoughness), 0.0f, 1.0f);
            lowlandFraction = clampValue(j.value("lowlandFraction", lowlandFraction), 0.0f, 0.80f);
            terrainDirection = j.value("terrainDirection", terrainDirection);
            valleyStrength = clampValue(j.value("valleyStrength", valleyStrength), 0.0f, 1.0f);
        }
    };

    // ============================================================================
    // PRIMARY LANDFORM GENERATORS
    // ============================================================================

    enum class MountainLandformPreset {
        AlpineChain = 0, RoundedMassif, DesertRanges,
        CoastalMountains, VolcanicHighlands, Custom
    };

    class MountainRangeNode : public TerrainNodeBase {
    public:
        MountainLandformPreset preset = MountainLandformPreset::AlpineChain;
        int seed = 4201;
        float direction = 25.0f;
        float centerX = 0.5f, centerY = 0.5f;
        float lengthFraction = 1.05f;
        // Fraction of the terrain, the same unit Length already used. Width was
        // the only absolute metre value on this node, and the mismatch was the
        // whole failure: on a 4096 m tile the default 420 m gave a full-length
        // ribbon 420 m wide, measured at 0.01 degrees median slope - a welt on
        // a flat plate, not a mountain range.
        float widthFraction = 0.28f;
        float reliefMeters = 120.0f;
        float ridgeSharpness = 3.2f;
        float warp = 0.32f;
        float branches = 0.62f;
        float detail = 0.24f;
        float massif = 0.46f;
        float foothills = 0.38f;
        float peakVariation = 0.58f;
        float asymmetry = 0.12f;

        MountainRangeNode() {
            name = "Mountain Range";
            terrainNodeType = NodeType::MountainRange;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Base Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Uplift", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Ridge", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            // Appended outputs preserve serialized pin indices used by existing
            // projects while exposing the geological fields needed downstream.
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Valley Seed", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Fracture", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Mountain Range / Orogeny";
            metadata.category = "Landform";
            metadata.description = "Metre-scaled orogeny with catchments, ridge hierarchy and geological detail";
            metadata.headerColor = IM_COL32(155, 105, 72, 255);
            metadata.iconType = (int)UIWidgets::IconType::DrawTool;
            headerColor = ImVec4(0.61f, 0.41f, 0.28f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        void applyPreset(MountainLandformPreset value);
        static const char* getPresetName(MountainLandformPreset value);
        std::string getTypeId() const override { return "TerrainV2.MountainRange"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["landformPreset"] = static_cast<int>(preset);
            j["seed"] = seed; j["direction"] = direction;
            j["centerX"] = centerX; j["centerY"] = centerY;
            j["lengthFraction"] = lengthFraction; j["widthFraction"] = widthFraction;
            j["reliefMeters"] = reliefMeters; j["ridgeSharpness"] = ridgeSharpness;
            j["warp"] = warp; j["branches"] = branches; j["detail"] = detail;
            j["massif"] = massif; j["foothills"] = foothills;
            j["peakVariation"] = peakVariation; j["asymmetry"] = asymmetry;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            preset = static_cast<MountainLandformPreset>(
                clampValue(j.value("landformPreset", static_cast<int>(preset)), 0, 5));
            seed = j.value("seed", seed); direction = j.value("direction", direction);
            centerX = clampValue(j.value("centerX", centerX), 0.0f, 1.0f);
            centerY = clampValue(j.value("centerY", centerY), 0.0f, 1.0f);
            lengthFraction = clampValue(j.value("lengthFraction", lengthFraction), 0.05f, 1.5f);
            // Renamed rather than reinterpreted: a stored `widthMeters` means
            // metres and must not be read as a fraction. Legacy values are
            // converted at the 1500 m reference the five landform presets
            // imply - a stated assumption, so a project authored on a very
            // different tile needs its Range Width looked at once.
            if (j.contains("widthFraction"))
                widthFraction = clampValue(j["widthFraction"].get<float>(), 0.005f, 2.0f);
            else if (j.contains("widthMeters"))
                widthFraction = clampValue(j["widthMeters"].get<float>() / 1500.0f,
                                           0.005f, 2.0f);
            reliefMeters = (std::max)(j.value("reliefMeters", reliefMeters), 0.0f);
            ridgeSharpness = clampValue(j.value("ridgeSharpness", ridgeSharpness), 0.5f, 8.0f);
            warp = clampValue(j.value("warp", warp), 0.0f, 1.0f);
            branches = clampValue(j.value("branches", branches), 0.0f, 1.0f);
            detail = clampValue(j.value("detail", detail), 0.0f, 1.0f);
            massif = clampValue(j.value("massif", massif), 0.0f, 1.0f);
            foothills = clampValue(j.value("foothills", foothills), 0.0f, 1.0f);
            peakVariation = clampValue(j.value("peakVariation", peakVariation), 0.0f, 1.0f);
            asymmetry = clampValue(j.value("asymmetry", asymmetry), -1.0f, 1.0f);
            // Named presets are authoritative; only Custom preserves authored
            // scalar values. This also migrates the former 10x relief presets.
            if (preset != MountainLandformPreset::Custom) applyPreset(preset);
        }
    };

    enum class ValleyProfile { VShape = 0, UShape = 1 };
    class BasinValleyNode : public TerrainNodeBase {
    public:
        int seed = 731;
        ValleyProfile profile = ValleyProfile::VShape;
        float direction = 0.0f;
        float center = 0.5f;
        float widthMeters = 180.0f;
        float floorWidth = 0.15f;
        float depthMeters = 120.0f;
        float meander = 0.18f;
        float shoulderSoftness = 0.55f;

        BasinValleyNode() {
            name = "Basin & Valley";
            terrainNodeType = NodeType::BasinValley;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Valley", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Basin & Valley";
            metadata.category = "Landform";
            metadata.description = "World-scale V/U valley profile with a meandering axis";
            metadata.headerColor = IM_COL32(92, 128, 104, 255);
            metadata.iconType = (int)UIWidgets::IconType::FlattenTool;
            headerColor = ImVec4(0.36f, 0.50f, 0.41f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.BasinValley"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["seed"] = seed; j["profile"] = static_cast<int>(profile);
            j["direction"] = direction; j["center"] = center;
            j["widthMeters"] = widthMeters; j["floorWidth"] = floorWidth;
            j["depthMeters"] = depthMeters; j["meander"] = meander;
            j["shoulderSoftness"] = shoulderSoftness;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            seed = j.value("seed", seed);
            profile = static_cast<ValleyProfile>(clampValue(j.value("profile", 0), 0, 1));
            direction = j.value("direction", direction);
            center = clampValue(j.value("center", center), 0.0f, 1.0f);
            widthMeters = (std::max)(j.value("widthMeters", widthMeters), 1.0f);
            floorWidth = clampValue(j.value("floorWidth", floorWidth), 0.0f, 0.9f);
            depthMeters = (std::max)(j.value("depthMeters", depthMeters), 0.0f);
            meander = clampValue(j.value("meander", meander), 0.0f, 1.0f);
            shoulderSoftness = clampValue(j.value("shoulderSoftness", shoulderSoftness), 0.05f, 1.0f);
        }
    };

    class TerrainDetailNode : public TerrainNodeBase {
    public:
        int seed = 991;
        float macroSizeMeters = 800.0f, macroAmountMeters = 30.0f;
        float mediumSizeMeters = 160.0f, mediumAmountMeters = 12.0f;
        float microSizeMeters = 28.0f, microAmountMeters = 3.0f;
        float roughness = 0.55f;
        bool preserveMean = true;

        TerrainDetailNode() {
            name = "Terrain Detail";
            terrainNodeType = NodeType::TerrainDetail;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            metadata.displayName = "Terrain Detail";
            metadata.category = "Landform";
            metadata.description = "Independent macro, medium and micro world-scale relief";
            metadata.headerColor = IM_COL32(116, 112, 150, 255);
            metadata.iconType = (int)UIWidgets::IconType::DrawSharpTool;
            headerColor = ImVec4(0.45f, 0.44f, 0.59f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.TerrainDetail"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["seed"] = seed;
            j["macroSizeMeters"] = macroSizeMeters; j["macroAmountMeters"] = macroAmountMeters;
            j["mediumSizeMeters"] = mediumSizeMeters; j["mediumAmountMeters"] = mediumAmountMeters;
            j["microSizeMeters"] = microSizeMeters; j["microAmountMeters"] = microAmountMeters;
            j["roughness"] = roughness; j["preserveMean"] = preserveMean;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            seed = j.value("seed", seed);
            macroSizeMeters = (std::max)(j.value("macroSizeMeters", macroSizeMeters), 1.0f);
            macroAmountMeters = (std::max)(j.value("macroAmountMeters", macroAmountMeters), 0.0f);
            mediumSizeMeters = (std::max)(j.value("mediumSizeMeters", mediumSizeMeters), 1.0f);
            mediumAmountMeters = (std::max)(j.value("mediumAmountMeters", mediumAmountMeters), 0.0f);
            microSizeMeters = (std::max)(j.value("microSizeMeters", microSizeMeters), 0.1f);
            microAmountMeters = (std::max)(j.value("microAmountMeters", microAmountMeters), 0.0f);
            roughness = clampValue(j.value("roughness", roughness), 0.05f, 0.95f);
            preserveMean = j.value("preserveMean", preserveMean);
        }
    };

    // ============================================================================
    // EROSION NODES
    // ============================================================================
    
    enum class HydraulicMultiPassPreset {
        Balanced = 0,
        Alpine,
        Humid,
        Arid,
        Custom
    };

    class HydraulicErosionNode : public TerrainNodeBase {
    public:
        HydraulicErosionParams params;
        bool useGPU = true;
        bool multiPass = true;
        HydraulicMultiPassPreset multiPassPreset = HydraulicMultiPassPreset::Balanced;
        double lastEroded = 0.0;
        double lastDeposited = 0.0;
        double lastDischarge = 0.0;
        double lastSedimentFlux = 0.0;
        float lastSolveMs = 0.0f;
        int lastExecutedPasses = 0;
        HydraulicErosionStats lastStats;

        // Edge Falloff Settings
        float edgeFalloffWidth = 0.0f;
        float edgeFalloffValue = 0.0f;

        HydraulicErosionNode() {
            name = "Hydraulic Erosion";
            terrainNodeType = NodeType::HydraulicErosion;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Area", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Wear", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Deposits", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            // ★★★★ THIS PIN WAS CALLED "Flow" AND IT CARRIES SEDIMENT.
            //
            // compute() returns `sediment` here - the suspended-load field,
            // log-normalised to 0..1 - while the physically meaningful
            // DISCHARGE the solver computes had no output pin at all and only
            // reached the RGBA preview texture's blue channel.
            //
            // The cost of the name: FlowMask.Discharge is the drainage
            // authority, its own comment says it "carries measured m3/s when
            // Hydraulic Erosion is connected", and the only pin here that
            // looked like it meant flow was this one. Wiring it makes
            // flow_authority report `measured` while measuring sediment - a
            // FALSE GREEN, strictly worse than the honest `unwired` state it
            // replaces. Renamed rather than left alone, because the next
            // person to look for "flow" on this node would make the same link.
            //
            // Saved graphs are safe: the loader matches by stableKey first and
            // falls back to the pin INDEX when the key is gone and the port
            // list has grown. An old link keyed "flow" therefore lands on index
            // 3 - this same pin, same data, new name.
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Sediment", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar));
            // The field the drainage authority actually wants. Appended, so
            // every serialized graph keeps its existing pin indices.
            // m3/s: catchment area times runoff, straight from the solver, NOT
            // normalised - FlowMask classifies against the field's own min/max
            // and a pre-normalised input would throw away the scale it needs.
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Discharge", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::PhysicalScalar,
                1, NodeSystem::ImageUnit::CubicMetersPerSecond));

            metadata.displayName = "Hydraulic Erosion";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(75, 125, 200, 255);
            metadata.iconType = (int)UIWidgets::IconType::Water;
            headerColor = ImVec4(0.3f, 0.5f, 0.8f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        void applyMultiPassPreset(HydraulicMultiPassPreset preset);
        static const char* getMultiPassPresetName(HydraulicMultiPassPreset preset);
        std::string getTypeId() const override { return "TerrainV2.HydraulicErosion"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["useGPU"] = useGPU;
            j["multiPass"] = multiPass;
            j["multiPassPreset"] = static_cast<int>(multiPassPreset);
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
                {"erosionRadius", params.erosionRadius},
                {"initialWater", params.initialWater},
                {"initialSpeed", params.initialSpeed},
                {"uphillErosion", params.uphillErosion},
                {"flatSettling", params.flatSettling},
                {"velocitySettling", params.velocitySettling},
                {"minWater", params.minWater},
                {"minSpeed", params.minSpeed},
                {"removeSpikes", params.removeSpikes},
                {"fillPits", params.fillPits},
                {"smoothSurface", params.smoothSurface},
                {"seed", params.seed},
                {"boundaryMode", static_cast<int>(params.boundaryMode)},
                {"boundaryWidth", params.boundaryWidth},
                {"boundaryLevel", params.boundaryLevel},
                {"channelEvolution", params.channelEvolution},
                {"channelIterations", params.channelIterations},
                {"channelErosion", params.channelErosion},
                {"channelDeposition", params.channelDeposition},
                {"channelWidthScale", params.channelWidthScale},
                {"channelDepthScale", params.channelDepthScale},
                {"macroDrainage", params.macroDrainage},
                {"macroValleyScaleMeters", params.macroValleyScaleMeters},
                {"macroHeadwaterAreaKm2", params.macroHeadwaterAreaKm2},
                {"macroValleyDepthMeters", params.macroValleyDepthMeters},
                {"macroValleyFloor", params.macroValleyFloor},
                {"fluvialCycle", params.fluvialCycle},
                {"fluvialIterations", params.fluvialIterations},
                {"fluvialTimeStep", params.fluvialTimeStep},
                {"rainRate", params.rainRate},
                {"orographicRain", params.orographicRain},
                {"rainWindDegrees", params.rainWindDegrees},
                {"incisionK", params.incisionK},
                {"streamPowerM", params.streamPowerM},
                {"streamPowerN", params.streamPowerN},
                {"slopeMin", params.slopeMin},
                {"slopeMax", params.slopeMax},
                {"transportK", params.transportK},
                {"sedimentCover", params.sedimentCover},
                {"settlingVelocity", params.settlingVelocity},
                {"sedimentRouteSteps", params.sedimentRouteSteps},
                {"avulsionInterval", params.avulsionInterval},
                {"alluviumSlopeDegrees", params.alluviumSlopeDegrees},
                {"alluviumRate", params.alluviumRate},
                {"alluviumSteps", params.alluviumSteps},
                {"alluviumConsolidation", params.alluviumConsolidation},
                {"drainageRefreshInterval", params.drainageRefreshInterval},
                {"maxDepositionMeters", params.maxDepositionMeters},
                {"drainageFillPasses", params.drainageFillPasses},
                {"drainageAccumulatePasses", params.drainageAccumulatePasses},
                {"drainageCoarsestSize", params.drainageCoarsestSize},
                {"flatGradient", params.flatGradient},
                {"flatResolvePasses", params.flatResolvePasses},
                {"massWasting", params.massWasting},
                {"reposeAngleDegrees", params.reposeAngleDegrees},
                {"massWastingRate", params.massWastingRate},
                {"massWastingSteps", params.massWastingSteps},
                {"hillslopeDiffusion", params.hillslopeDiffusion},
                {"channelRefAreaKm2", params.channelRefAreaKm2},
                {"incisionSafety", params.incisionSafety},
                {"depositionSafety", params.depositionSafety},
                {"maxStepMeters", params.maxStepMeters},
                {"lakeEpsilonMeters", params.lakeEpsilonMeters},
                {"fluvialWidthScale", params.fluvialWidthScale},
                {"fluvialDepthScale", params.fluvialDepthScale},
                {"fluvialHeadwaterAreaKm2", params.fluvialHeadwaterAreaKm2}
            };
            j["edgeFalloffWidth"] = edgeFalloffWidth;
            j["edgeFalloffValue"] = edgeFalloffValue;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("useGPU")) useGPU = j["useGPU"].get<bool>();
            multiPass = j.value("multiPass", multiPass);
            multiPassPreset = static_cast<HydraulicMultiPassPreset>(
                clampValue(j.value("multiPassPreset", static_cast<int>(multiPassPreset)), 0, 4));
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
                params.initialWater = p.value("initialWater", params.initialWater);
                params.initialSpeed = p.value("initialSpeed", params.initialSpeed);
                params.uphillErosion = p.value("uphillErosion", params.uphillErosion);
                params.flatSettling = p.value("flatSettling", params.flatSettling);
                params.velocitySettling = p.value("velocitySettling", params.velocitySettling);
                params.minWater = p.value("minWater", params.minWater);
                params.minSpeed = p.value("minSpeed", params.minSpeed);
                params.removeSpikes = p.value("removeSpikes", params.removeSpikes);
                params.fillPits = p.value("fillPits", params.fillPits);
                params.smoothSurface = p.value("smoothSurface", params.smoothSurface);
                params.seed = p.value("seed", params.seed);
                params.boundaryMode = static_cast<ErosionBoundaryMode>(
                    clampValue(p.value("boundaryMode", static_cast<int>(params.boundaryMode)), 0, 2));
                params.boundaryWidth = clampValue(p.value("boundaryWidth", params.boundaryWidth), 0, 512);
                params.boundaryLevel = p.value("boundaryLevel", params.boundaryLevel);
                params.channelEvolution = p.value("channelEvolution", params.channelEvolution);
                params.channelIterations = clampValue(p.value("channelIterations", params.channelIterations), 0, 128);
                params.channelErosion = clampValue(p.value("channelErosion", params.channelErosion), 0.0f, 2.0f);
                params.channelDeposition = clampValue(p.value("channelDeposition", params.channelDeposition), 0.0f, 2.0f);
                params.channelWidthScale = clampValue(p.value("channelWidthScale", params.channelWidthScale), 0.01f, 20.0f);
                params.channelDepthScale = clampValue(p.value("channelDepthScale", params.channelDepthScale), 0.01f, 20.0f);
                params.macroDrainage = p.value("macroDrainage", params.macroDrainage);
                params.macroValleyScaleMeters = clampValue(p.value("macroValleyScaleMeters", params.macroValleyScaleMeters), 20.0f, 6000.0f);
                params.macroHeadwaterAreaKm2 = clampValue(p.value("macroHeadwaterAreaKm2", params.macroHeadwaterAreaKm2), 0.0005f, 25.0f);
                params.macroValleyDepthMeters = clampValue(p.value("macroValleyDepthMeters", params.macroValleyDepthMeters), 0.0f, 500.0f);
                params.macroValleyFloor = clampValue(p.value("macroValleyFloor", params.macroValleyFloor), 0.0f, 1.0f);

                // Landscape Evolution Model cycle. Absent keys keep the
                // defaults, so an older project opens with the cycle ON: the
                // whole point is that the previous behaviour was the bug.
                params.fluvialCycle = p.value("fluvialCycle", params.fluvialCycle);
                params.fluvialIterations = clampValue(p.value("fluvialIterations", params.fluvialIterations), 0, 512);
                params.fluvialTimeStep = clampValue(p.value("fluvialTimeStep", params.fluvialTimeStep), 0.0f, 16.0f);
                params.rainRate = clampValue(p.value("rainRate", params.rainRate), 1.0e-4f, 100.0f);
                params.orographicRain = clampValue(p.value("orographicRain", params.orographicRain), 0.0f, 1.0f);
                params.rainWindDegrees = p.value("rainWindDegrees", params.rainWindDegrees);
                params.incisionK = clampValue(p.value("incisionK", params.incisionK), 0.0f, 20000.0f);
                params.streamPowerM = clampValue(p.value("streamPowerM", params.streamPowerM), 0.0f, 2.0f);
                params.streamPowerN = clampValue(p.value("streamPowerN", params.streamPowerN), 0.1f, 4.0f);
                params.slopeMin = clampValue(p.value("slopeMin", params.slopeMin), 1.0e-6f, 1.0f);
                params.slopeMax = clampValue(p.value("slopeMax", params.slopeMax), 1.0e-3f, 100.0f);
                params.transportK = clampValue(p.value("transportK", params.transportK), 0.0f, 20000.0f);
                params.sedimentCover = clampValue(p.value("sedimentCover", params.sedimentCover), 0.0f, 1.0f);
                params.settlingVelocity = clampValue(p.value("settlingVelocity", params.settlingVelocity), 0.0f, 100.0f);
                params.sedimentRouteSteps = clampValue(p.value("sedimentRouteSteps", params.sedimentRouteSteps), 0, 4096);
                params.avulsionInterval = clampValue(p.value("avulsionInterval", params.avulsionInterval), 0, 4096);
                params.alluviumSlopeDegrees = clampValue(p.value("alluviumSlopeDegrees", params.alluviumSlopeDegrees), 0.1f, 20.0f);
                params.alluviumRate = clampValue(p.value("alluviumRate", params.alluviumRate), 0.0f, 1.0f);
                params.alluviumSteps = clampValue(p.value("alluviumSteps", params.alluviumSteps), 0, 64);
                params.alluviumConsolidation = clampValue(p.value("alluviumConsolidation", params.alluviumConsolidation), 0.0f, 1.0f);
                params.drainageRefreshInterval = clampValue(p.value("drainageRefreshInterval", params.drainageRefreshInterval), 1, 64);
                params.maxDepositionMeters = clampValue(p.value("maxDepositionMeters", params.maxDepositionMeters), 0.0f, 500.0f);
                params.drainageFillPasses = clampValue(p.value("drainageFillPasses", params.drainageFillPasses), 8, 4096);
                params.drainageAccumulatePasses = clampValue(p.value("drainageAccumulatePasses", params.drainageAccumulatePasses), 8, 4096);
                params.drainageCoarsestSize = clampValue(p.value("drainageCoarsestSize", params.drainageCoarsestSize), 32, 512);
                params.flatGradient = clampValue(p.value("flatGradient", params.flatGradient), 0.0f, 0.05f);
                params.flatResolvePasses = clampValue(p.value("flatResolvePasses", params.flatResolvePasses), 0, 8192);
                params.massWasting = p.value("massWasting", params.massWasting);
                params.reposeAngleDegrees = clampValue(p.value("reposeAngleDegrees", params.reposeAngleDegrees), 1.0f, 80.0f);
                params.massWastingRate = clampValue(p.value("massWastingRate", params.massWastingRate), 0.0f, 1.0f);
                params.massWastingSteps = clampValue(p.value("massWastingSteps", params.massWastingSteps), 0, 64);
                params.hillslopeDiffusion = clampValue(p.value("hillslopeDiffusion", params.hillslopeDiffusion), 0.0f, 100.0f);
                params.channelRefAreaKm2 = clampValue(p.value("channelRefAreaKm2", params.channelRefAreaKm2), 1.0e-6f, 100.0f);
                params.incisionSafety = clampValue(p.value("incisionSafety", params.incisionSafety), 0.0f, 0.95f);
                params.depositionSafety = clampValue(p.value("depositionSafety", params.depositionSafety), 0.0f, 0.95f);
                params.maxStepMeters = clampValue(p.value("maxStepMeters", params.maxStepMeters), 0.0f, 1000.0f);
                params.lakeEpsilonMeters = clampValue(p.value("lakeEpsilonMeters", params.lakeEpsilonMeters), 0.0f, 100.0f);
                params.fluvialWidthScale = clampValue(p.value("fluvialWidthScale", params.fluvialWidthScale), 0.0f, 100.0f);
                params.fluvialDepthScale = clampValue(p.value("fluvialDepthScale", params.fluvialDepthScale), 0.0f, 100.0f);
                params.fluvialHeadwaterAreaKm2 = clampValue(p.value("fluvialHeadwaterAreaKm2", params.fluvialHeadwaterAreaKm2), 1.0e-6f, 100.0f);
            }
            if (j.contains("edgeFalloffWidth")) edgeFalloffWidth = j["edgeFalloffWidth"].get<float>();
            if (j.contains("edgeFalloffValue")) edgeFalloffValue = j["edgeFalloffValue"].get<float>();
        }
    };

    enum class ThermalErosionPreset {
        Balanced = 0, AlpineScree, DesertRock, SoftSediment, Custom
    };

    class ThermalErosionNode : public TerrainNodeBase {
    public:
        ThermalErosionParams params;
        ThermalErosionPreset preset = ThermalErosionPreset::Balanced;
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
            // Append-only output contract: keep Height Out at index 0 for
            // serialized graph stability.
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Erosion", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Deposition", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Talus", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Rock Exposure", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Thermal Erosion";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(75, 125, 200, 255);
            metadata.iconType = (int)UIWidgets::IconType::ScrapeTool;
            headerColor = ImVec4(0.3f, 0.5f, 0.8f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        void applyPreset(ThermalErosionPreset value);
        static const char* getPresetName(ThermalErosionPreset value);
        std::string getTypeId() const override { return "TerrainV2.ThermalErosion"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["useGPU"] = useGPU;
            j["preset"] = static_cast<int>(preset);
            j["params"] = {
                {"iterations", params.iterations},
                {"talusAngle", params.talusAngle},
                {"erosionAmount", params.erosionAmount},
                {"anisotropy", params.anisotropy},
                {"anisotropyDirection", params.anisotropyDirection},
                {"talusSettling", params.talusSettling},
                {"sedimentRemoval", params.sedimentRemoval},
                {"fineDetail", params.fineDetail},
                {"debrisSizeMeters", params.debrisSizeMeters}
            };
            j["edgeFalloffWidth"] = edgeFalloffWidth;
            j["edgeFalloffValue"] = edgeFalloffValue;
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("useGPU")) useGPU = j["useGPU"].get<bool>();
            preset = static_cast<ThermalErosionPreset>(
                clampValue(j.value("preset", static_cast<int>(preset)), 0, 4));
            if (j.contains("params")) {
                const auto& p = j["params"];
                params.iterations = p.value("iterations", params.iterations);
                params.talusAngle = p.value("talusAngle", params.talusAngle);
                params.erosionAmount = p.value("erosionAmount", params.erosionAmount);
                params.anisotropy = clampValue(p.value("anisotropy", params.anisotropy), 0.0f, 1.0f);
                params.anisotropyDirection = p.value("anisotropyDirection", params.anisotropyDirection);
                params.talusSettling = clampValue(p.value("talusSettling", params.talusSettling), 0.0f, 2.0f);
                params.sedimentRemoval = clampValue(p.value("sedimentRemoval", params.sedimentRemoval), 0.0f, 1.0f);
                params.fineDetail = p.value("fineDetail", params.fineDetail);
                params.debrisSizeMeters = (std::max)(p.value("debrisSizeMeters", params.debrisSizeMeters), 0.01f);
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
                "Flow Guide", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar, true,
                1, NodeSystem::ImageUnit::Unitless));
            inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::Mask);
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Erosion Map", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, 4));
            
            metadata.displayName = "Fluvial Erosion";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(75, 150, 230, 255);
            metadata.iconType = (int)UIWidgets::IconType::Vortex;
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
            metadata.iconType = (int)UIWidgets::IconType::Wind;
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
            metadata.iconType = (int)UIWidgets::IconType::World;
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
            metadata.iconType = (int)UIWidgets::IconType::Render;
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
                "Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PackedData, false, 4));
            // Appended for serialized pin stability. This texture is not
            // normalized with material weights.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Semantic (Flow/Wet/Ice/Hard)", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::PackedData, true, 4));
            
            metadata.displayName = "Splat Output";
            metadata.category = "Output";
            metadata.headerColor = IM_COL32(200, 75, 125, 255);
            metadata.iconType = (int)UIWidgets::IconType::ViewMatcap;
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
            metadata.iconType = (int)UIWidgets::IconType::ShadeFlatTool;
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

        // B is an OPERAND, not a modifier: a two-input operator with one input
        // is not an operation. It was declared optional so a missing B could
        // fall back to a `factor` scalar, and that second path cost exactly
        // what silent dual paths cost in this repo. Once every terrain node
        // received a default exposure profile, "optional" meant the socket
        // drew only while connected - so Math showed ONE input and the second
        // one could not be wired at all. Scaling a field by a constant is
        // Remap's job, and `factor` is gone rather than left as a dead dial.
        
        MathNode() {
            name = "Math";
            terrainNodeType = NodeType::Add;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "A", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "B", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Result", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Math";
            metadata.category = "Math";
            metadata.headerColor = IM_COL32(100, 100, 150, 255);
            metadata.iconType = (int)UIWidgets::IconType::AddKey;
            headerColor = ImVec4(0.4f, 0.4f, 0.6f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Math"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["operation"] = static_cast<int>(operation);
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            if (j.contains("operation")) operation = static_cast<MathOp>(j["operation"].get<int>());
            // `factor` is deliberately not read back. A project that leaned on
            // the scalar path now reports a missing input naming this node -
            // which is the point: it used to produce a plausible-looking
            // result out of half a graph.
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
            metadata.iconType = (int)UIWidgets::IconType::LayerTool;
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
            metadata.iconType = (int)UIWidgets::IconType::PivotEdit;
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
            metadata.iconType = (int)UIWidgets::IconType::ViewSolid;
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
            metadata.iconType = (int)UIWidgets::IconType::Rotate;
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
            metadata.iconType = (int)UIWidgets::IconType::MaskTool;
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
            metadata.iconType = (int)UIWidgets::IconType::CreaseTool;
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
    /**
     * @brief The single authority for "how much water is here".
     *
     * Two different quantities were both called flow, and consumers got
     * whichever happened to be wired:
     *   DISCHARGE - a physical field (drainage area / m3 per s). Hydraulic
     *     Erosion measures it against the landscape it is actually carving.
     *   CHANNEL   - a 0-1 selection: which cells read as a visible
     *     watercourse. detailLevel/softness/bankSpread are STYLE dials on
     *     that selection, not physics.
     *
     * This node now owns both, and states which source the discharge came
     * from. Wire Hydraulic Erosion's Discharge into it and the classification
     * runs on the measured field; leave it empty and the field is derived
     * from the height alone - a defensible drainage-area proxy, but it knows
     * nothing about lakes, infiltration or erosion history, so a graph that
     * HAS an erosion sim and does not wire it here is classifying one
     * landscape while rendering another. That case is called out in the panel
     * rather than left to look plausible.
     *
     * Renamed from "Flow / Soil": it stopped feeding Soil Depth's deposition
     * when flow and deposition were separated, and a name that outlives its
     * job is how the next reader gets misled.
     */
    class FlowMaskNode : public TerrainNodeBase {
    public:
        int detailLevel = 6;          // 1=main rivers, 8=finest tributaries
        int bankSpread = 1;           // Small channel-width expansion, never global blur
        float strength = 1.0f;        // Flow strength multiplier
        float decay = 0.995f;         // Discharge retained at each downstream step
        float channelSoftness = 0.06f;// Soft transition around the selected detail threshold
        bool normalize = true;        // Normalize output to 0-1

        /// Set by compute() so the panel can report which source was used
        /// instead of leaving the artist to guess. Not serialized: these are
        /// observations, not settings, and a stale observation restored from
        /// disk would be worse than none.
        bool lastEvaluated = false;
        bool lastDischargeMeasured = false;
        /// The expensive case: the graph HAS an erosion sim and this node is
        /// not reading it, so the channels are classified from bare geometry
        /// while the render shows an eroded landscape. It still looks like a
        /// river network, which is why it needs saying out loud.
        bool lastErosionUnwired = false;

        FlowMaskNode() {
            name = "Flow";
            terrainNodeType = NodeType::FlowMask;

            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            // Appended, so serialized graphs keep their pin indices. Measured
            // discharge takes over from the derived accumulation entirely -
            // this is an authority, not a blend.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Discharge", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::PhysicalScalar, true));
            inputs[1].acceptImageSemantic(NodeSystem::ImageSemantic::Mask);

            outputs.push_back(NodeSystem::Pin::createOutput(
                "Channel", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            // The canonical flow-magnitude field. It carries measured m3/s
            // when Hydraulic Erosion is connected, or derived drainage area
            // in m2 otherwise. Every consumer reads it here so a graph has one
            // source of truth; the runtime image unit states which quantity.
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Discharge", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::PhysicalScalar,
                1, NodeSystem::ImageUnit::Unknown));

            metadata.displayName = "Flow";
            metadata.category = "Mask";
            metadata.headerColor = IM_COL32(100, 150, 200, 255);
            metadata.iconType = (int)UIWidgets::IconType::Water;
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
            metadata.iconType = (int)UIWidgets::IconType::LightDir;
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
            metadata.iconType = (int)UIWidgets::IconType::SmoothTool;
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
            metadata.iconType = (int)UIWidgets::IconType::ScaleAxis;
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
            metadata.iconType = (int)UIWidgets::IconType::ShadeFlatTool;
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
            metadata.iconType = (int)UIWidgets::IconType::PinchTool;
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
            metadata.iconType = (int)UIWidgets::IconType::MergeVertices;
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
            metadata.iconType = (int)UIWidgets::IconType::LayerTool;
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
            metadata.iconType = (int)UIWidgets::IconType::ViewRendered;
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
            
            // Default material rules: Grass/Rock/Snow/Soil.
            // Layer 0 (R): Grass - flat, low-mid height
            rules[0] = { 0.0f, 50.0f, 0.0f, 25.0f, 0.5f, 0.5f, 10.0f, 0.05f, true };
            // Layer 1 (G): Rock - steep slopes
            rules[1] = { 0.0f, 200.0f, 30.0f, 90.0f, 0.2f, 0.8f, 5.0f, 0.03f, true };
            // Layer 2 (B): Snow - high altitude
            rules[2] = { 80.0f, 200.0f, 0.0f, 45.0f, 0.9f, 0.1f, 15.0f, 0.02f, true };
            // Layer 3 (A): exposed soil. Flow is kept in the semantic output.
            rules[3] = { 0.0f, 20.0f, 0.0f, 15.0f, 0.6f, 0.4f, 8.0f, 0.1f, true };
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            // Appended for serialized pin-index stability. Flow remains an
            // independent non-normalized semantic mask.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Appended: the rules below are authored in degrees, but the angle
            // used to be re-derived here on a private stencil. Wiring the
            // shared Terrain Analysis slope makes Auto Splat and Surface
            // Composer classify the same pixel the same way.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Slope", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Appended: the semantic output wrote Flow and hard-zeroed the
            // other three, so an Auto Splat graph was structurally unable to
            // drive a Wetness, Ice or Hardness material - a dead end with no
            // symptom. These pass through; nothing is synthesized, because
            // these channels reach erosion resistance and soil capacity.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Wetness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Ice", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            for (size_t index = 1; index < inputs.size(); ++index) {
                inputs[index].acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
            }

            // 4-channel output for splat map
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PackedData, 4));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Semantic (Flow/Wet/Ice/Hard)", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::PackedData, 4));
            
            metadata.displayName = "Auto Splat";
            metadata.category = "Texture";
            metadata.headerColor = IM_COL32(200, 150, 50, 255);
            metadata.iconType = (int)UIWidgets::IconType::PaintTool;
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
            metadata.iconType = (int)UIWidgets::IconType::PaintTool;
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
            metadata.iconType = (int)UIWidgets::IconType::Assets;
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
            metadata.iconType = (int)UIWidgets::IconType::DissolveTopology;
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
            metadata.iconType = (int)UIWidgets::IconType::FlattenTool;
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
            metadata.iconType = (int)UIWidgets::IconType::NudgeTool;
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
            metadata.iconType = (int)UIWidgets::IconType::Sensitivity;
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

    /**
     * @brief Moves/scales/rotates an Image2D field in the XZ plane.
     *
     * Samples the source field at the inverse-transformed position of each
     * output pixel, so a positive offset moves content in the +X/+Z world
     * direction rather than sliding the sampling window the other way.
     * Offsets are authored in world meters (via the terrain's scale_xz),
     * not pixels, so the same node setting behaves consistently across
     * resolution changes.
     */
    enum class TransformEdgeMode { Clamp = 0, Wrap = 1, Zero = 2 };

    class TransformNode : public TerrainNodeBase {
    public:
        float offsetXMeters = 0.0f;
        float offsetZMeters = 0.0f;
        float scaleX = 1.0f;
        float scaleZ = 1.0f;
        float rotationDegrees = 0.0f;
        TransformEdgeMode edgeMode = TransformEdgeMode::Clamp;
        ResampleSemantic semanticMode = ResampleSemantic::Height;

        TransformNode() {
            name = "Transform";
            terrainNodeType = NodeType::Transform;
            inputs.push_back(NodeSystem::Pin::createInput(
                "In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            // 1.0 where a pixel came from real source data, 0.0 in the padding
            // a downscale/offset opens up. Independent of Edge mode below, so
            // it stays correct whichever padding Out uses - wire it into a
            // downstream Blend's Mask input to composite this transformed
            // field onto a base (e.g. a mountain) WITHOUT the base being
            // overwritten/summed in the area the shrunk content no longer
            // covers. Without this, Add/Blend-without-a-mask paints the
            // padding (0, or a smeared edge value under Clamp) straight over
            // the base there - "boşalan alanların bozulması".
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Coverage", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Transform";
            metadata.category = "Utility";
            metadata.headerColor = IM_COL32(80, 145, 180, 255);
            metadata.iconType = (int)UIWidgets::IconType::Move;
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
        std::string getTypeId() const override { return "TerrainV2.Transform"; }

        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["offsetXMeters"] = offsetXMeters;
            j["offsetZMeters"] = offsetZMeters;
            j["scaleX"] = scaleX;
            j["scaleZ"] = scaleZ;
            j["rotationDegrees"] = rotationDegrees;
            j["edgeMode"] = static_cast<int>(edgeMode);
            j["semanticMode"] = static_cast<int>(semanticMode);
        }

        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            offsetXMeters = j.value("offsetXMeters", offsetXMeters);
            offsetZMeters = j.value("offsetZMeters", offsetZMeters);
            scaleX = clampValue(j.value("scaleX", scaleX), 0.01f, 100.0f);
            scaleZ = clampValue(j.value("scaleZ", scaleZ), 0.01f, 100.0f);
            rotationDegrees = j.value("rotationDegrees", rotationDegrees);
            edgeMode = static_cast<TransformEdgeMode>(clampValue(j.value("edgeMode", 0), 0, 2));
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
            metadata.iconType = (int)UIWidgets::IconType::EyedropperTool;
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
                "Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PackedData, 4));
            metadata.displayName = "Splat Compose";
            metadata.category = "Texture";
            metadata.headerColor = IM_COL32(195, 135, 55, 255);
            metadata.iconType = (int)UIWidgets::IconType::LayerTool;
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
            metadata.iconType = (int)UIWidgets::IconType::Settings;
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
            metadata.iconType = (int)UIWidgets::IconType::DodgeTool;
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
            metadata.iconType = (int)UIWidgets::IconType::InflateTool;
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
                "Flow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar, true,
                1, NodeSystem::ImageUnit::Unitless));
            inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::Mask);
            for (const char* outputName : {"Slope", "Concavity", "Convexity", "Valley", "Wetness"}) {
                outputs.push_back(NodeSystem::Pin::createOutput(
                    outputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            }
            metadata.displayName = "Terrain Analysis";
            metadata.category = "Data Maps";
            metadata.description = "Cached slope, curvature, valley and wetness fields";
            metadata.headerColor = IM_COL32(62, 145, 180, 255);
            metadata.iconType = (int)UIWidgets::IconType::Graph;
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
        // Numerical ladder of the priority flood ONLY. It has to stay above
        // float resolution near h = 0.5 (one ulp is about 6e-8) or the fill
        // becomes bitwise flat and steepest descent lays down parallel
        // "power line" channels. It is NOT a slope threshold - see
        // flatSlopePercent, which used to be derived from this and therefore
        // scaled with heightScale and mesh resolution instead of with terrain.
        float flatEpsilon = 0.00001f;
        // Below this PHYSICAL slope a cell is routed as a flat (Garbrecht &
        // Martz exit-distance BFS) instead of by steepest descent. A natural
        // alluvial valley floor runs 0.05-0.5 %, so anything at or above that
        // swallows real river reaches. Percent, resolution independent.
        float flatSlopePercent = 0.02f;
        // Hybrid breach-fill (Lindsay 2016). A pit whose outlet can be cut for
        // less than this many cubic metres is BREACHED - the sill is carved
        // down along the drainage path - instead of the basin being raised to
        // meet it. Filling is what puts a fictitious surface under the river
        // mesh and hides real hills from routing; breaching leaves the routed
        // path inside terrain that actually exists. Set to 0 to fill only
        // (the pre-2026-08-26 behaviour).
        // ★ The discriminating variable is DEPTH, not volume: a numerical pit
        // has a sill under a metre, a real lake basin sits many metres below
        // its sill. The first defaults (25000 m3 / 40 m) were so permissive
        // that every basin on a 1 km terrain was breachable and the scene
        // produced NO lakes at all.
        float breachBudgetCubicMeters = 2000.0f;
        // A breach may never cut deeper than this below the original surface.
        // Without it one pathological rim turns into a canyon.
        float breachMaximumDepthMeters = 3.0f;
        // ★ A volume budget alone does NOT bound the LENGTH of the cut, and
        // length is what makes a breach look wrong. On a 1 km / 2048 field the
        // cell is 0.24 m2, so 2000 m3 buys a one-metre-deep trench over eight
        // thousand cells - a canal ruled straight across the valley, through
        // whatever ridges were in the way. A sill notch is tens of metres.
        float breachMaximumLengthMeters = 60.0f;

        // Last solve's outcome. Deliberately NOT serialized - these are a
        // measurement, and a measurement restored from a file is a default
        // wearing a measurement's clothes.
        int   lastBreachedPits = 0;
        int   lastFilledPits = 0;
        float lastBreachVolumeCubicMeters = 0.0f;

        WatershedAnalysisNode() {
            name = "Watershed Analysis";
            terrainNodeType = NodeType::WatershedAnalysis;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            // Appended optional input: serialized graphs map pins by index, so
            // new pins must stay at the end. Scales per-cell rainfall — wire a
            // snow-melt or climate mask here so flow accounts for snow water.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Precipitation", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Water Input Depth", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::PhysicalScalar, true, 1, NodeSystem::ImageUnit::Meters));
            // NOT "Filled Height" any more: with breaching enabled this surface
            // can sit BELOW the input height as well as above it. The rename is
            // deliberate - a consumer that still wants "the lake surface" must
            // be re-read rather than silently handed a different quantity.
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Conditioned Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Accumulation", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                1, NodeSystem::ImageUnit::Unitless));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Flow Direction", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Direction,
                1, NodeSystem::ImageUnit::Unitless));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Drainage Basins", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Categorical,
                1, NodeSystem::ImageUnit::Identifier));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Catchment Area", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                1, NodeSystem::ImageUnit::SquareMeters));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Runoff Volume", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                1, NodeSystem::ImageUnit::CubicMeters));
            // Appended output: serialized graphs map pins by index, so it stays
            // last. How deep the breach cut the sill, in metres, zero where the
            // pit was filled instead. This has to LEAVE the node: a breach that
            // only exists in the conditioning surface removes the pit for
            // routing while the ridge still stands in the rendered terrain, so
            // the extracted river climbs over a ridge that is really there.
            // Wire it into River Bed Carve so the cut is committed.
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Breach Depth", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                1, NodeSystem::ImageUnit::Meters));
            metadata.displayName = "Watershed Analysis";
            metadata.category = "Hydrology";
            metadata.description = "Depression-safe D8 drainage, accumulation and catchments";
            metadata.headerColor = IM_COL32(48, 132, 190, 255);
            metadata.iconType = (int)UIWidgets::IconType::Console;
            headerColor = ImVec4(0.19f, 0.52f, 0.75f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.WatershedAnalysis"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["rainfall"] = rainfall;
            j["flatEpsilon"] = flatEpsilon;
            j["flatSlopePercent"] = flatSlopePercent;
            j["breachBudgetCubicMeters"] = breachBudgetCubicMeters;
            j["breachMaximumDepthMeters"] = breachMaximumDepthMeters;
            j["breachMaximumLengthMeters"] = breachMaximumLengthMeters;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            rainfall = clampValue(j.value("rainfall", rainfall), 0.001f, 100.0f);
            flatEpsilon = clampValue(j.value("flatEpsilon", flatEpsilon), 0.0000001f, 0.01f);
            flatSlopePercent = clampValue(j.value("flatSlopePercent", flatSlopePercent), 0.0f, 25.0f);
            breachBudgetCubicMeters = clampValue(
                j.value("breachBudgetCubicMeters", breachBudgetCubicMeters), 0.0f, 1.0e9f);
            breachMaximumDepthMeters = clampValue(
                j.value("breachMaximumDepthMeters", breachMaximumDepthMeters), 0.0f, 500.0f);
            breachMaximumLengthMeters = clampValue(
                j.value("breachMaximumLengthMeters", breachMaximumLengthMeters), 0.0f, 100000.0f);
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
        // Evaluation-local accepted footprint. Published as an internal terrain
        // hydrology field so visible river output cannot depend on manual wiring.
        NodeSystem::Image2DData pendingLakeMask;

        LakeBasinNode() {
            name = "Lake Basin";
            terrainNodeType = NodeType::LakeBasin;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Original Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Conditioned Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Direction", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Direction, true,
                1, NodeSystem::ImageUnit::Unitless));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Runoff Volume", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::PhysicalScalar, true, 1, NodeSystem::ImageUnit::CubicMeters));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Lake Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Lake Depth", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Water Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Shoreline", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Spill Points", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Lake IDs", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Categorical,
                1, NodeSystem::ImageUnit::Identifier));
            metadata.displayName = "Lake Basin";
            metadata.category = "Hydrology";
            metadata.description = "Extracts lake levels, shorelines, storage and spill outlets";
            metadata.headerColor = IM_COL32(35, 145, 190, 255);
            metadata.iconType = (int)UIWidgets::IconType::FillTool;
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
        struct AuthoredWaterProfile {
            int surfaceId = -1;
            std::string name;
            WaterWaveParams params;
            uint64_t featureId = 0;
            Vec3 anchor = Vec3(0.0f);
            float extent = 0.0f;
            bool hasIdentity = false;
        };

        float surfaceOffsetMeters = 0.02f;
        float uvScaleMeters = 4.0f;
        int maximumGeneratedLakes = 32;
        bool generateWaterMeshes = true;
        int sourceLakeNodeId = -1;
        std::vector<int> generatedWaterSurfaceIds;
        // Producer-owned copy of Water UI authorship. Generated scene objects
        // are disposable during project-open Evaluate; this payload is not.
        std::vector<AuthoredWaterProfile> authoredWaterProfiles;
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
                "Lake IDs", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Categorical,
                false, 1, NodeSystem::ImageUnit::Identifier));
            metadata.displayName = "Lake Surface Output";
            metadata.category = "Output";
            metadata.description = "Builds owned WaterSurface meshes from analytical lakes";
            metadata.headerColor = IM_COL32(25, 155, 198, 255);
            metadata.iconType = (int)UIWidgets::IconType::ViewPreview;
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
            nlohmann::json profiles = nlohmann::json::array();
            std::unordered_set<int> writtenSurfaceIds;
            std::unordered_set<uint64_t> writtenFeatureIds;
            std::unordered_set<std::string> writtenNames;
            for (int surfaceId : generatedWaterSurfaceIds) {
                const WaterSurface* surface = WaterManager::getInstance().getWaterSurface(surfaceId);
                if (!surface) continue;
                profiles.push_back({
                    {"surfaceId", surface->id}, {"name", surface->name},
                    {"params", surface->params.serializeParams()},
                    {"featureId", surface->generated_feature_id},
                    {"anchor", {surface->generated_anchor.x, surface->generated_anchor.y,
                                 surface->generated_anchor.z}},
                    {"extent", surface->generated_extent},
                    {"hasIdentity", surface->has_generated_identity}
                });
                writtenSurfaceIds.insert(surface->id);
                writtenFeatureIds.insert(surface->generated_feature_id);
                writtenNames.insert(surface->name);
            }
            // Preserve the last producer-owned copy if serialization happens
            // during a transient rebuild where the runtime object is absent.
            for (const AuthoredWaterProfile& profile : authoredWaterProfiles) {
                if (writtenSurfaceIds.count(profile.surfaceId) != 0 ||
                    writtenNames.count(profile.name) != 0 ||
                    (profile.hasIdentity && writtenFeatureIds.count(profile.featureId) != 0)) continue;
                profiles.push_back({
                    {"surfaceId", profile.surfaceId}, {"name", profile.name},
                    {"params", profile.params.serializeParams()},
                    {"featureId", profile.featureId},
                    {"anchor", {profile.anchor.x, profile.anchor.y, profile.anchor.z}},
                    {"extent", profile.extent}, {"hasIdentity", profile.hasIdentity}
                });
            }
            j["authoredWaterProfiles"] = std::move(profiles);
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            surfaceOffsetMeters = clampValue(j.value("surfaceOffsetMeters", surfaceOffsetMeters), -10.0f, 10.0f);
            uvScaleMeters = clampValue(j.value("uvScaleMeters", uvScaleMeters), 0.01f, 100000.0f);
            maximumGeneratedLakes = clampValue(j.value("maximumGeneratedLakes", maximumGeneratedLakes), 1, 4096);
            generateWaterMeshes = j.value("generateWaterMeshes", generateWaterMeshes);
            generatedWaterSurfaceIds = j.value("generatedWaterSurfaceIds", std::vector<int>{});
            authoredWaterProfiles.clear();
            if (j.contains("authoredWaterProfiles") && j["authoredWaterProfiles"].is_array()) {
                for (const auto& saved : j["authoredWaterProfiles"]) {
                    AuthoredWaterProfile profile;
                    profile.surfaceId = saved.value("surfaceId", -1);
                    profile.name = saved.value("name", std::string{});
                    profile.featureId = saved.value("featureId", uint64_t{0});
                    profile.extent = saved.value("extent", 0.0f);
                    profile.hasIdentity = saved.value("hasIdentity", false);
                    if (saved.contains("params") && saved["params"].is_object()) {
                        profile.params.deserializeParams(saved["params"]);
                    }
                    if (saved.contains("anchor") && saved["anchor"].is_array() &&
                        saved["anchor"].size() >= 3) {
                        profile.anchor = Vec3(saved["anchor"][0].get<float>(),
                                              saved["anchor"][1].get<float>(),
                                              saved["anchor"][2].get<float>());
                    }
                    authoredWaterProfiles.push_back(std::move(profile));
                }
            }
        }
    };

    // Converts the watershed's continuous accumulation/direction fields into a
    // pruned, topologically ordered stream network suitable for spline output.
    class RiverNetworkNode : public TerrainNodeBase {
    public:
        float catchmentThreshold = 0.0015f;
        float minimumCatchmentAreaSquareMeters = 5000.0f;
        int minimumBranchLength = 8;

        RiverNetworkNode() {
            name = "River Network";
            terrainNodeType = NodeType::RiverNetwork;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Accumulation", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                false, 1, NodeSystem::ImageUnit::Unknown));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Direction", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Direction,
                false, 0, NodeSystem::ImageUnit::Unitless));
            // Appended for pin-index serialization stability. When connected,
            // this physical area replaces the domain-relative legacy threshold.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Catchment Area", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                true, 1, NodeSystem::ImageUnit::SquareMeters));
            // Appended for pin-index serialization stability. A lake owns its
            // complete footprint, so extracted channels must leave it empty.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Optional explicit outlet authority for vector flow fields that
            // are intentionally still inside standing water. Appended for
            // pin-index serialization stability.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Spill Points", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::Mask, true));
            // Cells that must never be NAMED a channel: authored infrastructure.
            // Wire Road Network's Road Core (and Ditch, through a Math max) here.
            // A carved road is a linear depression, so to a threshold on
            // accumulation it is a perfect river bed - which is exactly how one
            // came to be painted along every road.
            //
            // It excludes only the classification. Routing and accumulation are
            // untouched, so the water still reaches the same downstream cells;
            // the road pushes it off the surface and the ditch carries it. The
            // failure this avoids is the quiet one - a drainage network cut in
            // half by an exclusion that also removed the flow.
            // Appended for pin-index serialization stability.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Channel Exclusion", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Channels", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Stream Order", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Categorical,
                1, NodeSystem::ImageUnit::Identifier));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Sources", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "River Network";
            metadata.category = "Hydrology";
            metadata.description = "Extracts and prunes a connected stream hierarchy";
            metadata.headerColor = IM_COL32(42, 151, 203, 255);
            metadata.iconType = (int)UIWidgets::IconType::AnimGraph;
            headerColor = ImVec4(0.16f, 0.59f, 0.80f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.RiverNetwork"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["catchmentThreshold"] = catchmentThreshold;
            j["minimumCatchmentAreaSquareMeters"] = minimumCatchmentAreaSquareMeters;
            j["minimumBranchLength"] = minimumBranchLength;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            catchmentThreshold = clampValue(j.value("catchmentThreshold", catchmentThreshold), 0.00001f, 0.95f);
            minimumCatchmentAreaSquareMeters = clampValue(
                j.value("minimumCatchmentAreaSquareMeters", minimumCatchmentAreaSquareMeters), 1.0f, 1000000000.0f);
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
        // How far above its own bed a backwater may stand, as a multiple of the
        // locally solved normal depth. This replaces the flat maximumDepthMeters
        // ceiling, which let a lake surface ride 12 m over the valley floor no
        // matter how small the river was.
        float backwaterDepthRatio = 2.5f;
        float surfaceOffsetMeters = 0.03f;
        float bankFreeboardRatio = 0.35f;
        float minimumFreeboardMeters = 0.08f;
        float maximumFreeboardMeters = 2.0f;
        float foamPersistenceMeters = 30.0f;

        RiverHydraulicsNode() {
            name = "River Hydraulics";
            terrainNodeType = NodeType::RiverHydraulics;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Bed Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Catchment Area", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                false, 1, NodeSystem::ImageUnit::SquareMeters));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Direction", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Direction,
                false, 0, NodeSystem::ImageUnit::Unitless));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Channels", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Reference Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Discharge", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                1, NodeSystem::ImageUnit::CubicMetersPerSecond));
            for (const char* outputName : {"River Width", "Water Depth"}) {
                outputs.push_back(NodeSystem::Pin::createOutput(
                    outputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                    1, NodeSystem::ImageUnit::Meters));
            }
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Flow Speed", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                1, NodeSystem::ImageUnit::MetersPerSecond));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Water Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Froude", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                1, NodeSystem::ImageUnit::Unitless));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Foam Potential", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "River Hydraulics";
            metadata.category = "Hydrology";
            metadata.description = "Manning discharge, normal depth, velocity and whitewater state";
            metadata.headerColor = IM_COL32(28, 126, 176, 255);
            metadata.iconType = (int)UIWidgets::IconType::Water;
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
            j["backwaterDepthRatio"] = backwaterDepthRatio;
            j["surfaceOffsetMeters"] = surfaceOffsetMeters;
            j["bankFreeboardRatio"] = bankFreeboardRatio;
            j["minimumFreeboardMeters"] = minimumFreeboardMeters;
            j["maximumFreeboardMeters"] = maximumFreeboardMeters;
            j["foamPersistenceMeters"] = foamPersistenceMeters;
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
            backwaterDepthRatio = clampValue(j.value("backwaterDepthRatio", backwaterDepthRatio), 1.0f, 50.0f);
            surfaceOffsetMeters = clampValue(j.value("surfaceOffsetMeters", surfaceOffsetMeters), 0.0f, 10.0f);
            bankFreeboardRatio = clampValue(j.value("bankFreeboardRatio", bankFreeboardRatio), 0.0f, 5.0f);
            minimumFreeboardMeters = clampValue(j.value("minimumFreeboardMeters", minimumFreeboardMeters), 0.0f, 20.0f);
            maximumFreeboardMeters = clampValue(
                j.value("maximumFreeboardMeters", maximumFreeboardMeters), minimumFreeboardMeters, 100.0f);
            foamPersistenceMeters = clampValue(
                j.value("foamPersistenceMeters", foamPersistenceMeters), 0.0f, 500.0f);
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
        /// Cross-section shape is a RESULT, not a dial.
        ///
        /// A young fluvial valley is V-shaped because water cuts a line; a
        /// glaciated one is U-shaped because ice cuts a plane. Exposing "pick
        /// V or U" as a free choice would make the panel lie about the cause,
        /// which is this repository's most expensive failure class. So the
        /// floor widens only where the Glacial input says ice worked, and the
        /// unwired case reproduces the old profile exactly.
        float glacialFloorFraction = 0.55f;
        float glacialWidening = 0.8f;

        RiverBedCarveNode() {
            name = "River Bed Carve";
            terrainNodeType = NodeType::RiverBedCarve;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Channels", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "River Width", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                true, 1, NodeSystem::ImageUnit::Meters));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Water Depth", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                true, 1, NodeSystem::ImageUnit::Meters));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Reference Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Water Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            // Appended for pin-index serialization stability. The drainage
            // graph may cross a lake, but visible bed carving must stop there.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Appended. Glacier Flow already publishes where ice worked and
            // nothing read it; this is the pin that turns that into form.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Glacial", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Appended. Commits the watershed's breach cut to the terrain. It
            // is deliberately NOT gated on channel strength or on the lake
            // mask: the sill a breach removes is often below the accumulation
            // threshold, and skipping it there is what leaves the ridge
            // standing while the drainage solve believes it is gone.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Breach Depth", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::PhysicalScalar, true, 1, NodeSystem::ImageUnit::Meters));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Carved Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "River Bed", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "River Bed Carve";
            metadata.category = "Hydrology";
            metadata.description = "Area-scaled, non-destructive channel and bank carving";
            metadata.headerColor = IM_COL32(38, 139, 184, 255);
            metadata.iconType = (int)UIWidgets::IconType::Sculpt;
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
            j["glacialFloorFraction"] = glacialFloorFraction;
            j["glacialWidening"] = glacialWidening;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            minimumWidth = clampValue(j.value("minimumWidth", minimumWidth), 0.1f, 100.0f);
            maximumWidth = clampValue(j.value("maximumWidth", maximumWidth), minimumWidth, 500.0f);
            minimumDepth = clampValue(j.value("minimumDepth", minimumDepth), 0.0f, 50.0f);
            maximumDepth = clampValue(j.value("maximumDepth", maximumDepth), minimumDepth, 200.0f);
            bankSoftness = clampValue(j.value("bankSoftness", bankSoftness), 0.05f, 1.0f);
            glacialFloorFraction = clampValue(j.value("glacialFloorFraction", glacialFloorFraction), 0.0f, 0.9f);
            glacialWidening = clampValue(j.value("glacialWidening", glacialWidening), 0.0f, 3.0f);
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
                "Accumulation", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                false, 1, NodeSystem::ImageUnit::Unknown));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Direction", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Direction,
                false, 0, NodeSystem::ImageUnit::Unitless));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Channels", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Lake Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "River Width", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                true, 1, NodeSystem::ImageUnit::Meters));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Water Depth", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                true, 1, NodeSystem::ImageUnit::Meters));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow Speed", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                true, 1, NodeSystem::ImageUnit::MetersPerSecond));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Discharge", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                true, 1, NodeSystem::ImageUnit::CubicMetersPerSecond));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Froude", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PhysicalScalar,
                true, 1, NodeSystem::ImageUnit::Unitless));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Foam Potential", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "River Water Level", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height, true));
            metadata.displayName = "River Spline Output";
            metadata.category = "Output";
            metadata.description = "Creates owned RiverSpline branches from a river network";
            metadata.headerColor = IM_COL32(35, 167, 214, 255);
            metadata.iconType = (int)UIWidgets::IconType::Gizmo;
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

    enum class RiverLakePreset {
        TemperateValleys = 0,
        AlpineSnowmelt,
        AridCanyons,
        TropicalDrainage,
        BroadLowlands
    };

    // Pass-through controller for the detailed hydrology layer. It changes no
    // height data itself: presets coherently drive the sibling expert nodes,
    // which remain available by expanding the containing graph layer.
    class RiverLakeEasyNode : public TerrainNodeBase {
    public:
        RiverLakePreset preset = RiverLakePreset::TemperateValleys;
        float waterAmount = 0.50f;
        bool generateWaterMeshes = true;
        bool expertOverrides = false;
        uint32_t watershedNodeId = 0;
        uint32_t lakeBasinNodeId = 0;
        uint32_t lakeOutputNodeId = 0;
        uint32_t networkNodeId = 0;
        uint32_t hydraulicsNodeId = 0;
        uint32_t carveNodeId = 0;
        uint32_t splineOutputNodeId = 0;

        RiverLakeEasyNode() {
            name = "River & Lake (Easy)";
            terrainNodeType = NodeType::RiverLakeEasy;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Precipitation / Snowmelt", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Precipitation", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "River & Lake (Easy)";
            metadata.category = "Hydrology";
            metadata.description = "One-control presets for the detailed physical river/lake layer";
            metadata.headerColor = IM_COL32(30, 151, 205, 255);
            metadata.iconType = (int)UIWidgets::IconType::Water;
            headerColor = ImVec4(0.12f, 0.59f, 0.80f, 1.0f);
        }

        static const char* getPresetName(RiverLakePreset value);
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.RiverLakeEasy"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["preset"] = static_cast<int>(preset);
            j["waterAmount"] = waterAmount;
            j["generateWaterMeshes"] = generateWaterMeshes;
            j["expertOverrides"] = expertOverrides;
            j["watershedNodeId"] = watershedNodeId; j["lakeBasinNodeId"] = lakeBasinNodeId;
            j["lakeOutputNodeId"] = lakeOutputNodeId; j["networkNodeId"] = networkNodeId;
            j["hydraulicsNodeId"] = hydraulicsNodeId; j["carveNodeId"] = carveNodeId;
            j["splineOutputNodeId"] = splineOutputNodeId;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            preset = static_cast<RiverLakePreset>(clampValue(j.value("preset", 0), 0, 4));
            waterAmount = clampValue(j.value("waterAmount", waterAmount), 0.0f, 1.0f);
            generateWaterMeshes = j.value("generateWaterMeshes", generateWaterMeshes);
            expertOverrides = j.value("expertOverrides", expertOverrides);
            watershedNodeId = j.value("watershedNodeId", 0u); lakeBasinNodeId = j.value("lakeBasinNodeId", 0u);
            lakeOutputNodeId = j.value("lakeOutputNodeId", 0u); networkNodeId = j.value("networkNodeId", 0u);
            hydraulicsNodeId = j.value("hydraulicsNodeId", 0u); carveNodeId = j.value("carveNodeId", 0u);
            splineOutputNodeId = j.value("splineOutputNodeId", 0u);
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
                                          "River Water Level", "River Froude", "River Foam Potential",
                                          "Erosion Wear", "Erosion Deposits",
                                          "Geology Hardness", "Geology Permeability",
                                          "Geology Fracture", "Geology ID"}) {
                inputs.push_back(NodeSystem::Pin::createInput(
                    inputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            }
            const auto setFieldContract = [&](size_t index, NodeSystem::ImageSemantic semantic,
                                              NodeSystem::ImageUnit unit = NodeSystem::ImageUnit::Unknown) {
                inputs[index].imageSemantic = semantic;
                inputs[index].imageUnit = unit;
                inputs[index].updateVisualCache();
            };
            setFieldContract(9, NodeSystem::ImageSemantic::PhysicalScalar, NodeSystem::ImageUnit::Unitless);
            setFieldContract(10, NodeSystem::ImageSemantic::Direction, NodeSystem::ImageUnit::Unitless);
            setFieldContract(11, NodeSystem::ImageSemantic::Categorical, NodeSystem::ImageUnit::Identifier);
            setFieldContract(13, NodeSystem::ImageSemantic::Categorical, NodeSystem::ImageUnit::Identifier);
            setFieldContract(17, NodeSystem::ImageSemantic::Height);
            setFieldContract(18, NodeSystem::ImageSemantic::Height);
            setFieldContract(21, NodeSystem::ImageSemantic::Categorical, NodeSystem::ImageUnit::Identifier);
            setFieldContract(22, NodeSystem::ImageSemantic::PhysicalScalar, NodeSystem::ImageUnit::SquareMeters);
            setFieldContract(23, NodeSystem::ImageSemantic::PhysicalScalar,
                             NodeSystem::ImageUnit::CubicMetersPerSecond);
            setFieldContract(24, NodeSystem::ImageSemantic::PhysicalScalar, NodeSystem::ImageUnit::Meters);
            setFieldContract(25, NodeSystem::ImageSemantic::PhysicalScalar, NodeSystem::ImageUnit::Meters);
            setFieldContract(26, NodeSystem::ImageSemantic::PhysicalScalar,
                             NodeSystem::ImageUnit::MetersPerSecond);
            setFieldContract(27, NodeSystem::ImageSemantic::Height);
            setFieldContract(28, NodeSystem::ImageSemantic::PhysicalScalar, NodeSystem::ImageUnit::Unitless);
            metadata.displayName = "Terrain Fields Output";
            metadata.category = "Output";
            metadata.description = "Publishes terrain, biome and hydrology fields for downstream systems";
            metadata.headerColor = IM_COL32(55, 170, 135, 255);
            metadata.iconType = (int)UIWidgets::IconType::Console;
            headerColor = ImVec4(0.22f, 0.67f, 0.53f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        std::string getTypeId() const override { return "TerrainV2.TerrainFieldsOutput"; }
    };

    // Publishes ONE authored field under a name of your choosing.
    //
    // Terrain Fields Output names measurements: its 36 pins are fixed because
    // each one IS a specific quantity the solver produced. A composed mask is a
    // different thing -- "not water and not steep" is an authoring decision, not
    // a measurement -- so it needs a name the author picks, and there was no way
    // to give it one. Consumers that read fields by name (scatter density and
    // exclusion masks, the foliage layer pickers) could therefore only ever see
    // the canonical 36.
    //
    // This is the node that makes combining masks worthwhile: build the
    // combination out of Math nodes, publish it here as e.g. "mask.no_water",
    // and point a single mask slot at it. The alternative -- N mask slots on the
    // consumer -- would have to carry its own combine semantics (and/or/weights/
    // per-slot thresholds), which is a small expression language hidden inside a
    // panel, where nothing can inspect it.
    class PublishFieldNode : public TerrainNodeBase {
    public:
        // Author-typed. Normalised on publish (see effectiveFieldName): trimmed,
        // lowercased, invalid characters to '_', and given the "mask." namespace
        // when no namespace was typed.
        std::string fieldName = "mask.custom";

        // Last evaluation's outcome, shown in the panel. A publisher that
        // silently does nothing is the failure this node would otherwise invite:
        // a mask slot pointed at a name nobody wrote reads as "no mask", which
        // looks exactly like a working scatter with a permissive mask.
        std::string lastPublishedName;
        bool lastPublishSucceeded = false;
        std::string lastPublishError;

        PublishFieldNode() {
            name = "Publish Field";
            terrainNodeType = NodeType::PublishField;
            // Generic, not Mask: this node makes NO claim about the quantity it
            // publishes -- the author names it, and the name is the only contract.
            // A Mask-typed input refused every non-mask field (height, slope in
            // degrees, discharge), which are exactly the compositions worth
            // naming. Measured live 2026-08-30: a Noise Generator output was
            // rejected as a "type/semantic mismatch".
            inputs.push_back(NodeSystem::Pin::createInput(
                "Field", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Generic, true));
            metadata.displayName = "Publish Field";
            metadata.category = "Output";
            metadata.description = "Publishes one field under an author-chosen name for mask consumers";
            metadata.headerColor = IM_COL32(55, 170, 135, 255);
            metadata.iconType = (int)UIWidgets::IconType::Console;
            headerColor = ImVec4(0.22f, 0.67f, 0.53f, 1.0f);
        }

        // The name this node will actually write, or empty when the authored
        // text cannot produce a legal one. Kept as a function rather than a
        // cached member so the panel and compute() can never disagree.
        std::string effectiveFieldName() const;
        // Canonical measurement names owned by Terrain Fields Output. Shadowing
        // one would make every downstream reader of that name silently receive
        // something else, so publishing to it is refused rather than allowed to
        // win by evaluation order.
        static bool isReservedFieldName(const std::string& name);

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.PublishField"; }
        float getCustomWidth() const override { return 210.0f; }
        void serializeToJson(nlohmann::json& j) const override;
        void deserializeFromJson(const nlohmann::json& j) override;
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
                // Wetness is routinely driven by hydraulic discharge, an SI
                // field. Accept it here and normalize at getMaskInput() rather
                // than letting the link be silently refused.
                inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
            }
            for (const char* outputName : {"Forest", "Grass", "Rock", "Alpine"}) {
                outputs.push_back(NodeSystem::Pin::createOutput(
                    outputName, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            }
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Biome Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PackedData, 4));
            metadata.displayName = "Biome Composer";
            metadata.category = "Data Maps";
            metadata.headerColor = IM_COL32(74, 154, 92, 255);
            metadata.iconType = (int)UIWidgets::IconType::World;
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
            // Random world-Y offset band applied per instance (meters). Mirrors
            // the Terrain UI source "Y-Off" control on the shared InstanceGroup.
            float yOffsetMin = 0.0f;
            float yOffsetMax = 0.0f;
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
        // Wide-open by default (mirrors InstanceGroup): the biome masks pick
        // altitude; a fixed +/-10 m absolute band broke on 1000 m terrains.
        float minimumHeight = -100000.0f;
        float maximumHeight = 100000.0f;
        // Terrain-border keep-out in meters; <0 = auto (two heightmap cells),
        // 0 = off. Mirrors InstanceGroup::BrushSettings::edge_margin.
        float edgeMargin = -1.0f;
        std::string densityField;
        // New foliage layers automatically respect the canonical road keep-out
        // field. If a terrain does not publish it, scatter's missing-field
        // fallback preserves the previous no-exclusion behavior.
        std::string exclusionField = "infrastructure.foliage_exclusion";
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
            metadata.iconType = (int)UIWidgets::IconType::HairCombTool;
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
            metadata.iconType = (int)UIWidgets::IconType::HairClumpTool;
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
            metadata.iconType = (int)UIWidgets::IconType::HairAddTool;
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
            metadata.iconType = (int)UIWidgets::IconType::EyedropperTool;
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
        /// Discharge that has slowed down leaves its load behind, and discharge
        /// that has not scours the bed. One accumulation field, two opposite
        /// effects separated by slope - this is what makes valley floors thick
        /// and channel beds thin without an erosion sim in the graph.
        float transportInfluence = 0.45f;
        float channelScour = 0.35f;
        /// How much bedrock hardness suppresses soil PRODUCTION. Left at 1.0,
        /// which is what the term did before it was a dial - it exists because
        /// Strata hardness is banded by elevation by construction, so a graph
        /// that wants soil to stop reading as a contour map has somewhere to
        /// turn. Transported soil in valleys keeps no memory of the rock
        /// beneath it either way.
        float hardnessInfluence = 1.0f;

        SoilDepthNode() {
            name = "Soil Depth";
            terrainNodeType = NodeType::SoilDepth;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            // Renamed from "Flow". The pin is consumed as DEPOSITION -
            // `soilDepth += depositionInfluence * value * flatness`, i.e. it
            // BUILDS soil. Feeding flow accumulation here made channels the
            // thickest soil on the terrain, which is backwards: channels are
            // scoured, not filled. Hydraulic Erosion's Deposition output is
            // the correct source; with no erosion sim there is no deposition
            // field and the pin belongs empty.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Deposition", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Erosion publishes deposition/discharge as SI fields; accept them
            // and normalize once at the input boundary.
            // Appended, so existing serialized graphs keep their pin indices.
            // Deposition is the erosion sim's answer; Flow is the accumulation
            // field every graph already has. Flow is NOT deposition - it is
            // split into a deposition and a scour term by local slope, which
            // is why it needs its own pin rather than being wired into the one
            // above.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Flow", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs[1].acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
            inputs[2].acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
            inputs[3].acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Soil Depth", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Soil Depth";
            metadata.category = "Data Map";
            metadata.headerColor = IM_COL32(135, 105, 65, 255);
            metadata.iconType = (int)UIWidgets::IconType::ClayStripsTool;
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
            j["transportInfluence"] = transportInfluence;
            j["channelScour"] = channelScour;
            j["hardnessInfluence"] = hardnessInfluence;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            production = clampValue(j.value("production", production), 0.0f, 2.0f);
            depositionInfluence = clampValue(j.value("depositionInfluence", depositionInfluence), 0.0f, 2.0f);
            concavityInfluence = clampValue(j.value("concavityInfluence", concavityInfluence), 0.0f, 2.0f);
            slopeLoss = clampValue(j.value("slopeLoss", slopeLoss), 0.0f, 2.0f);
            transportInfluence = clampValue(j.value("transportInfluence", transportInfluence), 0.0f, 2.0f);
            channelScour = clampValue(j.value("channelScour", channelScour), 0.0f, 2.0f);
            hardnessInfluence = clampValue(j.value("hardnessInfluence", hardnessInfluence), 0.0f, 1.0f);
        }
    };

    /**
     * @brief Volcanic cones, craters and collapsed calderas.
     *
     * The one macro landform the generator could not make: every other node
     * here builds ranges, folds and basins, all of which are elongated or
     * tectonic. A volcano is radial, and nothing radial could be authored
     * except by hand.
     *
     * The profile is built from the real thing rather than a bump: a cone,
     * an excavated bowl with a flat floor, a raised rim at the crater edge,
     * and an ejecta blanket outside it decaying with distance. A caldera is
     * the same shape with the summit collapsed - the ratio is a dial, not a
     * second node, because a caldera IS a crater whose roof fell in.
     */
    class CraterCalderaNode : public TerrainNodeBase {
    public:
        int count = 1;
        int seed = 1337;
        /// Metres. Radius is the crater rim crest, not the cone base.
        float craterRadius = 300.0f;
        float radiusVariation = 0.35f;
        float craterDepth = 120.0f;
        float rimHeight = 45.0f;
        float coneHeight = 400.0f;
        /// Cone base radius as a multiple of the crater radius. 1.0 gives a
        /// bare crater in flat ground; larger values grow a volcano under it.
        float coneRadiusScale = 4.0f;
        /// Fraction of the crater radius that is flat floor.
        float floorFraction = 0.35f;
        /// 0 = simple crater. Above 0 the summit collapses into a caldera of
        /// this fraction of the crater radius, with a terrace at its edge.
        float calderaRatio = 0.0f;
        /// How fast the ejecta blanket decays outside the rim.
        float ejectaFalloff = 2.6f;
        float ejectaStrength = 0.35f;

        CraterCalderaNode() {
            name = "Crater / Caldera";
            terrainNodeType = NodeType::CraterCaldera;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            // Where craters are ALLOWED, not where they are placed: placement
            // stays deterministic from the seed so the same graph rebuilds the
            // same mountain.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Placement Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Crater", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Rim", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            // Fresh volcanic rock is hard and unweathered, so the chain that
            // already consumes hardness can consume this too.
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Hardness", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Crater / Caldera";
            metadata.category = "Landform";
            metadata.headerColor = IM_COL32(170, 85, 70, 255);
            metadata.iconType = (int)UIWidgets::IconType::BlobTool;
            headerColor = ImVec4(0.67f, 0.33f, 0.27f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.CraterCaldera"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["count"] = count; j["seed"] = seed;
            j["craterRadius"] = craterRadius; j["radiusVariation"] = radiusVariation;
            j["craterDepth"] = craterDepth; j["rimHeight"] = rimHeight;
            j["coneHeight"] = coneHeight; j["coneRadiusScale"] = coneRadiusScale;
            j["floorFraction"] = floorFraction; j["calderaRatio"] = calderaRatio;
            j["ejectaFalloff"] = ejectaFalloff; j["ejectaStrength"] = ejectaStrength;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            count = clampValue(j.value("count", count), 1, 64);
            seed = j.value("seed", seed);
            craterRadius = clampValue(j.value("craterRadius", craterRadius), 1.0f, 100000.0f);
            radiusVariation = clampValue(j.value("radiusVariation", radiusVariation), 0.0f, 0.95f);
            craterDepth = clampValue(j.value("craterDepth", craterDepth), 0.0f, 20000.0f);
            rimHeight = clampValue(j.value("rimHeight", rimHeight), 0.0f, 20000.0f);
            coneHeight = clampValue(j.value("coneHeight", coneHeight), 0.0f, 20000.0f);
            coneRadiusScale = clampValue(j.value("coneRadiusScale", coneRadiusScale), 1.0f, 20.0f);
            floorFraction = clampValue(j.value("floorFraction", floorFraction), 0.0f, 0.9f);
            calderaRatio = clampValue(j.value("calderaRatio", calderaRatio), 0.0f, 0.95f);
            ejectaFalloff = clampValue(j.value("ejectaFalloff", ejectaFalloff), 0.5f, 8.0f);
            ejectaStrength = clampValue(j.value("ejectaStrength", ejectaStrength), 0.0f, 2.0f);
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
        float basePermeability = 0.35f;
        float permeabilityContrast = 0.55f;
        float fractureDensity = 0.30f;
        float fractureScaleMeters = 18.0f;
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
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Permeability", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Fracture", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Lithology";
            metadata.category = "Geology";
            metadata.headerColor = IM_COL32(150, 100, 72, 255);
            metadata.iconType = (int)UIWidgets::IconType::ViewMatcap;
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
            j["basePermeability"] = basePermeability;
            j["permeabilityContrast"] = permeabilityContrast;
            j["fractureDensity"] = fractureDensity;
            j["fractureScaleMeters"] = fractureScaleMeters;
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
            basePermeability = clampValue(j.value("basePermeability", basePermeability), 0.0f, 1.0f);
            permeabilityContrast = clampValue(j.value("permeabilityContrast", permeabilityContrast), 0.0f, 1.0f);
            fractureDensity = clampValue(j.value("fractureDensity", fractureDensity), 0.0f, 1.0f);
            fractureScaleMeters = (std::max)(j.value("fractureScaleMeters", fractureScaleMeters), 0.1f);
            seed = j.value("seed", seed);
        }
    };

    class PlateTectonicsNode : public TerrainNodeBase {
    public:
        int plateCount = 7;
        float plateScaleMeters = 900.0f;
        float upliftMeters = 18.0f;
        float riftDepthMeters = 6.0f;
        float boundaryWidthMeters = 90.0f;
        float interiorWarp = 0.12f;
        int seed = 271;

        PlateTectonicsNode() {
            name = "Plate Tectonics";
            terrainNodeType = NodeType::PlateTectonics;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Macro", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Ridge", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Valley", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Uplift", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Plate Boundary", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Crust ID", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Plate Tectonics";
            metadata.category = "Geology";
            metadata.headerColor = IM_COL32(135, 82, 62, 255);
            metadata.iconType = (int)UIWidgets::IconType::GrabTool;
            headerColor = ImVec4(0.53f, 0.32f, 0.24f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.PlateTectonics"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["plateCount"] = plateCount; j["plateScaleMeters"] = plateScaleMeters;
            j["upliftMeters"] = upliftMeters; j["riftDepthMeters"] = riftDepthMeters;
            j["boundaryWidthMeters"] = boundaryWidthMeters; j["interiorWarp"] = interiorWarp;
            j["seed"] = seed;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            plateCount = clampValue(j.value("plateCount", plateCount), 2, 24);
            plateScaleMeters = (std::max)(j.value("plateScaleMeters", plateScaleMeters), 10.0f);
            upliftMeters = clampValue(j.value("upliftMeters", upliftMeters), 0.0f, 5000.0f);
            riftDepthMeters = clampValue(j.value("riftDepthMeters", riftDepthMeters), 0.0f, 5000.0f);
            boundaryWidthMeters = (std::max)(j.value("boundaryWidthMeters", boundaryWidthMeters), 1.0f);
            interiorWarp = clampValue(j.value("interiorWarp", interiorWarp), 0.0f, 1.0f);
            seed = j.value("seed", seed);
        }
    };

    class FoldNode : public TerrainNodeBase {
    public:
        float wavelengthMeters = 180.0f;
        float amplitudeMeters = 6.0f;
        float directionDegrees = 35.0f;
        float asymmetry = 0.25f;
        float pinch = 0.35f;
        float fractureStrength = 0.55f;
        int seed = 419;

        FoldNode() {
            name = "Fold";
            terrainNodeType = NodeType::Fold;
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Compression", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Fold", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Fracture", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Fold / Compression";
            metadata.category = "Geology";
            metadata.headerColor = IM_COL32(155, 92, 66, 255);
            metadata.iconType = (int)UIWidgets::IconType::SnakeHookTool;
            headerColor = ImVec4(0.61f, 0.36f, 0.26f, 1.0f);
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.Fold"; }
        void serializeToJson(nlohmann::json& j) const override {
            TerrainNodeBase::serializeToJson(j);
            j["wavelengthMeters"] = wavelengthMeters; j["amplitudeMeters"] = amplitudeMeters;
            j["directionDegrees"] = directionDegrees; j["asymmetry"] = asymmetry;
            j["pinch"] = pinch; j["fractureStrength"] = fractureStrength; j["seed"] = seed;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            wavelengthMeters = (std::max)(j.value("wavelengthMeters", wavelengthMeters), 1.0f);
            amplitudeMeters = clampValue(j.value("amplitudeMeters", amplitudeMeters), 0.0f, 5000.0f);
            directionDegrees = j.value("directionDegrees", directionDegrees);
            asymmetry = clampValue(j.value("asymmetry", asymmetry), -0.95f, 0.95f);
            pinch = clampValue(j.value("pinch", pinch), 0.0f, 1.0f);
            fractureStrength = clampValue(j.value("fractureStrength", fractureStrength), 0.0f, 1.0f);
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
        float boundaryFracture = 0.65f;
        float permeabilityBoost = 0.35f;

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
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Permeability", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Fracture", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            metadata.displayName = "Strata";
            metadata.category = "Geology";
            metadata.headerColor = IM_COL32(165, 110, 70, 255);
            metadata.iconType = (int)UIWidgets::IconType::FaceMode;
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
            j["boundaryFracture"] = boundaryFracture;
            j["permeabilityBoost"] = permeabilityBoost;
        }
        void deserializeFromJson(const nlohmann::json& j) override {
            TerrainNodeBase::deserializeFromJson(j);
            layerThickness = clampValue(j.value("layerThickness", layerThickness), 0.05f, 1000.0f);
            dipDegrees = clampValue(j.value("dipDegrees", dipDegrees), -75.0f, 75.0f);
            dipAzimuth = j.value("dipAzimuth", dipAzimuth);
            reliefStrength = clampValue(j.value("reliefStrength", reliefStrength), 0.0f, 0.25f);
            edgeSharpness = clampValue(j.value("edgeSharpness", edgeSharpness), 0.25f, 8.0f);
            boundaryFracture = clampValue(j.value("boundaryFracture", boundaryFracture), 0.0f, 1.0f);
            permeabilityBoost = clampValue(j.value("permeabilityBoost", permeabilityBoost), 0.0f, 1.0f);
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
        // Renamed from normalizeOutput: Splat Output normalized regardless, so
        // the old name described an operation this node never controlled. The
        // dial now names what it actually does, and old files are not silently
        // reinterpreted because the key changed with the meaning.
        bool soilFillsRemainder = true;

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
            // Flow and climate Meltwater are merged into semantic Flow (R).
            inputs.push_back(NodeSystem::Pin::createInput(
                "Meltwater", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Explicit authored layers are appended so old serialized pin indices
            // remain stable. When either pin is unconnected, the legacy
            // height/slope/soil synthesis remains available as a fallback.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Grass / Base", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Rock / Slope", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Appended: without a Slope pin this node could not consume the
            // shared Terrain Analysis solve, so it re-derived its own slope on
            // a different curve and disagreed with every other splat author.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Slope", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Appended: Thermal Erosion has published a Talus mask all along
            // and nothing consumed it. Debris cones at a cliff base ARE rock -
            // loose, angular rock - so talus joins the rock claim and clears
            // the grass off itself rather than needing a layer of its own.
            inputs.push_back(NodeSystem::Pin::createInput(
                "Talus", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            // Hydrology publishes discharge, drainage area and water depth as
            // SI fields. They are the correct sources for these pins; the
            // normalization into 0-1 happens once, at getMaskInput().
            for (size_t index = 1; index < inputs.size(); ++index) {
                inputs[index].acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
            }
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Surface Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::PackedData, 4));
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Semantic (Flow/Wet/Ice/Hard)", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::PackedData, 4));
            metadata.displayName = "Surface Composer";
            metadata.category = "Texture";
            metadata.headerColor = IM_COL32(190, 135, 52, 255);
            metadata.iconType = (int)UIWidgets::IconType::ViewMatcap;
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
            j["soilFillsRemainder"] = soilFillsRemainder;
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
            soilFillsRemainder = j.value("soilFillsRemainder", soilFillsRemainder);
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
        bool useGPU = true;
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
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Meltwater Depth", NodeSystem::DataType::Image2D,
                NodeSystem::ImageSemantic::PhysicalScalar, 1, NodeSystem::ImageUnit::Meters));
            metadata.displayName = "Snow Layer";
            metadata.category = "Snow & Ice";
            metadata.headerColor = IM_COL32(105, 180, 220, 255);
            metadata.iconType = (int)UIWidgets::IconType::Wind;
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
            j["windAzimuth"] = windAzimuth; j["useGPU"] = useGPU; j["useSceneSun"] = useSceneSun;
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
            useGPU = j.value("useGPU", useGPU);
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
            metadata.iconType = (int)UIWidgets::IconType::Volumetric;
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
            metadata.iconType = (int)UIWidgets::IconType::Volumetric;
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
            metadata.iconType = (int)UIWidgets::IconType::SmudgeTool;
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
            metadata.iconType = (int)UIWidgets::IconType::BurnTool;
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
            metadata.iconType = (int)UIWidgets::IconType::ElasticDeformTool;
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

        // Explicit Terrain Object resolution change. The requested dimensions
        // seed procedural sources (Noise, etc.) without silently changing the
        // normal graph evaluation contract. File/Terrain sources may retain
        // their authored native resolution.
        bool evaluateTerrainAtResolution(TerrainObject* terrain, struct ::SceneData& scene,
                                         int targetWidth, int targetHeight);

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
        bool lastAsyncEvaluationCancelled() const { return lastAsyncEvaluationCancelled_; }
        const std::string& lastAsyncEvaluationError() const { return lastAsyncEvaluationError_; }
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

        // Observational inspector pull. Reuses clean upstream values from the
        // committed cache, evaluates dirty nodes against a detached terrain and
        // restores authoring dirty state before returning.
        bool evaluateInspectorImageOutput(uint32_t nodeId, int outputIndex,
                                          TerrainObject* terrain,
                                          NodeSystem::Image2DData& image);

        // Create a default graph with basic nodes
        void createDefaultGraph(TerrainObject* terrain);

        // ====================================================================
        // SETUP WIRING DIAGNOSTICS
        // ====================================================================

        /**
         * @brief One link a setup asked for and did not get.
         *
         * addLink() answers a refused connection with 0, and every setup
         * ignored that answer. A pin-semantic mismatch therefore produced a
         * graph that looked built but had holes in it: the composer fell back
         * to synthesized values and the result merely looked plausible, which
         * is the one failure nobody reports as a bug.
         */
        struct SetupWiringFault {
            std::string fromNode;
            std::string fromPin;
            std::string toNode;
            std::string toPin;
            std::string reason;
        };

        /// Faults recorded by the most recent add*Setup call.
        const std::vector<SetupWiringFault>& lastSetupWiringFaults() const {
            return setupWiringFaults_;
        }

        /**
         * @brief addLink that records what it could not connect.
         *
         * Setups must use this rather than addLink so a refused connection is
         * a reported fault instead of a missing wire nobody notices.
         */
        uint32_t addCheckedLink(uint32_t startPinId, uint32_t endPinId);

        // Non-destructive authoring helper: inserts a Snow Climate node before
        // the active Height Output and wires its material outputs to the active
        // Surface Composer/Splat Output chain. Existing grass/rock/flow inputs
        // on the composer are preserved.
        bool addSnowLayerSetup(float x = 520.0f, float y = 120.0f);
        bool removeSnowLayerSetup();

        // Non-destructive biome authoring helper. Reuses an existing analysis,
        // exposure, composer and fields-output chain when present, otherwise
        // creates and wires the missing nodes from the active Height Output source.
        bool addBiomeFieldsSetup(BiomeClimatePreset preset,
                                 float x = 520.0f, float y = 120.0f);
        bool removeBiomeFieldsSetup();

        // Adds four independent biome-driven foliage layers (forest, grass,
        // rock and alpine), grouped under one Foliage Set and output sink.
        bool addBiomeFoliageSetup(float x = 1120.0f, float y = 120.0f);

        // Optional look-development companion for terrain presets. Reuses
        // available analysis/erosion/biome/snow masks and leaves manually
        // authored SatMap input links untouched.
        bool addSatMapSetup(const std::string& preset = "Temperate",
                            float x = 1320.0f, float y = 420.0f);
        bool applySatMapPresetRecipe(const std::string& presetId,
                                     std::string* error = nullptr,
                                     std::vector<std::string>* warnings = nullptr,
                                     float x = 1320.0f, float y = 420.0f);

        // Non-destructive hydrology branch from the active authored height.
        // The Height Output connection is not replaced; the new branch publishes
        // a watershed, vector river network and owned RiverSpline sink.
        bool addRiverNetworkSetup(float x = 520.0f, float y = 520.0f);
        bool removeRiverNetworkSetup(TerrainObject* terrain, struct ::SceneData& scene);

        // Inserts a continuous heightfield geology foundation before the active
        // Height Output and publishes reusable fields for erosion/materials.
        bool addGeologyFoundationSetup(float x = 420.0f, float y = 320.0f);
        bool removeGeologyFoundationSetup();

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
        std::vector<SetupWiringFault> setupWiringFaults_;
        std::future<void> evalFuture_;
        TerrainObject* pendingFinalizeTerrain_ = nullptr;
        struct ::SceneData* pendingFinalizeScene_ = nullptr;
        std::unique_ptr<TerrainContext> activeTerrainCtx_;
        std::atomic<bool> lastEvaluateResized_{false};
        std::atomic<bool> lastFinalizeWasFullRebuild_{false};
        std::atomic<bool> lastHeightDataUpdated_{false};
        bool lastAsyncEvaluationCancelled_ = false;
        std::string lastAsyncEvaluationError_;
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

