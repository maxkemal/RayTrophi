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

namespace TerrainNodesV2 {

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
        
        TerrainContext() = default;
        explicit TerrainContext(TerrainObject* t) : terrain(t) {
            if (t) {
                width = t->heightmap.width;
                height = t->heightmap.height;
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
        SplatOutput
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
        
        TerrainContext* getTerrainContext(NodeSystem::EvaluationContext& ctx) {
            return ctx.getDomainContext<TerrainContext>();
        }
        
        // Helper to get Image2D from input
        NodeSystem::Image2DData getHeightInput(int inputIndex, NodeSystem::EvaluationContext& ctx) {
            NodeSystem::PinValue val = getInputValue(inputIndex, ctx);
            if (auto* img = std::get_if<NodeSystem::Image2DData>(&val)) {
                return *img;
            }
            return NodeSystem::Image2DData{};
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
        }
        
        /**
         * @brief Deserialize node-specific data from JSON
         * Override in derived classes to load custom parameters
         */
        virtual void deserializeFromJson(const nlohmann::json& j) {
            if (j.contains("x")) x = j["x"].get<float>();
            if (j.contains("y")) y = j["y"].get<float>();
            if (j.contains("name")) name = j["name"].get<std::string>();
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
        int smoothIterations = 0; // Smoothing pass count
        
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
            
            // Default: read from terrain
            TerrainObject* terrain = tctx->terrain;
            int w = terrain->heightmap.width;
            int h = terrain->heightmap.height;
            
            auto result = createHeightOutput(w, h);
            
            if (terrain->heightmap.data.size() == (size_t)(w * h)) {
                *result.data = terrain->heightmap.data;
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
            
            // Reload file if path exists
            if (sourceMode == SourceMode::File && strlen(filePath) > 0) {
                loadHeightmapFromFile();
            }
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
        
        HydraulicErosionNode() {
            name = "Hydraulic Erosion";
            terrainNodeType = NodeType::HydraulicErosion;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Hydraulic Erosion";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(75, 125, 200, 255);
            headerColor = ImVec4(0.3f, 0.5f, 0.8f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.HydraulicErosion"; }
    };

    class ThermalErosionNode : public TerrainNodeBase {
    public:
        ThermalErosionParams params;
        bool useGPU = true;
        
        ThermalErosionNode() {
            name = "Thermal Erosion";
            terrainNodeType = NodeType::ThermalErosion;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
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
    };

    class FluvialErosionNode : public TerrainNodeBase {
    public:
        HydraulicErosionParams params;
        bool useGPU = true;
        
        FluvialErosionNode() {
            name = "Fluvial Erosion";
            terrainNodeType = NodeType::FluvialErosion;
            
            // Fluvial erosion uses lower sediment capacity than hydraulic
            params.sedimentCapacity = 1.0f;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "Height In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            inputs.push_back(NodeSystem::Pin::createInput(
                "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Height Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Fluvial Erosion";
            metadata.category = "Erosion";
            metadata.headerColor = IM_COL32(75, 150, 230, 255);
            headerColor = ImVec4(0.3f, 0.6f, 0.9f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.FluvialErosion"; }
    };

    class WindErosionNode : public TerrainNodeBase {
    public:
        float strength = 0.5f;
        float direction = 45.0f;
        int iterations = 20;
        bool useGPU = true;
        
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
        float rainfallFactor = 1.0f;      // Affects hydraulic erosion
        float temperatureFactor = 1.0f;   // Affects thermal erosion
        float windFactor = 1.0f;          // Affects wind erosion
        
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
    };

    class SplatOutputNode : public TerrainNodeBase {
    public:
        char exportPath[256] = "";
        bool browseForExport = false;
        bool autoApplyToTerrain = true;
        
        SplatOutputNode() {
            name = "Splat Output";
            terrainNodeType = NodeType::SplatOutput;
            
            // Accept 4-channel splat data or height for auto-conversion
            inputs.push_back(NodeSystem::Pin::createInput(
                "Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Splat Output";
            metadata.category = "Output";
            metadata.headerColor = IM_COL32(200, 75, 125, 255);
            headerColor = ImVec4(0.8f, 0.3f, 0.5f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        void exportSplatMap(TerrainObject* terrain);
        std::string getTypeId() const override { return "TerrainV2.SplatOutput"; }
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
    };

    class InvertNode : public TerrainNodeBase {
    public:
        InvertNode() {
            name = "Invert";
            terrainNodeType = NodeType::Invert;
            
            inputs.push_back(NodeSystem::Pin::createInput(
                "In", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            outputs.push_back(NodeSystem::Pin::createOutput(
                "Out", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
            
            metadata.displayName = "Invert";
            metadata.category = "Math";
            metadata.headerColor = IM_COL32(100, 100, 100, 255);
            headerColor = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        std::string getTypeId() const override { return "TerrainV2.Invert"; }
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
    };
    
    /**
     * @brief Flow Mask - Simulates where soil/sediment would accumulate
     * 
     * Uses flow accumulation algorithm to find valleys and depressions.
     */
    class FlowMaskNode : public TerrainNodeBase {
    public:
        int iterations = 16;          // Flow simulation iterations (increased for more detail)
        float strength = 1.0f;        // Flow strength multiplier
        float decay = 0.8f;           // Flow decay per step (lowered for longer flow paths)
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
    };

    // ============================================================================
    // PROCEDURAL TEXTURE NODES (Gaea-style)
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
                "Splat", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            
            metadata.displayName = "Auto Splat";
            metadata.category = "Texture";
            metadata.headerColor = IM_COL32(200, 150, 50, 255);
            headerColor = ImVec4(0.8f, 0.6f, 0.2f, 1.0f);
        }
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        std::string getTypeId() const override { return "TerrainV2.AutoSplat"; }
        float getCustomWidth() const override { return 160.0f; }
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
    };

    // ============================================================================
    // TERRAIN GRAPH WRAPPER
    // ============================================================================
    
    /**
     * @brief Terrain-specialized graph using V2 system
     */
    class TerrainNodeGraphV2 : public NodeSystem::GraphBase {
    public:
        TerrainNodeGraphV2() = default;
        
        // Factory method for creating nodes by type
        NodeSystem::NodeBase* addTerrainNode(NodeType type, float x = 0, float y = 0);
        
        // Evaluate with terrain context
        void evaluateTerrain(TerrainObject* terrain, struct ::SceneData& scene);
        
        // Create a default graph with basic nodes
        void createDefaultGraph(TerrainObject* terrain);
        
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
    };

} // namespace TerrainNodesV2
