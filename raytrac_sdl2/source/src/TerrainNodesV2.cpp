/**
 * @file TerrainNodesV2.cpp
 * @brief Implementation of terrain nodes using V2 NodeSystem
 */

#include "TerrainNodesV2.h"
#include "scene_data.h"
#include "TerrainManager.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cmath>
#include <algorithm>
#include <cfloat>  // FLT_MAX
#include <fstream>
#include <string>
#include <cstring>

namespace TerrainNodesV2 {

    // C++14 compatible clamp helper (std::clamp requires C++17)
    template<typename T>
    inline T clampValue(T val, T lo, T hi) {
        return (val < lo) ? lo : ((val > hi) ? hi : val);
    }

    // ============================================================================
    // HEIGHTMAP FILE LOADING
    // ============================================================================
    
    void HeightmapInputNode::loadHeightmapFromFile() {
        if (strlen(filePath) == 0) return;
        
        std::string path(filePath);
        std::string ext = path.substr(path.find_last_of('.') + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        loadedHeightData.clear();
        fileLoaded = false;
        
        // Try RAW16 format first (common for heightmaps)
        if (ext == "raw" || ext == "r16") {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file) return;
            
            size_t fileSize = file.tellg();
            file.seekg(0);
            
            // Assume square heightmap, 16-bit
            size_t pixelCount = fileSize / 2;
            int side = (int)std::sqrt((double)pixelCount);
            
            if (side * side * 2 != fileSize) {
                // Try common sizes
                if (fileSize == 1025 * 1025 * 2) side = 1025;
                else if (fileSize == 2049 * 2049 * 2) side = 2049;
                else if (fileSize == 4097 * 4097 * 2) side = 4097;
                else if (fileSize == 513 * 513 * 2) side = 513;
                else if (fileSize == 257 * 257 * 2) side = 257;
            }
            
            loadedWidth = side;
            loadedHeight = side;
            loadedHeightData.resize(side * side);
            
            std::vector<uint16_t> rawData(side * side);
            file.read(reinterpret_cast<char*>(rawData.data()), side * side * 2);
            
            // Convert to normalized float (0-1 range scaled to typical height)
            for (size_t i = 0; i < rawData.size(); i++) {
                loadedHeightData[i] = (float)rawData[i] / 65535.0f; // Normalize 0-1
            }
            
            fileLoaded = true;
            dirty = true;
            return;
        }
        
        // PNG/JPG/BMP via stb_image
        if (ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "bmp" || ext == "tga") {
            int w, h, channels;
            unsigned char* img = stbi_load(path.c_str(), &w, &h, &channels, 1); // Force grayscale
            
            if (img) {
                // Calculate separate strides for X and Y to preserve aspect ratio
                int strideX = 1, strideY = 1;
                if (w > maxResolution) {
                    strideX = (w + maxResolution - 1) / maxResolution;  // Ceiling division
                }
                if (h > maxResolution) {
                    strideY = (h + maxResolution - 1) / maxResolution;  // Ceiling division
                }
                
                loadedWidth = w / strideX;
                loadedHeight = h / strideY;
                rawHeightData.resize(loadedWidth * loadedHeight);
                
                for (int y = 0; y < loadedHeight; y++) {
                    for (int x = 0; x < loadedWidth; x++) {
                         int srcIdx = (y * strideY) * w + (x * strideX);
                         rawHeightData[y * loadedWidth + x] = (float)img[srcIdx] / 255.0f; 
                    }
                }
                
                stbi_image_free(img);
                applySmoothing();
                fileLoaded = true;
                dirty = true;
            }
            return;
        }
        
        // Try 16-bit PNG loading
        if (ext == "png") {
            int w, h, channels;
            unsigned short* img16 = stbi_load_16(path.c_str(), &w, &h, &channels, 1);
            
            if (img16) {
                // Calculate separate strides for X and Y to preserve aspect ratio
                int strideX = 1, strideY = 1;
                if (w > maxResolution) {
                    strideX = (w + maxResolution - 1) / maxResolution;
                }
                if (h > maxResolution) {
                    strideY = (h + maxResolution - 1) / maxResolution;
                }
                
                loadedWidth = w / strideX;
                loadedHeight = h / strideY;
                rawHeightData.resize(loadedWidth * loadedHeight);
                
                for (int y = 0; y < loadedHeight; y++) {
                    for (int x = 0; x < loadedWidth; x++) {
                         int srcIdx = (y * strideY) * w + (x * strideX);
                         rawHeightData[y * loadedWidth + x] = (float)img16[srcIdx] / 65535.0f;
                    }
                }
            
            stbi_image_free(img16);
            applySmoothing();
            fileLoaded = true;
            dirty = true;
        }    }
     }
    
    void HeightmapInputNode::applySmoothing() {
        if (rawHeightData.empty() || loadedWidth == 0 || loadedHeight == 0) return;
        
        // Start with raw data
        loadedHeightData = rawHeightData;
        
        if (smoothIterations <= 0) return;
        
        std::vector<float> temp = loadedHeightData;
        int w = loadedWidth;
        int h = loadedHeight;
        
        for (int iter = 0; iter < smoothIterations; iter++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float sum = 0.0f;
                    int count = 0;
                    
                    // 3x3 Box Blur
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int nx = x + dx;
                            int ny = y + dy;
                            
                            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                                sum += temp[ny * w + nx];
                                count++;
                            }
                        }
                    }
                    loadedHeightData[y * w + x] = sum / count;
                }
            }
            if (iter < smoothIterations - 1) temp = loadedHeightData;
        }
    }
    
    // ============================================================================
    // NOISE GENERATOR IMPLEMENTATION
    // ============================================================================
    
    NodeSystem::PinValue NoiseGeneratorNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        if (!tctx || !tctx->terrain) {
            ctx.addError(id, "No terrain context");
            return NodeSystem::PinValue{};
        }
        
        int w = tctx->width;
        int h = tctx->height;
        auto result = createHeightOutput(w, h);
        
        TerrainObject* terrain = tctx->terrain;
        TerrainManager& mgr = TerrainManager::getInstance();
        
        // Store original heightmap
        auto backup = terrain->heightmap.data;
        
        // Generate noise directly into terrain
        // Note: If your terrain doesn't have these noise methods, you'll need to implement them
        // For now, we'll generate FBM noise directly here
        
        std::srand(seed);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float nx = (float)x * scale * frequency;
                float ny = (float)y * scale * frequency;
                
                float noise = 0.0f;
                float amp = 1.0f;
                float freq = 1.0f;
                float maxAmp = 0.0f;
                
                for (int o = 0; o < octaves; o++) {
                    // Simple noise approximation
                    float sn = std::sin(nx * freq * 12.9898f + ny * freq * 78.233f);
                    sn = std::sin(sn * 43758.5453f);
                    sn = (sn - std::floor(sn)) * 2.0f - 1.0f;
                    
                    noise += sn * amp;
                    maxAmp += amp;
                    amp *= persistance;
                    freq *= lacunarity;
                }
                
                noise = (noise / maxAmp + 1.0f) * 0.5f * amplitude;
                (*result.data)[y * w + x] = noise;
            }
        }
        
        return result;
    }

    // ============================================================================
    // EROSION NODE IMPLEMENTATIONS
    // ============================================================================
    
    NodeSystem::PinValue HydraulicErosionNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        if (!tctx || !tctx->terrain) {
            ctx.addError(id, "No terrain context");
            return NodeSystem::PinValue{};
        }
        
        // Get input heightmap
        auto inputHeight = getHeightInput(0, ctx);
        if (!inputHeight.isValid()) {
            ctx.addError(id, "No valid height input");
            return NodeSystem::PinValue{};
        }
        
        TerrainObject* terrain = tctx->terrain;
        TerrainManager& mgr = TerrainManager::getInstance();
        
        // ALWAYS apply input to terrain - handle size mismatch
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        terrain->heightmap.data = *inputHeight.data;
        terrain->dirty_mesh = true;
        
        // Get optional mask
        std::vector<float> mask;
        auto maskInput = getHeightInput(1, ctx);
        if (maskInput.isValid() && maskInput.data->size() == terrain->heightmap.data.size()) {
            mask = *maskInput.data;
        }
        
        // Run erosion via TerrainManager
        if (useGPU) {
            mgr.hydraulicErosionGPU(terrain, params, mask);
        } else {
            mgr.hydraulicErosion(terrain, params, mask);
        }
        
        // Create output using INPUT dimensions to propagate correctly
        auto result = createHeightOutput(inputHeight.width, inputHeight.height);
        *result.data = terrain->heightmap.data;
        
        return result;
    }
    
    void HydraulicErosionNode::drawContent() {
        ImGui::Checkbox("Use GPU", &useGPU);
        ImGui::DragInt("Iterations", &params.iterations, 100, 1000, 500000);
        ImGui::DragInt("Droplet Lifetime", &params.dropletLifetime, 1, 10, 256);
        ImGui::DragFloat("Inertia", &params.inertia, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Capacity", &params.sedimentCapacity, 0.1f, 0.1f, 50.0f);
        ImGui::DragFloat("Deposit Speed", &params.depositSpeed, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Erode Speed", &params.erodeSpeed, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Evaporate Speed", &params.evaporateSpeed, 0.001f, 0.0f, 0.1f);
        ImGui::DragFloat("Min Slope", &params.minSlope, 0.001f, 0.0f, 0.1f);
        ImGui::DragFloat("Gravity", &params.gravity, 0.1f, 1.0f, 20.0f);
        ImGui::DragInt("Erosion Radius", &params.erosionRadius, 1, 1, 8);
    }
    
    NodeSystem::PinValue ThermalErosionNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        if (!tctx || !tctx->terrain) {
            ctx.addError(id, "No terrain context");
            return NodeSystem::PinValue{};
        }
        
        auto inputHeight = getHeightInput(0, ctx);
        if (!inputHeight.isValid()) {
            ctx.addError(id, "No valid height input");
            return NodeSystem::PinValue{};
        }
        
        TerrainObject* terrain = tctx->terrain;
        TerrainManager& mgr = TerrainManager::getInstance();
        
        // ALWAYS apply input to terrain - handle size mismatch
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        terrain->heightmap.data = *inputHeight.data;
        terrain->dirty_mesh = true;
        
        // Get optional mask
        std::vector<float> mask;
        auto maskInput = getHeightInput(1, ctx);
        if (maskInput.isValid() && maskInput.data->size() == terrain->heightmap.data.size()) {
            mask = *maskInput.data;
        }
        
        if (useGPU) {
            mgr.thermalErosionGPU(terrain, params, mask);
        } else {
            mgr.thermalErosion(terrain, params, mask);
        }
        
        auto result = createHeightOutput(inputHeight.width, inputHeight.height);
        *result.data = terrain->heightmap.data;
        
        return result;
    }
    
    void ThermalErosionNode::drawContent() {
        ImGui::Checkbox("Use GPU", &useGPU);
        ImGui::DragInt("Iterations", &params.iterations, 1, 1, 500);
        ImGui::DragFloat("Talus Angle", &params.talusAngle, 0.01f, 0.1f, 1.5f);
        ImGui::DragFloat("Erosion Amount", &params.erosionAmount, 0.01f, 0.0f, 1.0f);
    }
    
    NodeSystem::PinValue FluvialErosionNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        if (!tctx || !tctx->terrain) {
            ctx.addError(id, "No terrain context");
            return NodeSystem::PinValue{};
        }
        
        auto inputHeight = getHeightInput(0, ctx);
        if (!inputHeight.isValid()) {
            ctx.addError(id, "No valid height input");
            return NodeSystem::PinValue{};
        }
        
        TerrainObject* terrain = tctx->terrain;
        TerrainManager& mgr = TerrainManager::getInstance();
        
        // ALWAYS apply input to terrain - handle size mismatch
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        terrain->heightmap.data = *inputHeight.data;
        terrain->dirty_mesh = true;
        
        // Get optional mask
        std::vector<float> mask;
        auto maskInput = getHeightInput(1, ctx);
        if (maskInput.isValid() && maskInput.data->size() == terrain->heightmap.data.size()) {
            mask = *maskInput.data;
        }
        
        if (useGPU) {
            mgr.fluvialErosionGPU(terrain, params, mask);
        } else {
            mgr.fluvialErosion(terrain, params, mask);
        }
        
        auto result = createHeightOutput(inputHeight.width, inputHeight.height);
        *result.data = terrain->heightmap.data;
        
        return result;
    }
    
    void FluvialErosionNode::drawContent() {
        ImGui::Checkbox("Use GPU", &useGPU);
        ImGui::DragInt("Iterations", &params.iterations, 100, 1000, 500000);
        ImGui::DragFloat("Evaporate Speed", &params.evaporateSpeed, 0.001f, 0.0f, 0.1f);
        ImGui::DragFloat("Capacity", &params.sedimentCapacity, 0.1f, 0.1f, 50.0f);
        ImGui::DragFloat("Erode Speed", &params.erodeSpeed, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Deposit Speed", &params.depositSpeed, 0.01f, 0.0f, 1.0f);
    }
    
    NodeSystem::PinValue WindErosionNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        if (!tctx || !tctx->terrain) {
            ctx.addError(id, "No terrain context");
            return NodeSystem::PinValue{};
        }
        
        auto inputHeight = getHeightInput(0, ctx);
        if (!inputHeight.isValid()) {
            ctx.addError(id, "No valid height input");
            return NodeSystem::PinValue{};
        }
        
        TerrainObject* terrain = tctx->terrain;
        TerrainManager& mgr = TerrainManager::getInstance();
        
        // ALWAYS apply input to terrain - handle size mismatch
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        terrain->heightmap.data = *inputHeight.data;
        terrain->dirty_mesh = true;
        
        // Get optional mask
        std::vector<float> mask;
        auto maskInput = getHeightInput(1, ctx);
        if (maskInput.isValid() && maskInput.data->size() == terrain->heightmap.data.size()) {
            mask = *maskInput.data;
        }
        
        if (useGPU) {
            mgr.windErosionGPU(terrain, strength, direction, iterations, mask);
        } else {
            mgr.windErosion(terrain, strength, direction, iterations, mask);
        }
        
        auto result = createHeightOutput(inputHeight.width, inputHeight.height);
        *result.data = terrain->heightmap.data;
        
        return result;
    }
    
    void WindErosionNode::drawContent() {
        ImGui::Checkbox("Use GPU", &useGPU);
        ImGui::DragFloat("Strength", &strength, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Direction", &direction, 1.0f, 0.0f, 360.0f);
        ImGui::DragInt("Iterations", &iterations, 1, 1, 100);
    }

    // ============================================================================
    // OUTPUT NODE IMPLEMENTATIONS
    // ============================================================================
    
    NodeSystem::PinValue HeightOutputNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        if (!tctx || !tctx->terrain) {
            ctx.addError(id, "No terrain context");
            return NodeSystem::PinValue{};
        }
        
        auto inputHeight = getHeightInput(0, ctx);
        if (!inputHeight.isValid()) {
            ctx.addError(id, "No valid height input");
            return NodeSystem::PinValue{};
        }
        
        TerrainObject* terrain = tctx->terrain;
        TerrainManager& mgr = TerrainManager::getInstance();
        
        // Resize terrain if dimensions changed
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        
        // Apply heightmap data
        terrain->heightmap.data = *inputHeight.data;
        terrain->dirty_mesh = true;
        terrain->dirty_region.markAllDirty();
        mgr.updateTerrainMesh(terrain);
        
        return NodeSystem::PinValue{}; // Output nodes don't produce data
    }
    
    NodeSystem::PinValue SplatOutputNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        if (!tctx || !tctx->terrain) {
            ctx.addError(id, "No terrain context");
            return NodeSystem::PinValue{};
        }
        
        auto splatInput = getHeightInput(0, ctx);
        if (!splatInput.isValid()) {
            return NodeSystem::PinValue{};
        }
        
        TerrainObject* terrain = tctx->terrain;
        
        // Apply splat data to terrain if autoApply enabled
        if (autoApplyToTerrain && terrain->splatMap) {
            int sw = terrain->splatMap->width;
            int sh = terrain->splatMap->height;
            
            // Check if input is 4-channel
            if (splatInput.channels == 4 && splatInput.data->size() == sw * sh * 4) {
                // Direct copy from 4-channel data
                for (int i = 0; i < sw * sh; i++) {
                    terrain->splatMap->pixels[i].r = (uint8_t)((*splatInput.data)[i * 4 + 0] * 255.0f);
                    terrain->splatMap->pixels[i].g = (uint8_t)((*splatInput.data)[i * 4 + 1] * 255.0f);
                    terrain->splatMap->pixels[i].b = (uint8_t)((*splatInput.data)[i * 4 + 2] * 255.0f);
                    terrain->splatMap->pixels[i].a = (uint8_t)((*splatInput.data)[i * 4 + 3] * 255.0f);
                }
                terrain->splatMap->updateGPU();
            }
        }
        
        return NodeSystem::PinValue{};
    }
    
    void SplatOutputNode::drawContent() {
        ImGui::Checkbox("Apply to Terrain", &autoApplyToTerrain);
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        if (strlen(exportPath) > 0) {
            std::string shortPath = exportPath;
            if (shortPath.length() > 20) {
                shortPath = "..." + shortPath.substr(shortPath.length() - 17);
            }
            ImGui::TextDisabled("%s", shortPath.c_str());
        }
        
        if (ImGui::Button("Export PNG...")) {
            browseForExport = true;
        }
    }
    
    void SplatOutputNode::exportSplatMap(TerrainObject* terrain) {
        if (!terrain || !terrain->splatMap || strlen(exportPath) == 0) return;
        
        int sw = terrain->splatMap->width;
        int sh = terrain->splatMap->height;
        std::vector<uint8_t> rgba(sw * sh * 4);
        
        for (int i = 0; i < sw * sh; i++) {
            rgba[i * 4 + 0] = terrain->splatMap->pixels[i].r;
            rgba[i * 4 + 1] = terrain->splatMap->pixels[i].g;
            rgba[i * 4 + 2] = terrain->splatMap->pixels[i].b;
            rgba[i * 4 + 3] = 255;  // Full alpha for visibility
        }
        
        stbi_write_png(exportPath, sw, sh, 4, rgba.data(), sw * 4);
    }

    // ============================================================================
    // MATH NODE IMPLEMENTATIONS
    // ============================================================================
    
    NodeSystem::PinValue MathNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto inputA = getHeightInput(0, ctx);
        if (!inputA.isValid()) {
            ctx.addError(id, "Input A not valid");
            return NodeSystem::PinValue{};
        }
        
        auto result = createHeightOutput(inputA.width, inputA.height);
        
        // Get optional B input
        auto inputB = getHeightInput(1, ctx);
        bool hasB = inputB.isValid();
        
        // Calculate scale ratios if dimensions differ
        float scaleX = 1.0f;
        float scaleY = 1.0f;
        if (hasB && (inputB.width != inputA.width || inputB.height != inputA.height)) {
            scaleX = (float)inputB.width / (float)inputA.width;
            scaleY = (float)inputB.height / (float)inputA.height;
        }
        
        int w = inputA.width;
        int h = inputA.height;
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                float a = (*inputA.data)[idx];
                float b = factor;
                
                if (hasB) {
                    if (scaleX == 1.0f && scaleY == 1.0f) {
                        b = (*inputB.data)[idx];
                    } else {
                        // Nearest Neighbor Resampling for Input B
                        int bx = std::min((int)(x * scaleX), inputB.width - 1);
                        int by = std::min((int)(y * scaleY), inputB.height - 1);
                        b = (*inputB.data)[by * inputB.width + bx];
                    }
                }
                
                switch (operation) {
                    case MathOp::Add: (*result.data)[idx] = a + b; break;
                    case MathOp::Subtract: (*result.data)[idx] = a - b; break;
                    case MathOp::Multiply: (*result.data)[idx] = a * b; break;
                    case MathOp::Divide: (*result.data)[idx] = (b != 0) ? a / b : 0; break;
                    case MathOp::Min: (*result.data)[idx] = std::min(a, b); break;
                    case MathOp::Max: (*result.data)[idx] = std::max(a, b); break;
                }
            }
        }
        
        return result;
    }
    
    void MathNode::drawContent() {
        const char* opNames[] = { "Add", "Subtract", "Multiply", "Divide", "Min", "Max" };
        int opIdx = (int)operation;
        if (ImGui::Combo("Op", &opIdx, opNames, 6)) {
            operation = (MathOp)opIdx;
            dirty = true;
        }
        ImGui::DragFloat("Factor", &factor, 0.1f);
    }
    
    NodeSystem::PinValue BlendNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto inputA = getHeightInput(0, ctx);
        auto inputB = getHeightInput(1, ctx);
        
        if (!inputA.isValid() || !inputB.isValid()) {
            ctx.addError(id, "Both inputs A and B required");
            return NodeSystem::PinValue{};
        }
        
        if (inputA.data->size() != inputB.data->size()) {
            ctx.addError(id, "Input size mismatch");
            return NodeSystem::PinValue{};
        }
        
        auto result = createHeightOutput(inputA.width, inputA.height);
        auto maskInput = getHeightInput(2, ctx);
        bool hasMask = maskInput.isValid() && maskInput.data->size() == inputA.data->size();
        
        size_t size = inputA.data->size();
        for (size_t i = 0; i < size; i++) {
            float blend = hasMask ? (*maskInput.data)[i] : alpha;
            (*result.data)[i] = (*inputA.data)[i] * (1.0f - blend) + (*inputB.data)[i] * blend;
        }
        
        return result;
    }
    
    void BlendNode::drawContent() {
        ImGui::DragFloat("Alpha", &alpha, 0.01f, 0.0f, 1.0f);
    }
    
    NodeSystem::PinValue ClampNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        if (!input.isValid()) {
            ctx.addError(id, "No valid input");
            return NodeSystem::PinValue{};
        }
        
        auto result = createHeightOutput(input.width, input.height);
        
        for (size_t i = 0; i < input.data->size(); i++) {
            (*result.data)[i] = clampValue((*input.data)[i], minVal, maxVal);
        }
        
        return result;
    }
    
    void ClampNode::drawContent() {
        ImGui::DragFloat("Min", &minVal, 0.1f);
        ImGui::DragFloat("Max", &maxVal, 0.1f);
    }
    
    NodeSystem::PinValue InvertNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        if (!input.isValid()) {
            ctx.addError(id, "No valid input");
            return NodeSystem::PinValue{};
        }
        
        auto result = createHeightOutput(input.width, input.height);
        
        // Find min/max
        float minH = FLT_MAX, maxH = -FLT_MAX;
        for (float h : *input.data) {
            minH = std::min(minH, h);
            maxH = std::max(maxH, h);
        }
        
        for (size_t i = 0; i < input.data->size(); i++) {
            (*result.data)[i] = maxH - ((*input.data)[i] - minH);
        }
        
        return result;
    }

    // ============================================================================
    // MASK NODE IMPLEMENTATIONS
    // ============================================================================
    
    NodeSystem::PinValue SlopeMaskNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        auto input = getHeightInput(0, ctx);
        
        if (!tctx || !input.isValid()) {
            ctx.addError(id, "Invalid context or input");
            return NodeSystem::PinValue{};
        }
        
        int w = input.width;
        int h = input.height;
        auto result = createMaskOutput(w, h);
        
        float cellSize = tctx->terrain ? (tctx->terrain->heightmap.scale_xz / w) : 1.0f;
        
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                
                float dzdx = ((*input.data)[idx + 1] - (*input.data)[idx - 1]) / (2.0f * cellSize);
                float dzdy = ((*input.data)[idx + w] - (*input.data)[idx - w]) / (2.0f * cellSize);
                float slope = std::atan(std::sqrt(dzdx * dzdx + dzdy * dzdy)) * 57.2957795f; // rad to deg
                
                float t = 0.0f;
                if (slope >= minSlope && slope <= maxSlope) {
                    t = 1.0f;
                } else if (slope < minSlope) {
                    t = std::max(0.0f, 1.0f - (minSlope - slope) / (falloff * (maxSlope - minSlope) + 0.001f));
                } else {
                    t = std::max(0.0f, 1.0f - (slope - maxSlope) / (falloff * (maxSlope - minSlope) + 0.001f));
                }
                (*result.data)[idx] = t;
            }
        }
        
        return result;
    }
    
    void SlopeMaskNode::drawContent() {
        ImGui::DragFloat("Min Slope", &minSlope, 0.1f, 0.0f, 90.0f);
        ImGui::DragFloat("Max Slope", &maxSlope, 0.1f, 0.0f, 90.0f);
        ImGui::DragFloat("Falloff", &falloff, 0.01f, 0.0f, 1.0f);
    }
    
    NodeSystem::PinValue HeightMaskNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        
        if (!input.isValid()) {
            ctx.addError(id, "Invalid input");
            return NodeSystem::PinValue{};
        }
        
        auto result = createMaskOutput(input.width, input.height);
        
        for (size_t i = 0; i < input.data->size(); i++) {
            float h = (*input.data)[i];
            float t = 0.0f;
            
            if (h >= minHeight && h <= maxHeight) {
                t = 1.0f;
            } else if (h < minHeight) {
                t = std::max(0.0f, 1.0f - (minHeight - h) / (falloff + 0.001f));
            } else {
                t = std::max(0.0f, 1.0f - (h - maxHeight) / (falloff + 0.001f));
            }
            (*result.data)[i] = t;
        }
        
        return result;
    }
    
    void HeightMaskNode::drawContent() {
        ImGui::DragFloat("Min Height", &minHeight, 1.0f);
        ImGui::DragFloat("Max Height", &maxHeight, 1.0f);
        ImGui::DragFloat("Falloff", &falloff, 0.1f, 0.0f, 100.0f);
    }
    
    NodeSystem::PinValue CurvatureMaskNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        
        if (!input.isValid()) {
            ctx.addError(id, "Invalid input");
            return NodeSystem::PinValue{};
        }
        
        int w = input.width;
        int h = input.height;
        auto result = createMaskOutput(w, h);
        
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                
                float center = (*input.data)[idx];
                float laplacian = 
                    (*input.data)[idx - 1] + (*input.data)[idx + 1] +
                    (*input.data)[idx - w] + (*input.data)[idx + w] - 4.0f * center;
                
                // Positive = concave (valleys), Negative = convex (ridges)
                float curv = selectConvex ? -laplacian : laplacian;
                curv = clampValue((curv - minCurve) / (maxCurve - minCurve + 0.001f), 0.0f, 1.0f);
                (*result.data)[idx] = curv;
            }
        }
        
        return result;
    }
    
    void CurvatureMaskNode::drawContent() {
        ImGui::DragFloat("Min Curve", &minCurve, 0.01f);
        ImGui::DragFloat("Max Curve", &maxCurve, 0.01f);
        ImGui::Checkbox("Select Convex (Ridges)", &selectConvex);
    }

    // ============================================================================
    // NEW OPERATOR NODE IMPLEMENTATIONS
    // ============================================================================
    
    // SMOOTH NODE
    NodeSystem::PinValue SmoothNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        if (!input.isValid()) {
            ctx.addError(id, "No valid input");
            return NodeSystem::PinValue{};
        }
        
        int w = input.width;
        int h = input.height;
        auto result = createHeightOutput(w, h);
        *result.data = *input.data;
        
        // Get optional mask
        auto maskInput = getHeightInput(1, ctx);
        bool hasMask = maskInput.isValid() && maskInput.data->size() == input.data->size();
        
        std::vector<float> temp = *result.data;
        int halfKernel = kernelSize / 2;
        
        for (int iter = 0; iter < iterations; iter++) {
            for (int y = halfKernel; y < h - halfKernel; y++) {
                for (int x = halfKernel; x < w - halfKernel; x++) {
                    int idx = y * w + x;
                    
                    float sum = 0.0f;
                    int count = 0;
                    
                    for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                        for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                            int nIdx = (y + ky) * w + (x + kx);
                            sum += temp[nIdx];
                            count++;
                        }
                    }
                    
                    float avg = sum / count;
                    float maskVal = hasMask ? (*maskInput.data)[idx] : 1.0f;
                    float blendedStrength = strength * maskVal;
                    
                    (*result.data)[idx] = temp[idx] * (1.0f - blendedStrength) + avg * blendedStrength;
                }
            }
            if (iter < iterations - 1) temp = *result.data;
        }
        
        return result;
    }
    
    void SmoothNode::drawContent() {
        ImGui::DragInt("Iterations", &iterations, 1, 1, 50);
        ImGui::DragFloat("Strength", &strength, 0.01f, 0.0f, 1.0f);
        
        const char* sizes[] = { "3x3", "5x5", "7x7" };
        int sizeIdx = (kernelSize == 3) ? 0 : ((kernelSize == 5) ? 1 : 2);
        if (ImGui::Combo("Kernel", &sizeIdx, sizes, 3)) {
            kernelSize = (sizeIdx == 0) ? 3 : ((sizeIdx == 1) ? 5 : 7);
            dirty = true;
        }
    }
    
    // NORMALIZE NODE
    NodeSystem::PinValue NormalizeNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        if (!input.isValid()) {
            ctx.addError(id, "No valid input");
            return NodeSystem::PinValue{};
        }
        
        auto result = createHeightOutput(input.width, input.height);
        
        // Find min/max in input
        float minH = FLT_MAX, maxH = -FLT_MAX;
        for (float h : *input.data) {
            minH = std::min(minH, h);
            maxH = std::max(maxH, h);
        }
        
        float range = maxH - minH;
        if (range < 0.0001f) range = 1.0f;
        
        float outRange = maxOutput - minOutput;
        
        for (size_t i = 0; i < input.data->size(); i++) {
            float normalized = ((*input.data)[i] - minH) / range;
            (*result.data)[i] = minOutput + normalized * outRange;
        }
        
        return result;
    }
    
    void NormalizeNode::drawContent() {
        ImGui::Checkbox("Auto Range", &autoRange);
        if (!autoRange) {
            ImGui::DragFloat("Min Out", &minOutput, 1.0f);
            ImGui::DragFloat("Max Out", &maxOutput, 1.0f);
        } else {
            ImGui::TextDisabled("Input: Auto");
            ImGui::DragFloat("Max Out", &maxOutput, 1.0f, 0.0f, 10000.0f);
            minOutput = 0.0f;
        }
    }
    
    // TERRACE NODE
    NodeSystem::PinValue TerraceNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        if (!input.isValid()) {
            ctx.addError(id, "No valid input");
            return NodeSystem::PinValue{};
        }
        
        int w = input.width;
        int h = input.height;
        auto result = createHeightOutput(w, h);
        
        // Get optional mask
        auto maskInput = getHeightInput(1, ctx);
        bool hasMask = maskInput.isValid() && maskInput.data->size() == input.data->size();
        
        // Find min/max
        float minH = FLT_MAX, maxH = -FLT_MAX;
        for (float hVal : *input.data) {
            minH = std::min(minH, hVal);
            maxH = std::max(maxH, hVal);
        }
        float range = maxH - minH;
        if (range < 0.0001f) range = 1.0f;
        
        float stepSize = range / levels;
        
        for (size_t i = 0; i < input.data->size(); i++) {
            float h = (*input.data)[i];
            float normalized = (h - minH) / range + offset / range;
            
            // Calculate terrace level
            float level = std::floor(normalized * levels) / levels;
            float fraction = (normalized * levels) - std::floor(normalized * levels);
            
            // Blend between hard step and smooth ramp based on sharpness
            float terraced;
            if (sharpness >= 1.0f) {
                terraced = level;
            } else {
                float smooth = level + fraction / levels;
                float step = level + (fraction > 0.5f ? 1.0f / levels : 0.0f);
                terraced = smooth * (1.0f - sharpness) + step * sharpness;
            }
            
            float newH = minH + terraced * range;
            
            // Apply mask
            float maskVal = hasMask ? (*maskInput.data)[i] : 1.0f;
            (*result.data)[i] = h * (1.0f - maskVal) + newH * maskVal;
        }
        
        return result;
    }
    
    void TerraceNode::drawContent() {
        if (ImGui::DragInt("Levels", &levels, 1, 2, 64)) dirty = true;
        if (ImGui::DragFloat("Sharpness", &sharpness, 0.01f, 0.0f, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Offset", &offset, 0.1f, -100.0f, 100.0f)) dirty = true;
    }
    
    // MASK COMBINE NODE
    NodeSystem::PinValue MaskCombineNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto inputA = getHeightInput(0, ctx);
        auto inputB = getHeightInput(1, ctx);
        
        if (!inputA.isValid() || !inputB.isValid()) {
            ctx.addError(id, "Both mask inputs required");
            return NodeSystem::PinValue{};
        }
        
        if (inputA.data->size() != inputB.data->size()) {
            ctx.addError(id, "Mask size mismatch");
            return NodeSystem::PinValue{};
        }
        
        auto result = createMaskOutput(inputA.width, inputA.height);
        
        for (size_t i = 0; i < inputA.data->size(); i++) {
            float a = (*inputA.data)[i];
            float b = (*inputB.data)[i];
            float out = 0.0f;
            
            switch (operation) {
                case MaskCombineOp::AND:
                    out = std::min(a, b);
                    break;
                case MaskCombineOp::OR:
                    out = std::max(a, b);
                    break;
                case MaskCombineOp::XOR:
                case MaskCombineOp::Difference:
                    out = std::abs(a - b);
                    break;
                case MaskCombineOp::Multiply:
                    out = a * b;
                    break;
                case MaskCombineOp::Add:
                    out = clampValue(a + b, 0.0f, 1.0f);
                    break;
                case MaskCombineOp::Subtract:
                    out = clampValue(a - b, 0.0f, 1.0f);
                    break;
            }
            
            (*result.data)[i] = out;
        }
        
        return result;
    }
    
    void MaskCombineNode::drawContent() {
        const char* opNames[] = { "AND (Min)", "OR (Max)", "XOR", "Multiply", "Add", "Subtract", "Difference" };
        int opIdx = (int)operation;
        if (ImGui::Combo("Operation", &opIdx, opNames, 7)) {
            operation = (MaskCombineOp)opIdx;
            dirty = true;
        }
    }
    
    // OVERLAY NODE
    NodeSystem::PinValue OverlayNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto inputBase = getHeightInput(0, ctx);
        auto inputBlend = getHeightInput(1, ctx);
        
        if (!inputBase.isValid() || !inputBlend.isValid()) {
            ctx.addError(id, "Both inputs required");
            return NodeSystem::PinValue{};
        }
        
        int w = inputBase.width;
        int h = inputBase.height;
        auto result = createHeightOutput(w, h);
        
        // Get optional mask
        auto maskInput = getHeightInput(2, ctx);
        bool hasMask = maskInput.isValid() && maskInput.data->size() == inputBase.data->size();
        
        // Need to normalize to 0-1 for overlay blend
        float minB = FLT_MAX, maxB = -FLT_MAX;
        float minL = FLT_MAX, maxL = -FLT_MAX;
        for (size_t i = 0; i < inputBase.data->size(); i++) {
            minB = std::min(minB, (*inputBase.data)[i]);
            maxB = std::max(maxB, (*inputBase.data)[i]);
            minL = std::min(minL, (*inputBlend.data)[i]);
            maxL = std::max(maxL, (*inputBlend.data)[i]);
        }
        float rangeB = (maxB - minB > 0.0001f) ? maxB - minB : 1.0f;
        float rangeL = (maxL - minL > 0.0001f) ? maxL - minL : 1.0f;
        
        for (size_t i = 0; i < inputBase.data->size(); i++) {
            float base = ((*inputBase.data)[i] - minB) / rangeB;
            float blend = ((*inputBlend.data)[i] - minL) / rangeL;
            
            // Overlay formula: a < 0.5 ? 2*a*b : 1 - 2*(1-a)*(1-b)
            float overlay;
            if (base < 0.5f) {
                overlay = 2.0f * base * blend;
            } else {
                overlay = 1.0f - 2.0f * (1.0f - base) * (1.0f - blend);
            }
            
            // Convert back to original scale
            float outValue = minB + overlay * rangeB;
            
            // Apply strength and mask
            float maskVal = hasMask ? (*maskInput.data)[i] : 1.0f;
            float blendAmount = strength * maskVal;
            
            (*result.data)[i] = (*inputBase.data)[i] * (1.0f - blendAmount) + outValue * blendAmount;
        }
        
        return result;
    }
    
    void OverlayNode::drawContent() {
        if (ImGui::DragFloat("Strength", &strength, 0.01f, 0.0f, 1.0f)) dirty = true;
    }
    
    // SCREEN NODE
    NodeSystem::PinValue ScreenNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto inputBase = getHeightInput(0, ctx);
        auto inputBlend = getHeightInput(1, ctx);
        
        if (!inputBase.isValid() || !inputBlend.isValid()) {
            ctx.addError(id, "Both inputs required");
            return NodeSystem::PinValue{};
        }
        
        int w = inputBase.width;
        int h = inputBase.height;
        auto result = createHeightOutput(w, h);
        
        // Get optional mask
        auto maskInput = getHeightInput(2, ctx);
        bool hasMask = maskInput.isValid() && maskInput.data->size() == inputBase.data->size();
        
        // Normalize to 0-1
        float minB = FLT_MAX, maxB = -FLT_MAX;
        float minL = FLT_MAX, maxL = -FLT_MAX;
        for (size_t i = 0; i < inputBase.data->size(); i++) {
            minB = std::min(minB, (*inputBase.data)[i]);
            maxB = std::max(maxB, (*inputBase.data)[i]);
            minL = std::min(minL, (*inputBlend.data)[i]);
            maxL = std::max(maxL, (*inputBlend.data)[i]);
        }
        float rangeB = (maxB - minB > 0.0001f) ? maxB - minB : 1.0f;
        float rangeL = (maxL - minL > 0.0001f) ? maxL - minL : 1.0f;
        
        for (size_t i = 0; i < inputBase.data->size(); i++) {
            float base = ((*inputBase.data)[i] - minB) / rangeB;
            float blend = ((*inputBlend.data)[i] - minL) / rangeL;
            
            // Screen formula: 1 - (1-a)*(1-b)
            float screen = 1.0f - (1.0f - base) * (1.0f - blend);
            
            // Convert back to original scale
            float outValue = minB + screen * rangeB;
            
            // Apply strength and mask
            float maskVal = hasMask ? (*maskInput.data)[i] : 1.0f;
            float blendAmount = strength * maskVal;
            
            (*result.data)[i] = (*inputBase.data)[i] * (1.0f - blendAmount) + outValue * blendAmount;
        }
        
        return result;
    }
    
    void ScreenNode::drawContent() {
        if (ImGui::DragFloat("Strength", &strength, 0.01f, 0.0f, 1.0f)) dirty = true;
    }

    // ============================================================================
    // PROCEDURAL TEXTURE NODE IMPLEMENTATIONS
    // ============================================================================
    
    // Helper: Simple hash-based noise
    static float simpleNoise(int x, int y, int seed) {
        int n = x + y * 57 + seed * 131;
        n = (n << 13) ^ n;
        return (1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f) * 0.5f + 0.5f;
    }
    
    // Helper: Smoothstep
    static float smoothstepLocal(float edge0, float edge1, float x) {
        float t = clampValue((x - edge0) / (edge1 - edge0 + 0.0001f), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }
    
    NodeSystem::PinValue AutoSplatNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        auto input = getHeightInput(0, ctx);
        
        if (!input.isValid()) {
            ctx.addError(id, "Height input required");
            return NodeSystem::PinValue{};
        }
        
        int w = input.width;
        int h = input.height;
        
        // Create 4-channel output (stored as 4 separate values per pixel)
        NodeSystem::Image2DData result;
        result.data = std::make_shared<std::vector<float>>(w * h * 4, 0.0f);
        result.width = w;
        result.height = h;
        result.channels = 4;
        result.semantic = NodeSystem::ImageSemantic::Mask;
        
        float cellSize = 1.0f;
        float heightScale = 100.0f; // Default scale if no terrain context
        
        if (tctx && tctx->terrain) {
            cellSize = tctx->terrain->heightmap.scale_xz / w; // Assuming uniform grid
            heightScale = tctx->terrain->heightmap.scale_y;
        }

        // Calculate weights for each pixel
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                // De-normalize height for rule evaluation (0-1 -> 0-heightScale)
                float height = (*input.data)[idx] * heightScale;
                
                // Calculate slope (degrees) using scaled heights
                float h_l = (*input.data)[idx - 1] * heightScale;
                float h_r = (*input.data)[idx + 1] * heightScale;
                float h_u = (*input.data)[idx - w] * heightScale;
                float h_d = (*input.data)[idx + w] * heightScale; // Down in image space is +Y
                
                float dzdx = (h_r - h_l) / (2.0f * cellSize);
                float dzdy = (h_d - h_u) / (2.0f * cellSize);
                float slope = std::atan(std::sqrt(dzdx * dzdx + dzdy * dzdy)) * 57.2957795f;
                
                float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                
                // Calculate weight for each layer
                for (int layer = 0; layer < 4; layer++) {
                    if (!rules[layer].enabled) continue;
                    
                    const auto& rule = rules[layer];
                    
                    // Height contribution
                    float heightWeight = 0.0f;
                    if (height >= rule.heightMin && height <= rule.heightMax) {
                        heightWeight = 1.0f;
                    } else if (height < rule.heightMin) {
                        heightWeight = smoothstepLocal(rule.heightMin - rule.falloff, rule.heightMin, height);
                    } else {
                        heightWeight = 1.0f - smoothstepLocal(rule.heightMax, rule.heightMax + rule.falloff, height);
                    }
                    
                    // Slope contribution
                    float slopeWeight = 0.0f;
                    if (slope >= rule.slopeMin && slope <= rule.slopeMax) {
                        slopeWeight = 1.0f;
                    } else if (slope < rule.slopeMin) {
                        slopeWeight = smoothstepLocal(rule.slopeMin - rule.falloff, rule.slopeMin, slope);
                    } else {
                        slopeWeight = 1.0f - smoothstepLocal(rule.slopeMax, rule.slopeMax + rule.falloff, slope);
                    }
                    
                    // Combine height and slope
                    float finalWeight = heightWeight * rule.heightWeight + slopeWeight * rule.slopeWeight;
                    
                    // Add noise variation
                    if (rule.noiseAmount > 0.0f) {
                        float noise = simpleNoise(x, y, noiseSeed + layer) * 2.0f - 1.0f;
                        finalWeight += noise * rule.noiseAmount;
                    }
                    
                    weights[layer] = clampValue(finalWeight, 0.0f, 1.0f);
                }
                
                // Normalize weights if requested
                if (normalizeOutput) {
                    float sum = weights[0] + weights[1] + weights[2] + weights[3];
                    if (sum > 0.001f) {
                        for (int i = 0; i < 4; i++) weights[i] /= sum;
                    } else {
                        weights[0] = 1.0f; // Default to first layer
                    }
                }
                
                // Store as RGBA
                (*result.data)[idx * 4 + 0] = weights[0];
                (*result.data)[idx * 4 + 1] = weights[1];
                (*result.data)[idx * 4 + 2] = weights[2];
                (*result.data)[idx * 4 + 3] = weights[3];
            }
        }
        
        // Fill edges
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 4; c++) {
                (*result.data)[x * 4 + c] = (*result.data)[(w + x) * 4 + c];
                (*result.data)[((h-1) * w + x) * 4 + c] = (*result.data)[((h-2) * w + x) * 4 + c];
            }
        }
        for (int y = 0; y < h; y++) {
            for (int c = 0; c < 4; c++) {
                (*result.data)[(y * w) * 4 + c] = (*result.data)[(y * w + 1) * 4 + c];
                (*result.data)[(y * w + w - 1) * 4 + c] = (*result.data)[(y * w + w - 2) * 4 + c];
            }
        }
        
        return result;
    }
    
    void AutoSplatNode::drawContent() {
        const char* layerNames[] = { "Layer 0 (R)", "Layer 1 (G)", "Layer 2 (B)", "Layer 3 (A)" };
        
        for (int i = 0; i < 4; i++) {
            if (ImGui::TreeNode(layerNames[i])) {
                auto& rule = rules[i];
                
                if (ImGui::Checkbox("Enabled", &rule.enabled)) dirty = true;
                
                if (rule.enabled) {
                    ImGui::Text("Height Range:");
                    if (ImGui::DragFloat("H Min##h", &rule.heightMin, 1.0f, 0.0f, 1000.0f)) dirty = true;
                    if (ImGui::DragFloat("H Max##h", &rule.heightMax, 1.0f, 0.0f, 1000.0f)) dirty = true;
                    
                    ImGui::Text("Slope Range (deg):");
                    if (ImGui::DragFloat("S Min##s", &rule.slopeMin, 1.0f, 0.0f, 90.0f)) dirty = true;
                    if (ImGui::DragFloat("S Max##s", &rule.slopeMax, 1.0f, 0.0f, 90.0f)) dirty = true;
                    
                    ImGui::Text("Weights:");
                    if (ImGui::SliderFloat("Height W", &rule.heightWeight, 0.0f, 1.0f)) dirty = true;
                    if (ImGui::SliderFloat("Slope W", &rule.slopeWeight, 0.0f, 1.0f)) dirty = true;
                    
                    if (ImGui::DragFloat("Falloff", &rule.falloff, 0.5f, 0.0f, 50.0f)) dirty = true;
                    if (ImGui::DragFloat("Noise", &rule.noiseAmount, 0.01f, 0.0f, 0.5f)) dirty = true;
                }
                
                ImGui::TreePop();
            }
        }
        
        ImGui::Separator();
        if (ImGui::Checkbox("Normalize", &normalizeOutput)) dirty = true;
        if (ImGui::DragInt("Noise Seed", &noiseSeed)) dirty = true;
    }
    
    // MASK PAINT NODE
    void MaskPaintNode::initBuffer(int width, int height) {
        if (bufferWidth != width || bufferHeight != height) {
            bufferWidth = width;
            bufferHeight = height;
            paintBuffer.resize(width * height, 0.0f);
            needsInit = false;
        }
    }
    
    void MaskPaintNode::paint(float u, float v, float strength) {
        if (paintBuffer.empty() || bufferWidth == 0 || bufferHeight == 0) return;
        
        int cx = (int)(u * (bufferWidth - 1));
        int cy = (int)(v * (bufferHeight - 1));
        int radiusPixels = (int)(brushRadius * bufferWidth / 100.0f);
        if (radiusPixels < 1) radiusPixels = 1;
        
        for (int dy = -radiusPixels; dy <= radiusPixels; dy++) {
            for (int dx = -radiusPixels; dx <= radiusPixels; dx++) {
                int px = cx + dx;
                int py = cy + dy;
                
                if (px < 0 || px >= bufferWidth || py < 0 || py >= bufferHeight) continue;
                
                float dist = std::sqrt((float)(dx * dx + dy * dy));
                if (dist > radiusPixels) continue;
                
                float falloff = 1.0f - (dist / radiusPixels);
                falloff = std::pow(falloff, 2.0f - brushFalloff * 2.0f);
                
                int idx = py * bufferWidth + px;
                paintBuffer[idx] += strength * brushStrength * falloff;
                paintBuffer[idx] = clampValue(paintBuffer[idx], 0.0f, 1.0f);
            }
        }
        
        dirty = true;
    }
    
    NodeSystem::PinValue MaskPaintNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        
        // Get resolution from reference input or terrain
        int w = 512, h = 512;
        auto refInput = getHeightInput(0, ctx);
        if (refInput.isValid()) {
            w = refInput.width;
            h = refInput.height;
        } else if (tctx && tctx->terrain) {
            w = tctx->terrain->heightmap.width;
            h = tctx->terrain->heightmap.height;
        }
        
        // Initialize buffer if needed
        if (needsInit || paintBuffer.empty() || bufferWidth != w || bufferHeight != h) {
            initBuffer(w, h);
        }
        
        auto result = createMaskOutput(bufferWidth, bufferHeight);
        *result.data = paintBuffer;
        
        return result;
    }
    
    void MaskPaintNode::drawContent() {
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.3f, 1.0f), "Brush Settings");
        
        if (ImGui::DragFloat("Radius", &brushRadius, 0.5f, 1.0f, 100.0f)) dirty = true;
        if (ImGui::DragFloat("Strength", &brushStrength, 0.01f, 0.0f, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Falloff", &brushFalloff, 0.01f, 0.0f, 1.0f)) dirty = true;
        
        ImGui::Spacing();
        
        if (ImGui::Button("Clear")) { clear(); dirty = true; }
        ImGui::SameLine();
        if (ImGui::Button("Fill")) { fill(1.0f); dirty = true; }
        
        ImGui::TextDisabled("Size: %dx%d", bufferWidth, bufferHeight);
        
        if (isPainting) {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "PAINTING...");
        }
    }
    
    // MASK IMAGE NODE
    void MaskImageNode::loadMaskFromFile() {
        if (strlen(filePath) == 0) return;
        
        int w, h, channels;
        unsigned char* data = stbi_load(filePath, &w, &h, &channels, 1);  // Force grayscale
        
        if (!data) {
            fileLoaded = false;
            return;
        }
        
        loadedWidth = w;
        loadedHeight = h;
        loadedMask.resize(w * h);
        
        for (int i = 0; i < w * h; i++) {
            float val = data[i] / 255.0f;
            
            // Apply adjustments
            val = (val - 0.5f) * contrast + 0.5f + brightness;
            if (invert) val = 1.0f - val;
            
            loadedMask[i] = clampValue(val, 0.0f, 1.0f);
        }
        
        stbi_image_free(data);
        fileLoaded = true;
        dirty = true;
    }
    
    NodeSystem::PinValue MaskImageNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        if (!fileLoaded || loadedMask.empty()) {
            ctx.addError(id, "No mask loaded");
            return NodeSystem::PinValue{};
        }
        
        auto result = createMaskOutput(loadedWidth, loadedHeight);
        *result.data = loadedMask;
        
        return result;
    }
    
    void MaskImageNode::drawContent() {
        if (fileLoaded) {
            std::string shortPath = filePath;
            if (shortPath.length() > 25) {
                shortPath = "..." + shortPath.substr(shortPath.length() - 22);
            }
            ImGui::TextDisabled("%s", shortPath.c_str());
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "Size: %dx%d", loadedWidth, loadedHeight);
        } else {
            ImGui::TextDisabled("No file loaded");
        }
        
        if (ImGui::Button("Browse...")) {
            browseForMask = true;
        }
        
        ImGui::Spacing();
        ImGui::Text("Adjustments:");
        if (ImGui::DragFloat("Contrast", &contrast, 0.01f, 0.0f, 3.0f)) {
            if (fileLoaded) loadMaskFromFile();
        }
        if (ImGui::DragFloat("Brightness", &brightness, 0.01f, -1.0f, 1.0f)) {
            if (fileLoaded) loadMaskFromFile();
        }
        if (ImGui::Checkbox("Invert", &invert)) {
            if (fileLoaded) loadMaskFromFile();
        }
    }

    // ============================================================================
    // TERRAIN GRAPH V2 IMPLEMENTATION
    // ============================================================================
    
    NodeSystem::NodeBase* TerrainNodeGraphV2::addTerrainNode(NodeType type, float x, float y) {
        NodeSystem::NodeBase* node = nullptr;
        
        switch (type) {
            case NodeType::HeightmapInput: node = addNode<HeightmapInputNode>(); break;
            case NodeType::NoiseGenerator: node = addNode<NoiseGeneratorNode>(); break;
            case NodeType::HydraulicErosion: node = addNode<HydraulicErosionNode>(); break;
            case NodeType::ThermalErosion: node = addNode<ThermalErosionNode>(); break;
            case NodeType::FluvialErosion: node = addNode<FluvialErosionNode>(); break;
            case NodeType::WindErosion: node = addNode<WindErosionNode>(); break;
            case NodeType::HeightOutput: node = addNode<HeightOutputNode>(); break;
            case NodeType::SplatOutput: node = addNode<SplatOutputNode>(); break;
            case NodeType::Add:
            case NodeType::Subtract:
            case NodeType::Multiply:
                node = addNode<MathNode>();
                break;
            case NodeType::Blend: node = addNode<BlendNode>(); break;
            case NodeType::Clamp: node = addNode<ClampNode>(); break;
            case NodeType::Invert: node = addNode<InvertNode>(); break;
            case NodeType::SlopeMask: node = addNode<SlopeMaskNode>(); break;
            case NodeType::HeightMask: node = addNode<HeightMaskNode>(); break;
            case NodeType::CurvatureMask: node = addNode<CurvatureMaskNode>(); break;
            // NEW OPERATORS
            case NodeType::Smooth: node = addNode<SmoothNode>(); break;
            case NodeType::Normalize: node = addNode<NormalizeNode>(); break;
            case NodeType::Terrace: node = addNode<TerraceNode>(); break;
            case NodeType::MaskCombine: node = addNode<MaskCombineNode>(); break;
            case NodeType::Overlay: node = addNode<OverlayNode>(); break;
            case NodeType::Screen: node = addNode<ScreenNode>(); break;
            // NEW: Procedural Texture Nodes
            case NodeType::AutoSplat: node = addNode<AutoSplatNode>(); break;
            case NodeType::MaskPaint: node = addNode<MaskPaintNode>(); break;
            case NodeType::MaskImage: node = addNode<MaskImageNode>(); break;
        }
        
        if (node) {
            node->x = x;
            node->y = y;
        }
        
        return node;
    }
    
    void TerrainNodeGraphV2::evaluateTerrain(TerrainObject* terrain, SceneData& scene) {
        if (!terrain) return;
        
        TerrainContext tctx;
        tctx.terrain = terrain;
        
        NodeSystem::EvaluationContext ctx(this);
        ctx.setDomainContext(&tctx);

        // Find Height Output Node and pull data
        HeightOutputNode* outputNode = nullptr;
        for (auto& node : nodes) {
            if (node->getTypeId() == "TerrainV2.HeightOutput") {
                outputNode = dynamic_cast<HeightOutputNode*>(node.get());
                break;
            }
        }
        
        if (outputNode) {
            // Pull data from the input of the output node (Input index 0 for Height)
            // This triggers the pull-based evaluation chain
            auto heightData = outputNode->getHeightInput(0, ctx);
            
            if (heightData.isValid() && heightData.data) {
                // Check if resize needed
                bool resized = (terrain->heightmap.width != heightData.width || terrain->heightmap.height != heightData.height);
                
                // Resize terrain heightmap manually (vector resize)
                if (resized) {
                    terrain->heightmap.width = heightData.width;
                    terrain->heightmap.height = heightData.height;
                    terrain->heightmap.data.resize(heightData.width * heightData.height);
                }
                
                // Copy data directly to terrain
                terrain->heightmap.data = *heightData.data;
                
                // Update mesh visualization (Rebuild if resized, Update if content changed)
                if (resized) {
                    TerrainManager::getInstance().rebuildTerrainMesh(scene, terrain);
                } else {
                    TerrainManager::getInstance().updateTerrainMesh(terrain);
                }
            }
        }
    }
    
    
    void TerrainNodeGraphV2::createDefaultGraph(TerrainObject* terrain) {
        clear();
        
        // Create default nodes
        auto* inputNode = addTerrainNode(NodeType::HeightmapInput, 50, 100);
        auto* outputNode = addTerrainNode(NodeType::HeightOutput, 400, 100);
        
        // Connect them
        if (inputNode && outputNode && 
            !inputNode->outputs.empty() && !outputNode->inputs.empty()) {
            addLink(inputNode->outputs[0].id, outputNode->inputs[0].id);
        }
    }

} // namespace TerrainNodesV2
