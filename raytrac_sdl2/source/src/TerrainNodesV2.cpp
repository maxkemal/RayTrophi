/**
 * @file TerrainNodesV2.cpp
 * @brief Implementation of terrain nodes using V2 NodeSystem
 */

#include "TerrainNodesV2.h"
#include "scene_data.h"
#include "TerrainManager.h"
#include "TerrainFFT.h"  // FFT-accelerated noise with CUDA fallback
#include "stb_image.h"
#include "stb_image_write.h"
#include <cmath>
#include <algorithm>
#include <cfloat>  // FLT_MAX
#include <fstream>
#include <string>
#include <cstring>
#include "perlin.h" // For gradient noise

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
        
        // SAFETY: Check if path has extension
        size_t dotPos = path.find_last_of('.');
        if (dotPos == std::string::npos || dotPos == path.length() - 1) {
            // No extension found - cannot determine format
            fileLoaded = false;
            return;
        }
        
        std::string ext = path.substr(dotPos + 1);
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
            sourceMode = SourceMode::File; // AUTO-SWITCH to File mode
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
                sourceMode = SourceMode::File; // AUTO-SWITCH to File mode
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
            sourceMode = SourceMode::File; // AUTO-SWITCH to File mode
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
        
        // Create NEW Perlin noise generator each time (seed affects random vectors)
        // static so parameters can change
       static Perlin perlin;
        
        // ═══════════════════════════════════════════════════════════
        // NOISE HELPER FUNCTIONS
        // ═══════════════════════════════════════════════════════════
        
        // Seed-based hash function
        auto hash = [this](float n) -> float {
            float seeded = n + (float)seed * 17.31f;
            float result = std::sin(seeded) * 43758.5453f;
            return result - std::floor(result);
        };
        
        // 2D hash for Voronoi
        auto hash2 = [this](float x, float y) -> std::pair<float, float> {
            float sx = x + (float)seed * 0.31f;
            float sy = y + (float)seed * 0.47f;
            float px = sx * 127.1f + sy * 311.7f;
            float py = sx * 269.5f + sy * 183.3f;
            float hx = std::sin(px) * 43758.5453f;
            float hy = std::sin(py) * 43758.5453f;
            return { hx - std::floor(hx), hy - std::floor(hy) };
        };
        
        // Simple 2D value noise
        auto noise2D = [&hash](float x, float y) -> float {
            float px = std::floor(x);
            float py = std::floor(y);
            float fx = x - px;
            float fy = y - py;
            
            // Smoothstep
            fx = fx * fx * (3.0f - 2.0f * fx);
            fy = fy * fy * (3.0f - 2.0f * fy);
            
            float n = px + py * 57.0f;
            float a = hash(n);
            float b = hash(n + 1.0f);
            float c = hash(n + 57.0f);
            float d = hash(n + 58.0f);
            
            return a + fx * (b - a) + fy * (c - a) + fx * fy * (a - b - c + d);
        };
        
        // FBM (Fractional Brownian Motion)
        auto fbm = [&noise2D, this](float x, float y) -> float {
            float value = 0.0f;
            float amp = 1.0f;
            float freq = 1.0f;
            float maxAmp = 0.0f;
            
            for (int i = 0; i < octaves; i++) {
                value += noise2D(x * freq, y * freq) * amp;
                maxAmp += amp;
                amp *= persistance;
                freq *= lacunarity;
            }
            return value / maxAmp;
        };
        
        // Voronoi/Worley noise
        auto voronoi = [&hash2, this](float x, float y) -> float {
            float ix = std::floor(x);
            float iy = std::floor(y);
            float fx = x - ix;
            float fy = y - iy;
            
            float minDist = 10.0f;
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    auto hashResult = hash2(ix + dx, iy + dy);
                    float hx = hashResult.first;
                    float hy = hashResult.second;
                    float px = dx + hx * jitter - fx;
                    float py = dy + hy * jitter - fy;
                    float dist = std::sqrt(px * px + py * py);
                    minDist = (std::min)(minDist, dist);
                }
            }
            return minDist;
        };
        
        // Voronoi FBM
        auto voronoiFbm = [&voronoi, this](float x, float y) -> float {
            float value = 0.0f;
            float amp = 1.0f;
            float freq = 1.0f;
            float maxAmp = 0.0f;
            
            for (int i = 0; i < octaves; i++) {
                value += voronoi(x * freq, y * freq) * amp;
                maxAmp += amp;
                amp *= persistance;
                freq *= lacunarity;
            }
            return value / maxAmp;
        };
        
        // ═══════════════════════════════════════════════════════════
        // GENERATE NOISE FOR EACH PIXEL
        // ═══════════════════════════════════════════════════════════
        
        // Seed offset for variation
        float seedOffsetX = (float)(seed % 1000) * 0.1f;
        float seedOffsetY = (float)((seed / 1000) % 1000) * 0.1f + 100.0f;
        
        // Get terrain aspect ratio for proper noise scaling
        float terrainAspect = 1.0f;
        if (tctx->terrain && w > 0 && h > 0) {
            // Use consistent scaling based on terrain's actual world size
            terrainAspect = (float)w / (float)h;
        }
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // Normalize to 0-1 range preserving aspect ratio
                // Both coordinates use the same base scale for isotropic noise
                float baseX = (float)x / (float)(std::max)(w, h);
                float baseY = (float)y / (float)(std::max)(w, h);
                
                // Apply scale and frequency
                float nx = baseX * scale * 10.0f + seedOffsetX;
                float ny = baseY * scale * 10.0f + seedOffsetY;
                
                // Additional frequency multiplier
                nx *= frequency * 100.0f;
                ny *= frequency * 100.0f;
                
                float noiseValue = 0.0f;
                
                switch (noiseType) {
                    case NoiseType::Perlin: {
                        // Use Perlin class with turb() for FBM
                        Vec3 p(nx, ny, 0.0f);
                        noiseValue = perlin.turb(p, octaves);
                        break;
                    }
                    
                    case NoiseType::Simplex: {
                        // Hash-based FBM (similar to CPUCloudNoise)
                        noiseValue = fbm(nx, ny);
                        break;
                    }
                    
                    case NoiseType::Voronoi: {
                        // Worley/cell noise
                        noiseValue = 1.0f - voronoiFbm(nx, ny);
                        break;
                    }
                    
                    case NoiseType::Ridge: {
                        // Ridge noise: sharp mountain ridges
                        float value = 0.0f;
                        float amp = 1.0f;
                        float freq = 1.0f;
                        float maxAmp = 0.0f;
                        float weight = 1.0f;
                        
                        for (int i = 0; i < octaves; i++) {
                            Vec3 p(nx * freq, ny * freq, 0.0f);
                            float n = perlin.noise(p);
                            n = ridge_offset - std::abs(n);
                            n = n * n; // Square for sharper ridges
                            n *= weight;
                            weight = clampValue(n * 2.0f, 0.0f, 1.0f);
                            
                            value += n * amp;
                            maxAmp += amp;
                            amp *= persistance;
                            freq *= lacunarity;
                        }
                        noiseValue = value / maxAmp;
                        break;
                    }
                    
                    case NoiseType::Billow: {
                        // Billow: soft, puffy hills
                        float value = 0.0f;
                        float amp = 1.0f;
                        float freq = 1.0f;
                        float maxAmp = 0.0f;
                        
                        for (int i = 0; i < octaves; i++) {
                            Vec3 p(nx * freq, ny * freq, 0.0f);
                            float n = perlin.noise(p);
                            n = std::abs(n) * 2.0f - 1.0f;
                            
                            value += n * amp;
                            maxAmp += amp;
                            amp *= persistance;
                            freq *= lacunarity;
                        }
                        noiseValue = (value / maxAmp + 1.0f) * 0.5f;
                        break;
                    }
                    
                    case NoiseType::Warped: {
                        // Domain warping: distort coordinates with noise
                        float warpX = fbm(nx, ny) * warp_strength * 5.0f;
                        float warpY = fbm(nx + 5.2f, ny + 1.3f) * warp_strength * 5.0f;
                        
                        Vec3 p(nx + warpX, ny + warpY, 0.0f);
                        noiseValue = perlin.turb(p, octaves);
                        break;
                    }
                    
                    // ═══════════════════════════════════════════════════════════
                    // FFT-ACCELERATED NOISE TYPES
                    // Uses TerrainFFT system: CUDA if available, CPU fallback otherwise
                    // ═══════════════════════════════════════════════════════════
                    
                    case NoiseType::FFT_Ocean:
                    case NoiseType::FFT_Ridge:
                    case NoiseType::FFT_Billow:
                    case NoiseType::FFT_Turb: {
                        // For FFT types, we'll generate the full heightmap once
                        // and then read from it. Since we're in a per-pixel loop,
                        // we'll use CPU fallback noise here (similar algorithm)
                        // The actual FFT generation happens in a separate code path
                        
                        // Use TerrainFFT CPU noise as these are compatible
                        TerrainFFT::FFTNoiseParams fftParams;
                        fftParams.seed = seed;
                        fftParams.scale = scale;
                        fftParams.frequency = frequency;
                        fftParams.octaves = octaves;
                        fftParams.persistence = persistance;
                        fftParams.lacunarity = lacunarity;
                        fftParams.ridgeOffset = ridge_offset;
                        
                        // Map noise type
                        switch (noiseType) {
                            case NoiseType::FFT_Ocean:
                                noiseValue = TerrainFFT::CPUNoise::fbmNoise(nx, ny, octaves, persistance, lacunarity, seed);
                                break;
                            case NoiseType::FFT_Ridge:
                                noiseValue = TerrainFFT::CPUNoise::ridgedNoise(nx, ny, octaves, persistance, lacunarity, ridge_offset, 2.0f, seed);
                                break;
                            case NoiseType::FFT_Billow:
                                noiseValue = TerrainFFT::CPUNoise::billowNoise(nx, ny, octaves, persistance, lacunarity, seed);
                                break;
                            case NoiseType::FFT_Turb:
                            default:
                                noiseValue = TerrainFFT::CPUNoise::fbmNoise(nx, ny, octaves, persistance, lacunarity, seed);
                                break;
                        }
                        break;
                    }
                }
                
                // Scale to amplitude
                (*result.data)[y * w + x] = noiseValue * amplitude;
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
    // SEDIMENT DEPOSITION NODE IMPLEMENTATIONS
    // ============================================================================
    
    NodeSystem::PinValue SedimentDepositionNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        auto inputHeight = getHeightInput(0, ctx);
        
        if (!tctx || !inputHeight.isValid()) {
            ctx.addError(id, "Invalid context or input");
            return NodeSystem::PinValue{};
        }
        
        int w = inputHeight.width;
        int h = inputHeight.height;
        
        // Get terrain's height scale
        float heightScale = (tctx->terrain) ? tctx->terrain->heightmap.scale_y : 1.0f;
        float cellSize = (tctx->terrain) ? (tctx->terrain->heightmap.scale_xz / (std::max)(w, h)) : 1.0f;
        
        // Working buffers
        std::vector<float> heightData = *inputHeight.data;
        std::vector<float> sediment(w * h, 0.0f);  // Sediment in transport
        std::vector<float> deposited(w * h, 0.0f); // Total deposited sediment
        
        // Optional flow mask input
        auto flowInput = getHeightInput(1, ctx);
        std::vector<float> flowMask(w * h, 1.0f);
        if (flowInput.isValid() && flowInput.data->size() == heightData.size()) {
            flowMask = *flowInput.data;
        }
        
        // D8 flow directions
        const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
        const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        const float dist[] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};
        
        // Flow-based sediment transport simulation
        for (int iter = 0; iter < iterations; iter++) {
            std::vector<float> newSediment(w * h, 0.0f);
            
            for (int y = 1; y < h - 1; y++) {
                for (int x = 1; x < w - 1; x++) {
                    int idx = y * w + x;
                    float centerH = heightData[idx] * heightScale;
                    float currentSed = sediment[idx];
                    float flow = flowMask[idx];
                    
                    // Calculate slope to steepest neighbor
                    float maxSlope = 0.0f;
                    int bestDir = -1;
                    float neighborH = centerH;
                    
                    for (int d = 0; d < 8; d++) {
                        int nx = x + dx[d];
                        int ny = y + dy[d];
                        int nidx = ny * w + nx;
                        
                        float nH = heightData[nidx] * heightScale;
                        float slope = (centerH - nH) / (dist[d] * cellSize);
                        
                        if (slope > maxSlope) {
                            maxSlope = slope;
                            bestDir = d;
                            neighborH = nH;
                        }
                    }
                    
                    // Transport capacity based on slope and flow
                    float capacity = maxSlope * transportCapacity * flow;
                    
                    if (maxSlope > 0.01f && bestDir >= 0) {
                        // Erosion (pick up sediment) on steep slopes
                        float erosion = (std::min)(maxSlope * 0.01f, 0.001f);
                        heightData[idx] -= erosion / heightScale;
                        currentSed += erosion;
                        
                        // Transport sediment downhill
                        int nx = x + dx[bestDir];
                        int ny = y + dy[bestDir];
                        int nidx = ny * w + nx;
                        
                        float transported = (std::min)(currentSed, capacity);
                        newSediment[nidx] += transported * (1.0f - settlingSpeed);
                        
                        // Deposit excess sediment
                        float excess = currentSed - transported;
                        if (excess > 0) {
                            deposited[idx] += excess * depositionRate;
                            heightData[idx] += excess * depositionRate / heightScale;
                        }
                    } else {
                        // Flat area: deposit all sediment
                        deposited[idx] += currentSed * depositionRate;
                        heightData[idx] += currentSed * depositionRate / heightScale;
                    }
                }
            }
            
            sediment = newSediment;
        }
        
        // Return appropriate output
        if (outputIndex == 0) {
            // Height output
            auto result = createHeightOutput(w, h);
            *result.data = heightData;
            return result;
        } else {
            // Sediment mask output
            auto result = createMaskOutput(w, h);
            // Normalize deposited map
            float maxDep = 0.001f;
            for (float d : deposited) maxDep = (std::max)(maxDep, d);
            for (int i = 0; i < w * h; i++) {
                (*result.data)[i] = clampValue(deposited[i] / maxDep, 0.0f, 1.0f);
            }
            return result;
        }
    }
    
    void SedimentDepositionNode::drawContent() {
        ImGui::SliderInt("Iterations", &iterations, 1, 50);
        ImGui::SliderFloat("Deposit Rate", &depositionRate, 0.1f, 1.0f);
        ImGui::SliderFloat("Transport Cap", &transportCapacity, 0.1f, 5.0f);
        ImGui::SliderFloat("Settling", &settlingSpeed, 0.1f, 0.9f);
    }
    
    NodeSystem::PinValue AlluvialFanNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        auto inputHeight = getHeightInput(0, ctx);
        
        if (!tctx || !inputHeight.isValid()) {
            ctx.addError(id, "Invalid context or input");
            return NodeSystem::PinValue{};
        }
        
        int w = inputHeight.width;
        int h = inputHeight.height;
        
        float heightScale = (tctx->terrain) ? tctx->terrain->heightmap.scale_y : 1.0f;
        float cellSize = (tctx->terrain) ? (tctx->terrain->heightmap.scale_xz / (std::max)(w, h)) : 1.0f;
        
        std::vector<float> heightData = *inputHeight.data;
        std::vector<float> fanMask(w * h, 0.0f);
        
        // Calculate slope map
        std::vector<float> slopeMap(w * h, 0.0f);
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                float dzdx = (heightData[idx + 1] - heightData[idx - 1]) * heightScale / (2.0f * cellSize);
                float dzdy = (heightData[idx + w] - heightData[idx - w]) * heightScale / (2.0f * cellSize);
                slopeMap[idx] = std::atan(std::sqrt(dzdx * dzdx + dzdy * dzdy)) * 57.2957795f;
            }
        }
        
        // Find steep-to-flat transition zones (fan apex points)
        float slopeThreshRad = slopeThreshold;
        
        for (int y = 2; y < h - 2; y++) {
            for (int x = 2; x < w - 2; x++) {
                int idx = y * w + x;
                float currentSlope = slopeMap[idx];
                
                // Check if this is a steep-to-flat transition
                bool isSteepAbove = false;
                bool isFlatBelow = false;
                
                // Check uphill neighbors (simplified: check north side)
                for (int dy = -2; dy <= 0; dy++) {
                    int nidx = (y + dy) * w + x;
                    if (slopeMap[nidx] > slopeThreshRad) isSteepAbove = true;
                }
                
                // Check downhill neighbors
                for (int dy = 1; dy <= 2; dy++) {
                    int nidx = (y + dy) * w + x;
                    if (slopeMap[nidx] < slopeThreshRad * 0.5f) isFlatBelow = true;
                }
                
                // If transition zone, create fan
                if (isSteepAbove && isFlatBelow && currentSlope < slopeThreshRad) {
                    float spreadRad = fanSpreadAngle * 3.14159f / 180.0f;
                    
                    // Spread sediment in fan pattern
                    for (int d = 0; d < fanLength; d++) {
                        float spread = (float)d / fanLength * spreadRad;
                        
                        for (float angle = -spread; angle <= spread; angle += 0.1f) {
                            int fx = x + (int)(std::sin(angle) * d);
                            int fy = y + d; // Fans spread downhill (south)
                            
                            if (fx >= 0 && fx < w && fy >= 0 && fy < h) {
                                int fidx = fy * w + fx;
                                float falloff = 1.0f - (float)d / fanLength;
                                falloff = falloff * falloff;
                                
                                float deposit = depositionStrength * falloff * 0.01f;
                                heightData[fidx] += deposit / heightScale;
                                fanMask[fidx] = (std::max)(fanMask[fidx], falloff);
                            }
                        }
                    }
                }
            }
        }
        
        if (outputIndex == 0) {
            auto result = createHeightOutput(w, h);
            *result.data = heightData;
            return result;
        } else {
            auto result = createMaskOutput(w, h);
            *result.data = fanMask;
            return result;
        }
    }
    
    void AlluvialFanNode::drawContent() {
        ImGui::SliderFloat("Slope Threshold", &slopeThreshold, 10.0f, 60.0f);
        ImGui::SliderFloat("Spread Angle", &fanSpreadAngle, 20.0f, 120.0f);
        ImGui::SliderFloat("Deposit Strength", &depositionStrength, 0.1f, 2.0f);
        ImGui::SliderInt("Fan Length", &fanLength, 10, 200);
    }
    
    NodeSystem::PinValue DeltaFormationNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        auto inputHeight = getHeightInput(0, ctx);
        
        if (!tctx || !inputHeight.isValid()) {
            ctx.addError(id, "Invalid context or input");
            return NodeSystem::PinValue{};
        }
        
        int w = inputHeight.width;
        int h = inputHeight.height;
        
        float heightScale = (tctx->terrain) ? tctx->terrain->heightmap.scale_y : 1.0f;
        
        std::vector<float> heightData = *inputHeight.data;
        std::vector<float> deltaMask(w * h, 0.0f);
        
        // Get flow mask for river detection
        auto flowInput = getHeightInput(1, ctx);
        std::vector<float> flowMask(w * h, 0.0f);
        if (flowInput.isValid() && flowInput.data->size() == heightData.size()) {
            flowMask = *flowInput.data;
        } else {
            // No flow input, skip delta formation
            if (outputIndex == 0) {
                auto result = createHeightOutput(w, h);
                *result.data = heightData;
                return result;
            } else {
                return createMaskOutput(w, h);
            }
        }
        
        // Find high-flow points at sea level (river mouths)
        float seaLevelThresh = seaLevel;
        float flowThreshold = 0.5f; // High flow accumulation
        
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                float height = heightData[idx];
                float flow = flowMask[idx];
                
                // Is this a river mouth? (high flow, low elevation)
                if (flow > flowThreshold && height < seaLevelThresh) {
                    float spreadRad = deltaSpread * 3.14159f / 180.0f;
                    
                    // Create branching delta pattern
                    for (int branch = 0; branch < branchingFactor; branch++) {
                        float branchAngle = -spreadRad + 2.0f * spreadRad * branch / (branchingFactor - 1);
                        if (branchingFactor == 1) branchAngle = 0;
                        
                        // Extend branch
                        float bx = (float)x;
                        float by = (float)y;
                        
                        for (int d = 0; d < 30; d++) {
                            bx += std::sin(branchAngle);
                            by += 1.0f; // Delta extends downward
                            
                            int fx = (int)bx;
                            int fy = (int)by;
                            
                            if (fx >= 0 && fx < w && fy >= 0 && fy < h) {
                                int fidx = fy * w + fx;
                                float falloff = 1.0f - (float)d / 30.0f;
                                
                                // Build up delta sediment
                                float deposit = sedimentRatio * falloff * flow * 0.01f;
                                heightData[fidx] += deposit / heightScale;
                                deltaMask[fidx] = (std::max)(deltaMask[fidx], falloff * flow);
                                
                                // Add some width to branches
                                for (int bw = -1; bw <= 1; bw++) {
                                    int wfx = fx + bw;
                                    if (wfx >= 0 && wfx < w) {
                                        int wfidx = fy * w + wfx;
                                        heightData[wfidx] += deposit * 0.5f / heightScale;
                                        deltaMask[wfidx] = (std::max)(deltaMask[wfidx], falloff * flow * 0.5f);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if (outputIndex == 0) {
            auto result = createHeightOutput(w, h);
            *result.data = heightData;
            return result;
        } else {
            auto result = createMaskOutput(w, h);
            // Normalize delta mask
            float maxVal = 0.001f;
            for (float v : deltaMask) maxVal = (std::max)(maxVal, v);
            for (int i = 0; i < w * h; i++) {
                (*result.data)[i] = clampValue(deltaMask[i] / maxVal, 0.0f, 1.0f);
            }
            return result;
        }
    }
    
    void DeltaFormationNode::drawContent() {
        ImGui::SliderFloat("Sea Level", &seaLevel, 0.0f, 0.5f);
        ImGui::SliderFloat("Delta Spread", &deltaSpread, 15.0f, 90.0f);
        ImGui::SliderInt("Branches", &branchingFactor, 1, 7);
        ImGui::SliderFloat("Sediment Ratio", &sedimentRatio, 0.1f, 1.0f);
    }

    // ============================================================================
    // EROSION WIZARD NODE IMPLEMENTATION
    // ============================================================================
    
    const char* ErosionWizardNode::getPresetName(ErosionPreset p) {
        switch (p) {
            case ErosionPreset::Custom: return "Custom";
            case ErosionPreset::YoungMountains: return "Young Mountains (1-10 My)";
            case ErosionPreset::MatureMountains: return "Mature Mountains (10-50 My)";
            case ErosionPreset::AncientPlateau: return "Ancient Plateau (100+ My)";
            case ErosionPreset::TropicalRainforest: return "Tropical Rainforest";
            case ErosionPreset::AridDesert: return "Arid Desert";
            case ErosionPreset::GlacialCarving: return "Glacial Carving";
            case ErosionPreset::CoastalErosion: return "Coastal Erosion";
            case ErosionPreset::VolcanicTerrain: return "Volcanic Terrain";
            case ErosionPreset::RiverDelta: return "River Delta";
            default: return "Unknown";
        }
    }
    
    void ErosionWizardNode::applyPreset(ErosionPreset p) {
        switch (p) {
            case ErosionPreset::YoungMountains:
                timeScaleMy = 5.0f;
                rainfallFactor = 1.0f;
                temperatureFactor = 0.8f;
                windFactor = 0.5f;
                break;
            case ErosionPreset::MatureMountains:
                timeScaleMy = 30.0f;
                rainfallFactor = 1.2f;
                temperatureFactor = 1.0f;
                windFactor = 0.7f;
                break;
            case ErosionPreset::AncientPlateau:
                timeScaleMy = 200.0f;
                rainfallFactor = 0.8f;
                temperatureFactor = 1.5f;
                windFactor = 1.0f;
                break;
            case ErosionPreset::TropicalRainforest:
                timeScaleMy = 20.0f;
                rainfallFactor = 2.0f;
                temperatureFactor = 0.5f;
                windFactor = 0.3f;
                break;
            case ErosionPreset::AridDesert:
                timeScaleMy = 50.0f;
                rainfallFactor = 0.1f;
                temperatureFactor = 2.0f;
                windFactor = 2.0f;
                break;
            case ErosionPreset::GlacialCarving:
                timeScaleMy = 1.0f;
                rainfallFactor = 0.5f;
                temperatureFactor = 2.0f;  // Freeze-thaw
                windFactor = 1.5f;
                break;
            case ErosionPreset::CoastalErosion:
                timeScaleMy = 10.0f;
                rainfallFactor = 1.5f;
                temperatureFactor = 0.8f;
                windFactor = 1.8f;
                break;
            case ErosionPreset::VolcanicTerrain:
                timeScaleMy = 0.5f;
                rainfallFactor = 1.2f;
                temperatureFactor = 1.5f;
                windFactor = 0.8f;
                break;
            case ErosionPreset::RiverDelta:
                timeScaleMy = 5.0f;
                rainfallFactor = 1.8f;
                temperatureFactor = 0.4f;
                windFactor = 0.2f;
                break;
            default:
                break;
        }
    }
    
    NodeSystem::PinValue ErosionWizardNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        auto inputHeight = getHeightInput(0, ctx);
        
        if (!tctx || !tctx->terrain || !inputHeight.isValid()) {
            ctx.addError(id, "Invalid context or input");
            return NodeSystem::PinValue{};
        }
        
        // Cache terrain for UI simulation updates
        cachedTerrain = tctx->terrain;
        
        int w = inputHeight.width;
        int h = inputHeight.height;
        TerrainObject* terrain = tctx->terrain;
        
        // Cache mask
        cachedMask.clear();
        auto maskInput = getHeightInput(1, ctx);
        if (maskInput.isValid() && maskInput.data->size() == (size_t)(w * h)) {
            cachedMask = *maskInput.data;
        }
        
        // 1. ALWAYS SYNC & RESET TERRAIN (Initialization)
        // Ensure dimensions match
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        
        // Reset terrain data from input (Start Fresh)
        terrain->heightmap.data = *inputHeight.data;
        
        // 2. AUTO-START SIMULATION
        isSimulating = true;
        currentPass = 0;
        
        // Setup Parameters
        float timeMultiplier = 1.0f + std::log10(1.0f + timeScaleMy) * timeScaleMy * 0.1f;
        int baseIterations = 5000 * qualityLevel;
        
        totalPasses = 1 + static_cast<int>(timeScaleMy / 50.0f);
        totalPasses = (std::min)(totalPasses, 20); // Limit max passes
        
        // Calculate iterations
        hydraulicItersPerPass = static_cast<int>(baseIterations * rainfallFactor * timeMultiplier / totalPasses);
        thermalItersPerPass = static_cast<int>(100 * qualityLevel * temperatureFactor * timeMultiplier / totalPasses);
        windItersPerPass = static_cast<int>(40 * qualityLevel * windFactor * timeMultiplier / totalPasses);
        
        // Clamp limits (Safety)
        hydraulicItersPerPass = (std::min)(hydraulicItersPerPass, 500000);
        thermalItersPerPass = (std::min)(thermalItersPerPass, 1000);
        windItersPerPass = (std::min)(windItersPerPass, 200);
        
        // Backup for mask calculation
        originalHeight = terrain->heightmap.data;
        
        SCENE_LOG_INFO("[ErosionWizard] Auto-started simulation: %d passes, %.1f My", totalPasses, timeScaleMy);
        
        // 3. RETURN INITIAL STATE (Passthrough)
        if (outputIndex == 0) {
            // Height output
            auto result = createHeightOutput(w, h);
            *result.data = terrain->heightmap.data;
            return result;
        } else {
            // Erosion mask (Empty initially)
            auto result = createMaskOutput(w, h);
            std::fill(result.data->begin(), result.data->end(), 0.0f);
            return result;
        }
    }
    
    void ErosionWizardNode::drawContent() {
        // Preset selector (Disable during sim)
        ImGui::BeginDisabled(isSimulating);
        if (ImGui::BeginCombo("Preset", getPresetName(preset))) {
            for (int i = 0; i <= (int)ErosionPreset::RiverDelta; i++) {
                ErosionPreset p = static_cast<ErosionPreset>(i);
                bool selected = (preset == p);
                if (ImGui::Selectable(getPresetName(p), selected)) {
                    preset = p;
                    if (p != ErosionPreset::Custom) applyPreset(p);
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
        ImGui::Separator();
        ImGui::SliderFloat("Time (My)", &timeScaleMy, 0.1f, 500.0f, "%.1f My");
        ImGui::Text("Climate:");
        ImGui::SliderFloat("Rainfall", &rainfallFactor, 0.0f, 2.0f);
        ImGui::SliderFloat("Temperature", &temperatureFactor, 0.0f, 2.0f);
        ImGui::SliderFloat("Wind", &windFactor, 0.0f, 2.0f);
        ImGui::SliderInt("Quality", &qualityLevel, 1, 3);
        ImGui::Checkbox("Use GPU", &useGPU);
        ImGui::EndDisabled();

        ImGui::Separator();
        
        // STATUS & CONTROLS
        if (isSimulating) {
            // Header
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "Simulating...");
            ImGui::SameLine();
            if (ImGui::Button("Stop")) {
                isSimulating = false;
                SCENE_LOG_INFO("[ErosionWizard] Simulation stopped by user.");
            }
            
            // Progress bar
            float progress = (totalPasses > 0) ? (float)currentPass / (float)totalPasses : 0.0f;
            char overlay[32];
            sprintf(overlay, "Pass %d/%d", currentPass, totalPasses);
            ImGui::ProgressBar(progress, ImVec2(-1, 0), overlay);
            
            // EXECUTE ONE PASS PER FRAME
            if (cachedTerrain && currentPass < totalPasses) {
                TerrainManager& mgr = TerrainManager::getInstance();
                
                // 1. Thermal
                if (temperatureFactor > 0.01f && thermalItersPerPass > 0) {
                    ThermalErosionParams tp;
                    tp.iterations = thermalItersPerPass;
                    tp.talusAngle = 25.0f - temperatureFactor * 5.0f;
                    tp.erosionAmount = 0.5f * temperatureFactor;
                    if (useGPU) mgr.thermalErosionGPU(cachedTerrain, tp, cachedMask);
                    else mgr.thermalErosion(cachedTerrain, tp, cachedMask);
                }
                
                // 2. Hydraulic
                if (rainfallFactor > 0.01f && hydraulicItersPerPass > 0) {
                    HydraulicErosionParams hp;
                    hp.iterations = hydraulicItersPerPass;
                    hp.erosionRadius = 3 + static_cast<int>(rainfallFactor * 2);
                    hp.depositSpeed = 0.1f;
                    hp.sedimentCapacity = 8.0f * rainfallFactor;
                    hp.evaporateSpeed = 0.01f;
                    hp.erodeSpeed = 0.5f * rainfallFactor;
                    if (useGPU) mgr.hydraulicErosionGPU(cachedTerrain, hp, cachedMask);
                    else mgr.hydraulicErosion(cachedTerrain, hp, cachedMask);
                }
                
                // 3. Wind
                if (windFactor > 0.01f && windItersPerPass > 0) {
                     if (useGPU) mgr.windErosionGPU(cachedTerrain, windFactor * 0.8f, 45.0f, windItersPerPass, cachedMask);
                     else mgr.windErosion(cachedTerrain, windFactor * 0.8f, 45.0f, windItersPerPass, cachedMask);
                }
                
                // Update Mesh & Viewport
                mgr.updateTerrainMesh(cachedTerrain);
                extern bool g_bvh_rebuild_pending;
                extern bool g_optix_rebuild_pending;
                g_bvh_rebuild_pending = true;
                g_optix_rebuild_pending = true;
                
                currentPass++;
            } else {
                isSimulating = false;
                SCENE_LOG_INFO("[ErosionWizard] Simulation complete!");
            }
        } else {
            ImGui::TextDisabled("Status: Ready (Press Evaluate)");
        }
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
                        int bx = (std::min)((int)(x * scaleX), inputB.width - 1);
                        int by = (std::min)((int)(y * scaleY), inputB.height - 1);
                        b = (*inputB.data)[by * inputB.width + bx];
                    }
                }
                
                switch (operation) {
                    case MathOp::Add: (*result.data)[idx] = a + b; break;
                    case MathOp::Subtract: (*result.data)[idx] = a - b; break;
                    case MathOp::Multiply: (*result.data)[idx] = a * b; break;
                    case MathOp::Divide: (*result.data)[idx] = (b != 0) ? a / b : 0; break;
                    case MathOp::Min: (*result.data)[idx] = (std::min)(a, b); break;
                    case MathOp::Max: (*result.data)[idx] = (std::max)(a, b); break;
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
            minH = (std::min)(minH, h);
            maxH = (std::max)(maxH, h);
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
        
        // Get terrain scales for proper gradient calculation
        float cellSize = tctx->terrain ? (tctx->terrain->heightmap.scale_xz / (std::max)(w, h)) : 1.0f;
        float heightScale = tctx->terrain ? tctx->terrain->heightmap.scale_y : 1.0f;
        
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                
                // Scale heights from normalized (0-1) to physical units for proper gradient
                float h_right = (*input.data)[idx + 1] * heightScale;
                float h_left = (*input.data)[idx - 1] * heightScale;
                float h_down = (*input.data)[idx + w] * heightScale;
                float h_up = (*input.data)[idx - w] * heightScale;
                
                float dzdx = (h_right - h_left) / (2.0f * cellSize);
                float dzdy = (h_down - h_up) / (2.0f * cellSize);
                float slope = std::atan(std::sqrt(dzdx * dzdx + dzdy * dzdy)) * 57.2957795f; // rad to deg
                
                float t = 0.0f;
                if (slope >= minSlope && slope <= maxSlope) {
                    t = 1.0f;
                } else if (slope < minSlope) {
                    t = (std::max)(0.0f, 1.0f - (minSlope - slope) / (falloff * (maxSlope - minSlope) + 0.001f));
                } else {
                    t = (std::max)(0.0f, 1.0f - (slope - maxSlope) / (falloff * (maxSlope - minSlope) + 0.001f));
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
        auto* tctx = getTerrainContext(ctx);
        auto input = getHeightInput(0, ctx);
        
        if (!input.isValid()) {
            ctx.addError(id, "Invalid input");
            return NodeSystem::PinValue{};
        }
        
        // Get height scale to convert normalized heights to physical units
        float heightScale = (tctx && tctx->terrain) ? tctx->terrain->heightmap.scale_y : 1.0f;
        
        auto result = createMaskOutput(input.width, input.height);
        
        for (size_t i = 0; i < input.data->size(); i++) {
            // Convert normalized height (0-1) to physical height 
            float h = (*input.data)[i] * heightScale;
            float t = 0.0f;
            
            if (h >= minHeight && h <= maxHeight) {
                t = 1.0f;
            } else if (h < minHeight) {
                t = (std::max)(0.0f, 1.0f - (minHeight - h) / (falloff + 0.001f));
            } else {
                t = (std::max)(0.0f, 1.0f - (h - maxHeight) / (falloff + 0.001f));
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
    // FLOW AND EXPOSURE MASK IMPLEMENTATIONS
    // ============================================================================
    
    // FLOW MASK - Soil/sediment accumulation
    NodeSystem::PinValue FlowMaskNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        auto input = getHeightInput(0, ctx);
        
        if (!tctx || !input.isValid()) {
            ctx.addError(id, "Invalid context or input");
            return NodeSystem::PinValue{};
        }
        
        int w = input.width;
        int h = input.height;
        auto result = createMaskOutput(w, h);
        
        // Get height scale for proper slope detection
        float heightScale = (tctx && tctx->terrain) ? tctx->terrain->heightmap.scale_y : 1.0f;
        
        // Flow accumulation buffer
        std::vector<float> flow(w * h, 1.0f);  // Start with uniform "rain"
        
        // D8 flow direction offsets
        const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
        const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        // D8 distance weights (diagonals are sqrt(2) apart)
        const float dist[] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};
        
        // Multiple iterations for flow propagation
        for (int iter = 0; iter < iterations; iter++) {
            std::vector<float> newFlow(w * h, 0.0f);
            
            for (int y = 1; y < h - 1; y++) {
                for (int x = 1; x < w - 1; x++) {
                    int idx = y * w + x;
                    float centerH = (*input.data)[idx] * heightScale;
                    float currentFlow = flow[idx];
                    
                    // Find steepest downhill neighbor
                    float maxSlope = 0.0f;
                    int bestDir = -1;
                    
                    for (int d = 0; d < 8; d++) {
                        int nx = x + dx[d];
                        int ny = y + dy[d];
                        int nidx = ny * w + nx;
                        
                        float neighborH = (*input.data)[nidx] * heightScale;
                        float slope = (centerH - neighborH) / dist[d];
                        
                        if (slope > maxSlope) {
                            maxSlope = slope;
                            bestDir = d;
                        }
                    }
                    
                    // Flow to steepest downhill neighbor
                    if (bestDir >= 0 && maxSlope > 0.0001f) {
                        int nx = x + dx[bestDir];
                        int ny = y + dy[bestDir];
                        int nidx = ny * w + nx;
                        newFlow[nidx] += currentFlow * strength * decay;
                    }
                    
                    // Keep some flow at current position
                    newFlow[idx] += currentFlow * (1.0f - decay);
                }
            }
            
            flow = newFlow;
        }
        
        // Find max for normalization
        float maxFlow = 0.001f;
        if (normalize) {
            for (int i = 0; i < w * h; i++) {
                if (flow[i] > maxFlow) maxFlow = flow[i];
            }
        } else {
            maxFlow = 1.0f;
        }
        
        // Write to output
        for (int i = 0; i < w * h; i++) {
            (*result.data)[i] = clampValue(flow[i] / maxFlow, 0.0f, 1.0f);
        }
        
        return result;
    }
    
    void FlowMaskNode::drawContent() {
        ImGui::SetNextItemWidth(80);
        ImGui::SliderInt("Iterations", &iterations, 1, 32);
        ImGui::SetNextItemWidth(80);
        ImGui::SliderFloat("Strength", &strength, 0.1f, 2.0f);
        ImGui::SetNextItemWidth(80);
        ImGui::SliderFloat("Decay", &decay, 0.5f, 0.99f);
        ImGui::Checkbox("Normalize", &normalize);
    }
    
    // EXPOSURE MASK - Sun-facing direction
    NodeSystem::PinValue ExposureMaskNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
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
        
        // Calculate sun direction vector
        float azimuthRad = sunAzimuth * 3.14159f / 180.0f;
        float elevationRad = sunElevation * 3.14159f / 180.0f;
        
        // Sun direction (pointing TO the sun)
        float sunX = std::sin(azimuthRad) * std::cos(elevationRad);
        float sunY = std::sin(elevationRad);
        float sunZ = std::cos(azimuthRad) * std::cos(elevationRad);
        
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                
                // Calculate surface normal from height gradient
                float dzdx = ((*input.data)[idx + 1] - (*input.data)[idx - 1]) / (2.0f * cellSize);
                float dzdy = ((*input.data)[idx + w] - (*input.data)[idx - w]) / (2.0f * cellSize);
                
                // Normal vector (unnormalized Y is up)
                float nx = -dzdx;
                float ny = 1.0f;
                float nz = -dzdy;
                float nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (nlen > 0.001f) { nx /= nlen; ny /= nlen; nz /= nlen; }
                
                // Dot product with sun direction
                float exposure = nx * sunX + ny * sunY + nz * sunZ;
                
                // Apply contrast
                exposure = (exposure - 0.5f) * contrast + 0.5f;
                exposure = clampValue(exposure, 0.0f, 1.0f);
                
                if (invert) exposure = 1.0f - exposure;
                
                (*result.data)[idx] = exposure;
            }
        }
        
        // Fill borders
        for (int x = 0; x < w; x++) {
            (*result.data)[x] = (*result.data)[w + x];
            (*result.data)[(h-1) * w + x] = (*result.data)[(h-2) * w + x];
        }
        for (int y = 0; y < h; y++) {
            (*result.data)[y * w] = (*result.data)[y * w + 1];
            (*result.data)[y * w + w - 1] = (*result.data)[y * w + w - 2];
        }
        
        return result;
    }
    
    void ExposureMaskNode::drawContent() {
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Azimuth", &sunAzimuth, 0.0f, 360.0f, "%.0f°");
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Elevation", &sunElevation, 0.0f, 90.0f, "%.0f°");
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Contrast", &contrast, 0.1f, 3.0f);
        ImGui::Checkbox("Invert (Shadow)", &invert);
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
            minH = (std::min)(minH, h);
            maxH = (std::max)(maxH, h);
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
            minH = (std::min)(minH, hVal);
            maxH = (std::max)(maxH, hVal);
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
                    out = (std::min)(a, b);
                    break;
                case MaskCombineOp::OR:
                    out = (std::max)(a, b);
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
            minB = (std::min)(minB, (*inputBase.data)[i]);
            maxB = (std::max)(maxB, (*inputBase.data)[i]);
            minL = (std::min)(minL, (*inputBlend.data)[i]);
            maxL = (std::max)(maxL, (*inputBlend.data)[i]);
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
            minB = (std::min)(minB, (*inputBase.data)[i]);
            maxB = (std::max)(maxB, (*inputBase.data)[i]);
            minL = (std::min)(minL, (*inputBlend.data)[i]);
            maxL = (std::max)(maxL, (*inputBlend.data)[i]);
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
    // GEOLOGICAL TRANSFORM NODES IMPLEMENTATION
    // ============================================================================
    
    // ------------------------------------------------------------------------
    // FAULT NODE - Strike-slip fault line
    // ------------------------------------------------------------------------
    NodeSystem::PinValue FaultNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        auto input = getHeightInput(0, ctx);
        
        if (!tctx || !input.isValid()) {
            ctx.addError(id, "Invalid input or context");
            return NodeSystem::PinValue{};
        }
        
        int w = input.width;
        int h = input.height;
        auto result = createHeightOutput(w, h);
        
        // Get optional mask (use getHeightInput for mask too)
        auto maskData = getHeightInput(1, ctx);
        bool hasMask = maskData.isValid();
        
        float dirRad = direction * 3.14159f / 180.0f;
        float cosD = std::cos(dirRad);
        float sinD = std::sin(dirRad);
        
        // Fault line position in grid coordinates
        float faultPos = position * (std::max)(w, h);
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                
                // Project point onto fault line direction
                float px = x - w * 0.5f;
                float py = y - h * 0.5f;
                float perpDist = px * sinD - py * cosD + faultPos;
                
                // Calculate offset based on which side of fault
                float t = 0.0f;
                if (width > 0.01f) {
                    t = perpDist / width;
                    t = clampValue(t, -1.0f, 1.0f);
                    // Smooth transition using smoothstep
                    t = t * 0.5f + 0.5f; // Map to 0-1
                    t = t * t * (3.0f - 2.0f * t); // Smoothstep
                } else {
                    t = perpDist > 0 ? 1.0f : 0.0f;
                }
                
                // Calculate source coordinates with offset
                float offsetAmount = offset * (t - 0.5f) * 2.0f;
                int srcX = x + (int)(offsetAmount * cosD);
                int srcY = y + (int)(offsetAmount * sinD);
                
                // Clamp source coordinates
                srcX = clampValue(srcX, 0, w - 1);
                srcY = clampValue(srcY, 0, h - 1);
                
                float srcHeight = (*input.data)[srcY * w + srcX];
                float vOffset = verticalOffset * (t - 0.5f) * 2.0f;
                
                float finalHeight = srcHeight + vOffset;
                
                // Apply mask
                if (hasMask) {
                    float mask = (*maskData.data)[idx];
                    finalHeight = (*input.data)[idx] * (1.0f - mask) + finalHeight * mask;
                }
                
                (*result.data)[idx] = finalHeight;
            }
        }
        
        return result;
    }
    
    void FaultNode::drawContent() {
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Direction", &direction, 0.0f, 360.0f, "%.0f°");
        ImGui::SetNextItemWidth(100);
        ImGui::DragFloat("Offset", &offset, 0.5f, -50.0f, 50.0f);
        ImGui::SetNextItemWidth(100);
        ImGui::DragFloat("V.Offset", &verticalOffset, 0.1f, -10.0f, 10.0f);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Position", &position, 0.0f, 1.0f);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Width", &width, 0.1f, 20.0f);
    }
    
    // ------------------------------------------------------------------------
    // MESA NODE - Flat-topped plateau
    // ------------------------------------------------------------------------
    NodeSystem::PinValue MesaNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        auto input = getHeightInput(0, ctx);
        
        if (!tctx || !input.isValid()) {
            ctx.addError(id, "Invalid input or context");
            return NodeSystem::PinValue{};
        }
        
        int w = input.width;
        int h = input.height;
        auto result = createHeightOutput(w, h);
        
        // Get optional mask (use getHeightInput for mask too)
        auto maskData = getHeightInput(1, ctx);
        bool hasMask = maskData.isValid();
        
        // Find height range for threshold calculation
        float minH = 1e9f, maxH = -1e9f;
        for (int i = 0; i < w * h; i++) {
            float v = (*input.data)[i];
            if (v < minH) minH = v;
            if (v > maxH) maxH = v;
        }
        float heightRange = maxH - minH;
        if (heightRange < 0.001f) heightRange = 1.0f;
        
        float absThreshold = minH + threshold * heightRange;
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                float srcHeight = (*input.data)[idx];
                
                float finalHeight = srcHeight;
                
                if (srcHeight > absThreshold) {
                    // Calculate distance from threshold
                    float above = (srcHeight - absThreshold) / heightRange;
                    
                    // Apply terrace levels
                    if (terraceCount > 1) {
                        float step = 1.0f / terraceCount;
                        above = std::floor(above / step) * step;
                    }
                    
                    // Smoothstep for cliff transition
                    float cliff = above / (1.0f - threshold + 0.001f);
                    cliff = clampValue(cliff, 0.0f, 1.0f);
                    
                    // Cliff steepness control
                    float steepCliff = std::pow(cliff, 1.0f / (1.0f - cliffSteepness * 0.9f + 0.1f));
                    
                    // Mesa top is flat at plateau height
                    float mesaTop = absThreshold + plateauHeight * heightRange * 0.3f;
                    
                    // Blend between original slope and flat top
                    finalHeight = absThreshold + (mesaTop - absThreshold) * steepCliff;
                }
                
                // Apply mask
                if (hasMask) {
                    float mask = (*maskData.data)[idx];
                    finalHeight = srcHeight * (1.0f - mask) + finalHeight * mask;
                }
                
                (*result.data)[idx] = finalHeight;
            }
        }
        
        return result;
    }
    
    void MesaNode::drawContent() {
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Threshold", &threshold, 0.0f, 1.0f);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Cliff", &cliffSteepness, 0.0f, 1.0f);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Height", &plateauHeight, 0.1f, 2.0f);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderInt("Terraces", &terraceCount, 1, 10);
    }
    
    // ------------------------------------------------------------------------
    // SHEAR NODE - Diagonal deformation
    // ------------------------------------------------------------------------
    NodeSystem::PinValue ShearNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        auto input = getHeightInput(0, ctx);
        
        if (!tctx || !input.isValid()) {
            ctx.addError(id, "Invalid input or context");
            return NodeSystem::PinValue{};
        }
        
        int w = input.width;
        int h = input.height;
        auto result = createHeightOutput(w, h);
        
        // Get optional mask (use getHeightInput for mask too)
        auto maskData = getHeightInput(1, ctx);
        bool hasMask = maskData.isValid();
        
        float angleRad = angle * 3.14159f / 180.0f;
        float tanA = std::tan(angleRad);
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                
                // Normalize coordinates to 0-1
                float nx = (float)x / w;
                float ny = (float)y / h;
                
                // Determine which band this pixel is in
                float bandProgress = ny * bands;
                int bandIndex = (int)bandProgress;
                float bandLocal = bandProgress - bandIndex;
                
                // Shear direction alternates if bidirectional
                float shearDir = 1.0f;
                if (bidirectional && (bandIndex % 2 == 1)) {
                    shearDir = -1.0f;
                }
                
                // Calculate shear amount within band
                float shearAmount = 0.0f;
                if (bandLocal < bandWidth) {
                    // Inside shear zone
                    float t = bandLocal / bandWidth;
                    shearAmount = std::sin(t * 3.14159f) * strength * shearDir;
                }
                
                // Calculate source coordinates
                float srcX = x + shearAmount * w * tanA;
                
                // Bilinear interpolation for smooth sampling
                int x0 = (int)std::floor(srcX);
                int x1 = x0 + 1;
                float tx = srcX - x0;
                
                x0 = clampValue(x0, 0, w - 1);
                x1 = clampValue(x1, 0, w - 1);
                
                float h0 = (*input.data)[y * w + x0];
                float h1 = (*input.data)[y * w + x1];
                float finalHeight = h0 * (1.0f - tx) + h1 * tx;
                
                // Apply mask
                if (hasMask) {
                    float mask = (*maskData.data)[idx];
                    finalHeight = (*input.data)[idx] * (1.0f - mask) + finalHeight * mask;
                }
                
                (*result.data)[idx] = finalHeight;
            }
        }
        
        return result;
    }
    
    void ShearNode::drawContent() {
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Angle", &angle, -60.0f, 60.0f, "%.0f°");
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Strength", &strength, 0.0f, 1.0f);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderInt("Bands", &bands, 1, 16);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Width", &bandWidth, 0.05f, 0.5f);
        ImGui::Checkbox("Bidirectional", &bidirectional);
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
            case NodeType::FlowMask: node = addNode<FlowMaskNode>(); break;
            case NodeType::ExposureMask: node = addNode<ExposureMaskNode>(); break;
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
            // NEW: Geological Transform Nodes
            case NodeType::Fault: node = addNode<FaultNode>(); break;
            case NodeType::Mesa: node = addNode<MesaNode>(); break;
            case NodeType::Shear: node = addNode<ShearNode>(); break;
            // Stacks and Anastomosing - TODO: implement later
            case NodeType::Stacks: break;
            case NodeType::Anastomosing: break;
            // NEW: Sediment Deposition Nodes
            case NodeType::SedimentDeposition: node = addNode<SedimentDepositionNode>(); break;
            case NodeType::AlluvialFan: node = addNode<AlluvialFanNode>(); break;
            case NodeType::DeltaFormation: node = addNode<DeltaFormationNode>(); break;
            // NEW: Erosion Wizard
            case NodeType::ErosionWizard: node = addNode<ErosionWizardNode>(); break;
        }
        
        if (node) {
            node->x = x;
            node->y = y;
        }
        
        return node;
    }
    
    void TerrainNodeGraphV2::evaluateTerrain(TerrainObject* terrain, SceneData& scene) {
        if (!terrain) return;
        
        // Use the proper constructor that sets width/height from terrain
        TerrainContext tctx(terrain);
        
        NodeSystem::EvaluationContext ctx(this);
        ctx.setDomainContext(&tctx);
        
        // CRITICAL FIX: Clear cache and mark all nodes dirty before evaluation
        // Without this, intermediate nodes (deformation nodes) may be skipped
        // because their cached values would be reused instead of recomputing
        ctx.clearCache();
        ctx.clearErrors();
        markAllDirty();

        // Find Height Output Node and pull data
        HeightOutputNode* outputNode = nullptr;
        for (auto& node : nodes) {
            if (node->getTypeId() == "TerrainV2.HeightOutput") {
                outputNode = dynamic_cast<HeightOutputNode*>(node.get());
                break;
            }
        }
        
        if (!outputNode) {
            // No output node found - nothing to evaluate
            return;
        }
        
        // Pull data from the input of the output node (Input index 0 for Height)
        // This triggers the pull-based evaluation chain through ALL connected nodes
        auto heightData = outputNode->getHeightInput(0, ctx);
        
        // Check for errors in the evaluation chain
        if (ctx.hasErrors()) {
            for (const auto& err : ctx.getErrors()) {
                // Log detailed errors
                SCENE_LOG_ERROR("Terrain Graph Error (Node " + std::to_string(err.nodeId) + "): " + err.message);
            }
        }
        
        // CRITICAL CHECK: Verify input data integrity
        if (!heightData.isValid() || !heightData.data || heightData.width < 2 || heightData.height < 2) {
            // If data is invalid or too small (e.g. uninitialized), do not update terrain
            // This prevents "scrambled" artifacts during initialization
            return;
        }
        
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

    
    // ============================================================================
    // TERRAIN NODE GRAPH SERIALIZATION
    // ============================================================================
    
    nlohmann::json TerrainNodeGraphV2::toJson() const {
        nlohmann::json j;
        
        // Save ID generators for proper restoration
        j["nextNodeId"] = nextNodeId;
        j["nextPinId"] = nextPinId;
        j["nextLinkId"] = nextLinkId;
        j["nextGroupId"] = nextGroupId;
        
        // Save nodes
        nlohmann::json nodesArray = nlohmann::json::array();
        for (const auto& nodePtr : nodes) {
            nlohmann::json nodeJson;
            
            // Save base node data
            nodeJson["id"] = nodePtr->id;
            
            // Save pin IDs for link restoration
            nlohmann::json inputPins = nlohmann::json::array();
            for (const auto& pin : nodePtr->inputs) {
                nlohmann::json pinJson;
                pinJson["id"] = pin.id;
                pinJson["name"] = pin.name;
                inputPins.push_back(pinJson);
            }
            nodeJson["inputPins"] = inputPins;
            
            nlohmann::json outputPins = nlohmann::json::array();
            for (const auto& pin : nodePtr->outputs) {
                nlohmann::json pinJson;
                pinJson["id"] = pin.id;
                pinJson["name"] = pin.name;
                outputPins.push_back(pinJson);
            }
            nodeJson["outputPins"] = outputPins;
            
            // Save node-specific data via virtual method
            if (auto* terrainNode = dynamic_cast<TerrainNodeBase*>(nodePtr.get())) {
                terrainNode->serializeToJson(nodeJson);
            }
            
            nodesArray.push_back(nodeJson);
        }
        j["nodes"] = nodesArray;
        
        // Save links
        nlohmann::json linksArray = nlohmann::json::array();
        for (const auto& link : links) {
            nlohmann::json linkJson;
            linkJson["id"] = link.id;
            linkJson["startPinId"] = link.startPinId;
            linkJson["endPinId"] = link.endPinId;
            linksArray.push_back(linkJson);
        }
        j["links"] = linksArray;
        
        // Save groups (optional)
        nlohmann::json groupsArray = nlohmann::json::array();
        for (const auto& group : groups) {
            nlohmann::json groupJson;
            groupJson["id"] = group.id;
            groupJson["name"] = group.name;
            groupJson["position"] = { group.position.x, group.position.y };
            groupJson["size"] = { group.size.x, group.size.y };
            groupJson["nodeIds"] = group.nodeIds;
            groupsArray.push_back(groupJson);
        }
        j["groups"] = groupsArray;
        
        return j;
    }
    
    void TerrainNodeGraphV2::fromJson(const nlohmann::json& j, TerrainObject* terrain) {
        // Clear existing graph
        clear();
        
        // Restore ID generators
        if (j.contains("nextNodeId")) nextNodeId = j["nextNodeId"].get<uint32_t>();
        if (j.contains("nextPinId")) nextPinId = j["nextPinId"].get<uint32_t>();
        if (j.contains("nextLinkId")) nextLinkId = j["nextLinkId"].get<uint32_t>();
        if (j.contains("nextGroupId")) nextGroupId = j["nextGroupId"].get<uint32_t>();
        
        // ID mapping for proper link restoration
        std::unordered_map<uint32_t, uint32_t> oldToNewPinId;
        std::unordered_map<uint32_t, uint32_t> oldToNewNodeId;
        
        // Load nodes
        if (j.contains("nodes") && j["nodes"].is_array()) {
            for (const auto& nodeJson : j["nodes"]) {
                std::string typeId = nodeJson.value("typeId", "");
                uint32_t oldNodeId = nodeJson.value("id", 0u);
                float x = nodeJson.value("x", 0.0f);
                float y = nodeJson.value("y", 0.0f);
                
                // Create node by type ID
                NodeSystem::NodeBase* newNode = nullptr;
                
                if (typeId == "TerrainV2.HeightmapInput") {
                    newNode = addTerrainNode(NodeType::HeightmapInput, x, y);
                } else if (typeId == "TerrainV2.NoiseGenerator") {
                    newNode = addTerrainNode(NodeType::NoiseGenerator, x, y);
                } else if (typeId == "TerrainV2.HydraulicErosion") {
                    newNode = addTerrainNode(NodeType::HydraulicErosion, x, y);
                } else if (typeId == "TerrainV2.ThermalErosion") {
                    newNode = addTerrainNode(NodeType::ThermalErosion, x, y);
                } else if (typeId == "TerrainV2.FluvialErosion") {
                    newNode = addTerrainNode(NodeType::FluvialErosion, x, y);
                } else if (typeId == "TerrainV2.WindErosion") {
                    newNode = addTerrainNode(NodeType::WindErosion, x, y);
                } else if (typeId == "TerrainV2.HeightOutput") {
                    newNode = addTerrainNode(NodeType::HeightOutput, x, y);
                } else if (typeId == "TerrainV2.SplatOutput") {
                    newNode = addTerrainNode(NodeType::SplatOutput, x, y);
                } else if (typeId == "TerrainV2.Math") {
                    newNode = addTerrainNode(NodeType::Add, x, y);
                } else if (typeId == "TerrainV2.Blend") {
                    newNode = addTerrainNode(NodeType::Blend, x, y);
                } else if (typeId == "TerrainV2.Clamp") {
                    newNode = addTerrainNode(NodeType::Clamp, x, y);
                } else if (typeId == "TerrainV2.Invert") {
                    newNode = addTerrainNode(NodeType::Invert, x, y);
                } else if (typeId == "TerrainV2.SlopeMask") {
                    newNode = addTerrainNode(NodeType::SlopeMask, x, y);
                } else if (typeId == "TerrainV2.HeightMask") {
                    newNode = addTerrainNode(NodeType::HeightMask, x, y);
                } else if (typeId == "TerrainV2.CurvatureMask") {
                    newNode = addTerrainNode(NodeType::CurvatureMask, x, y);
                } else if (typeId == "TerrainV2.FlowMask") {
                    newNode = addTerrainNode(NodeType::FlowMask, x, y);
                } else if (typeId == "TerrainV2.ExposureMask") {
                    newNode = addTerrainNode(NodeType::ExposureMask, x, y);
                } else if (typeId == "TerrainV2.Smooth") {
                    newNode = addTerrainNode(NodeType::Smooth, x, y);
                } else if (typeId == "TerrainV2.Normalize") {
                    newNode = addTerrainNode(NodeType::Normalize, x, y);
                } else if (typeId == "TerrainV2.Terrace") {
                    newNode = addTerrainNode(NodeType::Terrace, x, y);
                } else if (typeId == "TerrainV2.MaskCombine") {
                    newNode = addTerrainNode(NodeType::MaskCombine, x, y);
                } else if (typeId == "TerrainV2.Overlay") {
                    newNode = addTerrainNode(NodeType::Overlay, x, y);
                } else if (typeId == "TerrainV2.Screen") {
                    newNode = addTerrainNode(NodeType::Screen, x, y);
                } else if (typeId == "TerrainV2.AutoSplat") {
                    newNode = addTerrainNode(NodeType::AutoSplat, x, y);
                } else if (typeId == "TerrainV2.MaskPaint") {
                    newNode = addTerrainNode(NodeType::MaskPaint, x, y);
                } else if (typeId == "TerrainV2.MaskImage") {
                    newNode = addTerrainNode(NodeType::MaskImage, x, y);
                } else if (typeId == "TerrainV2.Fault") {
                    newNode = addTerrainNode(NodeType::Fault, x, y);
                } else if (typeId == "TerrainV2.Mesa") {
                    newNode = addTerrainNode(NodeType::Mesa, x, y);
                } else if (typeId == "TerrainV2.Shear") {
                    newNode = addTerrainNode(NodeType::Shear, x, y);
                } else if (typeId == "TerrainV2.SedimentDeposition") {
                    newNode = addTerrainNode(NodeType::SedimentDeposition, x, y);
                } else if (typeId == "TerrainV2.AlluvialFan") {
                    newNode = addTerrainNode(NodeType::AlluvialFan, x, y);
                } else if (typeId == "TerrainV2.DeltaFormation") {
                    newNode = addTerrainNode(NodeType::DeltaFormation, x, y);
                } else if (typeId == "TerrainV2.ErosionWizard") {
                    newNode = addTerrainNode(NodeType::ErosionWizard, x, y);
                }
                
                if (newNode) {
                    // Store ID mapping
                    oldToNewNodeId[oldNodeId] = newNode->id;
                    
                    // Map old pin IDs to new pin IDs
                    if (nodeJson.contains("inputPins")) {
                        size_t idx = 0;
                        for (const auto& pinJson : nodeJson["inputPins"]) {
                            if (idx < newNode->inputs.size()) {
                                uint32_t oldPinId = pinJson.value("id", 0u);
                                oldToNewPinId[oldPinId] = newNode->inputs[idx].id;
                            }
                            idx++;
                        }
                    }
                    if (nodeJson.contains("outputPins")) {
                        size_t idx = 0;
                        for (const auto& pinJson : nodeJson["outputPins"]) {
                            if (idx < newNode->outputs.size()) {
                                uint32_t oldPinId = pinJson.value("id", 0u);
                                oldToNewPinId[oldPinId] = newNode->outputs[idx].id;
                            }
                            idx++;
                        }
                    }
                    
                    // Deserialize node-specific data
                    if (auto* terrainNode = dynamic_cast<TerrainNodeBase*>(newNode)) {
                        terrainNode->deserializeFromJson(nodeJson);
                    }
                }
            }
        }
        
        // Restore links using ID mappings
        if (j.contains("links") && j["links"].is_array()) {
            for (const auto& linkJson : j["links"]) {
                uint32_t oldStartPin = linkJson.value("startPinId", 0u);
                uint32_t oldEndPin = linkJson.value("endPinId", 0u);
                
                auto startIt = oldToNewPinId.find(oldStartPin);
                auto endIt = oldToNewPinId.find(oldEndPin);
                
                if (startIt != oldToNewPinId.end() && endIt != oldToNewPinId.end()) {
                    addLink(startIt->second, endIt->second);
                }
            }
        }
        
        // Restore groups
        if (j.contains("groups") && j["groups"].is_array()) {
            for (const auto& groupJson : j["groups"]) {
                std::string name = groupJson.value("name", "Group");
                ImVec2 pos(0, 0), size(200, 150);
                if (groupJson.contains("position") && groupJson["position"].is_array()) {
                    pos.x = groupJson["position"][0].get<float>();
                    pos.y = groupJson["position"][1].get<float>();
                }
                if (groupJson.contains("size") && groupJson["size"].is_array()) {
                    size.x = groupJson["size"][0].get<float>();
                    size.y = groupJson["size"][1].get<float>();
                }
                
                uint32_t groupId = createGroup(name, pos, size);
                
                if (groupJson.contains("nodeIds")) {
                    for (uint32_t oldNodeId : groupJson["nodeIds"]) {
                        auto it = oldToNewNodeId.find(oldNodeId);
                        if (it != oldToNewNodeId.end()) {
                            addNodeToGroup(it->second, groupId);
                        }
                    }
                }
            }
        }
    }

} // namespace TerrainNodesV2
