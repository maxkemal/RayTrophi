/**
 * @file TerrainNodesV2.cpp
 * @brief Implementation of terrain nodes using V2 NodeSystem
 */

#include "TerrainNodesV2.h"
#include "NodeSystem/NodeRegistry.h"
#include "scene_data.h"
#include "TerrainManager.h"
#include "InstanceManager.h"
#include "FoliageAssetLibrary.h"
#include "RiverSpline.h"
#include "TerrainFFT.h"  // FFT-accelerated noise with CUDA fallback
#include "globals.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cmath>
#include <algorithm>
#include <cfloat>  // FLT_MAX
#include <limits>
#include <fstream>
#include <string>
#include <cstring>
#include <cctype>
#include <queue>
#include <deque>
#include <functional>
#include <unordered_set>
#include "perlin.h" // For gradient noise
#include <image_resample.h>
#include <image_filters.h>

namespace TerrainNodesV2 {

    static NodeSystem::Image2DData buildPackedErosionMap(
        const std::vector<float>& before,
        const std::vector<float>& after,
        int w,
        int h,
        const std::vector<float>* flowMap,
        const std::vector<float>* hardnessMap,
        const std::vector<float>* mask)
    {
        NodeSystem::Image2DData result;
        result.data = std::make_shared<std::vector<float>>(static_cast<size_t>(w) * h * 4, 0.0f);
        result.width = w;
        result.height = h;
        result.channels = 4;
        result.semantic = NodeSystem::ImageSemantic::Mask;

        float maxErosion = 0.0f;
        float maxDeposition = 0.0f;
        float maxFlow = 0.0f;

        const bool hasFlow = flowMap && flowMap->size() == static_cast<size_t>(w) * h;
        const bool hasHardness = hardnessMap && hardnessMap->size() == static_cast<size_t>(w) * h;
        const bool hasMask = mask && mask->size() == static_cast<size_t>(w) * h;

        std::vector<float> localFlow(static_cast<size_t>(w) * h, 0.0f);
        if (!hasFlow) {
            for (int y = 1; y < h - 1; ++y) {
                for (int x = 1; x < w - 1; ++x) {
                    const int idx = y * w + x;
                    const float slopeX = std::fabs(after[idx + 1] - after[idx - 1]) * 0.5f;
                    const float slopeY = std::fabs(after[idx + w] - after[idx - w]) * 0.5f;
                    const float delta = std::fabs(before[idx] - after[idx]);
                    localFlow[idx] = slopeX + slopeY + delta * 2.0f;
                    maxFlow = (std::max)(maxFlow, localFlow[idx]);
                }
            }
        }

        for (int i = 0; i < w * h; ++i) {
            const float erosion = (std::max)(before[i] - after[i], 0.0f);
            const float deposition = (std::max)(after[i] - before[i], 0.0f);
            maxErosion = (std::max)(maxErosion, erosion);
            maxDeposition = (std::max)(maxDeposition, deposition);
            if (hasFlow) {
                maxFlow = (std::max)(maxFlow, (*flowMap)[i]);
            }
        }

        const float logFlowDenom = static_cast<float>(std::log1p((std::max)(maxFlow, 1e-6f)));
        for (int i = 0; i < w * h; ++i) {
            const float erosion = (std::max)(before[i] - after[i], 0.0f);
            const float deposition = (std::max)(after[i] - before[i], 0.0f);
            const float rawFlow = hasFlow ? (*flowMap)[i] : localFlow[i];
            const float normalizedFlow = (logFlowDenom > 0.0f)
                ? (static_cast<float>(std::log1p((std::max)(rawFlow, 0.0f))) / logFlowDenom) : 0.0f;
            const float alpha = hasHardness ? std::clamp((*hardnessMap)[i], 0.0f, 1.0f)
                : (hasMask ? std::clamp((*mask)[i], 0.0f, 1.0f) : 1.0f);

            (*result.data)[i * 4 + 0] = (maxErosion > 1e-6f) ? (erosion / maxErosion) : 0.0f;
            (*result.data)[i * 4 + 1] = (maxDeposition > 1e-6f) ? (deposition / maxDeposition) : 0.0f;
            (*result.data)[i * 4 + 2] = std::clamp(normalizedFlow, 0.0f, 1.0f);
            (*result.data)[i * 4 + 3] = alpha;
        }

        return result;
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

                if (strideX == 1 && strideY == 1) {
                    // Direct copy with sRGB->linear conversion. Apply light dithering
                    // only on small/medium images to break banding; skip for very large inputs.
                    std::mt19937 rng(1337);
                    std::uniform_real_distribution<float> dither_dist(-0.5f / 255.0f, 0.5f / 255.0f);
                    const float gamma = 2.2f;
                    bool high_input = (w >= 3000 || h >= 3000);
                    bool apply_dither = !high_input;
                    for (int y = 0; y < loadedHeight; y++) {
                        for (int x = 0; x < loadedWidth; x++) {
                            int srcIdx = (y * strideY) * w + (x * strideX);
                            float v = (float)img[srcIdx] / 255.0f;
                            v = powf(v, gamma);
                            if (apply_dither) v = v + dither_dist(rng);
                            v = std::clamp(v, 0.0f, 1.0f);
                            rawHeightData[y * loadedWidth + x] = v;
                        }
                    }
                } else {
                    std::vector<uint8_t> dstBuf((size_t)loadedWidth * loadedHeight);
                    int lanczos_a = 2;
                    ImageResample::lanczos_resample_u8(img, w, h, dstBuf.data(), loadedWidth, loadedHeight, lanczos_a);
                    for (int i = 0; i < loadedWidth * loadedHeight; ++i) rawHeightData[i] = dstBuf[i] / 255.0f;
                }

                // Apply frequency separation (high-quality) to loadedHeightData with adaptive params
                try {
                    std::vector<float> fs_out;
                    const bool downsampled_local = !(strideX == 1 && strideY == 1);
                    float fs_sigma = downsampled_local ? 1.4f : 0.6f;
                    float fs_detail_strength = downsampled_local ? 0.8f : 0.97f;
                    ImageFilters::frequency_separation(rawHeightData, loadedWidth, loadedHeight, fs_out, fs_sigma, fs_detail_strength);
                    rawHeightData.swap(fs_out);
                } catch (...) {
                    // ignore failures
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

                if (strideX == 1 && strideY == 1) {
                    for (int y = 0; y < loadedHeight; y++) {
                        for (int x = 0; x < loadedWidth; x++) {
                            int srcIdx = (y * strideY) * w + (x * strideX);
                            rawHeightData[y * loadedWidth + x] = (float)img16[srcIdx] / 65535.0f;
                        }
                    }
                } else {
                    std::vector<uint16_t> dstBuf((size_t)loadedWidth * loadedHeight);
                    ImageResample::lanczos_resample_u16(img16, w, h, dstBuf.data(), loadedWidth, loadedHeight, 3);
                    for (int i = 0; i < loadedWidth * loadedHeight; ++i) rawHeightData[i] = dstBuf[i] / 65535.0f;
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
        
        // Create Perlin generator seeded by node parameter so output is deterministic
        Perlin perlin((unsigned int)seed);
        
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
        
        // CRITICAL: Use scale values from TerrainContext (set at evaluation start)
        // This ensures consistent scale throughout the entire node chain
        float preserved_scale_xz = tctx->scale_xz;
        float preserved_scale_y = tctx->scale_y;
        
        // ALWAYS apply input to terrain - handle size mismatch
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        const std::vector<float> originalHeight = *inputHeight.data;
        terrain->heightmap.data = *inputHeight.data;
        
        // Apply scale from context (guaranteed valid)
        terrain->heightmap.scale_xz = preserved_scale_xz;
        terrain->heightmap.scale_y = preserved_scale_y;
        terrain->dirty_mesh = true;
        
        // Get optional mask
        std::vector<float> mask;
        auto maskInput = getHeightInput(1, ctx);
        if (maskInput.isValid() && maskInput.data->size() == terrain->heightmap.data.size()) {
            mask = *maskInput.data;
        }
        
        // NEW: Get optional Hardness input
        auto hardnessInput = getHeightInput(2, ctx);
        if (hardnessInput.isValid() && hardnessInput.data->size() == terrain->heightmap.data.size()) {
            terrain->hardnessMap = *hardnessInput.data;
        }
        
        // Run erosion via TerrainManager
        if (useGPU) {
            mgr.hydraulicErosionGPU(terrain, params, mask);
        } else {
            mgr.hydraulicErosion(terrain, params, mask, [&ctx](float f) { ctx.reportNodeProgress(f); });
        }
        
        // Create output using INPUT dimensions to propagate correctly
        auto result = createHeightOutput(inputHeight.width, inputHeight.height);
        *result.data = terrain->heightmap.data;
        
        // Apply Edge Falloff if enabled
        if (this->edgeFalloffWidth > 0.01f) {
            applyEdgeFalloff(*result.data, inputHeight.width, inputHeight.height, this->edgeFalloffWidth, this->edgeFalloffValue);
        }

        auto erosionMap = buildPackedErosionMap(
            originalHeight,
            *result.data,
            inputHeight.width,
            inputHeight.height,
            nullptr,
            terrain->hardnessMap.empty() ? nullptr : &terrain->hardnessMap,
            mask.empty() ? nullptr : &mask);
        terrain->erosionMapRGBA = *erosionMap.data;

        ctx.setCachedValue(id, 0, result);
        ctx.setCachedValue(id, 1, erosionMap);
        return (outputIndex == 1) ? NodeSystem::PinValue{erosionMap} : NodeSystem::PinValue{result};
    }
    
    void HydraulicErosionNode::drawContent() {
        if (ImGui::Checkbox("Use GPU", &useGPU)) dirty = true;
        ImGui::TextDisabled("GPU uses Vulkan compute; CUDA/CPU remain fallbacks.");
        
        ImGui::TextColored(
            useGPU ? ImVec4(0.4f, 0.7f, 1.0f, 1.0f) : ImVec4(0.4f, 1.0f, 0.7f, 1.0f),
            useGPU ? "Mode: GPU Hydraulic Droplet Solver" : "Mode: CPU Hydraulic Droplet Solver");
        if (ImGui::DragInt("Droplets", &params.iterations, 25000, 25000, 1000000, "%d hits")) dirty = true;
        if (ImGui::DragInt("Lifetime", &params.dropletLifetime, 1, 16, 512)) dirty = true;
        if (ImGui::DragFloat("Inertia", &params.inertia, 0.01f, 0.0f, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Capacity", &params.sedimentCapacity, 0.05f, 0.1f, 20.0f)) dirty = true;
        if (ImGui::DragFloat("Erode Rate", &params.erodeSpeed, 0.01f, 0.01f, 2.0f)) dirty = true;
        if (ImGui::DragFloat("Deposit Rate", &params.depositSpeed, 0.01f, 0.0f, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Evaporate", &params.evaporateSpeed, 0.001f, 0.0f, 0.2f)) dirty = true;
        if (ImGui::DragFloat("Gravity", &params.gravity, 0.1f, 1.0f, 50.0f)) dirty = true;
        if (ImGui::DragFloat("Min Slope", &params.minSlope, 0.001f, 0.0f, 0.1f, "%.4f")) dirty = true;
        if (ImGui::DragInt("Channel Width", &params.erosionRadius, 1, 1, 15)) dirty = true;

        if (ImGui::Button(useGPU ? "Reset GPU Params" : "Reset CPU Params")) {
            params = HydraulicErosionParams();
            dirty = true;
        }

        ImGui::Separator();
        ImGui::Text("Edge Falloff");
        if (ImGui::DragFloat("Fade Width", &this->edgeFalloffWidth, 1.0f, 0.0f, 256.0f, "%.0f px")) dirty = true;
        if (ImGui::SliderFloat("Fade Value", &this->edgeFalloffValue, 0.0f, 1.0f)) dirty = true;
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
        
        // CRITICAL: Use scale values from TerrainContext (set at evaluation start)
        float preserved_scale_xz = tctx->scale_xz;
        float preserved_scale_y = tctx->scale_y;
        
        // ALWAYS apply input to terrain - handle size mismatch
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        terrain->heightmap.data = *inputHeight.data;
        
        // Apply scale from context
        terrain->heightmap.scale_xz = preserved_scale_xz;
        terrain->heightmap.scale_y = preserved_scale_y;
        terrain->dirty_mesh = true;
        
        // Get optional mask
        std::vector<float> mask;
        auto maskInput = getHeightInput(1, ctx);
        if (maskInput.isValid() && maskInput.data->size() == terrain->heightmap.data.size()) {
            mask = *maskInput.data;
        }

        // NEW: Get optional Hardness input
        auto hardnessInput = getHeightInput(2, ctx);
        if (hardnessInput.isValid() && hardnessInput.data->size() == terrain->heightmap.data.size()) {
            terrain->hardnessMap = *hardnessInput.data;
        }
        
        if (useGPU) {
            mgr.thermalErosionGPU(terrain, params, mask);
        } else {
            mgr.thermalErosion(terrain, params, mask, [&ctx](float f) { ctx.reportNodeProgress(f); });
        }
        
        auto result = createHeightOutput(inputHeight.width, inputHeight.height);
        *result.data = terrain->heightmap.data;
        
        // Apply Edge Falloff if enabled
        if (this->edgeFalloffWidth > 0.01f) {
            applyEdgeFalloff(*result.data, inputHeight.width, inputHeight.height, this->edgeFalloffWidth, this->edgeFalloffValue);
        }
        
        return result;
    }
    
    void ThermalErosionNode::drawContent() {
        if (ImGui::Checkbox("Use GPU", &useGPU)) dirty = true;
        ImGui::TextDisabled("GPU uses Vulkan compute; CUDA/CPU remain fallbacks.");
        if (ImGui::DragInt("Iterations", &params.iterations, 1, 1, 500)) dirty = true;
        
        float uiDegrees = std::atan(params.talusAngle) * 180.0f / 3.14159f;
        if (ImGui::DragFloat("Talus Angle", &uiDegrees, 0.1f, 0.0f, 80.0f, "%.1f deg")) {
            params.talusAngle = std::tan(uiDegrees * 3.14159f / 180.0f);
            dirty = true;
        }
        if (ImGui::DragFloat("Erosion Amount", &params.erosionAmount, 0.01f, 0.0f, 1.0f)) dirty = true;

        ImGui::Separator();
        ImGui::Text("Edge Falloff");
        if (ImGui::DragFloat("Fade Width", &this->edgeFalloffWidth, 1.0f, 0.0f, 256.0f, "%.0f px")) dirty = true;
        if (ImGui::SliderFloat("Fade Value", &this->edgeFalloffValue, 0.0f, 1.0f)) dirty = true;
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
        
        // CRITICAL: Preserve scales
        float preserved_scale_xz = tctx->scale_xz;
        float preserved_scale_y = tctx->scale_y;
        
        // Setup terrain data
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        const std::vector<float> originalHeight = *inputHeight.data;
        terrain->heightmap.data = *inputHeight.data;
        terrain->heightmap.scale_xz = preserved_scale_xz;
        terrain->heightmap.scale_y = preserved_scale_y;
        
        // Get optional mask
        std::vector<float> mask;
        auto maskInput = getHeightInput(1, ctx);
        if (maskInput.isValid() && maskInput.data->size() == terrain->heightmap.data.size()) {
            mask = *maskInput.data;
        }

        auto hardnessInput = getHeightInput(2, ctx);
        if (hardnessInput.isValid() && hardnessInput.data->size() == terrain->heightmap.data.size()) {
            terrain->hardnessMap = *hardnessInput.data;
        }

        std::vector<float> flowGuide;
        auto flowGuideInput = getHeightInput(3, ctx);
        if (flowGuideInput.isValid() && flowGuideInput.data->size() == terrain->heightmap.data.size()) {
            flowGuide = *flowGuideInput.data;
        }
        
        if (useGPU) {
            // ========================================================
            // GPU WAY: CPU-Parity Stream Power Erosion (Hybrid)
            // ========================================================
            mgr.fluvialErosionGPU(terrain, params, mask, flowGuide);
        } else {
            // ========================================================
            // CPU WAY: Global Hydrological Analysis
            // ========================================================
            mgr.fluvialErosion(terrain, params, mask,
                [&ctx](float f) { ctx.reportNodeProgress(f); }, flowGuide);
        }
        
        auto result = createHeightOutput(inputHeight.width, inputHeight.height);
        *result.data = terrain->heightmap.data;
        
        if (this->edgeFalloffWidth > 0.01f) {
            applyEdgeFalloff(*result.data, inputHeight.width, inputHeight.height, this->edgeFalloffWidth, this->edgeFalloffValue);
        }

        auto erosionMap = buildPackedErosionMap(
            originalHeight,
            *result.data,
            inputHeight.width,
            inputHeight.height,
            terrain->flowMap.empty() ? nullptr : &terrain->flowMap,
            terrain->hardnessMap.empty() ? nullptr : &terrain->hardnessMap,
            mask.empty() ? nullptr : &mask);
        terrain->erosionMapRGBA = *erosionMap.data;

        ctx.setCachedValue(id, 0, result);
        ctx.setCachedValue(id, 1, erosionMap);
        return (outputIndex == 1) ? NodeSystem::PinValue{erosionMap} : NodeSystem::PinValue{result};
    }
    
    void FluvialErosionNode::drawContent() {
        // Old serialized nodes may contain zero inertia from the former stream-power
        // UI.  Particle runoff needs a small directional memory to cross micro-relief.
        params.inertia = std::clamp(params.inertia, 0.12f, 0.98f);
        if (ImGui::Checkbox("Use GPU", &useGPU)) dirty = true;
        ImGui::TextDisabled("GPU uses device-resident particle runoff; CPU mode is unchanged.");
        ImGui::TextDisabled("Optional Flow Guide locks carving to watershed channels.");

        if (ImGui::Button(useGPU ? "Reset GPU Params" : "Reset CPU Params")) {
            params = HydraulicErosionParams();
            params.iterations = 250000;
            params.dropletLifetime = 384;
            params.inertia = 0.25f;
            params.sedimentCapacity = 1.5f;
            params.erosionRadius = 4;
            params.erodeSpeed = 0.12f;
            params.depositSpeed = 0.20f;
            params.evaporateSpeed = 0.001f;
            params.minSlope = 0.003f;
            dirty = true;
        }
        ImGui::TextColored(
            useGPU ? ImVec4(0.4f, 0.7f, 1.0f, 1.0f) : ImVec4(0.4f, 1.0f, 0.7f, 1.0f),
            useGPU ? "Mode: GPU Fluvial Runoff + Deposition" : "Mode: CPU Stream Power River Carver");
        ImGui::TextDisabled("Reach = Travel Length + Water Retention. GPU updates terrain every 32 cells.");
        if (ImGui::DragInt("Runoff Droplets", &params.iterations, 25000, 25000, 1500000, "%d parcels")) dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rainfall coverage and discharge sampling. More parcels reinforce shared channels.");
        if (ImGui::DragInt("Travel Length", &params.dropletLifetime, 4, 32, 1024, "%d cells")) dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Total parcel reach. It is internally continued in 32-cell physics segments for stable channel formation.");
        if (ImGui::DragFloat("Flow Inertia", &params.inertia, 0.01f, 0.12f, 0.98f)) dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Direction persistence. Too high can overshoot bends; 0.2-0.5 follows terrain well.");
        if (ImGui::DragFloat("Sediment Capacity", &params.sedimentCapacity, 0.05f, 0.1f, 8.0f)) dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How much soil water can carry before settling. Higher values transport material farther.");
        if (ImGui::DragFloat("Erosion Rate", &params.erodeSpeed, 0.01f, 0.01f, 1.5f)) dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Channel cutting strength, not travel distance. Excessive values deepen channels rapidly.");
        if (ImGui::DragFloat("Settling Rate", &params.depositSpeed, 0.01f, 0.0f, 1.0f)) dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Higher values deposit sediment sooner; lower values carry it farther downstream.");
        float waterRetention = 1.0f - params.evaporateSpeed;
        if (ImGui::DragFloat("Water Retention", &waterRetention, 0.0001f, 0.98f, 0.9999f, "%.4f")) {
            params.evaporateSpeed = 1.0f - waterRetention;
            dirty = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Higher retention means longer rivers. 0.999 is persistent runoff; 0.98 dries quickly.");
        if (ImGui::DragInt("Channel Width", &params.erosionRadius, 1, 1, 12)) dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum cross-flow bed radius. Effective range now matches the full 1-12 UI range.");
        if (ImGui::DragFloat("Flat Slope", &params.minSlope, 0.0005f, 0.0001f, 0.05f, "%.4f")) dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Below this physical slope, broad catchment guidance and meandering dominate.");
        
        ImGui::Separator();
        ImGui::Text("Edge Falloff");
        if (ImGui::DragFloat("Fade Width", &this->edgeFalloffWidth, 1.0f, 0.0f, 256.0f, "%.0f px")) dirty = true;
        if (ImGui::SliderFloat("Fade Value", &this->edgeFalloffValue, 0.0f, 1.0f)) dirty = true;
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
        
        // CRITICAL: Use scale values from TerrainContext (set at evaluation start)
        float preserved_scale_xz = tctx->scale_xz;
        float preserved_scale_y = tctx->scale_y;
        
        // ALWAYS apply input to terrain - handle size mismatch
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        terrain->heightmap.data = *inputHeight.data;
        
        // Apply scale from context
        terrain->heightmap.scale_xz = preserved_scale_xz;
        terrain->heightmap.scale_y = preserved_scale_y;
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
            mgr.windErosion(terrain, strength, direction, iterations, mask, [&ctx](float f) { ctx.reportNodeProgress(f); });
        }
        
        auto result = createHeightOutput(inputHeight.width, inputHeight.height);
        *result.data = terrain->heightmap.data;
        
        // Apply Edge Falloff if enabled
        if (this->edgeFalloffWidth > 0.01f) {
            applyEdgeFalloff(*result.data, inputHeight.width, inputHeight.height, this->edgeFalloffWidth, this->edgeFalloffValue);
        }
        
        return result;
    }
    
    void WindErosionNode::drawContent() {
        if (ImGui::Checkbox("Use GPU", &useGPU)) dirty = true;
        ImGui::TextDisabled("GPU uses Vulkan compute; CUDA/CPU remain fallbacks.");
        if (ImGui::SliderFloat("Strength", &strength, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Direction", &direction, 0.0f, 360.0f)) dirty = true;
        if (ImGui::DragInt("Iterations", &iterations, 10, 1, 1000)) dirty = true;

        ImGui::Separator();
        ImGui::Text("Edge Falloff");
        if (ImGui::DragFloat("Fade Width", &this->edgeFalloffWidth, 1.0f, 0.0f, 256.0f, "%.0f px")) dirty = true;
        if (ImGui::SliderFloat("Fade Value", &this->edgeFalloffValue, 0.0f, 1.0f)) dirty = true;
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
        if (ImGui::SliderInt("Iterations", &iterations, 1, 150)) dirty = true;
        if (ImGui::DragFloat("Deposit Rate", &depositionRate, 0.01f, 0.01f, 2.0f)) dirty = true;
        if (ImGui::DragFloat("Transport Cap", &transportCapacity, 0.05f, 0.05f, 20.0f)) dirty = true;
        if (ImGui::SliderFloat("Settling", &settlingSpeed, 0.0f, 0.99f)) dirty = true;
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
        if (ImGui::SliderFloat("Slope Threshold", &slopeThreshold, 5.0f, 70.0f)) dirty = true;
        if (ImGui::SliderFloat("Spread Angle", &fanSpreadAngle, 10.0f, 140.0f)) dirty = true;
        if (ImGui::DragFloat("Deposit Strength", &depositionStrength, 0.05f, 0.1f, 10.0f)) dirty = true;
        if (ImGui::SliderInt("Fan Length", &fanLength, 4, 300)) dirty = true;
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
        if (ImGui::SliderFloat("Sea Level", &seaLevel, 0.0f, 1.0f)) dirty = true;
        if (ImGui::SliderFloat("Delta Spread", &deltaSpread, 5.0f, 140.0f)) dirty = true;
        if (ImGui::SliderInt("Branches", &branchingFactor, 1, 9)) dirty = true;
        if (ImGui::DragFloat("Sediment Ratio", &sedimentRatio, 0.05f, 0.1f, 10.0f)) dirty = true;
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
                rainfallFactor = 0.15f;    // Tamed: Was 1.0
                temperatureFactor = 0.1f;
                windFactor = 0.05f;
                break;
            case ErosionPreset::MatureMountains:
                timeScaleMy = 40.0f;
                rainfallFactor = 0.3f;     // Tamed: Was 1.2
                temperatureFactor = 0.2f;
                windFactor = 0.15f;
                break;
            case ErosionPreset::AncientPlateau:
                timeScaleMy = 250.0f;
                rainfallFactor = 0.6f;     // Ancient needs high cumulative work
                temperatureFactor = 0.4f;
                windFactor = 0.3f;
                break;
            case ErosionPreset::TropicalRainforest:
                timeScaleMy = 20.0f;
                rainfallFactor = 0.5f;     // Was 2.0
                temperatureFactor = 0.15f;
                windFactor = 0.1f;
                break;
            case ErosionPreset::AridDesert:
                timeScaleMy = 80.0f;
                rainfallFactor = 0.05f;
                temperatureFactor = 0.5f;
                windFactor = 0.6f;
                break;
            case ErosionPreset::GlacialCarving:
                timeScaleMy = 2.0f;
                rainfallFactor = 0.2f;
                temperatureFactor = 0.6f;  // High freeze-thaw
                windFactor = 0.2f;
                break;
            case ErosionPreset::CoastalErosion:
                timeScaleMy = 15.0f;
                rainfallFactor = 0.25f;
                temperatureFactor = 0.15f;
                windFactor = 0.4f;
                break;
            case ErosionPreset::VolcanicTerrain:
                timeScaleMy = 0.8f;
                rainfallFactor = 0.3f;
                temperatureFactor = 0.4f;
                windFactor = 0.1f;
                break;
            case ErosionPreset::RiverDelta:
                timeScaleMy = 10.0f;
                rainfallFactor = 0.4f;
                temperatureFactor = 0.1f;
                windFactor = 0.05f;
                break;
            default:
                break;
        }
    }
    
    NodeSystem::PinValue ErosionWizardNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto inputHeight = getHeightInput(0, ctx);
        if (!inputHeight.isValid()) {
            ctx.addError(id, "Erosion Wizard requires a valid height input");
            return NodeSystem::PinValue{};
        }

        // The legacy wizard used to mutate TerrainObject from drawContent(),
        // outside graph evaluation/finalize. Keep serialized projects loadable,
        // but make the node deterministic and side-effect free.
        isSimulating = false;
        currentPass = 0;
        cachedTerrain = nullptr;
        cachedMask.clear();
        originalHeight.clear();
        SCENE_LOG_WARN("[ErosionWizard] Legacy simulation is disabled; node is operating as passthrough.");

        if (outputIndex == 0) {
            return inputHeight;
        }

        auto result = createMaskOutput(inputHeight.width, inputHeight.height);
        std::fill(result.data->begin(), result.data->end(), 0.0f);
        return result;
    }
    
    void ErosionWizardNode::drawContent() {
        ImGui::TextWrapped("Legacy node disabled. Existing projects use a safe passthrough.");
        ImGui::BeginDisabled();
        // Preset selector (Disable during sim)
        ImGui::BeginDisabled(isSimulating);
        if (ImGui::BeginCombo("Preset", getPresetName(preset))) {
            for (int i = 0; i <= (int)ErosionPreset::RiverDelta; i++) {
                ErosionPreset p = static_cast<ErosionPreset>(i);
                bool selected = (preset == p);
                if (ImGui::Selectable(getPresetName(p), selected)) {
                    preset = p;
                    if (p != ErosionPreset::Custom) applyPreset(p);
                    dirty = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
        ImGui::Separator();
        if (ImGui::SliderFloat("Time (My)", &timeScaleMy, 0.1f, 500.0f, "%.1f My")) dirty = true;
        ImGui::Text("Climate:");
        if (ImGui::SliderFloat("Rainfall", &rainfallFactor, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Temperature", &temperatureFactor, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Wind", &windFactor, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderInt("Quality", &qualityLevel, 1, 3)) dirty = true;
        if (!g_hasCUDA) useGPU = false;
        ImGui::BeginDisabled(!g_hasCUDA);
        if (ImGui::Checkbox("Use GPU", &useGPU)) dirty = true;
        ImGui::EndDisabled();
        if (!g_hasCUDA) ImGui::TextDisabled("CUDA required for GPU mode.");
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
                    // Normalize Talus Angle: UI uses degrees, Kernel uses tangent (h/w)
                    float degrees = 30.0f - temperatureFactor * 10.0f; // Soften as it gets hotter
                    tp.talusAngle = std::tan(degrees * 3.14159f / 180.0f);
                    tp.erosionAmount = 0.4f * temperatureFactor;
                    
                    if (useGPU) mgr.thermalErosionGPU(cachedTerrain, tp, cachedMask);
                    else mgr.thermalErosion(cachedTerrain, tp, cachedMask);
                }
                
                // 2. Hydraulic
                if (rainfallFactor > 0.01f && hydraulicItersPerPass > 0) {
                    HydraulicErosionParams hp;
                    // SCALE ITERATIONS by resolution for consistent density in Wizard
                    int w = cachedTerrain->heightmap.width;
                    int h = cachedTerrain->heightmap.height;
                    float resScale = (float)(w * h) / (512.0f * 512.0f);
                    hp.iterations = static_cast<int>(hydraulicItersPerPass * resScale);
                    
                    hp.erosionRadius = 2; // Fixed radius for stability
                    hp.depositSpeed = 0.2f; // Balanced for consistency
                    hp.sedimentCapacity = 2.0f * rainfallFactor; // Tamed from 12.0
                    hp.evaporateSpeed = 0.012f;
                    hp.erodeSpeed = 0.15f * rainfallFactor; // Tamed from 0.6
                    hp.gravity = 10.0f;
                    hp.dropletLifetime = 128;
                    
                    if (useGPU) mgr.fluvialErosionGPU(cachedTerrain, hp, cachedMask);
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
        ImGui::EndDisabled();
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
        
        // CRITICAL: Use scale values from TerrainContext (set at evaluation start)
        // This ensures consistent scale throughout the entire node chain
        float preserved_scale_xz = tctx->scale_xz;
        float preserved_scale_y = tctx->scale_y;
        
        // Resize terrain if dimensions changed
        if (inputHeight.width != terrain->heightmap.width || 
            inputHeight.height != terrain->heightmap.height) {
            terrain->heightmap.width = inputHeight.width;
            terrain->heightmap.height = inputHeight.height;
            terrain->heightmap.data.resize(inputHeight.width * inputHeight.height);
        }
        
        // Apply heightmap data
        terrain->heightmap.data = *inputHeight.data;
        
        // Apply scale from context (guaranteed valid)
        terrain->heightmap.scale_xz = preserved_scale_xz;
        terrain->heightmap.scale_y = preserved_scale_y;
        
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
        
        auto splatInput = getImageInput(0, ctx);
        if (!splatInput.isValid() || splatInput.channels != 4) {
            ctx.addError(id, "Splat input must be a valid RGBA image");
            return NodeSystem::PinValue{};
        }
        
        TerrainObject* terrain = tctx->terrain;
        
        // Apply splat data to terrain if autoApply enabled. Terrain splat maps
        // intentionally use at least 512x512, while graph data follows the
        // height resolution, so the output boundary owns the RGBA resample.
        if (autoApplyToTerrain) {
            TerrainManager& terrainManager = TerrainManager::getInstance();
            if (!terrain->splatMap) terrainManager.initLayers(terrain);
            terrainManager.resizeSplatMap(terrain);
            if (!terrain->splatMap || terrain->splatMap->width <= 0 || terrain->splatMap->height <= 0) {
                ctx.addError(id, "Terrain splat map could not be initialized");
                return NodeSystem::PinValue{};
            }

            int sw = terrain->splatMap->width;
            int sh = terrain->splatMap->height;
            const size_t destinationPixels = static_cast<size_t>(sw) * sh;
            if (terrain->splatMap->pixels.size() != destinationPixels) {
                terrain->splatMap->pixels.resize(destinationPixels);
            }

            const int sourceWidth = splatInput.width;
            const int sourceHeight = splatInput.height;
            for (int y = 0; y < sh; ++y) {
                // Terrain graph images use heightmap row order (row 0 = local
                // Z/UV 0). Texture::pixels uses image-storage order, where row
                // 0 is sampled at UV 1. Paint, autoMask and project import all
                // use the same inversion; perform it once at this output boundary.
                const int terrainRow = (sh - 1) - y;
                const float sourceY = clampValue(
                    ((static_cast<float>(terrainRow) + 0.5f) * sourceHeight / sh) - 0.5f,
                    0.0f, static_cast<float>(sourceHeight - 1));
                const int y0 = static_cast<int>(std::floor(sourceY));
                const int y1 = clampValue(y0 + 1, 0, sourceHeight - 1);
                const float ty = sourceY - static_cast<float>(y0);
                for (int x = 0; x < sw; ++x) {
                    const float sourceX = clampValue(
                        ((static_cast<float>(x) + 0.5f) * sourceWidth / sw) - 0.5f,
                        0.0f, static_cast<float>(sourceWidth - 1));
                    const int x0 = static_cast<int>(std::floor(sourceX));
                    const int x1 = clampValue(x0 + 1, 0, sourceWidth - 1);
                    const float tx = sourceX - static_cast<float>(x0);
                    float weights[4]{};
                    for (int channel = 0; channel < 4; ++channel) {
                        const float v00 = (*splatInput.data)[(static_cast<size_t>(y0) * sourceWidth + x0) * 4 + channel];
                        const float v10 = (*splatInput.data)[(static_cast<size_t>(y0) * sourceWidth + x1) * 4 + channel];
                        const float v01 = (*splatInput.data)[(static_cast<size_t>(y1) * sourceWidth + x0) * 4 + channel];
                        const float v11 = (*splatInput.data)[(static_cast<size_t>(y1) * sourceWidth + x1) * 4 + channel];
                        const float top = v00 + (v10 - v00) * tx;
                        const float bottom = v01 + (v11 - v01) * tx;
                        weights[channel] = clampValue(top + (bottom - top) * ty, 0.0f, 1.0f);
                    }

                    float sum = weights[0] + weights[1] + weights[2] + weights[3];
                    if (sum <= 1e-6f) { weights[0] = 1.0f; sum = 1.0f; }
                    auto& pixel = terrain->splatMap->pixels[static_cast<size_t>(y) * sw + x];
                    pixel.r = static_cast<uint8_t>(weights[0] / sum * 255.0f + 0.5f);
                    pixel.g = static_cast<uint8_t>(weights[1] / sum * 255.0f + 0.5f);
                    pixel.b = static_cast<uint8_t>(weights[2] / sum * 255.0f + 0.5f);
                    pixel.a = static_cast<uint8_t>(weights[3] / sum * 255.0f + 0.5f);
                }
            }
            terrain->splatMap->m_is_loaded = true;
            terrain->splatMap->updateGPU();
        }
        
        return NodeSystem::PinValue{};
    }
    
    void SplatOutputNode::drawContent() {
        if (ImGui::Checkbox("Apply to Terrain", &autoApplyToTerrain)) dirty = true;
        
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
        
        for (int y = 0; y < sh; ++y) {
            for (int x = 0; x < sw; ++x) {
                const int srcIdx = y * sw + x;
                const int dstIdx = ((sh - 1) - y) * sw + x;
                rgba[dstIdx * 4 + 0] = terrain->splatMap->pixels[srcIdx].r;
                rgba[dstIdx * 4 + 1] = terrain->splatMap->pixels[srcIdx].g;
                rgba[dstIdx * 4 + 2] = terrain->splatMap->pixels[srcIdx].b;
                rgba[dstIdx * 4 + 3] = terrain->splatMap->pixels[srcIdx].a;
            }
        }
        
        stbi_write_png(exportPath, sw, sh, 4, rgba.data(), sw * 4);
    }

    // HARDNESS OUTPUT
    NodeSystem::PinValue HardnessOutputNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        if (!tctx || !tctx->terrain) {
            ctx.addError(id, "No terrain context");
            return NodeSystem::PinValue{};
        }
        
        auto hardnessInput = getHeightInput(0, ctx);
        if (!hardnessInput.isValid()) {
            return NodeSystem::PinValue{};
        }
        
        TerrainObject* terrain = tctx->terrain;
        int w = hardnessInput.width;
        int h = hardnessInput.height;
        if (terrain->heightmap.width >= 2 && terrain->heightmap.height >= 2 &&
            (w != terrain->heightmap.width || h != terrain->heightmap.height)) {
            ctx.addError(id, "Hardness resolution must match terrain height resolution");
            return NodeSystem::PinValue{};
        }
        
        // Ensure hardness map matches size
        if (terrain->hardnessMap.size() != (size_t)(w * h)) {
            terrain->hardnessMap.resize(w * h, 0.5f);
        }
        
        // Copy data directly to terrain hardness map
        terrain->hardnessMap = *hardnessInput.data;
        
        // Pass-through for chaining
        return hardnessInput;
    }

    // HARDNESS INPUT
    NodeSystem::PinValue HardnessInputNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto* tctx = getTerrainContext(ctx);
        if (!tctx || !tctx->terrain) {
            ctx.addError(id, "No terrain context");
            return NodeSystem::PinValue{};
        }
        
        TerrainObject* terrain = tctx->terrain;
        int w = terrain->heightmap.width;
        int h = terrain->heightmap.height;
        if (w < 2 || h < 2) {
            w = tctx->width;
            h = tctx->height;
        }
        
        auto result = createMaskOutput(w, h);
        
        // If terrain doesn't have a hardness map, create a default one (0.5)
        if (terrain->hardnessMap.size() != (size_t)(w * h)) {
            terrain->hardnessMap.assign(w * h, 0.5f);
        }
        
        *result.data = terrain->hardnessMap;
        return result;
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
        if (ImGui::DragFloat("Factor", &factor, 0.1f)) dirty = true;
    }
    
    NodeSystem::PinValue BlendNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto inputA = getHeightInput(0, ctx);
        auto inputB = getHeightInput(1, ctx);
        
        if (!inputA.isValid() || !inputB.isValid()) {
            ctx.addError(id, "Both inputs A and B required");
            return NodeSystem::PinValue{};
        }
        
        if (inputA.width != inputB.width || inputA.height != inputB.height) {
            ctx.addError(id, "Input resolution mismatch; insert a Resample node");
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
        if (ImGui::DragFloat("Alpha", &alpha, 0.01f, 0.0f, 1.0f)) dirty = true;
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
        if (ImGui::DragFloat("Min", &minVal, 0.1f)) dirty = true;
        if (ImGui::DragFloat("Max", &maxVal, 0.1f)) dirty = true;
    }
    
    NodeSystem::PinValue InvertNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        if (!input.isValid()) {
            ctx.addError(id, "No valid input");
            return NodeSystem::PinValue{};
        }
        
        auto result = createMaskOutput(input.width, input.height);
        
        for (size_t i = 0; i < input.data->size(); i++) {
            (*result.data)[i] = 1.0f - (*input.data)[i];
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
        if (ImGui::DragFloat("Min Slope", &minSlope, 0.1f, 0.0f, 90.0f)) dirty = true;
        if (ImGui::DragFloat("Max Slope", &maxSlope, 0.1f, 0.0f, 90.0f)) dirty = true;
        if (ImGui::DragFloat("Falloff", &falloff, 0.01f, 0.0f, 1.0f)) dirty = true;
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
        if (ImGui::DragFloat("Min Height", &minHeight, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Max Height", &maxHeight, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Falloff", &falloff, 0.1f, 0.0f, 100.0f)) dirty = true;
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
        if (ImGui::DragFloat("Min Curve", &minCurve, 0.01f)) dirty = true;
        if (ImGui::DragFloat("Max Curve", &maxCurve, 0.01f)) dirty = true;
        if (ImGui::Checkbox("Select Convex (Ridges)", &selectConvex)) dirty = true;
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
        
        const int w = input.width;
        const int h = input.height;
        if (w < 2 || h < 2) {
            ctx.addError(id, "Flow Mask requires at least a 2x2 height image");
            return NodeSystem::PinValue{};
        }
        auto result = createMaskOutput(w, h);
        const int pixelCount = w * h;
        const float heightScale = std::max(0.0001f, tctx->scale_y);
        const float cellSize = std::max(0.0001f, tctx->scale_xz / static_cast<float>(std::max(w, h)));

        // Priority-flood removes closed numerical pits and records a guaranteed
        // route to the boundary. Unlike the old implementation, this drainage
        // solution is actually used by the accumulation pass.
        const std::vector<float>& height = *input.data;
        std::vector<float> filledHeight = height;
        std::vector<int> drainageParent(static_cast<size_t>(pixelCount), -1);
        std::vector<uint8_t> processed(static_cast<size_t>(pixelCount), 0);
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pq;
        const float eps = std::max(1.0e-7f, 1.0e-5f / heightScale);
        static constexpr int dx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        static constexpr int dy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        static constexpr float distance8[8] = {1.41421356f, 1.0f, 1.41421356f, 1.0f,
                                               1.0f, 1.41421356f, 1.0f, 1.41421356f};

        for (int x = 0; x < w; x++) {
            pq.push({ filledHeight[x], x });
            pq.push({ filledHeight[(h - 1) * w + x], (h - 1) * w + x });
            processed[x] = processed[(h - 1) * w + x] = 1;
        }
        for (int y = 1; y < h - 1; y++) {
            pq.push({ filledHeight[y * w], y * w });
            pq.push({ filledHeight[y * w + w - 1], y * w + w - 1 });
            processed[y * w] = processed[y * w + w - 1] = 1;
        }

        while (!pq.empty()) {
            const int idx = pq.top().second;
            pq.pop();
            const float centerFilled = filledHeight[static_cast<size_t>(idx)];
            const int x = idx % w;
            const int y = idx / w;

            for (int d = 0; d < 8; d++) {
                int nx = x + dx8[d], ny = y + dy8[d];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                int nIdx = ny * w + nx;
                if (processed[static_cast<size_t>(nIdx)]) continue;
                const float routedHeight = std::max(filledHeight[static_cast<size_t>(nIdx)], centerFilled + eps);
                filledHeight[static_cast<size_t>(nIdx)] = routedHeight;
                drainageParent[static_cast<size_t>(nIdx)] = idx;
                processed[static_cast<size_t>(nIdx)] = 1;
                pq.push({routedHeight, nIdx});
            }
            if ((idx & 0x7fff) == 0 && ctx.isCancelled()) return NodeSystem::PinValue{};
        }

        // One deterministic D8 receiver per cell creates continuous channel
        // trees. The previous MFD path sprayed every cell into many neighbors,
        // which was the source of the noisy, almost-everywhere grass blending.
        std::vector<int> receiver(static_cast<size_t>(pixelCount), -1);
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                const int idx = y * w + x;
                float bestSlope = 0.0f;
                int best = -1;
                for (int d = 0; d < 8; ++d) {
                    const int nidx = (y + dy8[d]) * w + (x + dx8[d]);
                    const float drop = (filledHeight[static_cast<size_t>(idx)] -
                                        filledHeight[static_cast<size_t>(nidx)]) * heightScale;
                    const float slope = drop / (distance8[d] * cellSize);
                    if (slope > bestSlope) {
                        bestSlope = slope;
                        best = nidx;
                    }
                }
                receiver[static_cast<size_t>(idx)] = best >= 0 ? best : drainageParent[static_cast<size_t>(idx)];
            }
        }

        std::vector<int> indices(static_cast<size_t>(pixelCount));
        for (int i = 0; i < pixelCount; ++i) indices[static_cast<size_t>(i)] = i;
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            if (filledHeight[static_cast<size_t>(a)] == filledHeight[static_cast<size_t>(b)]) return a > b;
            return filledHeight[static_cast<size_t>(a)] > filledHeight[static_cast<size_t>(b)];
        });

        std::vector<float> flow(static_cast<size_t>(pixelCount), 1.0f);
        const float retained = clampValue(decay, 0.95f, 1.0f);
        for (const int idx : indices) {
            const int next = receiver[static_cast<size_t>(idx)];
            if (next >= 0 && next != idx) {
                flow[static_cast<size_t>(next)] += flow[static_cast<size_t>(idx)] * retained;
            }
            if ((idx & 0x7fff) == 0 && ctx.isCancelled()) return NodeSystem::PinValue{};
        }

        if (tctx->terrain && tctx->publishTerrainState) {
            tctx->terrain->flowMap.resize(static_cast<size_t>(pixelCount));
            for (int i = 0; i < pixelCount; ++i) {
                tctx->terrain->flowMap[static_cast<size_t>(i)] = flow[static_cast<size_t>(i)] * strength;
            }
        }

        float maxAccumulation = 1.0f;
        for (const float value : flow) maxAccumulation = std::max(maxAccumulation, value);
        const float logMaximum = std::max(1.0e-6f, std::log1p(maxAccumulation - 1.0f));
        const float detailT = static_cast<float>(clampValue(detailLevel, 1, 8) - 1) / 7.0f;
        const float threshold = 0.72f + (0.16f - 0.72f) * detailT;
        const float softness = clampValue(channelSoftness, 0.01f, 0.20f);

        std::vector<float> channelMask(static_cast<size_t>(pixelCount), 0.0f);
        for (int i = 0; i < pixelCount; ++i) {
            const float magnitude = std::log1p(std::max(0.0f, flow[static_cast<size_t>(i)] - 1.0f)) / logMaximum;
            float channel = clampValue((magnitude - (threshold - softness)) / (2.0f * softness), 0.0f, 1.0f);
            channel = channel * channel * (3.0f - 2.0f * channel);
            channelMask[static_cast<size_t>(i)] = channel * (normalize ? magnitude : 1.0f);
        }

        // Expand only existing channels by a few pixels. This controls bank
        // width without the old whole-image blur that polluted every biome.
        for (int pass = 0; pass < clampValue(bankSpread, 0, 4); ++pass) {
            std::vector<float> expanded = channelMask;
            for (int y = 1; y < h - 1; ++y) {
                for (int x = 1; x < w - 1; ++x) {
                    const int idx = y * w + x;
                    float neighborMax = 0.0f;
                    for (int d = 0; d < 8; ++d) {
                        neighborMax = std::max(neighborMax, channelMask[static_cast<size_t>((y + dy8[d]) * w + x + dx8[d])]);
                    }
                    expanded[static_cast<size_t>(idx)] = std::max(channelMask[static_cast<size_t>(idx)], neighborMax * 0.65f);
                }
            }
            channelMask.swap(expanded);
        }

        for (int i = 0; i < pixelCount; ++i) {
            (*result.data)[static_cast<size_t>(i)] = clampValue(channelMask[static_cast<size_t>(i)] * strength, 0.0f, 1.0f);
        }

        return result;
    }
    
    void FlowMaskNode::drawContent() {
        ImGui::SetNextItemWidth(80);
        if (ImGui::SliderInt("Detail", &detailLevel, 1, 8)) dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("1: main rivers only, 8: finest tributaries");
        ImGui::SetNextItemWidth(80);
        if (ImGui::SliderInt("Bank Spread", &bankSpread, 0, 4)) dirty = true;
        ImGui::SetNextItemWidth(80);
        if (ImGui::SliderFloat("Strength", &strength, 0.1f, 2.0f)) dirty = true;
        ImGui::SetNextItemWidth(80);
        if (ImGui::SliderFloat("Persistence", &decay, 0.95f, 1.0f, "%.3f")) dirty = true;
        ImGui::SetNextItemWidth(80);
        if (ImGui::SliderFloat("Softness", &channelSoftness, 0.01f, 0.20f, "%.3f")) dirty = true;
        if (ImGui::Checkbox("Normalize", &normalize)) dirty = true;
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
        if (w < 2 || h < 2) {
            ctx.addError(id, "Exposure Mask requires at least a 2x2 height image");
            return NodeSystem::PinValue{};
        }
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
        if (ImGui::SliderFloat("Azimuth", &sunAzimuth, 0.0f, 360.0f, "%.0f\302\260")) {
            while(sunAzimuth < 0) sunAzimuth += 360;
            while(sunAzimuth >= 360) sunAzimuth -= 360;
            dirty = true;
        }
        
        // Visual Compass for Azimuth
        ImGui::SameLine();
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 cp = ImGui::GetCursorScreenPos();
        float c_size = 12.0f;
        cp.x += c_size + 5;
        cp.y += 12.0f;
        drawList->AddCircleFilled(cp, c_size + 2, IM_COL32(40, 40, 40, 255));
        drawList->AddCircle(cp, c_size, IM_COL32(200, 200, 200, 255), 16);
        
        float s_rad = sunAzimuth * 0.0174533f;
        ImVec2 s_needle(cp.x + sinf(s_rad) * c_size, cp.y - cosf(s_rad) * c_size);
        drawList->AddLine(cp, s_needle, IM_COL32(255, 200, 50, 255), 2.0f); // Golden sun color
        drawList->AddText(ImVec2(cp.x - 3, cp.y - c_size - 14), IM_COL32(200, 200, 200, 150), "N");
        ImGui::Dummy(ImVec2(c_size * 2 + 10, 1)); 

        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("Elevation", &sunElevation, 0.0f, 90.0f, "%.0f\302\260")) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("Contrast", &contrast, 0.1f, 3.0f)) dirty = true;
        if (ImGui::Checkbox("Invert (Shadow)", &invert)) dirty = true;
    }

    NodeSystem::PinValue TerrainAnalysisNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto flow = getHeightInput(1, ctx);
        auto* tctx = getTerrainContext(ctx);
        if (!height.isValid() || height.width < 2 || height.height < 2 || !tctx) {
            ctx.addError(id, "Terrain Analysis requires a valid height input");
            return NodeSystem::PinValue{};
        }
        if (flow.isValid() && (flow.width != height.width || flow.height != height.height)) {
            ctx.addError(id, "Terrain Analysis flow input must match height resolution");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const size_t pixelCount = static_cast<size_t>(w) * h;
        const float heightScale = (std::max)(tctx->scale_y, 0.1f);
        const float cellSize = (std::max)(tctx->scale_xz / (std::max)(w - 1, 1), 1e-5f);
        std::array<NodeSystem::Image2DData, 5> result = {
            createMaskOutput(w, h), createMaskOutput(w, h), createMaskOutput(w, h),
            createMaskOutput(w, h), createMaskOutput(w, h)
        };

        std::vector<float> meters(pixelCount, 0.0f);
        float minimumHeight = (std::numeric_limits<float>::max)();
        float maximumHeight = (std::numeric_limits<float>::lowest)();
        for (size_t i = 0; i < pixelCount; ++i) {
            meters[i] = (*height.data)[i] * heightScale;
            minimumHeight = (std::min)(minimumHeight, meters[i]);
            maximumHeight = (std::max)(maximumHeight, meters[i]);
        }
        const float relief = (std::max)(maximumHeight - minimumHeight, 0.01f);

        // Integral field supplies broad concavity/valley context in O(N), shared
        // by every output instead of repeating box scans in biome nodes.
        const size_t stride = static_cast<size_t>(w) + 1u;
        std::vector<float> integral(stride * (static_cast<size_t>(h) + 1u), 0.0f);
        for (int y = 0; y < h; ++y) {
            float rowSum = 0.0f;
            for (int x = 0; x < w; ++x) {
                rowSum += meters[static_cast<size_t>(y) * w + x];
                integral[(static_cast<size_t>(y) + 1u) * stride + static_cast<size_t>(x) + 1u] =
                    integral[static_cast<size_t>(y) * stride + static_cast<size_t>(x) + 1u] + rowSum;
            }
        }
        const auto boxAverage = [&](int cx, int cy, int radius) {
            const int x0 = (std::max)(cx - radius, 0), y0 = (std::max)(cy - radius, 0);
            const int x1 = (std::min)(cx + radius, w - 1), y1 = (std::min)(cy + radius, h - 1);
            const size_t ax = static_cast<size_t>(x0), ay = static_cast<size_t>(y0);
            const size_t bx = static_cast<size_t>(x1 + 1), by = static_cast<size_t>(y1 + 1);
            const float sum = integral[by * stride + bx] - integral[ay * stride + bx] -
                integral[by * stride + ax] + integral[ay * stride + ax];
            return sum / static_cast<float>((x1 - x0 + 1) * (y1 - y0 + 1));
        };

        const int nearRadius = clampValue(neighborhoodRadius, 1, (std::max)(1, (std::min)(w, h) / 4));
        const int farRadius = clampValue(nearRadius * 4, nearRadius, (std::max)(nearRadius, (std::min)(w, h) / 3));
        const float valleyDenominator = (std::max)(relief * valleyScale, cellSize * 0.25f);
        const float curvatureDenominator = (std::max)(cellSize * curvatureScale, relief * 0.001f);

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int xl = (std::max)(x - 1, 0), xr = (std::min)(x + 1, w - 1);
                const int yu = (std::max)(y - 1, 0), yd = (std::min)(y + 1, h - 1);
                const size_t index = static_cast<size_t>(y) * w + x;
                const float center = meters[index];
                const float dzdx = (meters[static_cast<size_t>(y) * w + xr] -
                    meters[static_cast<size_t>(y) * w + xl]) /
                    (std::max)((xr - xl) * cellSize, 1e-5f);
                const float dzdy = (meters[static_cast<size_t>(yd) * w + x] -
                    meters[static_cast<size_t>(yu) * w + x]) /
                    (std::max)((yd - yu) * cellSize, 1e-5f);
                const float slope = std::atan(std::sqrt(dzdx * dzdx + dzdy * dzdy)) / 1.57079632679f;

                const float localAverage = (
                    meters[static_cast<size_t>(y) * w + xl] +
                    meters[static_cast<size_t>(y) * w + xr] +
                    meters[static_cast<size_t>(yu) * w + x] +
                    meters[static_cast<size_t>(yd) * w + x]) * 0.25f;
                const float concavity = clampValue((localAverage - center) / curvatureDenominator, 0.0f, 1.0f);
                const float convexity = clampValue((center - localAverage) / curvatureDenominator, 0.0f, 1.0f);
                const float valleyRise = (std::max)(boxAverage(x, y, nearRadius),
                    boxAverage(x, y, farRadius)) - center;
                const float valley = clampValue(valleyRise / valleyDenominator, 0.0f, 1.0f);
                const float flatness = 1.0f - clampValue(slope, 0.0f, 1.0f);
                const float terrainWetness = clampValue(
                    valley * 0.58f + concavity * 0.27f + flatness * 0.15f, 0.0f, 1.0f);
                const float flowWetness = flow.isValid()
                    ? clampValue((*flow.data)[index], 0.0f, 1.0f) : 0.0f;

                (*result[0].data)[index] = clampValue(slope, 0.0f, 1.0f);
                (*result[1].data)[index] = concavity;
                (*result[2].data)[index] = convexity;
                (*result[3].data)[index] = valley;
                (*result[4].data)[index] = 1.0f - (1.0f - terrainWetness) * (1.0f - flowWetness);
            }
        }

        for (int i = 0; i < 5; ++i) ctx.setCachedValue(id, i, result[static_cast<size_t>(i)]);
        return (outputIndex >= 0 && outputIndex < 5)
            ? NodeSystem::PinValue{result[static_cast<size_t>(outputIndex)]}
            : NodeSystem::PinValue{};
    }

    void TerrainAnalysisNode::drawContent() {
        bool edited = false;
        edited |= ImGui::SliderFloat("Valley Scale", &valleyScale, 0.005f, 0.5f, "%.3f");
        edited |= ImGui::SliderFloat("Curvature Scale", &curvatureScale, 0.05f, 10.0f, "%.2f");
        edited |= ImGui::SliderInt("Feature Radius", &neighborhoodRadius, 1, 64);
        ImGui::TextDisabled("One solve, five cached fields");
        if (edited) dirty = true;
    }

    namespace {
        constexpr int kHydrologyDx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        constexpr int kHydrologyDy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

        int decodeFlowDirection(float encoded) {
            const int direction = static_cast<int>(std::lround(encoded * 9.0f)) - 1;
            return (direction >= 0 && direction < 8) ? direction : -1;
        }
    }

    NodeSystem::PinValue WatershedAnalysisNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        if (!height.isValid() || height.width < 2 || height.height < 2) {
            ctx.addError(id, "Watershed Analysis requires a valid height input");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const int count = w * h;
        auto* tctx = getTerrainContext(ctx);
        const float cellSizeX = tctx && tctx->terrain
            ? tctx->terrain->heightmap.scale_xz / static_cast<float>((std::max)(w - 1, 1)) : 1.0f;
        const float cellSizeZ = tctx && tctx->terrain
            ? tctx->terrain->heightmap.scale_xz / static_cast<float>((std::max)(h - 1, 1)) : 1.0f;
        const float catchmentCellArea = cellSizeX * cellSizeZ;
        const float rainfallPerCell = (std::max)(rainfall, 0.001f);
        std::array<NodeSystem::Image2DData, 5> result = {
            createHeightOutput(w, h), createMaskOutput(w, h),
            createMaskOutput(w, h), createMaskOutput(w, h), createHeightOutput(w, h)
        };
        std::vector<float>& filled = *result[0].data;
        filled = *height.data;
        std::vector<int> parent(static_cast<size_t>(count), -1);
        std::vector<int> visitOrder;
        visitOrder.reserve(static_cast<size_t>(count));
        std::vector<uint8_t> visited(static_cast<size_t>(count), 0);

        using QueueEntry = std::pair<float, int>;
        std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> frontier;
        const auto seedOutlet = [&](int index) {
            if (visited[static_cast<size_t>(index)]) return;
            visited[static_cast<size_t>(index)] = 1;
            frontier.push({filled[static_cast<size_t>(index)], index});
        };
        for (int x = 0; x < w; ++x) {
            seedOutlet(x);
            seedOutlet((h - 1) * w + x);
        }
        for (int y = 1; y < h - 1; ++y) {
            seedOutlet(y * w);
            seedOutlet(y * w + w - 1);
        }

        const float epsilon = (std::max)(flatEpsilon, 1e-7f);
        int processedCount = 0;
        while (!frontier.empty()) {
            const auto [priority, index] = frontier.top();
            frontier.pop();
            (void)priority;
            visitOrder.push_back(index);
            if ((++processedCount & 0x3fff) == 0) {
                if (ctx.isCancelled()) return NodeSystem::PinValue{};
                ctx.reportNodeProgress(0.65f * static_cast<float>(processedCount) /
                    static_cast<float>((std::max)(count, 1)));
            }
            const int x = index % w;
            const int y = index / w;
            for (int direction = 0; direction < 8; ++direction) {
                const int nx = x + kHydrologyDx[direction];
                const int ny = y + kHydrologyDy[direction];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                const int neighbor = ny * w + nx;
                if (visited[static_cast<size_t>(neighbor)]) continue;
                visited[static_cast<size_t>(neighbor)] = 1;
                parent[static_cast<size_t>(neighbor)] = index;
                // The tiny deterministic index term prevents equal-height queue
                // entries from producing platform-dependent drainage trees.
                const float tie = static_cast<float>((neighbor * 1664525u + 1013904223u) & 0xffffu) /
                    65535.0f * epsilon * 0.125f;
                filled[static_cast<size_t>(neighbor)] = (std::max)(
                    filled[static_cast<size_t>(neighbor)], filled[static_cast<size_t>(index)] + epsilon + tie);
                frontier.push({filled[static_cast<size_t>(neighbor)], neighbor});
            }
        }

        std::vector<float> accumulation(static_cast<size_t>(count), (std::max)(rainfall, 0.001f));
        for (auto it = visitOrder.rbegin(); it != visitOrder.rend(); ++it) {
            const int index = *it;
            const int downstream = parent[static_cast<size_t>(index)];
            if (downstream >= 0) {
                accumulation[static_cast<size_t>(downstream)] += accumulation[static_cast<size_t>(index)];
            }
        }
        ctx.reportNodeProgress(0.78f);
        const float maximumAccumulation = *(std::max_element)(accumulation.begin(), accumulation.end());

        std::vector<int> basinRoot(static_cast<size_t>(count), -1);
        for (int index : visitOrder) {
            const int downstream = parent[static_cast<size_t>(index)];
            basinRoot[static_cast<size_t>(index)] = downstream < 0
                ? index : basinRoot[static_cast<size_t>(downstream)];
        }

        for (int index = 0; index < count; ++index) {
            // Linear contributing-area fraction is the stable contract consumed
            // by River Network thresholds. UI preview may apply its own display
            // curve without changing the hydrology meaning of this field.
            (*result[1].data)[static_cast<size_t>(index)] = maximumAccumulation > 0.0f
                ? accumulation[static_cast<size_t>(index)] / maximumAccumulation : 0.0f;
            const int downstream = parent[static_cast<size_t>(index)];
            if (downstream >= 0) {
                const int dx = downstream % w - index % w;
                const int dy = downstream / w - index / w;
                for (int direction = 0; direction < 8; ++direction) {
                    if (kHydrologyDx[direction] == dx && kHydrologyDy[direction] == dy) {
                        (*result[2].data)[static_cast<size_t>(index)] =
                            static_cast<float>(direction + 1) / 9.0f;
                        break;
                    }
                }
            }
            const uint32_t root = static_cast<uint32_t>((std::max)(basinRoot[static_cast<size_t>(index)], 0));
            const uint32_t hash = root * 747796405u + 2891336453u;
            (*result[3].data)[static_cast<size_t>(index)] =
                static_cast<float>((hash >> 8u) & 0xffffu) / 65535.0f;
            (*result[4].data)[static_cast<size_t>(index)] =
                accumulation[static_cast<size_t>(index)] / rainfallPerCell * catchmentCellArea;
            if ((index & 0x7fff) == 0 && ctx.isCancelled()) return NodeSystem::PinValue{};
        }

        if (tctx && tctx->terrain && tctx->publishTerrainState) {
            tctx->terrain->flowMap = accumulation;
        }
        for (int i = 0; i < 5; ++i) ctx.setCachedValue(id, i, result[static_cast<size_t>(i)]);
        ctx.reportNodeProgress(1.0f);
        return (outputIndex >= 0 && outputIndex < 5)
            ? NodeSystem::PinValue{result[static_cast<size_t>(outputIndex)]}
            : NodeSystem::PinValue{};
    }

    void WatershedAnalysisNode::drawContent() {
        bool edited = false;
        edited |= ImGui::SliderFloat("Rainfall", &rainfall, 0.01f, 10.0f, "%.2f");
        edited |= ImGui::SliderFloat("Flat Epsilon", &flatEpsilon, 0.0000001f, 0.001f, "%.7f",
                                     ImGuiSliderFlags_Logarithmic);
        ImGui::TextDisabled("Priority flood + deterministic D8");
        if (edited) dirty = true;
    }

    NodeSystem::PinValue LakeBasinNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        pendingWaterBodies.clear();
        const auto original = getHeightInput(0, ctx);
        const auto filled = getHeightInput(1, ctx);
        const auto direction = getHeightInput(2, ctx);
        auto* tctx = getTerrainContext(ctx);
        if (!original.isValid() || !filled.isValid() || !tctx || !tctx->terrain ||
            original.width != filled.width || original.height != filled.height ||
            (direction.isValid() && (direction.width != original.width || direction.height != original.height))) {
            ctx.addError(id, "Lake Basin requires matching original and filled height inputs");
            return NodeSystem::PinValue{};
        }

        const int w = original.width;
        const int h = original.height;
        const int count = w * h;
        const float scaleY = (std::max)(tctx->terrain->heightmap.scale_y, 1e-6f);
        const float cellSizeX = tctx->terrain->heightmap.scale_xz /
            static_cast<float>((std::max)(w - 1, 1));
        const float cellSizeZ = tctx->terrain->heightmap.scale_xz /
            static_cast<float>((std::max)(h - 1, 1));
        const float cellArea = cellSizeX * cellSizeZ;
        const float minimumDepth = minimumDepthMeters / scaleY;

        std::array<NodeSystem::Image2DData, 6> result = {
            createMaskOutput(w, h), createHeightOutput(w, h), createHeightOutput(w, h),
            createMaskOutput(w, h), createMaskOutput(w, h), createMaskOutput(w, h)
        };
        std::vector<uint8_t> candidate(static_cast<size_t>(count), 0u);
        for (int index = 0; index < count; ++index) {
            const float depth = (*filled.data)[static_cast<size_t>(index)] -
                (*original.data)[static_cast<size_t>(index)];
            candidate[static_cast<size_t>(index)] = depth >= minimumDepth ? 1u : 0u;
        }

        struct BasinComponent {
            std::vector<int> cells;
            int spillIndex = -1;
            int outletIndex = -1;
            float surfaceLevel = 0.0f;
            bool closed = true;
        };
        std::vector<BasinComponent> components;
        std::vector<uint8_t> visited(static_cast<size_t>(count), 0u);
        std::vector<int> queue;
        for (int seed = 0; seed < count; ++seed) {
            if (!candidate[static_cast<size_t>(seed)] || visited[static_cast<size_t>(seed)]) continue;
            BasinComponent component;
            queue.clear();
            queue.push_back(seed);
            visited[static_cast<size_t>(seed)] = 1u;
            for (size_t head = 0; head < queue.size(); ++head) {
                const int index = queue[head];
                component.cells.push_back(index);
                const int x = index % w;
                const int y = index / w;
                constexpr int cardinalDirections[4] = {1, 3, 4, 6};
                for (int cardinalIndex = 0; cardinalIndex < 4; ++cardinalIndex) {
                    const int d = cardinalDirections[cardinalIndex];
                    const int nx = x + kHydrologyDx[d];
                    const int ny = y + kHydrologyDy[d];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    const int neighbor = ny * w + nx;
                    if (candidate[static_cast<size_t>(neighbor)] && !visited[static_cast<size_t>(neighbor)]) {
                        visited[static_cast<size_t>(neighbor)] = 1u;
                        queue.push_back(neighbor);
                    }
                }
            }
            components.push_back(std::move(component));
            if ((seed & 0x3fff) == 0 && ctx.isCancelled()) return NodeSystem::PinValue{};
        }

        // Label first, then find the D8 edge where each depression releases
        // into the downstream drainage network. The lowest such edge is the
        // stable spill level; without a direction input the lowest perimeter
        // cell remains a useful closed-basin fallback.
        std::vector<int> componentLabel(static_cast<size_t>(count), -1);
        for (int componentIndex = 0; componentIndex < static_cast<int>(components.size()); ++componentIndex) {
            for (int index : components[static_cast<size_t>(componentIndex)].cells) {
                componentLabel[static_cast<size_t>(index)] = componentIndex;
            }
        }
        for (int componentIndex = 0; componentIndex < static_cast<int>(components.size()); ++componentIndex) {
            BasinComponent& component = components[static_cast<size_t>(componentIndex)];
            float bestLevel = (std::numeric_limits<float>::max)();
            for (int index : component.cells) {
                const int x = index % w;
                const int y = index / w;
                int outlet = -1;
                if (direction.isValid()) {
                    const int d = decodeFlowDirection((*direction.data)[static_cast<size_t>(index)]);
                    if (d >= 0) {
                        const int nx = x + kHydrologyDx[d];
                        const int ny = y + kHydrologyDy[d];
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h) outlet = ny * w + nx;
                    }
                }
                const bool drainsOutside = outlet >= 0 &&
                    componentLabel[static_cast<size_t>(outlet)] != componentIndex;
                bool perimeter = drainsOutside;
                if (!direction.isValid()) {
                    for (int d = 0; d < 8 && !perimeter; ++d) {
                        const int nx = x + kHydrologyDx[d];
                        const int ny = y + kHydrologyDy[d];
                        perimeter = nx < 0 || nx >= w || ny < 0 || ny >= h ||
                            componentLabel[static_cast<size_t>(ny * w + nx)] != componentIndex;
                    }
                }
                if (perimeter && (*filled.data)[static_cast<size_t>(index)] < bestLevel) {
                    bestLevel = (*filled.data)[static_cast<size_t>(index)];
                    component.spillIndex = index;
                    component.outletIndex = drainsOutside ? outlet : -1;
                    component.closed = !drainsOutside;
                }
            }
            if (component.spillIndex < 0 && !component.cells.empty()) {
                component.spillIndex = *std::min_element(component.cells.begin(), component.cells.end(),
                    [&](int a, int b) { return (*filled.data)[static_cast<size_t>(a)] <
                        (*filled.data)[static_cast<size_t>(b)]; });
                bestLevel = (*filled.data)[static_cast<size_t>(component.spillIndex)];
            }
            component.surfaceLevel = bestLevel;
        }

        std::sort(components.begin(), components.end(), [](const BasinComponent& a, const BasinComponent& b) {
            return a.cells.size() > b.cells.size();
        });
        int acceptedCount = 0;
        for (BasinComponent& component : components) {
            if (acceptedCount >= maximumLakes) break;
            if (component.closed && !includeClosedBasins) continue;

            std::vector<int> wetCells;
            wetCells.reserve(component.cells.size());
            for (int index : component.cells) {
                if ((*original.data)[static_cast<size_t>(index)] < component.surfaceLevel) wetCells.push_back(index);
            }
            if (static_cast<float>(wetCells.size()) * cellArea < minimumAreaSquareMeters) continue;

            WaterBodyData body;
            body.sourceNodeId = static_cast<int>(id);
            body.terrainId = tctx->terrain->id;
            body.type = WaterBodyType::Lake;
            body.cellCount = static_cast<int>(wetCells.size());
            body.surfaceElevation = component.surfaceLevel * scaleY;
            body.area = static_cast<float>(wetCells.size()) * cellArea;
            body.closedBasin = component.closed;
            body.spillGridX = component.spillIndex >= 0 ? component.spillIndex % w : -1;
            body.spillGridY = component.spillIndex >= 0 ? component.spillIndex / w : -1;
            body.outletGridX = component.outletIndex >= 0 ? component.outletIndex % w : -1;
            body.outletGridY = component.outletIndex >= 0 ? component.outletIndex / w : -1;
            body.spillPoint = Vec3(body.spillGridX * cellSizeX, body.surfaceElevation,
                                   body.spillGridY * cellSizeZ);
            body.boundsMin = Vec3((std::numeric_limits<float>::max)(), body.surfaceElevation,
                                  (std::numeric_limits<float>::max)());
            body.boundsMax = Vec3((std::numeric_limits<float>::lowest)(), body.surfaceElevation,
                                  (std::numeric_limits<float>::lowest)());
            Vec3 centroidSum(0.0f);
            for (int index : wetCells) {
                const int x = index % w;
                const int y = index / w;
                const float depthNormalized = (std::max)(component.surfaceLevel -
                    (*original.data)[static_cast<size_t>(index)], 0.0f);
                const float depthMeters = depthNormalized * scaleY;
                const float px = x * cellSizeX;
                const float pz = y * cellSizeZ;
                body.maximumDepth = (std::max)(body.maximumDepth, depthMeters);
                body.volume += depthMeters * cellArea;
                centroidSum = centroidSum + Vec3(px, body.surfaceElevation, pz);
                body.boundsMin.x = (std::min)(body.boundsMin.x, px);
                body.boundsMin.z = (std::min)(body.boundsMin.z, pz);
                body.boundsMax.x = (std::max)(body.boundsMax.x, px);
                body.boundsMax.z = (std::max)(body.boundsMax.z, pz);
                (*result[0].data)[static_cast<size_t>(index)] = 1.0f;
                (*result[1].data)[static_cast<size_t>(index)] = depthNormalized;
                (*result[2].data)[static_cast<size_t>(index)] = component.surfaceLevel;
            }
            body.centroid = centroidSum * (1.0f / static_cast<float>((std::max)(body.cellCount, 1)));
            const uint64_t identitySeed = (static_cast<uint64_t>(static_cast<uint32_t>(id)) << 32u) ^
                static_cast<uint64_t>(static_cast<uint32_t>((std::max)(component.spillIndex, 0)));
            body.id = identitySeed * 11400714819323198485ull;
            body.name = "Lake_" + std::to_string(body.id);

            const float encodedId = static_cast<float>(acceptedCount + 1) /
                static_cast<float>((std::max)(maximumLakes + 1, 2));
            body.fieldValue = encodedId;
            for (int index : wetCells) {
                (*result[5].data)[static_cast<size_t>(index)] = encodedId;
                const int x = index % w;
                const int y = index / w;
                bool shoreline = false;
                for (int d = 0; d < 8 && !shoreline; ++d) {
                    const int nx = x + kHydrologyDx[d];
                    const int ny = y + kHydrologyDy[d];
                    shoreline = nx < 0 || nx >= w || ny < 0 || ny >= h ||
                        (*result[0].data)[static_cast<size_t>(ny * w + nx)] < 0.5f;
                }
                if (shoreline) (*result[3].data)[static_cast<size_t>(index)] = 1.0f;
            }
            if (component.spillIndex >= 0) {
                (*result[4].data)[static_cast<size_t>(component.spillIndex)] = 1.0f;
            }
            pendingWaterBodies.push_back(std::move(body));
            ++acceptedCount;
        }

        for (int i = 0; i < 6; ++i) ctx.setCachedValue(id, i, result[static_cast<size_t>(i)]);
        ctx.reportNodeProgress(1.0f);
        return (outputIndex >= 0 && outputIndex < 6)
            ? NodeSystem::PinValue{result[static_cast<size_t>(outputIndex)]}
            : NodeSystem::PinValue{};
    }

    void LakeBasinNode::publishWaterBodies(TerrainObject* terrain) const {
        if (!terrain) return;
        terrain->waterBodies.erase(
            std::remove_if(terrain->waterBodies.begin(), terrain->waterBodies.end(),
                [&](const WaterBodyData& body) { return body.sourceNodeId == static_cast<int>(id); }),
            terrain->waterBodies.end());
        terrain->waterBodies.insert(terrain->waterBodies.end(),
                                    pendingWaterBodies.begin(), pendingWaterBodies.end());
    }

    void LakeBasinNode::drawContent() {
        bool edited = false;
        edited |= ImGui::SliderFloat("Min Depth", &minimumDepthMeters, 0.01f, 10.0f, "%.2f m",
                                     ImGuiSliderFlags_Logarithmic);
        edited |= ImGui::SliderFloat("Min Area", &minimumAreaSquareMeters, 0.1f, 100000.0f, "%.1f m2",
                                     ImGuiSliderFlags_Logarithmic);
        edited |= ImGui::SliderInt("Max Lakes", &maximumLakes, 1, 512);
        edited |= ImGui::Checkbox("Include Closed Basins", &includeClosedBasins);
        ImGui::TextDisabled("Detected lakes: %d", static_cast<int>(pendingWaterBodies.size()));
        ImGui::TextDisabled("Level + storage + spill/outlet contract");
        if (edited) dirty = true;
    }

    NodeSystem::PinValue LakeSurfaceOutputNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        (void)outputIndex;
        for (int i = 0; i < 4; ++i) pendingFields[static_cast<size_t>(i)] = getHeightInput(i, ctx);
        const auto& mask = pendingFields[0];
        if (!mask.isValid()) {
            ctx.addError(id, "Lake Surface Output requires Lake Basin fields");
            return NodeSystem::PinValue{};
        }
        for (int i = 1; i < 4; ++i) {
            const auto& field = pendingFields[static_cast<size_t>(i)];
            if (!field.isValid() || field.width != mask.width || field.height != mask.height) {
                ctx.addError(id, "Lake Surface Output inputs must have matching resolutions");
                return NodeSystem::PinValue{};
            }
        }
        return NodeSystem::PinValue{};
    }

    bool LakeSurfaceOutputNode::applyGeneratedLakes(SceneData& scene, TerrainObject* terrain) {
        if (!terrain) return false;
        auto& waterManager = WaterManager::getInstance();
        const std::string ownershipPrefix =
            "AutoLake_T" + std::to_string(terrain->id) + "_N" + std::to_string(id) + "_";

        std::vector<int> surfacesToRemove = generatedWaterSurfaceIds;
        for (const auto& surface : waterManager.getWaterSurfaces()) {
            if (surface.name.rfind(ownershipPrefix, 0) == 0) surfacesToRemove.push_back(surface.id);
        }
        std::sort(surfacesToRemove.begin(), surfacesToRemove.end());
        surfacesToRemove.erase(std::unique(surfacesToRemove.begin(), surfacesToRemove.end()), surfacesToRemove.end());
        bool changed = false;
        for (int surfaceId : surfacesToRemove) {
            if (waterManager.getWaterSurface(surfaceId)) {
                waterManager.removeWaterSurface(scene, surfaceId);
                changed = true;
            }
        }
        generatedWaterSurfaceIds.clear();
        for (auto& body : terrain->waterBodies) {
            if ((sourceLakeNodeId < 0 || body.sourceNodeId == sourceLakeNodeId) &&
                body.type == WaterBodyType::Lake) body.waterSurfaceId = -1;
        }
        if (!generateWaterMeshes || !pendingFields[0].isValid()) return changed;

        const int w = pendingFields[0].width;
        const int h = pendingFields[0].height;
        if (w < 2 || h < 2) return changed;
        const float cellSizeX = terrain->heightmap.scale_xz / static_cast<float>(w - 1);
        const float cellSizeZ = terrain->heightmap.scale_xz / static_cast<float>(h - 1);
        const float scaleY = (std::max)(terrain->heightmap.scale_y, 1e-6f);
        const auto& mask = *pendingFields[0].data;
        const auto& depth = *pendingFields[1].data;
        const auto& lakeIds = *pendingFields[3].data;

        using HalfPoint = std::pair<int, int>;
        const HalfPoint c0{0, 0}, c1{2, 0}, c2{2, 2}, c3{0, 2};
        const HalfPoint e0{1, 0}, e1{2, 1}, e2{1, 2}, e3{0, 1};
        const auto polygonsForCase = [&](int code) {
            std::vector<std::vector<HalfPoint>> polygons;
            switch (code) {
                case 1: polygons = {{c0, e0, e3}}; break;
                case 2: polygons = {{c1, e1, e0}}; break;
                case 3: polygons = {{c0, c1, e1, e3}}; break;
                case 4: polygons = {{c2, e2, e1}}; break;
                case 5: polygons = {{c0, e0, e3}, {c2, e2, e1}}; break;
                case 6: polygons = {{e0, c1, c2, e2}}; break;
                case 7: polygons = {{c0, c1, c2, e2, e3}}; break;
                case 8: polygons = {{c3, e3, e2}}; break;
                case 9: polygons = {{c0, e0, e2, c3}}; break;
                case 10: polygons = {{c1, e1, e0}, {c3, e3, e2}}; break;
                case 11: polygons = {{c0, c1, e1, e2, c3}}; break;
                case 12: polygons = {{e3, e1, c2, c3}}; break;
                case 13: polygons = {{c0, e0, e1, c2, c3}}; break;
                case 14: polygons = {{e0, c1, c2, c3, e3}}; break;
                case 15: polygons = {{c0, c1, c2, c3}}; break;
                default: break;
            }
            return polygons;
        };

        int generatedCount = 0;
        for (auto& body : terrain->waterBodies) {
            if (generatedCount >= maximumGeneratedLakes) break;
            if (body.sourceNodeId < 0 ||
                (sourceLakeNodeId >= 0 && body.sourceNodeId != sourceLakeNodeId) ||
                body.type != WaterBodyType::Lake || body.fieldValue <= 0.0f) continue;

            std::vector<Vec3> positions;
            std::vector<Vec2> uvs;
            std::vector<float> vertexDepth;
            std::vector<float> shoreFactor;
            std::vector<uint32_t> indices;
            std::unordered_map<uint64_t, uint32_t> vertexCache;
            const auto isInside = [&](int x, int y) {
                if (x < 0 || x >= w || y < 0 || y >= h) return false;
                const size_t index = static_cast<size_t>(y) * w + x;
                return mask[index] >= 0.5f && std::fabs(lakeIds[index] - body.fieldValue) <= 1e-5f;
            };
            const auto sampleDepth = [&](int halfX, int halfY) {
                const int x0 = (std::max)(0, (std::min)(halfX / 2, w - 1));
                const int y0 = (std::max)(0, (std::min)(halfY / 2, h - 1));
                const int x1 = (std::min)(x0 + (halfX & 1), w - 1);
                const int y1 = (std::min)(y0 + (halfY & 1), h - 1);
                const float fx = (halfX & 1) ? 0.5f : 0.0f;
                const float fy = (halfY & 1) ? 0.5f : 0.0f;
                const auto at = [&](int x, int y) { return depth[static_cast<size_t>(y) * w + x]; };
                const float a = at(x0, y0) * (1.0f - fx) + at(x1, y0) * fx;
                const float b = at(x0, y1) * (1.0f - fx) + at(x1, y1) * fx;
                return (a * (1.0f - fy) + b * fy) * scaleY;
            };
            const auto getVertex = [&](int halfX, int halfY) -> uint32_t {
                const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(halfY)) << 32u) |
                    static_cast<uint32_t>(halfX);
                const auto found = vertexCache.find(key);
                if (found != vertexCache.end()) return found->second;
                const float px = static_cast<float>(halfX) * 0.5f * cellSizeX;
                const float pz = static_cast<float>(halfY) * 0.5f * cellSizeZ;
                const uint32_t index = static_cast<uint32_t>(positions.size());
                positions.emplace_back(px, body.surfaceElevation + surfaceOffsetMeters, pz);
                uvs.emplace_back(px / uvScaleMeters, pz / uvScaleMeters);
                vertexDepth.push_back(sampleDepth(halfX, halfY));
                const bool edgeVertex = (halfX & 1) || (halfY & 1);
                bool shore = edgeVertex;
                if (!shore) {
                    const int gx = halfX / 2;
                    const int gy = halfY / 2;
                    shore = !isInside(gx - 1, gy) || !isInside(gx + 1, gy) ||
                            !isInside(gx, gy - 1) || !isInside(gx, gy + 1);
                }
                shoreFactor.push_back(shore ? 1.0f : 0.0f);
                vertexCache.emplace(key, index);
                return index;
            };

            for (int y = 0; y < h - 1; ++y) {
                for (int x = 0; x < w - 1; ++x) {
                    const int code = (isInside(x, y) ? 1 : 0) |
                                     (isInside(x + 1, y) ? 2 : 0) |
                                     (isInside(x + 1, y + 1) ? 4 : 0) |
                                     (isInside(x, y + 1) ? 8 : 0);
                    if (code == 0) continue;
                    auto polygons = polygonsForCase(code);
                    for (auto& polygon : polygons) {
                        for (auto& point : polygon) {
                            point.first += x * 2;
                            point.second += y * 2;
                        }
                        float signedArea = 0.0f;
                        for (size_t i = 0; i < polygon.size(); ++i) {
                            const auto& a = polygon[i];
                            const auto& b = polygon[(i + 1) % polygon.size()];
                            signedArea += static_cast<float>(a.first * b.second - b.first * a.second);
                        }
                        if (signedArea > 0.0f) std::reverse(polygon.begin(), polygon.end());
                        const uint32_t first = getVertex(polygon[0].first, polygon[0].second);
                        for (size_t i = 1; i + 1 < polygon.size(); ++i) {
                            indices.push_back(first);
                            indices.push_back(getVertex(polygon[i].first, polygon[i].second));
                            indices.push_back(getVertex(polygon[i + 1].first, polygon[i + 1].second));
                        }
                    }
                }
            }
            if (indices.empty()) continue;

            WaterWaveParams lakeParams;
            lakeParams.applyPreset(WaterWaveParams::WaterPreset::Lake);
            lakeParams.use_fft_mesh_displacement = false;
            lakeParams.depth_max = (std::max)(body.maximumDepth, 0.25f);
            lakeParams.fft_ocean_size = (std::max)(
                (std::max)(body.boundsMax.x - body.boundsMin.x, body.boundsMax.z - body.boundsMin.z), 1.0f);
            lakeParams.shore_foam_distance = (std::max)(cellSizeX, cellSizeZ) * 1.5f;

            std::shared_ptr<Transform> transform = terrain->transform;
            if (!transform) transform = std::make_shared<Transform>();
            const std::string surfaceName = ownershipPrefix + std::to_string(body.id);
            WaterSurface* surface = waterManager.createWaterFromIndexedMesh(
                scene, surfaceName, positions, uvs, indices, transform,
                WaterSurface::Type::Lake, lakeParams, vertexDepth, shoreFactor);
            if (!surface) continue;
            body.waterSurfaceId = surface->id;
            generatedWaterSurfaceIds.push_back(surface->id);
            ++generatedCount;
            changed = true;
        }
        return changed;
    }

    void LakeSurfaceOutputNode::drawContent() {
        bool edited = false;
        edited |= ImGui::Checkbox("Generate Water Meshes", &generateWaterMeshes);
        edited |= ImGui::SliderFloat("Surface Offset", &surfaceOffsetMeters, -0.25f, 0.25f, "%.3f m");
        edited |= ImGui::SliderFloat("UV Scale", &uvScaleMeters, 0.1f, 100.0f, "%.2f m",
                                     ImGuiSliderFlags_Logarithmic);
        edited |= ImGui::SliderInt("Max Lakes", &maximumGeneratedLakes, 1, 256);
        ImGui::TextDisabled("Owned surfaces: %d", static_cast<int>(generatedWaterSurfaceIds.size()));
        ImGui::TextDisabled("Marching-squares shoreline mesh");
        if (edited) dirty = true;
    }

    NodeSystem::PinValue RiverNetworkNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto accumulation = getHeightInput(0, ctx);
        const auto direction = getHeightInput(1, ctx);
        if (!accumulation.isValid() || !direction.isValid() ||
            accumulation.width != direction.width || accumulation.height != direction.height) {
            ctx.addError(id, "River Network requires matching accumulation and direction inputs");
            return NodeSystem::PinValue{};
        }

        const int w = accumulation.width;
        const int h = accumulation.height;
        const int count = w * h;
        std::vector<int> parent(static_cast<size_t>(count), -1);
        std::vector<uint8_t> active(static_cast<size_t>(count), 0);
        for (int index = 0; index < count; ++index) {
            const int d = decodeFlowDirection((*direction.data)[static_cast<size_t>(index)]);
            if (d >= 0) {
                const int nx = index % w + kHydrologyDx[d];
                const int ny = index / w + kHydrologyDy[d];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) parent[static_cast<size_t>(index)] = ny * w + nx;
            }
            active[static_cast<size_t>(index)] =
                (*accumulation.data)[static_cast<size_t>(index)] >= catchmentThreshold ? 1u : 0u;
            if ((index & 0x7fff) == 0 && ctx.isCancelled()) return NodeSystem::PinValue{};
        }

        // Remove short source twigs without damaging their downstream junction.
        for (int pruningPass = 0; pruningPass < 4; ++pruningPass) {
            std::vector<int> incoming(static_cast<size_t>(count), 0);
            for (int index = 0; index < count; ++index) {
                const int downstream = parent[static_cast<size_t>(index)];
                if (active[static_cast<size_t>(index)] && downstream >= 0 && active[static_cast<size_t>(downstream)]) {
                    ++incoming[static_cast<size_t>(downstream)];
                }
            }
            bool removedAny = false;
            for (int source = 0; source < count; ++source) {
                if (!active[static_cast<size_t>(source)] || incoming[static_cast<size_t>(source)] != 0) continue;
                std::vector<int> branch;
                int cursor = source;
                while (cursor >= 0 && active[static_cast<size_t>(cursor)] &&
                       static_cast<int>(branch.size()) < minimumBranchLength) {
                    branch.push_back(cursor);
                    const int downstream = parent[static_cast<size_t>(cursor)];
                    if (downstream < 0 || !active[static_cast<size_t>(downstream)] ||
                        incoming[static_cast<size_t>(downstream)] > 1) break;
                    cursor = downstream;
                }
                const int endpoint = branch.empty() ? -1 : branch.back();
                const bool reachedJunction = endpoint >= 0 && parent[static_cast<size_t>(endpoint)] >= 0 &&
                    incoming[static_cast<size_t>(parent[static_cast<size_t>(endpoint)])] > 1;
                if (static_cast<int>(branch.size()) < minimumBranchLength && reachedJunction) {
                    for (int index : branch) active[static_cast<size_t>(index)] = 0;
                    removedAny = true;
                }
            }
            if (!removedAny) break;
        }

        std::vector<int> incoming(static_cast<size_t>(count), 0);
        for (int index = 0; index < count; ++index) {
            const int downstream = parent[static_cast<size_t>(index)];
            if (active[static_cast<size_t>(index)] && downstream >= 0 && active[static_cast<size_t>(downstream)]) {
                ++incoming[static_cast<size_t>(downstream)];
            }
        }
        std::queue<int> ready;
        std::vector<int> remainingIncoming = incoming;
        std::vector<int> order(static_cast<size_t>(count), 0);
        std::vector<int> highestUpstreamOrder(static_cast<size_t>(count), 0);
        std::vector<int> highestOrderCount(static_cast<size_t>(count), 0);
        for (int index = 0; index < count; ++index) {
            if (active[static_cast<size_t>(index)] && incoming[static_cast<size_t>(index)] == 0) {
                order[static_cast<size_t>(index)] = 1;
                ready.push(index);
            }
        }
        int maximumOrder = 1;
        while (!ready.empty()) {
            const int index = ready.front();
            ready.pop();
            maximumOrder = (std::max)(maximumOrder, order[static_cast<size_t>(index)]);
            const int downstream = parent[static_cast<size_t>(index)];
            if (downstream < 0 || !active[static_cast<size_t>(downstream)]) continue;
            const int currentOrder = order[static_cast<size_t>(index)];
            int& best = highestUpstreamOrder[static_cast<size_t>(downstream)];
            int& bestCount = highestOrderCount[static_cast<size_t>(downstream)];
            if (currentOrder > best) { best = currentOrder; bestCount = 1; }
            else if (currentOrder == best) { ++bestCount; }
            if (--remainingIncoming[static_cast<size_t>(downstream)] == 0) {
                order[static_cast<size_t>(downstream)] = best + (bestCount >= 2 ? 1 : 0);
                ready.push(downstream);
            }
        }

        std::array<NodeSystem::Image2DData, 3> result = {
            createMaskOutput(w, h), createMaskOutput(w, h), createMaskOutput(w, h)
        };
        for (int index = 0; index < count; ++index) {
            if (!active[static_cast<size_t>(index)]) continue;
            (*result[0].data)[static_cast<size_t>(index)] =
                (*accumulation.data)[static_cast<size_t>(index)];
            (*result[1].data)[static_cast<size_t>(index)] =
                static_cast<float>(order[static_cast<size_t>(index)]) / static_cast<float>(maximumOrder);
            (*result[2].data)[static_cast<size_t>(index)] =
                incoming[static_cast<size_t>(index)] == 0 ? 1.0f : 0.0f;
        }
        for (int i = 0; i < 3; ++i) ctx.setCachedValue(id, i, result[static_cast<size_t>(i)]);
        ctx.reportNodeProgress(1.0f);
        return (outputIndex >= 0 && outputIndex < 3)
            ? NodeSystem::PinValue{result[static_cast<size_t>(outputIndex)]}
            : NodeSystem::PinValue{};
    }

    void RiverNetworkNode::drawContent() {
        bool edited = false;
        edited |= ImGui::SliderFloat("Catchment", &catchmentThreshold, 0.0001f, 0.25f, "%.4f",
                                     ImGuiSliderFlags_Logarithmic);
        edited |= ImGui::SliderInt("Min Branch", &minimumBranchLength, 2, 128);
        ImGui::TextDisabled("Strahler ordered channel graph");
        if (edited) dirty = true;
    }

    NodeSystem::PinValue RiverHydraulicsNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto bed = getHeightInput(0, ctx);
        const auto catchment = getHeightInput(1, ctx);
        const auto direction = getHeightInput(2, ctx);
        const auto channels = getHeightInput(3, ctx);
        const auto lakeMask = getHeightInput(4, ctx);
        const auto lakeLevel = getHeightInput(5, ctx);
        const auto referenceHeight = getHeightInput(6, ctx);
        auto* tctx = getTerrainContext(ctx);
        const auto matches = [&](const NodeSystem::Image2DData& image) {
            return image.isValid() && image.width == bed.width && image.height == bed.height;
        };
        if (!bed.isValid() || !matches(catchment) || !matches(direction) || !matches(channels) ||
            !tctx || !tctx->terrain ||
            (lakeMask.isValid() && !matches(lakeMask)) || (lakeLevel.isValid() && !matches(lakeLevel)) ||
            (referenceHeight.isValid() && !matches(referenceHeight))) {
            ctx.addError(id, "River Hydraulics requires matching bed, catchment, direction and channel fields");
            return NodeSystem::PinValue{};
        }

        const int w = bed.width;
        const int h = bed.height;
        const int count = w * h;
        const float scaleY = (std::max)(tctx->terrain->heightmap.scale_y, 1e-6f);
        const float cellX = tctx->terrain->heightmap.scale_xz /
            static_cast<float>((std::max)(w - 1, 1));
        const float cellZ = tctx->terrain->heightmap.scale_xz /
            static_cast<float>((std::max)(h - 1, 1));
        const float rainfallMetersPerSecond =
            (std::max)(rainfallMillimetersPerHour, 0.0f) / 3600000.0f;
        const float runoff = clampValue(runoffCoefficient, 0.0f, 1.0f);
        const float roughness = (std::max)(manningRoughness, 0.005f);
        const float sideSlope = (std::max)(bankSideSlope, 0.0f);

        std::array<NodeSystem::Image2DData, 7> result = {
            createHeightOutput(w, h), createHeightOutput(w, h), createHeightOutput(w, h),
            createHeightOutput(w, h), createHeightOutput(w, h), createHeightOutput(w, h),
            createHeightOutput(w, h)
        };
        std::vector<int> parent(static_cast<size_t>(count), -1);
        std::vector<int> indegree(static_cast<size_t>(count), 0);
        std::vector<uint8_t> active(static_cast<size_t>(count), 0u);
        std::vector<float> bedSlope(static_cast<size_t>(count), 0.0f);
        std::vector<float> downstreamDistance(static_cast<size_t>(count), 0.0f);

        for (int index = 0; index < count; ++index) {
            if ((*channels.data)[static_cast<size_t>(index)] <= 0.0f) continue;
            active[static_cast<size_t>(index)] = 1u;
            const int d = decodeFlowDirection((*direction.data)[static_cast<size_t>(index)]);
            if (d < 0) continue;
            const int x = index % w;
            const int y = index / w;
            const int nx = x + kHydrologyDx[d];
            const int ny = y + kHydrologyDy[d];
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            parent[static_cast<size_t>(index)] = ny * w + nx;
            downstreamDistance[static_cast<size_t>(index)] = std::sqrt(
                static_cast<float>(kHydrologyDx[d] * kHydrologyDx[d]) * cellX * cellX +
                static_cast<float>(kHydrologyDy[d] * kHydrologyDy[d]) * cellZ * cellZ);
        }
        for (int index = 0; index < count; ++index) {
            const int downstream = parent[static_cast<size_t>(index)];
            if (active[static_cast<size_t>(index)] && downstream >= 0 &&
                active[static_cast<size_t>(downstream)]) {
                ++indegree[static_cast<size_t>(downstream)];
            }
        }

        std::vector<int> order;
        order.reserve(static_cast<size_t>(count));
        std::deque<int> queue;
        for (int index = 0; index < count; ++index) {
            if (active[static_cast<size_t>(index)] && indegree[static_cast<size_t>(index)] == 0) {
                queue.push_back(index);
            }
        }
        while (!queue.empty()) {
            const int index = queue.front();
            queue.pop_front();
            order.push_back(index);
            const int downstream = parent[static_cast<size_t>(index)];
            if (downstream >= 0 && active[static_cast<size_t>(downstream)] &&
                --indegree[static_cast<size_t>(downstream)] == 0) {
                queue.push_back(downstream);
            }
        }
        const size_t activeCount = static_cast<size_t>(
            std::count(active.begin(), active.end(), static_cast<uint8_t>(1u)));
        if (order.size() < activeCount) {
            std::vector<uint8_t> inOrder(static_cast<size_t>(count), 0u);
            for (int index : order) inOrder[static_cast<size_t>(index)] = 1u;
            for (int index = 0; index < count; ++index) {
                if (active[static_cast<size_t>(index)] && !inOrder[static_cast<size_t>(index)]) {
                    parent[static_cast<size_t>(index)] = -1;
                    order.push_back(index);
                }
            }
        }

        const auto sectionAtDepth = [&](float width, float depth, float slope,
                                        float& area, float& topWidth, float& velocity) {
            area = depth * (width + sideSlope * depth);
            topWidth = width + 2.0f * sideSlope * depth;
            const float perimeter = width + 2.0f * depth * std::sqrt(1.0f + sideSlope * sideSlope);
            const float hydraulicRadius = area / (std::max)(perimeter, 1e-6f);
            velocity = (1.0f / roughness) * std::pow((std::max)(hydraulicRadius, 1e-6f), 2.0f / 3.0f) *
                std::sqrt((std::max)(slope, minimumBedSlope));
            return area * velocity;
        };

        for (size_t ordinal = 0; ordinal < order.size(); ++ordinal) {
            const int index = order[ordinal];
            const int downstream = parent[static_cast<size_t>(index)];
            const bool inLake = lakeMask.isValid() &&
                (*lakeMask.data)[static_cast<size_t>(index)] > 0.5f;
            const float discharge = (std::max)(0.0f,
                (*catchment.data)[static_cast<size_t>(index)] * rainfallMetersPerSecond * runoff * dischargeScale);
            (*result[0].data)[static_cast<size_t>(index)] = discharge;

            float slope = minimumBedSlope;
            if (downstream >= 0 && downstreamDistance[static_cast<size_t>(index)] > 1e-6f) {
                const float dropMeters = ((*bed.data)[static_cast<size_t>(index)] -
                    (*bed.data)[static_cast<size_t>(downstream)]) * scaleY;
                slope = (std::max)(dropMeters / downstreamDistance[static_cast<size_t>(index)], minimumBedSlope);
            }
            bedSlope[static_cast<size_t>(index)] = slope;
            const float width = clampValue(widthCoefficient *
                std::pow((std::max)(discharge, 1e-6f), widthExponent),
                minimumWidthMeters, maximumWidthMeters);
            float low = minimumDepthMeters;
            float high = maximumDepthMeters;
            for (int iteration = 0; iteration < 28; ++iteration) {
                const float depth = 0.5f * (low + high);
                float area = 0.0f, topWidth = 0.0f, velocity = 0.0f;
                const float solvedDischarge = sectionAtDepth(width, depth, slope, area, topWidth, velocity);
                if (solvedDischarge < discharge) low = depth;
                else high = depth;
            }
            const float depth = 0.5f * (low + high);
            float area = 0.0f, topWidth = 0.0f, ignoredVelocity = 0.0f;
            sectionAtDepth(width, depth, slope, area, topWidth, ignoredVelocity);
            const float velocity = discharge / (std::max)(area, 1e-6f);
            const float hydraulicDepth = area / (std::max)(topWidth, 1e-6f);
            const float froude = velocity / std::sqrt(9.80665f * (std::max)(hydraulicDepth, 1e-4f));
            const float foam = clampValue((froude - 0.55f) * 0.9f +
                std::sqrt((std::max)(slope, 0.0f)) * 3.5f, 0.0f, 1.0f);
            (*result[1].data)[static_cast<size_t>(index)] = topWidth;
            (*result[2].data)[static_cast<size_t>(index)] = depth;
            (*result[3].data)[static_cast<size_t>(index)] = inLake ? 0.0f : velocity;
            const float existingIncision = referenceHeight.isValid()
                ? (std::max)(((*referenceHeight.data)[static_cast<size_t>(index)] -
                    (*bed.data)[static_cast<size_t>(index)]) * scaleY, 0.0f)
                : 0.0f;
            const float freeboard = clampValue(
                depth * bankFreeboardRatio, minimumFreeboardMeters, maximumFreeboardMeters);
            const float targetIncision = depth + freeboard;
            const float missingIncision = (std::max)(targetIncision - existingIncision, 0.0f);
            const float finalBedHeight = (*bed.data)[static_cast<size_t>(index)] -
                missingIncision / scaleY;
            (*result[4].data)[static_cast<size_t>(index)] = inLake && lakeLevel.isValid()
                ? (*lakeLevel.data)[static_cast<size_t>(index)]
                : finalBedHeight + (depth + surfaceOffsetMeters) / scaleY;
            (*result[5].data)[static_cast<size_t>(index)] = inLake ? 0.0f : froude;
            (*result[6].data)[static_cast<size_t>(index)] = inLake ? 0.0f : foam;
            if ((ordinal & 0x3fffu) == 0u && ctx.isCancelled()) return NodeSystem::PinValue{};
        }

        // Solve the free surface from outlets toward sources, with lakes acting
        // as fixed-level controls. This removes local uphill water surfaces.
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const int index = *it;
            const bool inLake = lakeMask.isValid() &&
                (*lakeMask.data)[static_cast<size_t>(index)] > 0.5f;
            if (inLake) continue;
            const int downstream = parent[static_cast<size_t>(index)];
            if (downstream < 0 || !active[static_cast<size_t>(downstream)]) continue;
            const float requiredStage = (*result[4].data)[static_cast<size_t>(downstream)] +
                minimumSurfaceSlope * downstreamDistance[static_cast<size_t>(index)] / scaleY;
            const float maximumStage = (*bed.data)[static_cast<size_t>(index)] + maximumDepthMeters / scaleY;
            (*result[4].data)[static_cast<size_t>(index)] =
                (std::min)((std::max)((*result[4].data)[static_cast<size_t>(index)], requiredStage), maximumStage);
        }

        for (int i = 0; i < 7; ++i) ctx.setCachedValue(id, i, result[static_cast<size_t>(i)]);
        ctx.reportNodeProgress(1.0f);
        return outputIndex >= 0 && outputIndex < 7
            ? NodeSystem::PinValue{result[static_cast<size_t>(outputIndex)]}
            : NodeSystem::PinValue{};
    }

    void RiverHydraulicsNode::drawContent() {
        bool edited = false;
        ImGui::TextDisabled("Hydrology");
        edited |= ImGui::SliderFloat("Rainfall", &rainfallMillimetersPerHour, 0.0f, 500.0f, "%.1f mm/h");
        edited |= ImGui::SliderFloat("Runoff", &runoffCoefficient, 0.0f, 1.0f, "%.2f");
        edited |= ImGui::SliderFloat("Discharge Scale", &dischargeScale, 0.05f, 20.0f, "%.2f",
                                     ImGuiSliderFlags_Logarithmic);
        ImGui::Separator();
        ImGui::TextDisabled("Channel section");
        edited |= ImGui::SliderFloat("Manning n", &manningRoughness, 0.01f, 0.15f, "%.3f");
        edited |= ImGui::SliderFloat("Width Coefficient", &widthCoefficient, 0.1f, 20.0f, "%.2f");
        edited |= ImGui::SliderFloat("Width Exponent", &widthExponent, 0.2f, 0.8f, "%.2f");
        edited |= ImGui::SliderFloat("Min Width", &minimumWidthMeters, 0.05f, 10.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Max Width", &maximumWidthMeters, minimumWidthMeters, 500.0f, "%.1f m");
        edited |= ImGui::SliderFloat("Min Depth", &minimumDepthMeters, 0.01f, 2.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Max Depth", &maximumDepthMeters, minimumDepthMeters, 50.0f, "%.1f m");
        edited |= ImGui::SliderFloat("Bank Side Slope", &bankSideSlope, 0.0f, 5.0f, "%.2f H:V");
        edited |= ImGui::SliderFloat("Min Bed Slope", &minimumBedSlope, 0.000001f, 0.01f, "%.6f",
                                     ImGuiSliderFlags_Logarithmic);
        edited |= ImGui::SliderFloat("Min Surface Slope", &minimumSurfaceSlope, 0.0f, 0.002f, "%.6f");
        edited |= ImGui::SliderFloat("Surface Offset", &surfaceOffsetMeters, 0.0f, 1.0f, "%.3f m");
        edited |= ImGui::SliderFloat("Bank Freeboard", &bankFreeboardRatio, 0.0f, 1.5f, "%.2f x depth");
        edited |= ImGui::SliderFloat("Min Freeboard", &minimumFreeboardMeters, 0.0f, 1.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Max Freeboard", &maximumFreeboardMeters,
                                     minimumFreeboardMeters, 10.0f, "%.2f m");
        ImGui::TextDisabled("Outputs: m3/s, m, m/s, stage, Froude, foam");
        if (edited) dirty = true;
    }

    NodeSystem::PinValue RiverBedCarveNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto channels = getHeightInput(1, ctx);
        const auto hydraulicWidth = getHeightInput(2, ctx);
        const auto hydraulicDepth = getHeightInput(3, ctx);
        const auto referenceHeight = getHeightInput(4, ctx);
        const auto waterLevel = getHeightInput(5, ctx);
        auto* tctx = getTerrainContext(ctx);
        if (!height.isValid() || !channels.isValid() || !tctx ||
            height.width != channels.width || height.height != channels.height ||
            (hydraulicWidth.isValid() &&
                (height.width != hydraulicWidth.width || height.height != hydraulicWidth.height)) ||
            (hydraulicDepth.isValid() &&
                (height.width != hydraulicDepth.width || height.height != hydraulicDepth.height)) ||
            (referenceHeight.isValid() &&
                (height.width != referenceHeight.width || height.height != referenceHeight.height)) ||
            (waterLevel.isValid() &&
                (height.width != waterLevel.width || height.height != waterLevel.height))) {
            ctx.addError(id, "River Bed Carve requires matching height and channel inputs");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const float cellX = (std::max)(tctx->scale_xz / static_cast<float>((std::max)(w - 1, 1)), 1e-5f);
        const float cellZ = (std::max)(tctx->scale_xz / static_cast<float>((std::max)(h - 1, 1)), 1e-5f);
        const float heightScale = (std::max)(tctx->scale_y, 1e-5f);
        const bool useHydraulicWidth = hydraulicWidth.isValid();
        const bool useHydraulicDepth = hydraulicDepth.isValid();
        const float safeMaximumWidth = useHydraulicWidth
            ? tctx->scale_xz * 0.08f : (std::min)(maximumWidth, tctx->scale_xz * 0.04f);
        const float safeMaximumDepth = useHydraulicDepth
            ? heightScale * 0.10f : (std::min)(maximumDepth, heightScale * 0.10f);
        std::vector<float> carveDepth(static_cast<size_t>(w) * h, 0.0f);
        std::vector<float> riverBedDepth(static_cast<size_t>(w) * h, 0.0f);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t channelIndex = static_cast<size_t>(y) * w + x;
                const float strength = clampValue((*channels.data)[channelIndex], 0.0f, 1.0f);
                if (strength <= 0.0f) continue;
                const float scale = std::sqrt(strength);
                const float safeMinimumWidth = (std::min)(minimumWidth, safeMaximumWidth);
                const float safeMinimumDepth = (std::min)(minimumDepth, safeMaximumDepth);
                const float fallbackWidth = safeMinimumWidth +
                    (safeMaximumWidth - safeMinimumWidth) * scale;
                const float fallbackDepth = safeMinimumDepth +
                    (safeMaximumDepth - safeMinimumDepth) * scale;
                // A small bank margin keeps the top-width water ribbon inside
                // the carved cross section instead of clipping the terrain.
                const float width = useHydraulicWidth
                    ? clampValue((*hydraulicWidth.data)[channelIndex] * 1.12f,
                                 safeMinimumWidth, safeMaximumWidth)
                    : fallbackWidth;
                const float depth = useHydraulicDepth
                    ? clampValue((*hydraulicDepth.data)[channelIndex],
                                 safeMinimumDepth, safeMaximumDepth)
                    : fallbackDepth;
                const float radius = width * 0.5f;
                const float waterRadius = useHydraulicWidth
                    ? radius / 1.12f : radius;
                const int radiusX = (std::max)(1, static_cast<int>(std::ceil(radius / cellX)));
                const int radiusZ = (std::max)(1, static_cast<int>(std::ceil(radius / cellZ)));
                for (int oy = -radiusZ; oy <= radiusZ; ++oy) {
                    const int py = y + oy;
                    if (py < 0 || py >= h) continue;
                    for (int ox = -radiusX; ox <= radiusX; ++ox) {
                        const int px = x + ox;
                        if (px < 0 || px >= w) continue;
                        const float distance = std::sqrt(
                            ox * cellX * ox * cellX + oy * cellZ * oy * cellZ);
                        if (distance > radius) continue;
                        const float channelT = clampValue(
                            distance / (std::max)(waterRadius, 1e-5f), 0.0f, 1.0f);
                        const float core = 1.0f - channelT * channelT;
                        const float exponent = 1.0f + (1.0f - bankSoftness) * 5.0f;
                        const float profile = std::pow((std::max)(core, 0.0f), exponent);
                        const size_t target = static_cast<size_t>(py) * w + px;
                        const float waterLevelIncision = referenceHeight.isValid() && waterLevel.isValid()
                            ? (std::max)(((*referenceHeight.data)[target] -
                                (*waterLevel.data)[channelIndex]) * heightScale, 0.0f)
                            : 0.0f;
                        float desiredDepth = waterLevelIncision + depth * profile;
                        if (distance > waterRadius && radius > waterRadius + 1e-5f) {
                            const float apronT = clampValue(
                                (distance - waterRadius) / (radius - waterRadius), 0.0f, 1.0f);
                            const float apron = 1.0f - apronT * apronT * (3.0f - 2.0f * apronT);
                            desiredDepth = waterLevelIncision * apron;
                        }
                        const float existingIncision = referenceHeight.isValid()
                            ? (std::max)(((*referenceHeight.data)[target] - (*height.data)[target]) * heightScale, 0.0f)
                            : 0.0f;
                        // Fluvial erosion may already have produced some or all
                        // of this cross section. Subtract only the missing depth.
                        carveDepth[target] = (std::max)(
                            carveDepth[target], (std::max)(desiredDepth - existingIncision, 0.0f));
                        riverBedDepth[target] = (std::max)(riverBedDepth[target], desiredDepth);
                    }
                }
            }
            if ((y & 0x1f) == 0) {
                if (ctx.isCancelled()) return NodeSystem::PinValue{};
                ctx.reportNodeProgress(0.85f * static_cast<float>(y + 1) / static_cast<float>(h));
            }
        }

        auto carved = createHeightOutput(w, h);
        auto bedMask = createMaskOutput(w, h);
        const float maskDenominator = (std::max)(safeMaximumDepth, 0.001f);
        for (size_t index = 0; index < carveDepth.size(); ++index) {
            (*carved.data)[index] = (*height.data)[index] - carveDepth[index] / heightScale;
            (*bedMask.data)[index] = clampValue(riverBedDepth[index] / maskDenominator, 0.0f, 1.0f);
        }
        ctx.setCachedValue(id, 0, carved);
        ctx.setCachedValue(id, 1, bedMask);
        ctx.reportNodeProgress(1.0f);
        return outputIndex == 1 ? NodeSystem::PinValue{bedMask} : NodeSystem::PinValue{carved};
    }

    void RiverBedCarveNode::drawContent() {
        bool edited = false;
        edited |= ImGui::SliderFloat("Min Width", &minimumWidth, 0.1f, 25.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Max Width", &maximumWidth, minimumWidth, 100.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Min Depth", &minimumDepth, 0.0f, 5.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Max Depth", &maximumDepth, minimumDepth, 20.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Bank Softness", &bankSoftness, 0.05f, 1.0f, "%.2f");
        ImGui::TextDisabled("Safety cap: 4%% terrain width / 10%% relief");
        if (edited) dirty = true;
    }

    NodeSystem::PinValue RiverSplineOutputNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        (void)outputIndex;
        pendingPaths.clear();
        const auto height = getHeightInput(0, ctx);
        const auto accumulation = getHeightInput(1, ctx);
        const auto direction = getHeightInput(2, ctx);
        const auto channels = getHeightInput(3, ctx);
        const auto lakeMask = getHeightInput(4, ctx);
        const auto lakeLevel = getHeightInput(5, ctx);
        const auto riverWidth = getHeightInput(6, ctx);
        const auto waterDepth = getHeightInput(7, ctx);
        const auto flowSpeed = getHeightInput(8, ctx);
        const auto discharge = getHeightInput(9, ctx);
        const auto froude = getHeightInput(10, ctx);
        const auto foamPotential = getHeightInput(11, ctx);
        const auto riverLevel = getHeightInput(12, ctx);
        const auto optionalMatches = [&](const NodeSystem::Image2DData& image) {
            return !image.isValid() || (height.width == image.width && height.height == image.height);
        };
        if (!height.isValid() || !accumulation.isValid() || !direction.isValid() || !channels.isValid() ||
            height.width != accumulation.width || height.height != accumulation.height ||
            height.width != direction.width || height.height != direction.height ||
            height.width != channels.width || height.height != channels.height ||
            (lakeMask.isValid() && (height.width != lakeMask.width || height.height != lakeMask.height)) ||
            (lakeLevel.isValid() && (height.width != lakeLevel.width || height.height != lakeLevel.height)) ||
            !optionalMatches(riverWidth) || !optionalMatches(waterDepth) || !optionalMatches(flowSpeed) ||
            !optionalMatches(discharge) || !optionalMatches(froude) || !optionalMatches(foamPotential) ||
            !optionalMatches(riverLevel)) {
            ctx.addError(id, "River Spline Output requires matching height, accumulation, direction and channel inputs");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const int count = w * h;
        std::vector<int> parent(static_cast<size_t>(count), -1);
        std::vector<uint8_t> active(static_cast<size_t>(count), 0);
        std::vector<int> incoming(static_cast<size_t>(count), 0);
        for (int index = 0; index < count; ++index) {
            const bool insideLake = lakeMask.isValid() &&
                (*lakeMask.data)[static_cast<size_t>(index)] >= 0.5f;
            active[static_cast<size_t>(index)] =
                (*channels.data)[static_cast<size_t>(index)] > 0.0f && !insideLake ? 1u : 0u;
            const int d = decodeFlowDirection((*direction.data)[static_cast<size_t>(index)]);
            if (d >= 0) {
                const int nx = index % w + kHydrologyDx[d];
                const int ny = index / w + kHydrologyDy[d];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) parent[static_cast<size_t>(index)] = ny * w + nx;
            }
        }
        for (int index = 0; index < count; ++index) {
            const int downstream = parent[static_cast<size_t>(index)];
            if (active[static_cast<size_t>(index)] && downstream >= 0 && active[static_cast<size_t>(downstream)]) {
                ++incoming[static_cast<size_t>(downstream)];
            }
        }

        struct SourceCandidate {
            std::vector<int> fullPath;
            float lengthCells = 0.0f;
        };
        std::vector<SourceCandidate> candidates;
        for (int source = 0; source < count; ++source) {
            if ((source & 0x7fff) == 0 && ctx.isCancelled()) return NodeSystem::PinValue{};
            if (!active[static_cast<size_t>(source)] || incoming[static_cast<size_t>(source)] != 0) continue;
            SourceCandidate candidate;
            if (lakeMask.isValid()) {
                const int sx = source % w;
                const int sy = source / w;
                int bestLakePredecessor = -1;
                float bestAccumulation = -1.0f;
                for (int d = 0; d < 8; ++d) {
                    const int nx = sx + kHydrologyDx[d];
                    const int ny = sy + kHydrologyDy[d];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    const int neighbor = ny * w + nx;
                    if ((*lakeMask.data)[static_cast<size_t>(neighbor)] >= 0.5f &&
                        parent[static_cast<size_t>(neighbor)] == source &&
                        (*accumulation.data)[static_cast<size_t>(neighbor)] > bestAccumulation) {
                        bestLakePredecessor = neighbor;
                        bestAccumulation = (*accumulation.data)[static_cast<size_t>(neighbor)];
                    }
                }
                if (bestLakePredecessor >= 0) candidate.fullPath.push_back(bestLakePredecessor);
            }
            int cursor = source;
            int guard = 0;
            while (cursor >= 0 && active[static_cast<size_t>(cursor)] && guard++ < count) {
                if (!candidate.fullPath.empty()) {
                    const int previous = candidate.fullPath.back();
                    const float dx = static_cast<float>(cursor % w - previous % w);
                    const float dy = static_cast<float>(cursor / w - previous / w);
                    candidate.lengthCells += std::sqrt(dx * dx + dy * dy);
                }
                candidate.fullPath.push_back(cursor);
                cursor = parent[static_cast<size_t>(cursor)];
            }
            if (cursor >= 0 && lakeMask.isValid() &&
                (*lakeMask.data)[static_cast<size_t>(cursor)] >= 0.5f) {
                const int previous = candidate.fullPath.empty() ? -1 : candidate.fullPath.back();
                if (previous != cursor) {
                    if (previous >= 0) {
                        const float dx = static_cast<float>(cursor % w - previous % w);
                        const float dy = static_cast<float>(cursor / w - previous / w);
                        candidate.lengthCells += std::sqrt(dx * dx + dy * dy);
                    }
                    candidate.fullPath.push_back(cursor);
                }
            }
            if (candidate.fullPath.size() >= 2) candidates.push_back(std::move(candidate));
        }

        // The longest headwater-to-outlet route claims the main stem first.
        // Tributaries then stop at the first claimed downstream edge. This keeps
        // every drainage edge in exactly one water mesh instead of stacking the
        // shared downstream portion once per source.
        std::sort(candidates.begin(), candidates.end(), [](const SourceCandidate& a, const SourceCandidate& b) {
            return a.lengthCells > b.lengthCells;
        });
        std::vector<uint8_t> claimedEdge(static_cast<size_t>(count), 0);

        const auto sampleField = [w, h](const NodeSystem::Image2DData& image, float x, float y) {
            x = clampValue(x, 0.0f, static_cast<float>(w - 1));
            y = clampValue(y, 0.0f, static_cast<float>(h - 1));
            const int x0 = static_cast<int>(std::floor(x));
            const int y0 = static_cast<int>(std::floor(y));
            const int x1 = (std::min)(x0 + 1, w - 1);
            const int y1 = (std::min)(y0 + 1, h - 1);
            const float tx = x - x0, ty = y - y0;
            const float a = (*image.data)[static_cast<size_t>(y0) * w + x0] * (1.0f - tx) +
                (*image.data)[static_cast<size_t>(y0) * w + x1] * tx;
            const float b = (*image.data)[static_cast<size_t>(y1) * w + x0] * (1.0f - tx) +
                (*image.data)[static_cast<size_t>(y1) * w + x1] * tx;
            return a * (1.0f - ty) + b * ty;
        };

        for (const SourceCandidate& candidate : candidates) {
            std::vector<int> uniquePath;
            uniquePath.reserve(candidate.fullPath.size());
            for (size_t i = 0; i < candidate.fullPath.size(); ++i) {
                const int index = candidate.fullPath[i];
                uniquePath.push_back(index);
                if (i + 1 >= candidate.fullPath.size()) break;
                if (claimedEdge[static_cast<size_t>(index)]) break;
                claimedEdge[static_cast<size_t>(index)] = 1;
            }
            if (static_cast<int>(uniquePath.size()) < minimumSplinePoints) continue;

            PendingPath path;
            const int spacing = (std::max)(pointSpacing, 1);
            for (size_t i = 0; i < uniquePath.size(); i += static_cast<size_t>(spacing)) {
                const int index = uniquePath[i];
                PendingPoint point;
                point.x = static_cast<float>(index % w);
                point.y = static_cast<float>(index / w);
                point.sourceIndex = index;
                path.points.push_back(point);
            }
            const int last = uniquePath.back();
            if (path.points.empty() || std::lround(path.points.back().x) != last % w ||
                std::lround(path.points.back().y) != last / w) {
                PendingPoint point;
                point.x = static_cast<float>(last % w);
                point.y = static_cast<float>(last / w);
                point.sourceIndex = last;
                path.points.push_back(point);
            }
            if (path.points.size() < 2) continue;

            // Suppress D8 stair-steps before Bezier tangent generation. Endpoints
            // remain exact so tributaries still meet their main stem.
            for (int smoothingPass = 0; smoothingPass < 2 && path.points.size() > 2; ++smoothingPass) {
                std::vector<PendingPoint> smoothed = path.points;
                for (size_t i = 1; i + 1 < path.points.size(); ++i) {
                    smoothed[i].x = (path.points[i - 1].x + path.points[i].x * 2.0f + path.points[i + 1].x) * 0.25f;
                    smoothed[i].y = (path.points[i - 1].y + path.points[i].y * 2.0f + path.points[i + 1].y) * 0.25f;
                }
                path.points.swap(smoothed);
            }

            path.importance = 0.0f;
            path.lengthCells = 0.0f;
            const auto sampleSource = [&](const NodeSystem::Image2DData& image,
                                          const PendingPoint& point) {
                if (!image.isValid()) return 0.0f;
                if (point.sourceIndex >= 0 && point.sourceIndex < count) {
                    return (*image.data)[static_cast<size_t>(point.sourceIndex)];
                }
                return sampleField(image, point.x, point.y);
            };
            for (size_t i = 0; i < path.points.size(); ++i) {
                PendingPoint& point = path.points[i];
                point.strength = clampValue(sampleSource(accumulation, point), 0.0f, 1.0f);
                const bool onLake = lakeMask.isValid() && sampleSource(lakeMask, point) >= 0.5f;
                point.widthMeters = riverWidth.isValid() ? sampleSource(riverWidth, point) : 0.0f;
                point.depthMeters = waterDepth.isValid() ? sampleSource(waterDepth, point) : 0.0f;
                point.flowSpeed = flowSpeed.isValid() ? sampleSource(flowSpeed, point) : 0.0f;
                point.discharge = discharge.isValid() ? sampleSource(discharge, point) : 0.0f;
                point.froude = froude.isValid() ? sampleSource(froude, point) : 0.0f;
                point.foamPotential = foamPotential.isValid() ? sampleSource(foamPotential, point) : 0.0f;
                point.surfaceHeight = riverLevel.isValid()
                    ? sampleSource(riverLevel, point)
                    : (onLake && lakeLevel.isValid()
                        ? sampleSource(lakeLevel, point)
                        : sampleSource(height, point));
                path.importance = (std::max)(path.importance, point.strength);
                if (i > 0) {
                    const float dx = point.x - path.points[i - 1].x;
                    const float dy = point.y - path.points[i - 1].y;
                    path.lengthCells += std::sqrt(dx * dx + dy * dy);
                }
            }
            // Paths are ordered source -> outlet. Bilinear sampling around D8
            // corners can reintroduce tiny uphill steps, so keep the authored
            // control profile non-increasing before Bezier interpolation.
            for (size_t i = 1; i < path.points.size(); ++i) {
                path.points[i].surfaceHeight = (std::min)(
                    path.points[i].surfaceHeight, path.points[i - 1].surfaceHeight);
            }
            if (path.lengthCells >= 1.5f) {
                path.importance *= std::sqrt(path.lengthCells);
                pendingPaths.push_back(std::move(path));
            }
        }

        std::sort(pendingPaths.begin(), pendingPaths.end(), [](const PendingPath& a, const PendingPath& b) {
            return a.importance > b.importance;
        });
        if (pendingPaths.size() > static_cast<size_t>(maximumRivers)) {
            pendingPaths.resize(static_cast<size_t>(maximumRivers));
        }
        return NodeSystem::PinValue{};
    }

    bool RiverSplineOutputNode::applyGeneratedRivers(SceneData& scene, TerrainObject* terrain) {
        if (!terrain) return false;
        auto& manager = RiverManager::getInstance();
        bool changed = false;
        const std::string ownershipPrefix =
            "AutoRiver_T" + std::to_string(terrain->id) + "_N" + std::to_string(id) + "_";
        std::vector<int> ownedIds = generatedRiverIds;
        for (const RiverSpline& river : manager.getRivers()) {
            if (river.name.rfind(ownershipPrefix, 0) == 0 &&
                std::find(ownedIds.begin(), ownedIds.end(), river.id) == ownedIds.end()) {
                ownedIds.push_back(river.id);
            }
        }
        for (int riverId : ownedIds) {
            if (manager.getRiver(riverId)) {
                manager.removeRiver(scene, riverId);
                changed = true;
            }
        }
        generatedRiverIds.clear();

        const int w = terrain->heightmap.width;
        const int h = terrain->heightmap.height;
        if (w < 2 || h < 2) return changed;
        const float size = terrain->heightmap.scale_xz;
        const float heightScale = terrain->heightmap.scale_y;
        const Matrix4x4 terrainTransform = terrain->transform
            ? terrain->transform->getFinal() : Matrix4x4::identity();

        int branchIndex = 0;
        for (const PendingPath& path : pendingPaths) {
            std::vector<Vec3> worldPoints;
            worldPoints.reserve(path.points.size());
            for (const PendingPoint& point : path.points) {
                const float legacySurfaceOffset = point.depthMeters > 0.0f ? 0.0f : 0.03f;
                const Vec3 local(
                    point.x / static_cast<float>(w - 1) * size,
                    point.surfaceHeight * heightScale + legacySurfaceOffset,
                    point.y / static_cast<float>(h - 1) * size);
                worldPoints.push_back(terrainTransform.multiplyVector(Vec4(local, 1.0f)).xyz());
            }
            float physicalLength = 0.0f;
            for (size_t i = 1; i < worldPoints.size(); ++i) {
                const float dx = worldPoints[i].x - worldPoints[i - 1].x;
                const float dz = worldPoints[i].z - worldPoints[i - 1].z;
                physicalLength += std::sqrt(dx * dx + dz * dz);
            }
            const float cellSize = size / static_cast<float>((std::max)((std::max)(w, h) - 1, 1));
            const bool hasHydraulicSection = std::any_of(path.points.begin(), path.points.end(),
                [](const PendingPoint& point) { return point.widthMeters > 0.0f; });
            const float geometricWidthCap = (std::min)(size * 0.035f, physicalLength * 0.18f);
            const float widthCap = hasHydraulicSection
                ? geometricWidthCap : (std::min)(maximumWidth, geometricWidthCap);
            if (worldPoints.size() < 2 || physicalLength < (std::max)(cellSize * 3.0f, widthCap * 2.5f) ||
                widthCap < cellSize * 0.35f) {
                continue;
            }
            RiverSpline* river = manager.createRiver(
                ownershipPrefix + "B" + std::to_string(branchIndex++));
            if (!river) continue;
            const int riverId = river->id;
            river->followTerrain = false;
            river->lengthSubdivisions = clampValue(static_cast<int>(path.points.size()) * 4, 24, 192);
            for (size_t pointIndex = 0; pointIndex < path.points.size(); ++pointIndex) {
                const PendingPoint& point = path.points[pointIndex];
                const float safeMinimumWidth = (std::min)(minimumWidth, widthCap);
                const float fallbackWidth = safeMinimumWidth +
                    (widthCap - safeMinimumWidth) * std::sqrt(point.strength);
                const float width = point.widthMeters > 0.0f
                    ? clampValue(point.widthMeters, 0.05f, widthCap) : fallbackWidth;
                const float depth = point.depthMeters > 0.0f
                    ? point.depthMeters : (std::max)(width * depthScale, 0.03f);
                river->addControlPoint(worldPoints[pointIndex], width, depth);
                RiverSpline::HydraulicPoint hydraulic;
                hydraulic.discharge = point.discharge;
                hydraulic.flowSpeed = point.flowSpeed;
                hydraulic.froude = point.froude;
                hydraulic.foamPotential = point.foamPotential;
                hydraulic.surfaceElevation = worldPoints[pointIndex].y;
                river->setHydraulicPoint(static_cast<int>(pointIndex), hydraulic);
            }
            // Preserve the smooth XZ route but remove vertical Catmull-Rom
            // overshoot. Zero-Y handles produce a monotone smoothstep between
            // already monotone water-level control points.
            for (BezierControlPoint& controlPoint : river->spline.points) {
                controlPoint.tangentIn.y = 0.0f;
                controlPoint.tangentOut.y = 0.0f;
                controlPoint.autoTangent = false;
            }
            generatedRiverIds.push_back(riverId);
            if (generateWaterMeshes) manager.generateMesh(manager.getRiver(riverId), scene);
            changed = true;
        }
        return changed;
    }

    void RiverSplineOutputNode::drawContent() {
        bool edited = false;
        edited |= ImGui::SliderFloat("Min Width", &minimumWidth, 0.1f, 25.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Max Width", &maximumWidth, minimumWidth, 100.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Depth Scale", &depthScale, 0.01f, 1.0f, "%.2f");
        edited |= ImGui::SliderInt("Min Points", &minimumSplinePoints, 2, 128);
        edited |= ImGui::SliderInt("Point Spacing", &pointSpacing, 1, 32);
        edited |= ImGui::SliderInt("Max Rivers", &maximumRivers, 1, 128);
        edited |= ImGui::Checkbox("Generate Water Mesh", &generateWaterMeshes);
        ImGui::TextDisabled("Owned rivers: %d", static_cast<int>(generatedRiverIds.size()));
        ImGui::TextDisabled("Width also capped by terrain and path length");
        if (edited) dirty = true;
    }

    NodeSystem::PinValue TerrainFieldsOutputNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        (void)outputIndex;
        auto* tctx = getTerrainContext(ctx);
        if (!tctx || !tctx->terrain) {
            ctx.addError(id, "Terrain Fields Output requires terrain context");
            return NodeSystem::PinValue{};
        }
        // Keep new fields appended: serialized graphs map pins by index and the
        // original terrain/biome pin order must remain stable.
        static const std::array<const char*, 30> names = {
            "terrain.slope", "terrain.concavity", "terrain.convexity", "terrain.valley", "terrain.wetness",
            "biome.forest", "biome.grass", "biome.rock", "biome.alpine",
            "hydrology.accumulation", "hydrology.direction", "hydrology.basins",
            "hydrology.channels", "hydrology.stream_order", "hydrology.sources", "hydrology.river_bed",
            "hydrology.lake_mask", "hydrology.lake_depth", "hydrology.lake_level",
            "hydrology.lake_shoreline", "hydrology.lake_spill", "hydrology.lake_id",
            "hydrology.catchment_area", "hydrology.river_discharge", "hydrology.river_width",
            "hydrology.river_depth", "hydrology.river_speed", "hydrology.river_level",
            "hydrology.river_froude", "hydrology.river_foam"
        };
        const size_t expected = static_cast<size_t>(tctx->terrain->heightmap.width) *
            tctx->terrain->heightmap.height;
        for (int inputIndex = 0; inputIndex < static_cast<int>(names.size()); ++inputIndex) {
            const auto field = getHeightInput(inputIndex, ctx);
            if (field.isValid() && field.data && field.data->size() == expected) {
                tctx->terrain->analysisFields[names[static_cast<size_t>(inputIndex)]] = field.data;
            }
        }
        return NodeSystem::PinValue{};
    }

    NodeSystem::PinValue BiomeComposerNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto slopeInput = getHeightInput(1, ctx);
        const auto valleyInput = getHeightInput(2, ctx);
        const auto wetnessInput = getHeightInput(3, ctx);
        const auto exposureInput = getHeightInput(4, ctx);
        if (!height.isValid() || height.width < 2 || height.height < 2) {
            ctx.addError(id, "Biome Composer requires a valid height input");
            return NodeSystem::PinValue{};
        }
        const auto matches = [&](const NodeSystem::Image2DData& image) {
            return !image.isValid() || (image.channels == 1 && image.width == height.width && image.height == height.height);
        };
        if (!matches(slopeInput) || !matches(valleyInput) || !matches(wetnessInput) || !matches(exposureInput)) {
            ctx.addError(id, "Biome Composer inputs must have matching resolutions");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const size_t pixelCount = static_cast<size_t>(w) * h;
        std::array<NodeSystem::Image2DData, 5> result = {
            createMaskOutput(w, h), createMaskOutput(w, h), createMaskOutput(w, h), createMaskOutput(w, h),
            NodeSystem::Image2DData{}
        };
        result[4].data = std::make_shared<std::vector<float>>(pixelCount * 4u, 0.0f);
        result[4].width = w;
        result[4].height = h;
        result[4].channels = 4;
        result[4].semantic = NodeSystem::ImageSemantic::Mask;

        float minimumHeight = (std::numeric_limits<float>::max)();
        float maximumHeight = (std::numeric_limits<float>::lowest)();
        for (float value : *height.data) {
            minimumHeight = (std::min)(minimumHeight, value);
            maximumHeight = (std::max)(maximumHeight, value);
        }
        const float heightRange = (std::max)(maximumHeight - minimumHeight, 1e-6f);
        auto* tctx = getTerrainContext(ctx);
        const float heightScale = tctx ? (std::max)(tctx->scale_y, 0.1f) : 1.0f;
        const float cellSize = tctx
            ? (std::max)(tctx->scale_xz / (std::max)(w - 1, 1), 1e-5f)
            : 1.0f;
        const auto smoothRange = [](float edge, float width, float value) {
            const float lo = edge - width;
            const float hi = edge + width;
            const float t = clampValue((value - lo) / (std::max)(hi - lo, 1e-6f), 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        };

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t index = static_cast<size_t>(y) * w + x;
                const float elevation = clampValue(((*height.data)[index] - minimumHeight) / heightRange, 0.0f, 1.0f);

                float slope = 0.0f;
                if (slopeInput.isValid()) {
                    slope = clampValue((*slopeInput.data)[index], 0.0f, 1.0f);
                } else {
                    const int xl = (std::max)(x - 1, 0), xr = (std::min)(x + 1, w - 1);
                    const int yu = (std::max)(y - 1, 0), yd = (std::min)(y + 1, h - 1);
                    const float dx = ((*height.data)[static_cast<size_t>(y) * w + xr] -
                        (*height.data)[static_cast<size_t>(y) * w + xl]) * heightScale /
                        (std::max)((xr - xl) * cellSize, 1e-5f);
                    const float dz = ((*height.data)[static_cast<size_t>(yd) * w + x] -
                        (*height.data)[static_cast<size_t>(yu) * w + x]) * heightScale /
                        (std::max)((yd - yu) * cellSize, 1e-5f);
                    slope = std::atan(std::sqrt(dx * dx + dz * dz)) / 1.57079632679f;
                }
                const float valley = valleyInput.isValid()
                    ? clampValue((*valleyInput.data)[index], 0.0f, 1.0f) : 0.0f;
                const float wetness = wetnessInput.isValid()
                    ? clampValue((*wetnessInput.data)[index], 0.0f, 1.0f)
                    : clampValue(0.20f + valley * 0.60f + (1.0f - slope) * 0.20f, 0.0f, 1.0f);
                const float exposure = exposureInput.isValid()
                    ? clampValue((*exposureInput.data)[index], 0.0f, 1.0f) : 0.5f;

                const float rockMask = smoothRange(rockSlope, transition, slope);
                const float alpineMask = smoothRange(alpineLine, transition, elevation);
                const float belowForestCeiling = 1.0f - smoothRange(forestCeiling, transition, elevation);
                const float moistureSuitability = smoothRange(forestMoisture, transition, wetness);
                const float sunRetention = 1.0f - exposureDrying * exposure;

                float forest = belowForestCeiling * moistureSuitability * sunRetention *
                    (0.70f + valley * 0.55f) * (1.0f - rockMask) * (1.0f - alpineMask) * 1.45f;
                float grass = (0.45f + (1.0f - moistureSuitability) * 0.60f) *
                    (0.75f + (1.0f - valley) * 0.25f) * (1.0f - rockMask) * (1.0f - alpineMask);
                float rock = (0.015f + rockMask * 2.8f) * (1.0f - alpineMask * 0.25f);
                float alpine = alpineMask * (1.15f + elevation) * (1.0f - rockMask * 0.35f);

                const float total = (std::max)(forest + grass + rock + alpine, 1e-6f);
                forest /= total; grass /= total; rock /= total; alpine /= total;
                (*result[0].data)[index] = forest;
                (*result[1].data)[index] = grass;
                (*result[2].data)[index] = rock;
                (*result[3].data)[index] = alpine;
                const size_t packed = index * 4u;
                (*result[4].data)[packed + 0u] = forest;
                (*result[4].data)[packed + 1u] = grass;
                (*result[4].data)[packed + 2u] = rock;
                (*result[4].data)[packed + 3u] = alpine;
            }
        }

        for (int i = 0; i < 5; ++i) ctx.setCachedValue(id, i, result[static_cast<size_t>(i)]);
        return (outputIndex >= 0 && outputIndex < 5)
            ? NodeSystem::PinValue{result[static_cast<size_t>(outputIndex)]}
            : NodeSystem::PinValue{};
    }

    const char* BiomeComposerNode::getPresetName(BiomeClimatePreset value) {
        switch (value) {
            case BiomeClimatePreset::Custom: return "Custom";
            case BiomeClimatePreset::TemperateMixed: return "Temperate Mixed";
            case BiomeClimatePreset::LushValleys: return "Lush Valleys";
            case BiomeClimatePreset::AlpineTundra: return "Alpine Tundra";
            case BiomeClimatePreset::AridHighlands: return "Arid Highlands";
            case BiomeClimatePreset::BorealMountains: return "Boreal Mountains";
            default: return "Custom";
        }
    }

    void BiomeComposerNode::applyPreset(BiomeClimatePreset value) {
        preset = value;
        switch (value) {
            case BiomeClimatePreset::TemperateMixed:
                forestCeiling = 0.72f; alpineLine = 0.68f; forestMoisture = 0.32f;
                rockSlope = 0.48f; transition = 0.10f; exposureDrying = 0.30f;
                break;
            case BiomeClimatePreset::LushValleys:
                forestCeiling = 0.84f; alpineLine = 0.86f; forestMoisture = 0.20f;
                rockSlope = 0.58f; transition = 0.14f; exposureDrying = 0.10f;
                break;
            case BiomeClimatePreset::AlpineTundra:
                forestCeiling = 0.52f; alpineLine = 0.58f; forestMoisture = 0.38f;
                rockSlope = 0.42f; transition = 0.08f; exposureDrying = 0.42f;
                break;
            case BiomeClimatePreset::AridHighlands:
                forestCeiling = 0.48f; alpineLine = 0.88f; forestMoisture = 0.72f;
                rockSlope = 0.34f; transition = 0.11f; exposureDrying = 0.88f;
                break;
            case BiomeClimatePreset::BorealMountains:
                forestCeiling = 0.67f; alpineLine = 0.64f; forestMoisture = 0.27f;
                rockSlope = 0.45f; transition = 0.09f; exposureDrying = 0.22f;
                break;
            case BiomeClimatePreset::Custom:
            default:
                break;
        }
        dirty = true;
    }

    void BiomeComposerNode::drawContent() {
        bool edited = false;
        if (ImGui::BeginCombo("Preset", getPresetName(preset))) {
            for (int i = 0; i <= static_cast<int>(BiomeClimatePreset::BorealMountains); ++i) {
                const auto value = static_cast<BiomeClimatePreset>(i);
                const bool selected = value == preset;
                if (ImGui::Selectable(getPresetName(value), selected)) {
                    applyPreset(value);
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        edited |= ImGui::SliderFloat("Forest Ceiling", &forestCeiling, 0.0f, 1.0f, "%.2f");
        edited |= ImGui::SliderFloat("Alpine Line", &alpineLine, 0.0f, 1.0f, "%.2f");
        edited |= ImGui::SliderFloat("Forest Moisture", &forestMoisture, 0.0f, 1.0f, "%.2f");
        edited |= ImGui::SliderFloat("Rock Slope", &rockSlope, 0.0f, 1.0f, "%.2f");
        edited |= ImGui::SliderFloat("Transition", &transition, 0.01f, 0.35f, "%.2f");
        edited |= ImGui::SliderFloat("Exposure Drying", &exposureDrying, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled("RGBA: Forest / Grass / Rock / Alpine");
        ImGui::TextDisabled("Outputs always sum to 1.0");
        if (edited) {
            preset = BiomeClimatePreset::Custom;
            dirty = true;
        }
    }

    namespace {
        const std::array<const char*, 14> kFoliageFieldNames = {
            "", "biome.forest", "biome.grass", "biome.rock", "biome.alpine",
            "terrain.slope", "terrain.concavity", "terrain.convexity", "terrain.valley",
            "terrain.wetness", "hydrology.channels", "hydrology.lake_mask",
            "hydrology.lake_shoreline", "hydrology.river_foam"
        };

        bool drawFoliageFieldPicker(const char* label, std::string& value) {
            const char* preview = value.empty() ? "<none>" : value.c_str();
            bool changed = false;
            if (ImGui::BeginCombo(label, preview)) {
                for (const char* field : kFoliageFieldNames) {
                    const bool selected = value == field;
                    const char* display = field[0] == '\0' ? "<none>" : field;
                    if (ImGui::Selectable(display, selected)) {
                        value = field;
                        changed = true;
                    }
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            return changed;
        }

        float defaultFoliageAssetHeight(const std::string& densityField) {
            return FoliageAssets::defaultTargetHeight(densityField);
        }

        void applyFoliageAssetRefSettings(const FoliageLayerNode::AssetRef& ref,
                                          ScatterSource& source) {
            FoliageAssets::configurePlacement(source, ref.targetHeight, ref.heightVariation,
                                              ref.alignToNormal, ref.normalInfluence);
            source.weight = (std::max)(0.0f, ref.weight);
        }

        void captureFoliageLibrarySources(FoliageLayerNode& layer, const InstanceGroup& group) {
            std::vector<FoliageLayerNode::AssetRef> captured;
            for (const auto& source : group.sources) {
                if (source.asset_relative_path.empty()) continue;
                FoliageLayerNode::AssetRef ref;
                ref.id = source.asset_id;
                ref.name = source.name;
                ref.relativeEntryPath = source.asset_relative_path;
                ref.weight = source.weight;
                ref.alignToNormal = source.settings.align_to_normal;
                ref.normalInfluence = source.settings.normal_influence;
                if (source.has_local_bbox) {
                    const float sourceHeight = source.local_bbox.max.y - source.local_bbox.min.y;
                    const float scaleMin = source.settings.scale_min;
                    const float scaleMax = source.settings.scale_max;
                    const float centerScale = (scaleMin + scaleMax) * 0.5f;
                    if (sourceHeight > 1e-5f && centerScale > 0.0f) {
                        ref.targetHeight = sourceHeight * centerScale;
                        ref.heightVariation = clampValue(
                            (scaleMax - scaleMin) / (scaleMax + scaleMin), 0.0f, 0.95f);
                    }
                }
                captured.push_back(std::move(ref));
            }
            layer.assetSources = std::move(captured);
        }

        bool synchronizeFoliageLibrarySources(
            const std::vector<FoliageLayerNode::AssetRef>& assetSources,
            InstanceGroup& group) {
            std::vector<ScatterSource> synchronized;
            synchronized.reserve(group.sources.size() + assetSources.size());
            // Scene-object sources are owned by the same group but are outside
            // the Asset Library picker. Preserve them byte-for-byte.
            for (auto& source : group.sources) {
                if (source.asset_relative_path.empty()) synchronized.push_back(std::move(source));
            }

            bool allLoaded = true;
            for (const auto& ref : assetSources) {
                if (ref.relativeEntryPath.empty()) continue;
                auto existing = std::find_if(group.sources.begin(), group.sources.end(),
                    [&](const ScatterSource& source) {
                        return source.asset_relative_path == ref.relativeEntryPath;
                    });
                ScatterSource source;
                if (existing != group.sources.end()) {
                    source = std::move(*existing);
                } else {
                    std::string error;
                    if (!FoliageAssets::loadScatterSource(ref.relativeEntryPath, ref.name,
                                                          ref.weight, source, &error)) {
                        allLoaded = false;
                        continue;
                    }
                }
                source.asset_id = ref.id.empty() ? source.asset_id : ref.id;
                source.name = ref.name.empty() ? source.name : ref.name;
                applyFoliageAssetRefSettings(ref, source);
                synchronized.push_back(std::move(source));
            }

            group.sources = std::move(synchronized);
            group.brush_settings.use_global_settings = false;
            group.source_bvh.reset();
            group.source_triangles_ptr.reset();
            group.blas_id = -1;
            group.gpu_dirty = true;
            return allLoaded;
        }

        void captureFoliageGroupSettings(FoliageLayerNode& layer, const InstanceGroup& group) {
            const auto& settings = group.brush_settings;
            layer.instanceGroupId = group.id;
            layer.instanceGroupName = group.name;
            layer.densityMultiplier = settings.density;
            layer.targetCount = settings.target_count;
            layer.seed = settings.seed;
            layer.minimumDistance = settings.min_distance;
            layer.maximumSlopeDegrees = settings.slope_max;
            layer.minimumHeight = settings.height_min;
            layer.maximumHeight = settings.height_max;
            layer.densityField = settings.density_mask_attribute;
            layer.exclusionField = settings.exclusion_mask_attribute;
            layer.exclusionThreshold = settings.exclusion_threshold;
            layer.scaleField = settings.scale_mask_attribute;
            layer.scaleFieldInfluence = settings.scale_mask_influence;
            layer.settingsCaptured = true;
        }

        void applyFoliageGroupSettings(const FoliageLayerNode& layer, InstanceGroup& group) {
            auto& settings = group.brush_settings;
            settings.density = clampValue(layer.densityMultiplier, 0.0f, 10000.0f);
            settings.target_count = clampValue(layer.targetCount, 0, 10000000);
            settings.seed = layer.seed;
            settings.min_distance = clampValue(layer.minimumDistance, 0.0f, 1000.0f);
            settings.slope_max = clampValue(layer.maximumSlopeDegrees, 0.0f, 90.0f);
            settings.height_min = (std::min)(layer.minimumHeight, layer.maximumHeight);
            settings.height_max = (std::max)(layer.minimumHeight, layer.maximumHeight);
            settings.density_mask_attribute = layer.densityField;
            settings.exclusion_mask_attribute = layer.exclusionField;
            settings.exclusion_threshold = clampValue(layer.exclusionThreshold, 0.0f, 1.0f);
            settings.scale_mask_attribute = layer.scaleField;
            settings.scale_mask_influence = clampValue(layer.scaleFieldInfluence, 0.0f, 1.0f);
            group.gpu_dirty = true;
        }
    }

    NodeSystem::PinValue FoliageLayerNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        (void)outputIndex;
        const InstanceGroup* liveGroup = instanceGroupId >= 0
            ? InstanceManager::getInstance().getGroup(instanceGroupId) : nullptr;
        if (!liveGroup && !instanceGroupName.empty()) {
            liveGroup = InstanceManager::getInstance().findGroupByName(instanceGroupName);
        }
        if (liveGroup && !liveGroup->transient) {
            captureFoliageGroupSettings(*this, *liveGroup);
            if (useAssetLibrary) captureFoliageLibrarySources(*this, *liveGroup);
        }
        std::string resolvedBiome = assetBiome;
        if (resolvedBiome.empty() || resolvedBiome == "Auto") {
            if (auto* graph = ctx.getGraph()) {
                for (const auto& graphNode : graph->nodes) {
                    if (const auto* composer = dynamic_cast<const BiomeComposerNode*>(graphNode.get())) {
                        resolvedBiome = BiomeComposerNode::getPresetName(composer->preset);
                        break;
                    }
                }
            }
        }
        nlohmann::json assets = nlohmann::json::array();
        for (const auto& asset : assetSources) {
            if (asset.relativeEntryPath.empty()) continue;
            assets.push_back({
                {"id", asset.id}, {"name", asset.name},
                {"relativeEntryPath", asset.relativeEntryPath},
                {"weight", (std::max)(0.0f, asset.weight)},
                {"targetHeight", (std::max)(0.0f, asset.targetHeight)},
                {"heightVariation", clampValue(asset.heightVariation, 0.0f, 0.95f)},
                {"alignToNormal", asset.alignToNormal},
                {"normalInfluence", clampValue(asset.normalInfluence, 0.0f, 1.0f)}
            });
        }
        const std::string resolvedGroupName = instanceGroupName.empty()
            ? (name + " Assets") : instanceGroupName;
        nlohmann::json recipe = {
            {"kind", "foliage_layer"},
            {"version", 2},
            {"group", resolvedGroupName},
            {"groupId", instanceGroupId},
            {"useAssetLibrary", useAssetLibrary},
            {"assetBiome", resolvedBiome},
            {"assets", std::move(assets)},
            {"settingsCaptured", settingsCaptured},
            {"enabled", layerEnabled},
            {"density", densityMultiplier},
            {"targetCount", targetCount},
            {"seed", seed},
            {"minimumDistance", minimumDistance},
            {"maximumSlopeDegrees", maximumSlopeDegrees},
            {"minimumHeight", minimumHeight},
            {"maximumHeight", maximumHeight},
            {"densityField", densityField},
            {"exclusionField", exclusionField},
            {"exclusionThreshold", exclusionThreshold},
            {"scaleField", scaleField},
            {"scaleFieldInfluence", scaleFieldInfluence}
        };
        return recipe.dump();
    }

    void FoliageLayerNode::drawContent() {
        bool edited = false;
        const char* sourceModes[] = {"Scene Layer", "Asset Library"};
        int sourceMode = useAssetLibrary ? 1 : 0;
        if (ImGui::Combo("Source", &sourceMode, sourceModes, 2)) {
            useAssetLibrary = sourceMode == 1;
            edited = true;
        }
        const InstanceGroup* boundGroup = instanceGroupId >= 0
            ? InstanceManager::getInstance().getGroup(instanceGroupId) : nullptr;
        if (!boundGroup && !instanceGroupName.empty()) {
            boundGroup = InstanceManager::getInstance().findGroupByName(instanceGroupName);
        }
        if (useAssetLibrary && boundGroup && !boundGroup->transient) {
            captureFoliageLibrarySources(*this, *boundGroup);
        }
        if (boundGroup) {
            if (!settingsCaptured) {
                const std::string authoredDensityField = densityField;
                const std::string authoredExclusionField = exclusionField;
                const std::string authoredScaleField = scaleField;
                captureFoliageGroupSettings(*this, *boundGroup);
                if (!authoredDensityField.empty()) densityField = authoredDensityField;
                if (!authoredExclusionField.empty()) exclusionField = authoredExclusionField;
                if (!authoredScaleField.empty()) scaleField = authoredScaleField;
                edited = true;
            } else {
                // InstanceGroup is the source of truth. Terrain UI edits become
                // visible in the node on the very next frame.
                captureFoliageGroupSettings(*this, *boundGroup);
            }
        }
        if (!useAssetLibrary) {
            const char* groupPreview = boundGroup ? boundGroup->name.c_str()
                : (instanceGroupName.empty() ? "<select layer>" : instanceGroupName.c_str());
            if (ImGui::BeginCombo("Foliage Layer", groupPreview)) {
                for (const auto& group : InstanceManager::getInstance().getGroups()) {
                    if (group.transient) continue;
                    const bool selected = instanceGroupId == group.id ||
                        (instanceGroupId < 0 && instanceGroupName == group.name);
                    if (ImGui::Selectable(group.name.c_str(), selected)) {
                        captureFoliageGroupSettings(*this, group);
                        edited = true;
                    }
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        } else {
            static const char* biomeNames[] = {
                "Auto", "Temperate Mixed", "Lush Valleys", "Alpine Tundra",
                "Arid Highlands", "Boreal Mountains"
            };
            int biomeIndex = 0;
            for (int i = 0; i < 6; ++i) if (assetBiome == biomeNames[i]) biomeIndex = i;
            if (ImGui::Combo("Biome", &biomeIndex, biomeNames, 6)) {
                assetBiome = biomeNames[biomeIndex];
                edited = true;
            }

            char searchBuffer[96] = {};
            std::strncpy(searchBuffer, assetSearch.c_str(), sizeof(searchBuffer) - 1);
            if (ImGui::InputTextWithHint("##AssetSearch", "Search recommended assets", searchBuffer,
                                         sizeof(searchBuffer))) {
                assetSearch = searchBuffer;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Refresh")) FoliageAssets::catalog(true);

            const auto candidates = FoliageAssets::recommendedAssets(densityField, assetBiome, assetSearch);
            const std::string addPreview = candidates.empty()
                ? "No matching assets" : "Add recommended asset...";
            if (ImGui::BeginCombo("Assets", addPreview.c_str())) {
                const size_t visibleCount = (std::min)(candidates.size(), size_t{200});
                for (size_t i = 0; i < visibleCount; ++i) {
                    const AssetRecord* asset = candidates[i];
                    if (!asset) continue;
                    const std::string path = asset->relative_entry_path.generic_string();
                    const bool selected = std::any_of(assetSources.begin(), assetSources.end(),
                        [&](const AssetRef& ref) { return ref.relativeEntryPath == path; });
                    const std::string label = asset->name + "  [" + asset->subcategory + "]";
                    if (ImGui::Selectable(label.c_str(), selected) && !selected) {
                        AssetRef ref;
                        ref.id = asset->id;
                        ref.name = asset->name;
                        ref.relativeEntryPath = path;
                        ref.targetHeight = defaultFoliageAssetHeight(densityField);
                        assetSources.push_back(std::move(ref));
                        if (instanceGroupName.empty()) instanceGroupName = name + " Assets";
                        settingsCaptured = true;
                        edited = true;
                    }
                }
                ImGui::EndCombo();
            }

            int removeAsset = -1;
            const std::string sourcesHeader = "Added Sources (" +
                std::to_string(assetSources.size()) + ")";
            const bool showSources = !assetSources.empty() &&
                ImGui::CollapsingHeader(sourcesHeader.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
            if (showSources) for (int i = 0; i < static_cast<int>(assetSources.size()); ++i) {
                auto& asset = assetSources[static_cast<size_t>(i)];
                ImGui::PushID(i);
                ImGui::BeginGroup();
                int previewWidth = 0;
                int previewHeight = 0;
                const ImTextureID preview = propertyThumbnailProvider
                    ? propertyThumbnailProvider(asset.relativeEntryPath, previewWidth, previewHeight)
                    : ImTextureID{};
                const ImVec2 thumbnailSize(44.0f, 44.0f);
                if (preview) ImGui::Image(preview, thumbnailSize);
                else ImGui::Button("No\nPreview", thumbnailSize);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(asset.name.c_str());
                    if (preview && previewWidth > 0 && previewHeight > 0) {
                        const float maxPreview = 240.0f;
                        const float previewScale = (std::min)(maxPreview / previewWidth,
                                                               maxPreview / previewHeight);
                        ImGui::Image(preview, ImVec2(previewWidth * previewScale,
                                                    previewHeight * previewScale));
                    } else {
                        ImGui::TextDisabled("No preview image");
                    }
                    ImGui::EndTooltip();
                }
                ImGui::SameLine();
                ImGui::BeginGroup();
                ImGui::TextUnformatted(asset.name.c_str());
                ImGui::TextDisabled("W %.2f  |  H %.2f m  |  %s",
                    asset.weight, asset.targetHeight,
                    asset.alignToNormal ? "Follow Slope" : "World Y-Up");
                if (ImGui::SmallButton("Edit")) ImGui::OpenPopup("AssetSourceEdit");
                ImGui::SameLine();
                if (ImGui::SmallButton("Remove")) removeAsset = i;
                ImGui::EndGroup();
                ImGui::EndGroup();

                if (ImGui::BeginPopup("AssetSourceEdit")) {
                    ImGui::TextUnformatted(asset.name.c_str());
                    ImGui::Separator();
                    ImGui::SetNextItemWidth(95.0f);
                    edited |= ImGui::DragFloat("Weight", &asset.weight, 0.05f, 0.0f, 100.0f, "%.2f");
                    ImGui::SetNextItemWidth(115.0f);
                    edited |= ImGui::DragFloat("Height", &asset.targetHeight,
                                               0.1f, 0.01f, 10000.0f, "%.2f m");
                    int variationPercent = static_cast<int>(std::round(asset.heightVariation * 100.0f));
                    ImGui::SetNextItemWidth(105.0f);
                    if (ImGui::SliderInt("Variation", &variationPercent, 0, 95, "%d%%")) {
                        asset.heightVariation = static_cast<float>(variationPercent) * 0.01f;
                        edited = true;
                    }
                    if (ImGui::Checkbox("Follow Slope", &asset.alignToNormal)) {
                        if (asset.alignToNormal && asset.normalInfluence <= 0.0f) {
                            asset.normalInfluence = 1.0f;
                        }
                        edited = true;
                    }
                    if (asset.alignToNormal) {
                        ImGui::SetNextItemWidth(110.0f);
                        edited |= ImGui::SliderFloat("Influence", &asset.normalInfluence,
                                                    0.0f, 1.0f, "%.2f");
                    } else {
                        asset.normalInfluence = 0.0f;
                        ImGui::TextDisabled("Orientation: World Y-Up");
                    }
                    ImGui::EndPopup();
                }
                ImGui::Separator();
                ImGui::PopID();
            }
            if (removeAsset >= 0) {
                assetSources.erase(assetSources.begin() + removeAsset);
                edited = true;
            }
            if (assetSources.empty()) ImGui::TextDisabled("Choose one or more library assets");
        }
        edited |= ImGui::Checkbox("Enabled", &layerEnabled);
        edited |= ImGui::DragFloat("Density", &densityMultiplier, 0.05f, 0.0f, 100.0f, "%.2f");
        edited |= ImGui::DragInt("Target Count", &targetCount, 10.0f, 0, 10000000);
        edited |= ImGui::InputInt("Seed", &seed);
        edited |= ImGui::DragFloat("Min Distance", &minimumDistance, 0.05f, 0.0f, 1000.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Max Slope", &maximumSlopeDegrees, 0.0f, 90.0f, "%.1f deg");
        edited |= ImGui::DragFloatRange2("Height", &minimumHeight, &maximumHeight,
                                        0.25f, -100000.0f, 100000.0f, "Min %.2f", "Max %.2f");
        ImGui::SeparatorText("Named Fields");
        edited |= drawFoliageFieldPicker("Density Field", densityField);
        edited |= drawFoliageFieldPicker("Exclusion Field", exclusionField);
        if (!exclusionField.empty()) {
            edited |= ImGui::SliderFloat("Reject At", &exclusionThreshold, 0.0f, 1.0f, "%.2f");
        }
        edited |= drawFoliageFieldPicker("Scale Field", scaleField);
        if (!scaleField.empty()) {
            edited |= ImGui::SliderFloat("Scale Influence", &scaleFieldInfluence, 0.0f, 1.0f, "%.2f");
        }
        ImGui::TextDisabled(useAssetLibrary
            ? "Assets load lazily; no scene object is created"
            : "Sources and transforms stay in InstanceGroup");
        if (edited) {
            if (useAssetLibrary && !assetSources.empty()) {
                auto& manager = InstanceManager::getInstance();
                InstanceGroup* mutableGroup = instanceGroupId >= 0
                    ? manager.getGroup(instanceGroupId) : nullptr;
                if (!mutableGroup && !instanceGroupName.empty()) {
                    mutableGroup = manager.findGroupByName(instanceGroupName);
                }
                if (!mutableGroup) {
                    if (instanceGroupName.empty()) instanceGroupName = name + " Assets";
                    instanceGroupId = manager.createGroup(instanceGroupName, std::string{}, {});
                    mutableGroup = manager.getGroup(instanceGroupId);
                }
                if (mutableGroup && !mutableGroup->transient) {
                    synchronizeFoliageLibrarySources(assetSources, *mutableGroup);
                }
            } else if (useAssetLibrary && assetSources.empty() && boundGroup) {
                // Removing the final library item from the node must remove it
                // from the shared group as well, while retaining scene sources.
                auto* mutableGroup = InstanceManager::getInstance().getGroup(boundGroup->id);
                if (mutableGroup) synchronizeFoliageLibrarySources(assetSources, *mutableGroup);
            }
            auto& manager = InstanceManager::getInstance();
            InstanceGroup* mutableGroup = instanceGroupId >= 0
                ? manager.getGroup(instanceGroupId) : nullptr;
            if (!mutableGroup && !instanceGroupName.empty()) {
                mutableGroup = manager.findGroupByName(instanceGroupName);
            }
            if (mutableGroup && !mutableGroup->transient) {
                applyFoliageGroupSettings(*this, *mutableGroup);
            }
            dirty = true;
        }
    }

    void FoliageLayerNode::serializeToJson(nlohmann::json& j) const {
        TerrainNodeBase::serializeToJson(j);
        j["instanceGroupName"] = instanceGroupName;
        j["instanceGroupId"] = instanceGroupId;
        j["useAssetLibrary"] = useAssetLibrary;
        j["assetBiome"] = assetBiome;
        j["assetSources"] = nlohmann::json::array();
        for (const auto& asset : assetSources) {
            j["assetSources"].push_back({
                {"id", asset.id}, {"name", asset.name},
                {"relativeEntryPath", asset.relativeEntryPath}, {"weight", asset.weight},
                {"targetHeight", asset.targetHeight},
                {"heightVariation", asset.heightVariation},
                {"alignToNormal", asset.alignToNormal},
                {"normalInfluence", asset.normalInfluence}
            });
        }
        j["settingsCaptured"] = settingsCaptured;
        j["enabled"] = layerEnabled;
        j["densityMultiplier"] = densityMultiplier;
        j["targetCount"] = targetCount;
        j["seed"] = seed;
        j["minimumDistance"] = minimumDistance;
        j["maximumSlopeDegrees"] = maximumSlopeDegrees;
        j["minimumHeight"] = minimumHeight;
        j["maximumHeight"] = maximumHeight;
        j["densityField"] = densityField;
        j["exclusionField"] = exclusionField;
        j["exclusionThreshold"] = exclusionThreshold;
        j["scaleField"] = scaleField;
        j["scaleFieldInfluence"] = scaleFieldInfluence;
    }

    void FoliageLayerNode::deserializeFromJson(const nlohmann::json& j) {
        TerrainNodeBase::deserializeFromJson(j);
        instanceGroupName = j.value("instanceGroupName", std::string{});
        instanceGroupId = j.value("instanceGroupId", -1);
        useAssetLibrary = j.value("useAssetLibrary", false);
        assetBiome = j.value("assetBiome", std::string{"Auto"});
        assetSources.clear();
        if (j.contains("assetSources") && j["assetSources"].is_array()) {
            for (const auto& item : j["assetSources"]) {
                if (!item.is_object()) continue;
                AssetRef ref;
                ref.id = item.value("id", std::string{});
                ref.name = item.value("name", ref.id);
                ref.relativeEntryPath = item.value("relativeEntryPath", std::string{});
                ref.weight = clampValue(item.value("weight", 1.0f), 0.0f, 100.0f);
                ref.targetHeight = clampValue(item.value("targetHeight", 0.0f), 0.0f, 10000.0f);
                ref.heightVariation = clampValue(item.value("heightVariation", 0.15f), 0.0f, 0.95f);
                ref.alignToNormal = item.value("alignToNormal", false);
                ref.normalInfluence = clampValue(item.value("normalInfluence", 0.0f), 0.0f, 1.0f);
                if (!ref.relativeEntryPath.empty()) assetSources.push_back(std::move(ref));
            }
        }
        settingsCaptured = j.value("settingsCaptured", false);
        layerEnabled = j.value("enabled", layerEnabled);
        densityMultiplier = clampValue(j.value("densityMultiplier", densityMultiplier), 0.0f, 100.0f);
        targetCount = clampValue(j.value("targetCount", targetCount), 0, 10000000);
        seed = j.value("seed", seed);
        minimumDistance = clampValue(j.value("minimumDistance", minimumDistance), 0.0f, 1000.0f);
        maximumSlopeDegrees = clampValue(j.value("maximumSlopeDegrees", maximumSlopeDegrees), 0.0f, 90.0f);
        minimumHeight = j.value("minimumHeight", minimumHeight);
        maximumHeight = j.value("maximumHeight", maximumHeight);
        if (minimumHeight > maximumHeight) std::swap(minimumHeight, maximumHeight);
        densityField = j.value("densityField", std::string{});
        exclusionField = j.value("exclusionField", std::string{});
        exclusionThreshold = clampValue(j.value("exclusionThreshold", exclusionThreshold), 0.0f, 1.0f);
        scaleField = j.value("scaleField", std::string{});
        scaleFieldInfluence = clampValue(j.value("scaleFieldInfluence", scaleFieldInfluence), 0.0f, 1.0f);
    }

    NodeSystem::PinValue FoliageSetNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        (void)outputIndex;
        nlohmann::json layers = nlohmann::json::array();
        for (int inputIndex = 0; inputIndex < static_cast<int>(inputs.size()); ++inputIndex) {
            const auto value = getInputValue(inputIndex, ctx);
            const auto* serialized = std::get_if<std::string>(&value);
            if (!serialized || serialized->empty()) continue;
            const auto layer = nlohmann::json::parse(*serialized, nullptr, false);
            if (!layer.is_discarded() && layer.is_object() &&
                layer.value("kind", std::string{}) == "foliage_layer") {
                layers.push_back(layer);
            }
        }
        nlohmann::json recipe = {
            {"kind", "foliage_set"},
            {"version", 1},
            {"name", setName},
            {"enabled", setEnabled},
            {"densityMultiplier", densityMultiplier},
            {"seedOffset", seedOffset},
            {"layers", std::move(layers)}
        };
        return recipe.dump();
    }

    void FoliageSetNode::drawContent() {
        bool edited = false;
        char setNameBuffer[128] = {};
        std::strncpy(setNameBuffer, setName.c_str(), sizeof(setNameBuffer) - 1);
        if (ImGui::InputText("Set Name", setNameBuffer, sizeof(setNameBuffer))) {
            setName = setNameBuffer;
            edited = true;
        }
        edited |= ImGui::Checkbox("Enabled", &setEnabled);
        edited |= ImGui::DragFloat("Density Multiplier", &densityMultiplier,
                                   0.05f, 0.0f, 100.0f, "%.2fx");
        edited |= ImGui::InputInt("Seed Offset", &seedOffset);
        ImGui::TextDisabled("Each input keeps its own placement rules");
        if (edited) dirty = true;
    }

    void FoliageSetNode::serializeToJson(nlohmann::json& j) const {
        TerrainNodeBase::serializeToJson(j);
        j["setName"] = setName;
        j["enabled"] = setEnabled;
        j["densityMultiplier"] = densityMultiplier;
        j["seedOffset"] = seedOffset;
    }

    void FoliageSetNode::deserializeFromJson(const nlohmann::json& j) {
        TerrainNodeBase::deserializeFromJson(j);
        setName = j.value("setName", setName);
        setEnabled = j.value("enabled", setEnabled);
        densityMultiplier = clampValue(j.value("densityMultiplier", densityMultiplier), 0.0f, 100.0f);
        seedOffset = j.value("seedOffset", seedOffset);
    }

    NodeSystem::PinValue FoliageOutputNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        (void)outputIndex;
        lastAppliedLayerCount = 0;
        lastMissingLayerCount = 0;
        lastMissingAssetCount = 0;
        lastSpawnedInstanceCount = 0;
        lastScatteredGroupIds.clear();
        TerrainContext* terrainContext = getTerrainContext(ctx);
        const auto value = getInputValue(0, ctx);
        const auto* serialized = std::get_if<std::string>(&value);
        if (!serialized || serialized->empty()) return NodeSystem::PinValue{};
        const auto set = nlohmann::json::parse(*serialized, nullptr, false);
        if (set.is_discarded() || !set.is_object() ||
            set.value("kind", std::string{}) != "foliage_set") {
            ctx.addError(id, "Foliage Output requires a Foliage Set input");
            return NodeSystem::PinValue{};
        }
        if (!set.value("enabled", true)) return NodeSystem::PinValue{};

        const float setDensity = clampValue(set.value("densityMultiplier", 1.0f), 0.0f, 100.0f);
        const int seedOffsetValue = set.value("seedOffset", 0);
        const auto layersIt = set.find("layers");
        if (layersIt == set.end() || !layersIt->is_array()) return NodeSystem::PinValue{};

        auto& manager = InstanceManager::getInstance();
        for (const auto& layer : *layersIt) {
            if (!layer.is_object()) continue;
            if (!layer.value("enabled", true)) continue;
            const std::string groupName = layer.value("group", std::string{});
            const int groupId = layer.value("groupId", -1);
            const bool useLibrary = layer.value("useAssetLibrary", false);
            const auto assetsIt = layer.find("assets");
            const bool hasLibraryAssets = useLibrary && assetsIt != layer.end() && assetsIt->is_array() &&
                !assetsIt->empty();
            InstanceGroup* group = groupId >= 0 ? manager.getGroup(groupId) : nullptr;
            if (!group && !groupName.empty()) group = manager.findGroupByName(groupName);
            bool groupWasCreated = false;
            if (!group && hasLibraryAssets && !groupName.empty()) {
                const int createdId = manager.createGroup(groupName, std::string{}, {});
                group = manager.getGroup(createdId);
                groupWasCreated = group != nullptr;
            }
            if (!group || group->transient) {
                ++lastMissingLayerCount;
                continue;
            }
            // Existing groups are authoritative: Terrain UI may have removed or
            // edited a source after this recipe was cached. Only bootstrap from
            // the graph recipe when the shared group genuinely did not exist.
            if (hasLibraryAssets && groupWasCreated) {
                std::vector<FoliageLayerNode::AssetRef> sharedBinding;
                for (const auto& asset : *assetsIt) {
                    FoliageLayerNode::AssetRef ref;
                    ref.id = asset.value("id", std::string{});
                    ref.name = asset.value("name", ref.id);
                    ref.relativeEntryPath = asset.value("relativeEntryPath", std::string{});
                    ref.weight = clampValue(asset.value("weight", 1.0f), 0.0f, 100.0f);
                    ref.targetHeight = clampValue(asset.value("targetHeight", 0.0f), 0.0f, 10000.0f);
                    ref.heightVariation = clampValue(asset.value("heightVariation", 0.15f), 0.0f, 0.95f);
                    ref.alignToNormal = asset.value("alignToNormal", false);
                    ref.normalInfluence = clampValue(asset.value("normalInfluence", 0.0f), 0.0f, 1.0f);
                    if (!ref.relativeEntryPath.empty()) sharedBinding.push_back(std::move(ref));
                }
                if (!synchronizeFoliageLibrarySources(sharedBinding, *group)) ++lastMissingAssetCount;
                if (sharedBinding.empty()) {
                    ++lastMissingLayerCount;
                    continue;
                }
            }
            auto& settings = group->brush_settings;
            const bool captured = layer.value("settingsCaptured", false);
            if (captured && groupWasCreated) {
                settings.density = clampValue(layer.value("density", 1.0f), 0.0f, 10000.0f);
                settings.target_count = clampValue(layer.value("targetCount", settings.target_count), 0, 10000000);
                settings.seed = layer.value("seed", settings.seed);
                settings.min_distance = clampValue(layer.value("minimumDistance", settings.min_distance), 0.0f, 1000.0f);
                settings.slope_max = clampValue(layer.value("maximumSlopeDegrees", settings.slope_max), 0.0f, 90.0f);
                settings.height_min = layer.value("minimumHeight", settings.height_min);
                settings.height_max = layer.value("maximumHeight", settings.height_max);
                if (settings.height_min > settings.height_max) std::swap(settings.height_min, settings.height_max);
                settings.density_mask_attribute = layer.value("densityField", std::string{});
                settings.exclusion_mask_attribute = layer.value("exclusionField", std::string{});
                settings.exclusion_threshold = clampValue(layer.value("exclusionThreshold", settings.exclusion_threshold), 0.0f, 1.0f);
                settings.scale_mask_attribute = layer.value("scaleField", std::string{});
                settings.scale_mask_influence = clampValue(layer.value("scaleFieldInfluence", settings.scale_mask_influence), 0.0f, 1.0f);
            } else if (!captured && groupWasCreated) {
                // Migration path for foliage nodes created before group settings
                // were captured: preserve the proven legacy Scatter settings and
                // only add explicit non-empty named-field bindings.
                const std::string densityFieldValue = layer.value("densityField", std::string{});
                const std::string exclusionFieldValue = layer.value("exclusionField", std::string{});
                const std::string scaleFieldValue = layer.value("scaleField", std::string{});
                if (!densityFieldValue.empty()) settings.density_mask_attribute = densityFieldValue;
                if (!exclusionFieldValue.empty()) settings.exclusion_mask_attribute = exclusionFieldValue;
                if (!scaleFieldValue.empty()) settings.scale_mask_attribute = scaleFieldValue;
            }
            ++lastAppliedLayerCount;
            if (scatterOnApply && terrainContext && terrainContext->terrain &&
                group->target_type == InstanceGroup::TargetType::TERRAIN) {
                const float authoredDensity = settings.density;
                const int authoredSeed = settings.seed;
                settings.density = clampValue(authoredDensity * setDensity, 0.0f, 10000.0f);
                settings.seed = authoredSeed + seedOffsetValue;
                group->clearInstances();
                const int spawned = group->scatterFillTerrain(terrainContext->terrain);
                // Set-level controls affect this batch without corrupting the
                // shared authoring values shown identically in UI and node.
                settings.density = authoredDensity;
                settings.seed = authoredSeed;
                group->gpu_dirty = true;
                lastSpawnedInstanceCount += spawned;
                lastScatteredGroupIds.push_back(group->id);
            }
        }
        return NodeSystem::PinValue{};
    }

    void FoliageOutputNode::drawContent() {
        if (ImGui::Checkbox("Scatter on Apply", &scatterOnApply)) dirty = true;
        ImGui::Text("Applied layers: %d", lastAppliedLayerCount);
        ImGui::Text("Spawned instances: %d", lastSpawnedInstanceCount);
        if (lastMissingLayerCount > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.67f, 0.22f, 1.0f),
                               "Missing layers: %d", lastMissingLayerCount);
        }
        if (lastMissingAssetCount > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.52f, 0.22f, 1.0f),
                               "Missing assets: %d", lastMissingAssetCount);
        }
        ImGui::TextDisabled("Runtime instances remain backend-owned");
    }

    void FoliageOutputNode::serializeToJson(nlohmann::json& j) const {
        TerrainNodeBase::serializeToJson(j);
        j["scatterOnApply"] = scatterOnApply;
    }

    void FoliageOutputNode::deserializeFromJson(const nlohmann::json& j) {
        TerrainNodeBase::deserializeFromJson(j);
        scatterOnApply = j.value("scatterOnApply", true);
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
        if (ImGui::DragInt("Iterations", &iterations, 1, 1, 50)) dirty = true;
        if (ImGui::DragFloat("Strength", &strength, 0.01f, 0.0f, 1.0f)) dirty = true;
        
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
        
        // Auto mode derives the input range. Manual mode treats the input as an
        // already-normalized 0..1 signal and only remaps the output range.
        float minH = 0.0f, maxH = 1.0f;
        if (autoRange) {
            minH = FLT_MAX;
            maxH = -FLT_MAX;
            for (float h : *input.data) {
                minH = (std::min)(minH, h);
                maxH = (std::max)(maxH, h);
            }
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
        if (ImGui::Checkbox("Auto Range", &autoRange)) dirty = true;
        if (!autoRange) {
            if (ImGui::DragFloat("Min Out", &minOutput, 1.0f)) dirty = true;
            if (ImGui::DragFloat("Max Out", &maxOutput, 1.0f)) dirty = true;
        } else {
            ImGui::TextDisabled("Input: Auto");
            if (ImGui::DragFloat("Max Out", &maxOutput, 1.0f, 0.0f, 10000.0f)) dirty = true;
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
    
    // EDGE FALLOFF NODE
    NodeSystem::PinValue EdgeFalloffNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        if (!input.isValid()) {
            ctx.addError(id, "No valid input");
            return NodeSystem::PinValue{};
        }
        
        int w = input.width;
        int h = input.height;
        auto result = createHeightOutput(w, h);
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                float height = (*input.data)[idx];
                
                // Calculate normalized distance from nearest edge (0 to 0.5)
                float dx = (float)x / (std::max(1, w - 1));
                float dy = (float)y / (std::max(1, h - 1));
                
                float distX = std::min(dx, 1.0f - dx);
                float distY = std::min(dy, 1.0f - dy);
                
                // IMPROVED MATH:
                // Instead of min(distX, distY) which creates a diagonal ridge (box distance),
                // we use a smooth combination that rounds the corners.
                float tx = std::clamp(distX / (std::max(0.001f, fadeWidth)), 0.0f, 1.0f);
                float ty = std::clamp(distY / (std::max(0.001f, fadeWidth)), 0.0f, 1.0f);
                
                // Multiplicative falloff eliminates the diagonal "crease"
                float t = tx * ty; 
                
                // Apply optional extra smoothing to the mask itself
                switch (mode) {
                    case FalloffMode::Linear:
                        break;
                    case FalloffMode::Smoothstep:
                        t = t * t * (3.0f - 2.0f * t);
                        break;
                    case FalloffMode::Cosine:
                        t = 0.5f * (1.0f - std::cos(t * 3.14159f));
                        break;
                }
                
                (*result.data)[idx] = fadeValue * (1.0f - t) + height * t;
            }
        }
        
        return result;
    }
    
    void EdgeFalloffNode::drawContent() {
        if (ImGui::SliderFloat("Fade Width", &fadeWidth, 0.001f, 0.5f, "%.3f")) dirty = true;
        if (ImGui::DragFloat("Fade Value", &fadeValue, 0.1f)) dirty = true;
        
        const char* modeNames[] = { "Linear", "Smoothstep", "Cosine" };
        int modeIdx = (int)mode;
        if (ImGui::Combo("Fade Mode", &modeIdx, modeNames, 3)) {
            mode = (FalloffMode)modeIdx;
            dirty = true;
        }
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
        if (inputBase.width != inputBlend.width || inputBase.height != inputBlend.height) {
            ctx.addError(id, "Input resolution mismatch; insert a Resample node");
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
        if (inputBase.width != inputBlend.width || inputBase.height != inputBlend.height) {
            ctx.addError(id, "Input resolution mismatch; insert a Resample node");
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
        if (w < 2 || h < 2) {
            ctx.addError(id, "Auto Splat requires at least a 2x2 height image");
            return NodeSystem::PinValue{};
        }
        
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
        constexpr int kMaxPaintDimension = 8192;
        if (width <= 0 || height <= 0 || width > kMaxPaintDimension || height > kMaxPaintDimension) {
            paintBuffer.clear();
            bufferWidth = 0;
            bufferHeight = 0;
            needsInit = true;
            return;
        }
        if (bufferWidth != width || bufferHeight != height) {
            bufferWidth = width;
            bufferHeight = height;
            paintBuffer.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
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
        if (paintBuffer.empty()) {
            ctx.addError(id, "Mask Paint resolution is invalid or exceeds 8192x8192");
            return NodeSystem::PinValue{};
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
        if (maskData.isValid() && (maskData.width != w || maskData.height != h)) {
            ctx.addError(id, "Fault mask resolution mismatch; insert Resample");
            return NodeSystem::PinValue{};
        }
        bool hasMask = maskData.isValid() &&
                       maskData.width == w && maskData.height == h;
        
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
        if (ImGui::SliderFloat("Direction", &direction, 0.0f, 360.0f, "%.0f°")) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::DragFloat("Offset", &offset, 0.5f, -50.0f, 50.0f)) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::DragFloat("V.Offset", &verticalOffset, 0.1f, -10.0f, 10.0f)) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("Position", &position, 0.0f, 1.0f)) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("Width", &width, 0.1f, 20.0f)) dirty = true;
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
        if (maskData.isValid() && (maskData.width != w || maskData.height != h)) {
            ctx.addError(id, "Mesa mask resolution mismatch; insert Resample");
            return NodeSystem::PinValue{};
        }
        bool hasMask = maskData.isValid() &&
                       maskData.width == w && maskData.height == h;
        
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
        if (ImGui::SliderFloat("Threshold", &threshold, 0.0f, 1.0f)) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("Cliff", &cliffSteepness, 0.0f, 1.0f)) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("Height", &plateauHeight, 0.1f, 2.0f)) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderInt("Terraces", &terraceCount, 1, 10)) dirty = true;
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
        if (maskData.isValid() && (maskData.width != w || maskData.height != h)) {
            ctx.addError(id, "Shear mask resolution mismatch; insert Resample");
            return NodeSystem::PinValue{};
        }
        bool hasMask = maskData.isValid() &&
                       maskData.width == w && maskData.height == h;
        
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
        if (ImGui::SliderFloat("Angle", &angle, -60.0f, 60.0f, "%.0f°")) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("Strength", &strength, 0.0f, 1.0f)) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderInt("Bands", &bands, 1, 16)) dirty = true;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("Width", &bandWidth, 0.05f, 0.5f)) dirty = true;
        if (ImGui::Checkbox("Bidirectional", &bidirectional)) dirty = true;
    }

    // ============================================================================
    // IMAGE CONTRACT / AUTHORING UTILITY NODE IMPLEMENTATIONS
    // ============================================================================

    NodeSystem::PinValue ResampleNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto source = getHeightInput(0, ctx);
        if (!source.isValid()) {
            ctx.addError(id, "Resample requires a single-channel source image");
            return NodeSystem::PinValue{};
        }

        int outW = clampValue(targetWidth, 2, 8192);
        int outH = clampValue(targetHeight, 2, 8192);
        if (matchReference) {
            auto reference = getHeightInput(1, ctx);
            if (reference.isValid()) {
                outW = reference.width;
                outH = reference.height;
            }
        }
        if (outW < 2 || outH < 2) {
            ctx.addError(id, "Resample target must be at least 2x2");
            return NodeSystem::PinValue{};
        }

        auto result = semanticMode == ResampleSemantic::Height
            ? createHeightOutput(outW, outH) : createMaskOutput(outW, outH);
        const int inW = source.width;
        const int inH = source.height;
        for (int y = 0; y < outH; ++y) {
            const float sy = (outH > 1) ? (static_cast<float>(y) * (inH - 1) / (outH - 1)) : 0.0f;
            const int y0 = clampValue(static_cast<int>(std::floor(sy)), 0, inH - 1);
            const int y1 = (std::min)(y0 + 1, inH - 1);
            const float ty = sy - y0;
            for (int x = 0; x < outW; ++x) {
                const float sx = (outW > 1) ? (static_cast<float>(x) * (inW - 1) / (outW - 1)) : 0.0f;
                float value = 0.0f;
                if (filter == ResampleFilter::Nearest) {
                    const int nx = clampValue(static_cast<int>(std::round(sx)), 0, inW - 1);
                    const int ny = clampValue(static_cast<int>(std::round(sy)), 0, inH - 1);
                    value = (*source.data)[ny * inW + nx];
                } else {
                    const int x0 = clampValue(static_cast<int>(std::floor(sx)), 0, inW - 1);
                    const int x1 = (std::min)(x0 + 1, inW - 1);
                    const float tx = sx - x0;
                    const float a = (*source.data)[y0 * inW + x0] * (1.0f - tx) + (*source.data)[y0 * inW + x1] * tx;
                    const float b = (*source.data)[y1 * inW + x0] * (1.0f - tx) + (*source.data)[y1 * inW + x1] * tx;
                    value = a * (1.0f - ty) + b * ty;
                }
                (*result.data)[y * outW + x] = value;
            }
        }
        return result;
    }

    void ResampleNode::drawContent() {
        int semantic = static_cast<int>(semanticMode);
        const char* semanticNames[] = { "Height", "Mask" };
        if (ImGui::Combo("Output", &semantic, semanticNames, 2)) {
            semanticMode = static_cast<ResampleSemantic>(semantic);
            syncSemantic();
            dirty = true;
        }
        if (ImGui::Checkbox("Match Reference", &matchReference)) dirty = true;
        ImGui::BeginDisabled(matchReference);
        if (ImGui::DragInt("Width", &targetWidth, 16, 2, 8192)) dirty = true;
        if (ImGui::DragInt("Height", &targetHeight, 16, 2, 8192)) dirty = true;
        ImGui::EndDisabled();
        int filterIndex = static_cast<int>(filter);
        const char* filterNames[] = { "Nearest", "Bilinear" };
        if (ImGui::Combo("Filter", &filterIndex, filterNames, 2)) {
            filter = static_cast<ResampleFilter>(filterIndex);
            dirty = true;
        }
    }

    NodeSystem::PinValue ChannelExtractNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getImageInput(0, ctx);
        if (!input.isValid() || input.channels < 1 || input.channels > 4) {
            ctx.addError(id, "Channel Extract requires a 1-4 channel image");
            return NodeSystem::PinValue{};
        }
        if (channel >= input.channels) {
            ctx.addError(id, "Requested channel does not exist in the input image");
            return NodeSystem::PinValue{};
        }
        auto result = createMaskOutput(input.width, input.height);
        const size_t pixels = input.pixelCount();
        for (size_t i = 0; i < pixels; ++i) {
            float value = (*input.data)[i * static_cast<size_t>(input.channels) + static_cast<size_t>(channel)];
            (*result.data)[i] = invert ? 1.0f - value : value;
        }
        return result;
    }

    void ChannelExtractNode::drawContent() {
        const char* names[] = { "R / Channel 0", "G / Channel 1", "B / Channel 2", "A / Channel 3" };
        if (ImGui::Combo("Channel", &channel, names, 4)) dirty = true;
        if (ImGui::Checkbox("Invert", &invert)) dirty = true;
    }

    NodeSystem::PinValue SplatComposeNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        NodeSystem::Image2DData channels[4];
        int w = 0, h = 0;
        for (int c = 0; c < 4; ++c) {
            channels[c] = getHeightInput(c, ctx);
            if (!channels[c].isValid()) continue;
            if (w == 0) { w = channels[c].width; h = channels[c].height; }
            else if (channels[c].width != w || channels[c].height != h) {
                ctx.addError(id, "Splat channels must share a resolution; insert Resample");
                return NodeSystem::PinValue{};
            }
        }
        if (w < 1 || h < 1) {
            ctx.addError(id, "Splat Compose requires at least one mask input");
            return NodeSystem::PinValue{};
        }
        NodeSystem::Image2DData result;
        result.width = w; result.height = h; result.channels = 4;
        result.semantic = NodeSystem::ImageSemantic::Mask;
        result.data = std::make_shared<std::vector<float>>(static_cast<size_t>(w) * h * 4, 0.0f);
        const size_t pixels = static_cast<size_t>(w) * h;
        for (size_t i = 0; i < pixels; ++i) {
            float values[4] = {};
            float sum = 0.0f;
            for (int c = 0; c < 4; ++c) {
                if (channels[c].isValid()) values[c] = clampValue((*channels[c].data)[i], 0.0f, 1.0f);
                sum += values[c];
            }
            if (normalize) {
                if (sum > 1e-6f) for (float& value : values) value /= sum;
                else values[0] = 1.0f;
            }
            for (int c = 0; c < 4; ++c) (*result.data)[i * 4 + c] = values[c];
        }
        return result;
    }

    void SplatComposeNode::drawContent() {
        if (ImGui::Checkbox("Normalize Weights", &normalize)) dirty = true;
    }

    NodeSystem::PinValue RemapNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        if (!input.isValid()) {
            ctx.addError(id, "Remap requires a single-channel height input");
            return NodeSystem::PinValue{};
        }
        auto mask = getHeightInput(1, ctx);
        const bool hasMask = mask.isValid() && mask.width == input.width && mask.height == input.height;
        auto result = maskMode ? createMaskOutput(input.width, input.height)
                               : createHeightOutput(input.width, input.height);
        const float range = inputMax - inputMin;
        const float safeGamma = (std::max)(gamma, 0.01f);
        for (size_t i = 0; i < input.pixelCount(); ++i) {
            const float original = (*input.data)[i];
            float t = std::fabs(range) > 1e-8f ? (original - inputMin) / range : 0.0f;
            if (clampOutput) t = clampValue(t, 0.0f, 1.0f);
            t = std::pow((std::max)(t, 0.0f), 1.0f / safeGamma);
            float value = outputMin + t * (outputMax - outputMin);
            if (clampOutput) value = clampValue(value, (std::min)(outputMin, outputMax), (std::max)(outputMin, outputMax));
            const float amount = hasMask ? clampValue((*mask.data)[i], 0.0f, 1.0f) : 1.0f;
            (*result.data)[i] = original * (1.0f - amount) + value * amount;
        }
        return result;
    }

    void RemapNode::drawContent() {
        if (ImGui::Checkbox("Mask Output", &maskMode)) { syncSemantic(); dirty = true; }
        if (ImGui::DragFloat("Input Min", &inputMin, 0.01f)) dirty = true;
        if (ImGui::DragFloat("Input Max", &inputMax, 0.01f)) dirty = true;
        if (ImGui::DragFloat("Output Min", &outputMin, 0.01f)) dirty = true;
        if (ImGui::DragFloat("Output Max", &outputMax, 0.01f)) dirty = true;
        if (ImGui::DragFloat("Gamma", &gamma, 0.01f, 0.01f, 8.0f)) dirty = true;
        if (ImGui::Checkbox("Clamp", &clampOutput)) dirty = true;
    }

    NodeSystem::PinValue MaskAdjustNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto input = getHeightInput(0, ctx);
        const auto effectMask = getHeightInput(1, ctx);
        if (!input.isValid()) {
            ctx.addError(id, "Mask Adjust requires a single-channel mask input");
            return NodeSystem::PinValue{};
        }
        if (effectMask.isValid() &&
            (effectMask.width != input.width || effectMask.height != input.height)) {
            ctx.addError(id, "Mask Adjust Effect Mask must match the input resolution");
            return NodeSystem::PinValue{};
        }

        auto result = createMaskOutput(input.width, input.height);
        const float safeGamma = (std::max)(gamma, 0.05f);
        const float globalMix = clampValue(mix, 0.0f, 1.0f);
        for (size_t index = 0; index < input.pixelCount(); ++index) {
            const float original = (*input.data)[index];
            float adjusted = original * intensity + brightness;
            adjusted = (adjusted - 0.5f) * contrast + 0.5f;
            adjusted = std::pow((std::max)(adjusted, 0.0f), 1.0f / safeGamma);
            if (invert) adjusted = 1.0f - adjusted;
            if (clampOutput) adjusted = clampValue(adjusted, 0.0f, 1.0f);
            const float localMask = effectMask.isValid()
                ? clampValue((*effectMask.data)[index], 0.0f, 1.0f) : 1.0f;
            float value = original + (adjusted - original) * (globalMix * localMask);
            if (clampOutput) value = clampValue(value, 0.0f, 1.0f);
            (*result.data)[index] = value;
        }
        return result;
    }

    void MaskAdjustNode::drawContent() {
        bool edited = false;
        edited |= ImGui::SliderFloat("Intensity", &intensity, 0.0f, 4.0f);
        edited |= ImGui::SliderFloat("Brightness", &brightness, -1.0f, 1.0f);
        edited |= ImGui::SliderFloat("Contrast", &contrast, 0.0f, 4.0f);
        edited |= ImGui::SliderFloat("Gamma", &gamma, 0.05f, 4.0f);
        edited |= ImGui::SliderFloat("Mix", &mix, 0.0f, 1.0f);
        edited |= ImGui::Checkbox("Invert", &invert);
        edited |= ImGui::Checkbox("Clamp 0-1", &clampOutput);
        if (ImGui::Button("Reset")) {
            intensity = 1.0f; brightness = 0.0f; contrast = 1.0f;
            gamma = 1.0f; mix = 1.0f; invert = false; clampOutput = true;
            edited = true;
        }
        if (edited) dirty = true;
    }

    NodeSystem::PinValue MaskMorphologyNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        auto input = getHeightInput(0, ctx);
        if (!input.isValid()) {
            ctx.addError(id, "Mask Morphology requires a single-channel mask");
            return NodeSystem::PinValue{};
        }
        const int w = input.width, h = input.height;
        const int r = clampValue(radius, 1, 12);
        std::vector<float> current = *input.data;
        std::vector<float> horizontal(current.size());
        std::vector<float> vertical(current.size());
        for (int iteration = 0; iteration < clampValue(iterations, 1, 8); ++iteration) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    float value = operation == MaskMorphologyOp::Erode ? 1.0f : 0.0f;
                    float sum = 0.0f;
                    for (int k = -r; k <= r; ++k) {
                        const float sample = current[y * w + clampValue(x + k, 0, w - 1)];
                        if (operation == MaskMorphologyOp::Dilate) value = (std::max)(value, sample);
                        else if (operation == MaskMorphologyOp::Erode) value = (std::min)(value, sample);
                        else sum += sample;
                    }
                    horizontal[y * w + x] = operation == MaskMorphologyOp::Blur ? sum / (2 * r + 1) : value;
                }
            }
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    float value = operation == MaskMorphologyOp::Erode ? 1.0f : 0.0f;
                    float sum = 0.0f;
                    for (int k = -r; k <= r; ++k) {
                        const float sample = horizontal[clampValue(y + k, 0, h - 1) * w + x];
                        if (operation == MaskMorphologyOp::Dilate) value = (std::max)(value, sample);
                        else if (operation == MaskMorphologyOp::Erode) value = (std::min)(value, sample);
                        else sum += sample;
                    }
                    vertical[y * w + x] = operation == MaskMorphologyOp::Blur ? sum / (2 * r + 1) : value;
                }
            }
            current.swap(vertical);
        }
        auto result = createMaskOutput(w, h);
        *result.data = std::move(current);
        return result;
    }

    void MaskMorphologyNode::drawContent() {
        int operationIndex = static_cast<int>(operation);
        const char* names[] = { "Dilate", "Erode", "Blur" };
        if (ImGui::Combo("Operation", &operationIndex, names, 3)) {
            operation = static_cast<MaskMorphologyOp>(operationIndex);
            dirty = true;
        }
        if (ImGui::DragInt("Radius", &radius, 1, 1, 12)) dirty = true;
        if (ImGui::DragInt("Iterations", &iterations, 1, 1, 8)) dirty = true;
    }

    namespace {
        struct TerrainMetricScale {
            float cellSize = 1.0f;
            float heightScale = 1.0f;
            float worldScale = 1.0f;
        };

        TerrainMetricScale resolveTerrainMetricScale(NodeSystem::EvaluationContext& ctx, int width) {
            TerrainMetricScale scale;
            if (auto* terrainCtx = ctx.getDomainContext<TerrainContext>()) {
                scale.worldScale = (std::max)(terrainCtx->scale_xz, 1.0f);
                scale.heightScale = (std::max)(terrainCtx->scale_y, 0.1f);
                scale.cellSize = scale.worldScale / (std::max)(width - 1, 1);
            }
            return scale;
        }

        bool sameImageExtent(const NodeSystem::Image2DData& image, int width, int height) {
            return !image.isValid() || (image.width == width && image.height == height);
        }

        float normalizedFlowValue(const NodeSystem::Image2DData& flow, size_t index, float maximum) {
            if (!flow.isValid()) return 0.0f;
            const float value = (std::max)((*flow.data)[index], 0.0f);
            return maximum > 1e-6f ? clampValue(value / maximum, 0.0f, 1.0f) : 0.0f;
        }

        float surfaceSlope01(const NodeSystem::Image2DData& height, int x, int y,
                             const TerrainMetricScale& scale) {
            const int w = height.width;
            const int h = height.height;
            const int xl = clampValue(x - 1, 0, w - 1);
            const int xr = clampValue(x + 1, 0, w - 1);
            const int yu = clampValue(y - 1, 0, h - 1);
            const int yd = clampValue(y + 1, 0, h - 1);
            const float dx = ((*height.data)[y * w + xr] - (*height.data)[y * w + xl]) * scale.heightScale /
                ((std::max)(2.0f * scale.cellSize, 1e-6f));
            const float dy = ((*height.data)[yd * w + x] - (*height.data)[yu * w + x]) * scale.heightScale /
                ((std::max)(2.0f * scale.cellSize, 1e-6f));
            const float degrees = std::atan(std::sqrt(dx * dx + dy * dy)) * 57.2957795f;
            return clampValue(degrees / 60.0f, 0.0f, 1.0f);
        }

        float directionalExposure01(const NodeSystem::Image2DData& height, int x, int y,
                                    const TerrainMetricScale& scale, float azimuthDegrees,
                                    float elevationDegrees) {
            const int w = height.width;
            const int h = height.height;
            const int xl = clampValue(x - 1, 0, w - 1);
            const int xr = clampValue(x + 1, 0, w - 1);
            const int yu = clampValue(y - 1, 0, h - 1);
            const int yd = clampValue(y + 1, 0, h - 1);
            const float dx = ((*height.data)[y * w + xr] - (*height.data)[y * w + xl]) * scale.heightScale /
                ((std::max)(2.0f * scale.cellSize, 1e-6f));
            const float dz = ((*height.data)[yd * w + x] - (*height.data)[yu * w + x]) * scale.heightScale /
                ((std::max)(2.0f * scale.cellSize, 1e-6f));
            float nx = -dx, ny = 1.0f, nz = -dz;
            const float normalLength = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (normalLength > 1e-6f) { nx /= normalLength; ny /= normalLength; nz /= normalLength; }
            const float azimuth = azimuthDegrees * 0.0174532925f;
            const float elevation = elevationDegrees * 0.0174532925f;
            const float sx = std::sin(azimuth) * std::cos(elevation);
            const float sy = std::sin(elevation);
            const float sz = std::cos(azimuth) * std::cos(elevation);
            return clampValue(nx * sx + ny * sy + nz * sz, 0.0f, 1.0f);
        }

        float positiveConcavity(const NodeSystem::Image2DData& height, int x, int y) {
            const int w = height.width;
            const int h = height.height;
            const int xl = clampValue(x - 1, 0, w - 1);
            const int xr = clampValue(x + 1, 0, w - 1);
            const int yu = clampValue(y - 1, 0, h - 1);
            const int yd = clampValue(y + 1, 0, h - 1);
            const float center = (*height.data)[y * w + x];
            const float neighborAverage = ((*height.data)[y * w + xl] + (*height.data)[y * w + xr] +
                (*height.data)[yu * w + x] + (*height.data)[yd * w + x]) * 0.25f;
            return (std::max)(neighborAverage - center, 0.0f);
        }

        float interpolatedPatchNoise(float u, float v, float scale, int seed) {
            const float px = u * scale;
            const float py = v * scale;
            const int x0 = static_cast<int>(std::floor(px));
            const int y0 = static_cast<int>(std::floor(py));
            const float txRaw = px - static_cast<float>(x0);
            const float tyRaw = py - static_cast<float>(y0);
            const float tx = txRaw * txRaw * (3.0f - 2.0f * txRaw);
            const float ty = tyRaw * tyRaw * (3.0f - 2.0f * tyRaw);
            const float a = simpleNoise(x0, y0, seed);
            const float b = simpleNoise(x0 + 1, y0, seed);
            const float c = simpleNoise(x0, y0 + 1, seed);
            const float d = simpleNoise(x0 + 1, y0 + 1, seed);
            const float top = a + (b - a) * tx;
            const float bottom = c + (d - c) * tx;
            return top + (bottom - top) * ty;
        }

        float maximumImageValue(const NodeSystem::Image2DData& image) {
            if (!image.isValid()) return 0.0f;
            float maximum = 0.0f;
            for (float value : *image.data) maximum = (std::max)(maximum, (std::max)(value, 0.0f));
            return maximum;
        }
    }

    NodeSystem::PinValue WetnessMapNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto flow = getHeightInput(1, ctx);
        const auto soil = getHeightInput(2, ctx);
        if (!height.isValid() || height.width < 2 || height.height < 2) {
            ctx.addError(id, "Wetness Map requires at least a 2x2 height input");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(flow, height.width, height.height) || !sameImageExtent(soil, height.width, height.height)) {
            ctx.addError(id, "Wetness Map inputs must have matching resolutions");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const TerrainMetricScale metric = resolveTerrainMetricScale(ctx, w);
        const float maxFlow = maximumImageValue(flow);
        float maxConcavity = 1e-6f;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                maxConcavity = (std::max)(maxConcavity, positiveConcavity(height, x, y));

        auto result = createMaskOutput(w, h);
        const float influenceSum = (std::max)(flowInfluence + concavityInfluence + flatnessInfluence, 1e-6f);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t index = static_cast<size_t>(y) * w + x;
                const float slope = surfaceSlope01(height, x, y, metric);
                const float flatness = 1.0f - slope;
                const float concavity = clampValue(positiveConcavity(height, x, y) / maxConcavity, 0.0f, 1.0f);
                const float flowValue = normalizedFlowValue(flow, index, maxFlow);
                const float retention = soil.isValid() ? (0.55f + 0.45f * clampValue((*soil.data)[index], 0.0f, 1.0f)) : 1.0f;
                float wetness = (flowValue * flowInfluence + concavity * concavityInfluence +
                    flatness * flatnessInfluence) / influenceSum;
                wetness *= retention * (1.0f - evaporation);
                (*result.data)[index] = clampValue(wetness, 0.0f, 1.0f);
            }
        }
        return result;
    }

    void WetnessMapNode::drawContent() {
        if (ImGui::SliderFloat("Flow", &flowInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Concavity", &concavityInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Flatness", &flatnessInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Evaporation", &evaporation, 0.0f, 1.0f)) dirty = true;
    }

    NodeSystem::PinValue SoilDepthNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto flow = getHeightInput(1, ctx);
        const auto hardness = getHeightInput(2, ctx);
        if (!height.isValid() || height.width < 2 || height.height < 2) {
            ctx.addError(id, "Soil Depth requires at least a 2x2 height input");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(flow, height.width, height.height) || !sameImageExtent(hardness, height.width, height.height)) {
            ctx.addError(id, "Soil Depth inputs must have matching resolutions");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const TerrainMetricScale metric = resolveTerrainMetricScale(ctx, w);
        const float maxFlow = maximumImageValue(flow);
        float maxConcavity = 1e-6f;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                maxConcavity = (std::max)(maxConcavity, positiveConcavity(height, x, y));

        auto result = createMaskOutput(w, h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t index = static_cast<size_t>(y) * w + x;
                const float slope = surfaceSlope01(height, x, y, metric);
                const float flatness = 1.0f - slope;
                const float rock = hardness.isValid() ? clampValue((*hardness.data)[index], 0.0f, 1.0f) : 0.35f;
                const float flowValue = normalizedFlowValue(flow, index, maxFlow);
                const float concavity = clampValue(positiveConcavity(height, x, y) / maxConcavity, 0.0f, 1.0f);
                float soilDepth = production * (1.0f - rock) * (0.25f + 0.75f * flatness);
                soilDepth += depositionInfluence * flowValue * flatness;
                soilDepth += concavityInfluence * concavity;
                soilDepth *= clampValue(1.0f - slope * slopeLoss, 0.0f, 1.0f);
                (*result.data)[index] = clampValue(soilDepth, 0.0f, 1.0f);
            }
        }
        return result;
    }

    void SoilDepthNode::drawContent() {
        if (ImGui::SliderFloat("Production", &production, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Deposition", &depositionInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Concavity", &concavityInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Slope Loss", &slopeLoss, 0.0f, 2.0f)) dirty = true;
    }

    NodeSystem::PinValue LithologyNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto warp = getHeightInput(1, ctx);
        if (!height.isValid()) {
            ctx.addError(id, "Lithology requires a height input");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(warp, height.width, height.height)) {
            ctx.addError(id, "Lithology Warp must match the height resolution");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const TerrainMetricScale metric = resolveTerrainMetricScale(ctx, w);
        const float azimuth = dipAzimuth * 0.0174532925f;
        const float dip = std::tan(dipDegrees * 0.0174532925f);
        const float thickness = (std::max)(layerThickness, 0.05f);
        const int count = clampValue(layerCount, 2, 32);
        auto result = createMaskOutput(w, h);

        for (int y = 0; y < h; ++y) {
            const float v = h > 1 ? static_cast<float>(y) / (h - 1) : 0.0f;
            for (int x = 0; x < w; ++x) {
                const float u = w > 1 ? static_cast<float>(x) / (w - 1) : 0.0f;
                const size_t index = static_cast<size_t>(y) * w + x;
                const float worldX = (u - 0.5f) * metric.worldScale;
                const float worldZ = (v - 0.5f) * metric.worldScale;
                const float projected = worldX * std::cos(azimuth) + worldZ * std::sin(azimuth);
                const float warpOffset = warp.isValid()
                    ? (clampValue((*warp.data)[index], 0.0f, 1.0f) - 0.5f) * thickness * warpStrength
                    : 0.0f;
                const float layerCoordinate = ((*height.data)[index] * metric.heightScale + projected * dip + warpOffset) / thickness;
                const int rawLayer = static_cast<int>(std::floor(layerCoordinate));
                const int layerIndex = ((rawLayer % count) + count) % count;
                if (outputIndex == 1) {
                    (*result.data)[index] = count > 1 ? static_cast<float>(layerIndex) / (count - 1) : 0.0f;
                } else {
                    const float alternating = (layerIndex & 1) ? 0.22f : 0.78f;
                    const float randomLayer = simpleNoise(layerIndex, 0, seed);
                    const float layerHardness = alternating * 0.72f + randomLayer * 0.28f;
                    (*result.data)[index] = clampValue(baseHardness +
                        (layerHardness - 0.5f) * hardnessContrast * 1.6f, 0.0f, 1.0f);
                }
            }
        }
        return result;
    }

    void LithologyNode::drawContent() {
        if (ImGui::DragInt("Layer Count", &layerCount, 1, 2, 32)) dirty = true;
        if (ImGui::DragFloat("Thickness", &layerThickness, 0.1f, 0.05f, 1000.0f)) dirty = true;
        if (ImGui::SliderFloat("Base Hardness", &baseHardness, 0.0f, 1.0f)) dirty = true;
        if (ImGui::SliderFloat("Contrast", &hardnessContrast, 0.0f, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Dip", &dipDegrees, 0.25f, -75.0f, 75.0f, "%.1f deg")) dirty = true;
        if (ImGui::DragFloat("Azimuth", &dipAzimuth, 1.0f, 0.0f, 360.0f, "%.0f deg")) dirty = true;
        if (ImGui::SliderFloat("Warp", &warpStrength, 0.0f, 2.0f)) dirty = true;
        if (ImGui::DragInt("Seed", &seed)) dirty = true;
    }

    NodeSystem::PinValue StrataNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto hardnessInput = getHeightInput(1, ctx);
        const auto mask = getHeightInput(2, ctx);
        if (!height.isValid()) {
            ctx.addError(id, "Strata requires a height input");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(hardnessInput, height.width, height.height) || !sameImageExtent(mask, height.width, height.height)) {
            ctx.addError(id, "Strata inputs must have matching resolutions");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const TerrainMetricScale metric = resolveTerrainMetricScale(ctx, w);
        const float azimuth = dipAzimuth * 0.0174532925f;
        const float dip = std::tan(dipDegrees * 0.0174532925f);
        const float thickness = (std::max)(layerThickness, 0.05f);
        NodeSystem::Image2DData result = outputIndex == 0 ? createHeightOutput(w, h) : createMaskOutput(w, h);

        for (int y = 0; y < h; ++y) {
            const float v = h > 1 ? static_cast<float>(y) / (h - 1) : 0.0f;
            for (int x = 0; x < w; ++x) {
                const float u = w > 1 ? static_cast<float>(x) / (w - 1) : 0.0f;
                const size_t index = static_cast<size_t>(y) * w + x;
                const float worldX = (u - 0.5f) * metric.worldScale;
                const float worldZ = (v - 0.5f) * metric.worldScale;
                const float projected = worldX * std::cos(azimuth) + worldZ * std::sin(azimuth);
                const float coordinate = ((*height.data)[index] * metric.heightScale + projected * dip) / thickness;
                const float phase = coordinate - std::floor(coordinate);
                float generatedHardness = 0.5f + 0.5f * std::cos(phase * 6.283185307f);
                generatedHardness = std::pow(clampValue(generatedHardness, 0.0f, 1.0f), edgeSharpness);
                const float hardness = hardnessInput.isValid()
                    ? clampValue((*hardnessInput.data)[index], 0.0f, 1.0f) : generatedHardness;
                if (outputIndex == 1) {
                    (*result.data)[index] = hardness;
                } else {
                    const float amount = mask.isValid() ? clampValue((*mask.data)[index], 0.0f, 1.0f) : 1.0f;
                    const float relief = (hardness - 0.5f) * 2.0f * reliefStrength * amount;
                    (*result.data)[index] = clampValue((*height.data)[index] + relief, 0.0f, 1.0f);
                }
            }
        }
        return result;
    }

    void StrataNode::drawContent() {
        if (ImGui::DragFloat("Thickness", &layerThickness, 0.1f, 0.05f, 1000.0f)) dirty = true;
        if (ImGui::DragFloat("Dip", &dipDegrees, 0.25f, -75.0f, 75.0f, "%.1f deg")) dirty = true;
        if (ImGui::DragFloat("Azimuth", &dipAzimuth, 1.0f, 0.0f, 360.0f, "%.0f deg")) dirty = true;
        if (ImGui::SliderFloat("Relief", &reliefStrength, 0.0f, 0.25f)) dirty = true;
        if (ImGui::SliderFloat("Edge Sharpness", &edgeSharpness, 0.25f, 8.0f)) dirty = true;
    }

    NodeSystem::PinValue SurfaceComposerNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto soil = getHeightInput(1, ctx);
        const auto flow = getHeightInput(2, ctx);
        const auto wetness = getHeightInput(3, ctx);
        const auto hardness = getHeightInput(4, ctx);
        const auto snow = getHeightInput(5, ctx);
        const auto ice = getHeightInput(6, ctx);
        const auto meltwater = getHeightInput(7, ctx);
        const auto grass = getHeightInput(8, ctx);
        const auto rock = getHeightInput(9, ctx);
        if (!height.isValid() || height.width < 2 || height.height < 2) {
            ctx.addError(id, "Surface Composer requires at least a 2x2 height input");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(soil, height.width, height.height) || !sameImageExtent(flow, height.width, height.height) ||
            !sameImageExtent(wetness, height.width, height.height) || !sameImageExtent(hardness, height.width, height.height) ||
            !sameImageExtent(snow, height.width, height.height) || !sameImageExtent(ice, height.width, height.height) ||
            !sameImageExtent(meltwater, height.width, height.height) ||
            !sameImageExtent(grass, height.width, height.height) || !sameImageExtent(rock, height.width, height.height)) {
            ctx.addError(id, "Surface Composer inputs must have matching resolutions");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const TerrainMetricScale metric = resolveTerrainMetricScale(ctx, w);
        NodeSystem::Image2DData result;
        if (outputIndex == 0) {
            result = createMaskOutput(w, h);
        } else {
            result.data = std::make_shared<std::vector<float>>(static_cast<size_t>(w) * h * 4, 0.0f);
            result.width = w; result.height = h; result.channels = 4;
            result.semantic = NodeSystem::ImageSemantic::Mask;
        }

        const float influenceTotal = (std::max)(patchiness + slopeInfluence + soilInfluence + flowInfluence +
            wetnessInfluence + hardnessInfluence + (grass.isValid() ? grassInfluence : 0.0f) +
            (rock.isValid() ? rockInfluence : 0.0f) + (snow.isValid() ? snowInfluence : 0.0f) +
            (ice.isValid() ? iceInfluence : 0.0f), 1e-6f);
        for (int y = 0; y < h; ++y) {
            const float v = h > 1 ? static_cast<float>(y) / (h - 1) : 0.0f;
            for (int x = 0; x < w; ++x) {
                const float u = w > 1 ? static_cast<float>(x) / (w - 1) : 0.0f;
                const size_t index = static_cast<size_t>(y) * w + x;
                const float patch = interpolatedPatchNoise(u, v, textureScale, seed);
                const float slope = surfaceSlope01(height, x, y, metric);
                const float hard = hardness.isValid() ? clampValue((*hardness.data)[index], 0.0f, 1.0f) : 0.45f;
                const float soilValue = soil.isValid() ? clampValue((*soil.data)[index], 0.0f, 1.0f)
                    : clampValue((1.0f - slope) * (1.0f - hard), 0.0f, 1.0f);
                const float grassValue = grass.isValid() ? clampValue((*grass.data)[index], 0.0f, 1.0f)
                    : soilValue;
                const float rockValue = rock.isValid() ? clampValue((*rock.data)[index], 0.0f, 1.0f)
                    : clampValue(slope * 0.75f + hard * hardnessInfluence * 0.55f, 0.0f, 1.0f);
                // Mask inputs already obey the 0..1 authoring contract. Re-
                // normalizing each image by its own maximum amplified weak flow
                // noise into full-strength rivers and changed masks that were
                // already correct in their node previews / Generate Mask path.
                const float erosionFlow = flow.isValid()
                    ? clampValue((*flow.data)[index], 0.0f, 1.0f) : 0.0f;
                const float climateFlow = meltwater.isValid()
                    ? clampValue((*meltwater.data)[index], 0.0f, 1.0f) : 0.0f;
                const float flowValue = 1.0f - (1.0f - erosionFlow) * (1.0f - climateFlow);
                const float wetValue = wetness.isValid() ? clampValue((*wetness.data)[index], 0.0f, 1.0f)
                    : flowValue * (1.0f - slope * 0.6f);
                const float snowValue = snow.isValid() ? clampValue((*snow.data)[index], 0.0f, 1.0f) : 0.0f;
                const float iceValue = ice.isValid() ? clampValue((*ice.data)[index], 0.0f, 1.0f) : 0.0f;
                const float weightedSnow = clampValue(snowValue * snowInfluence, 0.0f, 1.0f);
                const float weightedIce = clampValue(iceValue * iceInfluence, 0.0f, 1.0f);
                const float frozenCover = 1.0f - (1.0f - weightedSnow) * (1.0f - weightedIce);

                float surface = (patch * patchiness + slope * slopeInfluence + soilValue * soilInfluence +
                    flowValue * flowInfluence + wetValue * wetnessInfluence + hard * hardnessInfluence +
                    grassValue * grassInfluence + rockValue * rockInfluence +
                    snowValue * snowInfluence + iceValue * iceInfluence) / influenceTotal;
                surface = clampValue((surface - 0.5f) * contrast + 0.5f, 0.0f, 1.0f);
                if (outputIndex == 0) {
                    (*result.data)[index] = surface;
                    continue;
                }

                // Frozen cover is an overlay, not another peer in a final
                // normalization. Reserve its exact coverage first, then
                // normalize only the underlying authored layers in the
                // remainder. This keeps grass/rock/flow intact beneath snow.
                const float lowerCoverage = 1.0f - frozenCover;
                float lowerWeights[3];
                // Connected layer masks are authored truth: preserve their
                // shapes exactly. Procedural patch/slope/wetness synthesis is a
                // fallback only for missing pins, never a second interpretation
                // of an already-authored Grass, Rock or Flow mask.
                lowerWeights[0] = grass.isValid()
                    ? grassValue
                    : grassValue * grassInfluence * (0.75f + 0.25f * patch);
                lowerWeights[1] = rock.isValid()
                    ? rockValue
                    : rockValue * rockInfluence;
                lowerWeights[2] = flow.isValid()
                    ? flowValue
                    : clampValue((std::max)(wetValue, flowValue * 0.8f), 0.0f, 1.0f);
                float lowerSum = lowerWeights[0] + lowerWeights[1] + lowerWeights[2];
                if (lowerSum <= 1e-6f) {
                    lowerWeights[0] = 1.0f;
                    lowerSum = 1.0f;
                }
                (*result.data)[index * 4 + 0] = lowerWeights[0] / lowerSum * lowerCoverage;
                (*result.data)[index * 4 + 1] = lowerWeights[1] / lowerSum * lowerCoverage;
                (*result.data)[index * 4 + 2] = frozenCover;
                (*result.data)[index * 4 + 3] = lowerWeights[2] / lowerSum * lowerCoverage;
            }
        }
        return result;
    }

    void SurfaceComposerNode::drawContent() {
        ImGui::TextDisabled("RGBA: Base / Rock / Snow / Flow");
        ImGui::TextDisabled("Flow + Meltwater are merged into A");
        ImGui::TextDisabled("Connected Grass/Rock/Flow masks are preserved");
        if (ImGui::DragFloat("Texture Scale", &textureScale, 0.25f, 1.0f, 256.0f)) dirty = true;
        if (ImGui::SliderFloat("Patchiness", &patchiness, 0.0f, 1.0f)) dirty = true;
        if (ImGui::SliderFloat("Slope", &slopeInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Soil", &soilInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Flow", &flowInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Wetness", &wetnessInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Hardness", &hardnessInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Grass / Base", &grassInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Rock / Slope", &rockInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Snow", &snowInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Ice", &iceInfluence, 0.0f, 2.0f)) dirty = true;
        if (ImGui::SliderFloat("Contrast", &contrast, 0.1f, 4.0f)) dirty = true;
        if (ImGui::DragInt("Seed", &seed)) dirty = true;
    }

    const char* SnowClimateNode::getPresetName(SnowClimatePreset value) {
        switch (value) {
            case SnowClimatePreset::Custom: return "Custom";
            case SnowClimatePreset::AlpineBalanced: return "Alpine Balanced";
            case SnowClimatePreset::DeepWinter: return "Deep Winter";
            case SnowClimatePreset::SpringThaw: return "Spring Thaw";
            case SnowClimatePreset::WindblownPeaks: return "Windblown Peaks";
            case SnowClimatePreset::GlacierValley: return "Glacier Valley";
            default: return "Custom";
        }
    }

    void SnowClimateNode::applyPreset(SnowClimatePreset value) {
        if (value != SnowClimatePreset::Custom) {
            relativeSnowLine = true;
            affectGeometry = true;
            geometryAmount = 1.0f;
            coverageAmount = 1.0f;
        }
        switch (value) {
            case SnowClimatePreset::AlpineBalanced:
                snowfallMeters = 0.45f; maxDepthMeters = 1.8f; snowLineFraction = 0.58f; snowLineBlendFraction = 0.14f;
                baseTemperature = -3.0f; meltAmount = 0.18f; solarMelt = 0.30f;
                refreezeRate = 0.35f; valleyCapture = 0.75f; transportRate = 0.32f;
                slipAngle = 38.0f; settleIterations = 18; waterIterations = 10;
                windStrength = 0.08f;
                break;
            case SnowClimatePreset::DeepWinter:
                snowfallMeters = 1.25f; maxDepthMeters = 3.5f; snowLineFraction = 0.28f; snowLineBlendFraction = 0.22f;
                baseTemperature = -12.0f; meltAmount = 0.03f; solarMelt = 0.08f;
                refreezeRate = 0.55f; valleyCapture = 1.15f; transportRate = 0.38f;
                slipAngle = 36.0f; settleIterations = 28; waterIterations = 6;
                windStrength = 0.10f;
                break;
            case SnowClimatePreset::SpringThaw:
                snowfallMeters = 0.75f; maxDepthMeters = 2.5f; snowLineFraction = 0.52f; snowLineBlendFraction = 0.18f;
                baseTemperature = 3.0f; meltAmount = 0.58f; solarMelt = 0.65f;
                refreezeRate = 0.22f; valleyCapture = 0.85f; transportRate = 0.30f;
                slipAngle = 37.0f; settleIterations = 18; waterIterations = 22;
                windStrength = 0.04f;
                break;
            case SnowClimatePreset::WindblownPeaks:
                snowfallMeters = 0.35f; maxDepthMeters = 1.2f; snowLineFraction = 0.68f; snowLineBlendFraction = 0.10f;
                baseTemperature = -7.0f; meltAmount = 0.10f; solarMelt = 0.28f;
                refreezeRate = 0.40f; valleyCapture = 0.65f; transportRate = 0.40f;
                slipAngle = 39.0f; settleIterations = 24; waterIterations = 8;
                windStrength = 0.48f;
                break;
            case SnowClimatePreset::GlacierValley:
                snowfallMeters = 2.50f; maxDepthMeters = 8.0f; snowLineFraction = 0.22f; snowLineBlendFraction = 0.25f;
                baseTemperature = -14.0f; meltAmount = 0.05f; solarMelt = 0.10f;
                refreezeRate = 0.82f; valleyCapture = 1.65f; transportRate = 0.24f;
                slipAngle = 34.0f; settleIterations = 36; waterIterations = 14;
                windStrength = 0.06f;
                break;
            case SnowClimatePreset::Custom:
            default:
                break;
        }
    }

    NodeSystem::PinValue SnowClimateNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto exposureInput = getHeightInput(1, ctx);
        const auto authoredMask = getHeightInput(2, ctx);
        if (!height.isValid() || height.width < 2 || height.height < 2) {
            ctx.addError(id, "Snow Layer requires at least a 2x2 base height input");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(exposureInput, height.width, height.height) ||
            !sameImageExtent(authoredMask, height.width, height.height)) {
            ctx.addError(id, "Snow Layer inputs must have matching resolutions");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const size_t pixelCount = static_cast<size_t>(w) * h;
        const TerrainMetricScale metric = resolveTerrainMetricScale(ctx, w);
        const float safeHeightScale = (std::max)(metric.heightScale, 0.1f);
        const float safeMaxDepth = (std::max)(maxDepthMeters, 0.01f);
        float effectiveSunAzimuth = sunAzimuth;
        float effectiveSunElevation = sunElevation;
        if (useSceneSun) {
            if (const auto* terrainCtx = getTerrainContext(ctx); terrainCtx && terrainCtx->has_scene_sun) {
                const float sx = terrainCtx->scene_sun_x;
                const float sy = terrainCtx->scene_sun_y;
                const float sz = terrainCtx->scene_sun_z;
                const float length = std::sqrt(sx * sx + sy * sy + sz * sz);
                if (length > 1e-6f) {
                    effectiveSunAzimuth = std::atan2(sx / length, sz / length) * 57.2957795f;
                    if (effectiveSunAzimuth < 0.0f) effectiveSunAzimuth += 360.0f;
                    effectiveSunElevation = std::asin(clampValue(sy / length, -1.0f, 1.0f)) * 57.2957795f;
                }
            }
        }
        // These solvers are deterministic gather/scatter passes, not OpenMP
        // loops (parallel neighbor writes would race). Bound their total work
        // on large terrains so Evaluate cannot look like an infinite wait.
        constexpr size_t kTargetCellPasses = 32ull * 1024ull * 1024ull;
        const int resolutionPassBudget = static_cast<int>((std::max)(
            size_t{2}, kTargetCellPasses / (std::max)(pixelCount, size_t{1})));
        const int settlePasses = clampValue((std::min)(settleIterations, resolutionPassBudget), 2, 64);
        const int runoffPasses = clampValue((std::min)(waterIterations, resolutionPassBudget), 2, 64);
        SCENE_LOG_INFO("[Snow Layer] Starting " + std::to_string(w) + "x" + std::to_string(h) +
            " solve (settle=" + std::to_string(settlePasses) +
            ", runoff=" + std::to_string(runoffPasses) + ").");
        const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        const float distance[8] = {1.41421356f, 1.0f, 1.41421356f, 1.0f,
            1.0f, 1.41421356f, 1.0f, 1.41421356f};

        // Reuse the eventual output allocations as solver fields. This keeps
        // peak memory at five result images plus five scratch images instead
        // of retaining a second full set of climate/trace buffers.
        auto finalHeight = createHeightOutput(w, h);
        auto snowMask = createMaskOutput(w, h);
        auto iceMask = createMaskOutput(w, h);
        auto meltwaterMask = createMaskOutput(w, h);
        auto avalancheMask = createMaskOutput(w, h);
        std::vector<float>& baseMeters = *finalHeight.data;
        std::vector<float>& coldness = *snowMask.data;
        std::vector<float>& solar = *iceMask.data;
        std::vector<float>& runoffTrace = *meltwaterMask.data;
        std::vector<float>& avalancheDeposit = *avalancheMask.data;
        std::vector<float> snowDepth(pixelCount, 0.0f);
        std::vector<float> nextDepth(pixelCount, 0.0f);
        std::vector<float> windExposure(pixelCount, 0.0f);

        float minimumBase = (std::numeric_limits<float>::max)();
        float maximumBase = (std::numeric_limits<float>::lowest)();
        for (size_t index = 0; index < pixelCount; ++index) {
            const float base = (*height.data)[index] * safeHeightScale;
            baseMeters[index] = base;
            minimumBase = (std::min)(minimumBase, base);
            maximumBase = (std::max)(maximumBase, base);
        }
        const float terrainRelief = (std::max)(maximumBase - minimumBase, 0.01f);
        const float effectiveSnowLine = relativeSnowLine
            ? minimumBase + clampValue(snowLineFraction, 0.0f, 1.0f) * terrainRelief
            : snowLine;
        const float effectiveLineBlend = relativeSnowLine
            ? (std::max)(snowLineBlendFraction * terrainRelief, 0.01f)
            : (std::max)(snowLineTransition, 0.01f);

        // Integral height field gives a broad terrain-position signal in O(N).
        // Unlike the old immediate-neighbour test, it recognizes wide valley
        // floors and bowls instead of reducing accumulation to a height mask.
        const size_t integralStride = static_cast<size_t>(w) + 1u;
        std::vector<float> heightIntegral(integralStride * (static_cast<size_t>(h) + 1u), 0.0f);
        for (int y = 0; y < h; ++y) {
            float rowSum = 0.0f;
            for (int x = 0; x < w; ++x) {
                rowSum += baseMeters[static_cast<size_t>(y) * w + x];
                heightIntegral[(static_cast<size_t>(y) + 1u) * integralStride + static_cast<size_t>(x) + 1u] =
                    heightIntegral[static_cast<size_t>(y) * integralStride + static_cast<size_t>(x) + 1u] + rowSum;
            }
        }
        const auto boxAverageMeters = [&](int centerX, int centerY, int radius) {
            const int x0 = (std::max)(centerX - radius, 0);
            const int y0 = (std::max)(centerY - radius, 0);
            const int x1 = (std::min)(centerX + radius, w - 1);
            const int y1 = (std::min)(centerY + radius, h - 1);
            const size_t ax = static_cast<size_t>(x0), ay = static_cast<size_t>(y0);
            const size_t bx = static_cast<size_t>(x1 + 1), by = static_cast<size_t>(y1 + 1);
            const float sum = heightIntegral[by * integralStride + bx] -
                heightIntegral[ay * integralStride + bx] -
                heightIntegral[by * integralStride + ax] +
                heightIntegral[ay * integralStride + ax];
            const float area = static_cast<float>((x1 - x0 + 1) * (y1 - y0 + 1));
            return sum / (std::max)(area, 1.0f);
        };
        const int minimumExtent = (std::min)(w, h);
        const int nearRadius = (std::max)(2, minimumExtent / 96);
        const int farRadius = (std::max)(nearRadius + 1, minimumExtent / 24);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t index = static_cast<size_t>(y) * w + x;
                const float base = baseMeters[index];
                const float sun = exposureInput.isValid()
                    ? clampValue((*exposureInput.data)[index], 0.0f, 1.0f)
                    : directionalExposure01(height, x, y, metric, effectiveSunAzimuth, effectiveSunElevation);
                solar[index] = sun;
                const float altitudeAboveBase = base - minimumBase;
                const float temperature = baseTemperature - lapseRate * (altitudeAboveBase / 1000.0f) + sun * 6.0f;
                const float thermalCold = clampValue((2.0f - temperature) / 8.0f, 0.0f, 1.0f);
                const float altitudeCold = smoothstepLocal(
                    effectiveSnowLine - effectiveLineBlend, effectiveSnowLine + effectiveLineBlend, base);
                // Snowline is the coverage gate; temperature shapes retention
                // above it. Using max() here made a cold preset bypass the
                // snowline entirely and coat every altitude.
                const float cold = altitudeCold * (0.35f + thermalCold * 0.65f);
                coldness[index] = cold;

                const float nearRise = boxAverageMeters(x, y, nearRadius) - base;
                const float farRise = boxAverageMeters(x, y, farRadius) - base;
                const float valleyRise = (std::max)(nearRise, farRise);
                const float pocket = clampValue(valleyRise /
                    (std::max)(terrainRelief * 0.08f, 0.05f), 0.0f, 1.0f);
                const float slope = surfaceSlope01(height, x, y, metric);
                const float retention = clampValue(1.0f - slope * 0.72f, 0.15f, 1.0f);
                const float valleyBoost = 1.0f + pocket * valleyCapture *
                    (1.5f + 2.0f * (1.0f - slope));
                // Directional wind scour: wind-facing slopes lose snow while
                // broad concave pockets remain sheltered accumulation zones.
                // Remove the flat-surface elevation baseline so azimuth, not a
                // constant upward component, selects the exposed face.
                constexpr float windElevationDegrees = 8.0f;
                constexpr float flatWindBaseline = 0.1391731f; // sin(8 deg)
                // windAzimuth is the transport direction; exposure faces the
                // incoming wind, which is the opposite horizontal direction.
                const float incomingWindAzimuth = std::fmod(windAzimuth + 180.0f, 360.0f);
                const float rawWindExposure = directionalExposure01(
                    height, x, y, metric, incomingWindAzimuth, windElevationDegrees);
                const float directionalWind = clampValue(
                    (rawWindExposure - flatWindBaseline) / (1.0f - flatWindBaseline), 0.0f, 1.0f);
                windExposure[index] = directionalWind * (1.0f - pocket * 0.88f);
                const float maskValue = authoredMask.isValid()
                    ? clampValue((*authoredMask.data)[index], 0.0f, 1.0f) : 1.0f;
                const float sunRetention = 1.0f - sun * solarMelt * 0.35f;
                snowDepth[index] = clampValue(
                    snowfallMeters * cold * retention * valleyBoost * sunRetention * maskValue,
                    0.0f, safeMaxDepth);
            }
        }
        ctx.reportNodeProgress(0.12f);
        std::vector<float>().swap(heightIntegral);

        const float windRadians = windAzimuth * 0.0174532925f;
        const int windDx = clampValue(static_cast<int>(std::round(std::sin(windRadians))), -1, 1);
        const int windDy = clampValue(static_cast<int>(std::round(std::cos(windRadians))), -1, 1);
        // A one-cell transport step makes iteration count secretly depend on
        // heightmap resolution: even 64 passes travel only 3.1 m on a 2K / 100 m
        // terrain. Start on a coarse footprint and converge to one-cell passes.
        // This extends downhill reach without multiplying the O(N) pass budget,
        // while the final local passes still settle the actual mesh grid.
        const int maximumTransportStride = clampValue(minimumExtent / 64, 1, 32);
        for (int iteration = 0; iteration < settlePasses; ++iteration) {
            if (ctx.isCancelled()) return NodeSystem::PinValue{};
            const float coarseT = settlePasses > 1
                ? 1.0f - static_cast<float>(iteration) / static_cast<float>(settlePasses - 1)
                : 0.0f;
            const int transportStride = (std::max)(1, static_cast<int>(std::round(
                1.0f + (maximumTransportStride - 1) * coarseT * coarseT)));
            std::fill(nextDepth.begin(), nextDepth.end(), 0.0f);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t index = static_cast<size_t>(y) * w + x;
                    const float sourceDepth = (std::max)(snowDepth[index], 0.0f);
                    const float sourceSurface = baseMeters[index] + sourceDepth;
                    float weights[8] = {};
                    float weightSum = 0.0f;
                    float steepestDegrees = 0.0f;
                    for (int direction = 0; direction < 8; ++direction) {
                        const int nx = x + dx[direction] * transportStride;
                        const int ny = y + dy[direction] * transportStride;
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        const size_t neighbor = static_cast<size_t>(ny) * w + nx;
                        const float drop = sourceSurface - (baseMeters[neighbor] + snowDepth[neighbor]);
                        if (drop <= 0.0f) continue;
                        const float run = (std::max)(
                            metric.cellSize * distance[direction] * transportStride, 1e-6f);
                        const float degrees = std::atan(drop / run) * 57.2957795f;
                        weights[direction] = drop / run;
                        weightSum += weights[direction];
                        steepestDegrees = (std::max)(steepestDegrees, degrees);
                    }

                    const float avalancheExcess = steepestDegrees > slipAngle
                        ? clampValue((steepestDegrees - slipAngle) /
                            (std::max)(90.0f - slipAngle, 1.0f), 0.0f, 1.0f) : 0.0f;
                    const float settling = weightSum > 0.0f
                        ? clampValue(weightSum * metric.cellSize * transportStride /
                            (safeMaxDepth * 8.0f + 1e-6f), 0.0f, 1.0f) : 0.0f;
                    float movedFraction = transportRate * (avalancheExcess + settling * 0.16f);
                    movedFraction = clampValue(movedFraction, 0.0f, 0.65f);
                    float moved = sourceDepth * movedFraction;
                    float remaining = sourceDepth - moved;
                    if (weightSum > 0.0f && moved > 0.0f) {
                        for (int direction = 0; direction < 8; ++direction) {
                            if (weights[direction] <= 0.0f) continue;
                            const int nx = x + dx[direction] * transportStride;
                            const int ny = y + dy[direction] * transportStride;
                            const size_t neighbor = static_cast<size_t>(ny) * w + nx;
                            const float transfer = moved * (weights[direction] / weightSum);
                            nextDepth[neighbor] += transfer;
                            if (avalancheExcess > 0.0f) avalancheDeposit[neighbor] += transfer;
                        }
                    } else {
                        remaining += moved;
                    }

                    const int wx = x + windDx, wy = y + windDy;
                    if ((windDx != 0 || windDy != 0) && wx >= 0 && wx < w && wy >= 0 && wy < h) {
                        // Repeated passes can now genuinely scour the windward
                        // face and deposit that conserved mass downwind/into
                        // sheltered valleys. The old fixed 0.025 factor barely
                        // moved snow even with Wind at maximum.
                        const float windMoved = remaining * clampValue(
                            windStrength * (0.025f + windExposure[index] * 0.22f), 0.0f, 0.30f);
                        remaining -= windMoved;
                        nextDepth[static_cast<size_t>(wy) * w + wx] += windMoved;
                    }
                    nextDepth[index] += remaining;
                }
            }
            // Gather transport can converge many source cells into a single
            // sink. Enforce the authored physical capacity every pass so those
            // sinks cannot turn into vertex-height needles.
            for (size_t index = 0; index < pixelCount; ++index) {
                const float value = nextDepth[index];
                nextDepth[index] = std::isfinite(value) ? clampValue(value, 0.0f, safeMaxDepth) : 0.0f;
            }
            snowDepth.swap(nextDepth);
            ctx.reportNodeProgress(0.12f + 0.34f *
                (static_cast<float>(iteration + 1) / settlePasses));
        }

        // Two conservative, terrain-aware relaxation passes remove remaining
        // one-cell deposits without smearing snow across cliff discontinuities.
        const float relaxationHeightLimit = (std::max)(
            metric.cellSize + safeMaxDepth * 0.35f, 0.05f);
        for (int relaxationPass = 0; relaxationPass < 2; ++relaxationPass) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t index = static_cast<size_t>(y) * w + x;
                    float weightedDepth = snowDepth[index] * 4.0f;
                    float weightSum = 4.0f;
                    for (int direction = 0; direction < 8; ++direction) {
                        const int nx = x + dx[direction], ny = y + dy[direction];
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        const size_t neighbor = static_cast<size_t>(ny) * w + nx;
                        if (std::abs(baseMeters[neighbor] - baseMeters[index]) > relaxationHeightLimit) continue;
                        const float weight = distance[direction] > 1.0f ? 0.70f : 1.0f;
                        weightedDepth += snowDepth[neighbor] * weight;
                        weightSum += weight;
                    }
                    const float average = weightedDepth / (std::max)(weightSum, 1e-6f);
                    nextDepth[index] = clampValue(
                        snowDepth[index] * 0.60f + average * 0.40f, 0.0f, safeMaxDepth);
                }
            }
            snowDepth.swap(nextDepth);
        }
        ctx.reportNodeProgress(0.50f);

        // The settle ping-pong buffer is no longer needed; recycle it for ice.
        std::fill(nextDepth.begin(), nextDepth.end(), 0.0f);
        std::vector<float>& iceDepth = nextDepth;
        std::vector<float> water(pixelCount, 0.0f);
        std::vector<float> nextWater(pixelCount, 0.0f);
        const float solarClearStrength = clampValue(solarMelt + meltAmount * 0.70f, 0.0f, 1.0f);
        for (size_t index = 0; index < pixelCount; ++index) {
            const float warmth = 1.0f - coldness[index];
            // Melt=1 must not be defeated completely by the altitude/coldness
            // gate. It supplies a small global thaw and boosts directional sun
            // clearing; sheltered cold valleys can still retain their mass.
            const float thermalMelt = meltAmount * (0.10f + warmth * 0.90f);
            const float directionalSolarMelt = solar[index] * solarClearStrength;
            const float meltFraction = clampValue(
                1.0f - (1.0f - thermalMelt) * (1.0f - directionalSolarMelt), 0.0f, 1.0f);
            const float melted = snowDepth[index] * meltFraction;
            // Sun/wind-exposed meltwater should run off instead of immediately
            // becoming an ice mask that Surface Composer renders as snow again.
            const float exposedRefreezeSuppression = clampValue(
                solar[index] + windExposure[index] * windStrength, 0.0f, 1.0f);
            const float refrozen = melted * coldness[index] * refreezeRate *
                (1.0f - exposedRefreezeSuppression);
            snowDepth[index] -= melted;
            const float compacted = snowDepth[index] * coldness[index] * refreezeRate * 0.12f;
            snowDepth[index] -= compacted;
            iceDepth[index] = refrozen + compacted;
            water[index] = melted - refrozen;
        }

        // Melt can carve sharp one-vertex holes into an otherwise thick snow
        // field, while convergent transport can leave a cell at Max Depth next
        // to nearly empty neighbors. Re-stabilize the remaining snow as a soft,
        // mass-conserving layer using an angle-of-repose transfer. Wetter snow
        // slides at a lower angle, so increasing Melt also strengthens downhill
        // creep instead of only subtracting thickness in place.
        const float authoredSlipRadians = clampValue(slipAngle, 5.0f, 80.0f) * 0.0174532925f;
        const float wetSlipDegrees = (std::min)(slipAngle, 27.0f);
        const float wetSlipRadians = wetSlipDegrees * 0.0174532925f;
        const float effectiveSlipTangent = std::tan(
            authoredSlipRadians + (wetSlipRadians - authoredSlipRadians) * clampValue(meltAmount, 0.0f, 1.0f));
        const int softSnowPasses = clampValue(2 + settlePasses / 3, 3, 10);
        for (int iteration = 0; iteration < softSnowPasses; ++iteration) {
            if (ctx.isCancelled()) return NodeSystem::PinValue{};
            std::fill(nextDepth.begin(), nextDepth.end(), 0.0f);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t index = static_cast<size_t>(y) * w + x;
                    const float sourceDepth = clampValue(snowDepth[index], 0.0f, safeMaxDepth);
                    const float sourceSurface = baseMeters[index] + sourceDepth;
                    float excess[8] = {};
                    float excessSum = 0.0f;
                    float maximumExcess = 0.0f;
                    for (int direction = 0; direction < 8; ++direction) {
                        const int nx = x + dx[direction], ny = y + dy[direction];
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        const size_t neighbor = static_cast<size_t>(ny) * w + nx;
                        const float allowedDrop = effectiveSlipTangent * metric.cellSize * distance[direction];
                        const float actualDrop = sourceSurface - (baseMeters[neighbor] + snowDepth[neighbor]);
                        excess[direction] = (std::max)(actualDrop - allowedDrop, 0.0f);
                        excessSum += excess[direction];
                        maximumExcess = (std::max)(maximumExcess, excess[direction]);
                    }

                    const float stableStep = (std::max)(effectiveSlipTangent * metric.cellSize, 0.01f);
                    const float instability = clampValue(maximumExcess / stableStep, 0.0f, 1.0f);
                    const float mobility = clampValue(
                        0.16f + meltAmount * 0.34f + sourceDepth / safeMaxDepth * 0.16f,
                        0.16f, 0.66f);
                    const float moved = excessSum > 1e-8f
                        ? sourceDepth * mobility * instability : 0.0f;
                    nextDepth[index] += sourceDepth - moved;
                    if (moved <= 0.0f) continue;
                    for (int direction = 0; direction < 8; ++direction) {
                        if (excess[direction] <= 0.0f) continue;
                        const int nx = x + dx[direction], ny = y + dy[direction];
                        const size_t neighbor = static_cast<size_t>(ny) * w + nx;
                        nextDepth[neighbor] += moved * (excess[direction] / excessSum);
                    }
                }
            }
            for (size_t index = 0; index < pixelCount; ++index) {
                const float value = nextDepth[index];
                snowDepth[index] = std::isfinite(value)
                    ? clampValue(value, 0.0f, safeMaxDepth) : 0.0f;
            }
            ctx.reportNodeProgress(0.52f + 0.20f *
                (static_cast<float>(iteration + 1) / softSnowPasses));
        }

        // The soft-snow pass runs after the main melt and may carry snow back
        // onto a face that the sun/wind had just swept clean. Re-apply only the
        // directional clearing component (not the global thermal melt) and add
        // that released mass to runoff. This preserves valley accumulation but
        // keeps exposed slopes genuinely bare.
        for (size_t index = 0; index < pixelCount; ++index) {
            const float directionalClear = clampValue(
                solar[index] * solarClearStrength + windExposure[index] * windStrength,
                0.0f, 1.0f);
            const float clearedSnow = snowDepth[index] * directionalClear;
            snowDepth[index] -= clearedSnow;
            water[index] += clearedSnow;
        }

        for (int iteration = 0; iteration < runoffPasses; ++iteration) {
            if (ctx.isCancelled()) return NodeSystem::PinValue{};
            std::fill(nextWater.begin(), nextWater.end(), 0.0f);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t index = static_cast<size_t>(y) * w + x;
                    const float sourceWater = (std::max)(water[index], 0.0f);
                    const float sourceSurface = baseMeters[index] + iceDepth[index] + sourceWater * 0.10f;
                    float weights[8] = {};
                    float weightSum = 0.0f;
                    for (int direction = 0; direction < 8; ++direction) {
                        const int nx = x + dx[direction], ny = y + dy[direction];
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        const size_t neighbor = static_cast<size_t>(ny) * w + nx;
                        const float drop = sourceSurface - (baseMeters[neighbor] + iceDepth[neighbor] + water[neighbor] * 0.10f);
                        if (drop <= 0.0f) continue;
                        weights[direction] = drop / distance[direction];
                        weightSum += weights[direction];
                    }

                    const float mobile = weightSum > 0.0f ? sourceWater * 0.72f : 0.0f;
                    nextWater[index] += sourceWater - mobile;
                    if (mobile <= 0.0f) continue;
                    for (int direction = 0; direction < 8; ++direction) {
                        if (weights[direction] <= 0.0f) continue;
                        const int nx = x + dx[direction], ny = y + dy[direction];
                        const size_t neighbor = static_cast<size_t>(ny) * w + nx;
                        const float incoming = mobile * (weights[direction] / weightSum);
                        const float neighborExposure = clampValue(
                            solar[neighbor] + windExposure[neighbor] * windStrength, 0.0f, 1.0f);
                        const float frozen = incoming * coldness[neighbor] * refreezeRate * 0.08f *
                            (1.0f - neighborExposure);
                        iceDepth[neighbor] += frozen;
                        nextWater[neighbor] += incoming - frozen;
                        runoffTrace[neighbor] += incoming;
                    }
                }
            }
            for (size_t index = 0; index < pixelCount; ++index) {
                const float ice = iceDepth[index];
                iceDepth[index] = std::isfinite(ice) ? clampValue(ice, 0.0f, safeMaxDepth) : 0.0f;
                if (!std::isfinite(nextWater[index])) nextWater[index] = 0.0f;
            }
            water.swap(nextWater);
            ctx.reportNodeProgress(0.72f + 0.23f *
                (static_cast<float>(iteration + 1) / runoffPasses));
        }

        float maxRunoff = 1e-6f;
        float maxAvalanche = 1e-6f;
        for (size_t index = 0; index < pixelCount; ++index) {
            maxRunoff = (std::max)(maxRunoff, runoffTrace[index] + water[index]);
            maxAvalanche = (std::max)(maxAvalanche, avalancheDeposit[index]);
        }
        // Material coverage must follow the surviving physical pack.  The old
        // upper threshold (35% of snowfall) made a heavily melted cell look
        // identical to a full-depth cell: with 1 m authored snowfall, anything
        // above only 0.35 m still wrote a fully saturated Snow splat channel.
        // Normalize ordinary cover against one authored snowfall instead;
        // deeper valley deposits may saturate, while partially melted slopes
        // now fade continuously and reveal the lower material layers.
        const float snowCoverageDepth = (std::max)(
            (std::min)(snowfallMeters, safeMaxDepth), 0.02f);
        const float visibleSnowThreshold = (std::max)(snowCoverageDepth * 0.02f, 0.002f);
        const float iceCoverageDepth = (std::max)(snowfallMeters * 0.12f, 0.02f);
        const float visibleIceThreshold = (std::max)(snowfallMeters * 0.03f, 0.003f);
        const float geometryScale = affectGeometry ? clampValue(geometryAmount, 0.0f, 2.0f) : 0.0f;
        const float coverageScale = clampValue(coverageAmount, 0.0f, 2.0f);

        // Geometry gets a final low-frequency snowpack surface independent of
        // material-mask sharpness. Thick displacement needs more than a fixed
        // three-pass blur: bound the added snow-depth gradient itself so no
        // adjacent vertices can form a near-vertical snow wall or needle.
        std::vector<float> geometryDepth(pixelCount, 0.0f);
        std::vector<float> nextGeometryDepth(pixelCount, 0.0f);
        for (size_t index = 0; index < pixelCount; ++index) {
            const float combined = snowDepth[index] + iceDepth[index] * 0.92f;
            geometryDepth[index] = std::isfinite(combined)
                ? clampValue(combined, 0.0f, safeMaxDepth) : 0.0f;
        }
        const float safeGeometryScale = (std::max)(geometryScale, 0.05f);
        const float maximumAddedDepthStep = (std::max)(
            metric.cellSize * std::tan(32.0f * 0.0174532925f) / safeGeometryScale, 0.01f);
        const int geometryRelaxPasses = geometryScale > 0.0f
            ? clampValue(static_cast<int>(std::ceil(
                safeMaxDepth / maximumAddedDepthStep)), 4, 16)
            : 0;
        for (int pass = 0; pass < geometryRelaxPasses; ++pass) {
            // Jacobi/ping-pong update: every worker owns one destination row
            // and reads only the immutable previous-pass buffer.
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t index = static_cast<size_t>(y) * w + x;
                    float weightedDepth = geometryDepth[index] * 4.0f;
                    float weightSum = 4.0f;
                    float minimumAllowed = 0.0f;
                    float maximumAllowed = safeMaxDepth;
                    for (int direction = 0; direction < 8; ++direction) {
                        const int nx = x + dx[direction], ny = y + dy[direction];
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        const size_t neighbor = static_cast<size_t>(ny) * w + nx;
                        const float allowedStep = maximumAddedDepthStep * distance[direction];
                        minimumAllowed = (std::max)(minimumAllowed,
                            geometryDepth[neighbor] - allowedStep);
                        maximumAllowed = (std::min)(maximumAllowed,
                            geometryDepth[neighbor] + allowedStep);
                        const float weight = distance[direction] > 1.0f ? 0.65f : 1.0f;
                        weightedDepth += geometryDepth[neighbor] * weight;
                        weightSum += weight;
                    }
                    const float localAverage = weightedDepth / (std::max)(weightSum, 1e-6f);
                    // Conflicting neighbor envelopes can occur during the first
                    // few iterations around a tall isolated deposit. Collapse
                    // them toward their midpoint, then converge normally.
                    if (minimumAllowed > maximumAllowed) {
                        const float midpoint = (minimumAllowed + maximumAllowed) * 0.5f;
                        minimumAllowed = midpoint;
                        maximumAllowed = midpoint;
                    }
                    const float smoothed = geometryDepth[index] * 0.45f + localAverage * 0.55f;
                    nextGeometryDepth[index] = clampValue(
                        smoothed, minimumAllowed, maximumAllowed);
                }
            }
            geometryDepth.swap(nextGeometryDepth);
        }
        for (size_t index = 0; index < pixelCount; ++index) {
            (*finalHeight.data)[index] = (*height.data)[index] +
                geometryDepth[index] / safeHeightScale * geometryScale;
            // A physically negligible residual should not become a visible
            // high-altitude material blanket merely because Coverage is high.
            (*snowMask.data)[index] = smoothstepLocal(
                visibleSnowThreshold, snowCoverageDepth, snowDepth[index]) * coverageScale;
            (*snowMask.data)[index] = clampValue((*snowMask.data)[index], 0.0f, 1.0f);
            // Thin refrozen runoff is not enough physical depth to justify a
            // full Snow/Ice material layer in Surface Composer.
            (*iceMask.data)[index] = smoothstepLocal(
                visibleIceThreshold, iceCoverageDepth, iceDepth[index]) * coverageScale;
            (*iceMask.data)[index] = clampValue((*iceMask.data)[index], 0.0f, 1.0f);
            (*meltwaterMask.data)[index] = clampValue((runoffTrace[index] + water[index]) / maxRunoff, 0.0f, 1.0f);
            (*avalancheMask.data)[index] = clampValue(avalancheDeposit[index] / maxAvalanche, 0.0f, 1.0f);
        }
        ctx.reportNodeProgress(0.99f);
        SCENE_LOG_INFO("[Snow Layer] Solve complete.");

        // This node has one expensive coupled solve. Populate every output in
        // the evaluation cache now so downstream material/height branches do
        // not run the climate simulation again for each requested pin.
        ctx.setCachedValue(id, 0, finalHeight);
        ctx.setCachedValue(id, 1, snowMask);
        ctx.setCachedValue(id, 2, iceMask);
        ctx.setCachedValue(id, 3, meltwaterMask);
        ctx.setCachedValue(id, 4, avalancheMask);
        switch (outputIndex) {
            case 0: return finalHeight;
            case 1: return snowMask;
            case 2: return iceMask;
            case 3: return meltwaterMask;
            case 4: return avalancheMask;
            default: return NodeSystem::PinValue{};
        }
    }

    void SnowClimateNode::drawContent() {
        ImGui::TextDisabled("Surface Height -> Height Output only");
        ImGui::TextDisabled("Snow / Ice / Meltwater -> matching Composer pins");
        if (ImGui::BeginCombo("Preset", getPresetName(preset))) {
            for (int i = 0; i <= static_cast<int>(SnowClimatePreset::GlacierValley); ++i) {
                const auto value = static_cast<SnowClimatePreset>(i);
                const bool selected = value == preset;
                if (ImGui::Selectable(getPresetName(value), selected)) {
                    preset = value;
                    if (value != SnowClimatePreset::Custom) applyPreset(value);
                    dirty = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        bool edited = false;
        ImGui::TextDisabled("Quick Controls");
        if (relativeSnowLine) {
            float snowLinePercent = snowLineFraction * 100.0f;
            if (ImGui::SliderFloat("Snow Line", &snowLinePercent, 0.0f, 100.0f, "%.0f%%")) {
                snowLineFraction = snowLinePercent * 0.01f;
                edited = true;
            }
        } else {
            edited |= ImGui::DragFloat("Snow Line", &snowLine, 0.25f, -1000.0f, 10000.0f, "%.2f m");
        }
        edited |= ImGui::DragFloat("Snowfall", &snowfallMeters, 0.05f, 0.0f, 1000.0f, "%.2f m");
        edited |= ImGui::DragFloat("Max Depth", &maxDepthMeters, 0.10f, 0.01f, 1000.0f, "%.2f m");
        edited |= ImGui::SliderFloat("Coverage", &coverageAmount, 0.0f, 2.0f);
        edited |= ImGui::Checkbox("Affect Geometry", &affectGeometry);
        if (affectGeometry) edited |= ImGui::SliderFloat("Geometry Amount", &geometryAmount, 0.0f, 2.0f);
        edited |= ImGui::DragFloat("Temperature", &baseTemperature, 0.25f, -80.0f, 60.0f, "%.1f C");
        edited |= ImGui::SliderFloat("Melt", &meltAmount, 0.0f, 1.0f);
        edited |= ImGui::SliderFloat("Wind", &windStrength, 0.0f, 1.0f);

        if (ImGui::TreeNode("Advanced Climate & Flow")) {
            edited |= ImGui::Checkbox("Relative Snow Line", &relativeSnowLine);
            if (relativeSnowLine) {
                float blendPercent = snowLineBlendFraction * 100.0f;
                if (ImGui::SliderFloat("Line Blend", &blendPercent, 0.1f, 100.0f, "%.1f%%")) {
                    snowLineBlendFraction = blendPercent * 0.01f;
                    edited = true;
                }
            } else {
                edited |= ImGui::DragFloat("Line Blend", &snowLineTransition, 0.1f, 0.01f, 1000.0f, "%.2f m");
            }
            edited |= ImGui::DragFloat("Lapse / 1000", &lapseRate, 0.1f, 0.0f, 20.0f, "%.1f C");
            edited |= ImGui::SliderFloat("Solar Melt", &solarMelt, 0.0f, 1.0f);
            edited |= ImGui::SliderFloat("Refreeze", &refreezeRate, 0.0f, 1.0f);
            edited |= ImGui::SliderFloat("Valley Capture", &valleyCapture, 0.0f, 2.0f);
            edited |= ImGui::SliderFloat("Snow Transport", &transportRate, 0.0f, 1.0f);
            edited |= ImGui::DragFloat("Slip Angle", &slipAngle, 0.5f, 5.0f, 80.0f, "%.1f deg");
            edited |= ImGui::DragInt("Settle Passes", &settleIterations, 1.0f, 1, 64);
            edited |= ImGui::DragInt("Runoff Passes", &waterIterations, 1.0f, 1, 64);
            edited |= ImGui::DragFloat("Wind Azimuth", &windAzimuth, 1.0f, 0.0f, 360.0f, "%.0f deg");
            edited |= ImGui::Checkbox("Use Scene Sun", &useSceneSun);
            if (!useSceneSun) {
                edited |= ImGui::DragFloat("Sun Azimuth", &sunAzimuth, 1.0f, 0.0f, 360.0f, "%.0f deg");
                edited |= ImGui::DragFloat("Sun Elevation", &sunElevation, 0.5f, -89.0f, 89.0f, "%.1f deg");
            }
            ImGui::TreePop();
        }
        if (edited) {
            preset = SnowClimatePreset::Custom;
            dirty = true;
        }
    }

    NodeSystem::PinValue ClimateNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto exposure = getHeightInput(1, ctx);
        if (!height.isValid()) {
            ctx.addError(id, "Climate requires a height input");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(exposure, height.width, height.height)) {
            ctx.addError(id, "Climate Exposure must match the height resolution");
            return NodeSystem::PinValue{};
        }

        const TerrainMetricScale metric = resolveTerrainMetricScale(ctx, height.width);
        auto result = createMaskOutput(height.width, height.height);
        const float transition = (std::max)(temperatureTransition, 0.1f);
        const float altitudeTransition = (std::max)(snowLineTransition, 0.01f);
        for (int y = 0; y < height.height; ++y) {
            for (int x = 0; x < height.width; ++x) {
                const size_t index = static_cast<size_t>(y) * height.width + x;
                const float physicalHeight = (*height.data)[index] * metric.heightScale;
                const float solarExposure = exposure.isValid()
                    ? clampValue((*exposure.data)[index], 0.0f, 1.0f)
                    : directionalExposure01(height, x, y, metric, sunAzimuth, sunElevation);
                const float temperature = seaLevelTemperature - lapseRate * (physicalHeight / 1000.0f) +
                    solarExposure * solarHeating;
                const float thermalCold = clampValue(
                    (freezePoint + transition - temperature) / (2.0f * transition), 0.0f, 1.0f);
                const float altitudeCold = smoothstepLocal(
                    snowLine - altitudeTransition, snowLine + altitudeTransition, physicalHeight);
                const float coldness = (std::max)(thermalCold, altitudeCold);
                (*result.data)[index] = outputIndex == 0
                    ? coldness
                    : clampValue(solarExposure * (1.0f - coldness * 0.65f), 0.0f, 1.0f);
            }
        }
        return result;
    }

    void ClimateNode::drawContent() {
        if (ImGui::DragFloat("Base Temperature", &seaLevelTemperature, 0.25f, -50.0f, 50.0f, "%.1f C")) dirty = true;
        if (ImGui::DragFloat("Lapse / 1000", &lapseRate, 0.1f, 0.0f, 20.0f, "%.1f C")) dirty = true;
        if (ImGui::DragFloat("Freeze Point", &freezePoint, 0.1f, -20.0f, 10.0f, "%.1f C")) dirty = true;
        if (ImGui::DragFloat("Thermal Blend", &temperatureTransition, 0.1f, 0.1f, 20.0f)) dirty = true;
        if (ImGui::DragFloat("Snow Line", &snowLine, 0.25f)) dirty = true;
        if (ImGui::DragFloat("Line Blend", &snowLineTransition, 0.1f, 0.01f, 1000.0f)) dirty = true;
        if (ImGui::SliderFloat("Solar Heating", &solarHeating, 0.0f, 20.0f)) dirty = true;
        if (ImGui::DragFloat("Sun Azimuth", &sunAzimuth, 1.0f, 0.0f, 360.0f, "%.0f deg")) dirty = true;
        if (ImGui::DragFloat("Sun Elevation", &sunElevation, 0.5f, 1.0f, 89.0f, "%.1f deg")) dirty = true;
    }

    NodeSystem::PinValue SnowfallNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto coldnessInput = getHeightInput(1, ctx);
        const auto exposure = getHeightInput(2, ctx);
        const auto mask = getHeightInput(3, ctx);
        if (!height.isValid() || height.width < 2 || height.height < 2) {
            ctx.addError(id, "Snowfall requires at least a 2x2 height input");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(coldnessInput, height.width, height.height) ||
            !sameImageExtent(exposure, height.width, height.height) ||
            !sameImageExtent(mask, height.width, height.height)) {
            ctx.addError(id, "Snowfall inputs must have matching resolutions");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const TerrainMetricScale metric = resolveTerrainMetricScale(ctx, w);
        auto result = createMaskOutput(w, h);
        for (int y = 0; y < h; ++y) {
            const float v = h > 1 ? static_cast<float>(y) / (h - 1) : 0.0f;
            for (int x = 0; x < w; ++x) {
                const float u = w > 1 ? static_cast<float>(x) / (w - 1) : 0.0f;
                const size_t index = static_cast<size_t>(y) * w + x;
                const float physicalHeight = (*height.data)[index] * metric.heightScale;
                const float coldness = coldnessInput.isValid()
                    ? clampValue((*coldnessInput.data)[index], 0.0f, 1.0f)
                    : smoothstepLocal(snowLine - snowLineTransition, snowLine + snowLineTransition, physicalHeight);
                const float sun = exposure.isValid() ? clampValue((*exposure.data)[index], 0.0f, 1.0f) : 0.5f;
                const float slope = surfaceSlope01(height, x, y, metric);
                const float slopeRetention = clampValue(1.0f - slope * (1.45f - slopeAdhesion), 0.0f, 1.0f);
                const float patch = interpolatedPatchNoise(u, v, patchScale, seed);
                const float scour = 1.0f - windScour * (0.35f + 0.65f * patch);
                const float authoredMask = mask.isValid() ? clampValue((*mask.data)[index], 0.0f, 1.0f) : 1.0f;
                const float snowMass = amount * coldness * slopeRetention *
                    (1.0f - sunLoss * sun) * scour * authoredMask;
                (*result.data)[index] = clampValue(snowMass, 0.0f, 1.0f);
            }
        }
        return result;
    }

    void SnowfallNode::drawContent() {
        if (ImGui::SliderFloat("Amount", &amount, 0.0f, 4.0f)) dirty = true;
        if (ImGui::DragFloat("Snow Line", &snowLine, 0.25f)) dirty = true;
        if (ImGui::DragFloat("Line Blend", &snowLineTransition, 0.1f, 0.01f, 1000.0f)) dirty = true;
        if (ImGui::SliderFloat("Slope Adhesion", &slopeAdhesion, 0.0f, 1.0f)) dirty = true;
        if (ImGui::SliderFloat("Sun Loss", &sunLoss, 0.0f, 1.0f)) dirty = true;
        if (ImGui::SliderFloat("Wind Scour", &windScour, 0.0f, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Patch Scale", &patchScale, 0.25f, 1.0f, 256.0f)) dirty = true;
        if (ImGui::DragInt("Seed", &seed)) dirty = true;
    }

    NodeSystem::PinValue SnowSettleNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto snow = getHeightInput(1, ctx);
        const auto mask = getHeightInput(2, ctx);
        if (!height.isValid() || !snow.isValid()) {
            ctx.addError(id, "Snow Settle requires Height and Snow inputs");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(snow, height.width, height.height) || !sameImageExtent(mask, height.width, height.height)) {
            ctx.addError(id, "Snow Settle inputs must have matching resolutions");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const size_t pixelCount = static_cast<size_t>(w) * h;
        const TerrainMetricScale metric = resolveTerrainMetricScale(ctx, w);
        std::vector<float> current = *snow.data;
        std::vector<float> next(pixelCount, 0.0f);
        const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        const float distance[8] = {1.41421356f, 1.0f, 1.41421356f, 1.0f, 1.0f, 1.41421356f, 1.0f, 1.41421356f};
        const float windRadians = windAzimuth * 0.0174532925f;
        const int windDx = clampValue(static_cast<int>(std::round(std::sin(windRadians))), -1, 1);
        const int windDy = clampValue(static_cast<int>(std::round(std::cos(windRadians))), -1, 1);

        for (int iteration = 0; iteration < clampValue(iterations, 1, 64); ++iteration) {
            std::fill(next.begin(), next.end(), 0.0f);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t index = static_cast<size_t>(y) * w + x;
                    float remaining = (std::max)(current[index], 0.0f);
                    const float moveMask = mask.isValid() ? clampValue((*mask.data)[index], 0.0f, 1.0f) : 1.0f;
                    int bestIndex = -1;
                    float bestSlopeDegrees = 0.0f;
                    const float centerSurface = ((*height.data)[index] + current[index] * depthScale) * metric.heightScale;
                    for (int direction = 0; direction < 8; ++direction) {
                        const int nx = x + dx[direction], ny = y + dy[direction];
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        const size_t neighbor = static_cast<size_t>(ny) * w + nx;
                        const float neighborSurface = ((*height.data)[neighbor] + current[neighbor] * depthScale) * metric.heightScale;
                        const float drop = centerSurface - neighborSurface;
                        if (drop <= 0.0f) continue;
                        const float slopeDegrees = std::atan(drop / ((std::max)(metric.cellSize * distance[direction], 1e-6f))) * 57.2957795f;
                        if (slopeDegrees > bestSlopeDegrees) { bestSlopeDegrees = slopeDegrees; bestIndex = static_cast<int>(neighbor); }
                    }

                    if (bestIndex >= 0 && bestSlopeDegrees > slipAngle) {
                        const float excess = clampValue((bestSlopeDegrees - slipAngle) / (90.0f - slipAngle), 0.0f, 1.0f);
                        const float moved = remaining * avalancheRate * excess * moveMask;
                        remaining -= moved;
                        next[bestIndex] += moved;
                    }

                    const int wx = x + windDx, wy = y + windDy;
                    if ((windDx != 0 || windDy != 0) && wx >= 0 && wx < w && wy >= 0 && wy < h) {
                        const float moved = remaining * windStrength * moveMask;
                        remaining -= moved;
                        next[static_cast<size_t>(wy) * w + wx] += moved;
                    }
                    next[index] += remaining;
                }
            }
            current.swap(next);
        }

        if (outputIndex == 0) {
            auto result = createMaskOutput(w, h);
            for (size_t index = 0; index < pixelCount; ++index)
                (*result.data)[index] = clampValue(current[index], 0.0f, 1.0f);
            return result;
        }
        auto result = createHeightOutput(w, h);
        for (size_t index = 0; index < pixelCount; ++index) {
            const float visibleDepth = current[index] * depthScale * (1.0f - compaction * 0.75f);
            (*result.data)[index] = clampValue((*height.data)[index] + visibleDepth, 0.0f, 1.0f);
        }
        return result;
    }

    void SnowSettleNode::drawContent() {
        if (ImGui::DragInt("Iterations", &iterations, 1, 1, 64)) dirty = true;
        if (ImGui::DragFloat("Slip Angle", &slipAngle, 0.5f, 5.0f, 80.0f, "%.1f deg")) dirty = true;
        if (ImGui::SliderFloat("Avalanche", &avalancheRate, 0.0f, 1.0f)) dirty = true;
        if (ImGui::SliderFloat("Compaction", &compaction, 0.0f, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Wind Azimuth", &windAzimuth, 1.0f, 0.0f, 360.0f, "%.0f deg")) dirty = true;
        if (ImGui::SliderFloat("Wind Transport", &windStrength, 0.0f, 0.5f)) dirty = true;
        if (ImGui::SliderFloat("Depth", &depthScale, 0.0f, 0.25f)) dirty = true;
    }

    NodeSystem::PinValue SnowMeltFreezeNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto snow = getHeightInput(0, ctx);
        const auto coldness = getHeightInput(1, ctx);
        const auto solarHeat = getHeightInput(2, ctx);
        const auto wetness = getHeightInput(3, ctx);
        if (!snow.isValid()) {
            ctx.addError(id, "Snow Melt / Freeze requires a Snow input");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(coldness, snow.width, snow.height) ||
            !sameImageExtent(solarHeat, snow.width, snow.height) ||
            !sameImageExtent(wetness, snow.width, snow.height)) {
            ctx.addError(id, "Snow Melt / Freeze inputs must have matching resolutions");
            return NodeSystem::PinValue{};
        }

        auto result = createMaskOutput(snow.width, snow.height);
        for (size_t index = 0; index < snow.pixelCount(); ++index) {
            const float snowMass = clampValue((*snow.data)[index], 0.0f, 1.0f);
            const float cold = coldness.isValid() ? clampValue((*coldness.data)[index], 0.0f, 1.0f) : 0.5f;
            const float solar = solarHeat.isValid() ? clampValue((*solarHeat.data)[index], 0.0f, 1.0f) : 0.5f;
            const float water = wetness.isValid() ? clampValue((*wetness.data)[index], 0.0f, 1.0f) : 0.0f;
            const float meltFraction = clampValue((1.0f - cold) * meltRate + solar * solarMelt, 0.0f, 1.0f);
            const float melted = snowMass * meltFraction;
            const float compactedIce = snowMass * cold * iceCompaction;
            const float refrozen = clampValue((melted + water) * cold * freezeRate, 0.0f, 1.0f);
            const float remainingSnow = clampValue(snowMass - melted - compactedIce, 0.0f, 1.0f);
            const float meltwater = clampValue(melted - refrozen, 0.0f, 1.0f);
            const float ice = clampValue(compactedIce + refrozen, 0.0f, 1.0f);
            (*result.data)[index] = outputIndex == 0 ? remainingSnow : (outputIndex == 1 ? meltwater : ice);
        }
        return result;
    }

    void SnowMeltFreezeNode::drawContent() {
        if (ImGui::SliderFloat("Melt Rate", &meltRate, 0.0f, 1.0f)) dirty = true;
        if (ImGui::SliderFloat("Solar Melt", &solarMelt, 0.0f, 1.0f)) dirty = true;
        if (ImGui::SliderFloat("Freeze Rate", &freezeRate, 0.0f, 1.0f)) dirty = true;
        if (ImGui::SliderFloat("Ice Compaction", &iceCompaction, 0.0f, 1.0f)) dirty = true;
    }

    NodeSystem::PinValue GlacierFlowNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getHeightInput(0, ctx);
        const auto ice = getHeightInput(1, ctx);
        const auto mask = getHeightInput(2, ctx);
        if (!height.isValid() || !ice.isValid()) {
            ctx.addError(id, "Glacier Flow requires Height and Ice inputs");
            return NodeSystem::PinValue{};
        }
        if (!sameImageExtent(ice, height.width, height.height) || !sameImageExtent(mask, height.width, height.height)) {
            ctx.addError(id, "Glacier Flow inputs must have matching resolutions");
            return NodeSystem::PinValue{};
        }

        const int w = height.width;
        const int h = height.height;
        const size_t pixelCount = static_cast<size_t>(w) * h;
        const TerrainMetricScale metric = resolveTerrainMetricScale(ctx, w);
        std::vector<float> current = *ice.data;
        std::vector<float> next(pixelCount, 0.0f);
        std::vector<float> heightDelta(outputIndex == 0 ? pixelCount : 0, 0.0f);
        const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        const float distance[8] = {1.41421356f, 1.0f, 1.41421356f, 1.0f, 1.0f, 1.41421356f, 1.0f, 1.41421356f};

        for (int iteration = 0; iteration < clampValue(iterations, 1, 64); ++iteration) {
            std::fill(next.begin(), next.end(), 0.0f);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t index = static_cast<size_t>(y) * w + x;
                    const float sourceIce = (std::max)(current[index], 0.0f);
                    const float sourceSurface = ((*height.data)[index] + sourceIce * iceDepthScale) * metric.heightScale;
                    int bestIndex = -1;
                    float bestGradient = 0.0f;
                    for (int direction = 0; direction < 8; ++direction) {
                        const int nx = x + dx[direction], ny = y + dy[direction];
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        const size_t neighbor = static_cast<size_t>(ny) * w + nx;
                        const float neighborSurface = ((*height.data)[neighbor] + current[neighbor] * iceDepthScale) * metric.heightScale;
                        const float gradient = (sourceSurface - neighborSurface) /
                            ((std::max)(metric.cellSize * distance[direction], 1e-6f));
                        if (gradient > bestGradient) { bestGradient = gradient; bestIndex = static_cast<int>(neighbor); }
                    }

                    const float authoredMask = mask.isValid() ? clampValue((*mask.data)[index], 0.0f, 1.0f) : 1.0f;
                    const float moved = bestIndex >= 0
                        ? sourceIce * flowStrength * clampValue(bestGradient, 0.0f, 1.0f) * authoredMask : 0.0f;
                    next[index] += sourceIce - moved;
                    if (bestIndex >= 0) next[bestIndex] += moved;
                    if (outputIndex == 0 && moved > 0.0f) {
                        heightDelta[index] -= moved * carvingStrength;
                        const float lowGradient = 1.0f - clampValue(bestGradient * 4.0f, 0.0f, 1.0f);
                        heightDelta[bestIndex] += moved * depositionStrength * lowGradient;
                    }
                }
            }
            current.swap(next);
        }

        if (outputIndex == 1) {
            auto result = createMaskOutput(w, h);
            for (size_t index = 0; index < pixelCount; ++index)
                (*result.data)[index] = clampValue(current[index], 0.0f, 1.0f);
            return result;
        }
        auto result = createHeightOutput(w, h);
        for (size_t index = 0; index < pixelCount; ++index)
            (*result.data)[index] = clampValue((*height.data)[index] + heightDelta[index], 0.0f, 1.0f);
        return result;
    }

    void GlacierFlowNode::drawContent() {
        if (ImGui::DragInt("Iterations", &iterations, 1, 1, 64)) dirty = true;
        if (ImGui::SliderFloat("Flow", &flowStrength, 0.0f, 0.75f)) dirty = true;
        if (ImGui::SliderFloat("Ice Depth", &iceDepthScale, 0.0f, 0.25f)) dirty = true;
        if (ImGui::SliderFloat("Carving", &carvingStrength, 0.0f, 0.05f)) dirty = true;
        if (ImGui::SliderFloat("Deposition", &depositionStrength, 0.0f, 0.05f)) dirty = true;
    }

    // ============================================================================
    // NODE REGISTRY SELF-REGISTRATION
    // ============================================================================
    // Registers every terrain node type under its existing getTypeId() string
    // (already unique, already used for JSON serialization — see e.g. line
    // ~224 `j["typeId"] = getTypeId();`) with the domain-agnostic NodeRegistry.
    // This does NOT replace addTerrainNode()'s switch below (still the primary
    // path for the Terrain tab's "Add Node" menu / NodeType enum) — it adds a
    // second, generic creation path (NodeRegistry::instance().create(typeId))
    // that a future cross-domain node editor (Faz 8) can use without needing
    // to know about NodeType or link against TerrainNodesV2 directly. One
    // static per type, all in this single TU, so registration runs exactly
    // once per type regardless of how many other files include TerrainNodesV2.h.
    namespace {
        NodeSystem::AutoRegisterNode<HeightmapInputNode>     reg_HeightmapInput("TerrainV2.HeightmapInput");
        NodeSystem::AutoRegisterNode<HardnessInputNode>      reg_HardnessInput("TerrainV2.HardnessInput");
        NodeSystem::AutoRegisterNode<NoiseGeneratorNode>     reg_NoiseGenerator("TerrainV2.NoiseGenerator");
        NodeSystem::AutoRegisterNode<HydraulicErosionNode>   reg_HydraulicErosion("TerrainV2.HydraulicErosion");
        NodeSystem::AutoRegisterNode<ThermalErosionNode>     reg_ThermalErosion("TerrainV2.ThermalErosion");
        NodeSystem::AutoRegisterNode<FluvialErosionNode>     reg_FluvialErosion("TerrainV2.FluvialErosion");
        NodeSystem::AutoRegisterNode<WindErosionNode>        reg_WindErosion("TerrainV2.WindErosion");
        NodeSystem::AutoRegisterNode<SedimentDepositionNode> reg_SedimentDeposition("TerrainV2.SedimentDeposition");
        NodeSystem::AutoRegisterNode<AlluvialFanNode>        reg_AlluvialFan("TerrainV2.AlluvialFan");
        NodeSystem::AutoRegisterNode<DeltaFormationNode>     reg_DeltaFormation("TerrainV2.DeltaFormation");
        NodeSystem::AutoRegisterNode<ErosionWizardNode>      reg_ErosionWizard("TerrainV2.ErosionWizard");
        NodeSystem::AutoRegisterNode<HeightOutputNode>       reg_HeightOutput("TerrainV2.HeightOutput");
        NodeSystem::AutoRegisterNode<SplatOutputNode>        reg_SplatOutput("TerrainV2.SplatOutput");
        NodeSystem::AutoRegisterNode<HardnessOutputNode>     reg_HardnessOutput("TerrainV2.HardnessOutput");
        NodeSystem::AutoRegisterNode<MathNode>               reg_Math("TerrainV2.Math");
        NodeSystem::AutoRegisterNode<BlendNode>              reg_Blend("TerrainV2.Blend");
        NodeSystem::AutoRegisterNode<ClampNode>              reg_Clamp("TerrainV2.Clamp");
        NodeSystem::AutoRegisterNode<InvertNode>             reg_Invert("TerrainV2.Invert");
        NodeSystem::AutoRegisterNode<SlopeMaskNode>          reg_SlopeMask("TerrainV2.SlopeMask");
        NodeSystem::AutoRegisterNode<HeightMaskNode>         reg_HeightMask("TerrainV2.HeightMask");
        NodeSystem::AutoRegisterNode<CurvatureMaskNode>      reg_CurvatureMask("TerrainV2.CurvatureMask");
        NodeSystem::AutoRegisterNode<FlowMaskNode>           reg_FlowMask("TerrainV2.FlowMask");
        NodeSystem::AutoRegisterNode<ExposureMaskNode>       reg_ExposureMask("TerrainV2.ExposureMask");
        NodeSystem::AutoRegisterNode<TerrainAnalysisNode>    reg_TerrainAnalysis("TerrainV2.TerrainAnalysis");
        NodeSystem::AutoRegisterNode<TerrainFieldsOutputNode> reg_TerrainFieldsOutput("TerrainV2.TerrainFieldsOutput");
        NodeSystem::AutoRegisterNode<BiomeComposerNode>      reg_BiomeComposer("TerrainV2.BiomeComposer");
        NodeSystem::AutoRegisterNode<FoliageLayerNode>       reg_FoliageLayer("TerrainV2.FoliageLayer");
        NodeSystem::AutoRegisterNode<FoliageSetNode>         reg_FoliageSet("TerrainV2.FoliageSet");
        NodeSystem::AutoRegisterNode<FoliageOutputNode>      reg_FoliageOutput("TerrainV2.FoliageOutput");
        NodeSystem::AutoRegisterNode<WatershedAnalysisNode>  reg_WatershedAnalysis("TerrainV2.WatershedAnalysis");
        NodeSystem::AutoRegisterNode<LakeBasinNode>          reg_LakeBasin("TerrainV2.LakeBasin");
        NodeSystem::AutoRegisterNode<LakeSurfaceOutputNode>  reg_LakeSurfaceOutput("TerrainV2.LakeSurfaceOutput");
        NodeSystem::AutoRegisterNode<RiverNetworkNode>       reg_RiverNetwork("TerrainV2.RiverNetwork");
        NodeSystem::AutoRegisterNode<RiverHydraulicsNode>    reg_RiverHydraulics("TerrainV2.RiverHydraulics");
        NodeSystem::AutoRegisterNode<RiverSplineOutputNode>  reg_RiverSplineOutput("TerrainV2.RiverSplineOutput");
        NodeSystem::AutoRegisterNode<RiverBedCarveNode>      reg_RiverBedCarve("TerrainV2.RiverBedCarve");
        NodeSystem::AutoRegisterNode<SmoothNode>             reg_Smooth("TerrainV2.Smooth");
        NodeSystem::AutoRegisterNode<NormalizeNode>          reg_Normalize("TerrainV2.Normalize");
        NodeSystem::AutoRegisterNode<TerraceNode>            reg_Terrace("TerrainV2.Terrace");
        NodeSystem::AutoRegisterNode<EdgeFalloffNode>        reg_EdgeFalloff("TerrainV2.EdgeFalloff");
        NodeSystem::AutoRegisterNode<MaskCombineNode>        reg_MaskCombine("TerrainV2.MaskCombine");
        NodeSystem::AutoRegisterNode<OverlayNode>            reg_Overlay("TerrainV2.Overlay");
        NodeSystem::AutoRegisterNode<ScreenNode>             reg_Screen("TerrainV2.Screen");
        NodeSystem::AutoRegisterNode<AutoSplatNode>          reg_AutoSplat("TerrainV2.AutoSplat");
        NodeSystem::AutoRegisterNode<MaskPaintNode>          reg_MaskPaint("TerrainV2.MaskPaint");
        NodeSystem::AutoRegisterNode<MaskImageNode>          reg_MaskImage("TerrainV2.MaskImage");
        NodeSystem::AutoRegisterNode<FaultNode>              reg_Fault("TerrainV2.Fault");
        NodeSystem::AutoRegisterNode<MesaNode>               reg_Mesa("TerrainV2.Mesa");
        NodeSystem::AutoRegisterNode<ShearNode>              reg_Shear("TerrainV2.Shear");
        NodeSystem::AutoRegisterNode<ResampleNode>           reg_Resample("TerrainV2.Resample");
        NodeSystem::AutoRegisterNode<ChannelExtractNode>     reg_ChannelExtract("TerrainV2.ChannelExtract");
        NodeSystem::AutoRegisterNode<SplatComposeNode>       reg_SplatCompose("TerrainV2.SplatCompose");
        NodeSystem::AutoRegisterNode<RemapNode>              reg_Remap("TerrainV2.Remap");
        NodeSystem::AutoRegisterNode<MaskAdjustNode>         reg_MaskAdjust("TerrainV2.MaskAdjust");
        NodeSystem::AutoRegisterNode<MaskMorphologyNode>     reg_MaskMorphology("TerrainV2.MaskMorphology");
        NodeSystem::AutoRegisterNode<WetnessMapNode>         reg_WetnessMap("TerrainV2.WetnessMap");
        NodeSystem::AutoRegisterNode<SoilDepthNode>          reg_SoilDepth("TerrainV2.SoilDepth");
        NodeSystem::AutoRegisterNode<LithologyNode>          reg_Lithology("TerrainV2.Lithology");
        NodeSystem::AutoRegisterNode<StrataNode>             reg_Strata("TerrainV2.Strata");
        NodeSystem::AutoRegisterNode<SurfaceComposerNode>    reg_SurfaceComposer("TerrainV2.SurfaceComposer");
        NodeSystem::AutoRegisterNode<SnowClimateNode>       reg_SnowClimate("TerrainV2.SnowClimate");
        NodeSystem::AutoRegisterNode<ClimateNode>            reg_Climate("TerrainV2.Climate");
        NodeSystem::AutoRegisterNode<SnowfallNode>           reg_Snowfall("TerrainV2.Snowfall");
        NodeSystem::AutoRegisterNode<SnowSettleNode>         reg_SnowSettle("TerrainV2.SnowSettle");
        NodeSystem::AutoRegisterNode<SnowMeltFreezeNode>     reg_SnowMeltFreeze("TerrainV2.SnowMeltFreeze");
        NodeSystem::AutoRegisterNode<GlacierFlowNode>        reg_GlacierFlow("TerrainV2.GlacierFlow");
    } // anonymous namespace

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
            case NodeType::HardnessOutput: node = addNode<HardnessOutputNode>(); break;
            case NodeType::HardnessInput: node = addNode<HardnessInputNode>(); break;
            case NodeType::Add:
            case NodeType::Subtract:
            case NodeType::Multiply:
                node = addNode<MathNode>();
                if (auto* math = dynamic_cast<MathNode*>(node)) {
                    math->terrainNodeType = type;
                    math->operation = (type == NodeType::Subtract) ? MathOp::Subtract
                                    : (type == NodeType::Multiply) ? MathOp::Multiply
                                                                  : MathOp::Add;
                    math->name = (type == NodeType::Subtract) ? "Subtract"
                               : (type == NodeType::Multiply) ? "Multiply"
                                                             : "Add";
                }
                break;
            case NodeType::Blend: node = addNode<BlendNode>(); break;
            case NodeType::Clamp: node = addNode<ClampNode>(); break;
            case NodeType::Invert: node = addNode<InvertNode>(); break;
            case NodeType::SlopeMask: node = addNode<SlopeMaskNode>(); break;
            case NodeType::HeightMask: node = addNode<HeightMaskNode>(); break;
            case NodeType::CurvatureMask: node = addNode<CurvatureMaskNode>(); break;
            case NodeType::FlowMask: node = addNode<FlowMaskNode>(); break;
            case NodeType::ExposureMask: node = addNode<ExposureMaskNode>(); break;
            case NodeType::TerrainAnalysis: node = addNode<TerrainAnalysisNode>(); break;
            case NodeType::TerrainFieldsOutput: node = addNode<TerrainFieldsOutputNode>(); break;
            case NodeType::BiomeComposer: node = addNode<BiomeComposerNode>(); break;
            case NodeType::FoliageLayer: node = addNode<FoliageLayerNode>(); break;
            case NodeType::FoliageSet: node = addNode<FoliageSetNode>(); break;
            case NodeType::FoliageOutput: node = addNode<FoliageOutputNode>(); break;
            case NodeType::WatershedAnalysis: node = addNode<WatershedAnalysisNode>(); break;
            case NodeType::LakeBasin: node = addNode<LakeBasinNode>(); break;
            case NodeType::LakeSurfaceOutput: node = addNode<LakeSurfaceOutputNode>(); break;
            case NodeType::RiverNetwork: node = addNode<RiverNetworkNode>(); break;
            case NodeType::RiverHydraulics: node = addNode<RiverHydraulicsNode>(); break;
            case NodeType::RiverSplineOutput: node = addNode<RiverSplineOutputNode>(); break;
            case NodeType::RiverBedCarve: node = addNode<RiverBedCarveNode>(); break;
            // NEW OPERATORS
            case NodeType::Smooth: node = addNode<SmoothNode>(); break;
            case NodeType::Normalize: node = addNode<NormalizeNode>(); break;
            case NodeType::Terrace: node = addNode<TerraceNode>(); break;
            case NodeType::EdgeFalloff: node = addNode<EdgeFalloffNode>(); break;
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
            case NodeType::Resample: node = addNode<ResampleNode>(); break;
            case NodeType::ChannelExtract: node = addNode<ChannelExtractNode>(); break;
            case NodeType::SplatCompose: node = addNode<SplatComposeNode>(); break;
            case NodeType::Remap: node = addNode<RemapNode>(); break;
            case NodeType::MaskAdjust: node = addNode<MaskAdjustNode>(); break;
            case NodeType::MaskMorphology: node = addNode<MaskMorphologyNode>(); break;
            case NodeType::WetnessMap: node = addNode<WetnessMapNode>(); break;
            case NodeType::SoilDepth: node = addNode<SoilDepthNode>(); break;
            case NodeType::Lithology: node = addNode<LithologyNode>(); break;
            case NodeType::Strata: node = addNode<StrataNode>(); break;
            case NodeType::SurfaceComposer: node = addNode<SurfaceComposerNode>(); break;
            case NodeType::SnowClimate: node = addNode<SnowClimateNode>(); break;
            case NodeType::Climate: node = addNode<ClimateNode>(); break;
            case NodeType::Snowfall: node = addNode<SnowfallNode>(); break;
            case NodeType::SnowSettle: node = addNode<SnowSettleNode>(); break;
            case NodeType::SnowMeltFreeze: node = addNode<SnowMeltFreezeNode>(); break;
            case NodeType::GlacierFlow: node = addNode<GlacierFlowNode>(); break;
        }

        if (node) {
            node->x = x;
            node->y = y;
        }

        return node;
    }

    // Phase A: pure CPU height-data compute. Safe to run on a worker thread —
    // touches only terrain->heightmap (a std::vector<float>) and the node graph's
    // own cached PinValues, never scene.world.objects / mesh_triangles / GPU
    // textures. `ctx` must already have its domain context set by the caller and
    // must be kept alive (and reused, unmodified) for the subsequent
    // evaluateTerrainAuxOutputs() call so cached upstream node results aren't
    // recomputed.
    namespace {
        void captureTerrainSceneSun(TerrainContext& terrainCtx, const SceneData& scene) {
            // SceneData::world is the renderable HittableList, not the
            // atmosphere World object. The renderer keeps Nishita synchronized
            // from the first directional light, so snapshot that same authored
            // source here. Light::direction is the ray travel direction; climate
            // exposure needs the opposite vector, pointing toward the sun.
            for (const auto& light : scene.lights) {
                if (!light || !light->visible || light->type() != LightType::Directional) continue;
                const float sx = -light->direction.x;
                const float sy = -light->direction.y;
                const float sz = -light->direction.z;
                const float length = std::sqrt(sx * sx + sy * sy + sz * sz);
                if (length <= 1e-6f) continue;
                terrainCtx.has_scene_sun = true;
                terrainCtx.scene_sun_x = sx / length;
                terrainCtx.scene_sun_y = sy / length;
                terrainCtx.scene_sun_z = sz / length;
                break;
            }
        }

        struct ScopedTerrainMeshUpdateDeferral {
            TerrainObject* terrain = nullptr;
            bool previous = false;

            explicit ScopedTerrainMeshUpdateDeferral(TerrainObject* value) : terrain(value) {
                previous = terrain->defer_mesh_updates;
                terrain->defer_mesh_updates = true;
            }

            ~ScopedTerrainMeshUpdateDeferral() {
                terrain->defer_mesh_updates = previous;
            }
        };
    }

    void TerrainNodeGraphV2::captureCommittedTerrainForPreview(TerrainObject* terrain) {
        if (!terrain || previewActive_) return;
        committedPreviewWidth_ = terrain->heightmap.width;
        committedPreviewHeight_ = terrain->heightmap.height;
        committedPreviewScaleXZ_ = terrain->heightmap.scale_xz;
        committedPreviewScaleY_ = terrain->heightmap.scale_y;
        committedPreviewHeightData_ = terrain->heightmap.data;
        committedPreviewFlowMap_ = terrain->flowMap;
        committedPreviewHardnessMap_ = terrain->hardnessMap;
        committedPreviewErosionMapRGBA_ = terrain->erosionMapRGBA;
        previewActive_ = true;
    }

    void TerrainNodeGraphV2::restoreCommittedTerrainData(TerrainObject* terrain) {
        if (!terrain || !previewActive_) return;
        terrain->heightmap.width = committedPreviewWidth_;
        terrain->heightmap.height = committedPreviewHeight_;
        terrain->heightmap.scale_xz = committedPreviewScaleXZ_;
        terrain->heightmap.scale_y = committedPreviewScaleY_;
        terrain->heightmap.data = committedPreviewHeightData_;
        terrain->flowMap = committedPreviewFlowMap_;
        terrain->hardnessMap = committedPreviewHardnessMap_;
        terrain->erosionMapRGBA = committedPreviewErosionMapRGBA_;
    }

    void TerrainNodeGraphV2::clearCommittedPreviewSnapshot() {
        previewActive_ = false;
        displayedPreviewNodeId_ = 0;
        committedPreviewWidth_ = 0;
        committedPreviewHeight_ = 0;
        committedPreviewHeightData_.clear();
        committedPreviewFlowMap_.clear();
        committedPreviewHardnessMap_.clear();
        committedPreviewErosionMapRGBA_.clear();
    }

    bool TerrainNodeGraphV2::evaluateTerrainHeightData(TerrainObject* terrain, NodeSystem::EvaluationContext& ctx,
                                                        uint32_t previewNodeId) {
        if (!terrain) return false;

        // TerrainManager erosion functions also serve direct/non-node callers and
        // normally refresh the mesh themselves.  A node chain must defer those
        // side effects: on first evaluation they otherwise create flatMesh on the
        // worker without adding it to SceneData, and on a resolution-changing
        // input reload they replace the scene-owned mesh with an orphan.  Restore
        // the caller's state on every return (and during exception unwinding).
        ScopedTerrainMeshUpdateDeferral deferMeshUpdates(terrain);

        // CRITICAL FIX: Preserve AND RESTORE scale values at the START
        // If terrain is in a "broken" state from a previous failed evaluation,
        // we need to fix it immediately. Don't just preserve - actively repair.
        float preserved_scale_xz = terrain->heightmap.scale_xz;
        float preserved_scale_y = terrain->heightmap.scale_y;

        // Force valid scale values immediately - repair broken terrain state
        if (preserved_scale_xz < 1.0f) preserved_scale_xz = 100.0f;
        if (preserved_scale_y < 0.1f) preserved_scale_y = 10.0f;

        // IMMEDIATELY apply valid scales to terrain (repair before evaluation)
        terrain->heightmap.scale_xz = preserved_scale_xz;
        terrain->heightmap.scale_y = preserved_scale_y;

        NodeSystem::Image2DData heightData;
        if (previewNodeId != 0) {
            NodeSystem::NodeBase* previewNode = getNode(previewNodeId);
            if (!previewNode) return false;

            // Output nodes have no output socket; preview their Height input.
            if (auto* heightOutput = dynamic_cast<HeightOutputNode*>(previewNode)) {
                ctx.beginNode(heightOutput->id);
                heightData = heightOutput->getHeightInput(0, ctx);
                ctx.endNode();
                heightOutput->dirty = false;
            } else {
                // Pulling a selected Height output naturally evaluates only its
                // upstream dependency DAG, which is the selected-node preview behavior.
                for (size_t outputIndex = 0; outputIndex < previewNode->outputs.size(); ++outputIndex) {
                    const auto& pin = previewNode->outputs[outputIndex];
                    if (pin.dataType != NodeSystem::DataType::Image2D ||
                        pin.imageSemantic != NodeSystem::ImageSemantic::Height) continue;
                    NodeSystem::PinValue value = previewNode->requestOutput(static_cast<int>(outputIndex), ctx);
                    if (const auto* image = std::get_if<NodeSystem::Image2DData>(&value)) heightData = *image;
                    break;
                }
            }
        } else {
            HeightOutputNode* heightOutputNode = nullptr;
            for (auto& node : nodes) {
                if (node->getTypeId() == "TerrainV2.HeightOutput") {
                    heightOutputNode = dynamic_cast<HeightOutputNode*>(node.get());
                    if (heightOutputNode) break;
                }
            }
            if (!heightOutputNode) return false;
            ctx.beginNode(heightOutputNode->id);
            heightData = heightOutputNode->getHeightInput(0, ctx);
            ctx.endNode();
            heightOutputNode->dirty = false;
        }

        // Check for errors in the evaluation chain
        if (ctx.hasErrors()) {
            for (const auto& err : ctx.getErrors()) {
                SCENE_LOG_ERROR("Terrain Graph Error (Node " + std::to_string(err.nodeId) + "): " + err.message);
            }
        }

        // CRITICAL CHECK: Verify input data integrity
        if (!heightData.isValid() || !heightData.data || heightData.width < 2 || heightData.height < 2) {
            // If data is invalid or too small (e.g. uninitialized), do not update terrain
            // This prevents "scrambled" artifacts during initialization
            return false;
        }

        // Compare against the dimensions captured before evaluation. Intermediate
        // nodes may already have assigned their input dimensions to TerrainObject,
        // which would hide a real topology change at this point.
        TerrainContext* evaluationTerrain = ctx.getDomainContext<TerrainContext>();
        bool resized = !evaluationTerrain ||
                       evaluationTerrain->width != heightData.width ||
                       evaluationTerrain->height != heightData.height;
        lastEvaluateResized_.store(resized);
        if (resized) {
            terrain->heightmap.width = heightData.width;
            terrain->heightmap.height = heightData.height;
            terrain->heightmap.data.resize(heightData.width * heightData.height);
        }

        // Copy data directly to terrain
        terrain->heightmap.data = *heightData.data;

        const size_t pixelCount = static_cast<size_t>(heightData.width) * heightData.height;
        if (terrain->flowMap.size() != pixelCount) terrain->flowMap.clear();
        if (terrain->hardnessMap.size() != pixelCount) terrain->hardnessMap.clear();
        if (terrain->erosionMapRGBA.size() != pixelCount * 4) terrain->erosionMapRGBA.clear();

        // CRITICAL FIX: Restore scale values after node graph evaluation
        // This ensures terrain physical dimensions remain constant regardless of node chain
        terrain->heightmap.scale_xz = preserved_scale_xz;
        terrain->heightmap.scale_y = preserved_scale_y;

        ctx.setProgress(1.0f);

        return true;
    }

    // Phase B: MAIN THREAD ONLY. SplatOutputNode::compute() calls
    // terrain->splatMap->updateGPU() — a GPU texture upload.
    bool TerrainNodeGraphV2::evaluateTerrainAuxOutputs(
        TerrainObject* terrain, SceneData& scene, NodeSystem::EvaluationContext& ctx) {
        if (!terrain) return false;

        // An auxiliary output can pull an erosion node that is not part of the
        // height-output chain (or pull it first in synchronous evaluation). Keep
        // the same single-finalize rule used by evaluateTerrainHeightData().
        ScopedTerrainMeshUpdateDeferral deferMeshUpdates(terrain);

        std::vector<SplatOutputNode*> splatOutputNodes;
        std::vector<HardnessOutputNode*> hardnessOutputNodes;
        std::vector<TerrainFieldsOutputNode*> fieldOutputNodes;
        std::vector<FoliageOutputNode*> foliageOutputNodes;
        std::vector<RiverSplineOutputNode*> riverOutputNodes;
        std::vector<LakeBasinNode*> lakeBasinNodes;
        std::vector<LakeSurfaceOutputNode*> lakeSurfaceOutputNodes;
        const auto publicationAllowed = [this](const TerrainNodeBase* node) {
            if (!node || !node->publicationEnabled) return false;
            const NodeSystem::NodeGroup* group = node->groupId ? getGroup(node->groupId) : nullptr;
            return !group || group->publicationEnabled;
        };
        for (auto& node : nodes) {
            std::string typeId = node->getTypeId();
            if (typeId == "TerrainV2.SplatOutput") {
                if (auto* splat = dynamic_cast<SplatOutputNode*>(node.get())) {
                    splatOutputNodes.push_back(splat);
                }
            } else if (typeId == "TerrainV2.HardnessOutput") {
                if (auto* hardness = dynamic_cast<HardnessOutputNode*>(node.get())) {
                    hardnessOutputNodes.push_back(hardness);
                }
            } else if (typeId == "TerrainV2.TerrainFieldsOutput") {
                if (auto* fields = dynamic_cast<TerrainFieldsOutputNode*>(node.get())) {
                    fieldOutputNodes.push_back(fields);
                }
            } else if (typeId == "TerrainV2.FoliageOutput") {
                if (auto* foliage = dynamic_cast<FoliageOutputNode*>(node.get())) {
                    foliage->lastScatteredGroupIds.clear();
                    foliage->lastSpawnedInstanceCount = 0;
                    foliageOutputNodes.push_back(foliage);
                }
            } else if (typeId == "TerrainV2.LakeBasin") {
                if (auto* lakeBasin = dynamic_cast<LakeBasinNode*>(node.get())) {
                    lakeBasinNodes.push_back(lakeBasin);
                }
            } else if (typeId == "TerrainV2.LakeSurfaceOutput") {
                if (auto* lakeOutput = dynamic_cast<LakeSurfaceOutputNode*>(node.get())) {
                    lakeSurfaceOutputNodes.push_back(lakeOutput);
                }
            } else if (typeId == "TerrainV2.RiverSplineOutput") {
                if (auto* riverOutput = dynamic_cast<RiverSplineOutputNode*>(node.get())) {
                    riverOutputNodes.push_back(riverOutput);
                }
            }
        }

        // Rebuild the publication set once per evaluation. Individual output
        // nodes only add their connected fields, so multiple publishers can be
        // used without erasing one another and disconnected pins cannot leave
        // stale mesh attributes behind.
        terrain->analysisFields.clear();
        terrain->waterBodies.erase(
            std::remove_if(terrain->waterBodies.begin(), terrain->waterBodies.end(),
                [](const WaterBodyData& body) { return body.sourceNodeId >= 0; }),
            terrain->waterBodies.end());

        // Pull-based evaluation through the connected graph. Any node shared with
        // the height chain already evaluated in phase A is served from ctx's cache.
        for (auto* splatNode : splatOutputNodes) {
            if (!publicationAllowed(splatNode)) continue;
            ctx.beginNode(splatNode->id);
            splatNode->compute(0, ctx);
            ctx.endNode();
            splatNode->dirty = false;
        }
        for (auto* hardNode : hardnessOutputNodes) {
            if (!publicationAllowed(hardNode)) continue;
            ctx.beginNode(hardNode->id);
            hardNode->compute(0, ctx);
            ctx.endNode();
            hardNode->dirty = false;
        }
        for (auto* lakeNode : lakeBasinNodes) {
            if (!publicationAllowed(lakeNode)) continue;
            ctx.beginNode(lakeNode->id);
            lakeNode->compute(0, ctx);
            ctx.endNode();
            lakeNode->dirty = false;
            if (!ctx.isCancelled()) lakeNode->publishWaterBodies(terrain);
        }
        for (auto* fieldNode : fieldOutputNodes) {
            if (!publicationAllowed(fieldNode)) continue;
            ctx.beginNode(fieldNode->id);
            fieldNode->compute(0, ctx);
            ctx.endNode();
            fieldNode->dirty = false;
        }
        // Foliage recipes consume the named-field contract published above but
        // only update InstanceGroup authoring settings. They never own or rebuild
        // the runtime instance arrays during terrain evaluation.
        for (auto* foliageNode : foliageOutputNodes) {
            if (!publicationAllowed(foliageNode)) continue;
            ctx.beginNode(foliageNode->id);
            foliageNode->compute(0, ctx);
            ctx.endNode();
            foliageNode->dirty = false;
        }
        bool sceneTopologyChanged = false;
        for (auto* lakeOutput : lakeSurfaceOutputNodes) {
            const bool enabled = publicationAllowed(lakeOutput);
            lakeOutput->sourceLakeNodeId = -1;
            if (enabled && !lakeOutput->inputs.empty()) {
                for (const auto& link : links) {
                    if (link.endPinId != lakeOutput->inputs[0].id) continue;
                    if (auto* sourceLake = dynamic_cast<LakeBasinNode*>(getPinOwner(link.startPinId))) {
                        lakeOutput->sourceLakeNodeId = static_cast<int>(sourceLake->id);
                    }
                    break;
                }
            }
            if (enabled) {
                ctx.beginNode(lakeOutput->id);
                lakeOutput->compute(0, ctx);
                ctx.endNode();
                lakeOutput->dirty = false;
            } else {
                lakeOutput->pendingFields = {};
            }
            if (!ctx.isCancelled()) {
                // applyGeneratedLakes removes surfaces owned by this sink before
                // publishing replacements; calling it while disabled prevents
                // stale lake geometry from surviving an output toggle.
                sceneTopologyChanged |= lakeOutput->applyGeneratedLakes(scene, terrain);
            }
        }
        for (auto* riverNode : riverOutputNodes) {
            const bool enabled = publicationAllowed(riverNode);
            if (enabled) {
                ctx.beginNode(riverNode->id);
                riverNode->compute(0, ctx);
                ctx.endNode();
                riverNode->dirty = false;
            } else {
                riverNode->pendingPaths.clear();
            }
            if (!ctx.isCancelled()) {
                // The apply pass also removes previously owned river objects.
                sceneTopologyChanged |= riverNode->applyGeneratedRivers(scene, terrain);
            }
        }
        return sceneTopologyChanged;
    }

    TerrainNodeGraphV2::DirtyEvaluationImpact
    TerrainNodeGraphV2::classifyDirtyEvaluationImpact() {
        bool materialOnlySinkDirty = false;
        bool foliageOnlySinkDirty = false;
        const auto publicationAllowed = [this](const TerrainNodeBase* node) {
            if (!node || !node->publicationEnabled) return false;
            const NodeSystem::NodeGroup* group = node->groupId ? getGroup(node->groupId) : nullptr;
            return !group || group->publicationEnabled;
        };

        // markDirtyDownstream() has already propagated the edit to every sink it
        // can affect. Therefore sink dirtiness is the exact contract boundary;
        // node type guesses for Flow/Slope/etc. are unnecessary and unsafe.
        for (const auto& graphNode : nodes) {
            if (!graphNode || !graphNode->dirty) continue;
            const auto* terrainNode = dynamic_cast<const TerrainNodeBase*>(graphNode.get());
            if (!terrainNode || !publicationAllowed(terrainNode)) continue;
            const std::string type = terrainNode->getTypeId();
            if (type == "TerrainV2.HeightOutput" ||
                type == "TerrainV2.HardnessOutput" ||
                type == "TerrainV2.TerrainFieldsOutput" ||
                type == "TerrainV2.LakeBasin" ||
                type == "TerrainV2.LakeSurfaceOutput" ||
                type == "TerrainV2.RiverSplineOutput") {
                return DirtyEvaluationImpact::GeometryOrScene;
            }
            if (type == "TerrainV2.SplatOutput") materialOnlySinkDirty = true;
            if (type == "TerrainV2.FoliageOutput") foliageOnlySinkDirty = true;
        }
        if (materialOnlySinkDirty && foliageOnlySinkDirty) {
            return DirtyEvaluationImpact::GeometryOrScene;
        }
        return materialOnlySinkDirty ? DirtyEvaluationImpact::MaterialOnly
            : (foliageOnlySinkDirty ? DirtyEvaluationImpact::FoliageOnly
                                    : DirtyEvaluationImpact::None);
    }

    bool TerrainNodeGraphV2::evaluateDirtyMaterialOutputs(
        TerrainObject* terrain, SceneData& scene) {
        if (!terrain || isEvaluatingAsync() || previewActive_ || !cachedEvalContext_) return false;
        if (classifyDirtyEvaluationImpact() != DirtyEvaluationImpact::MaterialOnly) return false;

        if (!cachedTerrainCtx_) cachedTerrainCtx_ = std::make_unique<TerrainContext>(terrain);
        cachedTerrainCtx_->terrain = terrain;
        cachedTerrainCtx_->width = terrain->heightmap.width;
        cachedTerrainCtx_->height = terrain->heightmap.height;
        cachedTerrainCtx_->scale_xz = terrain->heightmap.scale_xz;
        cachedTerrainCtx_->scale_y = terrain->heightmap.scale_y;
        captureTerrainSceneSun(*cachedTerrainCtx_, scene);

        NodeSystem::EvaluationContext& ctx = *cachedEvalContext_;
        ctx.setDomainContext(cachedTerrainCtx_.get());
        ctx.clearErrors();
        ctx.setTotalNodes(static_cast<int>(nodeCount()));
        ScopedTerrainMeshUpdateDeferral deferMeshUpdates(terrain);

        bool evaluatedAny = false;
        for (const auto& graphNode : nodes) {
            auto* splat = dynamic_cast<SplatOutputNode*>(graphNode.get());
            if (!splat || !splat->dirty || !splat->publicationEnabled) continue;
            const NodeSystem::NodeGroup* group = splat->groupId ? getGroup(splat->groupId) : nullptr;
            if (group && !group->publicationEnabled) continue;

            ctx.beginNode(splat->id);
            splat->compute(0, ctx);
            ctx.endNode();
            evaluatedAny = true;
            if (!ctx.hasErrors()) splat->dirty = false;
        }

        if (ctx.hasErrors()) {
            for (const auto& error : ctx.getErrors()) {
                SCENE_LOG_ERROR("Terrain material evaluation failed (Node " +
                                std::to_string(error.nodeId) + "): " + error.message);
            }
        }
        return evaluatedAny;
    }

    bool TerrainNodeGraphV2::evaluateDirtyFoliageOutputs(TerrainObject* terrain) {
        if (!terrain || isEvaluatingAsync() || previewActive_ || !cachedEvalContext_) return false;
        if (classifyDirtyEvaluationImpact() != DirtyEvaluationImpact::FoliageOnly) return false;

        if (!cachedTerrainCtx_) cachedTerrainCtx_ = std::make_unique<TerrainContext>(terrain);
        cachedTerrainCtx_->terrain = terrain;
        cachedTerrainCtx_->width = terrain->heightmap.width;
        cachedTerrainCtx_->height = terrain->heightmap.height;
        cachedTerrainCtx_->scale_xz = terrain->heightmap.scale_xz;
        cachedTerrainCtx_->scale_y = terrain->heightmap.scale_y;
        NodeSystem::EvaluationContext& ctx = *cachedEvalContext_;
        ctx.setDomainContext(cachedTerrainCtx_.get());
        ctx.clearErrors();
        ctx.setTotalNodes(static_cast<int>(nodeCount()));
        bool evaluatedAny = false;
        for (const auto& graphNode : nodes) {
            auto* foliage = dynamic_cast<FoliageOutputNode*>(graphNode.get());
            if (!foliage || !foliage->dirty || !foliage->publicationEnabled) continue;
            const NodeSystem::NodeGroup* group = foliage->groupId ? getGroup(foliage->groupId) : nullptr;
            if (group && !group->publicationEnabled) continue;

            ctx.beginNode(foliage->id);
            foliage->compute(0, ctx);
            ctx.endNode();
            evaluatedAny = true;
            if (!ctx.hasErrors()) foliage->dirty = false;
        }
        if (ctx.hasErrors()) {
            for (const auto& error : ctx.getErrors()) {
                SCENE_LOG_ERROR("Terrain foliage evaluation failed (Node " +
                                std::to_string(error.nodeId) + "): " + error.message);
            }
        }
        return evaluatedAny;
    }

    std::vector<int> TerrainNodeGraphV2::getLastScatteredFoliageGroupIds() const {
        std::vector<int> result;
        for (const auto& node : nodes) {
            const auto* foliage = dynamic_cast<const FoliageOutputNode*>(node.get());
            if (!foliage) continue;
            for (int groupId : foliage->lastScatteredGroupIds) {
                if (std::find(result.begin(), result.end(), groupId) == result.end()) {
                    result.push_back(groupId);
                }
            }
        }
        return result;
    }

    // Phase C: MAIN THREAD ONLY. Touches scene.world.objects / mesh_triangles,
    // shared with the render/BVH thread.
    //
    // deferBackendSignal: when false (default — used by the synchronous
    // evaluateTerrain() wrapper for project load/deserialize), backend
    // rebuild flags are set exactly as before. When true (used by the async
    // "Evaluate" button path, pollEvaluateAsync()), the in-place-update branch
    // does NOT set the flags itself — the caller decides between a cheap
    // partial refit and the full-rebuild flags based on
    // lastFinalizeWasFullRebuild().
    void TerrainNodeGraphV2::finalizeTerrainMesh(SceneData& scene, TerrainObject* terrain, bool deferBackendSignal) {
        if (!terrain) return;
        if (terrain->heightmap.width < 2 || terrain->heightmap.height < 2) return;

        size_t expectedTriCount = static_cast<size_t>(terrain->heightmap.width - 1) *
                                   static_cast<size_t>(terrain->heightmap.height - 1) * 2ull;
        // Important: some intermediate terrain nodes mutate terrain->heightmap dimensions before the
        // final Height Output is pulled. In that case `resized` can be false even though the mesh still
        // has the old triangle topology. Rebuild when triangle count does not match the current height
        // grid, otherwise the terrain collapses into a thin / corrupted strip on first evaluate.
        bool topologyMismatch = !terrain->flatMesh || terrain->flatMesh->num_triangles() != expectedTriCount;
        bool fullRebuild = lastEvaluateResized_.load() || topologyMismatch;
        lastFinalizeWasFullRebuild_.store(fullRebuild);

        if (fullRebuild) {
            TerrainManager::getInstance().resizeSplatMap(terrain);
            TerrainManager::getInstance().rebuildTerrainMesh(scene, terrain);
        } else {
            TerrainManager::getInstance().updateTerrainMesh(terrain, /*signalRebuild=*/!deferBackendSignal);
        }
    }

    void TerrainNodeGraphV2::evaluateTerrain(TerrainObject* terrain, SceneData& scene) {
        if (!terrain) return;

        if (previewActive_) {
            restoreCommittedTerrainData(terrain);
            clearCommittedPreviewSnapshot();
        }

        TerrainContext tctx(terrain);
        captureTerrainSceneSun(tctx, scene);
        NodeSystem::EvaluationContext ctx(this);
        ctx.setDomainContext(&tctx);

        // CRITICAL FIX: Clear cache and mark all nodes dirty before evaluation
        // Without this, intermediate nodes (deformation nodes) may be skipped
        // because their cached values would be reused instead of recomputing
        ctx.clearCache();
        ctx.clearErrors();
        markAllDirty();
        ctx.setTotalNodes(static_cast<int>(nodeCount()));

        // Evaluate secondary outputs first (Splat, Hardness) — matches the
        // original synchronous order. Order doesn't affect correctness since any
        // node shared between chains is memoized in ctx regardless of which
        // chain visits it first; kept for behavioral parity with before.
        bool updated = evaluateTerrainHeightData(terrain, ctx);
        if (updated) {
            TerrainManager::getInstance().resizeSplatMap(terrain);
            evaluateTerrainAuxOutputs(terrain, scene, ctx);
            finalizeTerrainMesh(scene, terrain);
        }
    }

    void TerrainNodeGraphV2::evaluateTerrainAsync(TerrainObject* terrain, SceneData& scene) {
        if (!terrain) return;
        if (isEvaluating.exchange(true)) {
            // Already running — ignore re-entrant clicks.
            return;
        }

        if (previewActive_) {
            restoreCommittedTerrainData(terrain);
            clearCommittedPreviewSnapshot();
        }

        activeTerrainCtx_ = std::make_unique<TerrainContext>(terrain);
        captureTerrainSceneSun(*activeTerrainCtx_, scene);
        activeEvalContext = std::make_shared<NodeSystem::EvaluationContext>(this);
        activeEvalContext->setDomainContext(activeTerrainCtx_.get());
        activeEvalContext->clearCache();
        activeEvalContext->clearErrors();
        markAllDirty();
        activeEvalContext->setTotalNodes(static_cast<int>(nodeCount()));

        pendingFinalizeTerrain_ = terrain;
        pendingFinalizeScene_ = &scene;
        lastHeightDataUpdated_.store(false);
        pendingEvaluationIsPreview_ = false;
        pendingPreviewNodeId_ = 0;
        cachedEvalContext_.reset();
        cachedTerrainCtx_.reset();

        // Worker thread only ever touches terrain->heightmap and the node graph's
        // cached PinValues (see evaluateTerrainHeightData's doc comment) — no
        // scene/world/GPU access here.
        std::shared_ptr<NodeSystem::EvaluationContext> ctxForWorker = activeEvalContext;
        evalFuture_ = std::async(std::launch::async, [this, terrain, ctxForWorker]() {
            try {
                lastHeightDataUpdated_.store(evaluateTerrainHeightData(terrain, *ctxForWorker));
                isEvaluating.store(false);
            } catch (...) {
                isEvaluating.store(false);
                throw;
            }
        });
    }

    void TerrainNodeGraphV2::evaluateTerrainPreviewAsync(uint32_t nodeId, TerrainObject* terrain, SceneData& scene) {
        if (!terrain || nodeId == 0) return;
        NodeSystem::NodeBase* targetNode = getNode(nodeId);
        if (displayedPreviewNodeId_ == nodeId && previewActive_ &&
            targetNode && !targetNode->dirty) return;
        if (isEvaluating.exchange(true)) return;

        captureCommittedTerrainForPreview(terrain);
        restoreCommittedTerrainData(terrain);

        activeTerrainCtx_ = std::make_unique<TerrainContext>(terrain);
        captureTerrainSceneSun(*activeTerrainCtx_, scene);
        if (cachedEvalContext_) {
            activeEvalContext = std::move(cachedEvalContext_);
            cachedTerrainCtx_.reset();
        } else {
            activeEvalContext = std::make_shared<NodeSystem::EvaluationContext>(this);
        }
        activeEvalContext->setDomainContext(activeTerrainCtx_.get());
        activeEvalContext->clearErrors();
        // Keep cached PinValues: dirty dependencies recompute while unchanged
        // upstream nodes become zero-copy cache hits.
        activeEvalContext->setTotalNodes(static_cast<int>(nodeCount()));

        pendingFinalizeTerrain_ = terrain;
        pendingFinalizeScene_ = &scene;
        lastHeightDataUpdated_.store(false);
        pendingEvaluationIsPreview_ = true;
        pendingPreviewNodeId_ = nodeId;

        std::shared_ptr<NodeSystem::EvaluationContext> ctxForWorker = activeEvalContext;
        evalFuture_ = std::async(std::launch::async, [this, terrain, nodeId, ctxForWorker]() {
            try {
                lastHeightDataUpdated_.store(evaluateTerrainHeightData(terrain, *ctxForWorker, nodeId));
                isEvaluating.store(false);
            } catch (...) {
                ctxForWorker->markNodeFailed(nodeId);
                isEvaluating.store(false);
                throw;
            }
        });
    }

    bool TerrainNodeGraphV2::restoreTerrainPreview(TerrainObject* terrain, SceneData& scene) {
        if (!terrain || !previewActive_ || isEvaluating.load() || evalFuture_.valid()) return false;
        const int previewWidth = terrain->heightmap.width;
        const int previewHeight = terrain->heightmap.height;
        restoreCommittedTerrainData(terrain);
        lastEvaluateResized_.store(previewWidth != terrain->heightmap.width ||
                                   previewHeight != terrain->heightmap.height);
        finalizeTerrainMesh(scene, terrain, /*deferBackendSignal=*/true);
        clearCommittedPreviewSnapshot();
        return true;
    }

    bool TerrainNodeGraphV2::pollEvaluateAsync() {
        if (!evalFuture_.valid()) return false;
        if (evalFuture_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

        try {
            evalFuture_.get(); // join worker
        } catch (const std::exception& e) {
            SCENE_LOG_ERROR(std::string("Terrain node evaluation failed: ") + e.what());
            pendingFinalizeTerrain_ = nullptr;
            pendingFinalizeScene_ = nullptr;
            pendingEvaluationIsPreview_ = false;
            pendingPreviewNodeId_ = 0;
            lastHeightDataUpdated_.store(false);
            isEvaluating.store(false);
            cachedEvalContext_ = std::move(activeEvalContext);
            cachedTerrainCtx_ = std::move(activeTerrainCtx_);
            return false;
        } catch (...) {
            SCENE_LOG_ERROR("Terrain node evaluation failed with an unknown exception.");
            pendingFinalizeTerrain_ = nullptr;
            pendingFinalizeScene_ = nullptr;
            pendingEvaluationIsPreview_ = false;
            pendingPreviewNodeId_ = 0;
            lastHeightDataUpdated_.store(false);
            isEvaluating.store(false);
            cachedEvalContext_ = std::move(activeEvalContext);
            cachedTerrainCtx_ = std::move(activeTerrainCtx_);
            return false;
        }

        TerrainObject* terrain = pendingFinalizeTerrain_;
        SceneData* scene = pendingFinalizeScene_;
        pendingFinalizeTerrain_ = nullptr;
        pendingFinalizeScene_ = nullptr;

        const bool heightUpdated = lastHeightDataUpdated_.load();
        const bool wasPreview = pendingEvaluationIsPreview_;
        const uint32_t previewNode = pendingPreviewNodeId_;
        pendingEvaluationIsPreview_ = false;
        pendingPreviewNodeId_ = 0;
        bool auxiliaryTopologyChanged = false;
        if (heightUpdated && terrain && scene && activeEvalContext) {
            // Splat/hardness GPU upload + mesh/BVH rebuild — main thread only.
            if (!wasPreview) {
                TerrainManager::getInstance().resizeSplatMap(terrain);
                auxiliaryTopologyChanged =
                    evaluateTerrainAuxOutputs(terrain, *scene, *activeEvalContext);
            }
            if (terrain->heightmap.width >= 2 && terrain->heightmap.height >= 2) {
                // deferBackendSignal=true: let the toolbar (which has access to the
                // viewport/render backends) choose a cheap partial refit over a
                // full-scene rebuild when topology didn't change — see
                // lastFinalizeWasFullRebuild().
                finalizeTerrainMesh(*scene, terrain, /*deferBackendSignal=*/true);
                if (auxiliaryTopologyChanged) lastFinalizeWasFullRebuild_.store(true);
            }
            if (wasPreview) displayedPreviewNodeId_ = previewNode;
        }

        cachedEvalContext_ = std::move(activeEvalContext);
        cachedTerrainCtx_ = std::move(activeTerrainCtx_);
        if (cachedEvalContext_) {
            constexpr size_t kPreviewCacheBudgetBytes = 512ull * 1024ull * 1024ull;
            cachedEvalContext_->enforceImageCacheBudget(
                kPreviewCacheBudgetBytes, wasPreview ? previewNode : 0u);
        }
        return heightUpdated;
    }
    
    
    void TerrainNodeGraphV2::createDefaultGraph(TerrainObject* terrain) {
        if (previewActive_) restoreCommittedTerrainData(terrain);
        clearCommittedPreviewSnapshot();
        cachedEvalContext_.reset();
        cachedTerrainCtx_.reset();
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

    namespace {
        void ensureTerrainLayerGroup(TerrainNodeGraphV2& graph, const char* name, ImU32 color,
                                     std::initializer_list<NodeSystem::NodeBase*> members) {
            NodeSystem::NodeGroup* group = nullptr;
            for (auto& candidate : graph.groups) {
                if (candidate.name == name) { group = &candidate; break; }
            }
            if (!group) {
                group = graph.getGroup(graph.createGroup(name, ImVec2(0, 0), ImVec2(240, 160)));
            }
            if (!group) return;
            group->color = color;
            group->comment = "Auto-managed terrain layer";
            for (NodeSystem::NodeBase* member : members) {
                if (!member) continue;
                graph.removeNodeFromGroups(member->id);
                graph.addNodeToGroup(member->id, group->id);
            }

            // A named layer can receive members from several presets. Refit against ALL
            // current members so a later River/Snow/Biome setup cannot leave the old hard-
            // coded frame hundreds of pixels away from its nodes.
            float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
            for (uint32_t nodeId : group->nodeIds) {
                NodeSystem::NodeBase* node = graph.getNode(nodeId);
                if (!node) continue;
                const float customWidth = node->getCustomWidth();
                const float width = node->uiWidth > 0.0f ? node->uiWidth
                    : (customWidth > 0.0f ? customWidth : 180.0f);
                int inputCount = 0, outputCount = 0;
                for (const auto& pin : node->inputs) if (!pin.hidden) ++inputCount;
                for (const auto& pin : node->outputs) if (!pin.hidden) ++outputCount;
                const float height = node->collapsed ? 34.0f
                    : 54.0f + 22.0f * static_cast<float>(std::max(inputCount, outputCount));
                minX = std::min(minX, node->x); minY = std::min(minY, node->y);
                maxX = std::max(maxX, node->x + width); maxY = std::max(maxY, node->y + height);
            }
            if (minX != FLT_MAX) {
                group->position = ImVec2(minX - 24.0f, minY - 44.0f);
                group->size = ImVec2(maxX - minX + 48.0f, maxY - minY + 68.0f);
            }
        }

        void layoutAutoManagedTerrainLayers(TerrainNodeGraphV2& graph) {
            static constexpr const char* kLayerOrder[] = {
                "01 Analysis", "02 Biome", "03 River", "04 Lakes", "05 Snow", "06 Outputs"
            };
            auto estimatedNodeSize = [](const NodeSystem::NodeBase& node) {
                int inputs = 0, outputs = 0;
                for (const auto& pin : node.inputs) if (!pin.hidden) ++inputs;
                for (const auto& pin : node.outputs) if (!pin.hidden) ++outputs;
                const float custom = node.getCustomWidth();
                const float width = node.uiWidth > 0.0f ? node.uiWidth
                    : (custom > 0.0f ? custom : 180.0f);
                const float height = node.collapsed ? 34.0f
                    : 54.0f + 22.0f * static_cast<float>(std::max(inputs, outputs));
                return ImVec2(width, height);
            };

            // Put generated stages after the authored/base chain, never on top of it.
            float stageX = 40.0f;
            float stageY = FLT_MAX;
            for (const auto& node : graph.nodes) {
                const NodeSystem::NodeGroup* group = node->groupId ? graph.getGroup(node->groupId) : nullptr;
                if (group && group->comment == "Auto-managed terrain layer") continue;
                stageX = std::max(stageX, node->x + estimatedNodeSize(*node).x + 120.0f);
                stageY = std::min(stageY, node->y);
            }
            if (stageY == FLT_MAX) stageY = 80.0f;

            constexpr float columnGap = 34.0f;
            constexpr float nodeGap = 28.0f;
            constexpr float layerGap = 90.0f;
            for (const char* layerName : kLayerOrder) {
                NodeSystem::NodeGroup* layer = nullptr;
                for (auto& group : graph.groups) {
                    if (group.name == layerName && group.comment == "Auto-managed terrain layer") {
                        layer = &group;
                        break;
                    }
                }
                if (!layer || layer->nodeIds.empty()) continue;

                float columnY[2] = {stageY + 48.0f, stageY + 48.0f};
                const int columnCount = layer->nodeIds.size() > 2 ? 2 : 1;
                float cellWidth = 180.0f;
                for (uint32_t nodeId : layer->nodeIds) {
                    if (NodeSystem::NodeBase* node = graph.getNode(nodeId)) {
                        cellWidth = std::max(cellWidth, estimatedNodeSize(*node).x);
                    }
                }
                for (uint32_t nodeId : layer->nodeIds) {
                    NodeSystem::NodeBase* node = graph.getNode(nodeId);
                    if (!node) continue;
                    const ImVec2 size = estimatedNodeSize(*node);
                    const int column = columnCount == 1 ? 0 : (columnY[0] <= columnY[1] ? 0 : 1);
                    const float x = stageX + 24.0f + column * (cellWidth + columnGap);
                    node->x = x;
                    node->y = columnY[column];
                    columnY[column] += size.y + nodeGap;
                }

                const float contentWidth = cellWidth * columnCount +
                    (columnCount == 2 ? columnGap : 0.0f);
                const float contentBottom = std::max(columnY[0], columnY[1]) - nodeGap;
                layer->position = ImVec2(stageX, stageY);
                layer->size = ImVec2(std::max(240.0f, contentWidth + 48.0f),
                                     std::max(150.0f, contentBottom - stageY + 24.0f));
                stageX += layer->size.x + layerGap;
            }
        }
    }

    bool TerrainNodeGraphV2::addSnowLayerSetup(float x, float y) {
        HeightOutputNode* heightOutput = nullptr;
        for (const auto& node : nodes) {
            if (auto* output = dynamic_cast<HeightOutputNode*>(node.get())) {
                heightOutput = output;
                break;
            }
        }
        if (!heightOutput || heightOutput->inputs.empty()) return false;

        const uint32_t heightInputPin = heightOutput->inputs[0].id;
        uint32_t heightSourcePin = 0;
        for (const auto& link : links) {
            if (link.endPinId == heightInputPin) {
                heightSourcePin = link.startPinId;
                break;
            }
        }
        if (heightSourcePin == 0) return false;

        SnowClimateNode* snow = dynamic_cast<SnowClimateNode*>(getPinOwner(heightSourcePin));
        uint32_t baseHeightPin = heightSourcePin;
        if (snow && !snow->outputs.empty() && snow->outputs[0].id == heightSourcePin) {
            // Setup already has a snow geometry terminal. Recover its immutable
            // base-height branch and only repair the material-side wiring.
            baseHeightPin = 0;
            if (!snow->inputs.empty()) {
                for (const auto& link : links) {
                    if (link.endPinId == snow->inputs[0].id) {
                        baseHeightPin = link.startPinId;
                        break;
                    }
                }
            }
            if (baseHeightPin == 0) return false;
        } else {
            snow = dynamic_cast<SnowClimateNode*>(addTerrainNode(NodeType::SnowClimate, x, y));
            if (!snow || snow->inputs.empty() || snow->outputs.size() < 4) return false;
            addLink(baseHeightPin, snow->inputs[0].id);
            addLink(snow->outputs[0].id, heightInputPin);
        }

        SurfaceComposerNode* composer = nullptr;
        SplatOutputNode* splatOutput = nullptr;
        TerrainAnalysisNode* analysis = nullptr;
        FlowMaskNode* flowMask = nullptr;
        SoilDepthNode* soilDepth = nullptr;
        for (const auto& node : nodes) {
            if (!splatOutput) splatOutput = dynamic_cast<SplatOutputNode*>(node.get());
            if (!analysis) analysis = dynamic_cast<TerrainAnalysisNode*>(node.get());
            if (!flowMask) flowMask = dynamic_cast<FlowMaskNode*>(node.get());
            if (!soilDepth) soilDepth = dynamic_cast<SoilDepthNode*>(node.get());
        }
        // Prefer the composer that already owns the active splat branch.
        if (splatOutput && !splatOutput->inputs.empty()) {
            for (const auto& link : links) {
                if (link.endPinId != splatOutput->inputs[0].id) continue;
                composer = dynamic_cast<SurfaceComposerNode*>(getPinOwner(link.startPinId));
                if (composer) break;
            }
        }
        if (!composer) {
            for (const auto& node : nodes) {
                if ((composer = dynamic_cast<SurfaceComposerNode*>(node.get()))) break;
            }
        }
        if (!composer) {
            composer = dynamic_cast<SurfaceComposerNode*>(
                addTerrainNode(NodeType::SurfaceComposer, x + 300.0f, y + 220.0f));
        }
        if (!splatOutput) {
            splatOutput = dynamic_cast<SplatOutputNode*>(
                addTerrainNode(NodeType::SplatOutput, x + 560.0f, y + 220.0f));
        }
        if (!analysis) analysis = dynamic_cast<TerrainAnalysisNode*>(
            addTerrainNode(NodeType::TerrainAnalysis, x - 300.0f, y + 200.0f));
        if (!flowMask) flowMask = dynamic_cast<FlowMaskNode*>(
            addTerrainNode(NodeType::FlowMask, x - 300.0f, y + 500.0f));
        if (!soilDepth) soilDepth = dynamic_cast<SoilDepthNode*>(
            addTerrainNode(NodeType::SoilDepth, x - 40.0f, y + 500.0f));
        if (!composer || !splatOutput || composer->inputs.size() < 10 || composer->outputs.size() < 2 ||
            splatOutput->inputs.empty() || !analysis || analysis->inputs.empty() || analysis->outputs.size() < 5 ||
            !flowMask || flowMask->inputs.empty() || flowMask->outputs.empty() ||
            !soilDepth || soilDepth->inputs.size() < 2 || soilDepth->outputs.empty()) return false;

        // Composer always sees the pre-snow surface. Explicit grass/rock/flow
        // links are separate inputs and remain untouched by these calls.
        addLink(baseHeightPin, composer->inputs[0].id);
        addLink(baseHeightPin, analysis->inputs[0].id);
        addLink(baseHeightPin, flowMask->inputs[0].id);
        addLink(baseHeightPin, soilDepth->inputs[0].id);
        addLink(flowMask->outputs[0].id, soilDepth->inputs[1].id);
        addLink(soilDepth->outputs[0].id, composer->inputs[1].id);
        addLink(flowMask->outputs[0].id, composer->inputs[2].id);
        addLink(analysis->outputs[4].id, composer->inputs[3].id);
        addLink(snow->outputs[1].id, composer->inputs[5].id);
        addLink(snow->outputs[2].id, composer->inputs[6].id);
        addLink(snow->outputs[3].id, composer->inputs[7].id);
        addLink(composer->outputs[1].id, splatOutput->inputs[0].id);
        ensureTerrainLayerGroup(*this, "01 Analysis", IM_COL32(70, 105, 135, 90),
                                {analysis, flowMask, soilDepth});
        ensureTerrainLayerGroup(*this, "05 Snow", IM_COL32(100, 165, 205, 90), {snow});
        ensureTerrainLayerGroup(*this, "06 Outputs", IM_COL32(105, 90, 130, 90),
                                {heightOutput, composer, splatOutput});
        layoutAutoManagedTerrainLayers(*this);
        markAllDirty();
        return true;
    }

    bool TerrainNodeGraphV2::addBiomeFieldsSetup(BiomeClimatePreset biomePreset, float x, float y) {
        HeightOutputNode* heightOutput = nullptr;
        for (const auto& node : nodes) {
            if (auto* output = dynamic_cast<HeightOutputNode*>(node.get())) {
                heightOutput = output;
                break;
            }
        }
        if (!heightOutput || heightOutput->inputs.empty()) return false;

        auto sourcePinForInput = [&](uint32_t inputPin) -> uint32_t {
            for (const auto& link : links) {
                if (link.endPinId == inputPin) return link.startPinId;
            }
            return 0;
        };

        uint32_t baseHeightPin = sourcePinForInput(heightOutput->inputs[0].id);
        if (baseHeightPin == 0) return false;

        // Biomes describe the underlying landform. If snow owns the terminal
        // geometry branch, classify from its pre-snow height input instead.
        if (auto* snow = dynamic_cast<SnowClimateNode*>(getPinOwner(baseHeightPin))) {
            if (!snow->outputs.empty() && snow->outputs[0].id == baseHeightPin && !snow->inputs.empty()) {
                const uint32_t preSnowHeightPin = sourcePinForInput(snow->inputs[0].id);
                if (preSnowHeightPin != 0) baseHeightPin = preSnowHeightPin;
            }
        }

        BiomeComposerNode* biome = nullptr;
        TerrainFieldsOutputNode* fields = nullptr;
        for (const auto& node : nodes) {
            if (!biome) biome = dynamic_cast<BiomeComposerNode*>(node.get());
            if (!fields) fields = dynamic_cast<TerrainFieldsOutputNode*>(node.get());
        }

        TerrainAnalysisNode* analysis = nullptr;
        ExposureMaskNode* exposure = nullptr;
        FlowMaskNode* flowMask = nullptr;
        SoilDepthNode* soilDepth = nullptr;
        SurfaceComposerNode* surfaceComposer = nullptr;
        SplatOutputNode* splatOutput = nullptr;
        for (const auto& node : nodes) {
            if (!flowMask) flowMask = dynamic_cast<FlowMaskNode*>(node.get());
            if (!soilDepth) soilDepth = dynamic_cast<SoilDepthNode*>(node.get());
            if (!surfaceComposer) surfaceComposer = dynamic_cast<SurfaceComposerNode*>(node.get());
            if (!splatOutput) splatOutput = dynamic_cast<SplatOutputNode*>(node.get());
        }
        if (biome && biome->inputs.size() >= 5) {
            analysis = dynamic_cast<TerrainAnalysisNode*>(getPinOwner(sourcePinForInput(biome->inputs[1].id)));
            exposure = dynamic_cast<ExposureMaskNode*>(getPinOwner(sourcePinForInput(biome->inputs[4].id)));
        }
        if (!analysis && fields && !fields->inputs.empty()) {
            analysis = dynamic_cast<TerrainAnalysisNode*>(getPinOwner(sourcePinForInput(fields->inputs[0].id)));
        }

        if (!analysis) analysis = dynamic_cast<TerrainAnalysisNode*>(
            addTerrainNode(NodeType::TerrainAnalysis, x, y));
        if (!exposure) exposure = dynamic_cast<ExposureMaskNode*>(
            addTerrainNode(NodeType::ExposureMask, x, y + 330.0f));
        if (!biome) biome = dynamic_cast<BiomeComposerNode*>(
            addTerrainNode(NodeType::BiomeComposer, x + 290.0f, y + 80.0f));
        if (!fields) fields = dynamic_cast<TerrainFieldsOutputNode*>(
            addTerrainNode(NodeType::TerrainFieldsOutput, x + 590.0f, y + 80.0f));
        if (!flowMask) flowMask = dynamic_cast<FlowMaskNode*>(
            addTerrainNode(NodeType::FlowMask, x, y + 500.0f));
        if (!soilDepth) soilDepth = dynamic_cast<SoilDepthNode*>(
            addTerrainNode(NodeType::SoilDepth, x + 270.0f, y + 500.0f));
        if (!surfaceComposer) surfaceComposer = dynamic_cast<SurfaceComposerNode*>(
            addTerrainNode(NodeType::SurfaceComposer, x + 590.0f, y + 430.0f));
        if (!splatOutput) splatOutput = dynamic_cast<SplatOutputNode*>(
            addTerrainNode(NodeType::SplatOutput, x + 850.0f, y + 430.0f));

        if (!analysis || !exposure || !biome || !fields ||
            analysis->inputs.empty() || analysis->outputs.size() < 5 ||
            exposure->inputs.empty() || exposure->outputs.empty() ||
            biome->inputs.size() < 5 || biome->outputs.size() < 4 ||
            fields->inputs.size() < 9 || !flowMask || flowMask->inputs.empty() || flowMask->outputs.empty() ||
            !soilDepth || soilDepth->inputs.size() < 2 || soilDepth->outputs.empty() ||
            !surfaceComposer || surfaceComposer->inputs.size() < 10 || surfaceComposer->outputs.size() < 2 ||
            !splatOutput || splatOutput->inputs.empty()) {
            return false;
        }

        biome->applyPreset(biomePreset);
        addLink(baseHeightPin, analysis->inputs[0].id);
        addLink(baseHeightPin, exposure->inputs[0].id);
        addLink(baseHeightPin, biome->inputs[0].id);
        addLink(analysis->outputs[0].id, biome->inputs[1].id); // slope
        addLink(analysis->outputs[3].id, biome->inputs[2].id); // valley
        addLink(analysis->outputs[4].id, biome->inputs[3].id); // wetness
        addLink(exposure->outputs[0].id, biome->inputs[4].id);
        addLink(baseHeightPin, flowMask->inputs[0].id);
        addLink(baseHeightPin, soilDepth->inputs[0].id);
        addLink(flowMask->outputs[0].id, soilDepth->inputs[1].id);
        addLink(baseHeightPin, surfaceComposer->inputs[0].id);
        addLink(soilDepth->outputs[0].id, surfaceComposer->inputs[1].id);
        addLink(flowMask->outputs[0].id, surfaceComposer->inputs[2].id);
        addLink(analysis->outputs[4].id, surfaceComposer->inputs[3].id);
        addLink(biome->outputs[1].id, surfaceComposer->inputs[8].id);
        addLink(biome->outputs[2].id, surfaceComposer->inputs[9].id);
        addLink(surfaceComposer->outputs[1].id, splatOutput->inputs[0].id);

        for (int i = 0; i < 5; ++i) {
            addLink(analysis->outputs[static_cast<size_t>(i)].id,
                    fields->inputs[static_cast<size_t>(i)].id);
        }
        for (int i = 0; i < 4; ++i) {
            addLink(biome->outputs[static_cast<size_t>(i)].id,
                    fields->inputs[static_cast<size_t>(i + 5)].id);
        }
        ensureTerrainLayerGroup(*this, "01 Analysis", IM_COL32(70, 105, 135, 90),
                                {analysis, exposure, flowMask, soilDepth});
        ensureTerrainLayerGroup(*this, "02 Biome", IM_COL32(65, 135, 80, 90),
                                {biome});
        ensureTerrainLayerGroup(*this, "06 Outputs", IM_COL32(105, 90, 130, 90),
                                {heightOutput, fields, surfaceComposer, splatOutput});
        layoutAutoManagedTerrainLayers(*this);
        markAllDirty();
        return true;
    }

    bool TerrainNodeGraphV2::addBiomeFoliageSetup(float x, float y) {
        bool hasBiomeComposer = false;
        bool hasFieldOutput = false;
        for (const auto& node : nodes) {
            hasBiomeComposer |= dynamic_cast<BiomeComposerNode*>(node.get()) != nullptr;
            hasFieldOutput |= dynamic_cast<TerrainFieldsOutputNode*>(node.get()) != nullptr;
        }
        if ((!hasBiomeComposer || !hasFieldOutput) &&
            !addBiomeFieldsSetup(BiomeClimatePreset::TemperateMixed, x - 600.0f, y)) {
            return false;
        }

        BiomeComposerNode* activeBiomeComposer = nullptr;
        for (const auto& node : nodes) {
            if (auto* composer = dynamic_cast<BiomeComposerNode*>(node.get())) {
                activeBiomeComposer = composer;
                break;
            }
        }

        FoliageSetNode* foliageSet = nullptr;
        FoliageOutputNode* foliageOutput = nullptr;
        std::array<FoliageLayerNode*, 4> foliageLayers = {};
        const std::array<const char*, 4> fields = {
            "biome.forest", "biome.grass", "biome.rock", "biome.alpine"
        };
        const std::array<const char*, 4> labels = {
            "Forest", "Grass", "Rock", "Alpine"
        };

        for (const auto& node : nodes) {
            if (!foliageSet) foliageSet = dynamic_cast<FoliageSetNode*>(node.get());
            if (!foliageOutput) foliageOutput = dynamic_cast<FoliageOutputNode*>(node.get());
            if (auto* layer = dynamic_cast<FoliageLayerNode*>(node.get())) {
                for (size_t i = 0; i < fields.size(); ++i) {
                    if (!foliageLayers[i] && layer->densityField == fields[i]) {
                        foliageLayers[i] = layer;
                    }
                }
            }
        }

        auto findExistingGroup = [](const std::array<const char*, 4>& tokens)
            -> std::pair<int, std::string> {
            for (const auto& group : InstanceManager::getInstance().getGroups()) {
                if (group.transient) continue;
                std::string lower = group.name;
                std::transform(lower.begin(), lower.end(), lower.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                for (const char* token : tokens) {
                    if (token[0] != '\0' && lower.find(token) != std::string::npos) {
                        return {group.id, group.name};
                    }
                }
            }
            return {-1, {}};
        };
        const std::array<std::array<const char*, 4>, 4> groupTokens = {{
            {{"forest", "tree", "agac", "orman"}},
            {{"grass", "meadow", "cim", "ot"}},
            {{"rock", "stone", "kaya", "tas"}},
            {{"alpine", "snow", "dag", "mountain"}}
        }};

        for (size_t i = 0; i < foliageLayers.size(); ++i) {
            if (!foliageLayers[i]) {
                foliageLayers[i] = dynamic_cast<FoliageLayerNode*>(
                    addTerrainNode(NodeType::FoliageLayer,
                                   x, y + static_cast<float>(i) * 265.0f));
            }
            if (!foliageLayers[i]) return false;
            foliageLayers[i]->name = std::string(labels[i]) + " Foliage";
            if (activeBiomeComposer && foliageLayers[i]->assetBiome == "Auto") {
                foliageLayers[i]->assetBiome = BiomeComposerNode::getPresetName(activeBiomeComposer->preset);
            }
            if (foliageLayers[i]->instanceGroupId < 0 &&
                foliageLayers[i]->instanceGroupName.empty()) {
                const auto match = findExistingGroup(groupTokens[i]);
                if (const InstanceGroup* group = InstanceManager::getInstance().getGroup(match.first)) {
                    captureFoliageGroupSettings(*foliageLayers[i], *group);
                } else {
                    foliageLayers[i]->instanceGroupId = match.first;
                    foliageLayers[i]->instanceGroupName = match.second;
                    foliageLayers[i]->useAssetLibrary = true;
                }
            }
            foliageLayers[i]->densityField = fields[i];
        }
        if (!foliageSet) foliageSet = dynamic_cast<FoliageSetNode*>(
            addTerrainNode(NodeType::FoliageSet, x + 330.0f, y + 260.0f));
        if (!foliageOutput) foliageOutput = dynamic_cast<FoliageOutputNode*>(
            addTerrainNode(NodeType::FoliageOutput, x + 620.0f, y + 260.0f));
        if (!foliageSet || !foliageOutput || foliageSet->inputs.size() < foliageLayers.size() ||
            foliageSet->outputs.empty() || foliageOutput->inputs.empty()) return false;

        for (size_t i = 0; i < foliageLayers.size(); ++i) {
            addLink(foliageLayers[i]->outputs[0].id, foliageSet->inputs[i].id);
        }
        addLink(foliageSet->outputs[0].id, foliageOutput->inputs[0].id);
        ensureTerrainLayerGroup(*this, "05 Foliage", IM_COL32(55, 125, 70, 90),
                                {foliageLayers[0], foliageLayers[1], foliageLayers[2],
                                 foliageLayers[3], foliageSet});
        ensureTerrainLayerGroup(*this, "06 Outputs", IM_COL32(105, 90, 130, 90),
                                {foliageOutput});
        layoutAutoManagedTerrainLayers(*this);
        markAllDirty();
        return true;
    }

    bool TerrainNodeGraphV2::addRiverNetworkSetup(float x, float y) {
        HeightOutputNode* heightOutput = nullptr;
        WatershedAnalysisNode* watershed = nullptr;
        LakeBasinNode* lakeBasin = nullptr;
        LakeSurfaceOutputNode* lakeSurfaceOutput = nullptr;
        RiverNetworkNode* network = nullptr;
        RiverHydraulicsNode* hydraulics = nullptr;
        RiverSplineOutputNode* splineOutput = nullptr;
        RiverBedCarveNode* carve = nullptr;
        TerrainFieldsOutputNode* fields = nullptr;
        TerrainAnalysisNode* terrainAnalysis = nullptr;
        FlowMaskNode* flowMask = nullptr;
        SoilDepthNode* soilDepth = nullptr;
        SurfaceComposerNode* surfaceComposer = nullptr;
        for (const auto& node : nodes) {
            if (!heightOutput) heightOutput = dynamic_cast<HeightOutputNode*>(node.get());
            if (!watershed) watershed = dynamic_cast<WatershedAnalysisNode*>(node.get());
            if (!lakeBasin) lakeBasin = dynamic_cast<LakeBasinNode*>(node.get());
            if (!lakeSurfaceOutput) lakeSurfaceOutput = dynamic_cast<LakeSurfaceOutputNode*>(node.get());
            if (!network) network = dynamic_cast<RiverNetworkNode*>(node.get());
            if (!hydraulics) hydraulics = dynamic_cast<RiverHydraulicsNode*>(node.get());
            if (!splineOutput) splineOutput = dynamic_cast<RiverSplineOutputNode*>(node.get());
            if (!carve) carve = dynamic_cast<RiverBedCarveNode*>(node.get());
            if (!fields) fields = dynamic_cast<TerrainFieldsOutputNode*>(node.get());
            if (!terrainAnalysis) terrainAnalysis = dynamic_cast<TerrainAnalysisNode*>(node.get());
            if (!flowMask) flowMask = dynamic_cast<FlowMaskNode*>(node.get());
            if (!soilDepth) soilDepth = dynamic_cast<SoilDepthNode*>(node.get());
            if (!surfaceComposer) surfaceComposer = dynamic_cast<SurfaceComposerNode*>(node.get());
        }
        if (!heightOutput || heightOutput->inputs.empty()) return false;

        uint32_t authoredHeightPin = 0;
        for (const auto& link : links) {
            if (link.endPinId == heightOutput->inputs[0].id) {
                authoredHeightPin = link.startPinId;
                break;
            }
        }
        if (authoredHeightPin == 0) return false;

        // Re-running the setup repairs/reuses the existing branch instead of
        // feeding its carved result back into watershed analysis.
        if (auto* existingCarve = dynamic_cast<RiverBedCarveNode*>(getPinOwner(authoredHeightPin))) {
            carve = existingCarve;
            if (!carve->inputs.empty()) {
                for (const auto& link : links) {
                    if (link.endPinId == carve->inputs[0].id) {
                        authoredHeightPin = link.startPinId;
                        break;
                    }
                }
            }
        }

        // If the authored branch already contains Fluvial Erosion, solve the
        // watershed from its un-eroded input and feed that accumulation back as
        // a guide. This ordering is acyclic:
        // base height -> watershed -> fluvial -> river-bed carve -> Height Output.
        FluvialErosionNode* guidedFluvial = nullptr;
        std::vector<NodeSystem::NodeBase*> searchStack;
        std::unordered_set<uint32_t> visitedNodes;
        if (auto* owner = getPinOwner(authoredHeightPin)) searchStack.push_back(owner);
        while (!searchStack.empty() && !guidedFluvial) {
            NodeSystem::NodeBase* current = searchStack.back();
            searchStack.pop_back();
            if (!current || !visitedNodes.insert(current->id).second) continue;
            if (auto* fluvial = dynamic_cast<FluvialErosionNode*>(current)) {
                guidedFluvial = fluvial;
                break;
            }
            for (const auto& input : current->inputs) {
                for (const auto& link : links) {
                    if (link.endPinId == input.id) {
                        if (auto* upstream = getPinOwner(link.startPinId)) searchStack.push_back(upstream);
                    }
                }
            }
        }

        uint32_t hydrologyHeightPin = authoredHeightPin;
        if (guidedFluvial && !guidedFluvial->inputs.empty()) {
            for (const auto& link : links) {
                if (link.endPinId == guidedFluvial->inputs[0].id) {
                    hydrologyHeightPin = link.startPinId;
                    break;
                }
            }
        }

        if (!watershed) watershed = dynamic_cast<WatershedAnalysisNode*>(
            addTerrainNode(NodeType::WatershedAnalysis, x, y));
        if (!lakeBasin) lakeBasin = dynamic_cast<LakeBasinNode*>(
            addTerrainNode(NodeType::LakeBasin, x + 280.0f, y + 390.0f));
        if (!lakeSurfaceOutput) lakeSurfaceOutput = dynamic_cast<LakeSurfaceOutputNode*>(
            addTerrainNode(NodeType::LakeSurfaceOutput, x + 580.0f, y + 420.0f));
        if (!network) network = dynamic_cast<RiverNetworkNode*>(
            addTerrainNode(NodeType::RiverNetwork, x + 280.0f, y + 80.0f));
        if (!hydraulics) hydraulics = dynamic_cast<RiverHydraulicsNode*>(
            addTerrainNode(NodeType::RiverHydraulics, x + 560.0f, y + 210.0f));
        if (!splineOutput) splineOutput = dynamic_cast<RiverSplineOutputNode*>(
            addTerrainNode(NodeType::RiverSplineOutput, x + 570.0f, y + 50.0f));
        if (!carve) carve = dynamic_cast<RiverBedCarveNode*>(
            addTerrainNode(NodeType::RiverBedCarve, x + 560.0f, y - 180.0f));
        if (!fields) fields = dynamic_cast<TerrainFieldsOutputNode*>(
            addTerrainNode(NodeType::TerrainFieldsOutput, x + 850.0f, y + 270.0f));
        if (!terrainAnalysis) terrainAnalysis = dynamic_cast<TerrainAnalysisNode*>(
            addTerrainNode(NodeType::TerrainAnalysis, x - 310.0f, y + 180.0f));
        if (!flowMask) flowMask = dynamic_cast<FlowMaskNode*>(
            addTerrainNode(NodeType::FlowMask, x - 310.0f, y + 470.0f));
        if (!soilDepth) soilDepth = dynamic_cast<SoilDepthNode*>(
            addTerrainNode(NodeType::SoilDepth, x - 40.0f, y + 470.0f));
        if (!watershed || !lakeBasin || !lakeSurfaceOutput || !network || !hydraulics || !splineOutput || !carve ||
            watershed->inputs.empty() || watershed->outputs.size() < 5 ||
            lakeBasin->inputs.size() < 3 || lakeBasin->outputs.size() < 6 ||
            lakeSurfaceOutput->inputs.size() < 4 ||
            network->inputs.size() < 2 || network->outputs.size() < 3 ||
            hydraulics->inputs.size() < 7 || hydraulics->outputs.size() < 7 ||
            splineOutput->inputs.size() < 13 || carve->inputs.size() < 6 || carve->outputs.size() < 2 ||
            !fields || fields->inputs.size() < 30 || !terrainAnalysis ||
            terrainAnalysis->inputs.empty() || terrainAnalysis->outputs.size() < 5 ||
            !flowMask || flowMask->inputs.empty() || flowMask->outputs.empty() ||
            !soilDepth || soilDepth->inputs.size() < 2 || soilDepth->outputs.empty()) return false;

        addLink(hydrologyHeightPin, watershed->inputs[0].id);
        addLink(hydrologyHeightPin, lakeBasin->inputs[0].id);
        addLink(watershed->outputs[0].id, lakeBasin->inputs[1].id);
        addLink(watershed->outputs[2].id, lakeBasin->inputs[2].id);
        addLink(lakeBasin->outputs[0].id, lakeSurfaceOutput->inputs[0].id);
        addLink(lakeBasin->outputs[1].id, lakeSurfaceOutput->inputs[1].id);
        addLink(lakeBasin->outputs[2].id, lakeSurfaceOutput->inputs[2].id);
        addLink(lakeBasin->outputs[5].id, lakeSurfaceOutput->inputs[3].id);
        if (guidedFluvial && guidedFluvial->inputs.size() >= 4) {
            addLink(watershed->outputs[1].id, guidedFluvial->inputs[3].id);
        }
        addLink(watershed->outputs[1].id, network->inputs[0].id);
        addLink(watershed->outputs[2].id, network->inputs[1].id);
        addLink(hydrologyHeightPin, terrainAnalysis->inputs[0].id);
        if (terrainAnalysis->inputs.size() >= 2) {
            addLink(watershed->outputs[1].id, terrainAnalysis->inputs[1].id);
        }
        addLink(hydrologyHeightPin, flowMask->inputs[0].id);
        addLink(hydrologyHeightPin, soilDepth->inputs[0].id);
        addLink(flowMask->outputs[0].id, soilDepth->inputs[1].id);
        if (surfaceComposer && surfaceComposer->inputs.size() >= 10) {
            addLink(soilDepth->outputs[0].id, surfaceComposer->inputs[1].id);
            addLink(flowMask->outputs[0].id, surfaceComposer->inputs[2].id);
            addLink(terrainAnalysis->outputs[4].id, surfaceComposer->inputs[3].id);
        }
        addLink(watershed->outputs[0].id, splineOutput->inputs[0].id);
        addLink(watershed->outputs[1].id, splineOutput->inputs[1].id);
        addLink(watershed->outputs[2].id, splineOutput->inputs[2].id);
        addLink(network->outputs[0].id, splineOutput->inputs[3].id);
        addLink(lakeBasin->outputs[0].id, splineOutput->inputs[4].id);
        addLink(lakeBasin->outputs[2].id, splineOutput->inputs[5].id);
        addLink(authoredHeightPin, carve->inputs[0].id);
        addLink(network->outputs[0].id, carve->inputs[1].id);
        addLink(carve->outputs[0].id, heightOutput->inputs[0].id);
        addLink(authoredHeightPin, hydraulics->inputs[0].id);
        addLink(watershed->outputs[4].id, hydraulics->inputs[1].id);
        addLink(watershed->outputs[2].id, hydraulics->inputs[2].id);
        addLink(network->outputs[0].id, hydraulics->inputs[3].id);
        addLink(lakeBasin->outputs[0].id, hydraulics->inputs[4].id);
        addLink(lakeBasin->outputs[2].id, hydraulics->inputs[5].id);
        addLink(hydrologyHeightPin, hydraulics->inputs[6].id);
        addLink(hydraulics->outputs[1].id, carve->inputs[2].id);
        addLink(hydraulics->outputs[2].id, carve->inputs[3].id);
        addLink(hydrologyHeightPin, carve->inputs[4].id);
        addLink(hydraulics->outputs[4].id, carve->inputs[5].id);
        addLink(hydraulics->outputs[1].id, splineOutput->inputs[6].id);
        addLink(hydraulics->outputs[2].id, splineOutput->inputs[7].id);
        addLink(hydraulics->outputs[3].id, splineOutput->inputs[8].id);
        addLink(hydraulics->outputs[0].id, splineOutput->inputs[9].id);
        addLink(hydraulics->outputs[5].id, splineOutput->inputs[10].id);
        addLink(hydraulics->outputs[6].id, splineOutput->inputs[11].id);
        addLink(hydraulics->outputs[4].id, splineOutput->inputs[12].id);
        // Persistent hydrology contract for wetness/material/foliage today and
        // the low-cost water/foam system that will consume these fields next.
        addLink(watershed->outputs[1].id, fields->inputs[9].id);
        addLink(watershed->outputs[2].id, fields->inputs[10].id);
        addLink(watershed->outputs[3].id, fields->inputs[11].id);
        addLink(network->outputs[0].id, fields->inputs[12].id);
        addLink(network->outputs[1].id, fields->inputs[13].id);
        addLink(network->outputs[2].id, fields->inputs[14].id);
        addLink(carve->outputs[1].id, fields->inputs[15].id);
        for (int i = 0; i < 6; ++i) {
            addLink(lakeBasin->outputs[static_cast<size_t>(i)].id,
                    fields->inputs[static_cast<size_t>(16 + i)].id);
        }
        addLink(watershed->outputs[4].id, fields->inputs[22].id);
        for (int i = 0; i < 7; ++i) {
            addLink(hydraulics->outputs[static_cast<size_t>(i)].id,
                    fields->inputs[static_cast<size_t>(23 + i)].id);
        }

        ensureTerrainLayerGroup(*this, "01 Analysis", IM_COL32(70, 105, 135, 90),
                                {terrainAnalysis, flowMask, soilDepth});
        ensureTerrainLayerGroup(*this, "03 River", IM_COL32(32, 105, 175, 90),
                                {watershed, network, hydraulics, carve, splineOutput});
        ensureTerrainLayerGroup(*this, "04 Lakes", IM_COL32(25, 135, 185, 90),
                                {lakeBasin, lakeSurfaceOutput});
        ensureTerrainLayerGroup(*this, "06 Outputs", IM_COL32(105, 90, 130, 90),
                                {heightOutput, fields, surfaceComposer});
        layoutAutoManagedTerrainLayers(*this);
        markAllDirty();
        return true;
    }

    void TerrainNodeGraphV2::createSnowyMountainValleyGraph(TerrainObject* terrain) {
        if (previewActive_) restoreCommittedTerrainData(terrain);
        clearCommittedPreviewSnapshot();
        cachedEvalContext_.reset();
        cachedTerrainCtx_.reset();
        clear();

        auto* noise = dynamic_cast<NoiseGeneratorNode*>(addTerrainNode(NodeType::NoiseGenerator, 40.0f, 170.0f));
        auto* erosion = dynamic_cast<HydraulicErosionNode*>(addTerrainNode(NodeType::HydraulicErosion, 270.0f, 170.0f));
        auto* snow = dynamic_cast<SnowClimateNode*>(addTerrainNode(NodeType::SnowClimate, 530.0f, 110.0f));
        auto* slope = dynamic_cast<SlopeMaskNode*>(addTerrainNode(NodeType::SlopeMask, 520.0f, 470.0f));
        auto* grass = dynamic_cast<InvertNode*>(addTerrainNode(NodeType::Invert, 720.0f, 500.0f));
        auto* flow = dynamic_cast<FlowMaskNode*>(addTerrainNode(NodeType::FlowMask, 520.0f, 650.0f));
        auto* composer = dynamic_cast<SurfaceComposerNode*>(addTerrainNode(NodeType::SurfaceComposer, 850.0f, 350.0f));
        auto* heightOutput = dynamic_cast<HeightOutputNode*>(addTerrainNode(NodeType::HeightOutput, 900.0f, 90.0f));
        auto* splatOutput = dynamic_cast<SplatOutputNode*>(addTerrainNode(NodeType::SplatOutput, 1100.0f, 370.0f));
        if (!noise || !erosion || !snow || !slope || !grass || !flow || !composer || !heightOutput || !splatOutput) {
            createDefaultGraph(terrain);
            return;
        }

        noise->noiseType = NoiseType::Ridge;
        noise->seed = 1847;
        noise->scale = 0.75f;
        noise->frequency = 0.012f;
        noise->amplitude = 1.0f;
        noise->octaves = 7;
        noise->persistance = 0.52f;
        noise->lacunarity = 2.15f;
        noise->ridge_offset = 1.05f;

        erosion->useGPU = true;
        erosion->params.iterations = 75000;
        erosion->params.dropletLifetime = 96;
        erosion->params.inertia = 0.12f;
        erosion->params.sedimentCapacity = 1.8f;
        erosion->params.erodeSpeed = 0.065f;
        erosion->params.depositSpeed = 0.12f;
        erosion->params.evaporateSpeed = 0.008f;
        erosion->params.erosionRadius = 3;

        snow->applyPreset(SnowClimatePreset::AlpineBalanced);
        snow->snowLineFraction = 0.52f;
        snow->snowfallMeters = 0.55f;
        snow->maxDepthMeters = 2.0f;
        snow->geometryAmount = 0.75f;
        snow->coverageAmount = 1.0f;
        snow->valleyCapture = 1.10f;

        slope->minSlope = 25.0f;
        slope->maxSlope = 72.0f;
        slope->falloff = 0.22f;
        flow->detailLevel = 7;
        flow->bankSpread = 1;
        flow->strength = 1.0f;
        flow->decay = 0.995f;
        flow->channelSoftness = 0.055f;
        composer->patchiness = 0.25f;
        composer->snowInfluence = 1.0f;
        composer->iceInfluence = 0.85f;

        // Base geometry fans out to every geological mask and to the climate
        // solver. Only Snow Climate's Surface Height reaches Height Output.
        addLink(noise->outputs[0].id, erosion->inputs[0].id);
        addLink(erosion->outputs[0].id, snow->inputs[0].id);
        addLink(erosion->outputs[0].id, slope->inputs[0].id);
        addLink(erosion->outputs[0].id, flow->inputs[0].id);
        addLink(erosion->outputs[0].id, composer->inputs[0].id);
        addLink(snow->outputs[0].id, heightOutput->inputs[0].id);
        addLink(slope->outputs[0].id, grass->inputs[0].id);
        addLink(grass->outputs[0].id, composer->inputs[8].id);
        addLink(slope->outputs[0].id, composer->inputs[9].id);
        addLink(flow->outputs[0].id, composer->inputs[2].id);
        addLink(snow->outputs[1].id, composer->inputs[5].id);
        addLink(snow->outputs[2].id, composer->inputs[6].id);
        addLink(snow->outputs[3].id, composer->inputs[7].id);
        addLink(composer->outputs[1].id, splatOutput->inputs[0].id);
        markAllDirty();
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
            groupJson["color"] = group.color;
            groupJson["comment"] = group.comment;
            groupJson["collapsed"] = group.collapsed;
            groupJson["publicationEnabled"] = group.publicationEnabled;
            groupJson["nextInterfacePortId"] = group.nextInterfacePortId;
            nlohmann::json portsArray = nlohmann::json::array();
            for (const auto& port : group.interfacePorts) {
                portsArray.push_back({
                    {"id", port.id},
                    {"direction", static_cast<int>(port.direction)},
                    {"name", port.name},
                    {"internalPinId", port.internalPinId},
                    {"externalPinId", port.externalPinId},
                    {"dataType", static_cast<int>(port.dataType)},
                    {"imageSemantic", static_cast<int>(port.imageSemantic)},
                    {"imageChannels", port.imageChannels}
                });
            }
            groupJson["interfacePorts"] = std::move(portsArray);
            groupsArray.push_back(groupJson);
        }
        j["groups"] = groupsArray;
        
        return j;
    }
    
    void TerrainNodeGraphV2::fromJson(const nlohmann::json& j, TerrainObject* terrain) {
        // Clear existing graph
        if (previewActive_) restoreCommittedTerrainData(terrain);
        clearCommittedPreviewSnapshot();
        cachedEvalContext_.reset();
        cachedTerrainCtx_.reset();
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
                } else if (typeId == "TerrainV2.HardnessInput") {
                    newNode = addTerrainNode(NodeType::HardnessInput, x, y);
                } else if (typeId == "TerrainV2.HardnessOutput") {
                    newNode = addTerrainNode(NodeType::HardnessOutput, x, y);
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
                } else if (typeId == "TerrainV2.EdgeFalloff") {
                    newNode = addTerrainNode(NodeType::EdgeFalloff, x, y);
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
                } else if (typeId == "TerrainV2.Resample") {
                    newNode = addTerrainNode(NodeType::Resample, x, y);
                } else if (typeId == "TerrainV2.ChannelExtract") {
                    newNode = addTerrainNode(NodeType::ChannelExtract, x, y);
                } else if (typeId == "TerrainV2.SplatCompose") {
                    newNode = addTerrainNode(NodeType::SplatCompose, x, y);
                } else if (typeId == "TerrainV2.Remap") {
                    newNode = addTerrainNode(NodeType::Remap, x, y);
                } else if (typeId == "TerrainV2.MaskAdjust") {
                    newNode = addTerrainNode(NodeType::MaskAdjust, x, y);
                } else if (typeId == "TerrainV2.MaskMorphology") {
                    newNode = addTerrainNode(NodeType::MaskMorphology, x, y);
                } else if (typeId == "TerrainV2.WetnessMap") {
                    newNode = addTerrainNode(NodeType::WetnessMap, x, y);
                } else if (typeId == "TerrainV2.SoilDepth") {
                    newNode = addTerrainNode(NodeType::SoilDepth, x, y);
                } else if (typeId == "TerrainV2.Lithology") {
                    newNode = addTerrainNode(NodeType::Lithology, x, y);
                } else if (typeId == "TerrainV2.Strata") {
                    newNode = addTerrainNode(NodeType::Strata, x, y);
                } else if (typeId == "TerrainV2.SurfaceComposer") {
                    newNode = addTerrainNode(NodeType::SurfaceComposer, x, y);
                } else if (typeId == "TerrainV2.SnowClimate") {
                    newNode = addTerrainNode(NodeType::SnowClimate, x, y);
                } else if (typeId == "TerrainV2.Climate") {
                    newNode = addTerrainNode(NodeType::Climate, x, y);
                } else if (typeId == "TerrainV2.Snowfall") {
                    newNode = addTerrainNode(NodeType::Snowfall, x, y);
                } else if (typeId == "TerrainV2.SnowSettle") {
                    newNode = addTerrainNode(NodeType::SnowSettle, x, y);
                } else if (typeId == "TerrainV2.SnowMeltFreeze") {
                    newNode = addTerrainNode(NodeType::SnowMeltFreeze, x, y);
                } else if (typeId == "TerrainV2.GlacierFlow") {
                    newNode = addTerrainNode(NodeType::GlacierFlow, x, y);
                } else if (typeId == "TerrainV2.TerrainAnalysis") {
                    newNode = addTerrainNode(NodeType::TerrainAnalysis, x, y);
                } else if (typeId == "TerrainV2.TerrainFieldsOutput") {
                    newNode = addTerrainNode(NodeType::TerrainFieldsOutput, x, y);
                } else if (typeId == "TerrainV2.BiomeComposer") {
                    newNode = addTerrainNode(NodeType::BiomeComposer, x, y);
                } else if (typeId == "TerrainV2.FoliageLayer") {
                    newNode = addTerrainNode(NodeType::FoliageLayer, x, y);
                } else if (typeId == "TerrainV2.FoliageSet") {
                    newNode = addTerrainNode(NodeType::FoliageSet, x, y);
                } else if (typeId == "TerrainV2.FoliageOutput") {
                    newNode = addTerrainNode(NodeType::FoliageOutput, x, y);
                } else if (typeId == "TerrainV2.WatershedAnalysis") {
                    newNode = addTerrainNode(NodeType::WatershedAnalysis, x, y);
                } else if (typeId == "TerrainV2.LakeBasin") {
                    newNode = addTerrainNode(NodeType::LakeBasin, x, y);
                } else if (typeId == "TerrainV2.LakeSurfaceOutput") {
                    newNode = addTerrainNode(NodeType::LakeSurfaceOutput, x, y);
                } else if (typeId == "TerrainV2.RiverNetwork") {
                    newNode = addTerrainNode(NodeType::RiverNetwork, x, y);
                } else if (typeId == "TerrainV2.RiverHydraulics") {
                    newNode = addTerrainNode(NodeType::RiverHydraulics, x, y);
                } else if (typeId == "TerrainV2.RiverSplineOutput") {
                    newNode = addTerrainNode(NodeType::RiverSplineOutput, x, y);
                } else if (typeId == "TerrainV2.RiverBedCarve") {
                    newNode = addTerrainNode(NodeType::RiverBedCarve, x, y);
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
                if (NodeSystem::NodeGroup* group = getGroup(groupId)) {
                    group->color = groupJson.value("color", group->color);
                    group->comment = groupJson.value("comment", std::string{});
                    group->collapsed = groupJson.value("collapsed", false);
                    group->publicationEnabled = groupJson.value("publicationEnabled", true);
                    group->nextInterfacePortId = groupJson.value("nextInterfacePortId", 1u);
                    if (groupJson.contains("interfacePorts") && groupJson["interfacePorts"].is_array()) {
                        for (const auto& portJson : groupJson["interfacePorts"]) {
                            NodeSystem::LayerInterfacePort port;
                            port.id = portJson.contains("id")
                                ? portJson["id"].get<uint32_t>() : group->nextInterfacePortId++;
                            port.direction = static_cast<NodeSystem::LayerPortDirection>(
                                portJson.value("direction", 0));
                            port.name = portJson.value("name", std::string{});
                            const uint32_t oldInternalPinId = portJson.value("internalPinId", 0u);
                            const uint32_t oldExternalPinId = portJson.value("externalPinId", 0u);
                            const auto internalIt = oldToNewPinId.find(oldInternalPinId);
                            const auto externalIt = oldToNewPinId.find(oldExternalPinId);
                            port.internalPinId = internalIt != oldToNewPinId.end() ? internalIt->second : 0u;
                            port.externalPinId = externalIt != oldToNewPinId.end() ? externalIt->second : 0u;
                            port.dataType = static_cast<NodeSystem::DataType>(
                                portJson.value("dataType", static_cast<int>(NodeSystem::DataType::None)));
                            port.imageSemantic = static_cast<NodeSystem::ImageSemantic>(
                                portJson.value("imageSemantic", static_cast<int>(NodeSystem::ImageSemantic::Generic)));
                            port.imageChannels = portJson.value("imageChannels", 1);
                            port.connected = false; // Reconciled against restored links on first editor frame.
                            group->nextInterfacePortId = (std::max)(group->nextInterfacePortId, port.id + 1u);
                            group->interfacePorts.push_back(std::move(port));
                        }
                    }
                }
                
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
