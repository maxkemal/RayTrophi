#pragma once

/**
 * @file TerrainNodesV2.h
 * @brief Modern terrain node implementations using new node system
 * 
 * This file contains example implementations of terrain nodes
 * using the new TerrainNodeV2 base class and compute() pattern.
 */

#include "TerrainNodeBase.h"
#include <cmath>
#include <algorithm>

namespace TerrainNodes {

    // ============================================================================
    // INPUT NODES
    // ============================================================================
    
    /**
     * @brief Reads current terrain heightmap as starting point
     */
    class HeightmapInputNodeV2 : public TerrainNodeV2 {
    public:
        HeightmapInputNodeV2() {
            metadata.displayName = "Heightmap Input";
            metadata.category = Categories::Input;
            metadata.headerColor = IM_COL32(50, 150, 80, 255);
            metadata.typeId = "terrain.input.heightmap";
            
            addHeightOutput("Height");
        }
        
        void computeTerrain(TerrainObject* terrain, NodeSystem::EvaluationContext& ctx) override {
            if (!terrain) return;
            
            int res = terrain->heightmap.width;
            cachedHeightData.resize(res * res);
            
            // Copy from terrain's heightmap
            for (int y = 0; y < res; y++) {
                for (int x = 0; x < res; x++) {
                    cachedHeightData[y * res + x] = terrain->heightmap.data[y * res + x];
                }
            }
            
            hasCachedHeight = true;
        }
        
        void drawContent() override {
            ImGui::TextColored(ImVec4(0.7f, 0.9f, 0.7f, 1.0f), "Terrain Source");
        }
    };

    // ============================================================================
    // NOISE GENERATOR
    // ============================================================================
    
    enum class NoiseTypeV2 { Perlin, Voronoi, Simplex };
    
    /**
     * @brief Generates procedural noise heightmaps
     */
    class NoiseGeneratorNodeV2 : public TerrainNodeV2 {
    public:
        NoiseTypeV2 noiseType = NoiseTypeV2::Perlin;
        int seed = 1337;
        float scale = 0.01f;
        float frequency = 1.0f;
        float amplitude = 1.0f;
        int octaves = 6;
        float persistence = 0.5f;
        float lacunarity = 2.0f;
        float jitter = 1.0f;  // Voronoi
        
        NoiseGeneratorNodeV2() {
            metadata.displayName = "Noise Generator";
            metadata.category = Categories::Input;
            metadata.headerColor = IM_COL32(50, 150, 130, 255);
            metadata.typeId = "terrain.input.noise";
            
            addHeightOutput("Height");
        }
        
        void computeTerrain(TerrainObject* terrain, NodeSystem::EvaluationContext& ctx) override {
            if (!terrain) return;
            
            int res = terrain->heightmap.width;
            cachedHeightData.resize(res * res);
            
            // Generate noise
            for (int y = 0; y < res; y++) {
                for (int x = 0; x < res; x++) {
                    float nx = x * scale;
                    float ny = y * scale;
                    
                    float value = generateNoise(nx, ny);
                    cachedHeightData[y * res + x] = value * amplitude;
                }
                
                // Update progress
                ctx.setProgress((float)y / res);
            }
            
            hasCachedHeight = true;
        }
        
        void drawContent() override {
            const char* noiseNames[] = { "Perlin", "Voronoi", "Simplex" };
            int current = (int)noiseType;
            ImGui::PushItemWidth(100);
            if (ImGui::Combo("Type", &current, noiseNames, 3)) {
                noiseType = (NoiseTypeV2)current;
                dirty = true;
            }
            ImGui::DragInt("Seed", &seed, 1);
            ImGui::DragFloat("Scale", &scale, 0.001f, 0.001f, 1.0f);
            ImGui::DragFloat("Amplitude", &amplitude, 0.1f, 0.1f, 1000.0f);
            ImGui::DragInt("Octaves", &octaves, 1, 1, 12);
            ImGui::PopItemWidth();
        }
        
    private:
        float generateNoise(float x, float y) {
            // Simple FBM implementation
            float total = 0.0f;
            float amplitude = 1.0f;
            float freq = frequency;
            float maxValue = 0.0f;
            
            for (int i = 0; i < octaves; i++) {
                total += noise2D(x * freq + seed, y * freq + seed) * amplitude;
                maxValue += amplitude;
                amplitude *= persistence;
                freq *= lacunarity;
            }
            
            return total / maxValue;
        }
        
        float noise2D(float x, float y) {
            // Simple hash-based noise
            int xi = (int)std::floor(x);
            int yi = (int)std::floor(y);
            float xf = x - xi;
            float yf = y - yi;
            
            // Smooth interpolation
            xf = xf * xf * (3 - 2 * xf);
            yf = yf * yf * (3 - 2 * yf);
            
            auto hash = [](int x, int y) -> float {
                int n = x + y * 57;
                n = (n << 13) ^ n;
                return 1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741823.5f;
            };
            
            float a = hash(xi, yi);
            float b = hash(xi + 1, yi);
            float c = hash(xi, yi + 1);
            float d = hash(xi + 1, yi + 1);
            
            return a * (1 - xf) * (1 - yf) + b * xf * (1 - yf) +
                   c * (1 - xf) * yf + d * xf * yf;
        }
    };

    // ============================================================================
    // EROSION NODES
    // ============================================================================
    
    /**
     * @brief Hydraulic erosion simulation
     */
    class HydraulicErosionNodeV2 : public TerrainNodeV2 {
    public:
        int iterations = 50000;
        float erosionRate = 0.3f;
        float depositionRate = 0.3f;
        float evaporationRate = 0.02f;
        float gravity = 4.0f;
        float minSlope = 0.01f;
        int erosionRadius = 3;
        bool useGPU = true;
        
        HydraulicErosionNodeV2() {
            metadata.displayName = "Hydraulic Erosion";
            metadata.category = Categories::Erosion;
            metadata.headerColor = IM_COL32(80, 130, 200, 255);
            metadata.typeId = "terrain.erosion.hydraulic";
            
            addHeightInput("Height In");
            addMaskInput("Mask");
            addHeightOutput("Height Out");
        }
        
        void computeTerrain(TerrainObject* terrain, NodeSystem::EvaluationContext& ctx) override {
            // Get input height data
            const std::vector<float>* inputData = getInputHeightData(0, ctx);
            if (!inputData) {
                // If no input, use terrain's current heightmap
                int res = terrain->heightmap.width;
                cachedHeightData.resize(res * res);
                for (int y = 0; y < res; y++) {
                    for (int x = 0; x < res; x++) {
                        cachedHeightData[y * res + x] = terrain->heightmap.data[y * res + x];
                    }
                }
            } else {
                cachedHeightData = *inputData;
            }
            
            // Optional mask
            const std::vector<float>* maskData = getInputMaskData(1, ctx);
            
            // Apply erosion
            int res = terrain->heightmap.width;
           
            applyErosion(cachedHeightData.data(), res, maskData ? maskData->data() : nullptr, ctx);
            
            hasCachedHeight = true;
        }
        
        void drawContent() override {
            ImGui::PushItemWidth(100);
            ImGui::DragInt("Iterations", &iterations, 1000, 1000, 500000);
            ImGui::DragFloat("Erosion", &erosionRate, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Deposit", &depositionRate, 0.01f, 0.0f, 1.0f);
            ImGui::DragInt("Radius", &erosionRadius, 1, 1, 10);
            ImGui::Checkbox("GPU", &useGPU);
            ImGui::PopItemWidth();
        }
        
    private:
        void applyErosion(float* heights, int res, const float* mask, NodeSystem::EvaluationContext& ctx) {
            // Simplified droplet erosion
            for (int i = 0; i < iterations; i++) {
                if (ctx.isCancelled()) break;
                
                // Random droplet position
                float posX = (float)(rand() % res);
                float posY = (float)(rand() % res);
                float dirX = 0, dirY = 0;
                float speed = 1.0f;
                float water = 1.0f;
                float sediment = 0.0f;
                
                // Simulate droplet
                for (int step = 0; step < 64; step++) {
                    int xi = (int)posX;
                    int yi = (int)posY;
                    
                    if (xi < 0 || xi >= res - 1 || yi < 0 || yi >= res - 1) break;
                    
                    int idx = yi * res + xi;
                    float h = heights[idx];
                    
                    // Calculate gradient
                    float gx = heights[idx + 1] - h;
                    float gy = heights[idx + res] - h;
                    
                    // Update direction
                    dirX = dirX * 0.5f - gx;
                    dirY = dirY * 0.5f - gy;
                    float len = std::sqrt(dirX * dirX + dirY * dirY);
                    if (len > 0.001f) { dirX /= len; dirY /= len; }
                    
                    // Move droplet
                    posX += dirX;
                    posY += dirY;
                    
                    // New position
                    int nxi = (int)posX;
                    int nyi = (int)posY;
                    if (nxi < 0 || nxi >= res || nyi < 0 || nyi >= res) break;
                    
                    float newH = heights[nyi * res + nxi];
                    float diff = h - newH;
                    
                    // Check mask
                    float maskVal = mask ? mask[idx] : 1.0f;
                    
                    if (diff > 0) {
                        // Going downhill - erode
                        float erosionAmount = std::min(diff, erosionRate * speed * water * maskVal);
                        heights[idx] -= erosionAmount;
                        sediment += erosionAmount;
                    } else {
                        // Going uphill or flat - deposit
                        float depositAmount = std::min(sediment, -diff * depositionRate * maskVal);
                        heights[idx] += depositAmount;
                        sediment -= depositAmount;
                    }
                    
                    speed = std::sqrt(speed * speed + diff * gravity);
                    water *= (1.0f - evaporationRate);
                    
                    if (water < 0.01f) break;
                }
                
                // Progress update
                if (i % 5000 == 0) {
                    ctx.setProgress((float)i / iterations);
                }
            }
        }
    };

    // ============================================================================
    // MATH NODES
    // ============================================================================
    
    /**
     * @brief Blend two height inputs
     */
    class BlendNodeV2 : public TerrainNodeV2 {
    public:
        float alpha = 0.5f;
        bool useMaskAsAlpha = true;
        
        BlendNodeV2() {
            metadata.displayName = "Blend";
            metadata.category = Categories::Math;
            metadata.headerColor = IM_COL32(130, 100, 150, 255);
            metadata.typeId = "terrain.math.blend";
            
            addHeightInput("A");
            addHeightInput("B");
            addMaskInput("Mask");
            addHeightOutput("Result");
        }
        
        void computeTerrain(TerrainObject* terrain, NodeSystem::EvaluationContext& ctx) override {
            const std::vector<float>* dataA = getInputHeightData(0, ctx);
            const std::vector<float>* dataB = getInputHeightData(1, ctx);
            const std::vector<float>* mask = getInputMaskData(2, ctx);
            
            if (!dataA && !dataB) {
                ctx.addError(id, "At least one input required");
                return;
            }
            
            int res = terrain->heightmap.width;
           
            size_t size = res * res;
            cachedHeightData.resize(size);
            
            for (size_t i = 0; i < size; i++) {
                float a = dataA ? (*dataA)[i] : 0.0f;
                float b = dataB ? (*dataB)[i] : 0.0f;
                float t = (mask && useMaskAsAlpha) ? (*mask)[i] : alpha;
                
                cachedHeightData[i] = a * (1.0f - t) + b * t;
            }
            
            hasCachedHeight = true;
        }
        
        void drawContent() override {
            ImGui::PushItemWidth(80);
            ImGui::SliderFloat("Alpha", &alpha, 0.0f, 1.0f);
            ImGui::Checkbox("Use Mask", &useMaskAsAlpha);
            ImGui::PopItemWidth();
        }
    };

    // ============================================================================
    // OUTPUT NODES
    // ============================================================================
    
    /**
     * @brief Writes result back to terrain
     */
    class HeightOutputNodeV2 : public TerrainNodeV2 {
    public:
        HeightOutputNodeV2() {
            metadata.displayName = "Height Output";
            metadata.category = Categories::Output;
            metadata.headerColor = IM_COL32(200, 100, 80, 255);
            metadata.typeId = "terrain.output.height";
            
            addHeightInput("Height");
        }
        
        void computeTerrain(TerrainObject* terrain, NodeSystem::EvaluationContext& ctx) override {
            const std::vector<float>* inputData = getInputHeightData(0, ctx);
            if (!inputData) {
                ctx.addError(id, "No height input connected");
                return;
            }
            
            // Apply to terrain
            int res = terrain->heightmap.width;
            for (int y = 0; y < res; y++) {
                for (int x = 0; x < res; x++) {
                    terrain->heightmap.setHeight(x, y, (*inputData)[y * res + x]);
                }
            }
            
            // Mark mesh as dirty for recalculation
            terrain->dirty_mesh = true;
        }
    };

} // namespace TerrainNodes
