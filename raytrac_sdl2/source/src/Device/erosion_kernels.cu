#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include "erosion_ops.cuh"
#include <stdio.h>

using namespace TerrainPhysics;

// -----------------------------------------------------------------------
// Device Helpers (OPTIMIZED: No loops)
// -----------------------------------------------------------------------

__device__ float bilinearInterpolate(float* map, int width, int height, float x, float y) {
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // Clamp
    if (x0 < 0) x0 = 0; if (x0 >= width) x0 = width - 1;
    if (y0 < 0) y0 = 0; if (y0 >= height) y0 = height - 1;
    if (x1 < 0) x1 = 0; if (x1 >= width) x1 = width - 1;
    if (y1 < 0) y1 = 0; if (y1 >= height) y1 = height - 1;

    float h00 = map[y0 * width + x0];
    float h10 = map[y0 * width + x1];
    float h01 = map[y1 * width + x0];
    float h11 = map[y1 * width + x1];

    float tx = x - x0;
    float ty = y - y0;

    return (h00 * (1 - tx) * (1 - ty) +
            h10 * tx * (1 - ty) +
            h01 * (1 - tx) * ty +
            h11 * tx * ty);
}

// -----------------------------------------------------------------------
// Thermal Erosion Kernel - INDUSTRY STANDARD (Weighted Multi-Neighbor Transfer)
// Based on Musgrave's thermal weathering model with proportional distribution
// -----------------------------------------------------------------------
extern "C" __global__ void thermalErosionKernel(float* heightmap, ThermalErosionParamsGPU p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check (ignore edges)
    if (x <= 0 || x >= p.mapWidth - 1 || y <= 0 || y >= p.mapHeight - 1) return;
    
    int idx = y * p.mapWidth + x;
    float h = heightmap[idx];
    
    float diffs[8];
    int nIdxs[8];
    float totalExcess = 0.0f;
    int neighborCount = 0;
    
    // Physical slope = dh * heightScale / (dist * cellSize)
    float safeHeightScale = fmaxf(1.0f, p.heightScale);
    float invCellSize = 1.0f / fmaxf(0.001f, p.cellSize);
    float slopeScale = safeHeightScale * invCellSize;
    
    // D8 neighbor offsets and distance weights
    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float dists[8] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};
    
    for (int d = 0; d < 8; d++) {
        nIdxs[d] = (y + dy[d]) * p.mapWidth + (x + dx[d]);
        
        float diff = h - heightmap[nIdxs[d]];
        float slope = diff * slopeScale / dists[d];
        
        if (slope > p.talusAngle) {
            // Amount of normalized height to move
            float excess = (slope - p.talusAngle) * dists[d] * p.cellSize / safeHeightScale;
            diffs[d] = excess;
            totalExcess += excess;
            neighborCount++;
        } else {
            diffs[d] = 0.0f;
        }
    }
    
    // Second pass: Distribute material proportionally to all steep neighbors
    if (totalExcess > 0.0f && neighborCount > 0) {
        // Total amount to move (half of excess, scaled by erosion rate)
        float totalMove = totalExcess * 0.5f * p.erosionAmount;
        
        // STABILITY CAP: Never move more than 5% of height per step
        totalMove = fminf(totalMove, 0.05f);
        if (totalMove > h) totalMove = h * 0.9f;
        
        // Remove from center
        atomicAdd(&heightmap[idx], -totalMove);
        
        // Distribute to neighbors proportionally
        for (int d = 0; d < 8; d++) {
            if (diffs[d] > 0.0f) {
                float weight = diffs[d] / totalExcess;
                atomicAdd(&heightmap[nIdxs[d]], totalMove * weight);
            }
        }
    }
}



// -----------------------------------------------------------------------
// Fluvial (Pipe Method) Kernels
// -----------------------------------------------------------------------
// Note: Requires additional buffers (Water, Flux[4], Sediment) allocated by Host

// 1. Add Rain & Source Water
extern "C" __global__ void fluvialRainKernel(float* waterMap, int width, int height, float rainAmount, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    waterMap[idx] += rainAmount * dt;
}

// 2. Compute Flux (Outflow)
// fL, fR, fT, fB stored in fluxMap (4 * w * h floats)
extern "C" __global__ void fluvialFluxKernel(float* heightMap, float* waterMap, float* fluxMap, FluvialErosionParamsGPU p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 0 || x >= p.mapWidth - 1 || y <= 0 || y >= p.mapHeight - 1) return;
    int idx = y * p.mapWidth + x;
    
    // Inputs
    float H = heightMap[idx] + waterMap[idx];
    
    // Neighbors (Left, Right, Top, Bottom)
    int nIdx[4];
    nIdx[0] = idx - 1; // Left
    nIdx[1] = idx + 1; // Right
    nIdx[2] = idx - p.mapWidth; // Top
    nIdx[3] = idx + p.mapWidth; // Bottom
    
    // Current Flux // 0:L, 1:R, 2:T, 3:B
    float* fL = &fluxMap[idx * 4 + 0];
    float* fR = &fluxMap[idx * 4 + 1];
    float* fT = &fluxMap[idx * 4 + 2];
    float* fB = &fluxMap[idx * 4 + 3];
    
    // Calculate Height Diffs in meters
    float safeHeightScale = fmaxf(1.0f, p.heightScale);
    float dH[4];
    dH[0] = (H - (heightMap[nIdx[0]] + waterMap[nIdx[0]])) * safeHeightScale;
    dH[1] = (H - (heightMap[nIdx[1]] + waterMap[nIdx[1]])) * safeHeightScale;
    dH[2] = (H - (heightMap[nIdx[2]] + waterMap[nIdx[2]])) * safeHeightScale;
    dH[3] = (H - (heightMap[nIdx[3]] + waterMap[nIdx[3]])) * safeHeightScale;
    
    // Update Flux (Pipe Model)
    // fluxFactor = dt * A * g / L
    float invCellSize = 1.0f / fmaxf(0.01f, p.cellSize);
    float fluxFactor = p.fixedDeltaTime * p.gravity * invCellSize;
    
    *fL = fmaxf(0.0f, *fL + fluxFactor * dH[0]);
    *fR = fmaxf(0.0f, *fR + fluxFactor * dH[1]);
    *fT = fmaxf(0.0f, *fT + fluxFactor * dH[2]);
    *fB = fmaxf(0.0f, *fB + fluxFactor * dH[3]);
    
    // Scaling to prevent negative water
    float sumFlux = *fL + *fR + *fT + *fB;
    // float velocityLimit = 1000.0f; // Limit excessive velocity (unused variable warning fixed)
    if (sumFlux > 0) {
        float K = fminf(1.0f, (waterMap[idx] * p.cellSize * p.cellSize) / (sumFlux * p.fixedDeltaTime));
        *fL *= K; *fR *= K; *fT *= K; *fB *= K;
    }
}

// 3. Update Water Volume (and Velocity Field)
extern "C" __global__ void fluvialWaterKernel(float* waterMap, float* fluxMap, float* velocityMap, FluvialErosionParamsGPU p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 1 || x >= p.mapWidth - 2 || y <= 1 || y >= p.mapHeight - 2) return;
    int idx = y * p.mapWidth + x;
    
    // Inflow sum
    float inflow = 0;
    // From Left neighbor (their Right flux)
    inflow += fluxMap[(idx - 1) * 4 + 1];
    // From Right neighbor (their Left flux)
    inflow += fluxMap[(idx + 1) * 4 + 0];
    // From Top neighbor (their Bottom flux)
    inflow += fluxMap[(idx - p.mapWidth) * 4 + 3];
    // From Bottom neighbor (their Top flux)
    inflow += fluxMap[(idx + p.mapWidth) * 4 + 2];
    
    // Outflow sum
    float outflow = fluxMap[idx*4+0] + fluxMap[idx*4+1] + fluxMap[idx*4+2] + fluxMap[idx*4+3];
    
    // Change in volume
    float dV = p.fixedDeltaTime * (inflow - outflow);
    float oldW = waterMap[idx];
    float newW = fmaxf(0.0f, oldW + dV / (p.cellSize * p.cellSize));
    
    waterMap[idx] = newW;
    
    // Calculate Velocity (for Erosion)
    // vx = (inL - outL + inR - outR) / 2 ... (Average flux through x face)
    // Simplified: Velocity vector based on net flux direction
    float fluxL = fluxMap[idx*4+0]; // Out Left
    float fluxR = fluxMap[idx*4+1]; // Out Right
    float fluxT = fluxMap[idx*4+2]; // Out Top
    float fluxB = fluxMap[idx*4+3]; // Out Bottom
    
    // This is approximate velocity. Proper pipe method uses average.
    float avgWater = (oldW + newW) * 0.5f;
    if (avgWater > 0.0001f) {
        float vx = (fluxR - fluxL + fluxMap[(idx-1)*4+1] - fluxMap[(idx+1)*4+0]) * 0.5f; 
        float vy = (fluxB - fluxT + fluxMap[(idx-p.mapWidth)*4+3] - fluxMap[(idx+p.mapWidth)*4+2]) * 0.5f;
        // Notice: This assumes flux is volume/time. 
        
        velocityMap[idx * 2 + 0] = vx;
        velocityMap[idx * 2 + 1] = vy;
    } else {
        velocityMap[idx * 2 + 0] = 0;
        velocityMap[idx * 2 + 1] = 0;
    }
}

// 4. Erosion / Deposition - INDUSTRY STANDARD (Stream Power Law)
// Based on E = K * A^m * S^n (simplified as C = Kc * velocity * slope^0.5)
// Reference: Whipple & Tucker (1999), Howard (1994)
extern "C" __global__ void fluvialErosionKernel(float* heightMap, float* waterMap, float* velocityMap, float* sedimentMap, FluvialErosionParamsGPU p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 1 || x >= p.mapWidth - 2 || y <= 1 || y >= p.mapHeight - 2) return;
    int idx = y * p.mapWidth + x;
    
    float vx = velocityMap[idx*2+0];
    float vy = velocityMap[idx*2+1];
    float velocity = sqrtf(vx*vx + vy*vy);
    
    // Normalized gradient/slope
    float invCellSize = 1.0f / fmaxf(0.001f, p.cellSize);
    float dzdx = (heightMap[idx + 1] - heightMap[idx - 1]) * 0.5f * invCellSize;
    float dzdy = (heightMap[idx + p.mapWidth] - heightMap[idx - p.mapWidth]) * 0.5f * invCellSize;
    float slope = sqrtf(dzdx * dzdx + dzdy * dzdy) * p.heightScale;
    
    // Physical Downhill check
    float dotVGrad = (vx * dzdx + vy * dzdy) * p.heightScale;
    float capacityMultiplier = fmaxf(0.1f, -dotVGrad * invCellSize); 
    
    // Stream Power Law: C = Kc * velocity * slope^n
    // Use slightly higher power for slope to encourage channelization
    float slopeFactor = powf(slope + 0.001f, 0.7f);
    // Capacity in heightmap units (0..1)
    float C = p.sedimentCapacityConstant * velocity * slopeFactor * capacityMultiplier / fmaxf(1.0f, p.heightScale);
    C = fminf(C, 0.2f); // Cap maximum capacity to 20% of height range
    
    // Water depth factor
    float waterDepth = waterMap[idx];
    C *= fminf(1.0f + waterDepth * 0.5f, 2.0f);
    
    float st = sedimentMap[idx];
    float ht = heightMap[idx];
    
    if (C > st) {
        // EROSION
        float erode = p.erosionRate * (C - st) * p.fixedDeltaTime;
        // STABILITY: Bedrock protection (Even more conservative to avoid 'wrong' look)
        erode = fminf(erode, ht * 0.002f); 
        erode = fminf(erode, 0.001f); // Absolute cap per step
        
        heightMap[idx] -= erode;
        sedimentMap[idx] += erode;
    } else {
        // DEPOSITION
        float depositionFactor = 1.0f / (1.0f + velocity * 5.0f); // Higher penalty for velocity
        float depo = p.depositionRate * (st - C) * depositionFactor * p.fixedDeltaTime;
        // STABILITY: Prevent sudden spikes (Max 1% of height gap or small constant)
        depo = fminf(depo, 0.01f);
        
        heightMap[idx] += depo;
        sedimentMap[idx] -= depo;
    }
    
    // Evaporation
    waterMap[idx] *= (1.0f - p.evaporationRate * p.fixedDeltaTime);
}

// 5. Stream Power Law Erosion - Matches CPU "Global Hydrological" Model
// Based on E = K * sqrt(A) * S
extern "C" __global__ void fluvialStreamPowerKernel(float* heightMap, float* flowMap, float* hardnessMap, float* maskMap, StreamPowerParamsGPU p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 2 || x >= p.mapWidth - 3 || y <= 2 || y >= p.mapHeight - 3) return;
    int idx = y * p.mapWidth + x;
    
    // MASK CHECK: Stop execution if masked out
    if (maskMap != nullptr && maskMap[idx] < 0.01f) return;

    float flow = flowMap[idx];
    if (flow < 3.0f) return; // Ignore tiny streams
    
    // Physical slope calculation
    float invCellSize = 1.0f / fmaxf(0.001f, p.cellSize);
    float slopeX = (heightMap[idx + 1] - heightMap[idx - 1]) * 0.5f * invCellSize;
    float slopeY = (heightMap[idx + p.mapWidth] - heightMap[idx - p.mapWidth]) * 0.5f * invCellSize;
    float slope = sqrtf(slopeX * slopeX + slopeY * slopeY);
    
    // Apply minimum slope to ensure flow/erosion on flat plains
    slope = fmaxf(slope, p.minSlope);
    
    // Stream Power Law: E = K * sqrt(A) * S
    float Ks = p.erodeSpeed * 0.02f;
    float streamPower = Ks * sqrtf(flow) * slope;
    
    // Chaos/Turbulence Factor (from CPU version)
    // Helps break "Circuit Board" artifacts in flat areas
    if (slope < 0.05f) {
        float noise = (float)((idx * 1103515245 + 12345) & 0x7FFFFFFF) / 2147483647.0f;
        float flatIntensity = 1.0f - (slope / 0.05f);
        streamPower += Ks * sqrtf(flow) * 0.25f * flatIntensity * (0.7f + noise * 0.6f);
    }
    
    // Dynamic Channel Width
    float hardness = (hardnessMap != nullptr) ? hardnessMap[idx] : 0.3f;
    float flowScale = fminf(1.0f, (flow - 3.0f) / 100.0f);
    float baseRadius = (float)p.erosionRadius * 0.5f;
    float dynamicRadius = fmaxf(0.5f, baseRadius * sqrtf(flowScale) * (1.5f - hardness));
    int channelRadius = (int)ceilf(dynamicRadius);
    if (channelRadius > 10) channelRadius = 10;
    
    float hardnessMultiplier = 1.0f - hardness * 0.7f;
    float flowPower = fminf(1.5f, sqrtf(flow) * 0.2f);
    float slopeDampening = fminf(1.0f, slope * 60.0f); // Slightly steeper dampening
    
    streamPower *= hardnessMultiplier * p.sedimentCapacity * flowPower * slopeDampening;

    // Apply Mask Weight (Soft edges)
    if (maskMap != nullptr) streamPower *= maskMap[idx];
    
    // Apply erosion with circular falloff
    for (int dy = -channelRadius; dy <= channelRadius; dy++) {
        for (int dx = -channelRadius; dx <= channelRadius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= p.mapWidth || ny < 0 || ny >= p.mapHeight) continue;
            
            float distSq = (float)(dx * dx + dy * dy);
            if (distSq > dynamicRadius * dynamicRadius) continue;
            
            int nIdx = ny * p.mapWidth + nx;
            float dist = sqrtf(distSq);
            
            // Smooth circular falloff (Cubic)
            float t = dist / (dynamicRadius + 0.001f);
            float falloff = 1.0f - t * t * (3.0f - 2.0f * t);
            falloff = fmaxf(falloff, 0.0f);
            
            float erode = streamPower * falloff;
            // Protect bedrock (max 10% of height)
            erode = fminf(erode, heightMap[nIdx] * 0.1f);
            
            atomicAdd(&heightMap[nIdx], -erode);
        }
    }
}

// -----------------------------------------------------------------------
// Wind Erosion Kernel - INDUSTRY STANDARD (Shadow Zone + Saltation Model)
// Based on Bagnold's aeolian transport theory with wind shadow detection
// Reference: Bagnold (1941), Werner (1995) dune formation model
// -----------------------------------------------------------------------
// WindErosionParamsGPU is defined in erosion_ops.cuh

extern "C" __global__ void windErosionKernel(float* heightMap, WindErosionParamsGPU p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Bounds check with padding
    if (x <= 5 || x >= p.mapWidth - 6 || y <= 5 || y >= p.mapHeight - 6) return;
    int idx = y * p.mapWidth + x;
    float h = heightMap[idx];
    
    // Use PRNG for wind jitter/stochastic behavior
    curandState state;
    curand_init(1337, idx, 0, &state);
    // Physical slope = dh * heightScale / (dist * cellSize)
    float safeHeightScale = fmaxf(0.1f, p.heightScale);
    float invCellSize = 1.0f / fmaxf(0.01f, p.cellSize);
    float slopeScale = safeHeightScale * invCellSize;
    
    // Add 10% jitter to wind direction for natural feel
    float angleJitter = (curand_uniform(&state) - 0.5f) * 0.2f;
    float windLen = sqrtf(p.windDirX * p.windDirX + p.windDirY * p.windDirY);
    float cosJ = cosf(angleJitter);
    float sinJ = sinf(angleJitter);
    float normWindX = (windLen > 0.001f) ? (p.windDirX * cosJ - p.windDirY * sinJ) / windLen : 1.0f;
    float normWindY = (windLen > 0.001f) ? (p.windDirX * sinJ + p.windDirY * cosJ) / windLen : 0.0f;
    
    // INDUSTRY STANDARD: Resolution-Independent Shadow Zone
    // Shadow angle ~15 degrees means for every 1 unit upwind distance, 
    // the "blocking" height drops by tan(15) ~= 0.26
    float shadowFactor = 1.0f;
    const float SHADOW_TAN = 0.26f;
    
    // Search upwind up to 20 world units or map bounds
    float maxSearchDist = 20.0f; 
    int maxSteps = (int)(maxSearchDist * invCellSize);
    maxSteps = min(maxSteps, 15); // Performance cap for real-time
    
    for (int step = 1; step <= maxSteps; step++) {
        float dist = step * p.cellSize;
        int checkX = x - (int)(normWindX * step);
        int checkY = y - (int)(normWindY * step);
        
        if (checkX < 0 || checkX >= p.mapWidth || checkY < 0 || checkY >= p.mapHeight) break;
        
        float upwindH = heightMap[checkY * p.mapWidth + checkX];
        // Physical check: upwind peak higher than current + physical slope
        if (upwindH * p.heightScale > h * p.heightScale + dist * SHADOW_TAN) {
            shadowFactor *= 0.4f; 
            if (shadowFactor < 0.1f) break;
        }
    }
    
    // Local windward slope (physical)
    int ux = x - (int)normWindX;
    int uy = y - (int)normWindY;
    float upwindH = heightMap[uy * p.mapWidth + ux];
    float slope = (h - upwindH) * slopeScale;
    
    if (slope > 0.0f && shadowFactor > 0.5f) {
        // EROSION: Exposed windward face
        // Strength scales with slope (steeper = more exposed to wind force)
        float erosion = slope * p.strength * p.suspensionRate * shadowFactor;
        atomicAdd(&heightMap[idx], -erosion);
        
        // SALTATION: Material lands downwind
        // Target distance: jump depends on wind strength, but normalized to world space
        float jumpDistWorld = 4.0f * p.strength; // Base jump of 4m at full strength
        int jumpSteps = (int)(jumpDistWorld * invCellSize);
        jumpSteps = max(1, min(jumpSteps, 10));
        
        int dx_jump = x + (int)(normWindX * jumpSteps);
        int dy_jump = y + (int)(normWindY * jumpSteps);
        
        if (dx_jump > 0 && dx_jump < p.mapWidth && dy_jump > 0 && dy_jump < p.mapHeight) {
            atomicAdd(&heightMap[dy_jump * p.mapWidth + dx_jump], erosion * p.depositionRate);
        }
    } else if (shadowFactor < 0.7f) {
        // DEPOSITION: Settling in sheltered area (Wind Shadow)
        float settling = (1.0f - shadowFactor) * p.strength * 0.05f * p.depositionRate;
        atomicAdd(&heightMap[idx], settling);
    }
}

__device__ void updateHeight(float* map, int width, int height, float x, float y, float change) {
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    
    if (x0 < 0 || x0 >= width || y0 < 0 || y0 >= height) return;
    if (x1 < 0 || x1 >= width || y1 < 0 || y1 >= height) return;
    
    float tx = x - x0;
    float ty = y - y0;
    
    // Atomics are fast enough if not looped 100 times
    atomicAdd(&map[y0 * width + x0], change * (1 - tx) * (1 - ty));
    atomicAdd(&map[y0 * width + x1], change * tx * (1 - ty));
    atomicAdd(&map[y1 * width + x0], change * (1 - tx) * ty);
    atomicAdd(&map[y1 * width + x1], change * tx * ty);
}

// -----------------------------------------------------------------------
// Smoothing Kernel (Fixes Spikes in O(1) pass)
// -----------------------------------------------------------------------
extern "C" __global__ void smoothTerrainKernel(float* heightmap, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Simple 3x3 Average (Box Blur)
    // To do this strictly correctly we need a second buffer (ping-pong),
    // but for terrain smoothing, in-place is usually "okay enough" or we can just accept slight bias.
    // Ideally user provides a temporary buffer. 
    // For now, let's do a very soft read-modify-write which might race but removes spikes.
    // Or better: Read neighbors, compute avg, write self. 
    // Since we only write self, it's safeish (only race is reading neighbors being written by others).
    
    float sum = 0;
    float weightSum = 0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
             int nx = x + dx;
             int ny = y + dy;
             
             if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                 float w = 1.0f; 
                 // Center pixel usually gets more weight?
                 if (dx == 0 && dy == 0) w = 2.0f; 
                 
                 sum += heightmap[ny * width + nx] * w;
                 weightSum += w;
             }
        }
    }
    
    // Write back
    if (weightSum > 0) {
        heightmap[y * width + x] = sum / weightSum;
    }
}

// -----------------------------------------------------------------------
// Pit Filling Kernel - Eliminates micro-ponds that cause black triangles
// Matches CPU behavior in hydraulicErosion post-processing
// -----------------------------------------------------------------------
extern "C" __global__ void pitFillingKernel(float* heightmap, PostProcessParamsGPU p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Skip edges
    if (x <= 0 || x >= p.mapWidth - 1 || y <= 0 || y >= p.mapHeight - 1) return;
    
    int idx = y * p.mapWidth + x;
    float h = heightmap[idx];
    
    // Get 4 cardinal neighbors
    float hL = heightmap[idx - 1];
    float hR = heightmap[idx + 1];
    float hU = heightmap[idx - p.mapWidth];
    float hD = heightmap[idx + p.mapWidth];
    
    float minNeighbor = fminf(fminf(hL, hR), fminf(hU, hD));
    float avgNeighbor = (hL + hR + hU + hD) * 0.25f;
    
    // If significantly lower than all neighbors (pit), fill it up
    if (h < minNeighbor - p.pitThreshold) {
        // Fill to blend of average and current (matches CPU: 0.5f blend)
        heightmap[idx] = avgNeighbor * 0.5f + h * 0.5f;
    }
}

// -----------------------------------------------------------------------
// Spike Removal Kernel - Removes isolated peaks/spikes
// Matches CPU behavior in hydraulicErosion post-processing
// -----------------------------------------------------------------------
extern "C" __global__ void spikeRemovalKernel(float* heightmap, PostProcessParamsGPU p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Skip edges
    if (x <= 0 || x >= p.mapWidth - 1 || y <= 0 || y >= p.mapHeight - 1) return;
    
    int idx = y * p.mapWidth + x;
    float h = heightmap[idx];
    
    // Get 4 cardinal neighbors
    float hL = heightmap[idx - 1];
    float hR = heightmap[idx + 1];
    float hU = heightmap[idx - p.mapWidth];
    float hD = heightmap[idx + p.mapWidth];
    
    float avgNeighbor = (hL + hR + hU + hD) * 0.25f;
    
    // If significantly higher than neighbors (spike), smooth it down
    if (h > avgNeighbor + p.spikeThreshold) {
        // Blend down (matches CPU: 0.3f average, 0.7f original)
        heightmap[idx] = avgNeighbor * 0.3f + h * 0.7f;
    }
}

// -----------------------------------------------------------------------
// Edge Preservation Kernel - Prevents "wall" effect at terrain boundaries
// Matches CPU behavior in hydraulicErosion edge fade-out
// -----------------------------------------------------------------------
extern "C" __global__ void edgePreservationKernel(float* heightmap, float* originalHeights, PostProcessParamsGPU p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= p.mapWidth || y >= p.mapHeight) return;
    
    int idx = y * p.mapWidth + x;
    
    // Calculate distance from nearest edge
    int distFromEdge = x;
    if (y < distFromEdge) distFromEdge = y;
    if (p.mapWidth - 1 - x < distFromEdge) distFromEdge = p.mapWidth - 1 - x;
    if (p.mapHeight - 1 - y < distFromEdge) distFromEdge = p.mapHeight - 1 - y;
    
    if (distFromEdge < p.edgeFadeWidth) {
        // Smoothstep interpolation
        float t = (float)distFromEdge / (float)p.edgeFadeWidth;
        float smoothT = t * t * (3.0f - 2.0f * t);
        
        // Find inward reference point
        int inwardX = x;
        int inwardY = y;
        
        if (x < p.mapWidth / 2) inwardX = x + p.edgeFadeWidth;
        else inwardX = x - p.edgeFadeWidth;
        
        if (y < p.mapHeight / 2) inwardY = y + p.edgeFadeWidth;
        else inwardY = y - p.edgeFadeWidth;
        
        // Clamp
        inwardX = max(0, min(p.mapWidth - 1, inwardX));
        inwardY = max(0, min(p.mapHeight - 1, inwardY));
        
        int inwardIdx = inwardY * p.mapWidth + inwardX;
        float inwardHeight = heightmap[inwardIdx];
        
        // Blend current with inward reference
        heightmap[idx] = heightmap[idx] * smoothT + inwardHeight * 0.4f * (1.0f - smoothT);
    }
}

// -----------------------------------------------------------------------
// Thermal Erosion with Hardness Support - INDUSTRY STANDARD (Multi-Neighbor)
// -----------------------------------------------------------------------
extern "C" __global__ void thermalErosionWithHardnessKernel(float* heightmap, float* hardnessMap, ThermalErosionParamsGPU p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (x <= 0 || x >= p.mapWidth - 1 || y <= 0 || y >= p.mapHeight - 1) return;
    
    int idx = y * p.mapWidth + x;
    float h = heightmap[idx];
    
    // Get hardness at this location (0 = soft, 1 = hard)
    float hardness = (hardnessMap != nullptr) ? hardnessMap[idx] : 0.0f;
    
    // Hardness modifies effective talus angle
    float effectiveTalus = p.talusAngle + hardness * 0.3f;
    
    // D8 neighbor offsets and distance weights
    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float dists[8] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};

    float diffs[8];
    int nIdxs[8];
    float totalExcess = 0.0f;
    int neighborCount = 0;
    
    // Physical slope = dh * heightScale / (dist * cellSize)
    float safeHeightScale = fmaxf(0.1f, p.heightScale);
    float invCellSize = 1.0f / fmaxf(0.01f, p.cellSize);
    float slopeScale = safeHeightScale * invCellSize;
    
    for (int d = 0; d < 8; d++) {
        nIdxs[d] = (y + dy[d]) * p.mapWidth + (x + dx[d]);
        
        // Physical slope = dh * heightScale / (dist * cellSize)
        float diff = h - heightmap[nIdxs[d]];
        float slope = diff * slopeScale / dists[d];
        
        if (slope > effectiveTalus) {
            float excess = (slope - effectiveTalus) * dists[d] * p.cellSize / safeHeightScale;
            diffs[d] = excess;
            totalExcess += excess;
            neighborCount++;
        } else {
            diffs[d] = 0.0f;
        }
    }
    
    // Distribute material proportionally to all steep neighbors
    if (totalExcess > 0.0f && neighborCount > 0) {
        float totalMove = totalExcess * 0.5f * p.erosionAmount;
        
        // STABILITY CAP: Never move more than 5% of height per step
        totalMove = fminf(totalMove, 0.05f);
        if (totalMove > h) totalMove = h * 0.9f;
        
        // Hardness reduces erosion rate
        totalMove *= (1.0f - hardness * 0.7f);
        
        // Remove from center
        atomicAdd(&heightmap[idx], -totalMove);
        
        // Distribute to neighbors proportionally
        for (int d = 0; d < 8; d++) {
            if (diffs[d] > 0.0f) {
                float weight = diffs[d] / totalExcess;
                atomicAdd(&heightmap[nIdxs[d]], totalMove * weight);
            }
        }
    }
}

// -----------------------------------------------------------------------
// Main Erosion Kernel
// -----------------------------------------------------------------------
extern "C" __global__ void hydraulicErosionKernel(float* heightmap, HydraulicErosionParamsGPU p, unsigned long long seed_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState state;
    curand_init(1234 + seed_offset + p.seed, idx, 0, &state);
    
    // Random start position
    float posX = curand_uniform(&state) * (p.mapWidth - 1);
    float posY = curand_uniform(&state) * (p.mapHeight - 1);
    
    float dirX = 0.0f, dirY = 0.0f;
    float speed = 1.0f;
    float water = 1.0f;
    float sediment = 0.0f;
    
    float invCellSize = 1.0f / fmaxf(0.001f, p.cellSize);
    
    // Droplet simulation loop
    for (int step = 0; step < p.dropletLifetime; step++) {
        int nodeX = (int)posX;
        int nodeY = (int)posY;
        
        // Bounds check (with padding for gradient)
        if (nodeX <= 1 || nodeX >= p.mapWidth - 2 || nodeY <= 1 || nodeY >= p.mapHeight - 2) break;
        
        int gridIdx = nodeY * p.mapWidth + nodeX;
        
        // Normalize gradient by cellSize to ensure consistent force across resolutions
        // Using central difference for better stability
        float gradX = (heightmap[gridIdx + 1] - heightmap[gridIdx - 1]) * 0.5f * invCellSize;
        float gradY = (heightmap[gridIdx + p.mapWidth] - heightmap[gridIdx - p.mapWidth]) * 0.5f * invCellSize;
        
        // Update direction with inertia
        dirX = dirX * p.inertia - gradX * (1.0f - p.inertia);
        dirY = dirY * p.inertia - gradY * (1.0f - p.inertia);
        
        float len = sqrtf(dirX*dirX + dirY*dirY);
        if (len < 0.0001f) {
            dirX = curand_uniform(&state) * 2.0f - 1.0f;
            dirY = curand_uniform(&state) * 2.0f - 1.0f;
            len = sqrtf(dirX*dirX + dirY*dirY);
        }
        dirX /= len; dirY /= len;
        
        float newPosX = posX + dirX;
        float newPosY = posY + dirY;
        
        if (newPosX <= 1.0f || newPosX >= p.mapWidth - 2.0f || newPosY <= 1.0f || newPosY >= p.mapHeight - 2.0f) break;
        
        // Height sampling
        float oldH = bilinearInterpolate(heightmap, p.mapWidth, p.mapHeight, posX, posY);
        float newH = bilinearInterpolate(heightmap, p.mapWidth, p.mapHeight, newPosX, newPosY);
        float deltaHeight = newH - oldH;
        
        // Physical slope calculation
        float localSlope = -deltaHeight * p.heightScale * invCellSize; // Physical slope
        float capacity = fmaxf(localSlope * speed * water * p.sedimentCapacity * sqrtf(fmaxf(0.0f, localSlope) + 0.001f), p.minSlope);
        
        // Convert capacity back to normalized units if necessary? 
        // No, p.sedimentCapacity should be tuned for the physical slope.
        // BUT deltaHeight is in 0..1 units. Capacity must match sediment units.
        // If sediment is picked up in 0..1 units, capacity should also be in 0..1 units.
        capacity /= fmaxf(1.0f, p.heightScale);
        
        if (deltaHeight > 0.0f) {
            // UPHILL - MOMENTUM CHECK
            if (speed > 0.5f && sediment < capacity) {
                // High momentum: carve through obstacle (creates channels)
                // STABILITY: Cap at 30% of height diff and 5% of absolute height
                float erodeAmount = fminf(deltaHeight * 0.3f, oldH * 0.05f);
                updateHeight(heightmap, p.mapWidth, p.mapHeight, posX, posY, -erodeAmount);
                sediment += erodeAmount;
                speed *= 0.6f; // Significant penalty for climbing
            } else {
                // Low momentum: deposit and redirect/stop
                // STABILITY: Never deposit more than half of sediment or height diff in one step
                float depoAmount = fminf(sediment * 0.3f, deltaHeight * 0.5f);
                updateHeight(heightmap, p.mapWidth, p.mapHeight, posX, posY, depoAmount);
                sediment -= depoAmount;
                speed = 0.0f; // Droplet stops at the wall
            }
        } else if (sediment > capacity) {
            // DOWNHILL - DEPOSITION (Droplet overloaded)
            float depoAmount = (sediment - capacity) * p.depositSpeed;
            // STABILITY: Prevent massive spikes
            depoAmount = fminf(depoAmount, -deltaHeight * 0.5f);
            updateHeight(heightmap, p.mapWidth, p.mapHeight, posX, posY, depoAmount);
            sediment -= depoAmount;
        } else {
            // DOWNHILL - EROSION (Droplet hungry)
            float erodeAmount = fminf((capacity - sediment) * p.erodeSpeed, -deltaHeight);
            // STABILITY: Prevent deep pits (max 5% of local height)
            erodeAmount = fminf(erodeAmount, oldH * 0.05f);
            updateHeight(heightmap, p.mapWidth, p.mapHeight, posX, posY, -erodeAmount);
            sediment += erodeAmount;
        }
        
        // Physics update (Correct gravity logic)
        // Kinetic Energy: v_new^2 = v_old^2 + g * (h_old - h_new) = v_old^2 - g * deltaHeight
        speed = sqrtf(fmaxf(0.001f, speed*speed - deltaHeight * p.gravity));
        water *= (1.0f - p.evaporateSpeed);
        posX = newPosX; 
        posY = newPosY;
        
        if (water < 0.01f || speed < 0.01f) break;
    }
}
