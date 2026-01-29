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
    
    // INDUSTRY STANDARD: Weighted multi-neighbor transfer
    // Instead of transferring only to steepest neighbor, distribute proportionally
    // to ALL neighbors exceeding talus angle (more realistic, smoother results)
    
    float diffs[8];
    int nIdxs[8];
    float totalExcess = 0.0f;
    int neighborCount = 0;
    
    // D8 neighbor offsets (dx, dy) and distance weights
    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float distWeight[8] = {0.707f, 1.0f, 0.707f, 1.0f, 1.0f, 0.707f, 1.0f, 0.707f}; // 1/sqrt(2) for diagonals
    
    // First pass: Calculate all height differences and total excess
    for (int d = 0; d < 8; d++) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        nIdxs[d] = ny * p.mapWidth + nx;
        
        // Adjust talus for diagonal distance (diagonals can be steeper)
        float adjustedTalus = p.talusAngle * distWeight[d];
        float diff = h - heightmap[nIdxs[d]];
        
        if (diff > adjustedTalus) {
            diffs[d] = diff - adjustedTalus;
            totalExcess += diffs[d];
            neighborCount++;
        } else {
            diffs[d] = 0.0f;
        }
    }
    
    // Second pass: Distribute material proportionally to all steep neighbors
    if (totalExcess > 0.0f && neighborCount > 0) {
        // Total amount to move (half of excess, scaled by erosion rate)
        float totalMove = totalExcess * 0.5f * p.erosionAmount;
        
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
    
    // Calculate Height Diffs
    float dH[4];
    dH[0] = H - (heightMap[nIdx[0]] + waterMap[nIdx[0]]);
    dH[1] = H - (heightMap[nIdx[1]] + waterMap[nIdx[1]]);
    dH[2] = H - (heightMap[nIdx[2]] + waterMap[nIdx[2]]);
    dH[3] = H - (heightMap[nIdx[3]] + waterMap[nIdx[3]]);
    
    // Update Flux (Pipe Model)
    float pipeArea = p.pipeLength * p.pipeLength; // Section area? usually just 1
    float fluxFactor = p.fixedDeltaTime * pipeArea * p.gravity;
    
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
    
    // INDUSTRY STANDARD: Stream Power Law with slope term
    // Calculate local slope from height gradient
    float dzdx = (heightMap[idx + 1] - heightMap[idx - 1]) / (2.0f * p.cellSize);
    float dzdy = (heightMap[idx + p.mapWidth] - heightMap[idx - p.mapWidth]) / (2.0f * p.cellSize);
    float slope = sqrtf(dzdx * dzdx + dzdy * dzdy);
    
    // Stream Power Law: C = Kc * velocity * slope^n
    // Using n = 0.5 (common for fluvial systems)
    // Add small epsilon to prevent zero capacity on flat areas
    float slopeFactor = sqrtf(slope + 0.001f);
    float C = p.sedimentCapacityConstant * velocity * slopeFactor;
    
    // Water depth factor - deeper water can carry more sediment
    float waterDepth = waterMap[idx];
    C *= fminf(1.0f + waterDepth * 0.5f, 2.0f);
    
    float st = sedimentMap[idx];
    float ht = heightMap[idx];
    
    if (C > st) {
        // Erode (Pick up sediment) - Stream Power Erosion
        float erode = p.erosionRate * (C - st);
        
        // Bedrock protection: can't erode more than available height
        erode = fminf(erode, ht * 0.1f); // Max 10% per step for stability
        
        // Slope-dependent erosion boost (steeper = more erosion)
        erode *= (1.0f + slope * 0.5f);
        
        heightMap[idx] -= erode;
        sedimentMap[idx] += erode;
    } else {
        // Deposit - preferentially in low-velocity, low-slope areas
        float depositionFactor = 1.0f / (1.0f + velocity * 2.0f); // Slower = more deposition
        float depo = p.depositionRate * (st - C) * depositionFactor;
        heightMap[idx] += depo;
        sedimentMap[idx] -= depo;
    }
    
    // Evaporation
    waterMap[idx] *= (1.0f - p.evaporationRate * p.fixedDeltaTime);
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
    
    if (x <= 2 || x >= p.mapWidth - 3 || y <= 2 || y >= p.mapHeight - 3) return;
    int idx = y * p.mapWidth + x;
    
    float h = heightMap[idx];
    
    // Normalize wind direction
    float windLen = sqrtf(p.windDirX * p.windDirX + p.windDirY * p.windDirY);
    float normWindX = (windLen > 0.001f) ? p.windDirX / windLen : 1.0f;
    float normWindY = (windLen > 0.001f) ? p.windDirY / windLen : 0.0f;
    
    // INDUSTRY STANDARD: Shadow Zone Detection
    // Ray march upwind to detect if we're in a wind shadow
    float shadowFactor = 1.0f;
    const int SHADOW_STEPS = 6;
    const float SHADOW_ANGLE = 0.15f; // ~8.5 degrees shadow angle (typical for sand)
    
    for (int step = 1; step <= SHADOW_STEPS; step++) {
        int checkX = x - (int)(normWindX * step);
        int checkY = y - (int)(normWindY * step);
        
        // Clamp to bounds
        checkX = max(0, min(p.mapWidth - 1, checkX));
        checkY = max(0, min(p.mapHeight - 1, checkY));
        
        int checkIdx = checkY * p.mapWidth + checkX;
        float upwindH = heightMap[checkIdx];
        
        // Check if upwind point casts shadow over current point
        // Shadow extends at SHADOW_ANGLE below upwind peak
        float shadowHeight = upwindH - step * SHADOW_ANGLE;
        if (h < shadowHeight) {
            // We're in the wind shadow - reduce erosion, increase deposition
            shadowFactor *= 0.3f;
        }
    }
    
    // Calculate local windward slope
    int upwindX = x - (int)normWindX;
    int upwindY = y - (int)normWindY;
    upwindX = max(0, min(p.mapWidth - 1, upwindX));
    upwindY = max(0, min(p.mapHeight - 1, upwindY));
    int upwindIdx = upwindY * p.mapWidth + upwindX;
    
    int downwindX = x + (int)normWindX;
    int downwindY = y + (int)normWindY;
    downwindX = max(0, min(p.mapWidth - 1, downwindX));
    downwindY = max(0, min(p.mapHeight - 1, downwindY));
    int downwindIdx = downwindY * p.mapWidth + downwindX;
    
    // Further downwind for saltation deposition
    int farDownwindX = x + (int)(normWindX * 3);
    int farDownwindY = y + (int)(normWindY * 3);
    farDownwindX = max(0, min(p.mapWidth - 1, farDownwindX));
    farDownwindY = max(0, min(p.mapHeight - 1, farDownwindY));
    int farDownwindIdx = farDownwindY * p.mapWidth + farDownwindX;
    
    float upwindH = heightMap[upwindIdx];
    float downwindH = heightMap[downwindIdx];
    
    // Windward slope - exposed to erosion
    float windwardSlope = h - upwindH;
    
    // Leeward slope - sheltered, deposition zone
    float leewardSlope = h - downwindH;
    
    // EROSION: Windward faces (facing into wind)
    if (windwardSlope > 0.0f && shadowFactor > 0.5f) {
        // Saltation erosion - stronger on exposed windward slopes
        float erosionAmount = windwardSlope * p.suspensionRate * p.strength * shadowFactor;
        
        // Abrasion increases with slope angle
        float abrasionBoost = 1.0f + windwardSlope * 2.0f;
        erosionAmount *= abrasionBoost;
        
        atomicAdd(&heightMap[idx], -erosionAmount);
        
        // Saltation: material jumps and lands downwind
        // Split between immediate downwind and further saltation jump
        atomicAdd(&heightMap[downwindIdx], erosionAmount * p.depositionRate * 0.6f);
        atomicAdd(&heightMap[farDownwindIdx], erosionAmount * p.depositionRate * 0.4f);
    }
    
    // DEPOSITION: Leeward faces (wind shadow) and shadow zones
    if (shadowFactor < 0.7f || leewardSlope > 0.0f) {
        // In shadow zone or on leeward slope - sediment settles
        // Capture some passing sediment based on how sheltered we are
        float depositionAmount = (1.0f - shadowFactor) * p.strength * 0.02f;
        atomicAdd(&heightMap[idx], depositionAmount);
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
    const float distWeight[8] = {0.707f, 1.0f, 0.707f, 1.0f, 1.0f, 0.707f, 1.0f, 0.707f};
    
    // Calculate all height differences and total excess
    float diffs[8];
    int nIdxs[8];
    float totalExcess = 0.0f;
    int neighborCount = 0;
    
    for (int d = 0; d < 8; d++) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        nIdxs[d] = ny * p.mapWidth + nx;
        
        float adjustedTalus = effectiveTalus * distWeight[d];
        float diff = h - heightmap[nIdxs[d]];
        
        if (diff > adjustedTalus) {
            diffs[d] = diff - adjustedTalus;
            totalExcess += diffs[d];
            neighborCount++;
        } else {
            diffs[d] = 0.0f;
        }
    }
    
    // Distribute material proportionally to all steep neighbors
    if (totalExcess > 0.0f && neighborCount > 0) {
        float totalMove = totalExcess * 0.5f * p.erosionAmount;
        
        // Hardness reduces erosion rate
        totalMove *= (1.0f - hardness * 0.4f);
        
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
    
    float posX = curand_uniform(&state) * (p.mapWidth - 1);
    float posY = curand_uniform(&state) * (p.mapHeight - 1);
    
    float dirX = 0, dirY = 0;
    float speed = 1.0f;
    float water = 1.0f;
    float sediment = 0.0f;
    
    for (int step = 0; step < p.dropletLifetime; step++) {
        int nodeX = (int)posX;
        int nodeY = (int)posY;
        
        if (nodeX <= 0 || nodeX >= p.mapWidth - 1 || nodeY <= 0 || nodeY >= p.mapHeight - 1) break;
        
        int gridIdx = nodeY * p.mapWidth + nodeX;
        // float height = heightmap[gridIdx]; // Unused raw read? Use interp or raw for Gradient?
        // Gradient
        float gradX = (heightmap[gridIdx + 1] - heightmap[gridIdx - 1]);
        float gradY = (heightmap[gridIdx + p.mapWidth] - heightmap[gridIdx - p.mapWidth]);
        
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
        
        float oldH = bilinearInterpolate(heightmap, p.mapWidth, p.mapHeight, posX, posY);
        float newH = bilinearInterpolate(heightmap, p.mapWidth, p.mapHeight, newPosX, newPosY);
        float deltaHeight = newH - oldH;
        
        // INDUSTRY STANDARD: Stream Power Law sediment capacity
        // C = Kc * velocity * slope^n * water (n typically 0.5-1.0)
        // Reference: Whipple & Tucker (1999)
        float slope = fabsf(deltaHeight); // Local slope approximation
        float slopeFactor = sqrtf(slope + 0.001f); // slope^0.5 with epsilon
        float sedimentCapacity = fmaxf(-deltaHeight * speed * water * p.sedimentCapacity * slopeFactor, p.minSlope);
        
        if (deltaHeight > 0) {
            // Uphill: Carve if fast
             if (speed > 0.5f && sediment < sedimentCapacity) {
                 float carve = fminf(deltaHeight * 0.5f, oldH * 0.05f); 
                 // Use simple updateHeight (bilinear) - FAST
                 updateHeight(heightmap, p.mapWidth, p.mapHeight, posX, posY, -carve);
                 sediment += carve;
             } else {
                 float depo = fminf(deltaHeight, sediment);
                 sediment -= depo;
                 updateHeight(heightmap, p.mapWidth, p.mapHeight, posX, posY, depo);
             }
        } else if (sediment > sedimentCapacity) {
            float depo = (sediment - sedimentCapacity) * p.depositSpeed;
            sediment -= depo;
            updateHeight(heightmap, p.mapWidth, p.mapHeight, posX, posY, depo);
        } else {
            float erode = fminf((sedimentCapacity - sediment) * p.erodeSpeed, -deltaHeight);
            updateHeight(heightmap, p.mapWidth, p.mapHeight, posX, posY, -erode);
            sediment += erode;
        }
        
        speed = sqrtf(speed*speed + deltaHeight * p.gravity);
        water *= (1.0f - p.evaporateSpeed);
        posX = newPosX; posY = newPosY;
        if (water < 0.001f) break;
    }
}
