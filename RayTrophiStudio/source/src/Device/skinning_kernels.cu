#include "../include/skinning_kernels.cuh"
#include <device_launch_parameters.h>

// Helper: Transform Point by 4x4 Matrix (Row-Major)
// x' = m00*x + m01*y + m02*z + m03
__device__ inline float3 transform_point(const float* mat, const float3& p) {
    float3 r;
    r.x = mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3];
    r.y = mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7];
    r.z = mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11];
    return r;
}

// Helper: Transform Vector by 3x3 Upper-Left Matrix (Rotation/Scale only)
// Used for Normals (approximation for real-time skinning, assumes uniform scale/orthogonal)
__device__ inline float3 transform_vector(const float* mat, const float3& v) {
    float3 r;
    r.x = mat[0] * v.x + mat[1] * v.y + mat[2] * v.z;
    r.y = mat[4] * v.x + mat[5] * v.y + mat[6] * v.z;
    r.z = mat[8] * v.x + mat[9] * v.y + mat[10] * v.z;
    return r;
}

__global__ void skinningKernel(
    const float3* __restrict__ inputPositions,
    const float3* __restrict__ inputNormals,
    const int4* __restrict__ boneIndices,
    const float4* __restrict__ boneWeights,
    const float* __restrict__ boneMatrices,
    float3* __restrict__ outputPositions,
    float3* __restrict__ outputNormals,
    int numVertices,
    int numBones
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    // Load vertex data
    float3 inPos = inputPositions[idx];
    float3 inNorm = (inputNormals != nullptr) ? inputNormals[idx] : make_float3(0.0f, 1.0f, 0.0f);
    int4 indices = boneIndices[idx];
    float4 weights = boneWeights[idx];

    // Accumulators
    float3 finalPos = { 0.0f, 0.0f, 0.0f };
    float3 finalNorm = { 0.0f, 0.0f, 0.0f };
    float totalWeight = 0.0f;

    // Process 4 bones manually (unrolled)
    // Structure of access: indices.x, y, z, w
    int boneIDs[4] = { indices.x, indices.y, indices.z, indices.w };
    float boneWts[4] = { weights.x, weights.y, weights.z, weights.w };

    // First pass: check total weight
    for (int i = 0; i < 4; i++) {
        if (boneIDs[i] >= 0 && boneIDs[i] < numBones) {
            totalWeight += boneWts[i];
        }
    }

    if (totalWeight < 1e-6f) {
        // Fallback: If no valid weights, keep original position
        finalPos = inPos;
        finalNorm = inNorm;
    } else {
        // Second pass: apply skinning with normalization
        float invTotalWeight = 1.0f / totalWeight;

        for (int i = 0; i < 4; i++) {
            int boneID = boneIDs[i];
            float w = boneWts[i] * invTotalWeight; // Use normalized weight (most significant 4 bones)

            if (boneID >= 0 && boneID < numBones && w > 1e-7f) {
                const float* mat = &boneMatrices[boneID * 16];

                // Linear Blend Skinning:
                // v' = sum(w_i * (M_i * v))
                float3 tPos = transform_point(mat, inPos);
                
                finalPos.x += tPos.x * w;
                finalPos.y += tPos.y * w;
                finalPos.z += tPos.z * w;

                // Normal transformation (approximation)
                if (inputNormals != nullptr) {
                    float3 tNorm = transform_vector(mat, inNorm);
                    finalNorm.x += tNorm.x * w;
                    finalNorm.y += tNorm.y * w;
                    finalNorm.z += tNorm.z * w;
                }
            }
        }
    }

    // Normalize normal
    float lenSq = finalNorm.x * finalNorm.x + finalNorm.y * finalNorm.y + finalNorm.z * finalNorm.z;
    if (lenSq > 1e-12f) {
        float invLen = rsqrtf(lenSq);
        finalNorm.x *= invLen;
        finalNorm.y *= invLen;
        finalNorm.z *= invLen;
    }

    // Write output
    outputPositions[idx] = finalPos;
    outputNormals[idx] = finalNorm;
}

extern "C" void launchSkinningKernel(
    const float3* inputPositions,
    const float3* inputNormals,
    const int4* boneIndices,
    const float4* boneWeights,
    const float* boneMatrices,
    float3* outputPositions,
    float3* outputNormals,
    int numVertices,
    int numBones,
    cudaStream_t stream
) {
    int blockSize = 256;
    int numBlocks = (numVertices + blockSize - 1) / blockSize;

    skinningKernel<<<numBlocks, blockSize, 0, stream>>>(
        inputPositions,
        inputNormals,
        boneIndices,
        boneWeights,
        boneMatrices,
        outputPositions,
        outputNormals,
        numVertices,
        numBones
    );
}
