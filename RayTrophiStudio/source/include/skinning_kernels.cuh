#ifndef SKINNING_KERNELS_CUH
#define SKINNING_KERNELS_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch the skinning kernel
void launchSkinningKernel(
    const float3* vertices,
    const float3* normals,
    const int4* boneIndices,
    const float4* boneWeights,
    const float* boneMatrices,
    float3* outVertices,
    float3* outNormals,
    int vertexCount,
    int numBones,
    cudaStream_t stream = 0
);

#ifdef __cplusplus
}
#endif

#endif // SKINNING_KERNELS_CUH
