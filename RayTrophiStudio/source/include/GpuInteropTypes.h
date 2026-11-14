#pragma once
// ===========================================================================
// GpuInteropTypes.h - the ONE place first-party headers get CUDA vector types.
//
// The GPU-shared structs (GpuMaterial, HitGroupData, WorldData, OptixGeometryData)
// are laid out with CUDA's float3 / cudaTextureObject_t because the OptiX device
// programs read the same bytes. Those headers used to include <cuda_runtime.h>
// directly, which made the whole CUDA SDK a compile dependency of ~300
// translation units that never call CUDA.
//
// Studio build (default): this header is exactly <cuda_runtime.h> +
// <vector_types.h>. Nothing changes.
//
// Player build (RT_NO_CUDA defined): no CUDA SDK. The same names are provided
// as plain structs with the host-side layout CUDA uses on x64, so the shared
// structs still compile. There is no OptiX in that build, so nothing reads the
// bytes across an ABI. Size/alignment assertions below guard the stand-ins;
// actual GPU ABI behavior still requires runtime validation in the SDK build.
//
// Do not add `#ifdef RT_NO_CUDA` anywhere else. A header that needs a CUDA
// *function* (cudaMalloc, cudaMemcpy...) does not belong in the Player set;
// move that code into an offline-only .cpp instead.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md
// ===========================================================================

#if !defined(RT_NO_CUDA) || defined(__CUDACC__)

#include <cuda_runtime.h>
#include <vector_types.h>

#else

#ifndef __align__
#define __align__(n) __declspec(align(n))
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ __forceinline
#endif

// Host-side (non-__CUDACC__, x64) layouts from CUDA's vector_types.h:
// 2-component 32-bit vectors align to 8 bytes, 4-component ones to 16.
// CUDA uchar4 aligns to 4 bytes; 3-component vectors use natural alignment.
struct __align__(8) float2 { float x, y; };
struct float3 { float x, y, z; };
struct alignas(16) float4 { float x, y, z, w; };
struct __align__(8) int2 { int x, y; };
struct int3 { int x, y, z; };
struct alignas(16) int4 { int x, y, z, w; };
struct __align__(8) uint2 { unsigned int x, y; };
struct uint3 { unsigned int x, y, z; };
struct alignas(16) uint4 { unsigned int x, y, z, w; };
struct alignas(4) uchar4 { unsigned char x, y, z, w; };

typedef unsigned long long cudaTextureObject_t;
// Opaque handle only: same declaration as CUDA's driver_types.h.
struct cudaArray;
typedef struct cudaArray* cudaArray_t;

static inline float2 make_float2(float x, float y) {
    float2 v;
    v.x = x;
    v.y = y;
    return v;
}
static inline float3 make_float3(float x, float y, float z) {
    float3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}
static inline float4 make_float4(float x, float y, float z, float w) {
    float4 v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
}
static inline int2 make_int2(int x, int y) {
    int2 v;
    v.x = x;
    v.y = y;
    return v;
}
static inline int3 make_int3(int x, int y, int z) {
    int3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}
static inline int4 make_int4(int x, int y, int z, int w) {
    int4 v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
}
static inline uint2 make_uint2(unsigned int x, unsigned int y) {
    uint2 v;
    v.x = x;
    v.y = y;
    return v;
}
static inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) {
    uint3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}
static inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
    uint4 v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
}
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) {
    uchar4 v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
}

static_assert(sizeof(float2) == 8 && sizeof(float3) == 12 && sizeof(float4) == 16,
              "GpuInteropTypes: stand-in vector sizes drifted");
static_assert(alignof(float2) == 8 && alignof(float3) == 4 && alignof(float4) == 16,
              "GpuInteropTypes: float vector alignment drifted");
static_assert(sizeof(int2) == 8 && sizeof(int3) == 12 && sizeof(int4) == 16 &&
              alignof(int2) == 8 && alignof(int3) == 4 && alignof(int4) == 16,
              "GpuInteropTypes: int vector layout drifted");
static_assert(sizeof(uint2) == 8 && sizeof(uint3) == 12 && sizeof(uint4) == 16 &&
              alignof(uint2) == 8 && alignof(uint3) == 4 && alignof(uint4) == 16,
              "GpuInteropTypes: uint vector layout drifted");
static_assert(sizeof(uchar4) == 4 && alignof(uchar4) == 4,
              "GpuInteropTypes: byte vector layout drifted");
static_assert(sizeof(cudaTextureObject_t) == 8, "GpuInteropTypes: texture handle must be 64-bit");

#endif
