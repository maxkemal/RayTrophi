//
// StylizeKernel.cu — GPU stylize pass. Runs StylizeCore::applyPostProcess (the
// exact same code the CPU path compiles) on the OptiX backend's device AOV
// buffers + the uploaded (already-graded) display color. See StylizeKernel.h.
//
// Bire-bir notes (must match applyStylizeToSurfaceWithCamera in Main.cpp):
//   * Works in SURFACE space. For surface pixel (x, y) it reads d_color[y*w+x],
//     decodes RGB with the surface masks/shifts, samples the AOV at buffer
//     coords (x, height-1-y), takes edge neighbours at (x+1, height-1-y) and
//     (x, height-1-y+1), and passes noise coords (x, y) to applyPostProcess.
//   * AOV decode mirrors downloadDenoiserBuffers + fillStylizeAOVFromBackend:
//       albedo = denoiser_albedo.xyz (raw)
//       normal = denoiser_normal.xyz * 2 - 1   (device stores [0,1])
//       position.xyz = world pos, position.w encodes material
//         (w<0.5 miss, 0.5<=w<1.5 hit/unknown, w>=1.5 -> matid = w-2)
//       depth = |world - ray_origin|  (ray_origin = Camera::lookfrom)
//       hit = (w>=0.5) && depth>0
//

#include "Stylize/StylizeKernel.h"

#include <cuda_runtime.h>

namespace SC = StylizeCore;

namespace {

// Build the stylize AOV for buffer pixel (bx, by), mirroring makeStylizeAOV.
// Out-of-bounds returns a default AOV (valid=0, hit=0) like the CPU sampler.
__device__ inline SC::StylizeAOVCore buildAOV(int bx, int by,
                                              const StylizeGPU::KernelParams& kp,
                                              const float4* __restrict pos,
                                              const float4* __restrict alb,
                                              const float4* __restrict nrm) {
    SC::StylizeAOVCore aov = SC::defaultAOV();
    if (bx < 0 || by < 0 || bx >= kp.width || by >= kp.height) {
        return aov;
    }
    aov.valid = 1;
    aov.screen_u = ((float)bx + 0.5f) / fmaxf(1.0f, (float)kp.width);
    aov.screen_v = ((float)by + 0.5f) / fmaxf(1.0f, (float)kp.height);

    // Sun direction (makeStylizeAOV: fallback is the raw vector, un-normalized;
    // otherwise Vec3::normalize with its 1e-6 gate).
    SC::SV3 sd = SC::mk(kp.sun_direction.x, kp.sun_direction.y, kp.sun_direction.z);
    if (SC::lensq3(sd) <= 1e-8f) {
        aov.sun_dir = SC::mk(0.32f, 0.82f, 0.46f);
    } else {
        aov.sun_dir = SC::normalize3(sd);
    }
    aov.sun_size_degrees = fmaxf(0.01f, kp.sun_size);
    aov.sun_elevation_degrees = kp.sun_elevation;
    aov.nishita_clouds_enabled = kp.clouds_enabled ? 1 : 0;
    aov.nishita_cloud_coverage = SC::clampf(kp.cloud_coverage, 0.0f, 1.0f);
    aov.nishita_cloud_density = fmaxf(0.0f, kp.cloud_density);
    aov.nishita_cloud_scale = fmaxf(0.05f, kp.cloud_scale);
    aov.nishita_cloud_offset_x = kp.cloud_offset_x;
    aov.nishita_cloud_offset_z = kp.cloud_offset_z;
    aov.nishita_cloud_seed = kp.cloud_seed;

    // view_dir = lower_left + u*horizontal + v*vertical - origin (Vec3::normalize,
    // fallback (0,0,-1) when degenerate).
    SC::SV3 ll = SC::mk(kp.cam_lower_left.x, kp.cam_lower_left.y, kp.cam_lower_left.z);
    SC::SV3 hh = SC::mk(kp.cam_horizontal.x, kp.cam_horizontal.y, kp.cam_horizontal.z);
    SC::SV3 vv = SC::mk(kp.cam_vertical.x, kp.cam_vertical.y, kp.cam_vertical.z);
    SC::SV3 og = SC::mk(kp.cam_origin.x, kp.cam_origin.y, kp.cam_origin.z);
    SC::SV3 view = ll + hh * aov.screen_u + vv * aov.screen_v - og;
    aov.view_dir = SC::lensq3(view) > 1e-8f ? SC::normalize3(view) : SC::mk(0.0f, 0.0f, -1.0f);

    const size_t idx = (size_t)by * (size_t)kp.width + (size_t)bx;
    const float4 P = pos[idx];
    const float matEncoded = P.w;
    const bool hitRaw = matEncoded >= 0.5f;

    float depth = 0.0f;
    if (hitRaw) {
        const float dx = P.x - kp.ray_origin.x;
        const float dy = P.y - kp.ray_origin.y;
        const float dz = P.z - kp.ray_origin.z;
        depth = sqrtf(dx * dx + dy * dy + dz * dz);
    }

    unsigned int material_id = 0xFFFFFFFFu;
    if (matEncoded >= 1.5f) {
        material_id = (unsigned int)(matEncoded + 0.5f) - 2u;
    }

    float3 a = alb ? make_float3(alb[idx].x, alb[idx].y, alb[idx].z) : make_float3(0.0f, 0.0f, 0.0f);
    float3 n = nrm ? make_float3(nrm[idx].x, nrm[idx].y, nrm[idx].z) : make_float3(0.0f, 0.0f, 0.0f);

    aov.albedo = SC::mk(a.x, a.y, a.z);
    aov.normal = SC::mk(n.x * 2.0f - 1.0f, n.y * 2.0f - 1.0f, n.z * 2.0f - 1.0f);
    aov.world_position = SC::mk(P.x, P.y, P.z);
    aov.depth = depth;
    aov.material_id = material_id;
    aov.hit = (hitRaw && depth > 0.0f) ? 1 : 0;
    return aov;
}

// Edge strength from one neighbour, mirroring makeStylizeAOVWithEdges.
__device__ inline float edgeContribution(const SC::StylizeAOVCore& c,
                                         const SC::StylizeAOVCore& n) {
    if (!n.hit) {
        return 1.0f;
    }
    float e = 0.0f;
    const float depth_scale = fmaxf(0.025f, c.depth * 0.015f);
    e += fminf(1.0f, fabsf(c.depth - n.depth) / depth_scale);
    const SC::SV3 dn = c.normal - n.normal;
    e += fminf(1.0f, SC::length3(dn) * 0.75f);
    if (c.material_id != n.material_id) {
        e += 0.45f;
    }
    return e;
}

__global__ void stylizeKernel(uint32_t* __restrict color,
                              const float4* __restrict pos,
                              const float4* __restrict alb,
                              const float4* __restrict nrm,
                              StylizeGPU::KernelParams kp,
                              SC::StyleProfileCore profile) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kp.width || y >= kp.height) {
        return;
    }
    const size_t cidx = (size_t)y * (size_t)kp.width + (size_t)x;
    const int buffer_y = kp.height - 1 - y;

    // Center AOV + edge from neighbours (buffer space, exactly as the CPU path).
    SC::StylizeAOVCore aov = buildAOV(x, buffer_y, kp, pos, alb, nrm);
    if (aov.hit) {
        const SC::StylizeAOVCore right = buildAOV(x + 1, buffer_y, kp, pos, alb, nrm);
        const SC::StylizeAOVCore down  = buildAOV(x, buffer_y + 1, kp, pos, alb, nrm);
        const float edge = edgeContribution(aov, right) + edgeContribution(aov, down);
        aov.edge = SC::clampf(edge * 0.55f, 0.0f, 1.0f);
    }

    const uint32_t px = color[cidx];
    const float inv255 = 1.0f / 255.0f;   // match the CPU multiply-by-reciprocal
    SC::SV3 in = SC::mk(
        (float)((px & kp.rMask) >> kp.rShift) * inv255,
        (float)((px & kp.gMask) >> kp.gShift) * inv255,
        (float)((px & kp.bMask) >> kp.bShift) * inv255);

    // noise coords = surface (x, y), exactly as applyPostProcess is called on CPU.
    SC::SV3 out = SC::applyPostProcess(in, aov, x, y, kp.frame_index, profile);

    const unsigned int ri = (unsigned int)(SC::saturate(out.x) * 255.0f);
    const unsigned int gi = (unsigned int)(SC::saturate(out.y) * 255.0f);
    const unsigned int bi = (unsigned int)(SC::saturate(out.z) * 255.0f);
    color[cidx] = (px & kp.aMask)
                | (ri << kp.rShift)
                | (gi << kp.gShift)
                | (bi << kp.bShift);
}

} // namespace

namespace StylizeGPU {

bool launchStylize(uint32_t* d_color,
                   const float4* d_position,
                   const float4* d_albedo,
                   const float4* d_normal,
                   const KernelParams& params,
                   const StylizeCore::StyleProfileCore& profile,
                   void* stream) {
    if (!d_color || !d_position || params.width <= 0 || params.height <= 0) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid((params.width + block.x - 1) / block.x,
                    (params.height + block.y - 1) / block.y);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    stylizeKernel<<<grid, block, 0, s>>>(d_color, d_position, d_albedo, d_normal, params, profile);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaGetLastError();   // consume sticky error so the next launch is clean
        return false;
    }
    return true;
}

} // namespace StylizeGPU
