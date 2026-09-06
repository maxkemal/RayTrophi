#include "PostProcess/PostSurface.h"
#include "PostProcess/PostService.h"
#include "Api/RtApiInternal.h"
#include "Stylize/StylizePostProcess.h"
#include "Camera.h"
#include <execution>
#include <numeric>
#include <algorithm>
Vec3 applyVignette(const Vec3& color,int x,int y,int width,int height,float strength) {
    const float u=((x+.5f)/width-.5f)*2, v=((y+.5f)/height-.5f)*2;
    return color*std::clamp(1-strength*(u*u+v*v),0.0f,1.0f);
}
void applyToneMappingToSurfaceWithCamera(SDL_Surface* surface, SDL_Surface* original, ColorProcessor& processor, Renderer* renderer, const Camera* camera) {
    if (!surface || !surface->pixels) return;
    // GPU backends already wrote the complete display transform. Never grade
    // their 8-bit output a second time (including final animation exports).
    if (!renderer) { if (original && original != surface) SDL_BlitSurface(original,nullptr,surface,nullptr); return; }
    const Camera* activeCamera=camera?camera:(rtapi::g_ctx?rtapi::g_ctx->scene.camera.get():nullptr);
    rtpost::syncDisplay(processor,activeCamera,rtapi::renderJobActive() || (rtapi::g_ctx && rtapi::g_ctx->render_settings.is_final_render_mode));
    Uint32* pixels = (Uint32*)surface->pixels;
    int width = surface->w;
    int height = surface->h;
    SDL_PixelFormat* fmt = surface->format;

    const bool use_float_buffer = (renderer != nullptr) &&
                                  renderer->cpu_accumulation_valid &&
                                  (renderer->cpu_accumulation_buffer.size() == (size_t)(width * height));

    if (use_float_buffer) rtpost::meterCpu(reinterpret_cast<const float*>(renderer->cpu_accumulation_buffer.data()), width, height, 4);
    const size_t dstStride=surface->pitch/sizeof(Uint32);
    const size_t srcStride=original?original->pitch/sizeof(Uint32):dstStride;
    const ColorProcessor displayProcessor=processor;
    Uint32* src = (original && original->pixels) ? (Uint32*)original->pixels : nullptr;
    if (!use_float_buffer && !src) return;

    // Capture format masks/shifts once — avoid SDL_MapRGB/SDL_GetRGB per-pixel dispatch.
    const Uint32 rMask = fmt->Rmask, gMask = fmt->Gmask, bMask = fmt->Bmask, aMask = fmt->Amask;
    const Uint8  rShift = fmt->Rshift, gShift = fmt->Gshift, bShift = fmt->Bshift;
    const float  inv255 = 1.0f / 255.0f;

    // Precompute linear→sRGB→uint8 LUT so the per-pixel hot path avoids 3× std::pow.
    // Only needed for use_float_buffer branch; non-float branch consumes pre-sRGB pixels.
    constexpr int LUT_SIZE = 4096;
    constexpr float LUT_MAX = float(LUT_SIZE - 1);
    alignas(64) uint8_t srgbLut[LUT_SIZE];
    if (use_float_buffer) {
        for (int i = 0; i < LUT_SIZE; ++i) {
            float x = float(i) / LUT_MAX;
            float s = (x <= 0.0031308f) ? 12.92f * x
                                        : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
            if (s < 0.0f) s = 0.0f;
            if (s > 1.0f) s = 1.0f;
            srgbLut[i] = static_cast<uint8_t>(s * 255.0f + 0.5f);
        }
    }


    const bool use_denoised  = use_float_buffer && renderer->hasCPUDenoisedBuffer();
    const bool vignette_on   = processor.params.enable_vignette;
    const float vignette_strength = processor.params.vignette_strength;
    const Stylize::StylizeModeState* stylize_state = renderer ? &renderer->stylizeMode : nullptr;
    const int stylize_frame = renderer ? renderer->world.getGPUData().frame_count : 0;
    const WorldData stylize_world = renderer ? renderer->world.getGPUData() : WorldData{};
    const bool use_cpu_stylize_aov =
        stylize_state &&
        stylize_state->enabled &&
        use_float_buffer &&
        renderer->cpu_albedo_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer->cpu_normal_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer->cpu_world_position_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer->cpu_depth_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer->cpu_material_id_buffer.size() == static_cast<size_t>(width * height);

    auto makeStylizeAOV = [=](int sx, int sy) -> Stylize::StylizeAOVSample {
        Stylize::StylizeAOVSample aov;
        if (!use_cpu_stylize_aov ||
            sx < 0 || sy < 0 || sx >= width || sy >= height) {
            return aov;
        }
        aov.valid = true;
        aov.screen_u = (static_cast<float>(sx) + 0.5f) / std::max(1.0f, static_cast<float>(width));
        aov.screen_v = (static_cast<float>(sy) + 0.5f) / std::max(1.0f, static_cast<float>(height));
        aov.sun_dir = Vec3(stylize_world.nishita.sun_direction.x, stylize_world.nishita.sun_direction.y, stylize_world.nishita.sun_direction.z);
        if (aov.sun_dir.length_squared() <= 1e-8f) {
            aov.sun_dir = Vec3(0.32f, 0.82f, 0.46f);
        } else {
            aov.sun_dir = aov.sun_dir.normalize();
        }
        aov.sun_size_degrees = std::max(0.01f, stylize_world.nishita.sun_size);
        aov.sun_elevation_degrees = stylize_world.nishita.sun_elevation;
        aov.nishita_clouds_enabled = stylize_world.nishita.clouds_enabled != 0;
        aov.nishita_cloud_coverage = std::clamp(stylize_world.nishita.cloud_coverage, 0.0f, 1.0f);
        aov.nishita_cloud_density = std::max(0.0f, stylize_world.nishita.cloud_density);
        aov.nishita_cloud_scale = std::max(0.05f, stylize_world.nishita.cloud_scale);
        aov.nishita_cloud_offset_x = stylize_world.nishita.cloud_offset_x;
        aov.nishita_cloud_offset_z = stylize_world.nishita.cloud_offset_z;
        aov.nishita_cloud_seed = stylize_world.nishita.cloud_seed;
        float stylize_view_len = 0.0f;
        if (camera) {
            Vec3 view_dir = camera->lower_left_corner
                + aov.screen_u * camera->horizontal
                + aov.screen_v * camera->vertical
                - camera->origin;
            stylize_view_len = view_dir.length();
            aov.view_dir = view_dir.length_squared() > 1e-8f ? view_dir.normalize() : Vec3(0.0f, 0.0f, -1.0f);
        }
        const size_t idx = static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx);
        const Renderer::Vec4& albedo = renderer->cpu_albedo_accumulation_buffer[idx];
        const Renderer::Vec4& normal = renderer->cpu_normal_accumulation_buffer[idx];
        const Renderer::Vec4& world_position = renderer->cpu_world_position_accumulation_buffer[idx];
        aov.hit = albedo.w > 0.0f && renderer->cpu_depth_accumulation_buffer[idx] > 0.0f;
        aov.albedo = Vec3(albedo.x, albedo.y, albedo.z);
        aov.normal = Vec3(normal.x, normal.y, normal.z);
        aov.world_position = Vec3(world_position.x, world_position.y, world_position.z);
        aov.depth = renderer->cpu_depth_accumulation_buffer[idx];
        aov.material_id = renderer->cpu_material_id_buffer[idx];
        if (aov.hit && stylize_view_len > 1e-6f) {
            // world units per pixel at the hit — drives screen-constant brush daub sizing
            aov.pixel_scale = aov.depth * camera->vertical.length()
                            / (std::max(1.0f, static_cast<float>(height)) * stylize_view_len);
        }
        return aov;
    };

    auto makeStylizeAOVWithEdges = [=](int sx, int sy) -> Stylize::StylizeAOVSample {
        Stylize::StylizeAOVSample aov = makeStylizeAOV(sx, sy);
        if (!aov.hit) {
            return aov;
        }
        const Stylize::StylizeAOVSample right = makeStylizeAOV(sx + 1, sy);
        const Stylize::StylizeAOVSample down = makeStylizeAOV(sx, sy + 1);
        float edge = 0.0f;
        auto accumulateEdge = [&](const Stylize::StylizeAOVSample& n) {
            if (!n.hit) {
                edge += 1.0f;
                return;
            }
            const float depth_scale = std::max(0.025f, aov.depth * 0.015f);
            edge += std::min(1.0f, std::abs(aov.depth - n.depth) / depth_scale);
            edge += std::min(1.0f, (aov.normal - n.normal).length() * 0.75f);
            if (aov.material_id != n.material_id) {
                edge += 0.45f;
            }
        };
        accumulateEdge(right);
        accumulateEdge(down);
        aov.edge = std::clamp(edge * 0.55f, 0.0f, 1.0f);
        return aov;
    };

    const float* denoised_ptr = use_denoised ? renderer->cpu_denoised_buffer.data() : nullptr;
    const Renderer::Vec4* accum_ptr = (use_float_buffer && !use_denoised)
                                      ? renderer->cpu_accumulation_buffer.data() : nullptr;

    // std::for_each_n over row indices → inner loop vectorizable; avoids std::async thread spawn per call.
    std::vector<int> rowIndices(height);
    std::iota(rowIndices.begin(), rowIndices.end(), 0);

    std::for_each_n(std::execution::par_unseq, rowIndices.data(), (size_t)height,
        [=](int j) {
            const int buffer_y = height - 1 - j;
            Uint32* __restrict rowDst = pixels + (size_t)j * dstStride;
            const Uint32* __restrict rowSrc = src ? (src + (size_t)j * srcStride) : nullptr;

            for (int i = 0; i < width; ++i) {
                Vec3 raw_color;

                if (use_float_buffer) {
                    if (use_denoised) {
                        const size_t idx = ((size_t)j * (size_t)width + (size_t)i) * 3;
                        raw_color = Vec3(denoised_ptr[idx], denoised_ptr[idx + 1], denoised_ptr[idx + 2]);
                    } else {
                        const Renderer::Vec4& p = accum_ptr[(size_t)buffer_y * (size_t)width + (size_t)i];
                        raw_color = Vec3(p.x, p.y, p.z);
                    }

                } else {
                    Uint32 px = rowSrc[i];
                    float r = float((px & rMask) >> rShift) * inv255;
                    float g = float((px & gMask) >> gShift) * inv255;
                    float b = float((px & bMask) >> bShift) * inv255;
                    raw_color = Vec3(r, g, b);
                }

                Vec3 final_color = displayProcessor.processColor(raw_color, i, j);

                if (vignette_on)
                    final_color = applyVignette(final_color, i, j, width, height, vignette_strength);

                if (stylize_state && stylize_state->enabled) {
                    if (use_cpu_stylize_aov) {
                        final_color = Stylize::applyPostProcess(
                            final_color,
                            makeStylizeAOVWithEdges(i, buffer_y),
                            i, j, stylize_frame, *stylize_state);
                    } else {
                        final_color = Stylize::applyPostProcess(final_color, i, j, stylize_frame, *stylize_state);
                    }
                }

                Uint8 ri, gi, bi;
                if (use_float_buffer) {
                    float fx = final_color.x; if (fx < 0.0f) fx = 0.0f; else if (fx > 1.0f) fx = 1.0f;
                    float fy = final_color.y; if (fy < 0.0f) fy = 0.0f; else if (fy > 1.0f) fy = 1.0f;
                    float fz = final_color.z; if (fz < 0.0f) fz = 0.0f; else if (fz > 1.0f) fz = 1.0f;
                    ri = srgbLut[int(fx * LUT_MAX)];
                    gi = srgbLut[int(fy * LUT_MAX)];
                    bi = srgbLut[int(fz * LUT_MAX)];
                } else {
                    float fx = final_color.x; if (fx < 0.0f) fx = 0.0f; else if (fx > 1.0f) fx = 1.0f;
                    float fy = final_color.y; if (fy < 0.0f) fy = 0.0f; else if (fy > 1.0f) fy = 1.0f;
                    float fz = final_color.z; if (fz < 0.0f) fz = 0.0f; else if (fz > 1.0f) fz = 1.0f;
                    ri = uint8_t(fx * 255.0f);
                    gi = uint8_t(fy * 255.0f);
                    bi = uint8_t(fz * 255.0f);
                }

                Uint32 alpha = rowSrc ? (rowSrc[i] & aMask) : aMask;
                rowDst[i] = alpha
                          | ((Uint32)ri << rShift)
                          | ((Uint32)gi << gShift)
                          | ((Uint32)bi << bShift);
            }
        });
}

void applyToneMappingToSurface(SDL_Surface* surface, SDL_Surface* original, ColorProcessor& processor, Renderer* renderer) {
    applyToneMappingToSurfaceWithCamera(surface, original, processor, renderer, nullptr);
}

void applyStylizeToSurfaceWithCamera(SDL_Surface* surface, Renderer& renderer, bool use_cpu_aov, const Camera* camera) {
    if (!surface || !surface->pixels || !renderer.stylizeMode.enabled) return;

    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    SDL_PixelFormat* fmt = surface->format;
    const int width = surface->w;
    const int height = surface->h;

    const Uint32 rMask = fmt->Rmask, gMask = fmt->Gmask, bMask = fmt->Bmask, aMask = fmt->Amask;
    const Uint8 rShift = fmt->Rshift, gShift = fmt->Gshift, bShift = fmt->Bshift;
    const float inv255 = 1.0f / 255.0f;
    const int stylize_frame = renderer.world.getGPUData().frame_count;
    const WorldData stylize_world = renderer.world.getGPUData();
    const bool use_cpu_stylize_aov =
        use_cpu_aov &&
        renderer.cpu_albedo_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer.cpu_normal_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer.cpu_world_position_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer.cpu_depth_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer.cpu_material_id_buffer.size() == static_cast<size_t>(width * height);

    auto makeStylizeAOV = [&](int sx, int sy) -> Stylize::StylizeAOVSample {
        Stylize::StylizeAOVSample aov;
        if (!use_cpu_stylize_aov ||
            sx < 0 || sy < 0 || sx >= width || sy >= height) {
            return aov;
        }
        aov.valid = true;
        aov.screen_u = (static_cast<float>(sx) + 0.5f) / std::max(1.0f, static_cast<float>(width));
        aov.screen_v = (static_cast<float>(sy) + 0.5f) / std::max(1.0f, static_cast<float>(height));
        aov.sun_dir = Vec3(stylize_world.nishita.sun_direction.x, stylize_world.nishita.sun_direction.y, stylize_world.nishita.sun_direction.z);
        if (aov.sun_dir.length_squared() <= 1e-8f) {
            aov.sun_dir = Vec3(0.32f, 0.82f, 0.46f);
        } else {
            aov.sun_dir = aov.sun_dir.normalize();
        }
        aov.sun_size_degrees = std::max(0.01f, stylize_world.nishita.sun_size);
        aov.sun_elevation_degrees = stylize_world.nishita.sun_elevation;
        aov.nishita_clouds_enabled = stylize_world.nishita.clouds_enabled != 0;
        aov.nishita_cloud_coverage = std::clamp(stylize_world.nishita.cloud_coverage, 0.0f, 1.0f);
        aov.nishita_cloud_density = std::max(0.0f, stylize_world.nishita.cloud_density);
        aov.nishita_cloud_scale = std::max(0.05f, stylize_world.nishita.cloud_scale);
        aov.nishita_cloud_offset_x = stylize_world.nishita.cloud_offset_x;
        aov.nishita_cloud_offset_z = stylize_world.nishita.cloud_offset_z;
        aov.nishita_cloud_seed = stylize_world.nishita.cloud_seed;
        float stylize_view_len = 0.0f;
        if (camera) {
            Vec3 view_dir = camera->lower_left_corner
                + aov.screen_u * camera->horizontal
                + aov.screen_v * camera->vertical
                - camera->origin;
            stylize_view_len = view_dir.length();
            aov.view_dir = view_dir.length_squared() > 1e-8f ? view_dir.normalize() : Vec3(0.0f, 0.0f, -1.0f);
        }
        const size_t idx = static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx);
        const Renderer::Vec4& albedo = renderer.cpu_albedo_accumulation_buffer[idx];
        const Renderer::Vec4& normal = renderer.cpu_normal_accumulation_buffer[idx];
        const Renderer::Vec4& world_position = renderer.cpu_world_position_accumulation_buffer[idx];
        aov.hit = albedo.w > 0.0f && renderer.cpu_depth_accumulation_buffer[idx] > 0.0f;
        aov.albedo = Vec3(albedo.x, albedo.y, albedo.z);
        aov.normal = Vec3(normal.x, normal.y, normal.z);
        aov.world_position = Vec3(world_position.x, world_position.y, world_position.z);
        aov.depth = renderer.cpu_depth_accumulation_buffer[idx];
        aov.material_id = renderer.cpu_material_id_buffer[idx];
        if (aov.hit && stylize_view_len > 1e-6f) {
            // world units per pixel at the hit — drives screen-constant brush daub sizing
            aov.pixel_scale = aov.depth * camera->vertical.length()
                            / (std::max(1.0f, static_cast<float>(height)) * stylize_view_len);
        }
        return aov;
    };

    auto makeStylizeAOVWithEdges = [&](int sx, int sy) -> Stylize::StylizeAOVSample {
        Stylize::StylizeAOVSample aov = makeStylizeAOV(sx, sy);
        if (!aov.hit) return aov;
        const Stylize::StylizeAOVSample right = makeStylizeAOV(sx + 1, sy);
        const Stylize::StylizeAOVSample down = makeStylizeAOV(sx, sy + 1);
        float edge = 0.0f;
        auto accumulateEdge = [&](const Stylize::StylizeAOVSample& n) {
            if (!n.hit) {
                edge += 1.0f;
                return;
            }
            const float depth_scale = std::max(0.025f, aov.depth * 0.015f);
            edge += std::min(1.0f, std::abs(aov.depth - n.depth) / depth_scale);
            edge += std::min(1.0f, (aov.normal - n.normal).length() * 0.75f);
            if (aov.material_id != n.material_id) edge += 0.45f;
        };
        accumulateEdge(right);
        accumulateEdge(down);
        aov.edge = std::clamp(edge * 0.55f, 0.0f, 1.0f);
        return aov;
    };

    std::vector<int> rowIndices(height);
    std::iota(rowIndices.begin(), rowIndices.end(), 0);

    std::for_each_n(std::execution::par_unseq, rowIndices.data(), static_cast<size_t>(height),
        [=, &renderer](int y) {
            Uint32* row = pixels + static_cast<size_t>(y) * static_cast<size_t>(width);
            for (int x = 0; x < width; ++x) {
                const Uint32 px = row[x];
                Vec3 color(
                    static_cast<float>((px & rMask) >> rShift) * inv255,
                    static_cast<float>((px & gMask) >> gShift) * inv255,
                    static_cast<float>((px & bMask) >> bShift) * inv255
                );
                if (use_cpu_stylize_aov) {
                    const int buffer_y = height - 1 - y;
                    color = Stylize::applyPostProcess(
                        color,
                        makeStylizeAOVWithEdges(x, buffer_y),
                        x, y, stylize_frame, renderer.stylizeMode);
                } else {
                    color = Stylize::applyPostProcess(color, x, y, stylize_frame, renderer.stylizeMode);
                }

                const Uint8 ri = static_cast<Uint8>(std::clamp(color.x, 0.0f, 1.0f) * 255.0f);
                const Uint8 gi = static_cast<Uint8>(std::clamp(color.y, 0.0f, 1.0f) * 255.0f);
                const Uint8 bi = static_cast<Uint8>(std::clamp(color.z, 0.0f, 1.0f) * 255.0f);
                row[x] = (px & aMask)
                       | (static_cast<Uint32>(ri) << rShift)
                       | (static_cast<Uint32>(gi) << gShift)
                       | (static_cast<Uint32>(bi) << bShift);
            }
        });
}

void applyStylizeToSurface(SDL_Surface* surface, Renderer& renderer, bool use_cpu_aov) {
    applyStylizeToSurfaceWithCamera(surface, renderer, use_cpu_aov, nullptr);
}

void applyCPUDenoisedPreviewToSurface(SDL_Surface* surface,Renderer& renderer,const Camera* camera) {
    if(!renderer.hasCPUDenoisedBuffer() || !rtapi::g_ctx)return;
    applyToneMappingToSurfaceWithCamera(surface,surface,rtapi::g_ctx->color_processor,&renderer,camera);
}
