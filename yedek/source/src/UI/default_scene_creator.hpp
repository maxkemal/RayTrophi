/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          default_scene_creator.hpp
* Author:        Kemal Demirtaş
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef DEFAULT_SCENE_CREATOR_HPP
#define DEFAULT_SCENE_CREATOR_HPP

#include "scene_data.h"
#include "Triangle.h"
#include "HittableInstance.h"
#include "Vec2.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "Camera.h"
#include "Renderer.h"
#include "World.h"
#include "MaterialManager.h"
#include "Material.h"
#include "PrincipledBSDF.h"
#include "material_gpu.h"
#include "Transform.h"
#include "ParallelBVHNode.h"
#include <vector_types.h>
#include <SDL.h>
#include <filesystem>
#include <string>
#include <cmath>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

// ─────────────────────────────────────────────────────────────────────────
// Helper: create a unit cube (12 triangles) centered at origin, size 2×2×2
// ─────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<std::vector<std::shared_ptr<Triangle>>>
createCubeTriangles(uint16_t matId) {
    auto tris = std::make_shared<std::vector<std::shared_ptr<Triangle>>>();
    tris->reserve(12);

    const Vec3 v[8] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1},
    };

    struct Face { int a, b, c, d; Vec3 n; };
    const Face faces[6] = {
        {0, 3, 2, 1, { 0,  0, -1}},
        {4, 5, 6, 7, { 0,  0,  1}},
        {0, 1, 5, 4, { 0, -1,  0}},
        {3, 7, 6, 2, { 0,  1,  0}},
        {0, 4, 7, 3, {-1,  0,  0}},
        {1, 2, 6, 5, { 1,  0,  0}},
    };

    const float w = 1.0f / 4.0f;
    const float h = 1.0f / 3.0f;
    
    struct FaceUV { Vec2 a, b, c, d; };
    const FaceUV uvs[6] = {
        { Vec2(4*w, 1*h), Vec2(4*w, 2*h), Vec2(3*w, 2*h), Vec2(3*w, 1*h) }, // Face 0 (Back)
        { Vec2(1*w, 1*h), Vec2(2*w, 1*h), Vec2(2*w, 2*h), Vec2(1*w, 2*h) }, // Face 1 (Front)
        { Vec2(1*w, 0*h), Vec2(2*w, 0*h), Vec2(2*w, 1*h), Vec2(1*w, 1*h) }, // Face 2 (Bottom)
        { Vec2(1*w, 3*h), Vec2(1*w, 2*h), Vec2(2*w, 2*h), Vec2(2*w, 3*h) }, // Face 3 (Top)
        { Vec2(0*w, 1*h), Vec2(1*w, 1*h), Vec2(1*w, 2*h), Vec2(0*w, 2*h) }, // Face 4 (Left)
        { Vec2(3*w, 1*h), Vec2(3*w, 2*h), Vec2(2*w, 2*h), Vec2(2*w, 1*h) }, // Face 5 (Right)
    };

    for (int i = 0; i < 6; ++i) {
        const auto& f = faces[i];
        const auto& uv = uvs[i];
        auto t1 = std::make_shared<Triangle>(
            v[f.a], v[f.b], v[f.c], f.n, f.n, f.n, uv.a, uv.b, uv.c, matId);
        auto t2 = std::make_shared<Triangle>(
            v[f.a], v[f.c], v[f.d], f.n, f.n, f.n, uv.a, uv.c, uv.d, matId);
        tris->push_back(t1);
        tris->push_back(t2);
    }
    return tris;
}

// ─────────────────────────────────────────────────────────────────────────
// Main: Blender-style default scene (Cube + Point Light + Camera)
// ─────────────────────────────────────────────────────────────────────────
inline void createDefaultScene(SceneData& scene, Renderer& renderer, Backend::IBackend* backend) {
    // Startup can arrive here with scene.initialized=true and/or pre-existing lights,
    // but still no geometry. In that case we must still create the default cube.
    if (!scene.world.objects.empty()) {
        return;
    }

    SCENE_LOG_INFO("Creating default scene (Cube + Point Light + Camera).");

    // ═══════════════════════════════════════════════════════════
    // 1. World — Solid Color for viewport, Nishita ready for Rendered
    // ═══════════════════════════════════════════════════════════
    renderer.world.setMode(WORLD_MODE_COLOR);
    renderer.world.setColor(Vec3(0.224f, 0.224f, 0.224f));
    renderer.world.setColorIntensity(1.0f);
    scene.background_color = Vec3(0.224f, 0.224f, 0.224f);

    // Pre-configure Nishita so it's ready when switching to Rendered
    {
        NishitaSkyParams nishita = renderer.world.getNishitaParams();
        nishita.sun_elevation = 25.0f;
        nishita.sun_azimuth   = 150.0f;
        nishita.sun_intensity = 8.0f;
        nishita.atmosphere_intensity = 8.0f;

        float elevRad = nishita.sun_elevation * 3.14159265f / 180.0f;
        float azimRad = nishita.sun_azimuth   * 3.14159265f / 180.0f;
        Vec3 sunDir(cosf(elevRad) * sinf(azimRad),
                    sinf(elevRad),
                    cosf(elevRad) * cosf(azimRad));
        sunDir = sunDir.normalize();
        nishita.sun_direction = make_float3(sunDir.x, sunDir.y, sunDir.z);
        renderer.world.setNishitaParams(nishita);
    }

    // Quick startup workaround: prefer importing default GLB to avoid empty-geometry
    // startup states that can leave CPU BVH uninitialized.
    bool loadedDefaultGlb = false;
    {
        const auto cwd = std::filesystem::current_path();
        std::vector<std::filesystem::path> candidates = {
            std::filesystem::path("assets/scenes/default.glb"),
            cwd / "assets/scenes/default.glb",

        };
        // Prefer exe-relative assets (checked via SDL_GetBasePath below) for portability.

        if (char* basePathC = SDL_GetBasePath()) {
            std::filesystem::path basePath;
            // SDL_GetBasePath returns UTF-8 encoded C string. On Windows use Win32 API
            // MultiByteToWideChar to convert UTF-8 -> UTF-16 to avoid deprecated <codecvt>.
#ifdef _WIN32
            int required = MultiByteToWideChar(CP_UTF8, 0, basePathC, -1, nullptr, 0);
            if (required > 0) {
                std::wstring wide;
                wide.resize(required);
                MultiByteToWideChar(CP_UTF8, 0, basePathC, -1, &wide[0], required);
                if (!wide.empty() && wide.back() == L'\0') wide.pop_back();
                basePath = std::filesystem::path(wide);
            }
            else {
                basePath = std::filesystem::path(basePathC);
            }
#else
            basePath = std::filesystem::path(basePathC);
#endif
            SDL_free(basePathC);

            candidates.push_back(basePath / "assets/scenes/default.glb");
            candidates.push_back(basePath.parent_path() / "assets/scenes/default.glb");
            // Also check common exe-relative locations: top-level assets and nested 'default' scene folder
            candidates.push_back(basePath / "assets/default.glb");
            candidates.push_back(basePath / "assets/scenes/default/default.glb");
        }

        // Development fallback: derive project path from this source file location.
        {
            std::filesystem::path sourceFilePath(__FILE__);
            auto p = sourceFilePath.parent_path(); // .../source/src/UI
            if (!p.empty()) {
                auto raytracRoot = p.parent_path().parent_path().parent_path(); // .../raytrac_sdl2
                candidates.push_back(raytracRoot / "assets/default.glb");
            }
        }

        for (const auto& candidate : candidates) {
            std::error_code ec;
            const auto normalized = std::filesystem::weakly_canonical(candidate, ec);
            const auto finalPath = ec ? candidate : normalized;
            if (!std::filesystem::exists(finalPath, ec) || ec) continue;

            try {
                renderer.create_scene(scene, backend, finalPath.string(), nullptr, true, "default");
                if (!scene.world.objects.empty()) {
                    loadedDefaultGlb = true;
                    SCENE_LOG_INFO("Default GLB loaded at startup: " + finalPath.string());
                    break;
                }
            }
            catch (const std::exception& e) {
                SCENE_LOG_WARN(std::string("Default GLB load failed: ") + finalPath.string() + " | " + e.what());
            }
            catch (...) {
                SCENE_LOG_WARN(std::string("Default GLB load failed (unknown): ") + finalPath.string());
            }
        }

        if (!loadedDefaultGlb) {
            SCENE_LOG_WARN("Default GLB not found/loaded at startup. Falling back to procedural cube. cwd=" + cwd.string());
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 2. Cube Material (neutral clay)
    // ═══════════════════════════════════════════════════════════
    if (!loadedDefaultGlb || scene.world.objects.empty()) {
        auto cubeMat = std::make_shared<PrincipledBSDF>();
        cubeMat->gpuMaterial = std::make_shared<GpuMaterial>();
        cubeMat->gpuMaterial->albedo       = make_float3(0.8f, 0.8f, 0.8f);
        cubeMat->gpuMaterial->roughness    = 0.5f;
        cubeMat->gpuMaterial->metallic     = 0.0f;
        cubeMat->gpuMaterial->opacity      = 1.0f;
        cubeMat->gpuMaterial->emission     = make_float3(0.0f, 0.0f, 0.0f);
        cubeMat->gpuMaterial->transmission = 0.0f;
        cubeMat->gpuMaterial->ior          = 1.45f;

        uint16_t cubeMatId = MaterialManager::getInstance().addMaterial("Default Material", cubeMat);

        // ═══════════════════════════════════════════════════════════
        // 3. Default Cube (1m × 1m × 1m, sitting on Y=0)
        // ═══════════════════════════════════════════════════════════
        auto cubeTris = createCubeTriangles(cubeMatId);

        // Shared transform: scale ±1 cube to ±0.5, move up so base sits on Y=0
        auto cubeTransform = std::make_shared<Transform>();
        cubeTransform->position = Vec3(0.0f, 0.5f, 0.0f);
        cubeTransform->scale    = Vec3(0.5f, 0.5f, 0.5f);
        cubeTransform->updateMatrix();

        for (auto& tri : *cubeTris) {
            tri->setNodeName("Cube");
            tri->setTransformHandle(cubeTransform);
        }

        // Build a local BVH so CPU ray-picking/selection works
        std::vector<std::shared_ptr<Hittable>> cubeHittables(cubeTris->begin(), cubeTris->end());
        auto cubeBVH = std::make_shared<ParallelBVHNode>(
            cubeHittables, 0, cubeHittables.size(), 0.0, 1.0, 0);

        auto cubeInstance = std::make_shared<HittableInstance>(
            cubeBVH, cubeTris, cubeTransform->base, "Cube");
        scene.world.objects.push_back(cubeInstance);
    }

    // ═══════════════════════════════════════════════════════════
    // 4. Point Light
    // ═══════════════════════════════════════════════════════════
    if (scene.lights.empty()) {
        auto pointLight = std::make_shared<PointLight>(
            Vec3(4.0f, 5.0f, -3.0f),
            Vec3(1.0f, 1.0f, 1.0f),
            0.25f
        );
        pointLight->intensity = 800.0f;
        pointLight->nodeName = "Point Light";
        scene.lights.push_back(pointLight);
    }

    // ═══════════════════════════════════════════════════════════
    // 5. Camera (3/4 perspective looking at the cube)
    // ═══════════════════════════════════════════════════════════
    if (scene.cameras.empty()) {
        Vec3 lookfrom(5.0f, 4.0f, 6.0f);
        Vec3 lookat(0.0f, 0.5f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        float dist_to_focus = (lookfrom - lookat).length();

        auto cam = std::make_shared<Camera>(
            lookfrom, lookat, vup, 40.0f, 16.0f / 9.0f, 0.0f, dist_to_focus, 0);
        cam->nodeName = "Camera";
        scene.addCamera(cam);
    }

    // Only mark the scene initialized if we actually have geometry.
    scene.initialized = !scene.world.objects.empty();
    if (scene.initialized) {
        SCENE_LOG_INFO("Default scene ready: Cube + Point Light + Camera.");
    }
    else {
        SCENE_LOG_INFO("Default scene initialized but contains no geometry.");
    }
}

#endif
