#include "WaterSystem.h"
#include "scene_data.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "globals.h"
#include "fft_ocean.cuh"
#include "perlin.h" // For geometric waves
#include "KeyframeSystem.h" // For WaterKeyframe
#include <map>      // For smooth normal calculation

// CUDA Library Linking
#pragma comment(lib, "cufft.lib")
#pragma comment(lib, "cudart.lib")

// #include "GeometryUtils.h" // Removed: Not needed for manual mesh generation

WaterSurface* WaterManager::getWaterSurface(int id) {
    for (auto& surf : water_surfaces) {
        if (surf.id == id) return &surf;
    }
    return nullptr;
}

void WaterManager::removeWaterSurface(SceneData& scene, int id) {
    // 1. Check if surface exists
    auto it = std::find_if(water_surfaces.begin(), water_surfaces.end(), 
        [id](const WaterSurface& ws) { return ws.id == id; });
        
    if (it == water_surfaces.end()) return;
    
    // Cleanup FFT resources
    if (it->fft_state) {
        FFTOceanState* state = static_cast<FFTOceanState*>(it->fft_state);
        cleanupFFTOcean(state);
        delete state;
        it->fft_state = nullptr;
    }
    
    // 2. Remove triangles from scene
    for (auto& tri : it->mesh_triangles) {
        auto obj_it = std::find(scene.world.objects.begin(), scene.world.objects.end(), tri);
        if (obj_it != scene.world.objects.end()) {
            scene.world.objects.erase(obj_it);
        }
    }
    
    // 3. Remove from manager
    water_surfaces.erase(it);
}

void WaterManager::clear() {
    for (auto& surf : water_surfaces) {
         if (surf.fft_state) {
             FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
             cleanupFFTOcean(state);
             delete state;
             surf.fft_state = nullptr;
         }
    }
    water_surfaces.clear();
    next_id = 1;
}

bool WaterManager::update(float dt) {
    static float global_time = 0.0f;
    global_time += dt;
    bool needs_gpu_sync = false;

    for (auto& surf : water_surfaces) {
        // ════════════════════════════════════════════════════════════════════════
        // FFT OCEAN UPDATE (GPU-side animation - shader based)
        // ════════════════════════════════════════════════════════════════════════
        if (surf.params.use_fft_ocean) {
            // Manage FFT State
            if (!surf.fft_state) {
                 FFTOceanState* state = new FFTOceanState();
                 surf.fft_state = (void*)state;
            }
            
            FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
            
            // Map parameters
            FFTOceanParams fft_params;
            fft_params.resolution = surf.params.fft_resolution;
            fft_params.ocean_size = surf.params.fft_ocean_size;
            fft_params.wind_speed = surf.params.fft_wind_speed;
            fft_params.wind_direction = surf.params.fft_wind_direction;
            fft_params.choppiness = surf.params.fft_choppiness;
            fft_params.amplitude = surf.params.fft_amplitude;
            fft_params.time_scale = surf.params.fft_time_scale;
            
            // Check initialization
            if (!state->initialized || state->current_resolution != fft_params.resolution) {
                if (initFFTOcean(state, &fft_params)) {
                    needs_gpu_sync = true;
                }
            }

            // Run simulation
            updateFFTOcean(state, &fft_params, global_time);
            
            // Connect to Material
            if (surf.material_id > 0) {
                auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
                if (mat && mat->gpuMaterial) {
                    if (mat->gpuMaterial->fft_height_tex != state->tex_height ||
                        mat->gpuMaterial->fft_normal_tex != state->tex_normal) {
                        
                        mat->gpuMaterial->fft_height_tex = state->tex_height;
                        mat->gpuMaterial->fft_normal_tex = state->tex_normal;
                        needs_gpu_sync = true;
                    }
                }
            }
        } else {
             // Cleanup if disabled but state exists
             if (surf.fft_state) {
                 FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
                 cleanupFFTOcean(state);
                 delete state;
                 surf.fft_state = nullptr;
                 needs_gpu_sync = true;
                 
                 if (surf.material_id > 0) {
                     auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
                     if (mat && mat->gpuMaterial) {
                         mat->gpuMaterial->fft_height_tex = 0;
                         mat->gpuMaterial->fft_normal_tex = 0;
                     }
                 }
             }
        }
        
        // Note: Mesh animation is handled via keyframe system (applyKeyframe)
        // FFT animation is shader-based and doesn't need mesh updates
    }
    
    return needs_gpu_sync;
}

cudaTextureObject_t WaterManager::getFirstFFTHeightMap() {
    for (const auto& surf : water_surfaces) {
        if (surf.params.use_fft_ocean && surf.fft_state) {
            FFTOceanState* state = static_cast<FFTOceanState*>(surf.fft_state);
            return state->tex_height;
        }
    }
    return 0;
}

WaterSurface* WaterManager::createWaterPlane(SceneData& scene, const Vec3& pos, float size, float density) {
    WaterSurface surf;
    surf.id = next_id++;
    surf.name = "Water_Plane_" + std::to_string(surf.id);
    
    // 1. Create unique Water Material
    auto water_mat = std::make_shared<PrincipledBSDF>();
    auto gpu = std::make_shared<GpuMaterial>();
    
    // === BASE WATER MATERIAL ===
    // Albedo controls transmission tint - use deep_color for Beer's law
    gpu->albedo = make_float3(
        surf.params.deep_color.x, 
        surf.params.deep_color.y, 
        surf.params.deep_color.z
    );
    gpu->transmission = 1.0f;
    gpu->opacity = 1.0f;
    gpu->roughness = surf.params.roughness;
    gpu->ior = surf.params.ior;
    gpu->metallic = 0.0f;
    
    // === WAVE PARAMS (original packing) ===
    // anisotropic -> Wave Speed
    // sheen -> Wave Strength (Serves as IS_WATER flag if > 0)
    // sheen_tint -> Wave Frequency
    gpu->anisotropic = surf.params.wave_speed;
    gpu->sheen = fmaxf(0.001f, surf.params.wave_strength);  // >0 = IS_WATER flag
    gpu->sheen_tint = surf.params.wave_frequency;
    
    // === ADVANCED WATER PARAMS (new packing) ===
    // clearcoat -> Shore Foam Intensity
    // clearcoat_roughness -> Caustic Intensity
    gpu->clearcoat = surf.params.shore_foam_intensity;
    gpu->clearcoat_roughness = surf.params.caustic_intensity;
    
    // subsurface -> Depth Max (scaled: divide by 100 to fit 0-1 range)
    // subsurface_scale -> Absorption Density
    gpu->subsurface = surf.params.depth_max / 100.0f;
    gpu->subsurface_scale = surf.params.absorption_density;
    
    // subsurface_color -> Absorption Color
    gpu->subsurface_color = make_float3(
        surf.params.absorption_color.x,
        surf.params.absorption_color.y,
        surf.params.absorption_color.z
    );
    
    // subsurface_radius -> (shore_foam_distance, caustic_scale, sss_intensity)
    gpu->subsurface_radius = make_float3(
        surf.params.shore_foam_distance,
        surf.params.caustic_scale,
        surf.params.sss_intensity
    );
    
    // emission -> Shallow Color (repurposed for water)
    gpu->emission = make_float3(
        surf.params.shallow_color.x,
        surf.params.shallow_color.y,
        surf.params.shallow_color.z
    );
    
    // translucent -> Foam Level
    gpu->translucent = surf.params.foam_level;
    
    // subsurface_anisotropy -> Caustic Speed
    gpu->subsurface_anisotropy = surf.params.caustic_speed;

    // Water Details (New)
    gpu->micro_detail_strength = surf.params.micro_detail_strength;
    gpu->micro_detail_scale = surf.params.micro_detail_scale;
    gpu->foam_noise_scale = surf.params.foam_noise_scale;
    gpu->foam_threshold = surf.params.foam_threshold;
    
    // FFT
    gpu->fft_ocean_size = surf.params.fft_ocean_size;
    gpu->fft_choppiness = surf.params.fft_choppiness;
    
    water_mat->gpuMaterial = gpu;
    
    // Register material
    std::string mat_name = "Water_Mat_" + std::to_string(surf.id);
    surf.material_id = MaterialManager::getInstance().getOrCreateMaterialID(mat_name, water_mat);
    
    // 2. Generate Grid Mesh (NxN triangles for waves)
    // Resolution based on density
    int segments = static_cast<int>(size * density);
    if (segments < 2) segments = 2;
    if (segments > 256) segments = 256; // Limit for safety
    
    float step = size / segments;
    // Create vertices around origin (local space) - pivot will be at center
    float half_size = size * 0.5f;
    
    // Transform stores the actual world position
    std::shared_ptr<Transform> shared_transform = std::make_shared<Transform>();
    Matrix4x4 world_transform = Matrix4x4::translation(pos);
    shared_transform->setBase(world_transform);
    
    for (int z = 0; z < segments; z++) {
        for (int x = 0; x < segments; x++) {
            // Local space coordinates (centered around origin)
            float x0 = -half_size + (x * step);
            float z0 = -half_size + (z * step);
            float x1 = x0 + step;
            float z1 = z0 + step;
            
            // Grid cell vertices in local space (y=0 at local origin)
            Vec3 v0(x0, 0, z0);
            Vec3 v1(x1, 0, z0);
            Vec3 v2(x1, 0, z1);
            Vec3 v3(x0, 0, z1);
            
            // UVs
            float u0 = (float)x / segments;
            float v_0 = (float)z / segments; // v_0 to avoid variable name conflict
            float u1 = (float)(x + 1) / segments;
            float v_1 = (float)(z + 1) / segments;
            
            Vec3 n(0, 1, 0); // Up normal
            
            // Triangle 1
            auto tri1 = std::make_shared<Triangle>(v0, v1, v2, n, n, n, Vec2(u0, v_0), Vec2(u1, v_0), Vec2(u1, v_1), surf.material_id);
            tri1->setTransformHandle(shared_transform);
            tri1->setNodeName(surf.name);
            
            // Triangle 2
            auto tri2 = std::make_shared<Triangle>(v0, v2, v3, n, n, n, Vec2(u0, v_0), Vec2(u1, v_1), Vec2(u0, v_1), surf.material_id);
            tri2->setTransformHandle(shared_transform);
            tri2->setNodeName(surf.name);
            
            surf.mesh_triangles.push_back(tri1);
            surf.mesh_triangles.push_back(tri2);
            
            // Add to scene
            scene.world.objects.push_back(tri1);
            scene.world.objects.push_back(tri2);
        }
    }
    
    if (!surf.mesh_triangles.empty()) {
        surf.reference_triangle = surf.mesh_triangles[0];
    }
    
    water_surfaces.push_back(surf);
    return &water_surfaces.back();
}

void WaterManager::updateWaterMesh(WaterSurface* surf) {
    if (!surf || surf->type != WaterSurface::Type::Plane) return;
    
    bool use_geo = surf->params.use_geometric_waves;
    float max_height = surf->params.geo_wave_height;
    float scale = surf->params.geo_wave_scale;
    float chop = surf->params.geo_wave_choppiness;
    
    if (scale < 0.1f) scale = 0.1f;
    
    // Static perlin instance
    static Perlin perlin;
    
    // Noise Parameters
    int octaves = surf->params.geo_octaves;
    float persistence = surf->params.geo_persistence;
    float lacunarity = surf->params.geo_lacunarity;
    float ridge_offset = surf->params.geo_ridge_offset;
    auto noise_type = surf->params.geo_noise_type;
    
    // Ocean parameters
    float damping = surf->params.geo_damping;
    float alignment = surf->params.geo_alignment;
    float depth = surf->params.geo_depth;
    float swell_dir = surf->params.geo_swell_direction * 3.14159265f / 180.0f; // to radians
    float swell_amp = surf->params.geo_swell_amplitude;
    float sharpening = surf->params.geo_sharpening;
    float detail_scale = surf->params.geo_detail_scale;
    float detail_strength = surf->params.geo_detail_strength;
    bool smooth_normals = surf->params.geo_smooth_normals;

    // ════════════════════════════════════════════════════════════════════════
    // GERSTNER WAVE HELPER (Tessendorf-inspired, physically-based circular wave)
    // ════════════════════════════════════════════════════════════════════════
    struct GerstnerWave {
        float amplitude;
        float wavelength;
        float speed;
        float steepness; // Q parameter (0 = sine wave, 1 = trochoid)
        float direction; // radians
    };
    
    // Generate wave components based on wind direction and parameters
    auto getGerstnerDisplacement = [&](float x, float z, float time = 0.0f) -> Vec3 {
        if (!use_geo || noise_type != WaterWaveParams::NoiseType::Gerstner) 
            return Vec3(x, 0.0f, z);
        
        // Create 6-8 waves with different frequencies (Blender uses similar approach)
        std::vector<GerstnerWave> waves;
        float baseWavelength = scale;
        float baseDir = swell_dir;
        
        for (int i = 0; i < 6; ++i) {
            GerstnerWave w;
            float freqMult = powf(lacunarity, (float)i);
            float ampMult = powf(persistence, (float)i);
            
            w.wavelength = baseWavelength / freqMult;
            w.amplitude = max_height * ampMult * 0.25f;
            w.speed = sqrtf(9.81f * 2.0f * 3.14159265f / w.wavelength); // Deep water dispersion
            w.steepness = fminf(1.0f, chop * 0.5f); // Q parameter
            
            // Vary direction based on alignment
            float dirSpread = (1.0f - alignment) * 3.14159265f * 0.5f;
            float dirOffset = ((float)i - 2.5f) / 2.5f * dirSpread;
            
            // Apply damping for perpendicular waves
            float angleDiff = fabsf(dirOffset);
            float dampFactor = 1.0f - damping * sinf(angleDiff);
            w.amplitude *= fmaxf(0.1f, dampFactor);
            
            w.direction = baseDir + dirOffset;
            waves.push_back(w);
        }
        
        // Add swell (long-period waves from distant storms)
        if (swell_amp > 0.001f) {
            GerstnerWave swell;
            swell.wavelength = scale * 3.0f;
            swell.amplitude = max_height * swell_amp * 0.5f;
            swell.speed = sqrtf(9.81f * 2.0f * 3.14159265f / swell.wavelength);
            swell.steepness = 0.2f; // Swells are smooth
            swell.direction = swell_dir + 0.5f; // Slightly offset
            waves.push_back(swell);
        }
        
        // Sum all wave contributions
        Vec3 result(0.0f, 0.0f, 0.0f);
        for (const auto& w : waves) {
            float k = 2.0f * 3.14159265f / w.wavelength;
            float dx = cosf(w.direction);
            float dz = sinf(w.direction);
            float phase = k * (x * dx + z * dz) - w.speed * time;
            
            float Q = w.steepness / (k * w.amplitude * (float)waves.size());
            Q = fminf(Q, 1.0f);
            
            result.x += Q * w.amplitude * dx * cosf(phase);
            result.y += w.amplitude * sinf(phase);
            result.z += Q * w.amplitude * dz * cosf(phase);
        }
        
        return Vec3(x + result.x * chop, result.y, z + result.z * chop);
    };
    
    // ════════════════════════════════════════════════════════════════════════
    // TESSENDORF SIMPLIFIED (Predictable procedural ocean without FFT)
    // ════════════════════════════════════════════════════════════════════════
    auto getTessendorfSimplified = [&](float x, float z) -> float {
        if (!use_geo || noise_type != WaterWaveParams::NoiseType::TessendorfSimple) 
            return 0.0f;
        
        float height = 0.0f;
        float amp = max_height;
        float freq = 1.0f / scale;
        
        // Base direction from swell
        float dirX = cosf(swell_dir);
        float dirZ = sinf(swell_dir);
        
        for (int i = 0; i < octaves; ++i) {
            // Rotate direction slightly per octave
            float angle = (float)i * 0.3f * (1.0f - alignment);
            float dx = cosf(swell_dir + angle);
            float dz = sinf(swell_dir + angle);
            
            float phase = (x * dx + z * dz) * freq;
            float wave = sinf(phase * 2.0f * 3.14159265f);
            
            // Apply sharpening (sharper peaks)
            if (sharpening > 0.001f) {
                wave = powf(fabsf(wave), 1.0f - sharpening * 0.5f) * (wave >= 0 ? 1.0f : -1.0f);
            }
            
            // Damping for perpendicular
            float angleDiff = fabsf(angle);
            float dampFactor = 1.0f - damping * sinf(angleDiff);
            
            height += wave * amp * fmaxf(0.1f, dampFactor);
            
            amp *= persistence;
            freq *= lacunarity;
        }
        
        return height;
    };

    // ════════════════════════════════════════════════════════════════════════
    // ADVANCED NOISE GENERATION (Original noise types with detail layer)
    // ════════════════════════════════════════════════════════════════════════
    auto getNoiseValue = [&](float x, float z) -> float {
        if (!use_geo) return 0.0f;
        
        // Handle special wave types
        if (noise_type == WaterWaveParams::NoiseType::TessendorfSimple) {
            return getTessendorfSimplified(x, z);
        }
        
        float nx = x / scale;
        float nz = z / scale;
        
        float value = 0.0f;
        float amp = 1.0f;
        float freq = 1.0f;
        float maxAmp = 0.0f;
        float weight = 1.0f;
        
        for (int i = 0; i < octaves; i++) {
            Vec3 p(nx * freq, 0.0f, nz * freq); 
            float n = perlin.noise(p);
            
            if (noise_type == WaterWaveParams::NoiseType::Ridge) {
                // Ridged Multifractal
                n = ridge_offset - fabsf(n);
                n = n * n;
                if (chop > 0.0f) n = powf(fmaxf(0.0f, n), chop);
                n *= weight;
                weight = fmaxf(0.0f, fminf(1.0f, n * 2.0f));
            } 
            else if (noise_type == WaterWaveParams::NoiseType::Billow) {
                n = fabsf(n);
                n = 2.0f * n - 1.0f;
                n = fabsf(n); 
            }
            else if (noise_type == WaterWaveParams::NoiseType::FBM) {
                // Standard FBM
            }
            else if (noise_type == WaterWaveParams::NoiseType::Perlin) {
                if (i > 0) {
                     maxAmp = 1.0f;
                     continue; 
                }
            }
            else if (noise_type == WaterWaveParams::NoiseType::Voronoi) {
                // Simple worley-like approximation
                float vx = floorf(nx * freq);
                float vz = floorf(nz * freq);
                float minDist = 1.0f;
                for (int ox = -1; ox <= 1; ++ox) {
                    for (int oz = -1; oz <= 1; ++oz) {
                        Vec3 cellP(vx + ox + 0.5f, 0, vz + oz + 0.5f);
                        float hash = perlin.noise(cellP * 0.1f) * 0.5f + 0.5f;
                        float cellX = vx + ox + hash;
                        float cellZ = vz + oz + hash * 0.7f;
                        float dist = sqrtf((nx * freq - cellX) * (nx * freq - cellX) + 
                                          (nz * freq - cellZ) * (nz * freq - cellZ));
                        minDist = fminf(minDist, dist);
                    }
                }
                n = minDist * 2.0f - 1.0f;
            }
            
            value += n * amp;
            maxAmp += amp;
            amp *= persistence;
            freq *= lacunarity;
        }
        
        // Normalization
        if (noise_type != WaterWaveParams::NoiseType::Ridge && maxAmp > 0.001f) {
            value /= maxAmp;
        }
        
        // Add detail layer (high-frequency ripples)
        if (detail_strength > 0.001f) {
            float dx = x / (scale / detail_scale);
            float dz = z / (scale / detail_scale);
            float detail = perlin.noise(Vec3(dx, 0, dz)) * detail_strength;
            value += detail;
        }
        
        // Apply sharpening (sharper wave peaks)
        if (sharpening > 0.001f && value > 0.0f) {
            value = powf(value, 1.0f + sharpening);
        }
        
        return value * max_height;
    };
    
    // ════════════════════════════════════════════════════════════════════════
    // PHASE 1: Apply height displacement to all vertices
    // ════════════════════════════════════════════════════════════════════════
    
    // First, collect unique vertex positions and their heights
    // Using a map to store vertex position -> height
    struct VertexKey {
        int ix, iz; // Grid indices based on position
        bool operator<(const VertexKey& o) const {
            if (ix != o.ix) return ix < o.ix;
            return iz < o.iz;
        }
    };
    
    std::map<VertexKey, Vec3> vertexPositions; // Displaced positions
    std::map<VertexKey, Vec3> vertexNormals;   // Accumulated normals
    std::map<VertexKey, int> vertexCounts;     // Count for averaging
    
    float epsilon = 0.001f;
    auto makeKey = [epsilon](const Vec3& p) -> VertexKey {
        // Quantize to grid to handle floating point precision
        return { (int)roundf(p.x * 100.0f), (int)roundf(p.z * 100.0f) };
    };
    
    // First pass: Displace vertices and store positions
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        
        for (int v = 0; v < 3; ++v) {
            Vec3 p = tri->getOriginalVertexPosition(v);
            VertexKey key = makeKey(p);
            
            if (vertexPositions.find(key) == vertexPositions.end()) {
                Vec3 displaced = p;
                
                if (noise_type == WaterWaveParams::NoiseType::Gerstner) {
                    displaced = getGerstnerDisplacement(p.x, p.z, 0.0f);
                } else {
                    displaced.y = getNoiseValue(p.x, p.z);
                }
                
                vertexPositions[key] = displaced;
                vertexNormals[key] = Vec3(0, 0, 0);
                vertexCounts[key] = 0;
            }
        }
    }
    
    // ════════════════════════════════════════════════════════════════════════
    // PHASE 2: Calculate face normals and accumulate for smooth shading
    // ════════════════════════════════════════════════════════════════════════
    
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        
        // Get displaced positions for this triangle
        Vec3 orig0 = tri->getOriginalVertexPosition(0);
        Vec3 orig1 = tri->getOriginalVertexPosition(1);
        Vec3 orig2 = tri->getOriginalVertexPosition(2);
        
        VertexKey k0 = makeKey(orig0);
        VertexKey k1 = makeKey(orig1);
        VertexKey k2 = makeKey(orig2);
        
        Vec3 p0 = vertexPositions[k0];
        Vec3 p1 = vertexPositions[k1];
        Vec3 p2 = vertexPositions[k2];
        
        // Calculate face normal
        Vec3 edge1 = p1 - p0;
        Vec3 edge2 = p2 - p0;
        Vec3 faceNormal = edge1.cross(edge2);
        float len = faceNormal.length();
        if (len > 0.0001f) {
            faceNormal = faceNormal / len;
        } else {
            faceNormal = Vec3(0, 1, 0);
        }
        
        // Weight by face area (larger faces contribute more to smooth normal)
        float area = len * 0.5f;
        
        // Accumulate normals (for smooth shading)
        if (smooth_normals) {
            vertexNormals[k0] = vertexNormals[k0] + faceNormal * area;
            vertexNormals[k1] = vertexNormals[k1] + faceNormal * area;
            vertexNormals[k2] = vertexNormals[k2] + faceNormal * area;
            vertexCounts[k0]++;
            vertexCounts[k1]++;
            vertexCounts[k2]++;
        } else {
            // Flat shading: each vertex gets face normal
            vertexNormals[k0] = faceNormal;
            vertexNormals[k1] = faceNormal;
            vertexNormals[k2] = faceNormal;
        }
    }
    
    // Normalize accumulated normals
    for (auto& [key, normal] : vertexNormals) {
        float len = normal.length();
        if (len > 0.0001f) {
            normal = normal / len;
        } else {
            normal = Vec3(0, 1, 0);
        }
    }
    
    // ════════════════════════════════════════════════════════════════════════
    // PHASE 3: Apply displaced positions and smooth normals to triangles
    // ════════════════════════════════════════════════════════════════════════
    
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        
        for (int v = 0; v < 3; ++v) {
            Vec3 orig = tri->getOriginalVertexPosition(v);
            VertexKey key = makeKey(orig);
            
            Vec3 newPos = vertexPositions[key];
            Vec3 newNormal = vertexNormals[key];
            
            // Update position
            tri->setVertexPosition(v, newPos);
            tri->setOriginalVertexPosition(v, newPos);
            
            // Update normal (smooth or flat)
            tri->setVertexNormal(v, newNormal);
            tri->setOriginalVertexNormal(v, newNormal);
        }
        
        tri->markAABBDirty();
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// CACHE ORIGINAL POSITIONS (for animation base)
// ════════════════════════════════════════════════════════════════════════════════
void WaterManager::cacheOriginalPositions(WaterSurface* surf) {
    if (!surf) return;
    
    surf->original_positions.clear();
    
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        for (int v = 0; v < 3; ++v) {
            surf->original_positions.push_back(tri->getOriginalVertexPosition(v));
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// ANIMATED MESH UPDATE (time-based wave animation)
// ════════════════════════════════════════════════════════════════════════════════
void WaterManager::updateAnimatedWaterMesh(WaterSurface* surf, float time) {
    if (!surf || surf->type != WaterSurface::Type::Plane) return;
    if (!surf->params.use_geometric_waves) return;
    
    // Cache original positions if not done yet
    if (surf->original_positions.empty()) {
        // First time - need flat grid positions, not displaced ones
        // This is a fallback - ideally call cacheOriginalPositions after initial creation
        size_t idx = 0;
        for (auto& tri : surf->mesh_triangles) {
            if (!tri) continue;
            for (int v = 0; v < 3; ++v) {
                surf->original_positions.push_back(tri->getOriginalVertexPosition(v));
            }
        }
    }
    
    static Perlin perlin;
    
    float max_height = surf->params.geo_wave_height;
    float scale = surf->params.geo_wave_scale;
    float chop = surf->params.geo_wave_choppiness;
    int octaves = surf->params.geo_octaves;
    float persistence = surf->params.geo_persistence;
    float lacunarity = surf->params.geo_lacunarity;
    float swell_dir = surf->params.geo_swell_direction * 3.14159265f / 180.0f;
    float swell_amp = surf->params.geo_swell_amplitude;
    float alignment = surf->params.geo_alignment;
    float damping = surf->params.geo_damping;
    auto noise_type = surf->params.geo_noise_type;
    
    if (scale < 0.1f) scale = 0.1f;
    
    // ════════════════════════════════════════════════════════════════════════
    // ANIMATED GERSTNER WAVES
    // ════════════════════════════════════════════════════════════════════════
    auto getAnimatedHeight = [&](float x, float z) -> Vec3 {
        Vec3 result(0, 0, 0);
        
        if (noise_type == WaterWaveParams::NoiseType::Gerstner || 
            noise_type == WaterWaveParams::NoiseType::TessendorfSimple) {
            // Multi-wave Gerstner with time
            float numWaves = 6.0f;
            for (int i = 0; i < 6; ++i) {
                float freqMult = powf(lacunarity, (float)i);
                float ampMult = powf(persistence, (float)i);
                
                float wavelength = scale / freqMult;
                float amplitude = max_height * ampMult * 0.25f;
                float speed = sqrtf(9.81f * 2.0f * 3.14159265f / wavelength);
                float steepness = fminf(1.0f, chop * 0.5f);
                
                // Direction spreading
                float dirSpread = (1.0f - alignment) * 3.14159265f * 0.5f;
                float dirOffset = ((float)i - 2.5f) / 2.5f * dirSpread;
                float dir = swell_dir + dirOffset;
                
                // Damping
                float angleDiff = fabsf(dirOffset);
                float dampFactor = 1.0f - damping * sinf(angleDiff);
                amplitude *= fmaxf(0.1f, dampFactor);
                
                float k = 2.0f * 3.14159265f / wavelength;
                float dx = cosf(dir);
                float dz = sinf(dir);
                float phase = k * (x * dx + z * dz) - speed * time;
                
                float Q = steepness / (k * amplitude * numWaves);
                Q = fminf(Q, 1.0f);
                
                result.x += Q * amplitude * dx * cosf(phase);
                result.y += amplitude * sinf(phase);
                result.z += Q * amplitude * dz * cosf(phase);
            }
            
            // Add swell
            if (swell_amp > 0.001f) {
                float wavelength = scale * 3.0f;
                float amplitude = max_height * swell_amp * 0.5f;
                float speed = sqrtf(9.81f * 2.0f * 3.14159265f / wavelength);
                float k = 2.0f * 3.14159265f / wavelength;
                float dx = cosf(swell_dir + 0.5f);
                float dz = sinf(swell_dir + 0.5f);
                float phase = k * (x * dx + z * dz) - speed * time;
                result.y += amplitude * sinf(phase);
            }
        } else {
            // Simple animated noise (FBM, Ridge, etc.)
            float value = 0.0f;
            float amp = 1.0f;
            float freq = 1.0f;
            float maxAmp = 0.0f;
            
            for (int i = 0; i < octaves; i++) {
                float nx = x / scale * freq;
                float nz = z / scale * freq;
                // Add time for animation
                Vec3 p(nx + time * 0.1f, time * 0.05f, nz + time * 0.1f);
                float n = perlin.noise(p);
                
                if (noise_type == WaterWaveParams::NoiseType::Ridge) {
                    n = surf->params.geo_ridge_offset - fabsf(n);
                    n = n * n;
                }
                
                value += n * amp;
                maxAmp += amp;
                amp *= persistence;
                freq *= lacunarity;
            }
            
            if (maxAmp > 0.001f) value /= maxAmp;
            result.y = value * max_height;
        }
        
        return result;
    };
    
    // Apply animation to vertices
    size_t idx = 0;
    for (auto& tri : surf->mesh_triangles) {
        if (!tri) continue;
        
        Vec3 positions[3];
        for (int v = 0; v < 3; ++v) {
            if (idx < surf->original_positions.size()) {
                Vec3 orig = surf->original_positions[idx++];
                Vec3 disp = getAnimatedHeight(orig.x, orig.z);
                positions[v] = Vec3(orig.x + disp.x * chop, disp.y, orig.z + disp.z * chop);
            }
        }
        
        // Calculate face normal
        Vec3 edge1 = positions[1] - positions[0];
        Vec3 edge2 = positions[2] - positions[0];
        Vec3 normal = edge1.cross(edge2);
        float len = normal.length();
        if (len > 0.0001f) normal = normal / len;
        else normal = Vec3(0, 1, 0);
        
        // Apply to triangle
        for (int v = 0; v < 3; ++v) {
            tri->setVertexPosition(v, positions[v]);
            tri->setVertexNormal(v, normal);
        }
        tri->markAABBDirty();
    }
}

// ============================================================================
// APPLY KEYFRAME (for timeline animation)
// ============================================================================
void WaterManager::applyKeyframe(WaterSurface* surf, const WaterKeyframe& kf) {
    if (!surf) return;
    
    bool changed = false;
    
    // Apply only keyed properties
    if (kf.has_wave_height) {
        surf->params.geo_wave_height = kf.wave_height;
        changed = true;
    }
    if (kf.has_wave_scale) {
        surf->params.geo_wave_scale = kf.wave_scale;
        changed = true;
    }
    if (kf.has_wind_direction) {
        surf->params.geo_swell_direction = kf.wind_direction;
        changed = true;
    }
    if (kf.has_choppiness) {
        surf->params.geo_wave_choppiness = kf.choppiness;
        changed = true;
    }
    if (kf.has_alignment) {
        surf->params.geo_alignment = kf.alignment;
        changed = true;
    }
    if (kf.has_damping) {
        surf->params.geo_damping = kf.damping;
        changed = true;
    }
    if (kf.has_swell_amplitude) {
        surf->params.geo_swell_amplitude = kf.swell_amplitude;
        changed = true;
    }
    if (kf.has_sharpening) {
        surf->params.geo_sharpening = kf.sharpening;
        changed = true;
    }
    if (kf.has_detail_strength) {
        surf->params.geo_detail_strength = kf.detail_strength;
        changed = true;
    }
    
    // Rebuild mesh if any parameters changed
    if (changed && surf->params.use_geometric_waves) {
        updateWaterMesh(surf);
        
        // Signal BVH rebuild
        extern bool g_bvh_rebuild_pending;
        extern bool g_optix_rebuild_pending;
        g_bvh_rebuild_pending = true;
        g_optix_rebuild_pending = true;
    }
}

// ============================================================================
// CAPTURE KEYFRAME TO TRACK (for timeline recording)
// ============================================================================
void WaterManager::captureKeyframeToTrack(WaterSurface* surf, ObjectAnimationTrack& track, int frame) {
    if (!surf) return;
    
    Keyframe kf(frame);
    
    // Capture current water parameters
    kf.water.water_surface_id = surf->id;
    
    kf.water.wave_height = surf->params.geo_wave_height;
    kf.water.has_wave_height = true;
    
    kf.water.wave_scale = surf->params.geo_wave_scale;
    kf.water.has_wave_scale = true;
    
    kf.water.wind_direction = surf->params.geo_swell_direction;
    kf.water.has_wind_direction = true;
    
    kf.water.choppiness = surf->params.geo_wave_choppiness;
    kf.water.has_choppiness = true;
    
    kf.water.alignment = surf->params.geo_alignment;
    kf.water.has_alignment = true;
    
    kf.water.damping = surf->params.geo_damping;
    kf.water.has_damping = true;
    
    kf.water.swell_amplitude = surf->params.geo_swell_amplitude;
    kf.water.has_swell_amplitude = true;
    
    kf.water.sharpening = surf->params.geo_sharpening;
    kf.water.has_sharpening = true;
    
    kf.water.detail_strength = surf->params.geo_detail_strength;
    kf.water.has_detail_strength = true;
    
    kf.has_water = true;
    
    track.addKeyframe(kf);
}

// ============================================================================
// UPDATE FROM ANIMATION TRACK (called each frame during playback)
// ============================================================================
void WaterManager::updateFromTrack(WaterSurface* surf, const ObjectAnimationTrack& track, int currentFrame) {
    if (!surf) return;
    
    // Evaluate track at current frame (does interpolation)
    Keyframe kf = track.evaluate(currentFrame);
    
    // Apply if has water keyframe data
    // Note: Track is already scoped to this water surface via track name "Water_X"
    if (kf.has_water) {
        applyKeyframe(surf, kf.water);
    }
}

// ============================================================================
// SERIALIZATION
// ============================================================================

nlohmann::json WaterManager::serialize() const {
    nlohmann::json arr = nlohmann::json::array();
    
    for (const auto& surf : water_surfaces) {
        nlohmann::json ws;
        ws["id"] = surf.id;
        ws["name"] = surf.name;
        ws["material_id"] = surf.material_id;
        
        // Wave params
        ws["wave_speed"] = surf.params.wave_speed;
        ws["wave_strength"] = surf.params.wave_strength;
        ws["wave_frequency"] = surf.params.wave_frequency;
        
        // Colors
        ws["deep_color"] = {surf.params.deep_color.x, surf.params.deep_color.y, surf.params.deep_color.z};
        ws["shallow_color"] = {surf.params.shallow_color.x, surf.params.shallow_color.y, surf.params.shallow_color.z};
        
        // Physics
        ws["clarity"] = surf.params.clarity;
        ws["foam_level"] = surf.params.foam_level;
        ws["ior"] = surf.params.ior;
        ws["roughness"] = surf.params.roughness;
        
        // Advanced: Depth & Absorption
        ws["depth_max"] = surf.params.depth_max;
        ws["absorption_color"] = {surf.params.absorption_color.x, surf.params.absorption_color.y, surf.params.absorption_color.z};
        ws["absorption_density"] = surf.params.absorption_density;
        
        // Advanced: Shore Foam
        ws["shore_foam_distance"] = surf.params.shore_foam_distance;
        ws["shore_foam_intensity"] = surf.params.shore_foam_intensity;
        
        // Advanced: Caustics
        ws["caustic_intensity"] = surf.params.caustic_intensity;
        ws["caustic_scale"] = surf.params.caustic_scale;
        ws["caustic_speed"] = surf.params.caustic_speed;
        
        // Advanced: SSS
        ws["sss_intensity"] = surf.params.sss_intensity;
        ws["sss_color"] = {surf.params.sss_color.x, surf.params.sss_color.y, surf.params.sss_color.z};
        
        // Advanced: FFT Ocean
        ws["use_fft_ocean"] = surf.params.use_fft_ocean;
        ws["fft_resolution"] = surf.params.fft_resolution;
        ws["fft_ocean_size"] = surf.params.fft_ocean_size;
        ws["fft_wind_speed"] = surf.params.fft_wind_speed;
        ws["fft_wind_direction"] = surf.params.fft_wind_direction;
        ws["fft_choppiness"] = surf.params.fft_choppiness;
        ws["fft_amplitude"] = surf.params.fft_amplitude;
        ws["fft_time_scale"] = surf.params.fft_time_scale;
        
        // Advanced: Water Details
        ws["micro_detail_strength"] = surf.params.micro_detail_strength;
        ws["micro_detail_scale"] = surf.params.micro_detail_scale;
        ws["foam_noise_scale"] = surf.params.foam_noise_scale;
        ws["foam_threshold"] = surf.params.foam_threshold;
        
        // Geometric Displacement
        ws["use_geometric_waves"] = surf.params.use_geometric_waves;
        ws["geo_wave_height"] = surf.params.geo_wave_height;
        ws["geo_wave_scale"] = surf.params.geo_wave_scale;
        ws["geo_wave_choppiness"] = surf.params.geo_wave_choppiness;
        ws["geo_wave_speed"] = surf.params.geo_wave_speed;
        
        // Detailed Noise Params
        ws["geo_noise_type"] = (int)surf.params.geo_noise_type;
        ws["geo_octaves"] = surf.params.geo_octaves;
        ws["geo_persistence"] = surf.params.geo_persistence;
        ws["geo_lacunarity"] = surf.params.geo_lacunarity;
        ws["geo_ridge_offset"] = surf.params.geo_ridge_offset;
        
        // Blender-style Ocean Params
        ws["geo_damping"] = surf.params.geo_damping;
        ws["geo_alignment"] = surf.params.geo_alignment;
        ws["geo_depth"] = surf.params.geo_depth;
        ws["geo_swell_direction"] = surf.params.geo_swell_direction;
        ws["geo_swell_amplitude"] = surf.params.geo_swell_amplitude;
        ws["geo_sharpening"] = surf.params.geo_sharpening;
        ws["geo_detail_scale"] = surf.params.geo_detail_scale;
        ws["geo_detail_strength"] = surf.params.geo_detail_strength;
        ws["geo_smooth_normals"] = surf.params.geo_smooth_normals;
        
        // Animation state
        ws["animate_mesh"] = surf.animate_mesh;
        
        ws["type"] = (int)surf.type;

        // Position (from reference triangle or first triangle)
        if (surf.reference_triangle) {
            Vec3 v0 = surf.reference_triangle->getOriginalVertexPosition(0);
            ws["position"] = {v0.x, v0.y, v0.z};
        }

        
        // Grid info (calculate from triangles count)
        // triangles = segments * segments * 2
        size_t tri_count = surf.mesh_triangles.size();
        int segments = static_cast<int>(sqrt(tri_count / 2));
        ws["segments"] = segments;
        
        // Calculate size from triangle vertices
        if (surf.mesh_triangles.size() >= 2) {
            // First vertex of first triangle and last vertex of last triangle
            Vec3 v_first = surf.mesh_triangles[0]->getOriginalVertexPosition(0);
            Vec3 v_last = surf.mesh_triangles.back()->getOriginalVertexPosition(2);
            float size_x = std::abs(v_last.x - v_first.x);
            float size_z = std::abs(v_last.z - v_first.z);
            ws["size"] = std::max(size_x, size_z);
        }
        
        arr.push_back(ws);
    }
    
    nlohmann::json result;
    result["water_surfaces"] = arr;
    result["next_id"] = next_id;
    
    return result;
}

void WaterManager::deserialize(const nlohmann::json& j, SceneData& scene) {
    // Clear existing water surface metadata (but don't remove triangles - they're loaded from scene geometry)
    water_surfaces.clear();
    
    if (!j.contains("water_surfaces")) return;
    
    next_id = j.value("next_id", 1);
    
    for (const auto& ws : j["water_surfaces"]) {
        WaterSurface surf;
        surf.id = ws.value("id", next_id++);
        surf.name = ws.value("name", "Water_Plane_" + std::to_string(surf.id));
        surf.material_id = ws.value("material_id", 0);
        
        // Restore wave params
        surf.params.wave_speed = ws.value("wave_speed", 1.0f);
        surf.params.wave_strength = ws.value("wave_strength", 0.5f);
        surf.params.wave_frequency = ws.value("wave_frequency", 1.0f);
        surf.params.clarity = ws.value("clarity", 0.8f);
        surf.params.foam_level = ws.value("foam_level", 0.2f);
        surf.params.ior = ws.value("ior", 1.333f);
        surf.params.roughness = ws.value("roughness", 0.02f);
        
        // Colors
        if (ws.contains("deep_color")) {
            surf.params.deep_color = Vec3(ws["deep_color"][0], ws["deep_color"][1], ws["deep_color"][2]);
        }
        if (ws.contains("shallow_color")) {
            surf.params.shallow_color = Vec3(ws["shallow_color"][0], ws["shallow_color"][1], ws["shallow_color"][2]);
        }
        
        // Advanced: Depth & Absorption
        surf.params.depth_max = ws.value("depth_max", 15.0f);
        surf.params.absorption_density = ws.value("absorption_density", 0.5f);
        if (ws.contains("absorption_color")) {
            surf.params.absorption_color = Vec3(ws["absorption_color"][0], ws["absorption_color"][1], ws["absorption_color"][2]);
        }
        
        // Advanced: Shore Foam
        surf.params.shore_foam_distance = ws.value("shore_foam_distance", 1.5f);
        surf.params.shore_foam_intensity = ws.value("shore_foam_intensity", 0.6f);
        
        // Advanced: Caustics
        surf.params.caustic_intensity = ws.value("caustic_intensity", 0.4f);
        surf.params.caustic_scale = ws.value("caustic_scale", 2.0f);
        surf.params.caustic_speed = ws.value("caustic_speed", 1.0f);
        
        // Advanced: SSS
        surf.params.sss_intensity = ws.value("sss_intensity", 0.15f);
        if (ws.contains("sss_color")) {
            surf.params.sss_color = Vec3(ws["sss_color"][0], ws["sss_color"][1], ws["sss_color"][2]);
        }
        
        // Advanced: FFT Ocean
        surf.params.use_fft_ocean = ws.value("use_fft_ocean", false);
        surf.params.fft_resolution = ws.value("fft_resolution", 256);
        surf.params.fft_ocean_size = ws.value("fft_ocean_size", 100.0f);
        surf.params.fft_wind_speed = ws.value("fft_wind_speed", 10.0f);
        surf.params.fft_wind_direction = ws.value("fft_wind_direction", 0.0f);
        surf.params.fft_choppiness = ws.value("fft_choppiness", 1.0f);
        surf.params.fft_amplitude = ws.value("fft_amplitude", 0.0002f);
        surf.params.fft_time_scale = ws.value("fft_time_scale", 1.0f);
        
        // Advanced: Water Details
        surf.params.micro_detail_strength = ws.value("micro_detail_strength", 0.05f);
        surf.params.micro_detail_scale = ws.value("micro_detail_scale", 20.0f);
        surf.params.foam_noise_scale = ws.value("foam_noise_scale", 4.0f);
        surf.params.foam_threshold = ws.value("foam_threshold", 0.4f);
        
        // Geometric Displacement
        if (ws.contains("type")) {
            surf.type = (WaterSurface::Type)ws.value("type", (int)WaterSurface::Type::Plane);
        } else {
            surf.type = WaterSurface::Type::Plane;
        }
        
        surf.params.use_geometric_waves = ws.value("use_geometric_waves", false);
        surf.params.geo_wave_height = ws.value("geo_wave_height", 2.0f);
        surf.params.geo_wave_scale = ws.value("geo_wave_scale", 50.0f);
        surf.params.geo_wave_choppiness = ws.value("geo_wave_choppiness", 1.0f);
        surf.params.geo_wave_speed = ws.value("geo_wave_speed", 0.5f);
        
        // Detailed Noise Params
        surf.params.geo_noise_type = (WaterWaveParams::NoiseType)ws.value("geo_noise_type", (int)WaterWaveParams::NoiseType::Ridge);
        surf.params.geo_octaves = ws.value("geo_octaves", 4);
        surf.params.geo_persistence = ws.value("geo_persistence", 0.5f);
        surf.params.geo_lacunarity = ws.value("geo_lacunarity", 2.0f);
        surf.params.geo_ridge_offset = ws.value("geo_ridge_offset", 1.0f);
        
        // Geometric Ocean Params
        surf.params.geo_damping = ws.value("geo_damping", 0.0f);
        surf.params.geo_alignment = ws.value("geo_alignment", 0.5f);
        surf.params.geo_depth = ws.value("geo_depth", 200.0f);
        surf.params.geo_swell_direction = ws.value("geo_swell_direction", 0.0f);
        surf.params.geo_swell_amplitude = ws.value("geo_swell_amplitude", 0.2f);
        surf.params.geo_sharpening = ws.value("geo_sharpening", 0.0f);
        surf.params.geo_detail_scale = ws.value("geo_detail_scale", 3.0f);
        surf.params.geo_detail_strength = ws.value("geo_detail_strength", 0.15f);
        surf.params.geo_smooth_normals = ws.value("geo_smooth_normals", true);
        
        // Animation state
        surf.animate_mesh = ws.value("animate_mesh", false);

        // Find existing triangles in scene by nodeName (don't create new ones!)
        for (auto& obj : scene.world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->nodeName == surf.name) {
                surf.mesh_triangles.push_back(tri);
                if (!surf.reference_triangle) {
                    surf.reference_triangle = tri;
                }
            }
        }
        
        // Update material with restored params (material should already exist from scene load)
        if (surf.material_id > 0) {
            auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
            if (mat && mat->gpuMaterial) {
                auto& gpu = mat->gpuMaterial;
                
                // Base material properties
                gpu->albedo = make_float3(surf.params.deep_color.x, surf.params.deep_color.y, surf.params.deep_color.z);
                gpu->transmission = 1.0f;
                gpu->opacity = 1.0f;
                gpu->metallic = 0.0f;
                gpu->roughness = surf.params.roughness;
                gpu->ior = surf.params.ior;
                
                // Wave params
                gpu->anisotropic = surf.params.wave_speed;
                gpu->sheen = std::fmax(0.001f, surf.params.wave_strength);
                gpu->sheen_tint = surf.params.wave_frequency;
                
                // Advanced params
                gpu->clearcoat = surf.params.shore_foam_intensity;
                gpu->clearcoat_roughness = surf.params.caustic_intensity;
                gpu->subsurface = surf.params.depth_max / 100.0f;
                gpu->subsurface_scale = surf.params.absorption_density;
                gpu->subsurface_color = make_float3(surf.params.absorption_color.x, surf.params.absorption_color.y, surf.params.absorption_color.z);
                gpu->subsurface_radius = make_float3(surf.params.shore_foam_distance, surf.params.caustic_scale, surf.params.sss_intensity);
                gpu->emission = make_float3(surf.params.shallow_color.x, surf.params.shallow_color.y, surf.params.shallow_color.z);
                gpu->translucent = surf.params.foam_level;
                gpu->subsurface_anisotropy = surf.params.caustic_speed;
                
                // Water Details (New)
                gpu->micro_detail_strength = surf.params.micro_detail_strength;
                gpu->micro_detail_scale = surf.params.micro_detail_scale;
                gpu->foam_noise_scale = surf.params.foam_noise_scale;
                gpu->foam_threshold = surf.params.foam_threshold;
                
                // FFT
                gpu->fft_ocean_size = surf.params.fft_ocean_size;
                gpu->fft_choppiness = surf.params.fft_choppiness;
            }
        }
        
        water_surfaces.push_back(std::move(surf));
    }
    
    SCENE_LOG_INFO("[WaterManager] Loaded " + std::to_string(water_surfaces.size()) + " water surfaces (using existing geometry).");
}
