#pragma once
#include "Vec3.h"
#include <cuda_runtime.h>


// GPU-Compatible Enum
enum WorldMode {
    WORLD_MODE_COLOR = 0,
    WORLD_MODE_HDRI = 1,
    WORLD_MODE_NISHITA = 2
};

// GPU-Compatible Structs
struct NishitaSkyParams {
    float3 sun_direction;
    float sun_elevation;
    float sun_azimuth;
    float sun_intensity;
    float sun_density;
    float dust_density;
    float planet_radius;
    float atmosphere_height;
    float3 rayleigh_scattering;
    float3 mie_scattering;
    float mie_anisotropy; // g
    float rayleigh_density;
    float mie_density;
};

struct WorldData {
    int mode; // WorldMode
    
    // Solid Color Mode
    float3 color;
    float color_intensity;

    // HDRI Mode
    cudaTextureObject_t env_texture;
    float env_rotation; // Rotation in radians
    float env_intensity;
    int env_width;      // For importance sampling (future)
    int env_height;

    // Nishita Mode
    NishitaSkyParams nishita;

    // Volume (Placeholder for later)
    float volume_density;
    float volume_anisotropy;
};

#ifndef __CUDACC__
#include <optional>
#include <string>
#include <vector>
class Texture;

class World {
public:
    World();
    ~World();

    WorldData getGPUData() const;

    // Setters
    void setMode(WorldMode mode);
    WorldMode getMode() const;
    std::string getHDRIPath() const;
    void setNishitaParams(const NishitaSkyParams& params);
    
    // Color Mode
    void setColor(const Vec3& color);
    void setColorIntensity(float intensity);
    Vec3 getColor() const;
    float getColorIntensity() const;

    // HDRI Mode
    void setHDRI(const std::string& path);
    void setHDRIRotation(float rotation_degrees);
    void setHDRIIntensity(float intensity);
    float getHDRIRotation() const;
    float getHDRIIntensity() const;
    bool hasHDRI() const;

    // Nishita Mode
    void setSunDirection(const Vec3& direction);
    void setSunIntensity(float intensity);
    void setPlanetRadius(float radius);
    void setAtmosphereHeight(float height);
    void setRayleighScattering(const Vec3& scattering);
    void setMieScattering(const Vec3& scattering);
    void setMieAnisotropy(float g);
    void setDustDensity(float density);
    
    NishitaSkyParams getNishitaParams() const;

    // CPU Evaluation (for background missing)
    Vec3 evaluate(const Vec3& ray_dir);

private:
   WorldData data;
   std::string hdri_path; // Store path for getter
   Texture* hdri_texture = nullptr;
   
   // Internal helper for Nishita
   Vec3 calculateNishitaSky(const Vec3& ray_dir);
};
#endif
