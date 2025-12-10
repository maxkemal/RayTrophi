#include "EmissiveMaterial.h"

EmissiveMaterial::EmissiveMaterial(const Vec3& emit)
    : emission(emit) {}

bool EmissiveMaterial::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    return false; // Emissive materials do not scatter light
}

Vec3 EmissiveMaterial::getEmission(const Vec2& uv, const Vec3& p) const {
    return emission;
}
float EmissiveMaterial::get_opacity(const Vec2& uv) const {
    return 1.0f;  // Dielectric materyal tamamen opak, bu yüzden 1.0 döndür
}