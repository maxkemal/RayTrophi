#include "DiffuseLight.h"

DiffuseLight::DiffuseLight(Vec3 c)
    : emit(c) {}

bool DiffuseLight::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    return false; // DiffuseLight materials do not scatter light
}

Vec3 DiffuseLight::getEmission(const Vec2& uv, const Vec3& p) const {
    return emit;
}
float DiffuseLight::get_opacity(const Vec2& uv) const {
    return 1.0f;  // Dielectric materyal tamamen opak
}
