#include "Material.h"

// Implementations of Material member functions go here
bool Material::volumetric_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    
    return false;
}
 bool Material::sss_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    
     return false;
}
bool Material::sss_random_walk_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    return false;
}
bool Material::clearcoat_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    return false;
}
bool Material::translucent_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    return false;
}
 Vec3 Material::getAnisotropicDirection() const {
     
      return Vec3(1, 0, 0);
  }
  float Material::getAnisotropy() const {
      
      return 0.0f; 
  }

  Vec3 Material::getEmission() const {
   
       return Vec3(0, 0, 0);
   }
