#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include "Light.h"
#include "Vec3.h"
#include "Vec3SIMD.h"
#include <random>

class DirectionalLight : public Light {
public:
    DirectionalLight( const Vec3& dir, const Vec3& intens, double radius);
    int getSampleCount() const override;
    mutable Vec3 last_sampled_point;
    Vec3 getDirection(const Vec3& point) const override {
        return -direction.normalize(); // Iþýk yönünün tersi
    }

    Vec3 getIntensity(const Vec3& point, const Vec3& light_sample_point) const override {
        return intensity; // Mesafeden baðýmsýz
    }
    Vec3 getPosition() const  {
        return position; 
    }
    Vec3 random_point() const override;
    LightType type() const override;
    float pdf(const Vec3& hit_point, const Vec3& incoming_direction) const override;
    void setPosition(const Vec3& pos) {
        position = pos;
    }
    void setDirection(const Vec3& dir) {
        position = dir;
    }
    
    double disk_radius; // Disk'in yarýçapý
    float radius= disk_radius; // Disk'in yarýçapý
private:
   
  

};

#endif // DIRECTIONAL_LIGHT_H
