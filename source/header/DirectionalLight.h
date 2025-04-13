#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include "Light.h"
#include "Vec3.h"
#include "Vec3SIMD.h"
#include <random>

class DirectionalLight : public Light {
public:
    DirectionalLight( const Vec3& dir, const Vec3& intens, double radius);

    Vec3 getDirection(const Vec3& point) const override {
        return -direction.normalize(); // Iþýk yönünün tersi
    }

    Vec3 getIntensity(const Vec3& point) const override {
        return intensity; // Mesafeden baðýmsýz
    }
    Vec3 getPosition() const  {
        return position; 
    }
    Vec3 random_point() const override;
    LightType type() const override;
    void setPosition(const Vec3& pos) {
        position = pos;
    }
    void setDirection(const Vec3& dir) {
        position = dir;
    }
private:
    double disk_radius; // Disk'in yarýçapý
};

#endif // DIRECTIONAL_LIGHT_H
