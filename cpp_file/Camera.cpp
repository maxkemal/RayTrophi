#include "Camera.h"
#include <cmath>
#include <stdlib.h>
#include "Matrix4x4.h"
Camera::Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double vfov, double aspect, double aperture, double focus_dist, int blade_count) {
    double theta = vfov * M_PI / 180;
    double half_height = tan(theta / 2);
    double half_width = aspect * half_height;
    origin = lookfrom;
    w = (lookfrom - lookat).normalize();
    u = Vec3::cross(vup, w).normalize();
    v = Vec3::cross(w, u);
    lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
    horizontal = 2 * half_width * focus_dist * u;
    vertical = 2 * half_height * focus_dist * v;
    lens_radius = aperture / 2;

    // Frustum culling için ek alanlar
    near_dist = 0.1;  // Yakýn düzlem mesafesi, ihtiyaca göre ayarlayýn
    far_dist = focus_dist * 2;  // Uzak düzlem mesafesi, ihtiyaca göre ayarlayýn
    fov = vfov;
    aspect_ratio = aspect;
    updateFrustumPlanes();
    this->aperture = aperture;
    this->blade_count = blade_count; // Býçak sayýsýný sakla
}
Camera::Camera() 
    : aperture(0.0), aspect(0.0), aspect_ratio(0.0), blade_count(0), far_dist(0.0), 
      focus_dist(0.0), fov(0.0), lens_radius(0.0), near_dist(0.0), vfov(0.0) {
}
Ray Camera::get_ray(double s, double t) const {
    Vec3 rd = lens_radius * random_in_unit_polygon(blade_count);
    Vec3 offset = u * rd.x + v * rd.y;
    return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
}
int Camera::random_int(int min, int max) const {
    static std::random_device rd;
    static std::mt19937 gen(rd());  // Mersenne Twister RNG
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}
void Camera::update_camera_vectors() {
    Vec3 view = lookat - lookfrom;
    if (view.length_squared() < 1e-8) {
        view = Vec3(0, 0, -1);
    }
    w = (-view).normalize();

    // Eðer vup sýfýrsa veya w ile paralelse düzelt!
    if (vup.length_squared() < 1e-8 || std::abs(Vec3::dot(vup.normalize(), w)) > 0.999) {
        vup = Vec3(0, 1, 0); // Zorunlu varsayýlan yukarý yönü
    }

    u = Vec3::cross(vup, w).normalize();
    v = Vec3::cross(w, u).normalize();

    double theta = vfov * M_PI / 180.0;
    double half_height = tan(theta / 2.0);
    double half_width = aspect * half_height;

    horizontal = 2.0 * half_width * focus_dist * u;
    vertical = 2.0 * half_height * focus_dist * v;
    lower_left_corner = lookfrom - horizontal * 0.5 - vertical * 0.5 - focus_dist * w;
}

Vec3 Camera::random_in_unit_polygon(int sides) const {
    double step = 2 * M_PI / sides;

    // Rastgele bir kenar seç
    int edge_index = random_int(0, sides - 1);
    double edge_angle = edge_index * step;

    // Kenar boyunca rastgele bir nokta seç
    double t = random_double();
    double x1 = cos(edge_angle);
    double y1 = sin(edge_angle);

    double x2 = cos(edge_angle + step);
    double y2 = sin(edge_angle + step);

    // Rastgele nokta, kenar boyunca interpolasyon
    double px = x1 * (1 - t) + x2 * t;
    double py = y1 * (1 - t) + y2 * t;

    // Ýçeri rastgele bir kaydýrma yaparak tam dolu hale getir
    double shrink_factor = sqrt(random_double()); // Ýç içe eþit doluluk için
    px *= shrink_factor;
    py *= shrink_factor;

    return Vec3SIMD(px, py, 0);
}
// Yeni fonksiyon: Bokeh þiddetini hesapla
double Camera::calculate_bokeh_intensity(const Vec3& point) const {
    double distance = (point - origin).length();
    double focal_plane_distance = focus_dist;

    // Blur faktörünü daha yumuþak bir þekilde hesapla
    double blur_factor = std::abs(distance - focal_plane_distance) / focal_plane_distance;

    // Blur faktörünü daha kontrollü hale getir
    double scaled_aperture = aperture * 10.0; // aperture etkisini ölçekle
    return std::min(1.0, blur_factor * scaled_aperture);
}
// Iþýk kaynaklarý için özel bokeh þekli oluþtur
Vec3 Camera::create_bokeh_shape(const Vec3& color, double intensity) const {
    Vec3 bokeh_color = color * intensity;
    Vec3 shape = random_in_unit_polygon(blade_count);
    return bokeh_color * (shape * 0.5 + Vec3(0.5, 0.5, 0.5));
}

bool Camera::isPointInFrustum(const Vec3& point, double size) const {
    for (const auto& plane : frustum_planes) {
        if (plane.distanceToPoint(point) < -size) {
            return false;  // Point is outside the frustum
        }
    }
    return true;  // Point is inside or intersects the frustum
}

void Camera::updateFrustumPlanes() {
    // Frustum düzlemlerini hesapla
    Vec3 fc = origin - w * far_dist;
    float near_height = 2 * tan(fov * 0.5f * M_PI / 180) * near_dist;
    float far_height = 2 * tan(fov * 0.5f * M_PI / 180) * far_dist;
    float near_width = near_height * aspect_ratio;
    float far_width = far_height * aspect_ratio;

    Vec3 ntl = origin - w * near_dist - u * (near_width * 0.5f) + v * (near_height * 0.5f);
    Vec3 ntr = origin - w * near_dist + u * (near_width * 0.5f) + v * (near_height * 0.5f);
    Vec3 nbl = origin - w * near_dist - u * (near_width * 0.5f) - v * (near_height * 0.5f);
    Vec3 ftr = fc + u * (far_width * 0.5f) + v * (far_height * 0.5f);

    frustum_planes[0] = Plane(Vec3::cross(ntl - ntr, ntl - ftr).normalize(), ntl);  // top
    frustum_planes[1] = Plane(Vec3::cross(nbl - ntl, nbl - fc).normalize(), nbl);   // left
    frustum_planes[2] = Plane(w, origin - w * near_dist);                           // near
    frustum_planes[3] = Plane(Vec3::cross(ntr - ntl, ntr - fc).normalize(), ntr);   // right
    frustum_planes[4] = Plane(Vec3::cross(nbl - ntr, nbl - fc).normalize(), nbl);   // bottom
    frustum_planes[5] = Plane(-w, fc);                                              // far
}
Vec3 Camera::getViewDirection() const {
    return -w;  // Kameranýn baktýðý yön w vektörünün tersidir
}

Matrix4x4 Camera::getRotationMatrix() const {
    Matrix4x4 rotationMatrix;
    rotationMatrix.m[0][0] = u.x; rotationMatrix.m[0][1] = u.y; rotationMatrix.m[0][2] = u.z; rotationMatrix.m[0][3] = 0;
    rotationMatrix.m[1][0] = v.x; rotationMatrix.m[1][1] = v.y; rotationMatrix.m[1][2] = v.z; rotationMatrix.m[1][3] = 0;
    rotationMatrix.m[2][0] = w.x; rotationMatrix.m[2][1] = w.y; rotationMatrix.m[2][2] = w.z; rotationMatrix.m[2][3] = 0;
    rotationMatrix.m[3][0] = 0;   rotationMatrix.m[3][1] = 0;   rotationMatrix.m[3][2] = 0;   rotationMatrix.m[3][3] = 1;
    return rotationMatrix;
}
bool Camera::isAABBInFrustum(const AABB& aabb) const {
    for (const auto& plane : frustum_planes) {
        if (plane.distanceToPoint(aabb.getPositiveVertex(plane.normal)) < 0) {
            return false;  // AABB frustum dýþýnda
        }
    }
    return true;  // AABB frustum içinde
}

std::vector<AABB> Camera::performFrustumCulling(const std::vector<AABB>& objects) const {
    std::vector<AABB> visibleObjects;
    for (const auto& obj : objects) {
        if (isAABBInFrustum(obj)) {
            visibleObjects.push_back(obj);
        }
    }
    return visibleObjects;
}
