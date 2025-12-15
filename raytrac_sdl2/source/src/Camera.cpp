#include "Camera.h"
#include <cmath>
#include <stdlib.h>
#include "Matrix4x4.h"
Camera::Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect, float aperture, float focus_dist, int blade_count)
    : lookfrom(lookfrom), lookat(lookat), vup(vup), vfov(vfov),
    aspect_ratio(aspect), aperture(aperture), focus_dist(focus_dist),
    blade_count(blade_count), fov(vfov), origin(lookfrom)
{
    lens_radius = aperture * 0.5f;
    near_dist = 0.1; // Yakýn mesafe, genellikle 0.1 olarak ayarlanýr
    far_dist = focus_dist * 2.0; // Uzak mesafe, odak uzaklýðýnýn iki katý olarak ayarlanýr
    update_camera_vectors();
}

Camera::Camera()
    : aperture(0.0), aspect(0.0), aspect_ratio(0.0), blade_count(0), far_dist(0.0),
    focus_dist(0.0), fov(0.0), lens_radius(0.0), near_dist(0.0), vfov(0.0) {
}
Ray Camera::get_ray(float s, float t) const {
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
    if (view.length_squared() < 1e-8)
        view = Vec3(0, 0, -1);

    w = (-view).normalize();
    u = Vec3::cross(vup, w).normalize();
    v = Vec3::cross(w, u).normalize();

    float theta = vfov * M_PI / 180.0;
    float half_height = tan(theta / 2.0);
    float half_width = aspect_ratio * half_height;

    horizontal = 2.0 * half_width * focus_dist * u;
    vertical = 2.0 * half_height * focus_dist * v;
    lower_left_corner = lookfrom - horizontal * 0.5 - vertical * 0.5 - focus_dist * w;

    origin = lookfrom;
    fov = vfov;

    updateFrustumPlanes();
}
void Camera::moveToTargetLocked(const Vec3& new_position) {
    Vec3 view_dir = lookat - lookfrom;
    lookfrom = new_position;
    origin = new_position;
    lookat = new_position + view_dir; // yön sabit kalýr
    update_camera_vectors();
}
// Bu metodu da ekleyebilirsin
void Camera::setLookDirection(const Vec3& direction_normalized) {
    lookat = lookfrom + direction_normalized * focus_dist;
    update_camera_vectors();
}

Vec3 Camera::random_in_unit_polygon(int sides) const {
    float step = 2 * M_PI / sides;

    // Rastgele bir kenar seç
    int edge_index = random_int(0, sides - 1);
    float edge_angle = edge_index * step;

    // Kenar boyunca rastgele bir nokta seç
    float t = Vec3::random_float();
    float x1 = cos(edge_angle);
    float y1 = sin(edge_angle);

    float x2 = cos(edge_angle + step);
    float y2 = sin(edge_angle + step);

    // Rastgele nokta, kenar boyunca interpolasyon
    float px = x1 * (1 - t) + x2 * t;
    float py = y1 * (1 - t) + y2 * t;

    // Ýçeri rastgele bir kaydýrma yaparak tam dolu hale getir
    float shrink_factor = sqrt(Vec3::random_float()); // Ýç içe eþit doluluk için
    px *= shrink_factor;
    py *= shrink_factor;

    return Vec3(px, py, 0);
}
// Yeni fonksiyon: Bokeh þiddetini hesapla
float Camera::calculate_bokeh_intensity(const Vec3& point) const {
    float distance = (point - origin).length();
    float focal_plane_distance = focus_dist;

    // Blur faktörünü daha yumuþak bir þekilde hesapla
    float blur_factor = std::abs(distance - focal_plane_distance) / focal_plane_distance;

    // Blur faktörünü daha kontrollü hale getir
    float scaled_aperture = aperture * 10.0f; // aperture etkisini ölçekle
    return std::min(1.0f, blur_factor * scaled_aperture);
}
// Iþýk kaynaklarý için özel bokeh þekli oluþtur
Vec3 Camera::create_bokeh_shape(const Vec3& color, float intensity) const {
    Vec3 bokeh_color = color * intensity;
    Vec3 shape = random_in_unit_polygon(blade_count);
    return bokeh_color * (shape * 0.5 + Vec3(0.5, 0.5, 0.5));
}

bool Camera::isPointInFrustum(const Vec3& point, float size) const {
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
void Camera::reset() {
    lookfrom = init_lookfrom;
    lookat = init_lookat;
    vup = init_vup;
    vfov = init_vfov;
    aperture = init_aperture;
    focus_dist = init_focus_dist;
    update_camera_vectors();
}
void Camera::save_initial_state() {
    init_lookfrom = lookfrom;
    init_lookat = lookat;
    init_vup = vup;
    init_vfov = vfov;
    init_aperture = aperture;
    init_focus_dist = focus_dist;
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
