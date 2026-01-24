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
    near_dist = 0.1; // Yakın mesafe, genellikle 0.1 olarak ayarlanır
    far_dist = focus_dist * 2.0; // Uzak mesafe, odak uzaklığının iki katı olarak ayarlanır
    update_camera_vectors();
}

Camera::Camera()
    : aperture(0.0), aspect(0.0), aspect_ratio(0.0), blade_count(0), far_dist(0.0),
    focus_dist(0.0), fov(0.0), lens_radius(0.0), near_dist(0.0), vfov(0.0) {
}
Ray Camera::get_ray(float s, float t) const {
    float use_s = s;
    float use_t = t;

    // ---------------- LENS DISTORTION (CPU) ----------------
    if (std::abs(distortion) > 0.001f) {
        float h_len = horizontal.length();
        float v_len = vertical.length();
        float aspect = h_len / (v_len + 1e-6f);
        
        float u_centered = (s - 0.5f) * aspect;
        float v_centered = (t - 0.5f);
        
        float r2 = u_centered * u_centered + v_centered * v_centered;
        float factor = 1.0f + distortion * r2;
        
        u_centered *= factor;
        v_centered *= factor;
        
        use_s = (u_centered / aspect) + 0.5f;
        use_t = v_centered + 0.5f;
    }

    Vec3 rd = lens_radius * random_in_unit_polygon(blade_count);
    Vec3 offset = u * rd.x + v * rd.y;
    return Ray(origin + offset, lower_left_corner + use_s * horizontal + use_t * vertical - origin - offset);
}

#include "RayPacket.h"
void Camera::get_ray_packet(__m256 s, __m256 t, RayPacket& rp) const {
    __m256 use_s = s;
    __m256 use_t = t;

    // Lens Distortion (SIMD)
    if (std::abs(distortion) > 0.001f) {
        float h_len = horizontal.length();
        float v_len = vertical.length();
        __m256 aspect_v = _mm256_set1_ps(h_len / (v_len + 1e-6f));
        __m256 dist_v = _mm256_set1_ps(distortion);
        
        __m256 u_centered = _mm256_mul_ps(_mm256_sub_ps(s, _mm256_set1_ps(0.5f)), aspect_v);
        __m256 v_centered = _mm256_sub_ps(t, _mm256_set1_ps(0.5f));
        
        __m256 r2 = _mm256_add_ps(_mm256_mul_ps(u_centered, u_centered), _mm256_mul_ps(v_centered, v_centered));
        __m256 factor = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(dist_v, r2));
        
        u_centered = _mm256_mul_ps(u_centered, factor);
        v_centered = _mm256_mul_ps(v_centered, factor);
        
        use_s = _mm256_add_ps(_mm256_div_ps(u_centered, aspect_v), _mm256_set1_ps(0.5f));
        use_t = _mm256_add_ps(v_centered, _mm256_set1_ps(0.5f));
    }

    // Depth of Field (Thin Lens Model)
    // For simplicity in the first version of packet tracing, we splat one random offset for the whole packet
    // to avoid heavy random generation. Alternatively, we generate 8 offsets.
    // Let's generate 8 for better quality.
    
    alignas(32) float rd_x[8], rd_y[8];
    for(int i=0; i<8; ++i) {
        Vec3 rd = lens_radius * random_in_unit_polygon(blade_count);
        rd_x[i] = rd.x;
        rd_y[i] = rd.y;
    }
    
    Vec3SIMD offset_x = Vec3SIMD(u.x) * Vec3SIMD(rd_x) + Vec3SIMD(v.x) * Vec3SIMD(rd_y);
    Vec3SIMD offset_y = Vec3SIMD(u.y) * Vec3SIMD(rd_x) + Vec3SIMD(v.y) * Vec3SIMD(rd_y);
    Vec3SIMD offset_z = Vec3SIMD(u.z) * Vec3SIMD(rd_x) + Vec3SIMD(v.z) * Vec3SIMD(rd_y);

    rp.orig_x = Vec3SIMD(origin.x) + offset_x;
    rp.orig_y = Vec3SIMD(origin.y) + offset_y;
    rp.orig_z = Vec3SIMD(origin.z) + offset_z;

    Vec3SIMD target_x = Vec3SIMD(lower_left_corner.x) + Vec3SIMD(use_s) * Vec3SIMD(horizontal.x) + Vec3SIMD(use_t) * Vec3SIMD(vertical.x);
    Vec3SIMD target_y = Vec3SIMD(lower_left_corner.y) + Vec3SIMD(use_s) * Vec3SIMD(horizontal.y) + Vec3SIMD(use_t) * Vec3SIMD(vertical.y);
    Vec3SIMD target_z = Vec3SIMD(lower_left_corner.z) + Vec3SIMD(use_s) * Vec3SIMD(horizontal.z) + Vec3SIMD(use_t) * Vec3SIMD(vertical.z);

    rp.dir_x = target_x - rp.orig_x;
    rp.dir_y = target_y - rp.orig_y;
    rp.dir_z = target_z - rp.orig_z;

    rp.update_derived_data();
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

    float theta = vfov * static_cast<float>(M_PI) / 180.0;
    float half_height = tan(theta / 2.0);
    float half_width = aspect_ratio * half_height;

    horizontal = 2.0f * half_width * focus_dist * u;
    vertical = 2.0f * half_height * focus_dist * v;
    lower_left_corner = lookfrom - horizontal * 0.5 - vertical * 0.5 - focus_dist * w;

    origin = lookfrom;
    fov = vfov;

    // ---------------- PHYSICAL LENS DEFECTS (Auto-Calculated) ----------------
    // Calculate distortion based on estimated focal length (assuming 35mm full frame)
    // f = h / (2 * tan(vfov/2))
    // We use a fixed sensor height reference (24mm) to map "Wide Angle" feel consistently.
    // This allows the distortion to feel "physically correct" relative to standard FOV.
    
    // Note: theta is already calculated above as (vfov * PI / 180)
    float est_f = (24.0f / 2.0f) / tan(theta / 2.0f);
    
    if (est_f < 50.0f) {
         // Barrel Distortion for Wide Angle (< 50mm)
         // Quadratic curve: stronger effect as f decreases (e.g. 16mm is visible)
         float ratio = (50.0f - est_f) / 50.0f;
         distortion = -0.25f * (ratio * ratio); 
    } else {
         // Pincushion Distortion for Telephoto (> 50mm)
         // Linear, mild effect
         float ratio = (est_f - 50.0f) / 200.0f;
         distortion = 0.05f * ratio;
         if (distortion > 0.05f) distortion = 0.05f; // Cap at max
    }
    // -------------------------------------------------------------------------

    updateFrustumPlanes();
}
void Camera::moveToTargetLocked(const Vec3& new_position) {
    Vec3 view_dir = lookat - lookfrom;
    lookfrom = new_position;
    origin = new_position;
    lookat = new_position + view_dir; // yön sabit kalır
    update_camera_vectors();
}
// Bu metodu da ekleyebilirsin
void Camera::setLookDirection(const Vec3& direction_normalized) {
    lookat = lookfrom + direction_normalized * focus_dist;
    update_camera_vectors();
}

Vec3 Camera::random_in_unit_polygon(int sides) const {
    float step = 2 * static_cast<float>(M_PI) / sides;

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

    // İçeri rastgele bir kaydırma yaparak tam dolu hale getir
    float shrink_factor = sqrt(Vec3::random_float()); // İç içe eşit doluluk için
    px *= shrink_factor;
    py *= shrink_factor;

    return Vec3(px, py, 0);
}
// Yeni fonksiyon: Bokeh şiddetini hesapla
float Camera::calculate_bokeh_intensity(const Vec3& point) const {
    float distance = (point - origin).length();
    float focal_plane_distance = focus_dist;

    // Blur faktörünü daha yumuşak bir şekilde hesapla
    float blur_factor = std::abs(distance - focal_plane_distance) / focal_plane_distance;

    // Blur faktörünü daha kontrollü hale getir
    float scaled_aperture = aperture * 10.0f; // aperture etkisini ölçekle
    return std::min(1.0f, blur_factor * scaled_aperture);
}
// Işık kaynakları için özel bokeh şekli oluştur
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
    float near_height = 2 * tan(fov * 0.5f * static_cast<float>(M_PI) / 180) * near_dist;
    float far_height = 2 * tan(fov * 0.5f * static_cast<float>(M_PI) / 180) * far_dist;
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
    return -w;  // Kameranın baktığı yön w vektörünün tersidir
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
            return false;  // AABB frustum dışında
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

