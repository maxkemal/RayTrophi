#include "Vec3.h"
#include <cstdlib>
#include <limits>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <thread>

// --- Random Number Generator Optimization (XorShift32) ---
// Thread-local state for lock-free, fast generation
static thread_local uint32_t s_rng_state = 0;

void init_rng_if_needed() {
    if (s_rng_state == 0) {
        // Seed mixing: Time + Thread ID + Address
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        size_t h_thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
        uint32_t seed = static_cast<uint32_t>(now ^ h_thread);
        s_rng_state = (seed == 0) ? 0xDEADBEEF : seed;
    }
}

// random_float implementation using XorShift32
float Vec3::random_float(float min, float max) {
    if (s_rng_state == 0) init_rng_if_needed();

    uint32_t x = s_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s_rng_state = x;
    
    // Convert to [0, 1) float directly (faster than division)
    // 0x7FFFFFFF is max int, but we can just map to 23 mantissa bits for extreme speed
    // or standard division. Standard division is safe and fast enough here.
    float r = float(x) * 2.3283064365386963e-10f; // 1 / 2^32
    
    return min + (max - min) * r;
}

Vec3 Vec3::random() { 
    return Vec3(random_float(), random_float(), random_float()); 
}

Vec3 Vec3::random(float min, float max) {
    return Vec3(random_float(min, max), random_float(min, max), random_float(min, max));
}

Vec3 Vec3::random_in_unit_sphere() {
    // Rejection Sampling (Much faster than sin/cos/pow)
    // Cube volume = 8, Sphere volume = 4.18. Ratio ~0.52.
    // Average iterations: ~1.9. Very fast.
    while (true) {
        Vec3 p = Vec3::random(-1.0f, 1.0f);
        if (p.length_squared() >= 1.0f) continue;
        return p;
    }
}

Vec3 Vec3::random_in_hemisphere(const Vec3& normal) {
    Vec3 in_unit_sphere = random_in_unit_sphere();
    if (dot(in_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

Vec3 Vec3::random_unit_vector() {
    // Lambertian Sphere Sampling: Normalize point in unit sphere
    return random_in_unit_sphere().normalize();
}

Vec3 Vec3::random_in_unit_disk() {
    while (true) {
        Vec3 p = Vec3(random_float(-1.0f, 1.0f), random_float(-1.0f, 1.0f), 0.0f);
        if (p.length_squared() >= 1.0f) continue;
        return p;
    }
}

Vec3 Vec3::random_cosine_direction(const Vec3& normal) {
    // Malley's Method (Concentric Disk Sampling -> Hemisphere)
    // Standard cosine weighted sampling
    float r1 = random_float();
    float r2 = random_float();
    
    float z = std::sqrt(1.0f - r2);
    float phi = 2.0f * static_cast<float>(M_PI) * r1;
    float x = std::cos(phi) * std::sqrt(r2);
    float y = std::sin(phi) * std::sqrt(r2);
    
    // Orthonormal Basis Construction (Duff et al.)
    // More numerically stable than cross product with arbitrary axis
    float sign = std::copysign(1.0f, normal.z);
    float a = -1.0f / (1.0f + std::abs(normal.z));
    float b = normal.x * normal.y * a;
    
    Vec3 tangent = Vec3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    Vec3 bitangent = Vec3(b, sign + normal.y * normal.y * a, -normal.y);
    
    // Note: If normal is roughly Z up, tangent~X, bitangent~Y.
    // However, sticking to consistent ONB method helps.
    // Reverting to the simpler cross product method if this is too complex for now,
    // but Duff's method is better for path tracing.
    // Let's stick to the existing valid ONB logic or a simpler one:
    
    Vec3 up = std::abs(normal.z) < 0.999f ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
    Vec3 t = up.cross(normal).normalize();
    Vec3 bt = normal.cross(t);
    
    return t * x + bt * y + normal * z;
}

Vec3 Vec3::sample_hemisphere_cosine_weighted(const Vec3& normal, float u, float v) {
    // Equivalent to random_cosine_direction but with explicit u/v
    float z = std::sqrt(std::max(0.0f, 1.0f - u));
    float r = std::sqrt(u);
    float phi = 2.0f * static_cast<float>(M_PI) * v;
    
    float x = r * std::cos(phi);
    float y = r * std::sin(phi);

    Vec3 up = std::abs(normal.z) < 0.999f ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
    Vec3 tangent = up.cross(normal).normalize();
    Vec3 bitangent = normal.cross(tangent);
    
    return tangent * x + bitangent * y + normal * z;
}
Vec3::Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
Vec3::Vec3(float value) : x(value), y(value), z(value) {}
Vec3::Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
// double array constructor'u kaldırıldı

// Access operators (Daha temiz switch yapısı)
float Vec3::operator[](int index) const {
    switch (index) {
    case 0: return x;
    case 1: return y;
    case 2: return z;
    default: throw std::out_of_range("Vec3 index out of range");
    }
}

float& Vec3::operator[](int index) {
    switch (index) {
    case 0: return x;
    case 1: return y;
    case 2: return z;
    default: throw std::out_of_range("Vec3 index out of range");
    }
}

// Arithmetic operators
Vec3 Vec3::operator-() const { return Vec3(-x, -y, -z); }
Vec3 Vec3::operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
Vec3 Vec3::operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
Vec3 Vec3::operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
Vec3 Vec3::operator*(float t) const { return Vec3(x * t, y * t, z * t); }
Vec3 Vec3::operator/(float scalar) const { return Vec3(x / scalar, y / scalar, z / scalar); }
Vec3 Vec3::operator/(const Vec3& v) const { return Vec3(x / v.x, y / v.y, z / v.z); }
Vec3 Vec3::operator+(float scalar) const { return Vec3(x + scalar, y + scalar, z + scalar); }
Vec3 Vec3::operator-(float scalar) const { return Vec3(x - scalar, y - scalar, z - scalar); }

// Compound assignment operators
Vec3& Vec3::operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
Vec3& Vec3::operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
Vec3& Vec3::operator*=(float t) { x *= t; y *= t; z *= t; return *this; }
Vec3& Vec3::operator*=(const Vec3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }

Vec3& Vec3::operator/=(float t) {
    constexpr float EPSILON = 1e-6f; // float hassasiyeti için güvenli eşik
    if (std::abs(t) < EPSILON) {
        x = y = z = 0.0f;
        return *this;
    }
    float inv_t = 1.0f / t;
    x *= inv_t; y *= inv_t; z *= inv_t;
    return *this;
}

// Comparison operators
bool Vec3::operator==(const Vec3& other) const {
    const float epsilon = 1e-5f;
    return (std::abs(x - other.x) < epsilon &&
        std::abs(y - other.y) < epsilon &&
        std::abs(z - other.z) < epsilon);
}

bool Vec3::operator!=(const Vec3& other) const {
    return !(*this == other);
}

// Vector operations
float Vec3::length() const { return std::sqrt(length_squared()); }
float Vec3::length_squared() const { return x * x + y * y + z * z; }

Vec3 Vec3::normalize() const {
    float len_sq = length_squared();
    if (len_sq > 1e-6f) { // Sıfıra bölme ve near_zero kontrolü
        float inv_len = 1.0f / std::sqrt(len_sq);
        return Vec3(x * inv_len, y * inv_len, z * inv_len);
    }
    return Vec3(0, 0, 0);
}

Vec3 Vec3::cross(const Vec3& v) const {
    return Vec3(y * v.z - z * v.y,
        z * v.x - x * v.z,
        x * v.y - y * v.x);
}

Vec3 Vec3::abs() const {
    return Vec3(std::fabs(x), std::fabs(y), std::fabs(z));
}

Vec3 Vec3::orient(const Vec3& local) const {
    // Bu, TBN (Tangent, Bitangent, Normal) matrisi oluşturur.
    Vec3 tangent = (std::abs(z) > 0.999f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(0.0f, 0.0f, 1.0f);
    Vec3 bitangent = cross(tangent).normalize();
    tangent = bitangent.cross(*this);
    return (tangent * local.x + bitangent * local.y + (*this) * local.z).normalize();
}

Vec3 Vec3::cwiseProduct(const Vec3& v) const { return *this * v; }

// Component-wise operations
float Vec3::max_component() const { return std::max(x, std::max(y, z)); }

bool Vec3::near_zero() const {
    const auto s = 1e-6f; // Sabit float epsilon kullan
    return (std::fabs(x) < s) && (std::fabs(y) < s) && (std::fabs(z) < s);
}

float Vec3::luminance() const {
    return 0.2126f * x + 0.7152f * y + 0.0722f * z;
}

std::string Vec3::toString() const {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
}

// Static utility functions
float Vec3::dot(const Vec3& v1, const Vec3& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
Vec3 Vec3::cross(const Vec3& v1, const Vec3& v2) { return v1.cross(v2); }

Vec3 Vec3::reflect(const Vec3& v, const Vec3& n) {
    return v - 2 * Vec3::dot(v, n) * n;
}

Vec3 Vec3::refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    auto uv_normalized = uv.normalize();
    float cos_theta = std::fmin(Vec3::dot(-uv_normalized, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv_normalized + cos_theta * n);
    float k = 1.0f - r_out_perp.length_squared();
    if (k < 0.0f) return Vec3::reflect(uv_normalized, n);
    Vec3 r_out_parallel = -std::sqrt(k) * n;
    return (r_out_perp + r_out_parallel).normalize();
}

Vec3 Vec3::min(const Vec3& a, const Vec3& b) {
    return Vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

Vec3 Vec3::max(const Vec3& a, const Vec3& b) {
    return Vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

Vec3 Vec3::clamp(const Vec3& vec, float min, float max) {
    if (min > max) std::swap(min, max);
    return Vec3(std::clamp(vec.x, min, max),
        std::clamp(vec.y, min, max),
        std::clamp(vec.z, min, max));
}

Vec3 Vec3::lerp(const Vec3& a, const Vec3& b, float t) {
    return Vec3(a.x + t * (b.x - a.x),
        a.y + t * (b.y - a.y),
        a.z + t * (b.z - a.z));
}

Vec3 Vec3::lerp(const Vec3& a, const Vec3& b, const Vec3& t) {
    return a * (Vec3(1.0f) - t) + b * t;
}

float Vec3::lerpf(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

Vec3 Vec3::mix(const Vec3& a, const Vec3& b, float t) {
    return Vec3(a.x * (1.0f - t) + b.x * t,
        a.y * (1.0f - t) + b.y * t,
        a.z * (1.0f - t) + b.z * t);
}

Vec3 Vec3::mix(const Vec3& a, const Vec3& b, const Vec3& t) {
    return Vec3(a.x * (1.0f - t.x) + b.x * t.x,
        a.y * (1.0f - t.y) + b.y * t.y,
        a.z * (1.0f - t.z) + b.z * t.z);
}

Vec3 Vec3::exp(const Vec3& v) {
    return Vec3(std::exp(v.x), std::exp(v.y), std::exp(v.z));
}

float Vec3::average(const Vec3& v) {
    return (v.x + v.y + v.z) / 3.0f;
}

// Duplicate random functions removed


Vec3 Vec3::sphericalDirection(float sinTheta, float cosTheta, float phi) {
    return Vec3(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
}

Vec3 Vec3::from_spherical(float theta, float phi, float r) {
    return Vec3(r * std::sin(theta) * std::cos(phi),
        r * std::sin(theta) * std::sin(phi),
        r * std::cos(theta));
}

// Conversion operators (Kaldırıldı)

// Friend functions
std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << v.x << ", " << v.y << ", " << v.z;
    return os;
}

Vec3 operator*(float t, const Vec3& v) {
    return Vec3(v.x * t, v.y * t, v.z * t);
}

Vec3 operator/(float scalar, const Vec3& v) {
    return Vec3(scalar / v.x, scalar / v.y, scalar / v.z);
}

// Non-member functions
Vec3 unit_vector(const Vec3& v) {
    float len = v.length();
    return (len > 1e-6f) ? v / len : Vec3(0, 0, 0);
}
