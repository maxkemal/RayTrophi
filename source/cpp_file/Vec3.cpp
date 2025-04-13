#include "Vec3.h"
#include "Vec3SIMD.h"
#include <cstdlib> 
#include <limits>  
#include <random>
Vec3::Vec3() : x(0), y(0), z(0) {}

Vec3::Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

float Vec3::operator[](int index) const {
    if (index == 0) return x;
    else if (index == 1) return y;
    else return z;
}
Vec3::Vec3(float value) : x(value), y(value), z(value) {}
float& Vec3::operator[](int index) {
    if (index == 0) return x;
    else if (index == 1) return y;
    else return z;
}

float Vec3::max_component() const {
    // Assuming Vec3 has members x, y, z
    return std::max(x, std::max(y, z));  // Compare x, y, and z to find the maximum component
}


Vec3 Vec3::operator*(const Vec3& other) const {
    // Implementation of the operator
    return Vec3(x * other.x, y * other.y, z * other.z);
}
Vec3& Vec3::operator*=(float t) {
    x *= t;
    y *= t;
    z *= t;
    return *this;
}

Vec3& Vec3::operator*=(const Vec3& other) {
    x *= other.x;
    y *= other.y;
    z *= other.z;
    return *this;
}
Vec3 Vec3::operator-() const {
    return Vec3(-x, -y, -z);
}

Vec3 Vec3::operator+(const Vec3& other) const {
    return Vec3(x + other.x, y + other.y, z + other.z);
}

Vec3 Vec3::operator-(const Vec3& other) const {
    return Vec3(x - other.x, y - other.y, z - other.z);
}

Vec3 Vec3::operator*(float t) const {
    return Vec3(x * t, y * t, z * t);
}

Vec3& Vec3::operator+=(const Vec3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}
Vec3& Vec3::operator/=(float t) {
    constexpr float EPSILON = std::numeric_limits<float>::epsilon();
    if (fabs(t) < EPSILON) {
        *this = Vec3(0, 0, 0); // Bölme hatasýný önleyerek sýfýr vektörü döndür.
        return *this;
    }

    float inv_t = 1.0f / t;
    x *= inv_t;
    y *= inv_t;
    z *= inv_t;
    return *this;
}


float Vec3::length() const {
    return sqrt(length_squared());
}

float Vec3::length_squared() const {
    return x * x + y * y + z * z;
}

Vec3 Vec3::normalize() const {
    float len_sq = length_squared();
    if (len_sq > std::numeric_limits<float>::epsilon()) {
        float inv_len = 1.0f / std::sqrt(len_sq);
        return Vec3(x * inv_len, y * inv_len, z * inv_len);
    }
    return Vec3(0, 0, 0);
}

// Mersenne Twister rastgele sayý üreteci
std::mt19937 rng;
std::uniform_real_distribution<double> dist(0.0, 1.0);
Vec3 Vec3::from_spherical(double theta, double phi, double r = 1.0) {
    return Vec3(
        r * sin(theta) * cos(phi),
        r * sin(theta) * sin(phi),
        r * cos(theta)
    );
}
Vec3 Vec3::random_in_unit_sphere() {
    double u = random_double(-1, 1);
    double theta = random_double(0, 2 * M_PI);
    double r = pow(random_double(), 1.0 / 3.0);  // Cubic root for uniform distribution
    double sq = sqrt(1 - u * u);
    return Vec3(r * sq * cos(theta),
        r * sq * sin(theta),
        r * u);
}
Vec3 Vec3::random_in_hemisphere(const Vec3& normal) {
    Vec3 inUnitSphere = random_in_unit_sphere();
    if (Vec3::dot(inUnitSphere, normal) > 0.0f) // In the same hemisphere as the normal
        return inUnitSphere;
    else
        return -inUnitSphere;
}
// Rastgele sayý üreteciyi tohumla
void seed_random() {
    std::random_device rd;
    rng.seed(rd());
}
// [0, 1) aralýðýnda rastgele bir sayý üret
double random_double() {
    return dist(rng);
}
Vec3 Vec3::random(double min, double max) {
    return Vec3(random_double(min, max),
        random_double(min, max),
        random_double(min, max));
}
Vec3 Vec3::random_in_unit_disk() {
    // Polar koordinatlar kullanarak daha verimli bir versiyon
    double r = sqrt(random_double());
    double theta = 2 * M_PI * random_double();
    return Vec3(r * cos(theta), r * sin(theta), 0);
}

Vec3 unit_vector(const Vec3& v) {
    double len = v.length();
    if (len > 0.0) {
        return v / len;
    }
    return Vec3(0, 0, 0);  // veya exception fýrlatabilirsiniz
}
double random_double(double min, double max) {
    // Return a random real in [min,max)
    return min + (max - min) * random_double();
}

Vec3 Vec3::random_unit_vector() {
    auto a = random_double(0, 2 * M_PI);
    auto z = random_double(-1, 1);
    auto r = sqrt(1 - z * z);
    return Vec3(r * cos(a), r * sin(a), z);
}

Vec3 Vec3::reflect(const Vec3& v, const Vec3& n) {
    Vec3 result = v - 2 * Vec3::dot(v, n) * n;
    //result.z = -result.z;
    return result;
}

Vec3 Vec3::refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    auto uv_normalized = uv.normalize();
    double cos_theta = std::min(Vec3::dot(-uv_normalized, n), 1.0);
    Vec3 r_out_perp = etai_over_etat * (uv_normalized + cos_theta * n);
    float k = 1.0f - r_out_perp.length_squared();
    if (k < 0) {
        return Vec3::reflect(uv_normalized, n);
    }
    Vec3 r_out_parallel = -std::sqrt(k) * n;
    return (r_out_perp + r_out_parallel).normalize();  // Normalize eklenmiþ
}
Vec3 Vec3::cross(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x);
}


Vec3 Vec3::min(const Vec3& a, const Vec3& b) {
    return Vec3(
        std::min(a.x, b.x),
        std::min(a.y, b.y),
        std::min(a.z, b.z)
    );
}

Vec3 Vec3::max(const Vec3& a, const Vec3& b) {
    return Vec3(
        std::max(a.x, b.x),
        std::max(a.y, b.y),
        std::max(a.z, b.z)
    );
}

Vec3 operator*(double t, const Vec3& v) {
    return Vec3(v.x * t, v.y * t, v.z * t);
}

double Vec3::dot(const Vec3& v1, const Vec3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
