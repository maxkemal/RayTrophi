#include "Vec3.h"
#include <cstdlib>
#include <limits>
#include <random>

// Static random number generator initialization
static std::mt19937 rng;
static std::uniform_real_distribution<double> dist(0.0, 1.0);

// Constructors
Vec3::Vec3() : x(0), y(0), z(0) {}
Vec3::Vec3(float value) : x(value), y(value), z(value) {}
Vec3::Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
Vec3::Vec3(const std::array<double, 3>& arr) : x(arr[0]), y(arr[1]), z(arr[2]) {}

// Access operators
float Vec3::operator[](int index) const {
    if (index == 0) return x;
    else if (index == 1) return y;
    return z;
}

float& Vec3::operator[](int index) {
    if (index == 0) return x;
    else if (index == 1) return y;
    return z;
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
    constexpr float EPSILON = std::numeric_limits<float>::epsilon();
    if (fabs(t) < EPSILON) {
        *this = Vec3(0, 0, 0);
        return *this;
    }
    float inv_t = 1.0f / t;
    x *= inv_t; y *= inv_t; z *= inv_t;
    return *this;
}

// Comparison operators
bool Vec3::operator==(const Vec3& other) const {
    const float epsilon = std::numeric_limits<float>::epsilon() * 100.0f;
    return (std::abs(x - other.x) < epsilon &&
        std::abs(y - other.y) < epsilon &&
        std::abs(z - other.z) < epsilon);
}

bool Vec3::operator!=(const Vec3& other) const {
    return !(*this == other);
}

// Vector operations
float Vec3::length() const { return sqrt(length_squared()); }
float Vec3::length_squared() const { return x * x + y * y + z * z; }

Vec3 Vec3::normalize() const {
    float len_sq = length_squared();
    if (len_sq > std::numeric_limits<float>::epsilon()) {
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
    Vec3 tangent = (std::abs(z) > 0.999f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(0.0f, 0.0f, 1.0f);
    Vec3 bitangent = cross(tangent).normalize();
    tangent = bitangent.cross(*this);
    return (tangent * local.x + bitangent * local.y + (*this) * local.z).normalize();
}

Vec3 Vec3::cwiseProduct(const Vec3& v) const { return *this * v; }

// Component-wise operations
float Vec3::max_component() const { return std::max(x, std::max(y, z)); }

bool Vec3::near_zero() const {
    const auto s = std::numeric_limits<float>::epsilon();
    return (fabs(x) < s) && (fabs(y) < s) && (fabs(z) < s);
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
    double cos_theta = std::fmin(Vec3::dot(-uv_normalized, n), 1.0);
    Vec3 r_out_perp = etai_over_etat * (uv_normalized + cos_theta * n);
    float k = 1.0f - r_out_perp.length_squared();
    if (k < 0) return Vec3::reflect(uv_normalized, n);
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
    return a * (Vec3(1.0) - t) + b * t;
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

// Random generation functions
Vec3 Vec3::random() { return Vec3(random_double(), random_double(), random_double()); }
Vec3 Vec3::random(double min, double max) {
    return Vec3(random_double(min, max), random_double(min, max), random_double(min, max));
}

Vec3 Vec3::random_in_unit_sphere() {
    double u = random_double(-1, 1);
    double theta = random_double(0, 2 * M_PI);
    double r = pow(random_double(), 1.0 / 3.0);
    double sq = sqrt(1 - u * u);
    return Vec3(r * sq * cos(theta), r * sq * sin(theta), r * u);
}

Vec3 Vec3::random_in_hemisphere(const Vec3& normal) {
    Vec3 inUnitSphere = random_in_unit_sphere();
    return (Vec3::dot(inUnitSphere, normal) > 0.0f) ? inUnitSphere : -inUnitSphere;
}

Vec3 Vec3::random_unit_vector() {
    auto a = random_double(0, 2 * M_PI);
    auto z = random_double(-1, 1);
    auto r = sqrt(1 - z * z);
    return Vec3(r * cos(a), r * sin(a), z);
}

Vec3 Vec3::random_in_unit_disk() {
    double r = sqrt(random_double());
    double theta = 2 * M_PI * random_double();
    return Vec3(r * cos(theta), r * sin(theta), 0);
}

Vec3 Vec3::random_cosine_direction(const Vec3& normal) {
    double r1 = random_double();
    double r2 = random_double();
    double phi = 2 * M_PI * r1;
    double cos_theta = sqrt(1.0 - r2);
    double sin_theta = sqrt(r2);
    double x = cos(phi) * sin_theta;
    double y = sin(phi) * sin_theta;
    double z = cos_theta;

    Vec3 N = normal.normalize();
    Vec3 tangent = (fabs(N.x) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    tangent = (tangent - N * dot(N,tangent)).normalize();
    Vec3 bitangent = N.cross(tangent);
    return (tangent * x + bitangent * y + N * z).normalize();
}

Vec3 Vec3::sample_hemisphere_cosine_weighted(const Vec3& normal, float u, float v) {
    float r = std::sqrt(u);
    float theta = 2.0f * M_PI * v;
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(std::max(0.0f, 1.0f - u));

    Vec3 up = std::abs(normal.y) > 0.99f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    Vec3 tangent = up.cross(normal).normalize();
    Vec3 bitangent = normal.cross(tangent);
    return (tangent * x) + (bitangent * y) + (normal * z);
}

Vec3 Vec3::sphericalDirection(float sinTheta, float cosTheta, float phi) {
    return Vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

Vec3 Vec3::from_spherical(double theta, double phi, double r) {
    return Vec3(r * sin(theta) * cos(phi),
        r * sin(theta) * sin(phi),
        r * cos(theta));
}

// Random number utilities
double Vec3::random_double(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    return min + (max - min) * dis(gen);
}

// Conversion operators
Vec3::operator std::array<double, 3>() const {
    return { x, y, z };
}

// Friend functions
std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << v.x << ", " << v.y << ", " << v.z;
    return os;
}

Vec3 operator*(double t, const Vec3& v) {
    return Vec3(v.x * t, v.y * t, v.z * t);
}

Vec3 operator/(float scalar, const Vec3& v) {
    return Vec3(scalar / v.x, scalar / v.y, scalar / v.z);
}

// Non-member functions
Vec3 unit_vector(const Vec3& v) {
    double len = v.length();
    return (len > 0.0) ? v / len : Vec3(0, 0, 0);
}