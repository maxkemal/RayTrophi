#include "Vec2.h"
#include <cmath>

// Constructors
Vec2::Vec2(float u, float v) : u(u), v(v) {}
Vec2::Vec2(const Vec2& other) : u(other.u), v(other.v) {}

// Common vector operations
float Vec2::length() const { return std::sqrt(u * u + v * v); }
float Vec2::lengthSquared() const { return u * u + v * v; }

void Vec2::normalize() {
    float len = length();
    if (len > 0) {
        u /= len;
        v /= len;
    }
}

Vec2 Vec2::normalized() const {
    Vec2 result = *this;
    result.normalize();
    return result;
}

Vec2 Vec2::rotate(float angle) const {
    float cs = std::cos(angle);
    float sn = std::sin(angle);
    return Vec2(u * cs - v * sn, u * sn + v * cs);
}

// Dot and cross products
float Vec2::dot(const Vec2& other) const { return u * other.u + v * other.v; }
float Vec2::cross(const Vec2& other) const { return u * other.v - v * other.u; }

// Angle between vectors
float Vec2::angle(const Vec2& other) const {
    return std::atan2(cross(other), dot(other));
}

// Operator overloads
Vec2 Vec2::operator+(const Vec2& other) const { return Vec2(u + other.u, v + other.v); }
Vec2 Vec2::operator-(const Vec2& other) const { return Vec2(u - other.u, v - other.v); }
Vec2 Vec2::operator*(float scalar) const { return Vec2(u * scalar, v * scalar); }
Vec2 Vec2::operator/(float scalar) const { return Vec2(u / scalar, v / scalar); }

Vec2& Vec2::operator+=(const Vec2& other) {
    u += other.u;
    v += other.v;
    return *this;
}

Vec2& Vec2::operator-=(const Vec2& other) {
    u -= other.u;
    v -= other.v;
    return *this;
}

Vec2& Vec2::operator*=(float scalar) {
    u *= scalar;
    v *= scalar;
    return *this;
}

Vec2& Vec2::operator/=(float scalar) {
    u /= scalar;
    v /= scalar;
    return *this;
}

bool Vec2::operator==(const Vec2& other) const {
    return (u == other.u) && (v == other.v);
}

bool Vec2::operator!=(const Vec2& other) const {
    return !(*this == other);
}

Vec2 Vec2::operator-() const { return Vec2(-u, -v); }

float& Vec2::operator[](int i) {
    return (i == 0) ? u : v;
}

const float& Vec2::operator[](int i) const {
    return (i == 0) ? u : v;
}

// Static methods for common vectors
Vec2 Vec2::zero() { return Vec2(0, 0); }
Vec2 Vec2::one() { return Vec2(1, 1); }
Vec2 Vec2::up() { return Vec2(0, 1); }
Vec2 Vec2::down() { return Vec2(0, -1); }
Vec2 Vec2::left() { return Vec2(-1, 0); }
Vec2 Vec2::right() { return Vec2(1, 0); }

// Utility functions
void Vec2::clamp(float minVal, float maxVal) {
    u = std::max(minVal, std::min(u, maxVal));
    v = std::max(minVal, std::min(v, maxVal));
}

Vec2 Vec2::lerp(const Vec2& other, float t) const {
    return *this + (other - *this) * t;
}

// Stream insertion operator for easy printing
std::ostream& operator<<(std::ostream& os, const Vec2& vec) {
    os << "Vec2(" << vec.u << ", " << vec.v << ")";
    return os;
}

// Non-member operator overloads
Vec2 operator*(float scalar, const Vec2& vec) {
    return vec * scalar;
}