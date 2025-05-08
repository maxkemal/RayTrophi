#include "Vec3SIMD.h"
#include <stdexcept>
#include <cmath>

std::mt19937 Vec3SIMD::rng(std::random_device{}());

Vec3SIMD::Vec3SIMD() : data(_mm256_setzero_ps()) {}
Vec3SIMD::Vec3SIMD(__m256 d) : data(d) {}
Vec3SIMD::Vec3SIMD(const Vec3& v) : data(_mm256_set_ps(0, 0, 0, 0, 0, v.z, v.y, v.x)) {}
Vec3SIMD::Vec3SIMD(float x, float y, float z) : data(_mm256_set_ps(0, 0, 0, 0, 0, z, y, x)) {}

float Vec3SIMD::get(int index) const {
    alignas(32) float result[8];
    _mm256_store_ps(result, data);
    return result[index];
}

std::ostream& operator<<(std::ostream& os, const Vec3SIMD& vec) {
    os << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")";
    return os;
}

Vec3SIMD::operator Vec3() const {
    alignas(32) float result[8];
    _mm256_store_ps(result, data);
    return Vec3(result[0], result[1], result[2]);
}

Vec3SIMD Vec3SIMD::mix(const Vec3SIMD& a, const Vec3SIMD& b, float t)  {
    __m256 t_vec = _mm256_set1_ps(t);
    __m256 one_minus_t = _mm256_set1_ps(1.0f - t);
    return Vec3SIMD(_mm256_add_ps(_mm256_mul_ps(a.data, one_minus_t), _mm256_mul_ps(b.data, t_vec)));
}

float Vec3SIMD::x() const { return _mm256_cvtss_f32(data); }
float Vec3SIMD::y() const { return _mm256_cvtss_f32(_mm256_permute_ps(data, _MM_SHUFFLE(1, 1, 1, 1))); }
float Vec3SIMD::z() const { return _mm256_cvtss_f32(_mm256_permute_ps(data, _MM_SHUFFLE(2, 2, 2, 2))); }

Vec3SIMD Vec3SIMD::operator-() const {
    return Vec3SIMD(_mm256_xor_ps(data, _mm256_set1_ps(-0.0)));
}

Vec3SIMD& Vec3SIMD::operator+=(const Vec3SIMD& v) {
    data = _mm256_add_ps(data, v.data);
    return *this;
}

Vec3SIMD& Vec3SIMD::operator-=(const Vec3SIMD& v) {
    data = _mm256_sub_ps(data, v.data);
    return *this;
}

Vec3SIMD& Vec3SIMD::operator*=(const Vec3SIMD& v) {
    data = _mm256_mul_ps(data, v.data);
    return *this;
}

Vec3SIMD& Vec3SIMD::operator/=(const Vec3SIMD& v) {
    data = _mm256_div_ps(data, v.data);
    return *this;
}

Vec3SIMD& Vec3SIMD::operator*=(float t) {
    data = _mm256_mul_ps(data, _mm256_set1_ps(t));
    return *this;
}

Vec3SIMD& Vec3SIMD::operator/=(float t) {
    if (std::abs(t) < std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error("Division by zero in Vec3SIMD");
    }
    data = _mm256_div_ps(data, _mm256_set1_ps(t));
    return *this;
}

float Vec3SIMD::length() const {
    __m256 squared = _mm256_mul_ps(data, data);
    __m256 sum = _mm256_hadd_ps(squared, squared);
    sum = _mm256_hadd_ps(sum, sum);
    return _mm256_cvtss_f32(_mm256_sqrt_ps(sum));
}

__m256 Vec3SIMD::length_vec() const {
    __m256 squared = _mm256_mul_ps(data, data);
    __m256 sum = _mm256_hadd_ps(squared, squared);
    sum = _mm256_hadd_ps(sum, sum);
    return _mm256_sqrt_ps(sum);
}

float Vec3SIMD::length_squared() const {
    __m256 squared = _mm256_mul_ps(data, data);
    __m256 sum = _mm256_hadd_ps(squared, squared);
    sum = _mm256_hadd_ps(sum, sum);
    return _mm256_cvtss_f32(sum);
}

Vec3SIMD Vec3SIMD::cross(const Vec3SIMD& v) const {
    __m256 a = _mm256_permute_ps(data, _MM_SHUFFLE(3, 0, 2, 1));
    __m256 b = _mm256_permute_ps(v.data, _MM_SHUFFLE(3, 0, 2, 1));
    __m256 c = _mm256_mul_ps(a, v.data);
    __m256 d = _mm256_mul_ps(b, data);
    return Vec3SIMD(_mm256_sub_ps(c, d));
}

Vec3SIMD Vec3SIMD::safe_normalize(const Vec3SIMD& fallback) const {
    __m256 len = length_vec();
    __m256 is_zero = _mm256_cmp_ps(len, _mm256_setzero_ps(), _CMP_EQ_OQ);
    __m256 len_reciprocal = _mm256_div_ps(_mm256_set1_ps(1.0f), len);
    __m256 normalized = _mm256_mul_ps(data, len_reciprocal);
    return Vec3SIMD(_mm256_blendv_ps(normalized, fallback.data, is_zero));
}

Vec3SIMD Vec3SIMD::normalize() const {
    const __m256 EPSILON = _mm256_set1_ps(1e-10f);
    const __m256 ONE = _mm256_set1_ps(1.0f);
    __m256 len = length_vec();
    __m256 is_near_zero = _mm256_cmp_ps(len, EPSILON, _CMP_LT_OQ);   
    __m256 safe_len = _mm256_add_ps(len, EPSILON);
    __m256 len_reciprocal = _mm256_div_ps(ONE, safe_len);
    __m256 normalized = _mm256_mul_ps(data, len_reciprocal);   
    __m256 result = _mm256_blendv_ps(normalized, data, is_near_zero);  
    result = _mm256_min_ps(_mm256_max_ps(result, _mm256_set1_ps(-1.0f)), _mm256_set1_ps(1.0f));
    return Vec3SIMD(result);
}

Vec3SIMD Vec3SIMD::yxz() const {
    return Vec3SIMD(_mm256_permute_ps(data, _MM_SHUFFLE(3, 0, 1, 2)));
}

float Vec3SIMD::max_component() const {
    __m256 max1 = _mm256_max_ps(data, _mm256_permute_ps(data, _MM_SHUFFLE(3, 0, 0, 1)));
    __m256 max2 = _mm256_max_ps(max1, _mm256_permute_ps(max1, _MM_SHUFFLE(3, 0, 0, 2)));
    return _mm256_cvtss_f32(max2);
}

Vec3SIMD Vec3SIMD::to_vec3simd(const Vec3& vec) {
    return Vec3SIMD(vec);
}

Vec3SIMD operator+(const Vec3SIMD& u, const Vec3SIMD& v) {
    return Vec3SIMD(_mm256_add_ps(u.data, v.data));
}

Vec3SIMD operator-(const Vec3SIMD& u, const Vec3SIMD& v) {
    return Vec3SIMD(_mm256_sub_ps(u.data, v.data));
}

Vec3SIMD operator*(const Vec3SIMD& u, const Vec3SIMD& v) {
    return Vec3SIMD(_mm256_mul_ps(u.data, v.data));
}

Vec3SIMD operator/(const Vec3SIMD& u, const Vec3SIMD& v) {
    return Vec3SIMD(_mm256_div_ps(u.data, v.data));
}

Vec3SIMD operator*(float t, const Vec3SIMD& v) {
    return Vec3SIMD(_mm256_mul_ps(_mm256_set1_ps(t), v.data));
}

Vec3SIMD operator*(const Vec3SIMD& v, float t) {
    return t * v;
}

Vec3SIMD operator/(const Vec3SIMD& v, float t) {
    return Vec3SIMD(_mm256_div_ps(v.data, _mm256_set1_ps(t)));
}

Vec3SIMD Vec3SIMD::lerp(const Vec3SIMD& a, const Vec3SIMD& b, float t) {
    __m256 t_vec = _mm256_set1_ps(t);
    __m256 one_minus_t = _mm256_set1_ps(1.0f - t);
    return Vec3SIMD(_mm256_add_ps(_mm256_mul_ps(a.data, one_minus_t), _mm256_mul_ps(b.data, t_vec)));
}
float Vec3SIMD::to_float() const {
    return _mm_cvtss_f32(_mm256_castps256_ps128(data));
}

Vec3SIMD Vec3SIMD::reflect(const Vec3SIMD& v, const Vec3SIMD& n) {
    Vec3SIMD dot_vn = dot(v, n);
    Vec3SIMD scaled_n = n * (dot_vn * 2.0f);
    return v - scaled_n;
}


Vec3SIMD Vec3SIMD::refract(const Vec3SIMD& uv, const Vec3SIMD& n, float etai_over_etat) {
    Vec3SIMD neg_uv = uv * -1.0f;
    Vec3SIMD dot_neg_uvn = dot(neg_uv, n);
    Vec3SIMD cos_theta_vec = _mm256_min_ps(dot_neg_uvn.data, set1(1.0f).data);
    Vec3SIMD r_out_perp = (uv + (cos_theta_vec * n)) * etai_over_etat;
    float r_out_perp_length_squared = r_out_perp.length_squared();
    Vec3SIMD one_minus_r_out_perp_length_squared = set1(1.0f) - set1(r_out_perp_length_squared);
    Vec3SIMD sqrt_term = one_minus_r_out_perp_length_squared.sqrt();
    Vec3SIMD r_out_parallel = -sqrt_term * n;
    return r_out_perp + r_out_parallel;
}
float Vec3SIMD::dotfloat(const Vec3SIMD& u, const Vec3SIMD& v) {
    __m256 prod = _mm256_mul_ps(u.data, v.data);
    __m128 low = _mm256_castps256_ps128(prod);  
    __m128 high = _mm256_extractf128_ps(prod, 1);
    __m128 sum = _mm_add_ps(low, high); 
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
    return _mm_cvtss_f32(sum);
}


 Vec3SIMD Vec3SIMD::dot(const Vec3SIMD& u, const Vec3SIMD& v) {
     // Eleman eleman çarpým
     __m256 mul = _mm256_mul_ps(u.data, v.data);
     __m256 shuf1 = _mm256_permute_ps(mul, _MM_SHUFFLE(3, 0, 2, 1));
     __m256 shuf2 = _mm256_permute_ps(mul, _MM_SHUFFLE(3, 1, 0, 2)); 
     __m256 result = _mm256_add_ps(mul, shuf1);
     result = _mm256_add_ps(result, shuf2);
     result = _mm256_permute_ps(result, _MM_SHUFFLE(2, 2, 2, 2)); 
     return Vec3SIMD(result); 
 }
 float Vec3SIMD::dot(const Vec3SIMD& other) const {
     __m256 prod = _mm256_mul_ps(data, other.data);
     __m256 sum = _mm256_hadd_ps(prod, prod);
     sum = _mm256_hadd_ps(sum, sum);
     return _mm256_cvtss_f32(sum);
 }

bool Vec3SIMD::near_zero() const {
    const float epsilon = 1e-5f;
    __m256 eps = _mm256_set1_ps(epsilon);
    __m256 abs_data = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), data);
    __m256 cmp = _mm256_cmp_ps(abs_data, eps, _CMP_LT_OQ);
    return _mm256_movemask_ps(cmp) == 0xFF;
}

Vec3SIMD Vec3SIMD::cross(const Vec3SIMD& u, const Vec3SIMD& v) {
    __m256 a = _mm256_permute_ps(u.data, _MM_SHUFFLE(3, 0, 2, 1));
    __m256 b = _mm256_permute_ps(v.data, _MM_SHUFFLE(3, 1, 0, 2));
    __m256 c = _mm256_permute_ps(u.data, _MM_SHUFFLE(3, 1, 0, 2));
    __m256 d = _mm256_permute_ps(v.data, _MM_SHUFFLE(3, 0, 2, 1));
    return Vec3SIMD(_mm256_sub_ps(_mm256_mul_ps(a, b), _mm256_mul_ps(c, d)));
}

Vec3SIMD Vec3SIMD::random_in_unit_sphere() {   
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    while (true) {
        Vec3SIMD p(dist(rng), dist(rng), dist(rng));
        if (p.length_squared() < 1.0f)
            return p;
    }
}

Vec3SIMD Vec3SIMD::random(double min, double max) {  
    std::uniform_real_distribution<float> dist(static_cast<float>(min), static_cast<float>(max));
    return Vec3SIMD(dist(rng), dist(rng), dist(rng));
}

Vec3SIMD Vec3SIMD::random_in_unit_disk() {   
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    while (true) {
        Vec3SIMD p(dist(rng), dist(rng), 0.0f);
        if (p.length_squared() < 1.0f)
            return p;
    }
}

float Vec3SIMD::random_double() {   
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

Vec3SIMD Vec3SIMD::random_unit_vector() {  
    std::uniform_real_distribution<float> dist(0.0f, 2.0f * M_PI);
    float a = dist(rng);
    float z = dist(rng) * 2.0f - 1.0f;
    float r = std::sqrt(1.0f - z * z);
    return Vec3SIMD(r * std::cos(a), r * std::sin(a), z);
}

Vec3SIMD Vec3SIMD::random_cosine_direction(const Vec3SIMD& normal)  {
    Vec3SIMD random_dir = random_unit_vector();
    Vec3SIMD u, v;
    if (std::abs(normal.x()) > 0.1f) {
        u = Vec3SIMD(0.0f, 1.0f, 0.0f).cross(normal).normalize();
    }
    else {
        u = Vec3SIMD(1.0f, 0.0f, 0.0f).cross(normal).normalize();
    }
    v = normal.cross(u);
    Vec3SIMD weighted_dir = u * random_dir.x() + v * random_dir.y() + normal * random_dir.z();
    return weighted_dir.normalize();
}
Vec3SIMD Vec3SIMD::random_in_unit_hemisphere(const Vec3SIMD& normal) {    
    Vec3SIMD in_unit_sphere = Vec3SIMD::random_in_unit_sphere();
    __m256 dot_product = Vec3SIMD::dot(in_unit_sphere, normal).data;
    __m256 zero_vec = _mm256_set1_ps(0.0f);
    __m256 condition = _mm256_cmp_ps(dot_product, zero_vec, _CMP_GT_OS); // dot_product > 0 mý?
    __m256 negated_in_unit_sphere = _mm256_sub_ps(zero_vec, in_unit_sphere.data);
    __m256 selected_sphere = _mm256_blendv_ps(negated_in_unit_sphere, in_unit_sphere.data, condition);
    Vec3SIMD result;
    result.data = selected_sphere;
    return result;
}
Vec3SIMD Vec3SIMD::random_in_hemisphere(const Vec3SIMD& normal) {
    Vec3SIMD in_unit_sphere = Vec3SIMD::random_in_unit_sphere();
    __m256 dot_product = Vec3SIMD::dot(in_unit_sphere, normal).data;
    __m256 zero_vec = _mm256_set1_ps(0.0f);
    __m256 condition = _mm256_cmp_ps(dot_product, zero_vec, _CMP_GT_OS);
    __m256 negated_in_unit_sphere = _mm256_sub_ps(zero_vec, in_unit_sphere.data); // -inUnitSphere
    __m256 selected_sphere = _mm256_blendv_ps(negated_in_unit_sphere, in_unit_sphere.data, condition);
    Vec3SIMD result;
    result.data = selected_sphere;
    return result;
}
