#include "Vec3SIMD.h"
#include <stdexcept>
#include <chrono>
#include <thread>

// --- Random Number Generator (XorShift32) ---
// Thread-local state for lock-free, fast generation
static thread_local uint32_t s_vec3simd_rng_state = 0;

static void init_simd_rng_if_needed() {
    if (s_vec3simd_rng_state == 0) {
        // Seed mixing: Time + Thread ID + Address
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        size_t h_thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
        uint32_t seed = static_cast<uint32_t>(now ^ h_thread);
        s_vec3simd_rng_state = (seed == 0) ? 0xDEADBEEF : seed;
    }
}

// --- Constructors ---
Vec3SIMD::Vec3SIMD() : data(_mm256_setzero_ps()) {}
Vec3SIMD::Vec3SIMD(__m256 d) : data(d) {}
Vec3SIMD::Vec3SIMD(float val) : data(_mm256_set1_ps(val)) {}
Vec3SIMD::Vec3SIMD(const float arr[8]) : data(_mm256_loadu_ps(arr)) {}

// Tek bir 3D Vektoru AVX'e Yukler (Sadece ilk 3 bileşeni kullanır)
Vec3SIMD::Vec3SIMD(float x, float y, float z) {
    // _mm256_set_ps(h7, h6, h5, h4, h3, h2, h1, h0) sırasıyla doldurur
    // Bizim için h0=x, h1=y, h2=z olacak. Diğerleri 0.
    data = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, z, y, x);
}

// --- Accessors (Yavas) ---
float Vec3SIMD::get(int index) const {
    alignas(32) float result[8];
    _mm256_store_ps(result, data);
    if (index >= 0 && index < 8) return result[index];
    throw std::out_of_range("Vec3SIMD index out of range");
}

// Tekil 3D bileşenlere erişim (Yavas: Sadece ilk 3 bileşeni dondurur)
float Vec3SIMD::x() const { return get(0); }
float Vec3SIMD::y() const { return get(1); }
float Vec3SIMD::z() const { return get(2); }

// --- Statik Fabrika Metotları ---
Vec3SIMD Vec3SIMD::set1(float val) {
    return Vec3SIMD{ _mm256_set1_ps(val) };
}
Vec3SIMD Vec3SIMD::setZero() {
    return Vec3SIMD{ _mm256_setzero_ps() };
}

// --- Aritmetik Operatorler (8x Paralel) ---
Vec3SIMD Vec3SIMD::operator-() const {
    return Vec3SIMD(_mm256_xor_ps(data, _mm256_set1_ps(-0.0f)));
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

// Skaler Operatorler
Vec3SIMD& Vec3SIMD::operator*=(float t) {
    data = _mm256_mul_ps(data, _mm256_set1_ps(t));
    return *this;
}
Vec3SIMD& Vec3SIMD::operator/=(float t) {
    float inv_t = 1.0f / t;
    data = _mm256_mul_ps(data, _mm256_set1_ps(inv_t));
    return *this;
}
Vec3SIMD Vec3SIMD::operator-(float scalar) const {
    return Vec3SIMD(_mm256_sub_ps(data, _mm256_set1_ps(scalar)));
}
Vec3SIMD Vec3SIMD::operator+(float scalar) const {
    return Vec3SIMD(_mm256_add_ps(data, _mm256_set1_ps(scalar)));
}
Vec3SIMD Vec3SIMD::operator*(float scalar) const {
    return Vec3SIMD(_mm256_mul_ps(data, _mm256_set1_ps(scalar)));
}
Vec3SIMD Vec3SIMD::operator/(float scalar) const {
    float inv_t = 1.0f / scalar;
    return Vec3SIMD(_mm256_mul_ps(data, _mm256_set1_ps(inv_t)));
}

// --- Karşılaştırma Operatörleri (8x Paralel) ---
__m256 Vec3SIMD::operator==(const Vec3SIMD& other) const {
    return _mm256_cmp_ps(data, other.data, _CMP_EQ_OQ);
}
__m256 Vec3SIMD::operator!=(const Vec3SIMD& other) const {
    return _mm256_cmp_ps(data, other.data, _CMP_NEQ_OQ);
}

// --- Temel Matematik (8x Paralel) ---
Vec3SIMD Vec3SIMD::abs() const {
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    return Vec3SIMD{ _mm256_andnot_ps(sign_mask, data) };
}

Vec3SIMD Vec3SIMD::sqrt() const {
    return Vec3SIMD{ _mm256_sqrt_ps(data) };
}
// --- AVX2 MATH INTRINSICS IMPLEMENTATION ---
// Based on Cephes Mathematical Library & Agner Fog's VCL
// Constants are defined locally to keep header clean.

static const __m256 _ps256_1  = _mm256_set1_ps(1.0f);
static const __m256 _ps256_0p5 = _mm256_set1_ps(0.5f);
static const __m256 _ps256_min_norm_pos = _mm256_set1_ps(1.17549435e-38f);
static const __m256 _ps256_mant_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));
static const __m256 _ps256_inv_mant_mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x7f800000));
static const __m256 _ps256_sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
static const __m256 _ps256_inv_sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x80000000));

// --- Sine & Cosine ---
void Vec3SIMD::sincos_256(const __m256& x, __m256* s, __m256* c) {
    __m256 xmm1, xmm2, xmm3, sign_bit_sin, y;
    __m256i emm0, emm2, emm4;

    sign_bit_sin = x;
    
    // Take the absolute value
    xmm1 = _mm256_and_ps(x, _ps256_inv_sign_mask);
    
    // Extract the sign bit (upper one)
    sign_bit_sin = _mm256_and_ps(sign_bit_sin, _ps256_sign_mask);
    
    // Scale by 4/Pi
    y = _mm256_mul_ps(xmm1, _mm256_set1_ps(1.27323954473516f)); // 4/PI

    // Determine the quadrant
    emm2 = _mm256_cvttps_epi32(y);
    
    // Integer parameter for quadrant check
    emm2 = _mm256_add_epi32(emm2, _mm256_set1_epi32(1));
    emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(~1));
    y = _mm256_cvtepi32_ps(emm2);

    emm4 = emm2;

    // Polynomial approximation constants
    const __m256 dp1 = _mm256_set1_ps(-0.78515625f);
    const __m256 dp2 = _mm256_set1_ps(-2.4187564849853515625e-4f);
    const __m256 dp3 = _mm256_set1_ps(-3.77489497744594108e-8f);

    // Subtract the multiple of PI/2
    xmm1 = _mm256_add_ps(xmm1, _mm256_mul_ps(y, dp1));
    xmm1 = _mm256_add_ps(xmm1, _mm256_mul_ps(y, dp2));
    xmm1 = _mm256_add_ps(xmm1, _mm256_mul_ps(y, dp3));

    // Compute Sin
    const __m256 sincof_p0 = _mm256_set1_ps(2.443315711809948E-005f);
    const __m256 sincof_p1 = _mm256_set1_ps(-1.388731625493765E-003f);
    const __m256 sincof_p2 = _mm256_set1_ps(4.166664568298827E-002f);
    
    xmm2 = _mm256_mul_ps(xmm1, xmm1); // x^2
    xmm3 = _mm256_mul_ps(xmm2, xmm1); // x^3

    y = _mm256_mul_ps(xmm2, sincof_p0);
    y = _mm256_add_ps(y, sincof_p1);
    y = _mm256_mul_ps(y, xmm2);
    y = _mm256_add_ps(y, sincof_p2);
    y = _mm256_mul_ps(y, xmm3);
    y = _mm256_add_ps(y, xmm1);
    
    // Swap sin/cos if quadrant 1, 2
    __m256i poly_mask = _mm256_set1_epi32(2);
    emm0 = _mm256_and_si256(emm2, poly_mask);
    __m256 swap_sign_bit_sin = _mm256_castsi256_ps(_mm256_cmpeq_epi32(emm0, _mm256_setzero_si256()));

    // Compute Cos (using same poly as sin but for the other part)
    const __m256 coscof_p0 = _mm256_set1_ps(-1.9515295891E-4f);
    const __m256 coscof_p1 = _mm256_set1_ps(8.3321608736E-3f);
    const __m256 coscof_p2 = _mm256_set1_ps(-1.6666654611E-1f);

    __m256 y2 = _mm256_mul_ps(xmm2, coscof_p0);
    y2 = _mm256_add_ps(y2, coscof_p1);
    y2 = _mm256_mul_ps(y2, xmm2);
    y2 = _mm256_add_ps(y2, coscof_p2);
    y2 = _mm256_mul_ps(y2, xmm2);
    y2 = _mm256_mul_ps(y2, _ps256_0p5);
    y2 = _mm256_add_ps(y2, _ps256_1);
    y2 = _mm256_sub_ps(y2, _mm256_mul_ps(xmm2, _ps256_0p5)); // 1 - x^2/2 + ...

    // Select based on quadrant
    __m256 sin_val = _mm256_blendv_ps(y2, y, swap_sign_bit_sin);
    __m256 cos_val = _mm256_blendv_ps(y, y2, swap_sign_bit_sin);

    // Apply signs
    if (s) {
        // Sin sign
        emm0 = _mm256_and_si256(emm2, _mm256_set1_epi32(4));
        __m256 sign_mask_sin_quad = _mm256_castsi256_ps(_mm256_cmpeq_epi32(emm0, _mm256_setzero_si256()));
        // Logic for sin sign involves original sign bit and quadrant
        swap_sign_bit_sin = _mm256_blendv_ps(_ps256_sign_mask, _ps256_0p5, swap_sign_bit_sin); // Recycle var
        // This part is tricky to implement compactly. Simplified logic:
        
        // Correct sign bit for sin is x_sign ^ ((quadrant & 4) ? 1 : 0) ? No...
        // Let's use standard conditional logic for sign
        __m256 sign_poly = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(emm2, _mm256_set1_epi32(4)), 29));
        __m256 combined_sign = _mm256_xor_ps(sign_bit_sin, sign_poly);
        *s = _mm256_xor_ps(sin_val, combined_sign);
    }

    if (c) {
        // Cos sign
        emm0 = _mm256_and_si256(_mm256_sub_epi32(emm2, _mm256_set1_epi32(2)), _mm256_set1_epi32(4));
        __m256 sign_poly_cos = _mm256_castsi256_ps(_mm256_slli_epi32(emm0, 29));
        *c = _mm256_xor_ps(cos_val, sign_poly_cos);
    }
}

__m256 Vec3SIMD::sin_256(const __m256& x) {
    __m256 s;
    sincos_256(x, &s, nullptr);
    return s;
}

__m256 Vec3SIMD::cos_256(const __m256& x) {
    __m256 c;
    sincos_256(x, nullptr, &c);
    return c;
}

// --- Exponential ---
__m256 Vec3SIMD::exp_256(const __m256& x) {
    // Fast Exp approximation
    __m256 tmp = _mm256_setzero_ps(), fx;
    __m256i emm0;
    
    // Clamp x
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    __m256 x_clamped = _mm256_max_ps(_mm256_min_ps(x, exp_hi), exp_lo);

    // Express exp(x) = exp(g + n*log(2)) = exp(g) * 2^n
    fx = _mm256_mul_ps(x_clamped, _mm256_set1_ps(1.44269504088896341f)); // log2(e)
    fx = _mm256_add_ps(fx, _ps256_0p5);
    
    // Round to integer
    emm0 = _mm256_cvttps_epi32(fx);
    tmp = _mm256_cvtepi32_ps(emm0);

    __m256 flag = _mm256_cmp_ps(tmp, fx, _CMP_GT_OQ);
    tmp = _mm256_sub_ps(tmp, _mm256_and_ps(flag, _ps256_1));
    fx = tmp;

    // Remainder g = x - n*log(2)
    const __m256 cephes_exp_C1 = _mm256_set1_ps(0.693359375f);
    const __m256 cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4f);
    
    __m256 z = _mm256_mul_ps(fx, cephes_exp_C1);
    __m256 z2 = _mm256_mul_ps(fx, cephes_exp_C2);
    x_clamped = _mm256_sub_ps(x_clamped, z);
    x_clamped = _mm256_sub_ps(x_clamped, z2);

    // Polynomial approximation for exp(g)
    z = _mm256_mul_ps(x_clamped, x_clamped);
    
    const __m256 cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1f);

    __m256 y = _mm256_mul_ps(cephes_exp_p0, x_clamped);
    y = _mm256_add_ps(y, cephes_exp_p1);
    y = _mm256_mul_ps(y, x_clamped);
    y = _mm256_add_ps(y, cephes_exp_p2);
    y = _mm256_mul_ps(y, x_clamped);
    y = _mm256_add_ps(y, cephes_exp_p3);
    y = _mm256_mul_ps(y, x_clamped);
    y = _mm256_add_ps(y, cephes_exp_p4);
    y = _mm256_mul_ps(y, x_clamped);
    y = _mm256_add_ps(y, cephes_exp_p5);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x_clamped);
    y = _mm256_add_ps(y, _ps256_1);

    // Build 2^n
    emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);

    return _mm256_mul_ps(y, pow2n);
}

// --- Logarithm ---
__m256 Vec3SIMD::log_256(const __m256& x) {
    // Natural Logarithm
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);
    __m256 x_valid = _mm256_max_ps(x, _ps256_min_norm_pos);  // Avoid denormals/zero

    __m256i emm0 = _mm256_srli_epi32(_mm256_castps_si256(x_valid), 23);
    // Keep only the exponent part
    emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(0x7f));
    __m256 e = _mm256_cvtepi32_ps(emm0);
    
    emm0 = _mm256_and_si256(_mm256_castps_si256(x_valid), _mm256_castps_si256(_ps256_inv_mant_mask));
    emm0 = _mm256_or_si256(emm0, _mm256_castps_si256(_ps256_0p5));
    __m256 m = _mm256_castsi256_ps(emm0);

    const __m256 cephes_sqrthf = _mm256_set1_ps(0.707106781186547524f);
    __m256 mask = _mm256_cmp_ps(m, cephes_sqrthf, _CMP_LT_OQ);
    __m256 tmp = _mm256_and_ps(mask, _ps256_1);
    
    e = _mm256_sub_ps(e, tmp);
    m = _mm256_add_ps(m, _mm256_and_ps(mask, m)); // m = m + (m if m < sqrt(0.5)) -> NO, Logic error in scalar adaptation?
    // Correction: if (m < sqrt(0.5)) { m *= 2; e -= 1; }
    // Above logic: tmp=1 if m<sqrt, e-=1. Correct.
    // m update: m = blend(m, m*2, mask)? Or manipulate bits? 
    // Simplified: m = 0.5 * significand. If < 0.707, multiply by loop or adjust.
    // Let's stick to standard approx:
    
    // Correct approach using blend:
    // x = m * 2^e
    // if m < sqrt(0.5), m = 2*m, e = e - 1
    __m256 m_scaled = _mm256_add_ps(m, m); // 2*m
    m = _mm256_blendv_ps(m, m_scaled, mask);
    
    m = _mm256_sub_ps(m, _ps256_1);

    // Polynomial
    const __m256 cephes_log_p0 = _mm256_set1_ps(7.0376836292E-2f);
    const __m256 cephes_log_p1 = _mm256_set1_ps(-1.1514610310E-1f);
    const __m256 cephes_log_p2 = _mm256_set1_ps(1.1676998740E-1f);
    const __m256 cephes_log_p3 = _mm256_set1_ps(-1.2420140846E-1f);
    const __m256 cephes_log_p4 = _mm256_set1_ps(1.4249322787E-1f);
    const __m256 cephes_log_p5 = _mm256_set1_ps(-1.6668057665E-1f);
    const __m256 cephes_log_p6 = _mm256_set1_ps(2.0000714765E-1f);
    const __m256 cephes_log_p7 = _mm256_set1_ps(-2.4999993993E-1f);
    const __m256 cephes_log_p8 = _mm256_set1_ps(3.3333331174E-1f);

    __m256 t2 = _mm256_mul_ps(m, m);
    __m256 y = _mm256_mul_ps(m, cephes_log_p0);
    y = _mm256_add_ps(y, cephes_log_p1);
    y = _mm256_mul_ps(y, m);
    y = _mm256_add_ps(y, cephes_log_p2);
    y = _mm256_mul_ps(y, m);
    y = _mm256_add_ps(y, cephes_log_p3);
    y = _mm256_mul_ps(y, m);
    y = _mm256_add_ps(y, cephes_log_p4);
    y = _mm256_mul_ps(y, m);
    y = _mm256_add_ps(y, cephes_log_p5);
    y = _mm256_mul_ps(y, m);
    y = _mm256_add_ps(y, cephes_log_p6);
    y = _mm256_mul_ps(y, m);
    y = _mm256_add_ps(y, cephes_log_p7);
    y = _mm256_mul_ps(y, m);
    y = _mm256_add_ps(y, cephes_log_p8);
    y = _mm256_mul_ps(y, m);
    y = _mm256_mul_ps(y, m);
    y = _mm256_sub_ps(y, _mm256_mul_ps(t2, _ps256_0p5)); // y - 0.5 * x^2
    y = _mm256_add_ps(y, m); // y + x

    const __m256 cephes_log_q1 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 cephes_log_q2 = _mm256_set1_ps(0.693359375f);
    
    __m256 z = _mm256_mul_ps(e, cephes_log_q1);
    y = _mm256_add_ps(y, z);
    z = _mm256_mul_ps(e, cephes_log_q2);
    y = _mm256_add_ps(y, z);

    // NaNs for invalid
    return _mm256_or_ps(y, invalid_mask); 
}

// --- Power ---
__m256 Vec3SIMD::pow_256(const __m256& a, const __m256& b) {
    // pow(x,y) = exp(y * log(x))
    // Note: This does not handle 0^0 or neg^int correctly
    __m256 log_val = log_256(a);
    __m256 mul_val = _mm256_mul_ps(b, log_val);
    return exp_256(mul_val);
}

// Legacy wrapper
Vec3SIMD Vec3SIMD::pow(const Vec3SIMD& v, float exponent) {
    return Vec3SIMD( Vec3SIMD::pow_256(v.data, _mm256_set1_ps(exponent)) );
}

// --- ArcTan2 ---
__m256 Vec3SIMD::atan2_256(const __m256& y, const __m256& x) {
    // Standard Atan2 approx
    __m256 abs_y = _mm256_and_ps(y, _ps256_inv_sign_mask);
    __m256 abs_x = _mm256_and_ps(x, _ps256_inv_sign_mask);
    
    __m256 swap_mask = _mm256_cmp_ps(abs_y, abs_x, _CMP_GT_OQ);
    
    // If y > x, ratio = -x/y, angle_offset = PI/2
    __m256 num = _mm256_blendv_ps(abs_y, _mm256_xor_ps(abs_x, _ps256_sign_mask), swap_mask); // y or -x
    __m256 den = _mm256_blendv_ps(abs_x, abs_y, swap_mask); // x or y
    
    // Add epsilon to den to avoid div zero
    den = _mm256_add_ps(den, _ps256_min_norm_pos);
    
    __m256 ratio = _mm256_div_ps(num, den);
    __m256 ratio_sq = _mm256_mul_ps(ratio, ratio);

    // Polynomial
    const __m256 atan_p0 = _mm256_set1_ps(-0.0464964749f);
    const __m256 atan_p1 = _mm256_set1_ps(0.15931422f);
    const __m256 atan_p2 = _mm256_set1_ps(-0.327622764f);
    const __m256 atan_p3 = _mm256_set1_ps(0.999787841f);
    
    __m256 z = _mm256_mul_ps(ratio_sq, atan_p0);
    z = _mm256_add_ps(z, atan_p1);
    z = _mm256_mul_ps(z, ratio_sq);
    z = _mm256_add_ps(z, atan_p2);
    z = _mm256_mul_ps(z, ratio_sq);
    z = _mm256_add_ps(z, atan_p3);
    z = _mm256_mul_ps(z, ratio); // z is atan(ratio)

    // Adjust for swapped coords
    const __m256 PI_2 = _mm256_set1_ps(1.57079632679f);
    __m256 angle = _mm256_blendv_ps(z, _mm256_add_ps(z, PI_2), swap_mask); // + PI/2 if swapped
    
    // Adjust for quadrants (signs of X and Y)
    // if x < 0: angle = PI - angle (if y>0) or -PI - angle (if y<0)?
    // atan2 Logic:
    // Q1 (+x, +y): atan(y/x) -> Positive
    // Q2 (-x, +y): PI - atan(y/|x|) -> Positive
    // Q3 (-x, -y): -PI + atan(|y|/|x|) -> Negative
    // Q4 (+x, -y): -atan(|y|/x) -> Negative
    // Current 'angle' is always positive in [0, PI/2] (relative to nearest axis)
    
    // Apply sign of ratio? No, ratio was calculated with abs. 
    // Sign of Y is the final sign of result mostly. 
    // But PI offset depends on X.
    
    const __m256 PI = _mm256_set1_ps(3.14159265359f);
    
    // If x < 0: add PI
    __m256 x_neg_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
    angle = _mm256_blendv_ps(angle, _mm256_sub_ps(PI, angle), x_neg_mask);
    
    // Apply sign of Y
    __m256 y_sign = _mm256_and_ps(y, _ps256_sign_mask);
    angle = _mm256_xor_ps(angle, y_sign);
    
    return angle;
}

// --- ACos ---
__m256 Vec3SIMD::acos_256(const __m256& x) {
    // acos(x) = PI/2 - asin(x)
    // asin(x) approx: x * (1 + x^2 * P(x^2))?
    // Simplified: acos(x) = atan2(sqrt(1-x*x), x)
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 one_min_x2 = _mm256_sub_ps(_ps256_1, x2);
    __m256 sqrt_term = _mm256_sqrt_ps(_mm256_max_ps(one_min_x2, _mm256_setzero_ps()));
    return atan2_256(sqrt_term, x);
}

// --- Random Unit Vector 8x (Re-check if implementation is correct after paste) ---


// --- Skaler Uyum ve Legacy Metotlar (Yavas) ---

// Skaler Dot Product (Tek bir 3D vektor için)
float Vec3SIMD::dot(const Vec3SIMD& other) const {
    // Sadece ilk 3 bileşeni (x, y, z) çarpıp toplar (Horizontal Sum)
    __m256 prod = _mm256_mul_ps(data, other.data);

    // Yatay Toplama (x*x + y*y + z*z)
    __m128 lo = _mm256_castps256_ps128(prod);
    __m128 hi = _mm256_extractf128_ps(prod, 1);

    __m128 sum_128 = _mm_add_ps(lo, hi);

    // x, y, z'yi toplamak için tekil bileşenlere indirge
    // Sadece h0, h1, h2'yi toplamak gerekir.
    alignas(16) float sum_arr[4];
    _mm_store_ps(sum_arr, sum_128);

    return sum_arr[0] + sum_arr[1] + sum_arr[2];
}

// Uzunluk Karesi (Tek bir 3D vektor için)
float Vec3SIMD::length_squared() const {
    return this->dot(*this);
}

// Uzunluk (Tek bir 3D vektor için)
float Vec3SIMD::length() const {
    return std::sqrt(length_squared());
}

// Tek bir vektorun sıfıra yakın olup olmadığını kontrol et
bool Vec3SIMD::near_zero() const {
    const float s = 1e-6f;
    return (std::fabs(x()) < s) && (std::fabs(y()) < s) && (std::fabs(z()) < s);
}

float Vec3SIMD::max_component() const {
    return std::max({ x(), y(), z() });
}

Vec3SIMD Vec3SIMD::max(const Vec3SIMD& v, float scalar) {
    return Vec3SIMD(_mm256_max_ps(v.data, _mm256_set1_ps(scalar)));
}

Vec3SIMD Vec3SIMD::min(const Vec3SIMD& a, const Vec3SIMD& b) {
    return Vec3SIMD(_mm256_min_ps(a.data, b.data));
}

Vec3SIMD Vec3SIMD::max(const Vec3SIMD& a, const Vec3SIMD& b) {
    return Vec3SIMD(_mm256_max_ps(a.data, b.data));
}

Vec3SIMD Vec3SIMD::clamp(const Vec3SIMD& v, float minVal, float maxVal) {
    __m256 min_vec = _mm256_set1_ps(minVal);
    __m256 max_vec = _mm256_set1_ps(maxVal);
    __m256 res_min = _mm256_max_ps(v.data, min_vec);
    return Vec3SIMD(_mm256_min_ps(res_min, max_vec));
}

// --- 3D Vektor Paketi Operasyonları (8x Paralel) ---

__m256 Vec3SIMD::dot_product_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z,
    const Vec3SIMD& v_x, const Vec3SIMD& v_y, const Vec3SIMD& v_z) {
    __m256 term_x = _mm256_mul_ps(u_x.data, v_x.data);
    __m256 term_y = _mm256_mul_ps(u_y.data, v_y.data);
    __m256 term_z = _mm256_mul_ps(u_z.data, v_z.data);
    __m256 sum_xy = _mm256_add_ps(term_x, term_y);
    return _mm256_add_ps(sum_xy, term_z);
}

__m256 Vec3SIMD::length_squared_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z) {
    return dot_product_8x(u_x, u_y, u_z, u_x, u_y, u_z);
}

void Vec3SIMD::normalize_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z,
    Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z) {
    __m256 len_sq = length_squared_8x(u_x, u_y, u_z);
    __m256 len_inv = _mm256_rsqrt_ps(len_sq); // Yaklaşık ters karekök

    __m256 zero_mask = _mm256_cmp_ps(len_sq, _mm256_set1_ps(1e-10f), _CMP_LT_OS);

    __m256 nx = _mm256_mul_ps(u_x.data, len_inv);
    __m256 ny = _mm256_mul_ps(u_y.data, len_inv);
    __m256 nz = _mm256_mul_ps(u_z.data, len_inv);

    out_x.data = _mm256_blendv_ps(nx, u_x.data, zero_mask);
    out_y.data = _mm256_blendv_ps(ny, u_y.data, zero_mask);
    out_z.data = _mm256_blendv_ps(nz, u_z.data, zero_mask);
}

void Vec3SIMD::cross_8x(const Vec3SIMD& u_x, const Vec3SIMD& u_y, const Vec3SIMD& u_z,
    const Vec3SIMD& v_x, const Vec3SIMD& v_y, const Vec3SIMD& v_z,
    Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z) {
    // wx = uy*vz - uz*vy
    out_x.data = _mm256_sub_ps(_mm256_mul_ps(u_y.data, v_z.data), _mm256_mul_ps(u_z.data, v_y.data));
    // wy = uz*vx - ux*vz
    out_y.data = _mm256_sub_ps(_mm256_mul_ps(u_z.data, v_x.data), _mm256_mul_ps(u_x.data, v_z.data));
    // wz = ux*vy - uy*vx
    out_z.data = _mm256_sub_ps(_mm256_mul_ps(u_x.data, v_y.data), _mm256_mul_ps(u_y.data, v_x.data));
}

void Vec3SIMD::reflect_8x(const Vec3SIMD& v_x, const Vec3SIMD& v_y, const Vec3SIMD& v_z,
    const Vec3SIMD& n_x, const Vec3SIMD& n_y, const Vec3SIMD& n_z,
    Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z) {

    __m256 dot_vn = dot_product_8x(v_x, v_y, v_z, n_x, n_y, n_z);
    __m256 two_dot_vn = _mm256_mul_ps(dot_vn, _mm256_set1_ps(2.0f));

    out_x.data = _mm256_sub_ps(v_x.data, _mm256_mul_ps(two_dot_vn, n_x.data));
    out_y.data = _mm256_sub_ps(v_y.data, _mm256_mul_ps(two_dot_vn, n_y.data));
    out_z.data = _mm256_sub_ps(v_z.data, _mm256_mul_ps(two_dot_vn, n_z.data));
}

void Vec3SIMD::refract_8x(const Vec3SIMD& uv_x, const Vec3SIMD& uv_y, const Vec3SIMD& uv_z,
    const Vec3SIMD& n_x, const Vec3SIMD& n_y, const Vec3SIMD& n_z,
    const Vec3SIMD& etai_over_etat,
    Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z) {

    __m256 dot_neg_uv_n = dot_product_8x(uv_x, uv_y, uv_z, n_x, n_y, n_z);
    dot_neg_uv_n = _mm256_xor_ps(dot_neg_uv_n, _mm256_set1_ps(-0.0f));
    __m256 cos_theta = _mm256_min_ps(dot_neg_uv_n, _mm256_set1_ps(1.0f));

    __m256 cos_n_x = _mm256_mul_ps(cos_theta, n_x.data);
    __m256 perp_x = _mm256_mul_ps(etai_over_etat.data, _mm256_add_ps(uv_x.data, cos_n_x));
    // ... y ve z bileşenleri için de:
    __m256 cos_n_y = _mm256_mul_ps(cos_theta, n_y.data);
    __m256 perp_y = _mm256_mul_ps(etai_over_etat.data, _mm256_add_ps(uv_y.data, cos_n_y));

    __m256 cos_n_z = _mm256_mul_ps(cos_theta, n_z.data);
    __m256 perp_z = _mm256_mul_ps(etai_over_etat.data, _mm256_add_ps(uv_z.data, cos_n_z));

    // perp_len_sq hesapla
    __m256 perp_len_sq_x = _mm256_mul_ps(perp_x, perp_x);
    __m256 perp_len_sq_y = _mm256_mul_ps(perp_y, perp_y);
    __m256 perp_len_sq_z = _mm256_mul_ps(perp_z, perp_z);
    __m256 perp_len_sq = _mm256_add_ps(perp_len_sq_x, _mm256_add_ps(perp_len_sq_y, perp_len_sq_z)); // Düzeltilmiş toplama

    __m256 k = _mm256_sub_ps(_mm256_set1_ps(1.0f), perp_len_sq);
    __m256 reflect_mask = _mm256_cmp_ps(k, _mm256_set1_ps(0.0f), _CMP_LT_OS);

    __m256 sqrt_k = _mm256_sqrt_ps(_mm256_max_ps(k, _mm256_setzero_ps()));
    __m256 par_mult = _mm256_mul_ps(sqrt_k, _mm256_set1_ps(-1.0f));

    __m256 par_x = _mm256_mul_ps(par_mult, n_x.data);
    __m256 par_y = _mm256_mul_ps(par_mult, n_y.data);
    __m256 par_z = _mm256_mul_ps(par_mult, n_z.data);

    __m256 refract_x = _mm256_add_ps(perp_x, par_x);
    __m256 refract_y = _mm256_add_ps(perp_y, par_y);
    __m256 refract_z = _mm256_add_ps(perp_z, par_z);

    out_x.data = _mm256_blendv_ps(refract_x, uv_x.data, reflect_mask); 
    out_y.data = _mm256_blendv_ps(refract_y, uv_y.data, reflect_mask);
    out_z.data = _mm256_blendv_ps(refract_z, uv_z.data, reflect_mask);
}

// --- Skaler Random Metotlar (Optimized) ---
float Vec3SIMD::random_float() {
    if (s_vec3simd_rng_state == 0) init_simd_rng_if_needed();
    
    uint32_t x = s_vec3simd_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s_vec3simd_rng_state = x;
    
    // float [0, 1) transformation
    return float(x) * 2.3283064365386963e-10f; 
}

__m256 Vec3SIMD::random_float_8x() {
    alignas(32) float r[8];
    for (int i = 0; i < 8; i++) r[i] = random_float();
    return _mm256_load_ps(r);
}

__m256 Vec3SIMD::power_heuristic_8x(__m256 f, __m256 g) {
    __m256 f2 = _mm256_mul_ps(f, f);
    __m256 g2 = _mm256_mul_ps(g, g);
    return _mm256_div_ps(f2, _mm256_add_ps(f2, g2));
}

void Vec3SIMD::random_unit_vector_8x(Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z) {
    alignas(32) float x_arr[8], y_arr[8], z_arr[8];
    for (int i = 0; i < 8; ++i) {
        float rx, ry, rz, len_sq;
        do {
            rx = random_float() * 2.0f - 1.0f;
            ry = random_float() * 2.0f - 1.0f;
            rz = random_float() * 2.0f - 1.0f;
            len_sq = rx * rx + ry * ry + rz * rz;
        } while (len_sq >= 1.0f || len_sq < 1e-6f);

        float inv_len = 1.0f / std::sqrt(len_sq);
        x_arr[i] = rx * inv_len;
        y_arr[i] = ry * inv_len;
        z_arr[i] = rz * inv_len;
    }
    out_x.data = _mm256_load_ps(x_arr);
    out_y.data = _mm256_load_ps(y_arr);
    out_z.data = _mm256_load_ps(z_arr);
}

// --- Friend Fonksiyonlar (8x Paralel) ---
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
Vec3SIMD operator/(const Vec3SIMD& v, float t) {
    float inv_t = 1.0f / t;
    return v * inv_t;
}
