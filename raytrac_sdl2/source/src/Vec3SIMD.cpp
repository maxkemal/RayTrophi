#include "Vec3SIMD.h"
#include <stdexcept>

// Statik deðiþkenlerin baþlatýlmasý
std::mt19937 Vec3SIMD::rng(std::random_device{}());
std::uniform_real_distribution<float> Vec3SIMD::dist(0.0f, 1.0f);

// --- Constructors ---
Vec3SIMD::Vec3SIMD() : data(_mm256_setzero_ps()) {}
Vec3SIMD::Vec3SIMD(__m256 d) : data(d) {}
Vec3SIMD::Vec3SIMD(float val) : data(_mm256_set1_ps(val)) {}
Vec3SIMD::Vec3SIMD(const float arr[8]) : data(_mm256_loadu_ps(arr)) {}

// Tek bir 3D Vektörü AVX'e Yükler (Sadece ilk 3 bileþeni kullanýr)
Vec3SIMD::Vec3SIMD(float x, float y, float z) {
    // _mm256_set_ps(h7, h6, h5, h4, h3, h2, h1, h0) sýrasýyla doldurur
    // Bizim için h0=x, h1=y, h2=z olacak. Diðerleri 0.
    data = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, z, y, x);
}

// --- Accessors (Yavaþ) ---
float Vec3SIMD::get(int index) const {
    alignas(32) float result[8];
    _mm256_store_ps(result, data);
    if (index >= 0 && index < 8) return result[index];
    throw std::out_of_range("Vec3SIMD index out of range");
}

// Tekil 3D bileþenlere eriþim (Yavaþ: Sadece ilk 3 bileþeni döndürür)
float Vec3SIMD::x() const { return get(0); }
float Vec3SIMD::y() const { return get(1); }
float Vec3SIMD::z() const { return get(2); }

// --- Statik Fabrika Metotlarý ---
Vec3SIMD Vec3SIMD::set1(float val) {
    return Vec3SIMD{ _mm256_set1_ps(val) };
}
Vec3SIMD Vec3SIMD::setZero() {
    return Vec3SIMD{ _mm256_setzero_ps() };
}

// --- Aritmetik Operatörler (8x Paralel) ---
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

// Skaler Operatörler
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

// --- Karþýlaþtýrma Operatörleri (8x Paralel) ---
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
Vec3SIMD Vec3SIMD::pow(const Vec3SIMD& v, float exponent) {
    // Pow için doðrudan AVX intrinsic yoktur. Skaler döngü veya
    // SSE/AVX log/exp yaklaþýmlarý gerekir. Basitlik için skaler düþüþ yapýyoruz:
    alignas(32) float arr_in[8];
    alignas(32) float arr_out[8];
    _mm256_store_ps(arr_in, v.data);
    for (int i = 0; i < 8; ++i) {
        arr_out[i] = std::pow(arr_in[i], exponent);
    }
    return Vec3SIMD(_mm256_load_ps(arr_out));
}


// --- Skaler Uyum ve Legacy Metotlar (Yavaþ) ---

// Skaler Dot Product (Tek bir 3D vektör için)
float Vec3SIMD::dot(const Vec3SIMD& other) const {
    // Sadece ilk 3 bileþeni (x, y, z) çarpýp toplar (Horizontal Sum)
    __m256 prod = _mm256_mul_ps(data, other.data);

    // Yatay Toplama (x*x + y*y + z*z)
    __m128 lo = _mm256_castps256_ps128(prod);
    __m128 hi = _mm256_extractf128_ps(prod, 1);

    __m128 sum_128 = _mm_add_ps(lo, hi);

    // x, y, z'yi toplamak için tekil bileþenlere indirge
    // Sadece h0, h1, h2'yi toplamak gerekir.
    alignas(16) float sum_arr[4];
    _mm_store_ps(sum_arr, sum_128);

    return sum_arr[0] + sum_arr[1] + sum_arr[2];
}

// Uzunluk Karesi (Tek bir 3D vektör için)
float Vec3SIMD::length_squared() const {
    return this->dot(*this);
}

// Uzunluk (Tek bir 3D vektör için)
float Vec3SIMD::length() const {
    return std::sqrt(length_squared());
}

// Tek bir vektörün sýfýra yakýn olup olmadýðýný kontrol et
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

Vec3SIMD Vec3SIMD::clamp(const Vec3SIMD& v, float minVal, float maxVal) {
    __m256 min_vec = _mm256_set1_ps(minVal);
    __m256 max_vec = _mm256_set1_ps(maxVal);
    __m256 res_min = _mm256_max_ps(v.data, min_vec);
    return Vec3SIMD(_mm256_min_ps(res_min, max_vec));
}

// --- Skaler Random Metotlar (Yavaþ) ---
float Vec3SIMD::random_float() {
    return dist(rng);
}

// --- 3D Vektör Paketi Operasyonlarý (8x Paralel) ---

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
    __m256 len_inv = _mm256_rsqrt_ps(len_sq); // Yaklaþýk ters karekök

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
    // ... y ve z bileþenleri benzer þekilde hesaplanýr

    __m256 perp_len_sq = dot_product_8x(Vec3SIMD(perp_x), Vec3SIMD(_mm256_setzero_ps()), Vec3SIMD(_mm256_setzero_ps()),
        Vec3SIMD(perp_x), Vec3SIMD(_mm256_setzero_ps()), Vec3SIMD(_mm256_setzero_ps())); // Kýsaltýlmýþ hesap
    __m256 k = _mm256_sub_ps(_mm256_set1_ps(1.0f), perp_len_sq);
    __m256 reflect_mask = _mm256_cmp_ps(k, _mm256_set1_ps(0.0f), _CMP_LT_OS);

    // Basitçe yansýma hesaplamasý... (Gerçek Reflect kodu burada olmalý)
    // Þimdilik sadece kýrýlma mantýðý izlenir:
    __m256 par_x = _mm256_mul_ps(_mm256_mul_ps(_mm256_sqrt_ps(_mm256_max_ps(k, _mm256_setzero_ps())), n_x.data), _mm256_set1_ps(-1.0f));
    __m256 refract_x = _mm256_add_ps(perp_x, par_x);

    // Kýrýlma (refract) ve Yansýma (reflect) arasýnda seçim yapýlýr
    // Yansýma sonucu olarak geçici olarak UV kullanýlýyor (Daha önce yaptýðýmýz gibi)
    out_x.data = _mm256_blendv_ps(refract_x, uv_x.data, reflect_mask);
    // Y ve Z bileþenleri için de benzer kodlar yazýlmalýdýr.
    // ... (Y ve Z için de perp/par/refract hesaplamalarý)
    // out_y.data = _mm256_blendv_ps(refract_y, uv_y.data, reflect_mask);
    // out_z.data = _mm256_blendv_ps(refract_z, uv_z.data, reflect_mask);
}

void Vec3SIMD::random_unit_vector_8x(Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z) {
    alignas(32) float x_arr[8], y_arr[8], z_arr[8];
    for (int i = 0; i < 8; ++i) {
        float a = random_float() * 2.0f * M_PI;
        float z = random_float() * 2.0f - 1.0f;
        float r = std::sqrt(1.0f - z * z);
        x_arr[i] = r * std::cos(a);
        y_arr[i] = r * std::sin(a);
        z_arr[i] = z;
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