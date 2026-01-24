#include "AABB.h"
#include <globals.h>

AABB surrounding_box(const AABB& box0, const AABB& box1) {
    // AABB'nin min ve max vektörlerini SIMD'e yükle
    __m128 box0_min = _mm_set_ps(0.0f, box0.min.z, box0.min.y, box0.min.x);
    __m128 box1_min = _mm_set_ps(0.0f, box1.min.z, box1.min.y, box1.min.x);
    __m128 box0_max = _mm_set_ps(0.0f, box0.max.z, box0.max.y, box0.max.x);
    __m128 box1_max = _mm_set_ps(0.0f, box1.max.z, box1.max.y, box1.max.x);

    // Min ve max hesaplamaları
    __m128 small = _mm_min_ps(box0_min, box1_min);
    __m128 big = _mm_max_ps(box0_max, box1_max);

    // SIMD sonuçlarını bir diziye aktar
    float small_array[4], big_array[4];
    _mm_store_ps(small_array, small);
    _mm_store_ps(big_array, big);
	
    // Vec3 nesneleri oluştur ve döndür
    return AABB(
        Vec3(small_array[0], small_array[1], small_array[2]),
        Vec3(big_array[0], big_array[1], big_array[2])
    );
}
// AABB sınıfındaki bir fonksiyon
int AABB::max_axis() const {
    // Diagonal vektörünü al
    Vec3 diagonal_vec = max - min;

    // Bileşenlerin değerlerini karşılaştırarak en büyüğünü seç
    if (diagonal_vec.x > diagonal_vec.y) {
        if (diagonal_vec.x > diagonal_vec.z) {
            return 0; // X ekseni
        }
        else {
            return 2; // Z ekseni
        }
    }
    else {
        if (diagonal_vec.y > diagonal_vec.z) {
            return 1; // Y ekseni
        }
        else {
            return 2; // Z ekseni
        }
    }
}

float AABB::surface_area() const {
    if (cached_surface_area < 0.0) {
        const Vec3 d = max - min;  // diagonal() çağrısını atla
        cached_surface_area = 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
    return cached_surface_area;
}

