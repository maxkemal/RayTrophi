#include "Triangle.h"
#include "Ray.h"
#include "AABB.h"
#include "globals.h"
#include <Dielectric.h>


Triangle::Triangle()
    : smoothGroup(0) {
}

Triangle::Triangle(const Vec3& a, const Vec3& b, const Vec3& c, std::shared_ptr<Material> m)
    : v0(a), v1(b), v2(c), mat_ptr(m), smoothGroup(0) {
    initialize_transforms();  // Transform iþlemlerini baþlat
}


void Triangle::setUVCoordinates(const Vec2& t0, const Vec2& t1, const Vec2& t2) {
    this->t0 = t0;
    this->t1 = t1;
    this->t2 = t2;
}

std::tuple<Vec2, Vec2, Vec2> Triangle::getUVCoordinates() const {
    return std::make_tuple(t0, t1, t2);
}
void Triangle::set_normals(const Vec3& normal0, const Vec3& normal1, const Vec3& normal2) {
    n0 = normal0;
    n1 = normal1;
    n2 = normal2;
}

void Triangle::set_transform(const Matrix4x4& t) {
    transform = t;
    // Vertex pozisyonlarýný dönüþüm matrisi ile güncelle
    v0 = transform.transform_point(v0);
    v1 = transform.transform_point(v1);
    v2 = transform.transform_point(v2);
    update_bounding_box();
}
void Triangle::updateTriangleTransform(Triangle& triangle, const Matrix4x4& transform) {
    triangle.set_transform(transform);
}

void Triangle::update_bounding_box() {
    Vec3 transformed_v0 = transform.transform_point(v0);
    Vec3 transformed_v1 = transform.transform_point(v1);
    Vec3 transformed_v2 = transform.transform_point(v2);

    min_point = Vec3(
        std::min({ transformed_v0.x, transformed_v1.x, transformed_v2.x }),
        std::min({ transformed_v0.y, transformed_v1.y, transformed_v2.y }),
        std::min({ transformed_v0.z, transformed_v1.z, transformed_v2.z })
    );
    max_point = Vec3(
        std::max({ transformed_v0.x, transformed_v1.x, transformed_v2.x }),
        std::max({ transformed_v0.y, transformed_v1.y, transformed_v2.y }),
        std::max({ transformed_v0.z, transformed_v1.z, transformed_v2.z })
    );
}
float safeAcos(float x) {
    if (x < -1.0) x = -1.0;
    else if (x > 1.0) x = 1.0;
    return std::acos(x);
}
Vec3 Triangle::calculateBarycentricCoordinates(const Vec3& point) const {
    // Üçgen kenarlarý arasýndaki vektörler
    const Vec3 v0v1 = v1 - v0;
    const Vec3 v0v2 = v2 - v0;
    const Vec3 p = point - v0;

    // Daha önce tekrar eden hesaplamalar azaltýldý
    const float d00 = Vec3::dot(v0v1, v0v1);
    const float d01 = Vec3::dot(v0v1, v0v2);
    const float d11 = Vec3::dot(v0v2, v0v2);
    const float d20 = Vec3::dot(p, v0v1);
    const float d21 = Vec3::dot(p, v0v2);

    // Determinant ve barycentrik koordinatlar
    const float denom = 1.0 / (d00 * d11 - d01 * d01); // Tersini almak yerine burada çarpým daha verimli
    const float v = (d11 * d20 - d01 * d21) * denom;
    const float w = (d00 * d21 - d01 * d20) * denom;
    const float u = 1.0 - v - w;

    return Vec3(u, v, w);
}

bool Triangle::has_tangent_basis() const {
    return hasTangents;
}
// Eþik deðeri bir sabit olarak hesaplayýn
const float cos_threshold = std::cos(45.0f * M_PI / 180.0f);

float smoothstep(float edge0, float edge1, float x) {
    // x'i [0,1] aralýðýna clamp et
    x = std::clamp((x - edge0) / (edge1 - edge0), 0.01f, 1.0f);

    // Yumuþak geçiþ için 3. dereceden polinom: 3x^2 - 2x^3
    return x * x * (3.0f - 2.0f * x);
}
bool Triangle::hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    // Vertexleri dönüþtürme
    const Vec3 edge1 = transformed_v1 - transformed_v0;
    const Vec3 edge2 = transformed_v2 - transformed_v0;

    Vec3 h = Vec3::cross(r.direction, edge2);
    float a = Vec3::dot(edge1, h);   
    float f = 1.0 / a;
    Vec3 s = r.origin - transformed_v0;
    float u = f * Vec3::dot(s, h);

    if (u < 0.0 || u > 1.0)
        return false;

    Vec3 q = Vec3::cross(s, edge1);
    float v = f * Vec3::dot(r.direction, q);

    if (v < 0.0 || u + v > 1.0)
        return false;

    float t = f * Vec3::dot(edge2, q);

    if (t < t_min || t > t_max)
        return false;

    rec.triangle = this;
    rec.t = t;
    rec.point = r.at(t);
    const float w = 1.0 - u - v;
    rec.face_normal = Vec3::cross(edge1, edge2);    
  
    rec.set_face_normal(r, rec.face_normal);
    rec.interpolated_normal = (w * transformed_n0 + u * transformed_n1 + v * transformed_n2).normalize();
    rec.normal = rec.interpolated_normal.normalize();
     
    // Barycentric koordinatlarý hesapla
    Vec3 bary = calculateBarycentricCoordinates(rec.point);

    // UV koordinatlarýný interpolate et
    Vec2 uv = bary.x * t0 + bary.y * t1 + bary.z * t2;
   
    rec.uv = uv;
    rec.u = uv.u;
    rec.v = uv.v;
    rec.material = mat_ptr;  
	//if (rec.material->type() != MaterialType::Dielectric)		
 //   if (Vec3::dot(r.direction, rec.normal) < 0) {
 //       rec.normal = -rec.normal; // Normal yönü doðruysa, deðiþtirme
 //   }

    rec.set_face_normal(r, rec.interpolated_normal);
    // Normal interpolasyonu gibi tangent ve bitangent'ý da interpolate et
    if (hasTangents) {
        // Tangent ve Bitangent interpolasyonu (DÜZGÜN KULLANIM)
        if (hasTangents) {
            rec.tangent = (tangent0 * bary.x + tangent1 * bary.y + tangent2 * bary.z);
            rec.bitangent = Vec3::cross(rec.normal, rec.tangent);
            if (Vec3::dot(rec.bitangent, bitangent0) < 0.0f) {
                rec.bitangent = -rec.bitangent;
            }

            rec.has_tangent = true; 
        }
        else {
            rec.has_tangent = false;
        }
    }
    float opacity = rec.material->get_opacity(rec.uv);
        if(opacity<1)
			return false; // Iþýk geçiyor
    /* if (rec.material->type() == MaterialType::Dielectric)
            return false;*/
   
    return true;
}

bool Triangle::bounding_box(double time0, double time1, AABB& output_box) const {
    Vec3 small(
        std::min({ transformed_v0.x, transformed_v1.x, transformed_v2.x }),
        std::min({ transformed_v0.y, transformed_v1.y, transformed_v2.y }),
        std::min({ transformed_v0.z, transformed_v1.z, transformed_v2.z })
    );

    Vec3 big(
        std::max({ transformed_v0.x, transformed_v1.x, transformed_v2.x }),
        std::max({ transformed_v0.y, transformed_v1.y, transformed_v2.y }),
        std::max({ transformed_v0.z, transformed_v1.z, transformed_v2.z })
    );

    double edge1_length = (transformed_v1 - transformed_v0).length();
    double edge2_length = (transformed_v2 - transformed_v0).length();
    double epsilon = std::min(edge1_length, edge2_length) * EPSILON;

    output_box = AABB(small - epsilon, big + epsilon);
    return true;
}


