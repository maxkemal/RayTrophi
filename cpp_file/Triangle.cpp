#include "Triangle.h"
#include "Ray.h"
#include "AABB.h"
#include "globals.h"
#include <Dielectric.h>


Triangle::Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
    const Vec3& na, const Vec3& nb, const Vec3& nc,
    const Vec2& ta, const Vec2& tb, const Vec2& tc,
    const Vec3& tana, const Vec3& tanb, const Vec3& tanc,
    const Vec3& ba, const Vec3& bb, const Vec3& bc,
    bool hasTangentBasis,
    std::shared_ptr<Material> m,
    int sg)
    : v0(a), v1(b), v2(c),
    n0(na), n1(nb), n2(nc),
    t0(ta), t1(tb), t2(tc),
    tangent0(tana), tangent1(tanb), tangent2(tanc),
    bitangent0(ba), bitangent1(bb), bitangent2(bc),
    hasTangents(hasTangentBasis),
    mat_ptr(m),
    smoothingGroup(sg) {
    update_bounding_box();
    initialize_transforms();  // Transform iþlemlerini baþlat
    updateTransformedVertices(); // initialize_transforms sonrasý, çünkü transformed_vX hesaplamazsan AABB yanlýþ olur

    // Diðer gerekli baþlatmalar...
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
Vec3 Triangle::apply_bone_to_vertex(int vi, const std::vector<Matrix4x4>& finalBoneMatrices) const {
    Vec3 blended = Vec3(0);
    for (const auto& [boneIdx, weight] : vertexBoneWeights[vi]) {
        Vec3 transformed = finalBoneMatrices[boneIdx].transform_point(originalVertexPositions[vi]);
        blended += transformed * weight;
    }
    return blended;
}
void Triangle::apply_skinning(const std::vector<Matrix4x4>& finalBoneMatrices) {
    // Eksik veri durumunda erken çýkýþ ve orijinal pozisyon/normal atamasý
    if (vertexBoneWeights.size() != 3 || originalVertexPositions.size() != 3 ||
        original_n0.length_squared() < 1e-6 || original_n1.length_squared() < 1e-6 || original_n2.length_squared() < 1e-6) {
       // std::cerr << "[WARNING] Triangle skipping skinning — missing weight, position or normal data for " << getNodeName()
       //     << " (VBW size: " << vertexBoneWeights.size() << ", OVP size: " << originalVertexPositions.size()
        //    << ", Original Normals Valid: " << (original_n0.length_squared() >= 1e-6 && original_n1.length_squared() >= 1e-6 && original_n2.length_squared() >= 1e-6) << ").\n";
        // Eðer veri eksikse, transformed deðerleri orijinal deðerlere eþitle
        transformed_v0 = original_v0; transformed_v1 = original_v1; transformed_v2 = original_v2;
        transformed_n0 = original_n0; transformed_n1 = original_n1; transformed_n2 = original_n2;
        // Ana vertex ve normal üyelerini de güncelle (hit metodu için)
        v0 = original_v0; v1 = original_v1; v2 = original_v2;
        n0 = original_n0; n1 = original_n1; n2 = original_n2;
        update_bounding_box();
        return;
    }

    // Pozisyonlara skinning uygulama
    for (int vi = 0; vi < 3; ++vi) { // Her bir vertex için
        if (vertexBoneWeights[vi].empty()) {
           // std::cerr << "[WARNING] vertex " << vi << " has no bone weights for " << getNodeName() << ", keeping original position.\n";
            // Aðýrlýðý olmayan vertex, orijinal (bind pose) pozisyonunu korumalýdýr.
            if (vi == 0) transformed_v0 = original_v0;
            else if (vi == 1) transformed_v1 = original_v1;
            else if (vi == 2) transformed_v2 = original_v2;
        }
        else {
            Vec3 blendedPosition = Vec3(0);
            for (auto& [boneIdx, weight] : vertexBoneWeights[vi]) {
                if (boneIdx >= finalBoneMatrices.size()) {
                   // std::cerr << "[ERROR] POS: boneIdx " << boneIdx << " out of bounds! finalBoneMatrices.size(): " << finalBoneMatrices.size()
                     //   << " for vertex " << vi << " in triangle " << getNodeName() << " (Node: " << nodeName << ")\n";
                    continue; // Hata durumunda bu kemiði atla
                }
                Vec3 transformedVertex = finalBoneMatrices[boneIdx].transform_point(originalVertexPositions[vi]);
                blendedPosition += transformedVertex * weight;
            }
            if (vi == 0) transformed_v0 = blendedPosition;
            else if (vi == 1) transformed_v1 = blendedPosition;
            else if (vi == 2) transformed_v2 = blendedPosition;
        }
    }

    // Normallere skinning uygulama
    // apply_bone_to_normal metodunu burada çaðýrýyoruz
    transformed_n0 = apply_bone_to_normal(original_n0, vertexBoneWeights[0], finalBoneMatrices);
    transformed_n1 = apply_bone_to_normal(original_n1, vertexBoneWeights[1], finalBoneMatrices);
    transformed_n2 = apply_bone_to_normal(original_n2, vertexBoneWeights[2], finalBoneMatrices);

    // Hit metodu tarafýndan kullanýlan ana vertex ve normal üyelerini güncelle
    v0 = transformed_v0;
    v1 = transformed_v1;
    v2 = transformed_v2;

    n0 = transformed_n0.normalize(); // Normallerin normalize edildiðinden emin olun
    n1 = transformed_n1.normalize();
    n2 = transformed_n2.normalize();

    update_bounding_box();
}

Vec3 Triangle::apply_bone_to_normal(const Vec3& originalNormal,
    const std::vector<std::pair<int, float>>& boneWeights,
    const std::vector<Matrix4x4>& finalBoneMatrices) const {
    Vec3 blended = Vec3(0);
    for (const auto& [boneIdx, weight] : boneWeights) {
        Matrix4x4 normalMat = finalBoneMatrices[boneIdx].inverse().transpose();
        Vec3 transformed = normalMat.transform_vector(originalNormal);
        blended += transformed * weight;
    }
    return blended;
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
void Triangle::updateTransformedVertices() {
    transformed_v0 = finalTransform.transform_point(original_v0);
    transformed_v1 = finalTransform.transform_point(original_v1);
    transformed_v2 = finalTransform.transform_point(original_v2);

    // Normallarý güncelle
    Matrix4x4 normalTransform = finalTransform.inverse().transpose();
    transformed_n0 = normalTransform.transform_vector(original_n0).normalize();
    transformed_n1 = normalTransform.transform_vector(original_n1).normalize();
    transformed_n2 = normalTransform.transform_vector(original_n2).normalize();

    //  Embree ve raytracer için ana vertex'leri güncelle
    v0 = transformed_v0;
    v1 = transformed_v1;
    v2 = transformed_v2;

    update_bounding_box();
}
void Triangle::setNodeName(const std::string& name) {
    nodeName = name;
}

void Triangle::setBaseTransform(const Matrix4x4& transform) {
    baseTransform = transform;
    updateTransformedVertices();
}

void Triangle::initialize_transforms() {
    // Orijinal vertexleri sakla
    original_v0 = v0;
    original_v1 = v1;
    original_v2 = v2;

    original_n0 = n0;
    original_n1 = n1;
    original_n2 = n2;

    // Baþlangýçta transformed deðerler orijinal deðerlerle ayný
    transformed_v0 = v0;
    transformed_v1 = v1;
    transformed_v2 = v2;

    transformed_n0 = n0;
    transformed_n1 = n1;
    transformed_n2 = n2;

    // processTriangles'da vertexler ve normallar zaten transform edilmiþ olarak geliyor
    baseTransform = Matrix4x4::identity();
    currentTransform = Matrix4x4::identity();
    finalTransform = Matrix4x4::identity();
}

void Triangle::updateAnimationTransform(const Matrix4x4& animTransform) {
    currentTransform = animTransform;
    finalTransform = currentTransform;  // Base transform zaten identity olduðu için
    updateTransformedVertices();
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


