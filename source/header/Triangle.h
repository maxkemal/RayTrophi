#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <memory>
#include <string>
#include <algorithm>
#include "Hittable.h"
#include "Material.h"
#include "Vec2.h"
#include "Matrix4x4.h"
#include "Vec3SIMD.h"
#include <SDL.h>
#include <SDL_image.h>

class Triangle : public Hittable {
public:
   
    Vec3 v0, v1, v2;        // vertices
    Vec3 n0, n1, n2;        // normals
    Vec2 t0,t1, t2;     // texture coordinates
    Vec3 tangent0, tangent1, tangent2;     // tangents
    Vec3 bitangent0, bitangent1, bitangent2; // bitangents
    bool hasTangents;       // tangent basis var mý?
    std::shared_ptr<Material> mat_ptr;
    int smoothingGroup;
    Matrix4x4 transform;
    void setNodeName(const std::string& name) {
        nodeName = name;
    }

    const std::string& getNodeName() const {
        return nodeName;
    }

    void setBaseTransform(const Matrix4x4& transform) {
        baseTransform = transform;
        updateTransformedVertices();
    }

    void initialize_transforms() {
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
    void updateAnimationTransform(const Matrix4x4& animTransform) {
        currentTransform = animTransform;
        finalTransform = currentTransform;  // Base transform zaten identity olduðu için
        updateTransformedVertices();
    }

    std::shared_ptr<Texture> texture;
    std::string materialName;
    int smoothGroup;
    // Dönüþtürülmüþ haller
    Vec3 transformed_v0, transformed_v1, transformed_v2;
    Vec3 transformed_n0, transformed_n1, transformed_n2;
    // Default constructor
    Triangle();
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c, std::shared_ptr<Material> m);

    Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
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
        // Diðer gerekli baþlatmalar...
    }
  
    bool has_tangent_basis() const;
    void setUVCoordinates(const Vec2& uv0, const Vec2& uv1, const Vec2& uv2);
    void setMaterial(std::shared_ptr<Material> m) { mat_ptr = m; }
    void set_transform(const Matrix4x4& t);
    static void updateTriangleTransform(Triangle& triangle, const Matrix4x4& transform);
    void render(SDL_Renderer* renderer, SDL_Texture* texture);
    std::tuple<Vec2, Vec2, Vec2> getUVCoordinates() const;
    // Set normals
    void set_normals(const Vec3& normal0, const Vec3& normal1, const Vec3& normal2);
  
    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override;
    virtual bool bounding_box(double time0, double time1, AABB& output_box) const override;
   
private:
    Vec3 calculateBarycentricCoordinates(const Vec3& point) const;

    void applyUVCoordinatesToHitRecord(HitRecord& hitRecord, const std::shared_ptr<Triangle>& triangle);
    std::string nodeName;       // Baðlý olduðu node ismi
    Matrix4x4 baseTransform;    // Model uzayýndan dünya uzayýna dönüþüm
    Matrix4x4 currentTransform; // Animasyon dönüþümü
    Matrix4x4 finalTransform;   // Toplam dönüþüm (base * current)
    Vec3 min_point;
    Vec3 max_point;
    void update_bounding_box();
    Vec2 uv0, uv1, uv2;
    // Orijinal (baþlangýç) vertex pozisyonlarý
    Vec3 original_v0, original_v1, original_v2;
    // Orijinal (baþlangýç) normallar
    Vec3 original_n0, original_n1, original_n2;
    void updateTransformedVertices() {
        // Sadece animasyon transform'unu uygula
        transformed_v0 = finalTransform.transform_point(original_v0);
        transformed_v1 = finalTransform.transform_point(original_v1);
        transformed_v2 = finalTransform.transform_point(original_v2);
        // Normallarý güncelle
        Matrix4x4 normalTransform = finalTransform.inverse().transpose();
        transformed_n0 = normalTransform.transform_vector(original_n0).normalize();
        transformed_n1 = normalTransform.transform_vector(original_n1).normalize();
        transformed_n2 = normalTransform.transform_vector(original_n2).normalize();
        update_bounding_box();
    }
   
};

#endif // TRIANGLE_H
