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
    std::string materialName;
    // Animasyon için: her vertexin bone ağırlıkları
    std::vector<std::vector<std::pair<int, float>>> vertexBoneWeights;  // [vertex][(boneIndex, weight)]
 
    void setFaceIndex(int idx) { faceIndex = idx; }
    int getFaceIndex() const { return faceIndex; }
    // Animasyon için: her vertexin orijinal bind pose pozisyonu
    std::vector<Vec3> originalVertexPositions;
    // Orijinal (başlangıç) vertex pozisyonları
    Vec3 original_v0, original_v1, original_v2;
    // Orijinal (başlangıç) normallar
    Vec3 original_n0, original_n1, original_n2;
    Vec3 v0, v1, v2;        // vertices
    Vec3 n0, n1, n2;        // normals
    Vec2 t0,t1, t2;     // texture coordinates
   // Vec3 tangent0, tangent1, tangent2;     // tangents
   // Vec3 bitangent0, bitangent1, bitangent2; // bitangents
    bool hasTangents;       // tangent basis var mı?
    std::shared_ptr<Material> mat_ptr;
    std::shared_ptr<GpuMaterial> gpuMaterialPtr; // açık isim
    OptixGeometryData::TextureBundle textureBundle;
    int smoothingGroup;
    Matrix4x4 transform;
	// Texture nesnesi
    std::shared_ptr<Texture> texture;  
    int smoothGroup;
    // Dönüştürülmüş haller
    Vec3 transformed_v0, transformed_v1, transformed_v2;
    Vec3 transformed_n0, transformed_n1, transformed_n2;
    // Default constructor
    Triangle();  
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& na, const Vec3& nb, const Vec3& nc, const Vec2& ta, const Vec2& tb, const Vec2& tc, std::shared_ptr<Material> m, int sg);

  
    bool has_tangent_basis() const;
    void setUVCoordinates(const Vec2& uv0, const Vec2& uv1, const Vec2& uv2);
   
    void set_transform(const Matrix4x4& t);
    static void updateTriangleTransform(Triangle& triangle, const Matrix4x4& transform);
    void render(SDL_Renderer* renderer, SDL_Texture* texture);
    std::tuple<Vec2, Vec2, Vec2> getUVCoordinates() const;
    // Set normals
    void set_normals(const Vec3& normal0, const Vec3& normal1, const Vec3& normal2);
    void setAssimpVertexIndices(unsigned int i0, unsigned int i1, unsigned int i2) {
        assimpVertexIndices = { i0, i1, i2 };
    }

    const std::array<unsigned int, 3>& getAssimpVertexIndices() const {
        return assimpVertexIndices;
    }
    std::shared_ptr<Material> getMaterial() const {
        return mat_ptr;
    }
    void setMaterial(const std::shared_ptr<Material>& mat) {
        mat_ptr = mat;
    }

    const std::string& getNodeName() const {
        return nodeName;
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    void updateTransformedVertices();
    void setNodeName(const std::string& name);
    void setBaseTransform(const Matrix4x4& transform);
    void initialize_transforms();
    void updateAnimationTransform(const Matrix4x4& animTransform);
    virtual bool bounding_box(float time0, float time1, AABB& output_box) const override;
    void update_bounding_box();
    Vec3 apply_bone_to_vertex(int vi, const std::vector<Matrix4x4>& finalBoneMatrices) const;
    void apply_skinning(const std::vector<Matrix4x4>& finalBoneMatrices);
    Vec3 apply_bone_to_normal(const Vec3& originalNormal, const std::vector<std::pair<int, float>>& boneWeights, const std::vector<Matrix4x4>& finalBoneMatrices) const;
private:
    Vec3 calculateBarycentricCoordinates(const Vec3& point) const;

    void applyUVCoordinatesToHitRecord(HitRecord& hitRecord, const std::shared_ptr<Triangle>& triangle);
    std::string nodeName;       // Bağlı olduğu node ismi
    Matrix4x4 baseTransform;    // Model uzayından dünya uzayına dönüşüm
    Matrix4x4 currentTransform; // Animasyon dönüşümü
    Matrix4x4 finalTransform;   // Toplam dönüşüm (base * current)
    Vec3 min_point;
    Vec3 max_point;
    
    Vec2 uv0, uv1, uv2;
    int faceIndex = -1;
    std::array<unsigned int, 3> assimpVertexIndices;
};

#endif // TRIANGLE_H
