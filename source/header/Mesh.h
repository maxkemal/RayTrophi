#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "Hittable.h"
#include "Material.h"
#include "Vec3.h"
#include "Vec2.h"
#include "Triangle.h"
#include "Matrix4x4.h"

class Mesh : public Hittable {
public:
    struct Face {
        std::vector<int> vertexIndices;
        std::vector<int> normalIndices;
        std::vector<int> texCoordIndices;
    };
    Vec3 v0, v1, v2;  // Vertices
    Vec3 n0, n1, n2;  // Normals
    Vec2 t0, t1, t2;  // Texture coordinates
    std::vector<Vec2> globalUVs;
    Mesh(const std::vector<Vec3>& vertices, const std::vector<Vec3>& normals, const std::vector<Vec2>& texCoords, const std::vector<Face>& faces, const std::shared_ptr<Material>& assignedMaterial)
        : vertices(vertices), normals(normals), texCoords(texCoords), faces(faces), material(assignedMaterial) {
        // diðer üye deðiþkenler için baþlatma/güncelleme
    }    void triangulate();
    int smoothGroup;
    Mesh() = default;
    //explicit Mesh(const ObjLoader::ObjMesh& objMesh, const std::unordered_map<std::string, std::unique_ptr<ObjLoader::ObjMaterial>>& materials);

  

    void applyUVProjection();

    void calculateGlobalUVs();

    void updateUVCoordinates(const std::vector<Triangle>& updatedTriangles);

   virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override;
   bool rayPolygonIntersect(const Ray& r, const Face& face, double t_min, double t_max, HitRecord& rec) const;
   virtual bool bounding_box(double time0, double time1, AABB& output_box) const override;

    void setTransform(const Matrix4x4& newTransform);


    //static std::vector<std::shared_ptr<Mesh>> createFromObjModel(const ObjLoader::ObjModel& model);
     std::vector<std::shared_ptr<Triangle>> getTriangles() const;

     void setPosition(const Vec3& pos);
     void setScale(const Vec3& scale);
     void setRotation(const Vec3& rot);
     const Matrix4x4& getTransform() const;

private:
    Matrix4x4 transform;
    Matrix4x4 inverseTransform;
    Vec3 position;
    Vec3 scale;
    Vec3 rotation;

    void updateTransform();
  
    AABB originalBox;
    AABB transformedBox;
    std::shared_ptr<Triangle> triangles;
    //std::vector<Triangle> triangles;
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;
    std::vector<Vec2> texCoords;
    std::vector<Face> faces;
    std::shared_ptr<Material> material;
    Vec3 min_point;
    Vec3 max_point;
   
    void updateBoundingBox();
    bool rayTriangleIntersect(const Ray& r, const Vec3& v0, const Vec3& v1, const Vec3& v2,
        const Vec3& n0, const Vec3& n1, const Vec3& n2,
        const Vec2& t0, const Vec2& t1, const Vec2& t2,
        double t_min, double t_max, HitRecord& rec) const;
   
    
   
  
};