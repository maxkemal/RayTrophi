//// Mesh.cpp
//#include "Mesh.h"
//#include "AABB.h"
//#include "PrincipledBSDF.h"
//#include <algorithm>
//#include "globals.h"
//#include "SmartUVProjection.h"
//
//
//void Mesh::applyUVProjection() {
//    auto trianglePointers = getTriangles();
//    if (trianglePointers.empty()) {
//        std::cerr << "Warning: No triangles in the mesh for UV projection." << std::endl;
//        return;
//    }
//
//    std::vector<Triangle> triangles;
//    triangles.reserve(trianglePointers.size());
//    for (const auto& trianglePtr : trianglePointers) {
//        triangles.push_back(*trianglePtr);
//    }
//
//    SmartUVProjection smartUVProjection(trianglePointers);
//    smartUVProjection.apply(ProjectionType::PLANAR);
//    // PrincipledBSDF materyallerine SmartUVProjection kullanmalarýný söyle
//    for (auto& triangle : triangles) {
//        if (auto principledBSDF = std::dynamic_pointer_cast<PrincipledBSDF>(triangle.mat_ptr)) {
//            principledBSDF->useSmartUVProjection = true;
//        }
//    }
//    //std::cout << "UV projection applied, updating coordinates." << std::endl;
//
//    updateUVCoordinates(triangles);
//}
//
//
//void Mesh::updateUVCoordinates(const std::vector<Triangle>& updatedTriangles) {
//    if (updatedTriangles.size() != getTriangles().size()) {
//        throw std::runtime_error("Number of updated triangles does not match the mesh");
//    }
//
//    auto trianglePointers = getTriangles();
//
//    for (size_t i = 0; i < updatedTriangles.size(); ++i) {
//        const auto& updatedTriangle = updatedTriangles[i];
//        const auto& uvCoords = updatedTriangle.getUVCoordinates();
//
//        const auto& uv0 = std::get<0>(uvCoords);
//        const auto& uv1 = std::get<1>(uvCoords);
//        const auto& uv2 = std::get<2>(uvCoords);
//
//      /*  std::cout << "Updating UVs for triangle " << i << ": "
//            << uv0 << ", " << uv1 << ", " << uv2 << std::endl;*/
//
//        trianglePointers[i]->setUVCoordinates(uv0, uv1, uv2);
//    }
//
//    texCoords.clear();
//    for (const auto& trianglePtr : trianglePointers) {
//        const auto& uvCoords = trianglePtr->getUVCoordinates();
//        texCoords.push_back(std::get<0>(uvCoords));
//        texCoords.push_back(std::get<1>(uvCoords));
//        texCoords.push_back(std::get<2>(uvCoords));
//
//       
//    }
//
//    
//}
//
//
//
//
//
//bool Mesh::hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
//    bool hit_anything = false;
//    auto closest_so_far = t_max;
//
//    for (const auto& face : faces) {
//        if (rayPolygonIntersect(r, face, t_min, closest_so_far, rec)) {
//            hit_anything = true;
//            closest_so_far = rec.t;
//            rec.material = material;
//
//            // Transform hit point and normal back to world space
//            rec.point = transform.transform_point(rec.point);
//            rec.normal = transform.transform_vector(rec.normal).normalize();
//        }
//    }
//
//    return hit_anything;
//}
//bool Mesh::rayTriangleIntersect(const Ray& r, const Vec3& v0, const Vec3& v1, const Vec3& v2,
//    const Vec3& n0, const Vec3& n1, const Vec3& n2,
//    const Vec2& t0, const Vec2& t1, const Vec2& t2,
//    double t_min, double t_max, HitRecord& rec) const {
//
//    Vec3 edge1 = v1 - v0;
//    Vec3 edge2 = v2 - v0;
//    Vec3 h = Vec3SIMD::cross(r.direction, edge2);
//    double a = Vec3::dot(edge1, h);
//
//    if (a > -EPSILON && a < EPSILON) return false;
//
//    double f = 1.0 / a;
//    Vec3 s = r.origin - v0;
//    double u = f * Vec3::dot(s, h);
//
//    if (u < 0.0 || u > 1.0) return false;
//
//    Vec3 q = Vec3SIMD::cross(s, edge1);
//    double v = f * Vec3::dot(r.direction, q);
//
//    if (v < 0.0 || u + v > 1.0) return false;
//
//    double t = f * Vec3::dot(edge2, q);
//
//    if (t < t_min || t > t_max) return false;
//
//    rec.t = t;
//    rec.point = r.at(t);
//
//    // Interpolate normal and texture coordinates
//    double w = 1.0 - u - v;
//    rec.normal = unit_vector(w * n0 + u * n1 + v * n2);
//    rec.set_face_normal(r, rec.normal);
//   
//
//    return true;
//}
//
//
//bool Mesh::rayPolygonIntersect(const Ray& r, const Face& face, double t_min, double t_max, HitRecord& rec) const {
//    if (face.vertexIndices.size() == 3) {
//        // Üçgen için mevcut rayTriangleIntersect fonksiyonunu kullan
//        return rayTriangleIntersect(r,
//            vertices[face.vertexIndices[0]], vertices[face.vertexIndices[1]], vertices[face.vertexIndices[2]],
//            normals[face.normalIndices[0]], normals[face.normalIndices[1]], normals[face.normalIndices[2]],
//            texCoords[face.texCoordIndices[0]], texCoords[face.texCoordIndices[1]], texCoords[face.texCoordIndices[2]],
//            t_min, t_max, rec);
//    }
//    else {
//        // Çokgen için fan triangulation yöntemini kullan
//        const Vec3& v0 = vertices[face.vertexIndices[0]];
//        const Vec3& n0 = normals[face.normalIndices[0]];
//        const Vec2& t0 = texCoords[face.texCoordIndices[0]];
//
//        for (size_t i = 1; i < face.vertexIndices.size() - 1; ++i) {
//            const Vec3& v1 = vertices[face.vertexIndices[i]];
//            const Vec3& v2 = vertices[face.vertexIndices[i + 1]];
//            const Vec3& n1 = normals[face.normalIndices[i]];
//            const Vec3& n2 = normals[face.normalIndices[i + 1]];
//            const Vec2& t1 = texCoords[face.texCoordIndices[i]];
//            const Vec2& t2 = texCoords[face.texCoordIndices[i + 1]];
//
//            if (rayTriangleIntersect(r, v0, v1, v2, n0, n1, n2, t0, t1, t2, t_min, t_max, rec)) {
//                return true;
//            }
//        }
//    }
//    return false;
//}
//bool Mesh::bounding_box(double time0, double time1, AABB& output_box) const {
//    output_box = AABB(min_point - (EPSILON), max_point + (EPSILON));
//    return true;
//}
//void Mesh::setTransform(const Matrix4x4& newTransform) {
//    transform = newTransform;
//  
//    updateBoundingBox();
//}
//void Mesh::updateBoundingBox() {
//    if (vertices.empty()) return;
//
//    Vec3 min(INFINITY, INFINITY, INFINITY);
//    Vec3 max(-INFINITY, -INFINITY, -INFINITY);
//
//    for (const auto& vertex : vertices) {
//        for (int i = 0; i < 3; i++) {
//            min[i] = fmin(min[i], vertex[i]);
//            max[i] = fmax(max[i], vertex[i]);
//        }
//    }
//
//    originalBox = AABB(min, max);
//
//    // Transform the bounding box corners
//    std::vector<Vec3> corners = {
//        Vec3(min.x, min.y, min.z), Vec3(max.x, min.y, min.z),
//        Vec3(min.x, max.y, min.z), Vec3(max.x, max.y, min.z),
//        Vec3(min.x, min.y, max.z), Vec3(max.x, min.y, max.z),
//        Vec3(min.x, max.y, max.z), Vec3(max.x, max.y, max.z)
//    };
//
//    Vec3 transformedMin(INFINITY, INFINITY, INFINITY);
//    Vec3 transformedMax(-INFINITY, -INFINITY, -INFINITY);
//
//    for (const auto& corner : corners) {
//        Vec3 transformedCorner = transform.transform_point(corner);
//        for (int i = 0; i < 3; i++) {
//            transformedMin[i] = fmin(transformedMin[i], transformedCorner[i]);
//            transformedMax[i] = fmax(transformedMax[i], transformedCorner[i]);
//        }
//    }
//
//    transformedBox = AABB(transformedMin, transformedMax);
//}
//
//
//
//void Mesh::triangulate() {
//    std::vector<Face> triangulatedFaces;
//    for (const auto& face : faces) {
//        if (face.vertexIndices.size() == 3) {
//            triangulatedFaces.push_back(face);
//        }
//        else {
//            // Simple fan triangulation
//            for (size_t i = 1; i < face.vertexIndices.size() - 1; ++i) {
//                if (face.vertexIndices.size() < 3) {
//                    throw std::runtime_error("Error: Insufficient vertex indices for face triangulation.");
//                }
//
//                Face newFace;
//                newFace.vertexIndices = { face.vertexIndices[0], face.vertexIndices[i], face.vertexIndices[i + 1] };
//
//                if (!face.normalIndices.empty()) {
//                    // Eðer orijinal yüzün normal bilgisi varsa, onu kullan
//                    if (face.normalIndices.size() >= 3) {
//                        newFace.normalIndices = { face.normalIndices[0], face.normalIndices[i], face.normalIndices[i + 1] };
//                    }
//                    else {
//                        // Eðer yeterli normal bilgisi yoksa, ilk normal indeksini kullan
//                        newFace.normalIndices = { face.normalIndices[0], face.normalIndices[0], face.normalIndices[0] };
//                    }
//                }
//                if (!face.texCoordIndices.empty()) {
//                    if (face.texCoordIndices.size() >= 3) {
//                        newFace.texCoordIndices = { face.texCoordIndices[0], face.texCoordIndices[i], face.texCoordIndices[i + 1] };
//                    }
//                    else {
//                        newFace.texCoordIndices = { face.texCoordIndices[0], face.texCoordIndices[0], face.texCoordIndices[0] };
//                    }
//                }
//
//                triangulatedFaces.push_back(newFace);
//            }
//        }
//    }
//    faces = std::move(triangulatedFaces);
//}
//
//std::vector<std::shared_ptr<Triangle>> Mesh::getTriangles() const {
//    std::vector<std::shared_ptr<Triangle>> triangles;
//    std::cout << "Mesh::getTriangles() called. Faces count: " << faces.size() << std::endl;
//
//    const Vec3 DEFAULT_NORMAL(0.0f, 1.0f, 0.0f);
//    const Vec3 DEFAULT_VERTEX(0.0f, 0.0f, 0.0f);
//    const Vec2 DEFAULT_TEXCOORD(0.0f, 0.0f);
//
//    auto safeGetVertex = [this, &DEFAULT_VERTEX](size_t index) -> const Vec3& {
//        return (index < vertices.size()) ? vertices[index] : DEFAULT_VERTEX;
//        };
//
//    auto safeGetNormal = [this, &DEFAULT_NORMAL](size_t index) -> const Vec3& {
//        return (index < normals.size()) ? normals[index] : DEFAULT_NORMAL;
//        };
//
//    auto safeGetTexCoord = [this, &DEFAULT_TEXCOORD](size_t index) -> const Vec2& {
//        return (index < texCoords.size()) ? texCoords[index] : DEFAULT_TEXCOORD;
//        };
//
//    auto calculateNormal = [](const Vec3& v1, const Vec3& v2, const Vec3& v3) -> Vec3 {
//        Vec3 edge1 = v2 - v1;
//        Vec3 edge2 = v3 - v1;
//        Vec3 normal = edge1.cross(edge2);
//        return normal.normalize();
//        };
//
//    for (const auto& face : faces) {
//        if (face.vertexIndices.size() < 3) {
//            std::cout << "Warning: Skipping face with less than 3 vertices." << std::endl;
//            continue;
//        }
//
//        try {
//            const Vec3& v1 = safeGetVertex(face.vertexIndices[0]);
//            const Vec3& v2 = safeGetVertex(face.vertexIndices[1]);
//            const Vec3& v3 = safeGetVertex(face.vertexIndices[2]);
//
//            Vec3 n1, n2, n3;
//            if (face.normalIndices.empty()) {
//                // Eðer normal bilgisi yoksa, yüzün geometrisinden hesapla
//                Vec3 calculatedNormal = calculateNormal(v1, v2, v3);
//                n1 = n2 = n3 = calculatedNormal;
//            }
//            else {
//                n1 = safeGetNormal(face.normalIndices[0]);
//                n2 = face.normalIndices.size() > 1 ? safeGetNormal(face.normalIndices[1]) : n1;
//                n3 = face.normalIndices.size() > 2 ? safeGetNormal(face.normalIndices[2]) : n1;
//            }
//
//            Vec2 t1 = face.texCoordIndices.empty() ? DEFAULT_TEXCOORD : safeGetTexCoord(face.texCoordIndices[0]);
//            Vec2 t2 = face.texCoordIndices.size() > 1 ? safeGetTexCoord(face.texCoordIndices[1]) : t1;
//            Vec2 t3 = face.texCoordIndices.size() > 2 ? safeGetTexCoord(face.texCoordIndices[2]) : t1;
//
//            triangles.push_back(std::make_shared<Triangle>(
//                v1, v2, v3, n1, n2, n3, t1, t2, t3, material, 0
//            ));
//        }
//        catch (const std::exception& e) {
//            std::cerr << "Error creating triangle: " << e.what() << std::endl;
//        }
//    }
//
//    std::cout << "Mesh::getTriangles() returning " << triangles.size() << " triangles." << std::endl;
//    return triangles;
//}
//void Mesh::setPosition(const Vec3& pos) {
//    position = pos;
//    updateTransform();
//}
//
//void Mesh::setScale(const Vec3& scl) {
//    scale = scl;
//    updateTransform();
//}
//
//void Mesh::setRotation(const Vec3& rot) {
//    rotation = rot;
//    updateTransform();
//}
//
//const Matrix4x4& Mesh::getTransform() const {
//    return transform;
//}
//
//void Mesh::updateTransform() {
//    // Create translation matrix
//    Matrix4x4 translationMatrix = Matrix4x4::translation(position.x, position.y, position.z);
//
//    // Create rotation matrices
//    Matrix4x4 rotationMatrixX = Matrix4x4::rotationX(rotation.x);
//    Matrix4x4 rotationMatrixY = Matrix4x4::rotationY(rotation.y);
//    Matrix4x4 rotationMatrixZ = Matrix4x4::rotationZ(rotation.z);
//
//    // Create scale matrix
//    Matrix4x4 scaleMatrix = Matrix4x4::scaling(scale.x, scale.y, scale.z);
//
//    // Combine transformations
//    transform = translationMatrix * rotationMatrixZ * rotationMatrixY * rotationMatrixX * scaleMatrix;
//
//    // Update inverse transform
//    inverseTransform = transform.inverse();
//
//    updateBoundingBox();
//}