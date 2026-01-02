#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Matrix4x4.h"

/**
 * @brief Shared transform data for triangles.
 * 
 * This class allows multiple triangles from the same mesh to share
 * transform matrices, reducing memory usage from 256 bytes per triangle
 * to 8 bytes (shared_ptr) for static meshes or shared animated meshes.
 */
class Transform {
public:
    Matrix4x4 base;           // Base transform (from model loading)
    Matrix4x4 current;        // Current animation transform
    Matrix4x4 final;          // Combined: current * base
    Matrix4x4 normalTransform; // Cached inverse-transpose for normals
    
    // Decomposed Transform Components (for animation/gizmos)
    Vec3 position = Vec3(0,0,0);
    Vec3 rotation = Vec3(0,0,0); // Euler angles in degrees
    Vec3 scale = Vec3(1,1,1);

    Transform() 
        : base(Matrix4x4::identity())
        , current(Matrix4x4::identity())
        , final(Matrix4x4::identity())
        , normalTransform(Matrix4x4::identity())
        , dirty(true)
    {}
    
    // Reconstruct base matrix from components
    void updateMatrix() {
        const float deg2rad = 3.14159265358979f / 180.0f;
        Matrix4x4 T = Matrix4x4::translation(position);
        
        // Euler ZYX order (yaw, pitch, roll) matches standard decomposition
        Matrix4x4 Rx = Matrix4x4::rotationX(rotation.x * deg2rad);
        Matrix4x4 Ry = Matrix4x4::rotationY(rotation.y * deg2rad);
        Matrix4x4 Rz = Matrix4x4::rotationZ(rotation.z * deg2rad);
        Matrix4x4 R = Rz * Ry * Rx; 
        
        Matrix4x4 S = Matrix4x4::scaling(scale);
        
        base = T * R * S;
        dirty = true;
    }

    Transform(const Matrix4x4& baseTransform)
        : base(baseTransform)
        , current(Matrix4x4::identity())
        , final(baseTransform)
        , dirty(true)
    {
        updateNormalTransform();
    }

    void setBase(const Matrix4x4& baseTransform) {
        base = baseTransform;
        dirty = true;
    }

    void setCurrent(const Matrix4x4& currentTransform) {
        current = currentTransform;
        dirty = true;
    }

    void updateFinal() {
        if (dirty) {
            final = current * base;
            updateNormalTransform();
            dirty = false;
        }
    }

    const Matrix4x4& getFinal() {
        updateFinal();
        return final;
    }

    const Matrix4x4& getNormalTransform() {
        updateFinal();
        return normalTransform;
    }

    bool isDirty() const { return dirty; }
    void markDirty() { dirty = true; }

private:
    bool dirty;

    void updateNormalTransform() {
        normalTransform = final.inverse().transpose();
    }
};

#endif // TRANSFORM_H
