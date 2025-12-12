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

    Transform() 
        : base(Matrix4x4::identity())
        , current(Matrix4x4::identity())
        , final(Matrix4x4::identity())
        , normalTransform(Matrix4x4::identity())
        , dirty(true)
    {}

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
