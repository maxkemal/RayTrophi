#include "SceneSelection.h"
#include "Light.h"
#include "Camera.h"
#include "Triangle.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "AABB.h"

// ═══════════════════════════════════════════════════════════════════════════════
// SCENE SELECTION IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

void SceneSelection::updatePositionFromSelection() {
    if (!selected.is_valid()) return;
    
    switch (selected.type) {
        case SelectableType::Light:
            if (selected.light) {
                selected.position = selected.light->position;
                // Lights don't have rotation/scale in the current system
                selected.rotation = Vec3(0, 0, 0);
                selected.scale = Vec3(1, 1, 1);
            }
            break;
            
        case SelectableType::Camera:
            if (selected.camera) {
                selected.position = selected.camera->lookfrom;
                // Could extract rotation from lookfrom/lookat/vup
                selected.rotation = Vec3(0, 0, 0);
                selected.scale = Vec3(1, 1, 1);
            }
            break;
            
        case SelectableType::Object:
            if (selected.object) {
                // Get center of bounding box as position
                AABB bounds;
                if (selected.object->bounding_box(0, 0, bounds)) {
                    selected.position = (bounds.min + bounds.max) * 0.5f;
                }
                
                // Get transform from TransformHandle if available
                auto transform = selected.object->getTransformHandle();
                if (transform) {
                    // Access base directly (not getter method)
                    Matrix4x4 mat = transform->base;
                    selected.position = Vec3(mat.m[0][3], mat.m[1][3], mat.m[2][3]);
                }
            }
            break;
            
        default:
            break;
    }
}

void SceneSelection::applyTransformToSelection(const Matrix4x4& delta_transform) {
    if (!selected.is_valid()) return;
    
    switch (selected.type) {
        case SelectableType::Light:
            if (selected.light) {
                // Extract translation from matrix
                Vec3 translation(delta_transform.m[0][3], delta_transform.m[1][3], delta_transform.m[2][3]);
                selected.light->position = selected.light->position + translation;
                selected.position = selected.light->position;
            }
            break;
            
        case SelectableType::Camera:
            if (selected.camera) {
                Vec3 translation(delta_transform.m[0][3], delta_transform.m[1][3], delta_transform.m[2][3]);
                selected.camera->lookfrom = selected.camera->lookfrom + translation;
                selected.camera->lookat = selected.camera->lookat + translation;
                selected.camera->update_camera_vectors();
                selected.position = selected.camera->lookfrom;
            }
            break;
            
        case SelectableType::Object:
            if (selected.object) {
                auto transform = selected.object->getTransformHandle();
                if (transform) {
                    Matrix4x4 current = transform->base;  // Access base directly
                    Matrix4x4 new_transform = delta_transform * current;
                    transform->setBase(new_transform);
                    selected.object->updateTransformedVertices();
                    updatePositionFromSelection();
                }
            }
            break;
            
        default:
            break;
    }
}

Matrix4x4 SceneSelection::getSelectionMatrix() const {
    Matrix4x4 result = Matrix4x4::identity();
    
    if (!selected.is_valid()) return result;
    
    switch (selected.type) {
        case SelectableType::Light:
            if (selected.light) {
                result.m[0][3] = selected.light->position.x;
                result.m[1][3] = selected.light->position.y;
                result.m[2][3] = selected.light->position.z;
            }
            break;
            
        case SelectableType::Camera:
            if (selected.camera) {
                result.m[0][3] = selected.camera->lookfrom.x;
                result.m[1][3] = selected.camera->lookfrom.y;
                result.m[2][3] = selected.camera->lookfrom.z;
            }
            break;
            
        case SelectableType::Object:
            if (selected.object) {
                auto transform = selected.object->getTransformHandle();
                if (transform) {
                    result = transform->base;  // Access base directly
                }
            }
            break;
            
        default:
            break;
    }
    
    return result;
}

bool SceneSelection::deleteSelected() {
    if (!selected.is_valid()) return false;
    
    // Note: Actual deletion from scene arrays must be done by the caller
    // This just marks the intent and clears selection
    // The caller should check selected.type and index before calling this
    
    bool had_selection = selected.is_valid();
    selected.clear();
    
    return had_selection;
}
