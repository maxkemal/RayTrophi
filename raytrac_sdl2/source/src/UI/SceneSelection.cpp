#include "SceneSelection.h"
#include "Light.h"
#include "Camera.h"
#include "Triangle.h"
#include "Matrix4x4.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "AABB.h"
#include "VDBVolume.h"
#include "GasVolume.h"
#include "ForceField.h" 
#include <algorithm>

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
                selected.rotation = Vec3(0, 0, 0);
                selected.scale = Vec3(1, 1, 1);
            }
            break;

        case SelectableType::CameraTarget:
            if (selected.camera) {
                selected.position = selected.camera->lookat;
                selected.rotation = Vec3(0, 0, 0);
                selected.scale = Vec3(0.5f, 0.5f, 0.5f); 
            }
            break;
            
        case SelectableType::Object:
            if (selected.object) {
                // Get transform from TransformHandle if available
                auto transform = selected.object->getTransformHandle();
                if (transform) {
                    Matrix4x4 mat = transform->base;
                    // Use decompose to extract full transform (position, rotation, scale)
                    mat.decompose(selected.position, selected.rotation, selected.scale);
                    
                    // Fallback: If decomposed position is near origin but mesh isn't,
                    // use bounding box center (handles legacy world-space vertex data)
                    if (selected.position.length() < 0.001f) {
                        AABB bounds;
                        if (selected.object->bounding_box(0, 0, bounds)) {
                            Vec3 bb_center = (bounds.min + bounds.max) * 0.5f;
                            // Only use BB center if mesh is clearly not at origin
                            if (bb_center.length() > 1.0f) {
                                selected.position = bb_center;
                            }
                        }
                    }
                } else {
                    // Fallback: Get center of bounding box as position
                    AABB bounds;
                    if (selected.object->bounding_box(0, 0, bounds)) {
                        selected.position = (bounds.min + bounds.max) * 0.5f;
                    }
                    selected.rotation = Vec3(0, 0, 0);
                    selected.scale = Vec3(1, 1, 1);
                }
            }
            break;

        case SelectableType::VDBVolume:
            if (selected.vdb_volume) {
                selected.position = selected.vdb_volume->getPosition();
                selected.rotation = selected.vdb_volume->getRotation();
                selected.scale = selected.vdb_volume->getScale();
            }
            break;
            
      
            
        case SelectableType::GasVolume:
            if (selected.gas_volume) {
                selected.position = selected.gas_volume->getPosition();
                selected.rotation = selected.gas_volume->getRotation();
                selected.scale = selected.gas_volume->getScale();
            }
            break;
            
        case SelectableType::ForceField:
            if (selected.force_field) {
                // Use explicit namespace and ensure type is defined
                selected.position = selected.force_field->position;
                selected.rotation = selected.force_field->rotation;
                selected.scale = selected.force_field->scale;
            }
            break;
            
        default:
            break;
    }
}

// Helper to apply transform to a single item
static void ApplyTransformToItem(SelectableItem& item, const Matrix4x4& delta_transform) {
    if (!item.is_valid()) return;

    switch (item.type) {
        case SelectableType::Light:
            if (item.light) {
                Vec3 translation(delta_transform.m[0][3], delta_transform.m[1][3], delta_transform.m[2][3]);
                item.light->position = item.light->position + translation;
                item.position = item.light->position;
            }
            break;
            
        case SelectableType::Camera:
            if (item.camera) {
                Vec3 translation(delta_transform.m[0][3], delta_transform.m[1][3], delta_transform.m[2][3]);
                item.camera->lookfrom = item.camera->lookfrom + translation;
                item.camera->lookat = item.camera->lookat + translation;
                item.camera->update_camera_vectors();
                item.position = item.camera->lookfrom;
            }
            break;

        case SelectableType::CameraTarget:
            if (item.camera) {
                Vec3 translation(delta_transform.m[0][3], delta_transform.m[1][3], delta_transform.m[2][3]);
                item.camera->lookat = item.camera->lookat + translation;
                item.camera->update_camera_vectors();
                item.position = item.camera->lookat;
            }
            break;
            
        case SelectableType::Object:
            if (item.object) {
                auto transform = item.object->getTransformHandle();
                if (transform) {
                    // Apply Delta to current transform
                    Matrix4x4 current = transform->base; 
                    // Note: This applies delta in World Space relative to object
                    // For proper multi-model rotation around a pivot, logic is complex.
                    // This creates "Individual Origins" behavior for rotation/scale usually.
                    // For translation, it is correct (all move same amount).
                    Matrix4x4 new_transform = delta_transform * current;
                    transform->setBase(new_transform);
                    item.object->updateTransformedVertices();
                    
                    // Update item position cache
                    Matrix4x4 mat = new_transform;
                    item.position = Vec3(mat.m[0][3], mat.m[1][3], mat.m[2][3]);
                }
            }
            break;

        case SelectableType::VDBVolume:
            if (item.vdb_volume) {
                Matrix4x4 current = item.vdb_volume->getTransform();
                Matrix4x4 new_transform = delta_transform * current;
                
                Vec3 p, r, s;
                new_transform.decompose(p, r, s);
                
                item.vdb_volume->setPosition(p);
                item.vdb_volume->setRotation(r);
                item.vdb_volume->setScale(s);
                
                item.position = p;
            }
            break;
            
        case SelectableType::GasVolume:
            if (item.gas_volume) {
                Matrix4x4 current = Matrix4x4::identity();
                if (auto t = item.gas_volume->getTransformHandle()) current = t->base;

                Matrix4x4 new_transform = delta_transform * current;
                
                Vec3 p, r, s;
                new_transform.decompose(p, r, s);
                
                item.gas_volume->setPosition(p);
                item.gas_volume->setRotation(r);
                item.gas_volume->setScale(s);
                
                item.position = p;
            }
            break;
            
        case SelectableType::ForceField:
            if (item.force_field) {
                // Construct TRS matrix manually if fromTRS is causing issues
                Matrix4x4 current = Matrix4x4::translation(item.force_field->position) * 
                                  Matrix4x4::rotationX(item.force_field->rotation.x * 0.0174533f) * 
                                  Matrix4x4::rotationY(item.force_field->rotation.y * 0.0174533f) * 
                                  Matrix4x4::rotationZ(item.force_field->rotation.z * 0.0174533f) * 
                                  Matrix4x4::scaling(item.force_field->scale);
                
                Matrix4x4 new_transform = delta_transform * current;
                Vec3 p, r, s;
                new_transform.decompose(p, r, s);
                
                item.force_field->position = p;
                item.force_field->rotation = r;
                item.force_field->scale = s;
                item.position = p;
            }
            break;
            
        default:
            break;
    }
}

void SceneSelection::applyTransformToSelection(const Matrix4x4& delta_transform) {
    if (multi_selection.empty()) return;

    for (auto& item : multi_selection) {
        ApplyTransformToItem(item, delta_transform);
    }
    
    // Update primary selection position for Gizmo
    updatePositionFromSelection();
}

Matrix4x4 SceneSelection::getSelectionMatrix() const {
    Matrix4x4 result = Matrix4x4::identity();
    
    if (!selected.is_valid()) return result;
    
    // Gizmo is placed at PRIMARY/Active selection
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

        case SelectableType::CameraTarget:
            if (selected.camera) {
                result.m[0][3] = selected.camera->lookat.x;
                result.m[1][3] = selected.camera->lookat.y;
                result.m[2][3] = selected.camera->lookat.z;
            }
            break;
            
        case SelectableType::Object:
            if (selected.object) {
                auto transform = selected.object->getTransformHandle();
                if (transform) {
                    result = transform->base;
                }
            }
            break;

        case SelectableType::VDBVolume:
            if (selected.vdb_volume) {
                result = selected.vdb_volume->getTransform();
            }
            break;
            
        case SelectableType::GasVolume:
            if (selected.gas_volume) {
                if (auto t = selected.gas_volume->getTransformHandle()) {
                    result = t->base;
                }
            }
            break;
            
        case SelectableType::ForceField:
            if (selected.force_field) {
                result = Matrix4x4::translation(selected.force_field->position) * 
                         Matrix4x4::rotationX(selected.force_field->rotation.x * 0.0174533f) * 
                         Matrix4x4::rotationY(selected.force_field->rotation.y * 0.0174533f) * 
                         Matrix4x4::rotationZ(selected.force_field->rotation.z * 0.0174533f) * 
                         Matrix4x4::scaling(selected.force_field->scale);
            }
            break;
            
        default:
            break;
    }
    
    return result;
}

bool SceneSelection::deleteSelected() {
    if (multi_selection.empty()) return false;
    clearSelection();
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// New Multi-Selection Operations
// ─────────────────────────────────────────────────────────────────────────

void SceneSelection::clearSelection() {
    multi_selection.clear();
    selected.clear();
}

bool SceneSelection::hasSelection() const {
    return !multi_selection.empty();
}

bool SceneSelection::isSelected(const SelectableItem& item) const {
    for (const auto& s : multi_selection) {
        if (s.type != item.type) continue;
        
        // For objects, compare by name (since same object may have different triangle pointers)
        if (s.type == SelectableType::Object) {
            if (!s.name.empty() && s.name == item.name) return true;
            // Fallback to pointer comparison if names are empty
            if (s.object == item.object) return true;
        }
        else if (s.type == SelectableType::Light) {
            if (s.light == item.light) return true;
        }
        else if (s.type == SelectableType::Camera || s.type == SelectableType::CameraTarget) {
                if (s.camera == item.camera) return true;
        }
        else if (s.type == SelectableType::VDBVolume) {
            if (s.vdb_volume == item.vdb_volume) return true;
        }
        else if (s.type == SelectableType::GasVolume) {
            if (s.gas_volume == item.gas_volume) return true;
        }
    }
    return false;
}

void SceneSelection::addToSelection(const SelectableItem& item) {
    if (!item.is_valid()) return;

    // Check if already selected
    if (!isSelected(item)) {
        multi_selection.push_back(item);
    }
    
    // Make this the active/primary selection
    selected = item;
    updatePositionFromSelection();
}

void SceneSelection::removeFromSelection(const SelectableItem& item) {
    // Custom removal with name-based comparison for objects
    auto it = std::remove_if(multi_selection.begin(), multi_selection.end(), 
        [&item](const SelectableItem& s) {
            if (s.type != item.type) return false;
            
            if (s.type == SelectableType::Object) {
                // Compare by name for objects
                if (!s.name.empty() && s.name == item.name) return true;
                return s.object == item.object;
            }
            else if (s.type == SelectableType::Light) {
                return s.light == item.light;
            }
            else if (s.type == SelectableType::Camera || s.type == SelectableType::CameraTarget) {
                return s.camera == item.camera;
            }
            else if (s.type == SelectableType::VDBVolume) {
                return s.vdb_volume == item.vdb_volume;
            }
            else if (s.type == SelectableType::GasVolume) {
                return s.gas_volume == item.gas_volume;
            }
            return false;
        });
    
    if (it != multi_selection.end()) {
        multi_selection.erase(it, multi_selection.end());
    }
    
    syncPrimarySelection();
}

void SceneSelection::syncPrimarySelection() {
    if (multi_selection.empty()) {
        selected.clear();
    } else {
        // Active item is the last selected one
        selected = multi_selection.back();
        updatePositionFromSelection();
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Legacy Wrappers using Multi-Selection
// ─────────────────────────────────────────────────────────────────────────

void SceneSelection::selectObject(std::shared_ptr<Triangle> obj, int index, const std::string& name) {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::Object;
    item.object = obj;
    item.object_index = index;
    item.name = name;
    addToSelection(item);
}

void SceneSelection::selectLight(std::shared_ptr<Light> light, int index, const std::string& name) {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::Light;
    item.light = light;
    item.light_index = index;
    item.name = name;
    addToSelection(item);
}

void SceneSelection::selectVDBVolume(std::shared_ptr<VDBVolume> vdb, int index, const std::string& name) {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::VDBVolume;
    item.vdb_volume = vdb;
    item.vdb_index = index;
    item.name = name;
    addToSelection(item);
}

void SceneSelection::selectCamera(std::shared_ptr<Camera> camera) {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::Camera;
    item.camera = camera;
    item.name = "Camera";
    addToSelection(item);
}

void SceneSelection::selectCameraTarget(std::shared_ptr<Camera> camera) {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::CameraTarget;
    item.camera = camera;
    item.name = "Camera Target";
    addToSelection(item);
}

void SceneSelection::selectWorld() {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::World;
    item.name = "World";
    addToSelection(item);
}

void SceneSelection::selectGasVolume(std::shared_ptr<GasVolume> gas, int index, const std::string& name) {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::GasVolume;
    item.gas_volume = gas;
    item.vdb_index = index; 
    item.name = name;
    addToSelection(item);
}

void SceneSelection::selectForceField(std::shared_ptr<Physics::ForceField> field, int index, const std::string& name) {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::ForceField;
    item.force_field = field;
    item.force_field_index = index;
    item.name = name;
    addToSelection(item);
}
