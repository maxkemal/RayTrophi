#include "SceneSelection.h"
#include "Light.h"
#include "Camera.h"
#include "Triangle.h"
#include "TriangleMesh.h"
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
#include <utility>

// ═══════════════════════════════════════════════════════════════════════════════
// SCENE SELECTION IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

namespace {
Transform* selectedObjectTransform(const SelectableItem& item) {
    if (item.mesh_object && item.mesh_object->transform) return item.mesh_object->transform.get();
    return item.object ? item.object->getTransformPtr() : nullptr;
}

bool selectedObjectBounds(const SelectableItem& item, AABB& bounds) {
    if (item.mesh_object) return item.mesh_object->bounding_box(0, 0, bounds);
    return item.object && item.object->bounding_box(0, 0, bounds);
}
}

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
            if (selected.mesh_object || selected.object) {
                // Get transform from TransformHandle if available
                Transform* transform = selectedObjectTransform(selected);
                if (transform) {
                    Matrix4x4 mat = transform->getPivotMatrix();
                    mat.decompose(selected.position, selected.rotation, selected.scale);
                    
                    // Fallback: If decomposed position is near origin but mesh isn't,
                    // use bounding box center (handles legacy world-space vertex data)
                    if (selected.position.length() < 0.001f) {
                        AABB bounds;
                        if (selectedObjectBounds(selected, bounds)) {
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
                    if (selectedObjectBounds(selected, bounds)) {
                        selected.position = (bounds.min + bounds.max) * 0.5f;
                    }
                    selected.rotation = Vec3(0, 0, 0);
                    selected.scale = Vec3(1, 1, 1);
                }
            }
            break;

        case SelectableType::VDBVolume:
            if (selected.vdb_volume) {
                Matrix4x4 mat = selected.vdb_volume->getPivotMatrix();
                mat.decompose(selected.position, selected.rotation, selected.scale);
            }
            break;
            
      
            
        case SelectableType::GasVolume:
            if (selected.gas_volume) {
                Matrix4x4 mat = selected.gas_volume->getPivotMatrix();
                mat.decompose(selected.position, selected.rotation, selected.scale);
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

        case SelectableType::ParticleSystem:
            selected.position = Vec3(0, 0, 0);
            selected.rotation = Vec3(0, 0, 0);
            selected.scale = Vec3(1, 1, 1);
            break;

        case SelectableType::SimulationDomain:
            selected.rotation = Vec3(0, 0, 0);
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
            if (item.mesh_object || item.object) {
                Transform* transform = selectedObjectTransform(item);
                if (transform) {
                    Matrix4x4 current = transform->getPivotMatrix();
                    // Note: This applies delta in World Space relative to object
                    // For proper multi-model rotation around a pivot, logic is complex.
                    // This creates "Individual Origins" behavior for rotation/scale usually.
                    // For translation, it is correct (all move same amount).
                    Matrix4x4 new_transform = delta_transform * current;
                    transform->setPivotMatrix(new_transform);
                    
                    // Update item position cache
                    item.position = Vec3(new_transform.m[0][3], new_transform.m[1][3], new_transform.m[2][3]);
                }
            }
            break;

        case SelectableType::VDBVolume:
            if (item.vdb_volume) {
                Matrix4x4 current = item.vdb_volume->getPivotMatrix();
                Matrix4x4 new_transform = delta_transform * current;
                item.vdb_volume->setPivotMatrix(new_transform);
                Vec3 p, r, s;
                new_transform.decompose(p, r, s);
                item.position = p;
            }
            break;
            
        case SelectableType::GasVolume:
            if (item.gas_volume) {
                Matrix4x4 current = item.gas_volume->getPivotMatrix();
                Matrix4x4 new_transform = delta_transform * current;
                item.gas_volume->setPivotMatrix(new_transform);
                Vec3 p, r, s;
                new_transform.decompose(p, r, s);
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

        case SelectableType::ParticleSystem:
            break;

        case SelectableType::SimulationDomain:
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
            if (selected.mesh_object || selected.object) {
                Transform* transform = selectedObjectTransform(selected);
                if (transform) {
                    result = transform->getPivotMatrix();
                }
            }
            break;

        case SelectableType::VDBVolume:
            if (selected.vdb_volume) {
                result = selected.vdb_volume->getPivotMatrix();
            }
            break;
            
        case SelectableType::GasVolume:
            if (selected.gas_volume) {
                result = selected.gas_volume->getPivotMatrix();
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

        case SelectableType::ParticleSystem:
            break;

        case SelectableType::SimulationDomain:
            result.m[0][3] = selected.position.x;
            result.m[1][3] = selected.position.y;
            result.m[2][3] = selected.position.z;
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
        
        // Flat meshes own object identity.  The Triangle pointer is only a
        // compatibility facade for tools that have not migrated yet.
        if (s.type == SelectableType::Object) {
            if (s.mesh_object || item.mesh_object) {
                if (s.mesh_object && item.mesh_object &&
                    s.mesh_object == item.mesh_object) return true;

                Transform* s_trans = selectedObjectTransform(s);
                Transform* item_trans = selectedObjectTransform(item);
                if (s_trans && s_trans == item_trans) return true;
            }
            if (s.object_index >= 0 && s.object_index == item.object_index) return true;
            if (s.object && item.object) {
                Transform* s_trans = selectedObjectTransform(s);
                Transform* item_trans = selectedObjectTransform(item);
                if (s_trans && s_trans == item_trans) return true;
            }
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
        else if (s.type == SelectableType::ForceField) {
            if (s.force_field == item.force_field) return true;
        }
        else if (s.type == SelectableType::ParticleSystem) {
            if (s.particle_system_index == item.particle_system_index) return true;
        }
        else if (s.type == SelectableType::SimulationDomain) {
            if (s.particle_system_index == item.particle_system_index &&
                s.simulation_domain_index == item.simulation_domain_index) return true;
        }
    }
    return false;
}

void SceneSelection::addToSelection(const SelectableItem& item) {
    if (!item.is_valid()) return;

    SelectableItem canonicalItem = item;
    if (canonicalItem.type == SelectableType::Object &&
        !canonicalItem.mesh_object && canonicalItem.object &&
        canonicalItem.object->parentMesh) {
        canonicalItem.mesh_object = canonicalItem.object->parentMesh;
        canonicalItem.mesh_face_index = canonicalItem.object->faceIndex;
    }

    // Check if already selected
    if (!isSelected(canonicalItem)) {
        multi_selection.push_back(canonicalItem);
    }
    
    // Make this the active/primary selection
    selected = canonicalItem;
    updatePositionFromSelection();
}

void SceneSelection::removeFromSelection(const SelectableItem& item) {
    // Custom removal with name-independent comparison for objects
    auto it = std::remove_if(multi_selection.begin(), multi_selection.end(), 
        [&item](const SelectableItem& s) {
            if (s.type != item.type) return false;
            
            if (s.type == SelectableType::Object) {
                if (s.mesh_object || item.mesh_object) {
                    if (s.mesh_object && item.mesh_object &&
                        s.mesh_object == item.mesh_object) return true;

                    Transform* s_trans = selectedObjectTransform(s);
                    Transform* item_trans = selectedObjectTransform(item);
                    if (s_trans && s_trans == item_trans) return true;
                }
                if (s.object_index >= 0 && s.object_index == item.object_index) return true;
                if (s.object && item.object) {
                    Transform* s_trans = selectedObjectTransform(s);
                    Transform* item_trans = selectedObjectTransform(item);
                    if (s_trans && s_trans == item_trans) return true;
                }
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
            else if (s.type == SelectableType::ForceField) {
                return s.force_field == item.force_field;
            }
            else if (s.type == SelectableType::ParticleSystem) {
                return s.particle_system_index == item.particle_system_index;
            }
            else if (s.type == SelectableType::SimulationDomain) {
                return s.particle_system_index == item.particle_system_index &&
                       s.simulation_domain_index == item.simulation_domain_index;
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
    if (obj && obj->parentMesh) {
        item.mesh_object = obj->parentMesh;
        item.mesh_face_index = obj->faceIndex;
    }
    item.object_index = index;
    item.name = name;
    addToSelection(item);
}

void SceneSelection::selectObject(std::shared_ptr<TriangleMesh> mesh, int index,
                                  const std::string& name, uint32_t face_index,
                                  std::shared_ptr<Triangle> compatibility_facade) {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::Object;
    item.mesh_object = std::move(mesh);
    item.mesh_face_index = face_index;
    if (!compatibility_facade && item.mesh_object && item.mesh_object->num_triangles() > 0) {
        const uint32_t safeFace = (std::min)(
            face_index, static_cast<uint32_t>(item.mesh_object->num_triangles() - 1));
        compatibility_facade = std::make_shared<Triangle>(item.mesh_object, safeFace);
    }
    item.object = std::move(compatibility_facade);
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

void SceneSelection::selectParticleSystem(int index, const std::string& name) {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::ParticleSystem;
    item.particle_system_index = index;
    item.name = name;
    addToSelection(item);
}

void SceneSelection::selectSimulationDomain(int particle_system_index, int domain_index, const std::string& name) {
    clearSelection();
    SelectableItem item;
    item.type = SelectableType::SimulationDomain;
    item.particle_system_index = particle_system_index;
    item.simulation_domain_index = domain_index;
    item.name = name;
    addToSelection(item);
}
