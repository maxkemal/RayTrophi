#pragma once

#include <memory>
#include <string>
#include <vector>
#include "Vec3.h"
#include "matrix4x4.h"
#include "AABB.h"

// Forward declarations
class Light;
class Camera;
class Hittable;
class Triangle;

// ═══════════════════════════════════════════════════════════════════════════════
// SCENE SELECTION SYSTEM
// Unified selection handling for objects, lights, and camera
// ═══════════════════════════════════════════════════════════════════════════════

enum class SelectableType {
    None,
    Object,     // Mesh/Triangle
    Light,      // Any light type
    Camera,     // Scene camera (Position/Origin)
    CameraTarget // Scene camera target (LookAt)
};

enum class TransformMode {
    Translate,
    Rotate,
    Scale
};

enum class TransformSpace {
    World,
    Local
};

// ───────────────────────────────────────────────────────────────────────────────
// Selectable Item - Base interface for anything that can be selected
// ───────────────────────────────────────────────────────────────────────────────
struct SelectableItem {
    SelectableType type = SelectableType::None;
    std::string name = "Unnamed";
    
    // Pointers to actual objects (only one is valid at a time)
    std::shared_ptr<Light> light = nullptr;
    std::shared_ptr<Camera> camera = nullptr;
    std::shared_ptr<Triangle> object = nullptr;  // For mesh objects
    
    // For objects loaded from scene (index-based reference)
    int object_index = -1;
    int light_index = -1;
    
    // Transform data (cached for gizmo)
    Vec3 position = Vec3(0, 0, 0);
    Vec3 rotation = Vec3(0, 0, 0);  // Euler angles (degrees)
    Vec3 scale = Vec3(1, 1, 1);
    
    // Cached bounds for selection box performance
    AABB cached_aabb;
    bool has_cached_aabb = false;
    
    bool is_valid() const { return type != SelectableType::None; }

    bool operator==(const SelectableItem& other) const {
        if (type != other.type) return false;
        if (type == SelectableType::Object) return object == other.object;
        if (type == SelectableType::Light) return light == other.light;
        if (type == SelectableType::Camera) return camera == other.camera;
        if (type == SelectableType::CameraTarget) return camera == other.camera; // Same camera for target
        return false;
    }
    
    void clear() {
        type = SelectableType::None;
        light = nullptr;
        camera = nullptr;
        object = nullptr;
        object_index = -1;
        light_index = -1;
        has_cached_aabb = false;
    }
};

// ───────────────────────────────────────────────────────────────────────────────
// Scene Selection Manager
// ───────────────────────────────────────────────────────────────────────────────
class SceneSelection {
public:
    // Current PRIMARY selection (for gizmo placement)
    SelectableItem selected;
    
    // Multi-selection list (contains all selected items, including primary)
    std::vector<SelectableItem> multi_selection;
    bool multi_select_enabled = true; // Always enabled now

    // Transform mode settings
    TransformMode transform_mode = TransformMode::Translate;
    TransformSpace transform_space = TransformSpace::World;
    
    // Gizmo visibility
    bool show_gizmo = true;
    float gizmo_size = 1.0f;
    
    
    // ─────────────────────────────────────────────────────────────────────────
    // Selection Methods
    // ─────────────────────────────────────────────────────────────────────────
    void selectObject(std::shared_ptr<Triangle> obj, int index, const std::string& name = "Object");
    void selectLight(std::shared_ptr<Light> light, int index = -1, const std::string& name = "Light");
    void selectCamera(std::shared_ptr<Camera> camera);
    void selectCameraTarget(std::shared_ptr<Camera> camera);
    void clearSelection();
    
    // Multi-Selection Helpers
    void addToSelection(const SelectableItem& item);
    void removeFromSelection(const SelectableItem& item);
    bool isSelected(const SelectableItem& item) const;
    void syncPrimarySelection(); // Updates 'selected' from multi_selection list

    bool hasSelection() const;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Transform Helpers
    // ─────────────────────────────────────────────────────────────────────────
    
    void updatePositionFromSelection();
    void applyTransformToSelection(const Matrix4x4& transform);
    Matrix4x4 getSelectionMatrix() const;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Delete Selected
    // ─────────────────────────────────────────────────────────────────────────
    
    // Returns true if something was deleted
    bool deleteSelected();
    
    // ─────────────────────────────────────────────────────────────────────────
    // Keyboard Shortcuts
    // ─────────────────────────────────────────────────────────────────────────
    
    void handleKeyPress(int key) {
        switch (key) {
            case 'G': // Grab/Move (Blender style)
            case 'W': // Move (3ds Max style)
                transform_mode = TransformMode::Translate;
                break;
            case 'R': // Rotate
            case 'E': // Rotate (alt)
                transform_mode = TransformMode::Rotate;
                break;
            case 'S': // Scale
                transform_mode = TransformMode::Scale;
                break;
            case 'X': // Delete
            case 127: // Delete key
                deleteSelected();
                break;
        }
    }
};

// ───────────────────────────────────────────────────────────────────────────────
// Scene Hierarchy Node (for tree view)
// ───────────────────────────────────────────────────────────────────────────────
struct HierarchyNode {
    std::string name;
    SelectableType type;
    int index;  // Index in respective array
    std::vector<HierarchyNode> children;
    bool expanded = false;
    
    // Icon identifiers
    static constexpr const char* ICON_CAMERA = "[CAM]";
    static constexpr const char* ICON_LIGHT_POINT = "[*]";
    static constexpr const char* ICON_LIGHT_DIR = "[>]";
    static constexpr const char* ICON_LIGHT_SPOT = "[V]";
    static constexpr const char* ICON_LIGHT_AREA = "[#]";
    static constexpr const char* ICON_MESH = "[M]";
    static constexpr const char* ICON_UNKNOWN = "[?]";
};
