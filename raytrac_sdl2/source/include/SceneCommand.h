/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          SceneCommand.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include <memory>
#include <array>
#include <vector>
#include <string>
#include "Hittable.h"
#include "Triangle.h"
#include "Texture.h"
#include "MeshModifiers.h"
#include "Paint/PaintLayerData.h"

// Forward declarations
struct UIContext;

// ============================================================================
// COMMAND PATTERN - Base Class for Undo/Redo Operations
// ============================================================================
// Each command represents a reversible operation on the scene.
// Commands are stored in a history stack for undo/redo functionality.
// ============================================================================

class SceneCommand {
public:
    virtual ~SceneCommand() = default;
    
    // Execute the command (forward operation)
    virtual void execute(UIContext& ctx) = 0;
    
    // Undo the command (reverse operation)
    virtual void undo(UIContext& ctx) = 0;
    
    // Command Type for selective history pruning
    enum class Type {
        Generic,
        Transform,
        Heavy // Delete, Duplicate (Memory intensive)
    };
    virtual Type getType() const = 0;
    
    // Get human-readable description for UI
    virtual std::string getDescription() const = 0;
};

// ============================================================================
// DELETE COMMAND - Stores deleted objects for restoration
// ============================================================================
class DeleteObjectCommand : public SceneCommand {
public:
    DeleteObjectCommand(const std::string& object_name,
                       const std::vector<std::shared_ptr<Triangle>>& deleted_triangles)
        : object_name_(object_name)
        , deleted_triangles_(deleted_triangles) {}
    
    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Heavy; }
    std::string getDescription() const override {
        return "Delete " + object_name_;
    }
    
private:
    std::string object_name_;
    std::vector<std::shared_ptr<Triangle>> deleted_triangles_;
};

// ============================================================================
// DUPLICATE COMMAND - Stores created objects for removal
// ============================================================================
class DuplicateObjectCommand : public SceneCommand {
public:
    DuplicateObjectCommand(const std::string& source_name,
                          const std::string& new_name,
                          const std::vector<std::shared_ptr<Triangle>>& new_triangles)
        : source_name_(source_name)
        , new_name_(new_name)
        , new_triangles_(new_triangles) {}
    
    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Heavy; }
    std::string getDescription() const override {
        return "Duplicate " + source_name_ + " -> " + new_name_;
    }
    
private:
    std::string source_name_;
    std::string new_name_;
    std::vector<std::shared_ptr<Triangle>> new_triangles_;
};

// ============================================================================
// TRANSFORM COMMAND - Stores position, rotation, scale changes
// ============================================================================
#include "Matrix4x4.h"

// ============================================================================
// TRANSFORM COMMAND - Stores position, rotation, scale changes
// ============================================================================
struct TransformState {
    Matrix4x4 matrix;
    Vec3 position;
    Vec3 rotation;
    Vec3 scale;
};

class TransformCommand : public SceneCommand {
public:
    TransformCommand(const std::string& object_name,
                    const TransformState& old_state,
                    const TransformState& new_state)
        : object_name_(object_name)
        , old_state_(old_state)
        , new_state_(new_state) {}
    
    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Transform; }
    std::string getDescription() const override {
        return "Transform " + object_name_;
    }
    
private:
    std::string object_name_;
    TransformState old_state_;
    TransformState new_state_;
    
    // Helper to apply state to objects
    void applyState(UIContext& ctx, const TransformState& state);
};

// ============================================================================
// LIGHT COMMANDS
// ============================================================================
#include "Light.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "PointLight.h"

struct LightState {
    Vec3 position;
    Vec3 direction;
    Vec3 u, v;
    float width = 0, height = 0;
    float radius = 0;
    float angle = 0;
    float falloff = 0;
    
    static LightState capture(const Light& light) {
        LightState s;
        s.position = light.position;
        
        switch (light.type()) {
            case LightType::Directional: {
                auto& l = (const DirectionalLight&)light;
                s.direction = l.direction;
                s.radius = l.radius;
                break;
            }
            case LightType::Spot: {
                auto& l = (const SpotLight&)light;
                s.direction = l.direction;
                s.radius = l.radius; // Range
                s.angle = l.getAngleDegrees();
                s.falloff = l.getFalloff();
                break;
            }
            case LightType::Area: {
                auto& l = (const AreaLight&)light;
                s.u = l.u;
                s.v = l.v;
                s.width = l.width;
                s.height = l.height;
                break;
            }
            case LightType::Point: {
                auto& l = (const PointLight&)light;
                s.radius = l.radius;
                break;
            }
        }
        return s;
    }
    
    void apply(Light& light) const {
        light.position = position;
        
        switch (light.type()) {
            case LightType::Directional: {
                auto& l = (DirectionalLight&)light;
                l.setDirection(direction);
                l.radius = radius;
                break;
            }
            case LightType::Spot: {
                auto& l = (SpotLight&)light;
                l.direction = direction;
                l.radius = radius;
                l.setAngleDegrees(angle);
                l.setFalloff(falloff);
                break;
            }
            case LightType::Area: {
                auto& l = (AreaLight&)light;
                l.u = u;
                l.v = v;
                l.width = width;
                l.height = height;
                break;
            }
            case LightType::Point: {
                auto& l = (PointLight&)light;
                l.radius = radius;
                break;
            }
        }
    }
};

class TransformLightCommand : public SceneCommand {
public:
    TransformLightCommand(std::shared_ptr<Light> light,
                         const LightState& old_state,
                         const LightState& new_state)
        : light_(light)
        , old_state_(old_state)
        , new_state_(new_state) {}
    
    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Transform; }
    std::string getDescription() const override { return "Transform Light"; }
    
private:
    std::shared_ptr<Light> light_;
    LightState old_state_;
    LightState new_state_;
};

class DeleteLightCommand : public SceneCommand {
public:
    DeleteLightCommand(std::shared_ptr<Light> light) : light_(light) {}
    
    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Heavy; }
    std::string getDescription() const override { return "Delete Light"; }
    
private:
    std::shared_ptr<Light> light_;
};

class AddLightCommand : public SceneCommand {
public:
    AddLightCommand(std::shared_ptr<Light> light) : light_(light) {}
    
    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Heavy; }
    std::string getDescription() const override { return "Add Light"; }
    
private:
    std::shared_ptr<Light> light_;
};

class PaintTextureCommand : public SceneCommand {
public:
    PaintTextureCommand(const std::string& object_name,
                        uint16_t material_id,
                        const std::shared_ptr<Texture>& texture,
                        std::vector<CompactVec4> before_pixels,
                        std::vector<CompactVec4> after_pixels);

    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override {
        return "Paint " + object_name_;
    }

private:
    std::string object_name_;
    uint16_t material_id_ = 0xFFFF;
    std::shared_ptr<Texture> texture_;
    std::vector<CompactVec4> before_pixels_;
    std::vector<CompactVec4> after_pixels_;
    bool region_mode_ = false;
    int width_ = 0;
    int height_ = 0;
    int region_x_ = 0;
    int region_y_ = 0;
    int region_w_ = 0;
    int region_h_ = 0;

    void applyPixels(UIContext& ctx, const std::vector<CompactVec4>& pixels);
};

// ============================================================================
// PAINT LAYER COMMAND - Undo/redo for layer-based painting
// ============================================================================
// Stores before/after pixels for a specific layer + channel, and also keeps
// the flat texture ref so it can recomposite after restoring layer data.
class PaintLayerCommand : public SceneCommand {
public:
    PaintLayerCommand(const std::string& object_name,
                      uint16_t material_id,
                      const std::string& layer_stack_key,
                      uint32_t layer_id,
                      Paint::PaintChannel channel,
                      std::vector<CompactVec4> before_pixels,
                      std::vector<CompactVec4> after_pixels);

    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override {
        return "Paint Layer " + object_name_;
    }

private:
    std::string object_name_;
    uint16_t material_id_ = 0xFFFF;
    std::string layer_stack_key_;
    uint32_t layer_id_ = 0;
    Paint::PaintChannel channel_;
    std::vector<CompactVec4> before_pixels_;
    std::vector<CompactVec4> after_pixels_;
    bool region_mode_ = false;
    bool before_empty_ = false;
    bool after_empty_ = false;
    int width_ = 0;
    int height_ = 0;
    int region_x_ = 0;
    int region_y_ = 0;
    int region_w_ = 0;
    int region_h_ = 0;

    void applyPixels(UIContext& ctx, const std::vector<CompactVec4>& pixels, bool empty_state);
};

struct TriangleUVSetState {
    std::shared_ptr<Triangle> triangle;
    size_t uv_set_index = 0;
    std::array<Vec2, 3> uvs{};
};

class UVProjectionCommand : public SceneCommand {
public:
    UVProjectionCommand(const std::string& object_name,
                        std::vector<TriangleUVSetState> before_states,
                        std::vector<TriangleUVSetState> after_states)
        : object_name_(object_name)
        , before_states_(std::move(before_states))
        , after_states_(std::move(after_states)) {}

    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override {
        return "Project UVs: " + object_name_;
    }

private:
    std::string object_name_;
    std::vector<TriangleUVSetState> before_states_;
    std::vector<TriangleUVSetState> after_states_;

    void applyStates(UIContext& ctx, const std::vector<TriangleUVSetState>& states);
};

struct MeshEditTriangleState {
    std::shared_ptr<Triangle> triangle;
    std::array<Vec3, 3> positions{};
};

class MeshEditCommand : public SceneCommand {
public:
    MeshEditCommand(const std::string& object_name,
                    std::vector<MeshEditTriangleState> before_states,
                    std::vector<MeshEditTriangleState> after_states)
        : object_name_(object_name)
        , before_states_(std::move(before_states))
        , after_states_(std::move(after_states)) {}

    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Transform; }
    std::string getDescription() const override {
        return "Edit Mesh " + object_name_;
    }

private:
    std::string object_name_;
    std::vector<MeshEditTriangleState> before_states_;
    std::vector<MeshEditTriangleState> after_states_;

    void applyStates(UIContext& ctx, const std::vector<MeshEditTriangleState>& states);
};

class ReplaceMeshGeometryCommand : public SceneCommand {
public:
    ReplaceMeshGeometryCommand(const std::string& object_name,
                              std::vector<std::shared_ptr<Triangle>> before_display_mesh,
                              std::vector<std::shared_ptr<Triangle>> after_display_mesh,
                              std::vector<std::shared_ptr<Triangle>> before_base_mesh,
                              std::vector<std::shared_ptr<Triangle>> after_base_mesh,
                              MeshModifiers::ModifierStack before_stack,
                              MeshModifiers::ModifierStack after_stack)
        : object_name_(object_name)
        , before_display_mesh_(std::move(before_display_mesh))
        , after_display_mesh_(std::move(after_display_mesh))
        , before_base_mesh_(std::move(before_base_mesh))
        , after_base_mesh_(std::move(after_base_mesh))
        , before_stack_(std::move(before_stack))
        , after_stack_(std::move(after_stack)) {}

    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Heavy; }
    std::string getDescription() const override { return "Replace Mesh Geometry " + object_name_; }

private:
    std::string object_name_;
    std::vector<std::shared_ptr<Triangle>> before_display_mesh_;
    std::vector<std::shared_ptr<Triangle>> after_display_mesh_;
    std::vector<std::shared_ptr<Triangle>> before_base_mesh_;
    std::vector<std::shared_ptr<Triangle>> after_base_mesh_;
    MeshModifiers::ModifierStack before_stack_;
    MeshModifiers::ModifierStack after_stack_;

    void applyMesh(UIContext& ctx,
                   const std::vector<std::shared_ptr<Triangle>>& display_mesh,
                   const std::vector<std::shared_ptr<Triangle>>& base_mesh,
                   const MeshModifiers::ModifierStack& stack);
};

class CompositeSceneCommand : public SceneCommand {
public:
    explicit CompositeSceneCommand(std::string description)
        : description_(std::move(description)) {}

    void add(std::unique_ptr<SceneCommand> command) {
        if (command) {
            commands_.push_back(std::move(command));
        }
    }

    bool empty() const { return commands_.empty(); }

    void execute(UIContext& ctx) override;
    void undo(UIContext& ctx) override;
    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override { return description_; }

private:
    std::string description_;
    std::vector<std::unique_ptr<SceneCommand>> commands_;
};

