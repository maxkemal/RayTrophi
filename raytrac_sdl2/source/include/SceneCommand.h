#pragma once

#include <memory>
#include <vector>
#include <string>
#include "Hittable.h"
#include "Triangle.h"

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
