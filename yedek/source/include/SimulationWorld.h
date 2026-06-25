#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Vec3.h"
#include "SimulationCompute.h"

namespace Physics {
class ForceField;
class ForceFieldManager;
}

namespace RayTrophiSim {

enum class SimulationBackend {
    CPU,
    CUDA
};

enum class SimulationMode {
    Realtime,
    Baked,
    Paused
};

enum class SimulationSystemKind : uint32_t {
    Gas = 1u << 0,
    Hair = 1u << 1,
    WetSurface = 1u << 2,
    Particle = 1u << 3,
    Cloth = 1u << 4,
    RigidBody = 1u << 5,
    Terrain = 1u << 6,
    Fluid = 1u << 7,
    Custom = 1u << 31
};

uint32_t toSimulationSystemMask(SimulationSystemKind kind);

struct PackedForceField {
    int id = -1;
    int type = 0;
    int shape = 0;
    int falloff_type = 0;
    uint32_t affect_mask = 0;
    int enabled = 0;

    float position_x = 0.0f;
    float position_y = 0.0f;
    float position_z = 0.0f;
    float rotation_x = 0.0f;
    float rotation_y = 0.0f;
    float rotation_z = 0.0f;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    float scale_z = 1.0f;

    float strength = 1.0f;
    float direction_x = 0.0f;
    float direction_y = -1.0f;
    float direction_z = 0.0f;
    float falloff_radius = 5.0f;
    float inner_radius = 0.0f;

    float axis_x = 0.0f;
    float axis_y = 1.0f;
    float axis_z = 0.0f;
    float inward_force = 0.0f;
    float upward_force = 0.0f;
    float linear_drag = 0.1f;
    float quadratic_drag = 0.0f;

    float noise_frequency = 0.5f;
    float noise_lacunarity = 2.0f;
    float noise_persistence = 0.5f;
    float noise_amplitude = 1.0f;
    float noise_speed = 0.1f;
    int noise_octaves = 4;
    int noise_seed = 42;
    int use_noise = 0;
};

class SimulationForceFieldSnapshot {
public:
    void rebuild(const Physics::ForceFieldManager* manager, float frame);
    void clear();

    Vec3 evaluateAt(const Vec3& world_pos,
                    float time,
                    const Vec3& velocity,
                    SimulationSystemKind kind) const;

    const std::vector<const Physics::ForceField*>& activeFields() const { return active_fields_; }
    const std::vector<PackedForceField>& packedFields() const { return packed_fields_; }
    uint64_t version() const { return version_; }
    bool empty() const { return active_fields_.empty(); }

private:
    static uint32_t affectMaskForField(const Physics::ForceField& field);
    static PackedForceField packField(const Physics::ForceField& field, uint32_t affect_mask);

    std::vector<const Physics::ForceField*> active_fields_;
    std::vector<PackedForceField> packed_fields_;
    uint64_t version_ = 0;
};

struct SimulationForceFieldComputeBuffer {
    ComputeBufferHandle buffer;
    std::size_t count = 0;
    std::size_t stride_bytes = sizeof(PackedForceField);
    std::size_t size_bytes = 0;
    uint64_t source_version = 0;

    bool valid() const { return buffer.valid() && count > 0 && size_bytes > 0; }
};

struct SimulationContext {
    float dt = 0.0f;
    float fixed_dt = 1.0f / 60.0f;
    float time_seconds = 0.0f;
    int frame = 0;
    int substep_index = 0;
    int substep_count = 1;
    SimulationBackend backend = SimulationBackend::CUDA;
    SimulationMode mode = SimulationMode::Realtime;
    const Physics::ForceFieldManager* force_fields = nullptr;
    const SimulationForceFieldSnapshot* force_snapshot = nullptr;
    const SimulationForceFieldComputeBuffer* force_compute_buffer = nullptr;
    SimulationComputeContext* compute = nullptr;
};

class ISimulationSystem {
public:
    virtual ~ISimulationSystem() = default;

    virtual const char* name() const = 0;
    virtual SimulationSystemKind kind() const = 0;
    virtual int order() const { return 0; }
    virtual bool enabled() const { return true; }

    virtual void prepare(const SimulationContext& context) { (void)context; }
    virtual void step(const SimulationContext& context) = 0;
    virtual void finalize(const SimulationContext& context) { (void)context; }
};

struct SimulationWorldStats {
    int registered_systems = 0;
    int active_systems = 0;
    int active_force_fields = 0;
    int packed_force_fields = 0;
    std::size_t force_compute_buffer_bytes = 0;
    int substeps_last_advance = 0;
    float simulated_time_seconds = 0.0f;
    uint64_t force_snapshot_version = 0;
};

class SimulationWorld {
public:
    void setForceFieldManager(const Physics::ForceFieldManager* manager);
    const Physics::ForceFieldManager* getForceFieldManager() const;

    void setBackend(SimulationBackend backend);
    SimulationBackend getBackend() const;

    void setMode(SimulationMode mode);
    SimulationMode getMode() const;

    void setFixedTimestep(float fixed_dt);
    float getFixedTimestep() const;

    void setMaxSubsteps(int max_substeps);
    int getMaxSubsteps() const;

    void resetTime(float time_seconds = 0.0f, int frame = 0);
    void clearSystems();

    void addSystem(std::shared_ptr<ISimulationSystem> system);
    bool removeSystem(const ISimulationSystem* system);

    void advance(float dt);
    void stepOnce(float dt);
    void refreshForceFieldSnapshot();

    SimulationComputeContext& compute();
    const SimulationComputeContext& compute() const;

    SimulationContext makeContext(float dt, int substep_index, int substep_count);
    const SimulationWorldStats& getStats() const;
    const SimulationForceFieldSnapshot& getForceFieldSnapshot() const;
    const SimulationForceFieldComputeBuffer& getForceFieldComputeBuffer() const;

private:
    void sortSystems();
    void rebuildForceFieldSnapshot();
    void uploadForceFieldSnapshotToCompute();
    void executeStep(float dt, int substep_index, int substep_count);

    const Physics::ForceFieldManager* force_fields_ = nullptr;
    SimulationForceFieldSnapshot force_snapshot_;
    SimulationForceFieldComputeBuffer force_compute_buffer_;
    SimulationComputeContext compute_context_;
    std::vector<std::shared_ptr<ISimulationSystem>> systems_;
    SimulationBackend backend_ = SimulationBackend::CUDA;
    SimulationMode mode_ = SimulationMode::Realtime;
    float fixed_dt_ = 1.0f / 60.0f;
    float accumulator_ = 0.0f;
    float time_seconds_ = 0.0f;
    int frame_ = 0;
    int max_substeps_ = 8;
    bool systems_dirty_ = false;
    SimulationWorldStats stats_;
};

} // namespace RayTrophiSim
