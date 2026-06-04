#include "SimulationWorld.h"

#include "ForceField.h"

namespace RayTrophiSim {

uint32_t toSimulationSystemMask(SimulationSystemKind kind) {
    return static_cast<uint32_t>(kind);
}

void SimulationForceFieldSnapshot::rebuild(const Physics::ForceFieldManager* manager, float frame) {
    active_fields_.clear();
    packed_fields_.clear();

    if (!manager) {
        ++version_;
        return;
    }

    const auto& fields = manager->getForceFields();
    active_fields_.reserve(fields.size());
    packed_fields_.reserve(fields.size());

    for (const auto& field : fields) {
        if (!field || !field->isActiveAtFrame(frame)) {
            continue;
        }

        const uint32_t affect_mask = affectMaskForField(*field);
        active_fields_.push_back(field.get());
        packed_fields_.push_back(packField(*field, affect_mask));
    }

    ++version_;
}

void SimulationForceFieldSnapshot::clear() {
    active_fields_.clear();
    packed_fields_.clear();
    ++version_;
}

Vec3 SimulationForceFieldSnapshot::evaluateAt(const Vec3& world_pos,
                                              float time,
                                              const Vec3& velocity,
                                              SimulationSystemKind kind) const {
    const uint32_t system_mask = toSimulationSystemMask(kind);
    Vec3 total_force(0.0f, 0.0f, 0.0f);

    for (size_t i = 0; i < active_fields_.size(); ++i) {
        const Physics::ForceField* field = active_fields_[i];
        if (!field) {
            continue;
        }

        const uint32_t affect_mask = i < packed_fields_.size() ? packed_fields_[i].affect_mask : 0u;
        if ((affect_mask & system_mask) == 0u) {
            continue;
        }

        total_force = total_force + field->evaluate(world_pos, time, velocity);
    }

    return total_force;
}

uint32_t SimulationForceFieldSnapshot::affectMaskForField(const Physics::ForceField& field) {
    uint32_t mask = 0u;

    if (field.affects_gas) {
        mask |= toSimulationSystemMask(SimulationSystemKind::Gas);
    }
    if (field.affects_particles) {
        mask |= toSimulationSystemMask(SimulationSystemKind::Particle);
    }
    if (field.affects_cloth) {
        mask |= toSimulationSystemMask(SimulationSystemKind::Cloth);
    }
    if (field.affects_rigidbody) {
        mask |= toSimulationSystemMask(SimulationSystemKind::RigidBody);
    }
    if (field.affects_fluid) {
        mask |= toSimulationSystemMask(SimulationSystemKind::Fluid);
    }

    // These systems historically sampled all enabled force fields.
    mask |= toSimulationSystemMask(SimulationSystemKind::Hair);
    mask |= toSimulationSystemMask(SimulationSystemKind::WetSurface);
    mask |= toSimulationSystemMask(SimulationSystemKind::Terrain);

    return mask;
}

PackedForceField SimulationForceFieldSnapshot::packField(const Physics::ForceField& field, uint32_t affect_mask) {
    PackedForceField packed;

    packed.id = field.id;
    packed.type = static_cast<int>(field.type);
    packed.shape = static_cast<int>(field.shape);
    packed.falloff_type = static_cast<int>(field.falloff_type);
    packed.affect_mask = affect_mask;
    packed.enabled = field.enabled ? 1 : 0;

    packed.position_x = field.position.x;
    packed.position_y = field.position.y;
    packed.position_z = field.position.z;
    packed.rotation_x = field.rotation.x;
    packed.rotation_y = field.rotation.y;
    packed.rotation_z = field.rotation.z;
    packed.scale_x = field.scale.x;
    packed.scale_y = field.scale.y;
    packed.scale_z = field.scale.z;

    packed.strength = field.strength;
    packed.direction_x = field.direction.x;
    packed.direction_y = field.direction.y;
    packed.direction_z = field.direction.z;
    packed.falloff_radius = field.falloff_radius;
    packed.inner_radius = field.inner_radius;

    packed.axis_x = field.axis.x;
    packed.axis_y = field.axis.y;
    packed.axis_z = field.axis.z;
    packed.inward_force = field.inward_force;
    packed.upward_force = field.upward_force;
    packed.linear_drag = field.linear_drag;
    packed.quadratic_drag = field.quadratic_drag;

    packed.noise_frequency = field.noise.frequency;
    packed.noise_lacunarity = field.noise.lacunarity;
    packed.noise_persistence = field.noise.persistence;
    packed.noise_amplitude = field.noise.amplitude;
    packed.noise_speed = field.noise.speed;
    packed.noise_octaves = field.noise.octaves;
    packed.noise_seed = field.noise.seed;
    packed.use_noise = field.use_noise ? 1 : 0;

    return packed;
}

void SimulationWorld::setForceFieldManager(const Physics::ForceFieldManager* manager) {
    force_fields_ = manager;
}

const Physics::ForceFieldManager* SimulationWorld::getForceFieldManager() const {
    return force_fields_;
}

void SimulationWorld::setBackend(SimulationBackend backend) {
    backend_ = backend;
}

SimulationBackend SimulationWorld::getBackend() const {
    return backend_;
}

void SimulationWorld::setMode(SimulationMode mode) {
    mode_ = mode;
}

SimulationMode SimulationWorld::getMode() const {
    return mode_;
}

void SimulationWorld::setFixedTimestep(float fixed_dt) {
    if (fixed_dt > 0.0f) {
        fixed_dt_ = fixed_dt;
    }
}

float SimulationWorld::getFixedTimestep() const {
    return fixed_dt_;
}

void SimulationWorld::setMaxSubsteps(int max_substeps) {
    max_substeps_ = std::max(1, max_substeps);
}

int SimulationWorld::getMaxSubsteps() const {
    return max_substeps_;
}

void SimulationWorld::resetTime(float time_seconds, int frame) {
    accumulator_ = 0.0f;
    time_seconds_ = std::max(0.0f, time_seconds);
    frame_ = std::max(0, frame);
    stats_.substeps_last_advance = 0;
    stats_.simulated_time_seconds = time_seconds_;
}

void SimulationWorld::clearSystems() {
    systems_.clear();
    force_snapshot_.clear();
    if (force_compute_buffer_.buffer.valid()) {
        compute_context_.destroyBuffer(force_compute_buffer_.buffer);
        force_compute_buffer_ = {};
    }
    systems_dirty_ = false;
    stats_.registered_systems = 0;
    stats_.active_systems = 0;
    stats_.active_force_fields = 0;
    stats_.packed_force_fields = 0;
    stats_.force_compute_buffer_bytes = 0;
    stats_.force_snapshot_version = force_snapshot_.version();
}

void SimulationWorld::addSystem(std::shared_ptr<ISimulationSystem> system) {
    if (!system) {
        return;
    }

    const auto exists = std::find_if(systems_.begin(), systems_.end(),
        [&](const std::shared_ptr<ISimulationSystem>& existing) {
            return existing.get() == system.get();
        });
    if (exists != systems_.end()) {
        return;
    }

    systems_.push_back(std::move(system));
    systems_dirty_ = true;
    stats_.registered_systems = static_cast<int>(systems_.size());
}

bool SimulationWorld::removeSystem(const ISimulationSystem* system) {
    const auto before = systems_.size();
    systems_.erase(
        std::remove_if(systems_.begin(), systems_.end(),
            [&](const std::shared_ptr<ISimulationSystem>& existing) {
                return existing.get() == system;
            }),
        systems_.end());

    const bool removed = systems_.size() != before;
    if (removed) {
        stats_.registered_systems = static_cast<int>(systems_.size());
    }
    return removed;
}

void SimulationWorld::advance(float dt) {
    stats_.substeps_last_advance = 0;
    stats_.active_systems = 0;

    if (mode_ == SimulationMode::Paused || dt <= 0.0f || fixed_dt_ <= 0.0f) {
        return;
    }

    if (systems_dirty_) {
        sortSystems();
    }

    rebuildForceFieldSnapshot();

    accumulator_ += dt;
    int substeps = 0;
    while (accumulator_ >= fixed_dt_ && substeps < max_substeps_) {
        executeStep(fixed_dt_, substeps, max_substeps_);
        accumulator_ -= fixed_dt_;
        ++substeps;
    }

    if (substeps == max_substeps_ && accumulator_ >= fixed_dt_) {
        accumulator_ = 0.0f;
    }

    stats_.substeps_last_advance = substeps;
    stats_.simulated_time_seconds = time_seconds_;
}

void SimulationWorld::stepOnce(float dt) {
    stats_.substeps_last_advance = 0;
    stats_.active_systems = 0;

    if (mode_ == SimulationMode::Paused || dt <= 0.0f) {
        return;
    }

    if (systems_dirty_) {
        sortSystems();
    }

    rebuildForceFieldSnapshot();
    executeStep(dt, 0, 1);

    stats_.substeps_last_advance = 1;
    stats_.simulated_time_seconds = time_seconds_;
}

void SimulationWorld::refreshForceFieldSnapshot() {
    rebuildForceFieldSnapshot();
}

SimulationComputeContext& SimulationWorld::compute() {
    return compute_context_;
}

const SimulationComputeContext& SimulationWorld::compute() const {
    return compute_context_;
}

SimulationContext SimulationWorld::makeContext(float dt, int substep_index, int substep_count) {
    SimulationContext context;
    context.dt = dt;
    context.fixed_dt = fixed_dt_;
    context.time_seconds = time_seconds_;
    context.frame = frame_;
    context.substep_index = substep_index;
    context.substep_count = substep_count;
    context.backend = backend_;
    context.mode = mode_;
    context.force_fields = force_fields_;
    context.force_snapshot = &force_snapshot_;
    context.force_compute_buffer = &force_compute_buffer_;
    context.compute = &compute_context_;
    return context;
}

const SimulationWorldStats& SimulationWorld::getStats() const {
    return stats_;
}

const SimulationForceFieldSnapshot& SimulationWorld::getForceFieldSnapshot() const {
    return force_snapshot_;
}

const SimulationForceFieldComputeBuffer& SimulationWorld::getForceFieldComputeBuffer() const {
    return force_compute_buffer_;
}

void SimulationWorld::sortSystems() {
    std::stable_sort(systems_.begin(), systems_.end(),
        [](const std::shared_ptr<ISimulationSystem>& lhs,
           const std::shared_ptr<ISimulationSystem>& rhs) {
            if (!lhs) {
                return false;
            }
            if (!rhs) {
                return true;
            }
            return lhs->order() < rhs->order();
        });
    systems_dirty_ = false;
}

void SimulationWorld::rebuildForceFieldSnapshot() {
    force_snapshot_.rebuild(force_fields_, static_cast<float>(frame_));
    uploadForceFieldSnapshotToCompute();
    stats_.active_force_fields = static_cast<int>(force_snapshot_.activeFields().size());
    stats_.packed_force_fields = static_cast<int>(force_snapshot_.packedFields().size());
    stats_.force_compute_buffer_bytes = force_compute_buffer_.size_bytes;
    stats_.force_snapshot_version = force_snapshot_.version();
}

void SimulationWorld::uploadForceFieldSnapshotToCompute() {
    const auto& packed = force_snapshot_.packedFields();
    const std::size_t required_bytes = packed.size() * sizeof(PackedForceField);

    if (required_bytes == 0) {
        if (force_compute_buffer_.buffer.valid()) {
            compute_context_.destroyBuffer(force_compute_buffer_.buffer);
        }
        force_compute_buffer_ = {};
        force_compute_buffer_.source_version = force_snapshot_.version();
        return;
    }

    if (!force_compute_buffer_.buffer.valid() ||
        force_compute_buffer_.buffer.backend != compute_context_.backendType()) {
        if (force_compute_buffer_.buffer.valid()) {
            compute_context_.destroyBuffer(force_compute_buffer_.buffer);
        }

        ComputeBufferDesc desc;
        desc.debug_name = "SimulationForceFields";
        desc.size_bytes = required_bytes;
        desc.usage = ComputeBufferUsage::Storage |
                     ComputeBufferUsage::Upload |
                     ComputeBufferUsage::ReadOnly;
        force_compute_buffer_.buffer = compute_context_.createBuffer(desc);
    } else if (compute_context_.getBufferSize(force_compute_buffer_.buffer) != required_bytes) {
        if (!compute_context_.resizeBuffer(force_compute_buffer_.buffer, required_bytes)) {
            compute_context_.destroyBuffer(force_compute_buffer_.buffer);
            ComputeBufferDesc desc;
            desc.debug_name = "SimulationForceFields";
            desc.size_bytes = required_bytes;
            desc.usage = ComputeBufferUsage::Storage |
                         ComputeBufferUsage::Upload |
                         ComputeBufferUsage::ReadOnly;
            force_compute_buffer_.buffer = compute_context_.createBuffer(desc);
        }
    }

    force_compute_buffer_.count = packed.size();
    force_compute_buffer_.stride_bytes = sizeof(PackedForceField);
    force_compute_buffer_.size_bytes = required_bytes;
    force_compute_buffer_.source_version = force_snapshot_.version();

    if (!force_compute_buffer_.buffer.valid() ||
        !compute_context_.uploadBuffer(force_compute_buffer_.buffer, packed.data(), required_bytes)) {
        force_compute_buffer_ = {};
        force_compute_buffer_.source_version = force_snapshot_.version();
    }
}

void SimulationWorld::executeStep(float dt, int substep_index, int substep_count) {
    compute_context_.beginFrame(static_cast<uint64_t>(frame_));
    SimulationContext context = makeContext(dt, substep_index, substep_count);

    for (const auto& system : systems_) {
        if (system && system->enabled()) {
            system->prepare(context);
        }
    }

    int active_this_step = 0;
    for (const auto& system : systems_) {
        if (system && system->enabled()) {
            system->step(context);
            ++active_this_step;
        }
    }

    for (const auto& system : systems_) {
        if (system && system->enabled()) {
            system->finalize(context);
        }
    }

    compute_context_.endFrame();
    stats_.active_systems = std::max(stats_.active_systems, active_this_step);
    time_seconds_ += dt;
    ++frame_;
}

} // namespace RayTrophiSim
