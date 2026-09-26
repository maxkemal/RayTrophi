#pragma once

#include "Matrix4x4.h"
#include "Vec3.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace RayTrophiSim {

constexpr uint32_t kMaxKinematicProxySets = 256;
constexpr uint32_t kMaxKinematicProxiesPerSet = 256;

enum class KinematicProxyShape : uint8_t {
    Sphere = 0,
    Capsule = 1,
    Box = 2
};

enum KinematicColliderConsumer : uint32_t {
    KinematicConsumerFluid = 1u << 0u,
    KinematicConsumerGas = 1u << 1u,
    KinematicConsumerGranular = 1u << 2u,
    KinematicConsumerParticles = 1u << 3u,
    KinematicConsumerMaterialState = 1u << 4u,
    KinematicConsumerAll = (1u << 5u) - 1u
};

struct KinematicProxyDesc {
    uint64_t id = 0;
    std::string name;
    std::string bone;
    KinematicProxyShape shape = KinematicProxyShape::Capsule;
    bool enabled = true;

    // Bone-local authoring. Capsules use local_axis directly so auto-fit does
    // not need an unstable Euler decomposition merely to align a limb.
    Vec3 local_position = Vec3(0.0f);
    Vec3 local_rotation_degrees = Vec3(0.0f);
    Vec3 local_axis = Vec3(0.0f, 1.0f, 0.0f);
    float radius = 0.08f;
    float half_length = 0.20f;
    Vec3 half_extents = Vec3(0.10f, 0.08f, 0.20f);
};

struct KinematicProxySet {
    uint64_t id = 0;
    std::string name;
    std::string target_character;
    // Reserved for the scene-wide stable node identity planned in K1. The
    // current rig API resolves by character name and rejects a non-empty id
    // rather than pretending it was honored.
    std::string target_node_id;
    bool enabled = true;
    uint32_t consumer_mask = KinematicConsumerAll;
    float friction = 0.45f;
    float restitution = 0.0f;
    float thickness = 0.0f;
    uint64_t revision = 1;
    std::vector<KinematicProxyDesc> proxies;
};

struct KinematicJointPose {
    std::string name;
    std::string parent;
    Matrix4x4 world = Matrix4x4::identity();
    bool weighted = false;
};

struct KinematicAutoFitOptions {
    bool replace_existing = true;
    bool weighted_bones_only = true;
    float radius_fraction = 0.18f;
    float minimum_radius = 0.025f;
    float maximum_radius = 0.20f;
    float minimum_bone_length = 0.04f;
    uint32_t maximum_proxies = 64;
};

struct KinematicProxySample {
    uint64_t set_id = 0;
    uint64_t proxy_id = 0;
    std::string set_name;
    std::string proxy_name;
    std::string target_character;
    std::string bone;
    KinematicProxyShape shape = KinematicProxyShape::Capsule;
    bool resolved = false;
    std::string unresolved_reason;

    Matrix4x4 world_transform = Matrix4x4::identity();
    Vec3 center = Vec3(0.0f);
    Vec3 capsule_start = Vec3(0.0f);
    Vec3 capsule_end = Vec3(0.0f);
    float radius = 0.0f;
    Vec3 half_extents = Vec3(0.0f);
    Vec3 linear_velocity = Vec3(0.0f);
    Vec3 angular_velocity = Vec3(0.0f);
    bool velocity_valid = false;
};

using KinematicBoneResolver =
    std::function<bool(const std::string& character,
                       const std::string& bone,
                       Matrix4x4& world,
                       std::string& reason)>;

class KinematicColliderRegistry {
public:
    const std::vector<KinematicProxySet>& sets() const { return sets_; }
    std::vector<KinematicProxySet>& sets() { return sets_; }

    const KinematicProxySet* findSet(uint64_t id) const;
    KinematicProxySet* findSet(uint64_t id);
    const KinematicProxySet* findSet(const std::string& name) const;
    KinematicProxySet* findSet(const std::string& name);

    bool createSet(const KinematicProxySet& requested,
                   KinematicProxySet& created,
                   std::string& error);
    bool updateSet(uint64_t id,
                   const KinematicProxySet& requested,
                   std::string& error);
    bool removeSet(uint64_t id, std::string& error);

    bool setProxy(uint64_t set_id,
                  const KinematicProxyDesc& requested,
                  KinematicProxyDesc& stored,
                  std::string& error);
    bool removeProxy(uint64_t set_id, uint64_t proxy_id, std::string& error);

    bool autoFit(uint64_t set_id,
                 const std::vector<KinematicJointPose>& joints,
                 const KinematicAutoFitOptions& options,
                 uint32_t& created_count,
                 std::string& error);

    bool sampleSet(uint64_t set_id,
                   float dt,
                   bool discontinuity,
                   const KinematicBoneResolver& resolver,
                   std::vector<KinematicProxySample>& samples,
                   std::string& error);
    void resetMotionHistory();
    void resetMotionHistory(uint64_t set_id);
    bool restoreSets(const std::vector<KinematicProxySet>& sets,
                     std::string& error);
    void clear();

private:
    struct MotionHistory {
        Matrix4x4 world = Matrix4x4::identity();
        Vec3 center = Vec3(0.0f);
        bool valid = false;
    };

    std::vector<KinematicProxySet> sets_;
    std::unordered_map<uint64_t, MotionHistory> motion_history_;
    uint64_t next_set_id_ = 1;
    uint64_t next_proxy_id_ = 1;

    bool validateSet(const KinematicProxySet& set,
                     uint64_t ignore_id,
                     std::string& error) const;
    static bool validateProxy(const KinematicProxyDesc& proxy,
                              std::string& error);
};

const char* kinematicProxyShapeName(KinematicProxyShape shape);
bool parseKinematicProxyShape(const std::string& text,
                              KinematicProxyShape& shape);

} // namespace RayTrophiSim
