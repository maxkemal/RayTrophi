#pragma once
#include "Paint/HiddenGameObjects.h"
#include "Matrix4x4.h"
#include <map>
#include <set>
#include <cstdint>
#include <vector>

namespace Paint::HiddenDetail {
using json = nlohmann::json;
struct Box {
    Vec3 low, high;
};
struct Entry {
    std::string object;
    std::string visibility = "paint";
    bool randomize = true;
    bool blocks_player = false;
    bool blocks_balloon = false;
    float yaw_min = -15, yaw_max = 15;
    Matrix4x4 base = Matrix4x4::identity();
    std::string collider = "none";
    int sdf_resolution = 1;
};
struct Plane { Vec3 point, inward; };
struct State {
    SceneData* scene = nullptr;
    std::map<std::string, Entry> entries;
    bool manual_room = false;
    std::string placement = "walls_floor";
    float wall_band = 0.65f;
    float camera_clearance = 3.0f;
    Box manual;
    std::uint64_t round = 0;
    std::string error;
};
State& state();
void require(bool ok, const std::string& error);
Vec3 vector3(const json& value);
json vectorJson(const Vec3& value);
json entryJson(const Entry& entry);
bool roomNode(const std::string& name);
void discover(SceneData& scene);
Box objectBox(const std::string& name);
bool overlaps(const Box& a, const Box& b, float margin = 0.02f);
bool roomGeometry(SceneData& scene, Box& room, std::vector<Plane>& planes,
                  std::set<std::string>* boundary_slabs = nullptr);
json roomStatus(SceneData& scene);
bool planHiddenPlacements(SceneData& scene,
                          std::vector<std::pair<std::string, Matrix4x4>>& placements,
                          std::string& error);
void applyVisibility(SceneData& scene, const Entry& entry);
void waitHiddenResetWork(SceneData& scene);
bool moveHiddenObject(SceneData& scene, const std::string& name,
                      const Matrix4x4& matrix, std::string& error);
} // namespace Paint::HiddenDetail
