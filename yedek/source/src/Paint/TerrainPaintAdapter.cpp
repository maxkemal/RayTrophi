#include "Paint/TerrainPaintAdapter.h"
#include "TerrainSystem.h"
#include "TerrainManager.h"
#include "Material.h"
#include <array>

namespace Paint {

namespace {
const std::array<const char*, 4> kFallbackLayerNames = {
    "Grass",
    "Rock",
    "Snow",
    "Flow"
};
}

TerrainPaintAdapter::TerrainPaintAdapter(TerrainObject* terrain) : terrain_(terrain) {}

SurfaceType TerrainPaintAdapter::getSurfaceType() const {
    return SurfaceType::Terrain;
}

PaintSurfaceTarget TerrainPaintAdapter::getTarget() const {
    PaintSurfaceTarget target;
    target.type = SurfaceType::Terrain;
    if (terrain_) {
        target.object_id = terrain_->id;
        target.display_name = terrain_->name;
    }
    return target;
}

PaintAdapterCapabilities TerrainPaintAdapter::getCapabilities() const {
    PaintAdapterCapabilities caps;
    caps.supports_layers = true;
    caps.supports_terrain_channels = true;
    return caps;
}

bool TerrainPaintAdapter::isValid() const {
    return terrain_ != nullptr && terrain_->splatMap != nullptr;
}

int TerrainPaintAdapter::getLayerCount() const {
    return 4;
}

std::string TerrainPaintAdapter::getLayerName(int index) const {
    if (index < 0 || index >= 4) {
        return "Layer";
    }

    if (terrain_ && index < static_cast<int>(terrain_->layers.size()) && terrain_->layers[index]) {
        const std::string& material_name = terrain_->layers[index]->materialName;
        if (!material_name.empty()) {
            return material_name;
        }
    }

    return std::string(kFallbackLayerNames[index]);
}

bool TerrainPaintAdapter::beginStroke(const PaintStrokeContext& ctx) {
    stroke_active_ = isValid() && ctx.layer_index >= 0 && ctx.layer_index < getLayerCount();
    return stroke_active_;
}

bool TerrainPaintAdapter::applyDab(const Vec3& world_hit_point, const BrushSettings& brush, const PaintStrokeContext& ctx) {
    if (!stroke_active_ || !isValid()) {
        return false;
    }

    TerrainManager::getInstance().paintSplatMap(
        terrain_,
        world_hit_point,
        ctx.layer_index,
        brush.radius,
        brush.strength * brush.flow,
        ctx.dt);

    return true;
}

void TerrainPaintAdapter::endStroke() {
    stroke_active_ = false;
}

} // namespace Paint
