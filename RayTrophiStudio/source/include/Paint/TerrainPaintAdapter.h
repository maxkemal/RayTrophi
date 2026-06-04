#pragma once

#include "Paint/IPaintSurfaceAdapter.h"

struct TerrainObject;

namespace Paint {

class TerrainPaintAdapter final : public IPaintSurfaceAdapter {
public:
    explicit TerrainPaintAdapter(TerrainObject* terrain);

    SurfaceType getSurfaceType() const override;
    PaintSurfaceTarget getTarget() const override;
    PaintAdapterCapabilities getCapabilities() const override;
    bool isValid() const override;

    int getLayerCount() const override;
    std::string getLayerName(int index) const override;

    bool beginStroke(const PaintStrokeContext& ctx) override;
    bool applyDab(const Vec3& world_hit_point, const BrushSettings& brush, const PaintStrokeContext& ctx) override;
    void endStroke() override;

    TerrainObject* getTerrain() const { return terrain_; }

private:
    TerrainObject* terrain_ = nullptr;
    bool stroke_active_ = false;
};

} // namespace Paint
