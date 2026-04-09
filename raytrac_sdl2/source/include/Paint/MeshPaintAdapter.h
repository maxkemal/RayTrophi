#pragma once

#include "Paint/IPaintSurfaceAdapter.h"
#include "Paint/PaintTextureSet.h"
#include "Paint/PaintLayerStack.h"

struct SceneData;
class Triangle;

namespace Paint {

class MeshPaintAdapter final : public IPaintSurfaceAdapter {
public:
    MeshPaintAdapter(SceneData* scene, const std::shared_ptr<Triangle>& triangle);

    SurfaceType getSurfaceType() const override;
    PaintSurfaceTarget getTarget() const override;
    PaintAdapterCapabilities getCapabilities() const override;
    bool isValid() const override;

    int getLayerCount() const override;
    std::string getLayerName(int index) const override;

    bool beginStroke(const PaintStrokeContext& ctx) override;
    bool applyDab(const Vec3& world_hit_point, const BrushSettings& brush, const PaintStrokeContext& ctx) override;
    void endStroke() override;

    std::shared_ptr<Triangle> getTriangle() const { return triangle_; }
    uint16_t getMaterialID() const;
    std::string getNodeName() const;
    std::string getMaterialName() const;

    PaintTextureSet* getTextureSet() const;
    PaintTextureSet& ensureTextureSet(int resolution);
    std::shared_ptr<Texture> getChannelSourceTexture(PaintChannel channel) const;
    bool assignTextureToChannel(PaintChannel channel);
    bool createTextureSet();
    bool paintAtUV(PaintChannel channel, const Vec2& uv, const BrushSettings& brush, float dt);
    bool cloneAtUV(PaintChannel channel, const Vec2& dst_uv, const Vec2& src_uv, const BrushSettings& brush, float dt);
    bool fillChannel(PaintChannel channel, const BrushSettings& brush);
    bool generateNormalFromHeight(float strength);
    bool updateNormalFromHeightArea(const Vec2& center_uv, float radius_px, float strength);
    bool bakeHeightIntoNormal(float strength, bool clear_height_mask);
    bool restoreOriginalMaterialTextures();
    bool resizeTextureSet(int resolution);
    void bindTextureSetToMaterial();

    // -------- Layer Stack --------
    PaintLayerStack*       getLayerStack();
    const PaintLayerStack* getLayerStack() const;
    PaintLayerStack&       ensureLayerStack();

    // Paint onto a specific layer (by index) instead of the flat texture.
    // Supports Paint, Erase, Soften tools.  Returns dirty rect in pixel coords.
    PaintDirtyRect paintLayerAtUV(int layer_index, PaintChannel channel, const Vec2& uv,
                                  const BrushSettings& brush, float dt);

    // Clone from src_uv to dst_uv on a specific layer.  Returns dirty rect.
    PaintDirtyRect cloneLayerAtUV(int layer_index, PaintChannel channel,
                                  const Vec2& dst_uv, const Vec2& src_uv,
                                  const BrushSettings& brush, float dt);

    // Composite all layers and push result into PaintTextureSet + GPU.
    void compositeAndUpload();

    // Composite only the specified channels (cheaper than full compositeAndUpload).
    void compositeAndUploadChannels(const PaintChannel* channels, int count);

    // Composite only the dirty region of the specified channels (fastest path).
    void compositeAndUploadRegion(const PaintChannel* channels, int count,
                                  const PaintDirtyRect& dirty);

private:
    SceneData* scene_ = nullptr;
    std::shared_ptr<Triangle> triangle_;
};

} // namespace Paint
