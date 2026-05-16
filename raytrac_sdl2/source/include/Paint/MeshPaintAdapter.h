#pragma once

#include <vector>

#include "Paint/IPaintSurfaceAdapter.h"
#include "Paint/PaintTextureSet.h"
#include "Paint/PaintLayerStack.h"

struct SceneData;
class Triangle;
struct WeatherParams;

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
    bool fillChannel(PaintChannel channel, const BrushSettings& brush, int layer_index = -1);

    void releaseLayerStackFromScene() override;
    bool generateNormalFromHeight(float strength);
    bool updateNormalFromHeightArea(const Vec2& center_uv, float radius_px, float strength);
    bool updateNormalFromHeightRegion(const PaintDirtyRect& dirty, float strength);
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
    // `aspect_u` / `aspect_v` stretch the brush footprint in UV pixel space so
    // the stamp remains roughly circular in world space when the UV mapping is
    // anisotropic (e.g. resized/stretched meshes). Defaults = 1 = no correction.
    PaintDirtyRect paintLayerAtUV(int layer_index, PaintChannel channel, const Vec2& uv,
                                  const BrushSettings& brush, float dt,
                                  float aspect_u = 1.0f, float aspect_v = 1.0f);

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

    void noteWetDab(int layer_index, const Vec2& uv, const BrushSettings& brush,
                    float dt, float deposit_ratio);
    bool noteWeatherExposure(int layer_index, const WeatherParams& weather, float dt);
    float sampleWetPickupReservoir(const Vec2& uv) const;
    bool tickWetPaint(const BrushSettings& brush, float dt,
                      bool auto_normal_from_height_enabled,
                      float normal_strength);
    void clearWetSimulation();

private:
    struct WetSeamLink {
        std::array<Vec2, 3> source_uvs{};
        std::array<Vec2, 3> target_uvs{};
        int source_edge_a = 0;
        int source_edge_b = 1;
        int source_opposite = 2;
        int target_edge_a = 0;
        int target_edge_b = 1;
        int target_opposite = 2;
    };

    struct WetSimulationState {
        std::string target_key;
        int width = 0;
        int height = 0;
        bool uses_layers = false;
        uint32_t layer_id = 0;
        float simulation_time_accumulator = 0.0f;
        std::vector<float> wetness;
        std::vector<float> pigment;
        std::vector<float> thickness;
        PaintDirtyRect active_region;
        float weather_spawn_accumulator = 0.0f;
        uint32_t weather_spawn_cursor = 0;
        float weather_cluster_u = 0.5f;
        float weather_cluster_v = 0.5f;
        float weather_cluster_refresh_accumulator = 1.0f;
    };

    struct WetFlowTriangleInfo {
        std::array<Vec2, 3> uvs{};
        float min_u = 0.0f;
        float max_u = 0.0f;
        float min_v = 0.0f;
        float max_v = 0.0f;
        float flow_x = 0.0f;
        float flow_y = 0.0f;
        float flow_length = 0.0f;
        float slope = 0.0f;
    };

    struct WetFlowFieldCache {
        std::string target_key;
        int width = 0;
        int height = 0;
        int uv_set = 0;
        int lookup_resolution = 0;
        float max_slope = 0.0f;
        float max_flow_length = 0.0f;
        std::vector<WetFlowTriangleInfo> infos;
        std::vector<int> lookup_indices;
    };

    void rebuildWetSeamLinks();
    void invalidateWetFlowField();
    void rebuildWetFlowField(int width, int height);
    int findWetFlowTriangleIndex(const Vec2& uv, int hint_index) const;
    void mirrorWetRegionAcrossSeams(std::vector<CompactVec4>& pixels, int width, int height,
                                    PaintDirtyRect region, float blend_strength);

    SceneData* scene_ = nullptr;
    std::shared_ptr<Triangle> triangle_;
    WetSimulationState wet_basecolor_state_;
    WetFlowFieldCache wet_flow_field_cache_;
    std::vector<WetSeamLink> wet_seam_links_;
    std::string wet_seam_key_;
};

} // namespace Paint
