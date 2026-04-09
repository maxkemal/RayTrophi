#pragma once

#include <optional>
#include <vector>
#include <memory>
#include <string>
#include <array>
#include "Paint/PaintLayer.h"
#include "Paint/IPaintSurfaceAdapter.h"
#include "Paint/PaintTextureSet.h"
#include "Paint/PaintLayerStack.h"
#include "Texture.h"

namespace Paint {

enum class MaterialBindingMode : unsigned char {
    UseCurrentMaterial = 0,
    MakeUniqueForPaint
};

struct PaintStroke {
    bool active = false;
    bool changed = false;
    int layer_index = 0;
    float elapsed = 0.0f;
    bool has_last_uv = false;
    float last_u = 0.0f;
    float last_v = 0.0f;
    bool stamp_applied = false;
    bool has_clone_source = false;
    float clone_source_u = 0.0f;
    float clone_source_v = 0.0f;
    float clone_offset_u = 0.0f;
    float clone_offset_v = 0.0f;
    bool clone_offset_initialized = false;
    bool has_pickup_color = false;
    Vec3 pickup_color = Vec3(1.0f, 1.0f, 1.0f);
    Vec3 wet_color = Vec3(1.0f, 1.0f, 1.0f);
    bool has_wet_color = false;
    float remaining_paint_load = 1.0f;
    PaintChannel channel = PaintChannel::BaseColor;
    std::shared_ptr<Texture> texture_snapshot_ref;
    std::vector<CompactVec4> before_pixels;
    std::array<std::shared_ptr<Texture>, 7> texture_snapshot_refs{};
    std::array<std::vector<CompactVec4>, 7> before_pixels_by_channel{};

    // Layer-aware undo snapshots
    bool using_layers = false;
    std::string layer_stack_key;           // e.g. "NodeName#42"
    uint32_t layer_id = 0;                 // id of the painted layer
    std::array<std::vector<CompactVec4>, 7> before_layer_pixels{};
};

class PaintModeState {
public:
    bool enabled = false;
    bool compact_ui = true;
    std::string active_target_name;
    int active_layer_index = 0;
    int active_material_slot = 0;
    MaterialBindingMode material_binding_mode = MaterialBindingMode::UseCurrentMaterial;
    PaintChannel active_channel = PaintChannel::BaseColor;
    int requested_texture_resolution = 1024;
    float height_to_normal_strength = 4.0f;
    bool auto_normal_from_height = false;
    bool clear_height_after_bake = true;
    std::array<bool, 7> linked_channels{};
    BrushSettings brush;
    PaintStroke stroke;
    std::vector<PaintLayer> ui_layers;

    PaintModeState();

    void setAdapter(const PaintSurfaceAdapterPtr& adapter);
    void clearAdapter();
    PaintSurfaceAdapterPtr getAdapter() const;
    bool hasValidTarget() const;
    void syncLayersFromAdapter();
    void clampActiveLayer();

    // -------- Layer Stack bridge --------
    // Bind an external PaintLayerStack (owned by SceneData) to drive ui_layers.
    void bindLayerStack(PaintLayerStack* stack);
    PaintLayerStack* getBoundLayerStack() const { return bound_layer_stack_; }

    // Synchronise ui_layers metadata from the bound layer stack.
    void syncLayersFromStack();

    // Push ui_layers metadata edits (visibility, opacity, blend, name, lock)
    // back into the bound layer stack.
    void pushLayerMetaToStack();

    // Layer operations (delegate to bound stack, then sync back).
    int  addLayerAboveCurrent(const std::string& name = "");
    int  duplicateCurrentLayer();
    bool removeCurrentLayer();
    bool mergeCurrentLayerDown();
    bool moveCurrentLayer(int delta);  // +1 = up, -1 = down
    void flattenAllLayers();

private:
    PaintSurfaceAdapterPtr active_adapter_;
    PaintLayerStack* bound_layer_stack_ = nullptr;
};

} // namespace Paint
