#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <iosfwd>
#include "Paint/PaintLayerData.h"
#include "Paint/PaintTextureSet.h"
#include "json.hpp"

namespace Paint {

// Manages an ordered stack of paint layers and composites them into a
// PaintTextureSet so the renderer always sees a single flat texture per channel.
//
// Layer 0 is the bottom; the last element is the top-most layer.
class PaintLayerStack {
public:
    PaintLayerStack() = default;

    // -------- layer access --------

    int  layerCount() const { return static_cast<int>(layers_.size()); }
    bool empty()      const { return layers_.empty(); }

    PaintLayerData*       layerAt(int index);
    const PaintLayerData* layerAt(int index) const;
    PaintLayerData*       layerById(uint32_t id);

    int indexOfId(uint32_t id) const;

    // -------- layer mutation --------

    // Adds a new blank layer at the given position (clamped to [0, count]).
    // Returns the index where it was inserted.
    int  addLayer(const std::string& name, int insert_at = -1);

    // Duplicate layer at `source_index`, placing the copy directly above it.
    int  duplicateLayer(int source_index);

    // Remove the layer at `index`. Returns false if index is invalid or
    // this is the last remaining layer (stack must always have >= 1 layer).
    bool removeLayer(int index);

    // Move layer from `from_index` to `to_index`. Returns false on invalid indices.
    bool moveLayer(int from_index, int to_index);

    // Merge the layer at `index` down into `index - 1`.
    // The upper layer is removed after merging. Returns false if index <= 0.
    bool mergeDown(int index);

    // Flatten ALL visible layers into layer 0 and remove the rest.
    void flattenAll();

    // -------- compositing --------

    // Composite all visible layers for a single channel and write the result
    // into `dst_pixels` (sized width*height). The caller owns the vector.
    void compositeChannel(PaintChannel channel,
                          std::vector<CompactVec4>& dst_pixels,
                          int width, int height) const;

    // Flatten every channel into the provided PaintTextureSet's textures.
    // Channels where no layer has pixel data are left untouched.
    void flattenInto(PaintTextureSet& texture_set) const;

    // Flatten a single channel into the texture set.
    void flattenChannelInto(PaintChannel channel, PaintTextureSet& texture_set) const;

    // Flatten only the dirty region of a single channel into the texture set.
    void flattenChannelRegionInto(PaintChannel channel, PaintTextureSet& texture_set,
                                  const PaintDirtyRect& dirty) const;

    // Returns true if any layer has pixel data for the given channel.
    bool anyLayerHasPixels(PaintChannel channel) const;

    // -------- resolution --------

    int  width()  const { return width_; }
    int  height() const { return height_; }

    // (Re)initialise the stack resolution. Existing layers are resized.
    void setResolution(int w, int h);

    // -------- initialisation helpers --------

    // Seed layer 0 from an existing PaintTextureSet (copies pixel data
    // from each channel texture into the bottom layer's buffers).
    void seedFromTextureSet(const PaintTextureSet& src);

    // Convenience: ensure at least one layer exists.
    void ensureBaseLayer();

    // -------- serialization --------

    // Write layer metadata to JSON and pixel data as PNG blobs into the
    // binary stream.  Each non-empty channel is stored as a separate PNG
    // whose offset/size is recorded in the JSON.
    void serialize(nlohmann::json& j, std::ostream& bin) const;

    // Restore from a JSON + binary pair produced by serialize().
    void deserialize(const nlohmann::json& j, std::istream& bin);

private:
    uint32_t nextId();

    std::vector<PaintLayerData> layers_;
    uint32_t next_id_ = 1;
    int width_  = 0;
    int height_ = 0;
};

} // namespace Paint
