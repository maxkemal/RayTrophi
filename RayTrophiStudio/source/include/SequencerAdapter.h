/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          SequencerAdapter.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include "../external/ImSequencer.h"
#include "KeyframeSystem.h"
#include "scene_data.h"
#include <vector>
#include <string>

// Track type enumeration
enum class TrackType {
    Object_Transform = 0,
    Object_Material = 1,
    Light = 2,
    Camera = 3,
    World = 4
};

// Single sequencer item (one track row)
struct SequencerItem {
    std::string name;           // "Cube.001", "Light.001", "World"
    std::string entity_name;    // Reference to timeline track
    TrackType type;
    int start_frame = 0;
    int end_frame = 250;
    unsigned int color = 0xFFFFFFFF;
    bool expanded = true;
};

// Adapter class implementing ImSequencer::SequenceInterface
class SequencerAdapter : public ImSequencer::SequenceInterface {
public:
    SequencerAdapter(TimelineManager& timeline, SceneData& scene);
    
    // Build track list from timeline data
    void rebuildTracks();
    
    // ImSequencer::SequenceInterface implementation
    int GetFrameMin() const override { return frame_min; }
    int GetFrameMax() const override { return frame_max; }
    int GetItemCount() const override { return static_cast<int>(items.size()); }
    
    void Get(int index, int** start, int** end, int* type, unsigned int* color) override;
    const char* GetItemLabel(int index) const override;
    
    void Add(int type) override;
    void Del(int index) override;
    void Duplicate(int index) override;
    
    size_t GetCustomHeight(int index) override;
    void CustomDraw(int index, ImDrawList* draw_list, const ImRect& rc, 
                   const ImRect& legendRect, const ImRect& clippingRect, 
                   const ImRect& legendClippingRect) override;
    
    // Track management
    std::vector<SequencerItem>& getItems() { return items; }
    
private:
    TimelineManager& timeline;
    SceneData& scene;
    std::vector<SequencerItem> items;
    int frame_min = 0;
    int frame_max = 250;
};

